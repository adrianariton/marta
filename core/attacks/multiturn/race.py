from core.attacks.interfaces import TextGenerator, MultiturnStrat
from core.attacks.interfaces import LinearHistory
from core.attacks.classifiers.refusal.base import RefusalClassifier
from typing import Callable, Optional
import re
from core.attacks.helpers import RandomQueryGenerator, AgentRandomQueryGenerator
from core.attacks.evaluators import GGEEvaluator
from core.attacks.helpers.simulators import SelfPlaySimulator, SPSProps
from typing import cast
from core.attacks.helpers.realigner import ReAligner
from core.attacks.evaluators import PEC7Evaluator, PEC7
from core.attacks.interfaces import OneShotConversation, Conversation
from typing import Optional, Callable, Union, Awaitable

import asyncio

race_lock = (
    asyncio.Lock()
)  # global lock for printing best candidate and rating to avoid jumbled prints in async environment


class RACEAttack(MultiturnStrat):
    """
    [RACE]https://arxiv.org/pdf/2502.11054

    evaluator (GGEEvaluator) : ❶ The Gainguided Exploration module selects queries that remain
    semantically aligned with the target while extracting
    useful information, ensuring steady attack progression.

    selfplay (SPSProps) : ❷ The Self-play module preemptively refines
    queries within the shadow model by simulating potential rejection responses, improving attack efficiency
    before engaging the victim model.

    refusalFeedback (ReAligner) : ❸ The Rejection Feedback module analyzes failed interactions
    and restructures queries into alternative reasoning
    challenges, enabling quick recovery and maintaining
    attack stability. Credit to [FITD]https://arxiv.org/pdf/2502.19820 as well.

    We assume the attack model is the same as the target model so
    we initialize selfplay with two copies of attack_model
    """

    def __init__(
        self,
        attack_model: TextGenerator,
        gge_evaluator: GGEEvaluator,
        goal_evaluator: PEC7Evaluator,
        refusal_model: RefusalClassifier,
        refusalFeedback: ReAligner,
        goal: str,
        random_query_generator: Optional[RandomQueryGenerator] = None,
        use_self_play: bool = True,
        rounds: int = 5,
        max_refusals: int = 10,
        verbose: bool = True,
        candidates_per_round: int = 3,
        sps_props: Optional[SPSProps] = None,
        stateful: bool = True,
        samples_per_candidate: int = 3,
        victim_simulator: Optional[TextGenerator] = None,
        use_async_sps: bool = True,
    ):
        super().__init__()
        self.RF = refusalFeedback
        self.i = 0
        self.sps_props = SPSProps()
        self.candidates_per_round = candidates_per_round
        self.A = attack_model
        self.victim_simulator = victim_simulator or attack_model
        self.RQG = random_query_generator or AgentRandomQueryGenerator(5, attack_model)
        self.GGE = gge_evaluator
        self.J = goal_evaluator
        self.Ref = refusal_model
        self.use_self_play = use_self_play
        self.rounds = rounds
        self.goal = goal
        self.max_refusals = max_refusals
        self.verbose = verbose
        self.sps_props = sps_props or SPSProps()
        self.stateful = stateful
        self.samples_per_candidate = samples_per_candidate
        self.C_refu = 0
        self.H_A = LinearHistory(
            metrics=["user", "assistant", "system"],
            descriptions=["User attack query", "Assistant reply", "System message"],
        )
        self.H_B = LinearHistory(
            metrics=["user", "assistant", "eval"],
            descriptions=["User attack query", "Assistant reply", "Evaluation"],
        )
        self.r = None
        self.use_async_sps = use_async_sps

        def ev(h: LinearHistory, q: str, r: str):
            return self.GGE.evaluate(h, q, r, self.goal)

        async def async_ev(h: LinearHistory, q: str, r: str):
            return await self.GGE.aevaluate(h, q, r, self.goal)

        if self.use_self_play:
            self.self_play_sim: SelfPlaySimulator = SelfPlaySimulator(
                shadow_model=self.A,
                victim_simulator=self.victim_simulator,  # assume same if not specified
                refusal_classifier=self.Ref,
                evaluator_function=self.sps_props.evaluator_function
                or cast(
                    Callable[[LinearHistory, str, str], Union[float, Awaitable[float]]],
                    async_ev if use_async_sps else ev,
                ),
                random_modifier=self.sps_props.random_modifier,
                repeatablity=self.sps_props.repeatablity,
                depth=self.sps_props.depth,
                random_state=self.sps_props.random_state,
                gamma=self.sps_props.gamma,
                max_evaluation_score=self.sps_props.max_evaluation_score,
                attack_strategy=self.sps_props.self_play_strategy,
            )

    def _IG(self, q: str, r: str):
        evv = self.sps_props.evaluator_function or cast(
            Callable[[LinearHistory, str, str], float], self.GGE.evaluate
        )
        return evv(self.H_A, q, r, self.goal)

    async def a_IG(self, q: str, r: str):
        evv = self.sps_props.evaluator_function or cast(
            Awaitable[Callable[[LinearHistory, str, str], float]], self.GGE.aevaluate
        )
        return await evv(self.H_A, q, r, self.goal)

    def _get_best_candidate(self, initial_quety_or_goal: Optional[str] = None) -> tuple[str, float]:
        initial_quety_or_goal = initial_quety_or_goal or self.goal
        if not self.stateful:
            initial_quety_or_goal = self.goal  # only ask from goal if not stateful
        candidates = self.RQG.get_candidates(
            query=initial_quety_or_goal,
            n=self.candidates_per_round,
            history=self.H_B,
            turns=self.rounds - self.i,
            max_turns=self.rounds,
        )
        candidates = list(reversed(candidates))
        if self.verbose:
            print(f"{candidates=}")
        if len(candidates) < self.candidates_per_round:
            # something went wrong
            return None, None

        if self.use_self_play:
            best_candidate_query, max_rating = self.self_play_sim.optimize(
                candidates,
                self.H_A.to_conversation("user", "assistant", "system"),
                include_evaluation=True,
                samples=self.samples_per_candidate,
            )
        else:
            candidates_and_eval_scores = []
            for c in candidates:
                r = "<Pretend there is a response here to the query and use it for rating>"
                _ev = self._IG(q=c, r=r)
                candidates_and_eval_scores.append((c, _ev))

            candidates_and_eval_scores.sort(
                key=lambda tpl: tpl[1], reverse=True
            )  # sort from big to small eval
            best_candidate_query = candidates_and_eval_scores[0][0]
            max_rating = candidates_and_eval_scores[0][1]
        if self.verbose:
            print("\n^^^^^^^^^^^^^^^^^^^")
            print(f"{best_candidate_query=} {max_rating=}")
            print("^^^^^^^^^^^^^^^^^^^")
        return best_candidate_query, max_rating

    async def a_get_best_candidate(
        self,
        initial_quety_or_goal: Optional[str] = None,
        messages: Optional[OneShotConversation] = None,
    ) -> tuple[str, float]:
        initial_quety_or_goal = initial_quety_or_goal or self.goal
        if not self.stateful:
            initial_quety_or_goal = self.goal  # only ask from goal if not stateful
        candidates = await self.RQG.aget_candidates(
            query=initial_quety_or_goal,
            n=self.candidates_per_round,
            history=self.H_B,
            turns=self.rounds - self.i,
            max_turns=self.rounds,
        )
        candidates = list(reversed(candidates))
        if self.verbose:
            print(f"{candidates=}")
        if len(candidates) < self.candidates_per_round:
            # something went wrong
            return None, None

        if self.use_self_play:
            # if self.verbose:
            #     print("=========== HAHAHA ============")
            #     print(
            #         self.H_A.to_conversation("user", "assistant", "system")
            #         .latest_oneshot()
            #         ._messages
            #     )
            #     print("--")
            #     print(
            #         Conversation.from_oneshot(messages).latest_oneshot()._messages
            #         if messages
            #         else "No messages provided"
            #     )
            #     print("=======================")
            best_candidate_query, max_rating = await self.self_play_sim.aoptimize(
                candidates,
                Conversation.from_oneshot(messages),
                include_evaluation=True,
                samples=self.samples_per_candidate,
            )
        else:
            candidates_and_eval_scores = []
            for c in candidates:
                r = "<Pretend there is a response here to the query and use it for rating>"
                _ev = await self.a_IG(q=c, r=r)
                candidates_and_eval_scores.append((c, _ev))

            candidates_and_eval_scores.sort(
                key=lambda tpl: tpl[1], reverse=True
            )  # sort from big to small eval
            best_candidate_query = candidates_and_eval_scores[0][0]
            max_rating = candidates_and_eval_scores[0][1]
        if self.verbose:
            print("\n^^^^^^^^^^^^^^^^^^^")
            print(f"{best_candidate_query=} {max_rating=}")
            print("^^^^^^^^^^^^^^^^^^^")
        return best_candidate_query, max_rating

    # -------------------------------------------------------------
    # Main multiturn attack() step (1 round of algorithm)
    # -------------------------------------------------------------
    def attack(self, messages: OneShotConversation) -> list[str]:
        """
        Executes exactly **one** step of Algorithm 1 per call.
        Produces the next query q, respecting backtracking.
        """

        self.update_history_with_answer(messages, replace_empty=True)
        self.log(messages, self.verbose)

        last_query, last_response = self.H_A.get_last(metrics=["user", "assistant"])
        if last_response is not None:
            self.H_B.add("assistant", last_response)

        if (last_response is not None) and (self.Ref.is_refused(last_response)):
            self.H_B.add("eval", "response refused!")
            self.rollback(2)
            self.H_A.pop(metric="user")
            self.H_A.pop(metric="assistant")
            self.C_refu += 1
            if self.C_refu > self.max_refusals:
                return self.set_done()
            realignedQuery = self.RF.reAlign(
                last_query, concern=last_response, goal=self.goal, score=0
            )
            # self.H_A.add("user", realignedQuery)
            self.H_B.new_message(self.i + self.C_refu)
            self.H_B.add("user", realignedQuery)
            return self.new_query(realignedQuery)
        elif (last_response is not None) and (last_query is not None):
            eval_ = self.J.evaluate(last_query, last_response, goal=self.goal)
            self.H_B.add("eval", eval_.to_jsonstring())
            if eval_.goal_revealed:
                self.set_goal_achieved()
                return self.set_done()

        # u, a, u, a, u, a
        if last_response == None:
            candidate, rating = self._get_best_candidate()
        else:
            candidate, rating = self._get_best_candidate(last_query)
        #with race_lock:
        #    print(f"best_{candidate=} {rating=}")
        if candidate is None:
            self.C_refu += 1
            if self.C_refu > self.max_refusals:
                return self.set_done()
            return self.ask_nothing()

        # move on only if query was not refused
        self.i += 1
        if self.i > self.rounds:
            return self.set_done()

        # Return the next query to the target model
        self.H_B.new_message(self.i + self.C_refu)
        self.H_B.add("user", candidate)
        return self.new_query(query=candidate)

    async def aattack(self, messages: OneShotConversation) -> list[str]:
        """
        Executes exactly **one** step of Algorithm 1 per call.
        Produces the next query q, respecting backtracking.
        """

        self.update_history_with_answer(messages, replace_empty=True)
        self.log(messages, self.verbose)

        last_query, last_response = self.H_A.get_last(metrics=["user", "assistant"])
        if last_response is not None:
            self.H_B.add("assistant", last_response)

        if (last_response is not None) and (await self.Ref.ais_refused(last_response)):
            self.H_B.add("eval", "response refused!")
            self.rollback(2)
            self.H_A.pop(metric="user")
            self.H_A.pop(metric="assistant")
            self.C_refu += 1
            if self.C_refu > self.max_refusals:
                return self.set_done()
            realignedQuery = await self.RF.areAlign(
                last_query, concern=last_response, goal=self.goal, score=0
            )
            # self.H_A.add("user", realignedQuery)
            self.H_B.new_message(self.i + self.C_refu)
            self.H_B.add("user", realignedQuery)
            return self.new_query(realignedQuery)
        elif (last_response is not None) and (last_query is not None):
            eval_ = await self.J.aevaluate(last_query, last_response, goal=self.goal)
            self.H_B.add("eval", eval_.to_jsonstring())
            if eval_.goal_revealed:
                self.set_goal_achieved()
                return self.set_done()

        # u, a, u, a, u, a
        if last_response == None:
            candidate, rating = await self.a_get_best_candidate(messages=messages)
        else:
            candidate, rating = await self.a_get_best_candidate(last_query, messages=messages)
        #async with race_lock:
        #    print(f"best_{candidate=} {rating=}")
        if candidate is None:
            self.C_refu += 1
            if self.C_refu > self.max_refusals:
                return self.set_done()
            return self.ask_nothing()

        # move on only if query was not refused
        self.i += 1
        if self.i > self.rounds:
            return self.set_done()

        # Return the next query to the target model
        self.H_B.new_message(self.i + self.C_refu)
        self.H_B.add("user", candidate)
        return self.new_query(query=candidate)
