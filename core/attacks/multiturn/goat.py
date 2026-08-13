from core.attacks.interfaces import TextGenerator, MultiturnStrat
from core.attacks.interfaces import LinearHistory
from core.attacks.classifiers.refusal.base import RefusalClassifier
from typing import Callable, Optional
import re
from core.attacks.evaluators import PEC7Evaluator
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from core.attacks.helpers.approaches import AttackApproach
from core.attacks.utils import parse_simple
from core.attacks.interfaces import Conversation
from core.attacks.helpers.realigner import ReAligner
from core.attacks.conversation import OneShotConversation


class GoatAttack(MultiturnStrat):
    def __init__(
        self,
        attack_model: TextGenerator,
        evaluator: PEC7Evaluator,
        chain_of_thought_strategy: Strategy,
        approaches: list[AttackApproach],
        goal: str,
        refusal_model: Optional[RefusalClassifier] = None,
        rounds: int = 10,
        max_refusals: int = 10,
        evaluate_every: int = 2,
        verbose: bool = True,
    ):
        super().__init__()
        self.verbose = verbose
        self.evaluate_every = evaluate_every
        self.strategy = chain_of_thought_strategy
        self.strategy.assert_conforms(system_prompt=["goal", "approach_descriptions"])
        self.strategy.assert_conforms(conversation_start=["goal"])
        self.strategy.assert_conforms(
            conversation_followup=[
                "goal",
                "adversarial_prev_prompt",
                "prev_model_response",
            ]
        )

        self.A = attack_model
        self.J = evaluator
        self.Ref = refusal_model
        self.goal = goal
        self.rounds = rounds
        self.max_refusals = max_refusals
        self.C_refu = 0
        self.approaches = approaches
        self.approaches_descs = AttackApproach.stringify(self.approaches)
        self.H_A = LinearHistory.for_simple_logging()
        self.A_conversation = Conversation()
        system_prompt = self.strategy.system_prompt(
            goal=self.goal, approach_descriptions=self.approaches_descs
        )
        self.A_conversation.set_system_message(system_prompt)
        self.i = 0
        self.should_eval = True

    def _conversation_start(self) -> str:
        user_prompt = self.strategy.conversation_start(goal=self.goal)
        self.A_conversation.add_text_message(user_prompt, "user")
        output = self.A.generate(self.A_conversation.latest_oneshot())[0]
        dict_ = parse_simple(output, markers=["O:", "T:", "S:", "R:"])
        self.A_conversation.add_text_message(output, "assistant")
        self.i += 1
        return dict_["R:"]

    def _conversation_followup(self, last_q, last_r) -> str:
        user_prompt = self.strategy.conversation_followup(
            goal=self.goal, adversarial_prev_prompt=last_q, prev_model_response=last_r
        )
        self.A_conversation.add_text_message(user_prompt, "user")
        output = self.A.generate(self.A_conversation.latest_oneshot())[0]
        dict_ = parse_simple(output, markers=["O:", "T:", "S:", "R:"])
        self.A_conversation.add_text_message(output, "assistant")
        self.i += 1
        return dict_["R:"]

    async def a_conversation_start(self) -> str:
        user_prompt = self.strategy.conversation_start(goal=self.goal)
        self.A_conversation.add_text_message(user_prompt, "user")
        output = (await self.A.agenerate(self.A_conversation.latest_oneshot()))[0]
        dict_ = parse_simple(output, markers=["O:", "T:", "S:", "R:"])
        self.A_conversation.add_text_message(output, "assistant")
        self.i += 1
        if dict_ is None:
            return "continue please"
        return dict_["R:"]

    async def a_conversation_followup(self, last_q, last_r) -> str:
        user_prompt = self.strategy.conversation_followup(
            goal=self.goal, adversarial_prev_prompt=last_q, prev_model_response=last_r
        )
        dict_ = None
        self.A_conversation.add_text_message(user_prompt, "user")
        output = (await self.A.agenerate(self.A_conversation.latest_oneshot()))[0]
        dict_ = parse_simple(output, markers=["O:", "T:", "S:", "R:"])
        self.A_conversation.add_text_message(output, "assistant")
        self.i += 1
        if dict_ is None:
            return "continue please"
        return dict_["R:"]

    def attack(self, messages: OneShotConversation):
        self.update_history_with_answer(messages)

        last_q, last_r = self.H_A.get_last(metrics=["user", "assistant"])
        self.log(messages, verbose=self.verbose)
        # if self.verbose:
        #     print("************************ A_convo ************************")
        #     print(f"{self.A_conversation.to_linear_history().promptify()}")
        #     print("************************ ******* ************************")
        if last_r is None or last_q is None:
            return self.new_query(self._conversation_start())
        else:
            if self.i > self.rounds:
                return self.set_done()

            if self.Ref is not None:
                if self.Ref.is_refused(r=last_r):
                    self.rollback(2)
                    self.H_A.popall(metrics=["user", "assistant"])
                    return self.ask_nothing()

            if self.i % self.evaluate_every == 0 or self.evaluate_every == 1:
                eval_ = self.J.evaluate(
                    last_q, last_r, goal=self.goal, history=self.H_A.promptify()
                )
                if eval_.goal_revealed:
                    self.set_goal_achieved()
                    return self.set_done()
            return self.new_query(self._conversation_followup(last_q, last_r))

    async def aattack(self, messages: OneShotConversation):
        self.update_history_with_answer(messages)

        last_q, last_r = self.H_A.get_last(metrics=["user", "assistant"])
        self.log(messages, verbose=self.verbose)
        # if self.verbose:
        #     print("************************ A_convo ************************")
        #     print(f"{self.A_conversation.to_linear_history().promptify()}")
        #     print("************************ ******* ************************")
        if last_r is None or last_q is None:
            self.should_eval = True
            return self.new_query(await self.a_conversation_start())
        else:
            if self.i > self.rounds:
                return self.set_done()

            if self.Ref is not None:
                if await self.Ref.ais_refused(r=last_r):
                    self.rollback(2)
                    self.H_A.popall(metrics=["user", "assistant"])
                    self.should_eval = False
                    return self.ask_nothing()
            if self.should_eval:
                if self.i % self.evaluate_every == 0 or self.evaluate_every == 1:
                    eval_ = await self.J.aevaluate(
                        last_q, last_r, goal=self.goal, history=self.H_A.promptify()
                    )
                    if eval_.goal_revealed:
                        self.set_goal_achieved()
                        return self.set_done()
            self.should_eval = True
            return self.new_query(await self.a_conversation_followup(last_q, last_r))
