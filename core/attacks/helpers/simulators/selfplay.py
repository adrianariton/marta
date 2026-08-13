from core.attacks.interfaces import TextGenerator
from core.attacks.classifiers.refusal.base import RefusalClassifier
from core.attacks.interfaces import Conversation
from core.attacks.helpers.modifiers import QueryModifier, RandomStateModifier
from typing import Optional, Callable
import random
from dataclasses import dataclass
from core.attacks.interfaces import LinearHistory
from typing import cast
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from typing import Optional, Callable, Union, Awaitable


@dataclass
class SPSProps:
    random_modifier: Optional[QueryModifier] = None
    repeatablity: int = 4
    depth: int = 1
    random_state: Optional[random.Random] = None
    gamma: float = 0.8
    max_evaluation_score: int = 7
    evaluator_function: Optional[
        Callable[[LinearHistory, str, str], Union[float, Awaitable[float]]]
    ] = None
    self_play_strategy: Optional[Strategy] = None


class SelfPlaySimulator:
    """[RACE]https://arxiv.org/pdf/2502.11054"""

    def __init__(
        self,
        shadow_model: TextGenerator,
        victim_simulator: TextGenerator,
        refusal_classifier: RefusalClassifier,
        attack_strategy: Optional[Strategy] = None,
        evaluator_function: Optional[
            Callable[[LinearHistory, str, str], Union[float, Awaitable[float]]]
        ] = None,
        random_modifier: Optional[QueryModifier] = None,
        repeatablity: int = 4,
        depth: int = 1,
        random_state: Optional[random.Random] = None,
        gamma: float = 0.8,
        max_evaluation_score: int = 7,
        goal: str = "How to build a bomb?",
    ):
        """_summary_

        Args:
            shadow_model (TextGenerator): _description_
            victim_simulator (TextGenerator): _description_
            refusal_classifier (RefusalClassifier): _description_
            evaluator_function (Optional[Callable[[LinearHistory, str, str], Union[float, Awaitable[float]]]], optional): (linear_history, q, r) -> float (0.0-max_evaluation_score). Defaults to None.
            random_modifier (Optional[QueryModifier], optional): _description_. Defaults to None.
            repeatablity (int, optional): _description_. Defaults to 4.
            depth (int, optional): _description_. Defaults to 1.
            attack_strategy: Strategy, has a non-param system prompt and a user prompt(history, goal)
            random_state (Optional[random.Random], optional): _description_. Defaults to None.
            gamma (float, optional): _description_. Defaults to 0.8.
            max_evaluation_score (int, optional): _description_. Defaults to 7.
        """
        self.goal = goal
        self.ms = shadow_model  # The attacker
        self.mv_prime = victim_simulator  # The simulated victim
        self.refusal_classifier = refusal_classifier
        self.random_modifier = random_modifier or RandomStateModifier()
        self.repeatability = repeatablity
        self.random_state = random_state or random.Random()
        self.depth = depth
        self.gamma = gamma
        self.max_evaluation_score = max_evaluation_score
        self.maybe_strategy = attack_strategy
        self.evaluator_function = evaluator_function or cast(
            Callable[[LinearHistory, str, str], float], lambda h, s, r: 0.0
        )
        self.is_evaluator_async = isinstance(self.evaluator_function, Awaitable)
        if self.maybe_strategy:
            self.maybe_strategy.assert_conforms(system_prompt=[])
            self.maybe_strategy.assert_conforms(user_prompt=["history", "goal"])
        self.has_evaluator_function = not (evaluator_function is None)

    def utility(self, response: str) -> float:
        """Implements Equation (4): u_Ms(s, qc, rc)"""
        # Check if the response contains any rejection keywords
        is_rejection = self.refusal_classifier.is_refused(response)
        return 0.0 if is_rejection else 1.0

    def get_variate_queries(self, query_qc: str, n: int, random_state: random.Random):
        modified_queries = self.random_modifier.modify(
            query_qc, n // self.repeatability + 1, random_state=random_state
        )
        modified_queries = modified_queries * self.repeatability
        return modified_queries[:n]

    def get_expected_utility(
        self,
        state_s: Conversation,
        query_qc: str,
        samples: int = 3,
        include_evaluation: bool = False,
        stop_if_first_utility_is_zero: bool = True,
    ) -> tuple[float, Optional[float]]:
        """Implements Equation (5): U_Ms using Monte Carlo sampling
        Returns:
            utility: float, evaluation: Optional[float]
        """
        total_utility = 0.0
        total_eval = 0.0
        modified_queries = self.get_variate_queries(
            query_qc, n=samples, random_state=self.random_state
        )

        print(f"{modified_queries=}")

        include_evaluation = include_evaluation and self.has_evaluator_function

        for query_qc_modified in modified_queries:
            state_s.add_text_message(query_qc_modified, "user")
            simulated_response = self.mv_prime.generate(state_s.latest_oneshot())[0]
            utility = self.utility(simulated_response)

            evaluation = (
                self.evaluator_function(
                    state_s.to_linear_history(), query_qc_modified, simulated_response
                )
                if include_evaluation
                else 0.0
            )
            if (utility > 0) or (not stop_if_first_utility_is_zero):
                for d in range(self.depth - 1):
                    state_s.add_text_message(simulated_response, "assistant")
                    what_user_says = self.ms.generate(
                        state_s.flipped(remove_first_non_system=True).latest_oneshot()
                    )[0]
                    state_s.add_text_message(what_user_says, "user")
                    print(f"{state_s.latest_oneshot()=}")
                    if self.maybe_strategy:
                        simulated_response = self.mv_prime.query(
                            q=self.maybe_strategy.user_prompt(
                                history=state_s.to_linear_history().promptify(trunc_text_at=300),
                                goal=self.goal,
                            ),
                            system_prompt=self.maybe_strategy.system_prompt(),
                        )
                    else:
                        simulated_response = self.mv_prime.generate(state_s.latest_oneshot())[0]
                    utility = (1 - self.gamma) * utility + self.gamma * self.utility(
                        simulated_response
                    )  # lr formula
                    evaluation_prime = (
                        self.evaluator_function(
                            state_s.to_linear_history(), what_user_says, simulated_response
                        )
                        if include_evaluation
                        else 0.0
                    )
                    evaluation = (1 - self.gamma) * evaluation + self.gamma * evaluation_prime
                # add 2 * (d - 1) + 1
                for d in range(self.depth - 1):
                    state_s.pop()
                    state_s.pop()
            state_s.pop()
            # pop 2 * (d - 1) + 1
            total_utility += utility
            total_eval += evaluation

        return total_utility / samples, (total_eval / samples if include_evaluation else None)

    async def aget_expected_utility(
        self,
        state_s: Conversation,
        query_qc: str,
        samples: int = 3,
        include_evaluation: bool = False,
        stop_if_first_utility_is_zero: bool = True,
    ) -> tuple[float, Optional[float]]:
        """Implements Equation (5): U_Ms using Monte Carlo sampling
        Returns:
            utility: float, evaluation: Optional[float]
        """
        total_utility = 0.0
        total_eval = 0.0
        modified_queries = self.get_variate_queries(
            query_qc, n=samples, random_state=self.random_state
        )

        print(f"{modified_queries=}")

        include_evaluation = include_evaluation and self.has_evaluator_function

        for query_qc_modified in modified_queries:
            state_s.add_text_message(query_qc_modified, "user")
            simulated_response = (await self.mv_prime.agenerate(state_s.latest_oneshot()))[0]
            utility = self.utility(simulated_response)

            evaluation = (
                (
                    await self.evaluator_function(
                        state_s.to_linear_history(), query_qc_modified, simulated_response
                    )
                )
                if include_evaluation
                else 0.0
            )
            if (utility > 0) or (not stop_if_first_utility_is_zero):
                for d in range(self.depth - 1):
                    state_s.add_text_message(simulated_response, "assistant")
                    what_user_says = (
                        await self.ms.agenerate(
                            state_s.flipped(remove_first_non_system=True).latest_oneshot()
                        )
                    )[0]
                    state_s.add_text_message(what_user_says, "user")
                    print(f"{state_s.latest_oneshot()=}")
                    if self.maybe_strategy:
                        simulated_response = await self.mv_prime.aquery(
                            q=self.maybe_strategy.user_prompt(
                                history=state_s.to_linear_history().promptify(trunc_text_at=300),
                                goal=self.goal,
                            ),
                            system_prompt=self.maybe_strategy.system_prompt(),
                        )
                    else:
                        simulated_response = (
                            await self.mv_prime.agenerate(state_s.latest_oneshot())
                        )[0]
                    utility = (1 - self.gamma) * utility + self.gamma * self.utility(
                        simulated_response
                    )  # lr formula
                    evaluation_prime = (
                        (
                            await self.evaluator_function(
                                state_s.to_linear_history(), what_user_says, simulated_response
                            )
                        )
                        if include_evaluation
                        else 0.0
                    )
                    evaluation = (1 - self.gamma) * evaluation + self.gamma * evaluation_prime
                # add 2 * (d - 1) + 1
                for d in range(self.depth - 1):
                    state_s.pop()
                    state_s.pop()
            state_s.pop()
            # pop 2 * (d - 1) + 1
            total_utility += utility
            total_eval += evaluation

        return total_utility / samples, (total_eval / samples if include_evaluation else None)

    def optimize(
        self,
        candidates: list[str],
        history: Conversation,
        include_evaluation: bool = False,
        samples: int = 3,
    ) -> tuple[str, float]:
        """Implements Equation (6): Selecting q* that maximizes utility"""
        state_s = history  # Convert history object to stack-string
        best_query = candidates[0]
        include_evaluation = include_evaluation and self.has_evaluator_function
        max_u = -self.max_evaluation_score

        for qc in candidates:
            u, e = self.get_expected_utility(
                state_s, qc, samples=samples, include_evaluation=include_evaluation
            )
            print(f"{u=} {e=} {qc=}\n")
            u = u if not include_evaluation else (u * e / self.max_evaluation_score)
            if u > max_u:
                max_u = u
                best_query = qc

            # Optimization: If we find a query with 1.0 utility, we can break early
            if max_u == self.max_evaluation_score * 1.0:
                break

        return self.random_modifier.modify(best_query, 1, self.random_state)[0], max_u

    async def aoptimize(
        self,
        candidates: list[str],
        history: Conversation,
        include_evaluation: bool = False,
        samples: int = 3,
    ) -> tuple[str, float]:
        """Implements Equation (6): Selecting q* that maximizes utility"""
        state_s = history  # Convert history object to stack-string
        best_query = candidates[0]
        include_evaluation = include_evaluation and self.has_evaluator_function
        max_u = -self.max_evaluation_score

        for qc in candidates:
            u, e = await self.aget_expected_utility(
                state_s, qc, samples=samples, include_evaluation=include_evaluation
            )
            print(f"{u=} {e=} {qc=}\n")
            u = u if not include_evaluation else (u * e / self.max_evaluation_score)
            if u > max_u:
                max_u = u
                best_query = qc

            # Optimization: If we find a query with 1.0 utility, we can break early
            if max_u == self.max_evaluation_score * 1.0:
                break

        return self.random_modifier.modify(best_query, 1, self.random_state)[0], max_u
