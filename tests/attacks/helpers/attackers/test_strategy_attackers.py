from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path
import pytest


class TestStrategies:
    @pytest.mark.dummy
    def test_strategy_parsing(self):
        path = Path("core/core_strats") / "PLANATTACKER.strat"
        strat = Strategy(path)
        strat.assert_conforms(system_prompt=["target_behavior", "strategy", "max_turns"])
        strat.assert_conforms(user_prompt_first_turn=["turn_1_conversation_flow"])
        strat.assert_conforms(
            user_prompt_subsequent_turns=[
                "previous_turn_number",
                "turn_number",
                "turn_number_conversation_flow",
                "conversation_history",
            ]
        )
        strat.assert_conforms(
            user_prompt_final_turn=[
                "conversation_history",
                "final_turn_conversation_flow",
            ]
        )

        print(strat.strategies())
