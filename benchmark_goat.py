print("import core.utils", flush=True)
from core.utils import get_device

print(f"{get_device()=}", flush=True)
print("importing core.attacks", flush=True)

from core.attacks import (
    IFoundThis,
    Simple,
    GoalTrackerStrat,
    MultiturnStrat,
    YesICan,
    StatelessRemoteGeminiAgent,
    CrescendomationAttack,
    AutoAgent,
    DummyAgent,
    AutoRefusalClassifier,
    HeuristicRefusalClassifier,
    MistralAgent,
    FITDAttack,
    RACEAttack,
    GoatAttack,
    StatelessRemoteClaudeAgent,
)
from core.attacks.helpers.approaches import RefusalSuppressionApproach, PersonaModificationApproach
from core.attacks.helpers.generators.randomquerygenerator import AgentRandomQueryGenerator
from core.attacks.evaluators.GGE import AgentGGEEvaluator
from core.attacks.helpers.realigner import AgentReAligner, PRAlignmentTester
from core.attacks.helpers.generators.bridger import AgentBridger
from core.attacks.helpers.mid import AgentMidQuery
from core.attacks.evaluators.MSMARCO import HfMSMARCOEvaluator
from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from random import choices
from core.datasets.safety.advbench import AdvBench
from core.attacks.helpers.simulators.selfplay import SPSProps
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path

safety_df = BertievidgenSimpleSafetyTests().get_first(n=80)[50:]

BASE_FOLDER = "datasets/goat_bss_gemini_vs_circuitbreaker"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")

for i, row in safety_df.iterrows():
    goal = row["prompt"]
    print("\n\n\n\n============================================================")
    print("************************************************************\n\n")
    print(f"{goal=}")
    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/goat_bss_{row['id']}", buffer_max_size=20
    ) as logger:

        agent = Yielder(logger, tag=Tag.AGENT).of(MistralAgent(max_new_tokens=256))
        attacker = Yielder(logger, tag=Tag.ATTACK).of(StatelessRemoteGeminiAgent())
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
            AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())

        print("agents done", flush=True)
        strat_path = Path("core/core_strats") / "GOATATTACKER.strat"
        multiturn_strat = Yielder(logger, Tag.GOAT).of(
            GoatAttack(
                attack_model=attacker,
                evaluator=evaluator,
                refusal_model=refual_ev,
                chain_of_thought_strategy=Strategy(strat_path),
                approaches=[PersonaModificationApproach(), RefusalSuppressionApproach()],
                goal=goal,
                rounds=10,
                max_refusals=10,
                evaluate_every=1,
            )
        )

        print("multiturn")

        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = agent.apply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"{convo}")

        with open(f"{BASE_FOLDER}/goat_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        print("=====================")
        print("=====================")
        print("=====================")
        print("=====================")

        print(multiturn_strat.H_A.promptify())

        print("=====================")
        print("=====================")
        print("=====================")
        print("=====================")

        for message in convo.latest_oneshot():
            print(f"{message}")
            print("\n\n\n")
            print("=====================")
        print("=====================")
        # print(agent.apply_strategy(Simple("How to rob a bank?")))
