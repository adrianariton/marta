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
    LATSAttack,
)
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
from core.attacks.evaluators.SimilarityEvaluator import SimilarityEvaluator

safety_df = BertievidgenSimpleSafetyTests().get_first(n=10)

BASE_FOLDER = "datasets/lats_bss_gemini_vs_circuitbreaker"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")

for i, row in safety_df.iterrows():
    goal = row["prompt"]
    print("\n\n\n\n============================================================")
    print("************************************************************\n\n")
    print(f"{goal=}")
    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/lats_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(MistralAgent(max_new_tokens=1024))
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
        )
        print("agents done", flush=True)
        multiturn_strat = Yielder(logger, Tag.LATS).of(
            LATSAttack(
                similarityevaluator=SimilarityEvaluator(),
                jailbreakevaluator=evaluator,
                k=5,
                seed_pool_child_size=10,
                seed_pool_size=10,
                expansion_factor=2,
                D_max=2,
                goal=goal,
            )  ## 5 + 100 + 100 * 10 = 1150 DAMN!
        )

        multiturn_strat.reset_goal_achieved()
        print("multiturn")
        convo = agent.apply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"{convo}")

        with open(f"{BASE_FOLDER}/lats_bss_{row['id']}/info.txt", "w+") as f:
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
