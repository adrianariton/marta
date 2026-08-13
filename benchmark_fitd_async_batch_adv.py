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
    StatelessRemoteClaudeAgent,
    CrescendomationAttack,
    AutoAgent,
    DummyAgent,
    AutoRefusalClassifier,
    HeuristicRefusalClassifier,
    MistralAgent,
    FITDAttack,
)

from core.attacks.agents.batch import Batch

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.donotanswer import LAODoNotAnswer
from core.datasets.safety.advbench import AdvBench
from random import choices


from core.attacks.helpers.realigner import (
    AgentReAligner,
    PRAlignmentTester,
    PredefinedPromptReAligner,
    AgentAlignmentTester,
)
from core.attacks.helpers.generators.bridger import AgentBridger
from core.attacks.helpers.mid import AgentMidQuery
from core.attacks.evaluators.MSMARCO import HfMSMARCOEvaluator
from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.advbench import AdvBench
from random import choices

safety_df = AdvBench().get_first(n=200)
batch = Batch(max_wait_seconds=3, max_batch_size=3)
mbatch = Batch(max_wait_seconds=3, max_batch_size=3)

BASE_FOLDER = "datasets/fitd_adv_gemini_vs_circuitbreaker__async_batch"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")


async def run_one(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/fitd_adv_{row['id']}", buffer_max_size=20
    ) as logger:
        mistral_ = MistralAgent(model="mistralai/Mistral-Small-24B-Instruct-2501").with_batch(
            mbatch
        )
        agent = Yielder(logger, tag=Tag.AGENT).of(
            MistralAgent(max_new_tokens=1024).with_batch(batch)
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(StatelessRemoteGeminiAgent())
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())
        bridger = Yielder(logger, tag=Tag.BRIDGER).of(AgentBridger(batch=5, generator=mistral_))
        realigner = Yielder(logger, tag=Tag.REALIGNER).of(PredefinedPromptReAligner())
        midder = Yielder(logger, tag=Tag.MIDDER).of(AgentMidQuery(mistral_))
        alignment_tester = Yielder(logger, tag=Tag.PRALIGNMENTTESTER).of(
            AgentAlignmentTester(generator=mistral_)
            # PRAlignmentTester(prompt_response_evaluator=HfMSMARCOEvaluator())
        )
        print("agents done", flush=True)
        multiturn_strat = Yielder(logger, Tag.FITD).of(
            FITDAttack(
                paraphraser=attacker,
                refusal_model=refual_ev,
                align_tester=alignment_tester,
                evaluator=evaluator,
                bridger=bridger,
                realigner=realigner,
                midder=midder,
                goal=goal,
                n=10,
                max_refusals=20,
                verbose=False,
            )
        )

        multiturn_strat.reset_goal_achieved()
        print("multiturn", flush=True)
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        print(f"{convo}")

        with open(f"{BASE_FOLDER}/fitd_adv_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        print(f"[{row['id']}] Done")
        return row["id"], convo, multiturn_strat


import asyncio
import traceback


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row):
        async with semaphore:
            try:
                return await run_one(row)
            except Exception as e:
                print(f"[{row['id']}] FAILED: {e}")
                traceback.print_exc()
                return row["id"], None, None

    tasks = [run_with_semaphore(row) for _, row in safety_df.iterrows()]
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
