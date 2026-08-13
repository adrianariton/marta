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
)

from core.attacks.agents.batch import Batch

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.donotanswer import LAODoNotAnswer
from core.datasets.safety.advbench import AdvBench
from random import choices


safety_df = AdvBench().get_first(n=200)
batch = Batch(max_wait_seconds=3, max_batch_size=3)

BASE_FOLDER = "datasets/crescendo_adv_gemini_vs_circuitbreaker__async_batch"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")


async def run_one(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/crescendo_adv_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            MistralAgent(max_new_tokens=512).with_batch(batch)
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(StatelessRemoteGeminiAgent())
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
            # AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())
        question_refual_ev = Yielder(logger, tag=Tag.IS_QUESTION_REFUSED).of(
            HeuristicRefusalClassifier()
        )

        multiturn_strat = Yielder(logger, Tag.CRESCENDO).of(
            CrescendomationAttack(
                attack_model=attacker,
                evaluator=evaluator,
                refusal_model=refual_ev,
                goal=goal,
                rounds=8,
                iterations=3,
                on_new_round=lambda round: logger.new_id(),
                question_refusal_model=question_refual_ev,
                verbose=False,
            )
        )

        multiturn_strat.reset_goal_achieved()
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)

        with open(f"{BASE_FOLDER}/crescendo_adv_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        print(f"[{row['id']}] Done")
        return row["id"], convo, multiturn_strat


import asyncio


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row):
        async with semaphore:
            try:
                return await run_one(row)
            except Exception as e:
                print(f"[{row['id']}] FAILED: {e}")
                return row["id"], None, None

    tasks = [run_with_semaphore(row) for _, row in safety_df.iterrows()]
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
