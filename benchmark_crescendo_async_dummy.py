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

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7, DummyPEC7Evaluator
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.advbench import AdvBench
from random import choices
from core.attacks.agents.batch import Batch

batch = Batch(max_wait_seconds=2)
safety_df = BertievidgenSimpleSafetyTests().get_first(n=10)

BASE_FOLDER = "datasets/crescendo_bss_gemini_vs_circuitbreaker__async_batch_2"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")


async def run_one(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/crescendo_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            DummyAgent(format=f"{goal}$answer").with_batch(batch)
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            DummyAgent(format=f"Q: {goal} $question\nS: $text")
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            DummyPEC7Evaluator()
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
                iterations=5,
                on_new_round=lambda round: logger.new_id(),
                question_refusal_model=question_refual_ev,
            )
        )

        multiturn_strat.reset_goal_achieved()
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)

        with open(f"{BASE_FOLDER}/crescendo_bss_{row['id']}/info.txt", "w+") as f:
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
    r = await asyncio.gather(*tasks)
    await batch.end()
    return r


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=5))


# for i, row in safety_df.iterrows():
#     goal = row["prompt"]
#     print("\n\n\n\n============================================================")
#     print("************************************************************\n\n")
#     print(f"{goal=}")
#     with MessageFileLogger(
#         output_folder_name=f"{BASE_FOLDER}/crescendo_bss_{row['id']}", buffer_max_size=20
#     ) as logger:
#         agent = Yielder(logger, tag=Tag.AGENT).of(MistralAgent(max_new_tokens=512))
#         attacker = Yielder(logger, tag=Tag.ATTACK).of(
#             # DummyAgent(format="Q: $question\nS: $text", type_="input")
#             StatelessRemoteGeminiAgent()
#         )
#         evaluator = Yielder(logger, tag=Tag.EVAL).of(
#             # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent(model_name="gemini-2.5-pro"))
#             AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
#         )  # AutoAgent(model="Qwen/Qwen3-4B-Instruct-2507")
#         refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())
#         question_refual_ev = Yielder(logger, tag=Tag.IS_QUESTION_REFUSED).of(
#             HeuristicRefusalClassifier()
#         )

#         print("agents done", flush=True)
#         multiturn_strat = Yielder(logger, Tag.CRESCENDO).of(
#             CrescendomationAttack(
#                 attack_model=attacker,
#                 evaluator=evaluator,
#                 refusal_model=refual_ev,
#                 goal=goal,
#                 rounds=8,  # 8 mesaje per iteratie
#                 iterations=5,  # 5 iterations
#                 on_new_round=lambda round: logger.new_id(),
#                 question_refusal_model=question_refual_ev,
#             )
#         )

#         print("multiturn")

#         multiturn_strat.reset_goal_achieved()
#         # This is async
#         convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
#         print(f"{convo}")

#         with open(f"{BASE_FOLDER}/crescendo_bss_{row['id']}/info.txt", "w+") as f:
#             f.write(f"{multiturn_strat.goal_achieved_info()}")

#         print("=====================")
#         print("=====================")
#         print("=====================")
#         print("=====================")

#         print(multiturn_strat.H_B.promptify())

#         print("=====================")
#         print("=====================")
#         print("=====================")
#         print("=====================")

#         for message in convo.latest_oneshot():
#             print(f"{message}")
#             print("\n\n\n")
#             print("=====================")
#         print("=====================")
# print(agent.apply_strategy(Simple("How to rob a bank?")))
