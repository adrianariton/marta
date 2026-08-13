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
    StatelessSelfHostedOllamaAgent,
    StatelessSelfHostedVLLMAgent,
)

from core.attacks.agents.batch import Batch

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.advbench import AdvBench
from random import choices


safety_df = BertievidgenSimpleSafetyTests().get_first(n=80)
batch = Batch(max_wait_seconds=3, max_batch_size=3)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "GraySwanAI/Mistral-7B-Instruct-RR"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load adi LoRA adapter on top
model = PeftModel.from_pretrained(model, "./mistral-rr-safety-sft/final")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
BASE_FOLDER = "datasets/crescendo_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval"
BASE_FOLDER_ADI = "datasets/crescendo_bss_qwen_vs_adi__async_batch_qwen_eval"
with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")
with open(f"{BASE_FOLDER_ADI}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")


async def run_one(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/crescendo_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            MistralAgent(max_new_tokens=512).with_batch(batch)
            # MistralAgent.from_model_and_tokenizer(model, tokenizer, max_new_tokens=512).with_batch(
            #     batch
            # )
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            # StatelessSelfHostedOllamaAgent(
            #     host="dgxa100-ncit-wn01.grid.pub.ro", think="low", model="llama3.3:70b", ctx=16384
            # )
            StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
            )
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
                verbose=False,
            )
        )

        multiturn_strat.reset_goal_achieved()
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)

        with open(f"{BASE_FOLDER}/crescendo_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        print(f"[{row['id']}] Done")
        return row["id"], convo, multiturn_strat


async def run_one_adi(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER_ADI}/crescendo_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            # MistralAgent(max_new_tokens=512).with_batch(batch)
            MistralAgent.from_model_and_tokenizer(model, tokenizer, max_new_tokens=512).with_batch(
                batch
            )
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            # StatelessSelfHostedOllamaAgent(
            #     host="dgxa100-ncit-wn01.grid.pub.ro", think="low", model="llama3.3:70b", ctx=16384
            # )
            StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
            )
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
                verbose=False,
            )
        )

        multiturn_strat.reset_goal_achieved()
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)

        with open(f"{BASE_FOLDER_ADI}/crescendo_bss_{row['id']}/info.txt", "w+") as f:
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

                traceback.print_exc()
                print(f"[{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    async def run_with_semaphore_adi(row):
        async with semaphore:
            try:
                return await run_one_adi(row)
            except Exception as e:

                traceback.print_exc()
                print(f"[{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    tasks = [run_with_semaphore_adi(row) for _, row in safety_df.iterrows()] + [
        run_with_semaphore(row) for _, row in safety_df.iterrows()
    ]  # Run both BSS and ADI versions
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
