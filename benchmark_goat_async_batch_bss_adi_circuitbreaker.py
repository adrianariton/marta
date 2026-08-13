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
    GoatAttack,
    StatelessSelfHostedVLLMAgent,
    StatelessSelfHostedOllamaAgent,
)

from core.attacks.agents.batch import Batch

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.donotanswer import LAODoNotAnswer
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.advbench import AdvBench
from random import choices
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path

safety_df = BertievidgenSimpleSafetyTests().get_first(n=80)
batch = Batch(max_wait_seconds=3, max_batch_size=3)

BASE_FOLDER = "datasets/goat_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval"
BASE_FOLDER_ADI = "datasets/goat_bss_qwen_vs_adi__async_batch_qwen_eval"

with open(f"{BASE_FOLDER}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")
with open(f"{BASE_FOLDER_ADI}/_choices.txt", "w+") as f:
    _indices = safety_df[["id"]]
    f.write(f"{_indices.to_dict()}")
from core.attacks.helpers.approaches import RefusalSuppressionApproach, PersonaModificationApproach

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


async def run_one(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER}/goat_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            MistralAgent(max_new_tokens=512).with_batch(batch)
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
                # generator=StatelessRemoteClaudeAgent(max_tokens=1024)
            )
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())

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
                verbose=False,
            )
        )
        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"[{goal}] done", flush=True)
        with open(f"{BASE_FOLDER}/goat_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


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


async def run_one_adi(row):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")

    with MessageFileLogger(
        output_folder_name=f"{BASE_FOLDER_ADI}/goat_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        agent = Yielder(logger, tag=Tag.AGENT).of(
            MistralAgent.from_model_and_tokenizer(model, tokenizer, max_new_tokens=512).with_batch(
                batch
            )
        )
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
            # AgentPEC7Evaluator(generator=StatelessRemoteClaudeAgent(max_tokens=1024))
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host="dgxh100-precis-wn01.grid.pub.ro")
            )
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())

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
                verbose=False,
            )
        )
        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"[{goal}] done", flush=True)
        with open(f"{BASE_FOLDER_ADI}/goat_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row):
        async with semaphore:
            try:
                return await run_one(row)
            except Exception as e:
                print(f"[GSAI] [{row['id']}] FAILED: {e}")
                return row["id"], None, None

    async def run_with_semaphore_adi(row):
        async with semaphore:
            try:
                return await run_one_adi(row)
            except Exception as e:
                print(f"[ADI] [{row['id']}] FAILED: {e}")
                return row["id"], None, None

    tasks = [
        run_with_semaphore_adi(row) for _, row in safety_df.iterrows()
    ]  # Run both BSS and ADI versions
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
