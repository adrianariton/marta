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
    StatelessSelfHostedVLLMAgent,
    StatelessSelfHostedOllamaAgent,
    LlamaAgent,
    AutoAgentWithGuard,
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

VLLM_HOST = "dgxh100-precis-wn01.grid.pub.ro"
# rm -rf datasets/crescendo_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval/* datasets/crescendo_bss_qwen_vs_derta__async_batch_qwen_eval/* datasets/crescendo_bss_qwen_vs_door__async_batch_qwen_eval/* datasets/crescendo_bss_qwen_vs_guard__async_batch_qwen_eval/* datasets/crescendo_bss_qwen_vs_adi__async_batch_qwen_eval/*


def get_base_folder(index):
    if index == 0:
        return "datasets/crescendo_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval"
    elif index == 1:
        return "datasets/crescendo_bss_qwen_vs_derta__async_batch_qwen_eval"
    elif index == 2:
        return "datasets/crescendo_bss_qwen_vs_door__async_batch_qwen_eval"
    elif index == 3:
        return "datasets/crescendo_bss_qwen_vs_guard__async_batch_qwen_eval"
    elif index == 4:
        return "datasets/crescendo_bss_qwen_vs_adi__async_batch_qwen_eval"
    else:
        raise ValueError("Invalid index")


for i in range(5):
    base_folder = get_base_folder(i)
    Path(base_folder).mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {base_folder}", flush=True)
    with open(f"{base_folder}/_choices.txt", "w+") as f:
        _indices = safety_df[["id"]]
        f.write(f"{_indices.to_dict()}")

from core.attacks.helpers.approaches import RefusalSuppressionApproach, PersonaModificationApproach

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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


def get_agent(index):
    if index == 0:
        return MistralAgent(max_new_tokens=512).with_batch(batch)
    elif index == 1:
        return LlamaAgent(max_new_tokens=512).with_batch(batch)
    elif index == 2:
        return LlamaAgent(
            model="wicai24/Llama-3-8B-Instruct-W-DOOR-exponential", max_new_tokens=512
        ).with_batch(batch)
    elif index == 3:
        return AutoAgentWithGuard(
            guard_model="Qwen/Qwen3Guard-Gen-8B", max_new_tokens=512
        ).with_batch(batch)
    elif index == 4:
        return MistralAgent.from_model_and_tokenizer(
            model, tokenizer, max_new_tokens=512
        ).with_batch(batch)


async def run_one(row, agent_index=0, shared_session=None):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")
    base_folder = get_base_folder(agent_index)
    with MessageFileLogger(
        output_folder_name=f"{base_folder}/crescendo_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        if logger.already_saved():
            print(f"[{row['id']}] already saved, skipping...", flush=True)
            return row["id"], None, None
        agent = Yielder(logger, tag=Tag.AGENT).of(get_agent(agent_index))
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(host=VLLM_HOST, session=shared_session)
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host=VLLM_HOST, session=shared_session)
            )
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
                verbose=False,
                history_message_trunc_limit=512 * 4,
            )
        )

        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"[{goal}] done", flush=True)
        with open(f"{base_folder}/crescendo_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio

import aiohttp
import traceback


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(
        limit=12000,  # Total simultaneous connections
        keepalive_timeout=300,  # How many seconds to keep an idle connection open
        enable_cleanup_closed=True,
    )

    async def run_with_semaphore(row, agent_index, shared_session):
        async with semaphore:
            try:
                return await run_one(row, agent_index, shared_session)
            except Exception as e:
                traceback.print_exc()
                print(f"[AGI:{agent_index}] [{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    # async with aiohttp.ClientSession(connector=connector) as shared_session:
    tasks = []
    for agent_index in [4, 0, 2, 1, 3]:
        tasks.extend(
            [run_with_semaphore(row, agent_index, None) for _, row in safety_df.iterrows()]
        )
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
