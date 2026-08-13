print("import core.utils", flush=True)
from core.utils import get_device

print(f"{get_device()=}", flush=True)
print("importing core.attacks", flush=True)
import sys
import gc
from core.models import unload_all_models

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
    LlamaAgent,
    AutoAgentWithGuard,
    AutoAgentWithNemotronGuard,
)
import gc

from core.attacks.agents.batch import Batch

from core.attacks.evaluators import AgentPEC7Evaluator, PEC7
from core.attacks.datastore import MessageFileLogger
from core.attacks.datastore.yielder import Yielder, Tag
from core.datasets.safety.donotanswer import LAODoNotAnswer
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.xstest import XSTest
from core.datasets.safety.advbench import AdvBench
from random import choices
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path
from core.datasets.safety.harmbench import HarmBench

safety_df = HarmBench().get_first(400)
batch = Batch(max_wait_seconds=10, max_batch_size=30)
batch1 = Batch(max_wait_seconds=10, max_batch_size=30)
batch2 = Batch(max_wait_seconds=10, max_batch_size=30)
batch3 = Batch(max_wait_seconds=10, max_batch_size=30)
batch4 = Batch(max_wait_seconds=10, max_batch_size=30)
batch5 = Batch(max_wait_seconds=10, max_batch_size=30)
batch6 = Batch(max_wait_seconds=10, max_batch_size=30)
batch7 = Batch(max_wait_seconds=10, max_batch_size=30)
batch8 = Batch(max_wait_seconds=10, max_batch_size=30)

batch_vllm1 = Batch(max_wait_seconds=10, max_batch_size=30, name="VLLM1")
batch_vllm2 = Batch(max_wait_seconds=30, max_batch_size=90, name="VLLM2")

VLLM_HOST_BAGEL = "dgxa100-ncit-wn02.grid.pub.ro"
VLLM_HOST_QWEN = "dgxh100-precis-wn02.grid.pub.ro"


def get_base_folder(index):
    if index == 0:
        return "datasets/goat_hb_qwen3n_vs_circuitbreaker__async_batch_bagel_eval_3repl"
    elif index == 1:
        return "datasets/goat_hb_qwen3n_vs_dertains__async_batch_bagel_eval_3repl"
    elif index == 2:
        return "datasets/goat_hb_qwen3n_vs_door__async_batch_bagel_eval_3repl"
    elif index == 3:
        return "datasets/goat_hb_qwen3n_vs_guardr__async_batch_bagel_eval_3repl"
    elif index == 4:
        return "datasets/goat_hb_qwen3n_vs_llamainstruct__async_batch_bagel_eval_3repl"
    elif index == 5:
        return "datasets/goat_hb_qwen3n_vs_cbllama__async_batch_bagel_eval_3repl"
    elif index == 6:
        return "datasets/goat_hb_qwen3n_vs_guardq__async_batch_bagel_eval_3repl"
    elif index == 7:
        return "datasets/goat_hb_qwen3n_vs_nemor__async_batch_bagel_eval_3repl"
    elif index == 8:
        return "datasets/goat_hb_qwen3n_vs_nemoq__async_batch_bagel_eval_3repl"

    else:
        raise ValueError("Invalid index")


for i in range(9):
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
from core.models import unload_all_models


def get_agent(index):
    if index == 0:
        return MistralAgent(max_new_tokens=512).with_batch(batch)
    elif index == 1:
        return LlamaAgent(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            peft_model="Youliang/llama3-8b-instruct-lora-derta-100step",
            max_new_tokens=512,
        ).with_batch(batch1)
        # return LlamaAgent(max_new_tokens=512).with_batch(batch1)
    elif index == 2:
        return LlamaAgent(
            model="wicai24/Llama-3-8B-Instruct-W-DOOR-exponential", max_new_tokens=512
        ).with_batch(batch2)
    elif index == 3:
        return AutoAgentWithGuard(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            guard_model="Qwen/Qwen3Guard-Gen-8B",
            max_new_tokens=512,
            guard_type="response_moderation",
        ).with_batch(batch3)
    elif index == 4:
        return AutoAgent(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=512,
        ).with_batch(batch4)
    elif index == 5:
        return LlamaAgent(model="GraySwanAI/Llama-3-8B-Instruct-RR", max_new_tokens=512).with_batch(
            batch5
        )
    elif index == 6:
        return AutoAgentWithGuard(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            guard_model="Qwen/Qwen3Guard-Gen-8B",
            max_new_tokens=512,
            guard_type="question_moderation",
        ).with_batch(batch6)
    elif index == 7:
        return AutoAgentWithNemotronGuard(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=512,
            guard_type="response_moderation",
        ).with_batch(batch7)
    elif index == 8:
        return AutoAgentWithNemotronGuard(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=512,
            guard_type="question_moderation",
        ).with_batch(batch8)


async def run_one(row, agent_index=0, shared_session=None, i=0):
    goal = row["Behavior"]
    print(f"\n[{row['id']}] Starting: {goal}")
    base_folder = get_base_folder(agent_index)
    suff = "" if i == 0 else f"_i{i}"
    with MessageFileLogger(
        output_folder_name=f"{base_folder}/goat_hb_{row['id']}{suff}", buffer_max_size=20
    ) as logger:
        if logger.already_saved():
            print(f"[{base_folder}/{row['id']}] already saved, skipping...", flush=True)
            return row["id"], None, None
        agent = Yielder(logger, tag=Tag.AGENT).of(get_agent(agent_index))
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(
                host=VLLM_HOST_QWEN, model="/models/Qwen3-Next-80B-A3B-Instruct-FP8", port=8002
            ).with_batch(batch_vllm1)
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_BAGEL,
                    model="/models/bagel-34b-v0.2",
                    temperature=0.7,
                    max_tokens=1024,
                    stop=["<|im_end|>", "<|endoftext|>", "User:", "Assistant:"],
                    presence_penalty=0.0,
                    repetition_penalty=1.1,
                    frequency_penalty=0.0,
                    top_p=0.9,
                    structured_outputs={"json": PEC7.PEC7.schema()},
                ).with_batch(batch_vllm2),
                replicas=3,
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
        with open(f"{base_folder}/goat_hb_{row['id']}{suff}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio

import aiohttp
import traceback


async def run_all(safety_df, max_concurrent: int = 5, _agent_index=0):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row, agent_index, shared_session, i=0):
        async with semaphore:
            try:
                return await run_one(row, agent_index, shared_session, i)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"[{row['id']}] OOM error, skipping: {e}", flush=True)
                return row["id"], None, None
            except Exception as e:
                traceback.print_exc()
                print(f"[AGI:{agent_index}] [{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    async def run_model_index(agent_index):
        tasks = []
        for _, row in safety_df.iterrows():
            for k in range(1):
                tasks.append(run_with_semaphore(row, agent_index, None, k))
        return await asyncio.gather(*tasks)

    batches_per_index = [batch, batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8]
    all_tasks = []
    for agent_index in [1, 2, 3, 4, 5, 6, 7, 8]:
        if agent_index != _agent_index:
            continue
        all_tasks.extend(await run_model_index(agent_index))
        await batches_per_index[agent_index].end()
        unload_all_models()  # 1. drop model references from cache
        gc.collect()  # 2. force Python GC to release any lingering refs
        torch.cuda.empty_cache()
    await batch_vllm1.end()
    await batch_vllm2.end()
    return all_tasks


if __name__ == "__main__":
    agent_index = int(sys.argv[1])
    results = asyncio.run(run_all(safety_df, max_concurrent=60, _agent_index=agent_index))
