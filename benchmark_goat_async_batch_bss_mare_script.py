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


def get_base_folder(index):
    if index == 0:
        return "datasets/goat_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval"
    elif index == 1:
        return "datasets/goat_bss_qwen_vs_derta__async_batch_qwen_eval"
    elif index == 2:
        return "datasets/goat_bss_qwen_vs_door__async_batch_qwen_eval"
    elif index == 3:
        return "datasets/goat_bss_qwen_vs_guard__async_batch_qwen_eval"
    else:
        raise ValueError("Invalid index")


for i in range(4):
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


async def run_one(row, agent_index=0, shared_session=None):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")
    base_folder = get_base_folder(agent_index)
    with MessageFileLogger(
        output_folder_name=f"{base_folder}/goat_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        if logger.already_saved():
            print(f"[{base_folder}/{row['id']}] already saved, skipping...", flush=True)
            return row["id"], None, None
        agent = Yielder(logger, tag=Tag.AGENT).of(get_agent(agent_index))
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(host=VLLM_HOST, session=shared_session)
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            # AgentPEC7Evaluator(generator=StatelessRemoteGeminiAgent())
            AgentPEC7Evaluator(
                generator=StatelessSelfHostedVLLMAgent(host=VLLM_HOST, session=shared_session)
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
        with open(f"{base_folder}/goat_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio

import aiohttp
import traceback


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row, agent_index, shared_session):
        async with semaphore:
            try:
                return await run_one(row, agent_index, shared_session)
            except Exception as e:
                print(f"[AGI:{agent_index}] [{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    connector = aiohttp.TCPConnector(
        limit=200,  # Total simultaneous connections
        keepalive_timeout=300,  # How many seconds to keep an idle connection open
    )

    async with aiohttp.ClientSession(connector=connector) as shared_session:
        tasks = []
        for agent_index in [2, 3]:
            tasks.extend(
                [
                    run_with_semaphore(row, agent_index, shared_session)
                    for _, row in safety_df.iterrows()
                ]
            )
        all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
