print("import core.utils", flush=True)
from core.utils import get_device

print(f"{get_device()=}", flush=True)
print("importing core.attacks", flush=True)
from core.models import unload_all_models
import gc
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
    RACEAttack,
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
from core.attacks.helpers.simulators.selfplay import SPSProps
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path

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

safety_df = BertievidgenSimpleSafetyTests().get_first(80)
batch = Batch(max_wait_seconds=10, max_batch_size=30)
batch1 = Batch(max_wait_seconds=10, max_batch_size=30)
batch2 = Batch(max_wait_seconds=10, max_batch_size=30)
batch3 = Batch(max_wait_seconds=10, max_batch_size=30)
batch4 = Batch(max_wait_seconds=10, max_batch_size=30)
batch5 = Batch(max_wait_seconds=10, max_batch_size=30)
batch6 = Batch(max_wait_seconds=10, max_batch_size=30)

batch_vllm1A = Batch(max_wait_seconds=10, max_batch_size=30, name="VLLM1")
batch_vllm1 = Batch(max_wait_seconds=10, max_batch_size=30, name="VLLM1")
batch_vllm2 = Batch(max_wait_seconds=10, max_batch_size=90, name="VLLM2")
batch_vllm22 = Batch(max_wait_seconds=10, max_batch_size=30, name="VLLM22")

VLLM_HOST_BAGEL = "dgxh100-precis-wn02.grid.pub.ro"
VLLM_HOST_QWEN = "dgxa100-ncit-wn03.grid.pub.ro"


def get_base_folder(index):
    if index == 0:
        return (
            "datasets/race_bss_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl_powerful_r10"
        )
    elif index == 1:
        return "datasets/race_bss_qwen25_vs_derta__async_batch_bagel_eval_3repl_powerful_r10"
    elif index == 2:
        return "datasets/race_bss_qwen25_vs_door__async_batch_bagel_eval_3repl_powerful_r10"
    elif index == 3:
        return "datasets/race_bss_qwen25_vs_guardllama__async_batch_bagel_eval_3repl_powerful_r10"
    elif index == 4:
        return (
            "datasets/race_bss_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl_powerful_r10"
        )
    elif index == 5:
        return "datasets/race_bss_qwen25_vs_cbllama__async_batch_bagel_eval_3repl_powerful_r10"
    elif index == 6:
        return "datasets/race_bss_qwen25_vs_guardresp__async_batch_bagel_eval_3repl_powerful_r10"
    else:
        raise ValueError("Invalid index")


for i in range(7):
    base_folder = get_base_folder(i)
    # create folder if not exists
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
        return LlamaAgent(max_new_tokens=512).with_batch(batch1)
    elif index == 2:
        return LlamaAgent(
            model="wicai24/Llama-3-8B-Instruct-W-DOOR-exponential", max_new_tokens=512
        ).with_batch(batch2)
    elif index == 3:
        return AutoAgentWithGuard(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            guard_model="Qwen/Qwen3Guard-Gen-8B",
            max_new_tokens=512,
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
            guard_type="response_moderation",
        ).with_batch(batch6)


async def run_one(row, agent_index=0):
    goal = row["prompt"]
    print(f"\n[{row['id']}] Starting: {goal}")
    base_folder = get_base_folder(agent_index)
    with MessageFileLogger(
        output_folder_name=f"{base_folder}/race_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        # check if {base_folder}/race_bss_{row['id']} contains info.txt, if yes, skip
        if logger.already_saved():
            print(f"[{row['id']}] already saved, skipping...", flush=True)
            return row["id"], None, None
        victim_sim = Yielder(logger, tag=Tag.AGENT_VICTIM_SIM).of(get_agent(agent_index))
        agent = Yielder(logger, tag=Tag.AGENT).of(get_agent(agent_index))

        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(
                host=VLLM_HOST_BAGEL,
                model="/models/bagel-34b-v0.2",
            ).with_batch(batch_vllm1A)
        )
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
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
            )
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())
        realigner = Yielder(logger, tag=Tag.REALIGNER).of(
            AgentReAligner(
                StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_QWEN, model="/models/Qwen2.5-32B-Instruct", port=8002
                ).with_batch(batch_vllm1)
            )
        )
        gge = Yielder(logger, tag=Tag.GGE).of(
            AgentGGEEvaluator(
                generator=StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_BAGEL,
                    model="/models/bagel-34b-v0.2",
                    temperature=0.7,
                    max_tokens=16,
                    stop=["<|im_end|>", "<|endoftext|>", "User:", "Assistant:"],
                    presence_penalty=0.0,
                    repetition_penalty=1.1,
                    frequency_penalty=0.0,
                    top_p=0.9,
                    structured_outputs={"regex": r"-?\d+(\.\d+)?"},
                ).with_batch(batch_vllm22)
            )
        )
        rqg = Yielder(logger, tag=Tag.RQG).of(
            AgentRandomQueryGenerator(
                batch=5,
                generator=StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_QWEN, model="/models/Qwen2.5-32B-Instruct", port=8002
                ),
            )
        )

        print("agents done", flush=True)
        strat_path = Path("core/core_strats") / "RACEATTACKER.strat"
        multiturn_strat = Yielder(logger, Tag.RACE).of(
            RACEAttack(
                attack_model=attacker,
                gge_evaluator=gge,
                goal_evaluator=evaluator,
                refusal_model=refual_ev,
                refusalFeedback=realigner,
                random_query_generator=rqg,
                goal=goal,
                max_refusals=15,
                rounds=10,
                candidates_per_round=3,
                samples_per_candidate=1,
                victim_simulator=victim_sim,
                sps_props=SPSProps(depth=1, self_play_strategy=Strategy(file_path=strat_path)),
                stateful=False,
                verbose=False,
            )
        )
        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"[{goal}] done", flush=True)
        with open(f"{base_folder}/race_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio
import traceback


async def run_all(safety_df, max_concurrent: int = 5, _agent_index=0):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row, agent_index):
        async with semaphore:
            try:
                return await run_one(row, agent_index)
            except Exception as e:
                traceback.print_exc()
                print(f"[AGI:{agent_index}] [{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    async def run_model_index(agent_index):
        tasks = []
        for _, row in safety_df.iterrows():
            tasks.append(run_with_semaphore(row, agent_index))
        return await asyncio.gather(*tasks)

    batches_per_index = [batch, batch1, batch2, batch3, batch4, batch5, batch6]
    all_tasks = []
    for agent_index in [1, 2, 3, 4, 5, 6]:
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

    # tasks = []
    # for agent_index in range(4):
    #     tasks.extend([run_with_semaphore(row, agent_index) for _, row in safety_df.iterrows()])
    # all_tasks = await asyncio.gather(*tasks)
    # await batch.end()
    # return all_tasks


import sys

if __name__ == "__main__":
    agent_index = int(sys.argv[1])
    results = asyncio.run(run_all(safety_df, max_concurrent=60, _agent_index=agent_index))
