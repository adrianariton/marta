print("import core.utils", flush=True)
from core.utils import get_device

print(f"{get_device()=}", flush=True)
print("importing core.attacks", flush=True)

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
    StatelessSelfHostedVLLMAgent,
    StatelessSelfHostedOllamaAgent,
    LlamaAgent,
    AutoAgentWithGuard,
    FITDAttack,
)

from core.attacks.helpers.realigner import (
    AgentReAligner,
    PRAlignmentTester,
    PredefinedPromptReAligner,
    AgentAlignmentTester,
    SimplerPredefinedPromptReAligner,
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

safety_df = AdvBench().sample_randomly(80)
batch = Batch(max_wait_seconds=10, max_batch_size=10)
batch1 = Batch(max_wait_seconds=10, max_batch_size=10)
batch2 = Batch(max_wait_seconds=10, max_batch_size=10)
batch3 = Batch(max_wait_seconds=10, max_batch_size=10)
batch4 = Batch(max_wait_seconds=10, max_batch_size=10)
batch5 = Batch(max_wait_seconds=10, max_batch_size=10)
batch6 = Batch(max_wait_seconds=10, max_batch_size=10)

batch_vllm1 = Batch(max_wait_seconds=10, max_batch_size=30, name="VLLM1")
# here no need for high level since the model is used seldom and non parralel
batch_vllm2 = Batch(max_wait_seconds=10, max_batch_size=90, name="VLLM2")

VLLM_HOST_BAGEL = "dgxh100-precis-wn01.grid.pub.ro"
VLLM_HOST_QWEN = "dgxa100-ncit-wn03.grid.pub.ro"


def get_base_folder(index):
    if index == 0:
        return "datasets/fitd_adv_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl_r10"
    elif index == 1:
        return "datasets/fitd_adv_qwen25_vs_derta__async_batch_bagel_eval_3repl_r10"
    elif index == 2:
        return "datasets/fitd_adv_qwen25_vs_door__async_batch_bagel_eval_3repl_r10"
    elif index == 3:
        return "datasets/fitd_adv_qwen25_vs_guardllama__async_batch_bagel_eval_3repl_r10"
    elif index == 4:
        return "datasets/fitd_adv_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl_r10"
    elif index == 5:
        return "datasets/fitd_adv_qwen25_vs_cbllama__async_batch_bagel_eval_3repl_r10"
    elif index == 6:
        return "datasets/fitd_adv_qwen25_vs_guardresp__async_batch_bagel_eval_3repl_r10"
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
        output_folder_name=f"{base_folder}/fitd_bss_{row['id']}", buffer_max_size=20
    ) as logger:
        # check if {base_folder}/fitd_bss_{row['id']} contains info.txt, if yes, skip
        if logger.already_saved():
            print(f"[{row['id']}] already saved, skipping...", flush=True)
            return row["id"], None, None
        agent = Yielder(logger, tag=Tag.AGENT).of(get_agent(agent_index))
        attacker = Yielder(logger, tag=Tag.ATTACK).of(
            StatelessSelfHostedVLLMAgent(
                host=VLLM_HOST_QWEN, model="/models/Qwen2.5-32B-Instruct", port=8002
            )
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
        bridger = Yielder(logger, tag=Tag.BRIDGER).of(
            AgentBridger(
                batch=3,
                generator=StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_QWEN, model="/models/Qwen2.5-32B-Instruct", port=8002
                ),
            )
        )
        realigner = Yielder(logger, tag=Tag.REALIGNER).of(SimplerPredefinedPromptReAligner())
        midder = Yielder(logger, tag=Tag.MIDDER).of(
            AgentMidQuery(
                StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_QWEN, model="/models/Qwen2.5-32B-Instruct", port=8002
                ).with_batch(batch_vllm1)
            )
        )
        alignment_tester = Yielder(logger, tag=Tag.PRALIGNMENTTESTER).of(
            AgentAlignmentTester(
                generator=StatelessSelfHostedVLLMAgent(
                    host=VLLM_HOST_BAGEL,
                    model="/models/bagel-34b-v0.2",
                )
            )
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
                n=12,
                max_refusals=10,
                verbose=False,
            )
        )

        multiturn_strat.reset_goal_achieved()
        # multiturn_strat = GoalTrackerStrat(text_generator=attacker, goal="How to rob a bank?")
        convo = await agent.aapply_strategy(strat=multiturn_strat, use_trigger=True)
        print(f"[{goal}] done", flush=True)
        with open(f"{base_folder}/fitd_bss_{row['id']}/info.txt", "w+") as f:
            f.write(f"{multiturn_strat.goal_achieved_info()}")

        return row["id"], convo, multiturn_strat


import asyncio
import traceback


async def run_all(safety_df, max_concurrent: int = 5, _agent_index=0):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row, agent_index, s):
        async with semaphore:
            try:
                return await run_one(row, agent_index)
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
            tasks.append(run_with_semaphore(row, agent_index, None))
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


import sys

if __name__ == "__main__":
    agent_index = int(sys.argv[1])
    results = asyncio.run(run_all(safety_df, max_concurrent=30, _agent_index=agent_index))
