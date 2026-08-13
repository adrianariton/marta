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
from core.datasets.safety.advbench import AdvBench
from random import choices
from core.attacks.helpers.attackers.strategy_attacker import Strategy
from pathlib import Path

safety_df = BertievidgenSimpleSafetyTests().get_first(n=80)
batch = Batch(max_wait_seconds=3, max_batch_size=3)

VLLM_HOST = "dgxh100-precis-wn01.grid.pub.ro"


def get_base_folder(index):
    if index == 0:
        return "datasets/fitd_bss_qwen_vs_circuitbreaker__async_batch_qwen_eval"
    elif index == 1:
        return "datasets/fitd_bss_qwen_vs_derta__async_batch_qwen_eval"
    elif index == 2:
        return "datasets/fitd_bss_qwen_vs_door__async_batch_qwen_eval"
    elif index == 3:
        return "datasets/fitd_bss_qwen_vs_guard__async_batch_qwen_eval"
    else:
        raise ValueError("Invalid index")


for i in range(4):
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
        return MistralAgent(max_new_tokens=1024).with_batch(batch)
    elif index == 1:
        return LlamaAgent(max_new_tokens=1024).with_batch(batch)
    elif index == 2:
        return LlamaAgent(
            model="wicai24/Llama-3-8B-Instruct-W-DOOR-exponential", max_new_tokens=1024
        ).with_batch(batch)
    elif index == 3:
        return AutoAgentWithGuard(
            guard_model="Qwen/Qwen3Guard-Gen-8B", max_new_tokens=1024
        ).with_batch(batch)


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
        attacker = Yielder(logger, tag=Tag.ATTACK).of(StatelessSelfHostedVLLMAgent(host=VLLM_HOST))
        evaluator = Yielder(logger, tag=Tag.EVAL).of(
            AgentPEC7Evaluator(generator=StatelessSelfHostedVLLMAgent(host=VLLM_HOST))
        )
        refual_ev = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(HeuristicRefusalClassifier())
        bridger = Yielder(logger, tag=Tag.BRIDGER).of(
            AgentBridger(batch=3, generator=StatelessSelfHostedVLLMAgent(host=VLLM_HOST))
        )
        realigner = Yielder(logger, tag=Tag.REALIGNER).of(SimplerPredefinedPromptReAligner())
        midder = Yielder(logger, tag=Tag.MIDDER).of(
            AgentMidQuery(StatelessSelfHostedVLLMAgent(host=VLLM_HOST))
        )
        alignment_tester = Yielder(logger, tag=Tag.PRALIGNMENTTESTER).of(
            AgentAlignmentTester(generator=StatelessSelfHostedVLLMAgent(host=VLLM_HOST))
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
                max_refusals=20,
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


async def run_all(safety_df, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(row, agent_index):
        async with semaphore:
            try:
                return await run_one(row, agent_index)
            except Exception as e:
                traceback.print_exc()
                print(f"[AGI:{agent_index}] [{row['id']}] FAILED: {e}", flush=True)
                return row["id"], None, None

    tasks = []
    for agent_index in range(4):
        tasks.extend([run_with_semaphore(row, agent_index) for _, row in safety_df.iterrows()])
    all_tasks = await asyncio.gather(*tasks)
    await batch.end()
    return all_tasks


if __name__ == "__main__":
    results = asyncio.run(run_all(safety_df, max_concurrent=6))
