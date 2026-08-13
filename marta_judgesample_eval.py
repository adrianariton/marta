import asyncio
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tempfile
from core.attacks.agents.batch import Batch
from core.attacks.agents import StatelessSelfHostedVLLMAgent
from core.attacks.evaluators import AgentPEC7Evaluator, PEC7, DummyPEC7Evaluator
from core.attacks.interfaces import LinearHistory, OneShotConversation, Conversation

# ── Config ─────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("marta_corrected_neweval_checkpoint.parquet")
OUTPUT_PATH     = Path("marta_corrected_neweval/")
CHECKPOINT_EVERY = 15360

EVALUATOR_DEFS = {
    "phi-4-AWQ":                       (8001, "/models/phi-4-AWQ",                       ["<|end|>", "<|endoftext|>"]),
    "Mixtral-8x22B-Instruct-v0.1-AWQ":(8002, "/models/Mixtral-8x22B-Instruct-v0.1-AWQ",["</s>"]),
    "Qwen2.5-32B-Instruct-AWQ":       (8003, "/models/Qwen2.5-32B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
    "Qwen2.5-72B-Instruct-AWQ":       (8014, "/models/Qwen2.5-72B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
}

EVAL_COLS = [f"eval_PEC7_{name}" for name in EVALUATOR_DEFS]

# ── Build batches & evaluators ─────────────────────────────────────────────────

batches = [
    Batch(max_wait_seconds=3, max_batch_size=90, name=f"VLLM_{8001 + i}")
    for i in range(len(EVALUATOR_DEFS))
]

evaluators = {}
for i, (name, (port, model_path, stop_tokens)) in enumerate(EVALUATOR_DEFS.items()):
    if "mixtral" in name.lower():
        structured_outputs = None
    else:
        structured_outputs = {
            "json": PEC7.PEC7.schema()
        }
    evaluators[name] = AgentPEC7Evaluator(
        generator=StatelessSelfHostedVLLMAgent(
            host="localhost",
            port=port,
            model=model_path,
            temperature=0.7,
            max_tokens=1024,
            stop=stop_tokens + ["User:", "Assistant:"],
            presence_penalty=0.0,
            repetition_penalty=1.1,
            frequency_penalty=0.0,
            top_p=0.9,
            # structured_outputs={"json": PEC7.PEC7.schema()},
            structured_outputs=structured_outputs,
            default_system_message=not ("mixtral" in name.lower())

        ),#.with_batch(batches[i]),
        replicas=3,
        no_system="mixtral" in name.lower()
    )
    # evaluators[name] = DummyPEC7Evaluator()

# ── Helpers ────────────────────────────────────────────────────────────────────

    
# but instead of reading from marta_corrected,
# read from AIQVB100.jsonl

# like this 

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import os


def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def pec7_to_json(pec7) -> str:
    if isinstance(pec7, Exception):
        return f'{{"error": "{str(pec7)[:200]}"}}'
    try:
        return pec7.to_jsonstring()
    except Exception as e:
        return f'{{"error": "serialization failed: {str(e)[:100]}"}}'

async def evaluate_row(user, system) -> dict:
    async def _eval_one(name, evaluator: AgentPEC7Evaluator):
        # Fetch the port dynamically based on the current evaluator's name key
        port = EVALUATOR_DEFS.get(name, ("Unknown",))[0]
        # print(f"{evaluator.generator.identifier.lower()=}", flush=True)
        if 'mixtral' in evaluator.generator.identifier.lower():
            s = None
            # print("mixtral no", flush=True)
        else:
            s = system
        try:
            pec7 = await evaluator.aevaluate_user_system(user=user, system=s)
            return name, pec7_to_json(pec7)
        except Exception as e:
            # Print the error alongside the port to the console immediately
            print(f"\n[ERROR] Evaluator '{name}' on port {port} failed: {e}")
            # Inject the port into the JSON output string for your dataset log
            return name, f'{{"error": "Port {port}: {str(e)[:200]}"}}'

    results = await asyncio.gather(
        *[_eval_one(name, ev) for name, ev in evaluators.items()]
    )
    return {f"eval_PEC7_{name}": val for name, val in results}

async def worker(queue, convos, pbar):
    """
    Consumes indices from the queue and processes them sequentially.
    Ensures that only a strict number of workers hit the servers.
    """
    while not queue.empty():
        try:
            i = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        
        if convos[i]['frame']['eval'] is not None:
            
        
            msgs = ((convos[i]['frame']['eval']['conversation']['_messages']))
            all_msgs = ([''.join([t['text'] for t in m[1]['content']]) for m in msgs])
            system = all_msgs[0]
            user = all_msgs[1]
            
            result = await evaluate_row(user=user, system=system)
            for col, val in result.items():
                convos[i][col] = val    
        pbar.update(1)
        queue.task_done()

async def main():   
    convos = read_jsonl('human_labels/AIQVB100.jsonl')
    done_indices = set([])
    pending = [idx for idx in range(len(convos)) if idx not in done_indices]
    CONCURRENCY = 32 
    
    with tqdm(total=len(pending), desc="Evaluating", unit="row") as pbar:
        batch = pending
        queue = asyncio.Queue()
        for idx in batch:
            queue.put_nowait(idx)

        # Create strict, bounded worker tasks
        workers = [
            asyncio.create_task(worker(queue, convos, pbar))
            for _ in range(CONCURRENCY)
        ]
        
        await asyncio.gather(*workers)
        
    with open('human_labels/MORAAIQVB100.jsonl', 'w', encoding='utf-8') as f:
        for item in convos:
            f.write(json.dumps(item) + '\n')

        
    
    



if __name__ == "__main__":
    asyncio.run(main())
    
    
    