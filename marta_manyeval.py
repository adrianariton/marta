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

def json_format_to_linhist(row) -> LinearHistory:
    ocv = OneShotConversation.from_default(row["conversation"])
    ocv.assert_ready_for_completion()
    linhist = LinearHistory.from_oneshot(
        ocv
    )
    # linhist.add('assistant', resp)
    return linhist

def pec7_to_json(pec7) -> str:
    if isinstance(pec7, Exception):
        return f'{{"error": "{str(pec7)[:200]}"}}'
    try:
        return pec7.to_jsonstring()
    except Exception as e:
        return f'{{"error": "serialization failed: {str(e)[:100]}"}}'

async def evaluate_row(row, r, q, history: LinearHistory) -> dict:
    goal = row.get("goal", "")

    async def _eval_one(name, evaluator):
        # Fetch the port dynamically based on the current evaluator's name key
        port = EVALUATOR_DEFS.get(name, ("Unknown",))[0]
        try:
            pec7 = await evaluator.aevaluate(q=q, r=r, goal=goal, history=history)
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
import numpy as np

def normalize_nested(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
import json
class NumPyEncoder(json.JSONEncoder):
    """Recursively converts any numpy types inside nested structures to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
def load_checkpoint(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    for col in EVAL_COLS:
        if col not in df.columns:
            df[col] = ""

    done_indices: set[int] = set()
    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        try:
            ckpt = pd.read_parquet(CHECKPOINT_PATH, engine="pyarrow")
            ckpt.index = ckpt.index.astype(df.index.dtype)
            
            # Unpack the JSON text blocks back into real python lists/dicts
            if "conversation" in ckpt.columns:
                ckpt["conversation"] = ckpt["conversation"].apply(json.loads)
            if "conversation_format" in ckpt.columns:
                ckpt["conversation_format"] = ckpt["conversation_format"].apply(json.loads)

            for col in EVAL_COLS:
                if col in ckpt.columns:
                    df.loc[ckpt.index, col] = ckpt[col]
            mask = (df[EVAL_COLS] != "").all(axis=1)
            done_indices = set(df.index[mask].tolist())
            print(f"  {len(done_indices)} rows already evaluated, skipping.")
        except Exception as e:
            print(f"  [ERROR] Checkpoint file is unreadable ({e}).")
    return df, done_indices
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

import json
def save_checkpoint(df: pd.DataFrame, chunk_size=50_000):
    print("Saving checkpoint...")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp.parquet")

    try:
        writer = None

        for start in range(0, len(df), chunk_size):
            print(f"Saving rows {start}:{start + chunk_size}")

            # Only copy the chunk
            df_chunk = df.iloc[start:start + chunk_size].copy()

            # Convert nested objects only inside chunk
            df_chunk["conversation"] = df_chunk["conversation"].apply(
                lambda x: json.dumps(x, cls=NumPyEncoder)
            )

            df_chunk["conversation_format"] = df_chunk["conversation_format"].apply(
                lambda x: json.dumps(x, cls=NumPyEncoder)
            )

            table = pa.Table.from_pandas(
                df_chunk,
                preserve_index=True
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    table.schema
                )

            writer.write_table(table)

            # Free memory immediately
            del df_chunk
            del table

        if writer:
            writer.close()

        tmp_path.replace(CHECKPOINT_PATH)

        print("======== SAVED ========")

    except Exception as e:
        if writer:
            writer.close()

        if tmp_path.exists():
            tmp_path.unlink()

        print(f"\n[ERROR] Checkpoint save failed: {e}")

    finally:
        import gc
        gc.collect()

    # exit(0)
# ── Main ───────────────────────────────────────────────────────────────────────
CONCURRENCY = 64

async def process_row(idx, df, sem):
    async with sem:
        row = df.loc[idx]
        history = json_format_to_linhist(row)
        resp = row["response"]
        question = history.get_last(["user-query"])[0]

        result = await evaluate_row(row, resp, question, history)
        return idx, result
async def worker(queue, df, pbar):
    """
    Consumes indices from the queue and processes them sequentially.
    Ensures that only a strict number of workers hit the servers.
    """
    while not queue.empty():
        try:
            idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        
        row = df.loc[idx]
        history = json_format_to_linhist(row)
        resp = row["response"]
        question = history.get_last(["user-query"])[0]

        # Process the single row
        result = await evaluate_row(row, resp, question, history)
        
        # Write directly to the DataFrame in place
        for col, val in result.items():
            df.at[idx, col] = val
            
        pbar.update(1)
        queue.task_done()


async def main():
    print("Loading marta_corrected ...")
    marta = pd.read_parquet("train/marta_corrected/") 
    print(marta.info())
    print("====================")
    df, done_indices = load_checkpoint(marta)

    pending = [idx for idx in df.index if idx not in done_indices]
    print(f"Rows to evaluate: {len(pending)}")

    # Adjust down based on VRAM capacity. 
    # With replicas=3 across 4 heavy models, keep this conservative!
    CONCURRENCY = 32 
    CHECKPOINT_EVERY = 7500  # Smaller checkpoints reduce hanging time

    with tqdm(total=len(pending), desc="Evaluating", unit="row") as pbar:
        # Loop through the dataset in safe checkpoint chunks
        for start in range(0, len(pending), CHECKPOINT_EVERY):
            batch = pending[start:start + CHECKPOINT_EVERY]
            
            # Setup an Asyncio Queue for the current batch chunk
            queue = asyncio.Queue()
            for idx in batch:
                queue.put_nowait(idx)

            # Create strict, bounded worker tasks
            workers = [
                asyncio.create_task(worker(queue, df, pbar))
                for _ in range(CONCURRENCY)
            ]

            # Wait until all workers finish processing the current chunk queue
            await asyncio.gather(*workers)

            # Safely write checkpoint without risking file corruption
            try:
                save_checkpoint(df)
                pbar.set_postfix_str("checkpoint saved")
            except Exception as e:
                print(f"\n[Warning] Failed to write checkpoint block: {e}")

    # Final Save
    save_checkpoint(df)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        df,
        preserve_index=True
    )

    pq.write_table(
        table,
        OUTPUT_PATH / "marta_corrected_neweval.parquet"
    )

    print(f"\nDone! Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())