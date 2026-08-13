"""
prepare_data.py

Converts raw multi-turn conversation JSONL into a verl-compatible parquet file.
Includes prefix augmentation: for each conversation of length n, all strict
prefixes of length k < n (ending on an assistant turn) are also added as
separate training samples. This gives early turns stronger gradient signal
and corrects the length-reward correlation bias from harmful-terminated convos.

Input JSONL schema (one conversation per line):
{
  "turns": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...", "eval": {"proximity": 5, "cooperation": 4, "erosion": 1, "harmful": false}},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...", "eval": {"proximity": 3, "cooperation": 2, "erosion": 4, "harmful": true}},
    ...
  ]
}

Output parquet columns expected by verl RLHFDataset:
  - input_ids           : full tokenized conversation (prompt + all responses)
  - attention_mask      : 1 everywhere
  - position_ids        : 0..seq_len-1
  - loss_mask           : 1 only on assistant response tokens
  - token_level_rewards : scalar reward placed on the LAST token of each assistant turn, 0 elsewhere
  - responses           : tokenized assistant turns concatenated (needed by verl internally)
  - response_mask       : 1 on response tokens (same as loss_mask here)
  - conversation_id      : int, same for a conversation and all its prefixes (used to prevent
                           co-occurrence in the same training batch)
  - conversation_length  : int, number of assistant turns in this sample

Usage:
  python prepare_data.py \
    --input      data/conversations.jsonl \
    --output     data/train.parquet \
    --model      mistralai/Mistral-7B-Instruct-v0.3 \
    --max_len    4096 \
    --no_prefixes        # optional: disable prefix augmentation
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

EROSION_WEIGHT = 1.5  # erosion counts more than 1x — tune as needed
HARMFUL_PENALTY = 1.5  # soft penalty, not hard -1, because judge can be noisy


def eval_to_reward(e: dict) -> float:
    """
    Map a single turn eval dict to a scalar reward in roughly [-1, 1].

    - harmful=True  → strong penalty, overrides other dims
    - proximity + cooperation → positive signal (normalized)
    - erosion → subtracted penalty (weighted)
    """
    if e["harmful"]:
        # Soft penalty: still incorporate other dims faintly so gradients
        # are not completely zeroed for turns near a harmful one.
        cooperation = e["cooperation"]
        proximity = e["proximity"]
        erosion = e["erosion"] or 0
        if cooperation is None:
            r = proximity / 7.0
        elif proximity is None:
            r = cooperation / 7.0
        else:
            r = (proximity + cooperation) / 14.0
        r -= EROSION_WEIGHT * erosion / 7.0
        r -= HARMFUL_PENALTY
        return float(np.clip(r, -1.0, 1.0))

    cooperation = e["cooperation"]
    proximity = e["proximity"]
    erosion = e["erosion"] or 0
    if cooperation is None:
        cooperation = proximity
    if proximity is None:
        proximity = cooperation
    r = (proximity + cooperation) / 14.0  # [0, 1]
    r -= EROSION_WEIGHT * erosion / 7.0  # subtract up to ~0.21*weight
    return float(np.clip(r, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_conversation(turns: list[dict], tokenizer, max_len: int) -> dict | None:
    """
    Tokenizes a full multi-turn conversation.

    Strategy: build the conversation token by token using the chat template,
    tracking where each assistant turn starts/ends via delta tokenization.

    Returns a dict with numpy arrays, or None if the sequence exceeds max_len.
    """
    # Collect assistant turn evals in order
    assistant_evals = [t["eval"] for t in turns if t["role"] == "assistant"]

    # Build messages list (without eval field — that's metadata)
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]

    # Tokenize the full conversation with the chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_len:
        return None  # skip sequences that are too long

    seq_len = len(full_ids)
    input_ids = np.array(full_ids, dtype=np.int32)
    attention_mask = np.ones(seq_len, dtype=np.int32)
    position_ids = np.arange(seq_len, dtype=np.int32)
    loss_mask = np.zeros(seq_len, dtype=np.int32)
    token_level_rewards = np.zeros(seq_len, dtype=np.float32)

    # -------------------------------------------------------------------
    # Delta tokenization: find each assistant turn's token span
    # -------------------------------------------------------------------
    # We rebuild the conversation prefix up to (and including) each turn,
    # tokenize it, and the delta gives us the assistant span.
    # -------------------------------------------------------------------
    assistant_idx = 0
    prefix_messages: list[dict] = []

    for turn in turns:
        prefix_messages.append({"role": turn["role"], "content": turn["content"]})

        if turn["role"] != "assistant":
            continue

        # Tokenize prefix up to and including this assistant turn
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        # Tokenize prefix up to (but NOT including) this assistant turn
        prev_text = tokenizer.apply_chat_template(
            prefix_messages[:-1],
            tokenize=False,
            add_generation_prompt=True,  # simulate the "waiting for response" state
        )
        prev_ids = tokenizer.encode(prev_text, add_special_tokens=False)

        turn_start = len(prev_ids)
        turn_end = len(prefix_ids)  # exclusive

        if turn_start >= turn_end:
            # Degenerate case: skip
            assistant_idx += 1
            continue

        # Mark loss mask on assistant tokens
        loss_mask[turn_start:turn_end] = 1

        # Place reward on the LAST token of this assistant turn
        eval_dict = assistant_evals[assistant_idx]
        reward = eval_to_reward(eval_dict)
        token_level_rewards[turn_end - 1] = reward

        # If this turn is harmful, stop propagating — later turns were never generated
        if eval_dict["harmful"]:
            break

        assistant_idx += 1

    # response_mask == loss_mask for verl's internal use
    response_mask = loss_mask.copy()

    # responses: just the response token IDs (concatenated assistant turns)
    response_ids = input_ids[loss_mask == 1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "response_mask": response_mask,
        "token_level_rewards": token_level_rewards,
        "responses": response_ids,
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Prefix augmentation
# ---------------------------------------------------------------------------


def get_prefix_turn_lists(turns: list[dict]) -> list[list[dict]]:
    """
    Returns all strict prefixes of a conversation that end on an assistant turn.

    Example: for a conversation with turns [u1, a1, u2, a2, u3, a3],
    returns [[u1, a1], [u1, a1, u2, a2]] — i.e. all prefixes except the full
    conversation itself (which is added separately as the canonical sample).

    A prefix ending just before a harmful turn is intentionally included:
    the model should learn that turn n-1 in a harmful trajectory was a
    contributing step, even if it wasn't itself flagged harmful.
    """
    # Find indices of all assistant turns
    assistant_turn_indices = [i for i, t in enumerate(turns) if t["role"] == "assistant"]

    if len(assistant_turn_indices) < 2:
        # Need at least 2 assistant turns to have a meaningful strict prefix
        return []

    prefixes = []
    # All assistant turn boundaries except the last (that's the full conversation)
    for end_idx in assistant_turn_indices[:-1]:
        prefixes.append(turns[: end_idx + 1])

    return prefixes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output parquet file")
    parser.add_argument("--model", required=True, help="HuggingFace model name/path for tokenizer")
    parser.add_argument(
        "--max_len", type=int, default=4096, help="Max sequence length (longer convos are dropped)"
    )
    parser.add_argument(
        "--no_prefixes",
        action="store_true",
        help="Disable prefix augmentation (ignored if --has_prefixes)",
    )
    parser.add_argument(
        "--has_prefixes",
        action="store_true",
        help="Input JSONL already contains prefixes — skip generation, read conversation_id and conversation_length from the data",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    skipped = 0
    total = 0

    with open(args.input) as f:
        for convo_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            convo = json.loads(line)
            turns = convo["turns"]

            if args.has_prefixes:
                # ------------------------------------
                # Data already includes prefixes.
                # Read conversation_id and conversation_length
                # directly from the JSON if present,
                # otherwise fall back to line index.
                # ------------------------------------
                result = tokenize_conversation(turns, tokenizer, args.max_len)
                if result is None:
                    skipped += 1
                else:
                    result["conversation_id"] = convo.get("conversation_id", convo_id)
                    result["conversation_length"] = convo.get(
                        "conversation_length", sum(1 for t in turns if t["role"] == "assistant")
                    )
                    records.append(result)

            else:
                # ------------------------------------
                # Generate prefixes from full convos
                # ------------------------------------
                result = tokenize_conversation(turns, tokenizer, args.max_len)
                if result is None:
                    skipped += 1
                else:
                    result["conversation_id"] = convo_id
                    result["conversation_length"] = sum(
                        1 for t in turns if t["role"] == "assistant"
                    )
                    records.append(result)

                if not args.no_prefixes:
                    for prefix_turns in get_prefix_turn_lists(turns):
                        prefix_result = tokenize_conversation(prefix_turns, tokenizer, args.max_len)
                        if prefix_result is None:
                            continue
                        prefix_result["conversation_id"] = convo_id
                        prefix_result["conversation_length"] = sum(
                            1 for t in prefix_turns if t["role"] == "assistant"
                        )
                        records.append(prefix_result)

    print(
        f"Processed {total} conversations → {len(records)} total samples "
        f"({skipped} dropped as too long)"
    )

    df = pd.DataFrame(
        [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()} for r in records]
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Avg seq_len: {df['seq_len'].mean():.1f}  Max: {df['seq_len'].max()}")
    print()
    print("NOTE: Pass conversation_id to your DataLoader's batch sampler to prevent")
    print("a conversation and its prefixes from appearing in the same training batch.")
    print("See ConversationGroupedBatchSampler in multiturn_dataset.py.")


if __name__ == "__main__":
    main()
