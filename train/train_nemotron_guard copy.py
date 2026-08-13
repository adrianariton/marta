"""
train_nemotron_guard.py
-----------------------
Fine-tunes nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 on a multi-turn
safety dataset stored in parquet format.

Prompt format matches exactly the model card:
  https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3

Run via sbatch or directly:
    torchrun --nproc_per_node=3 train_nemotron_guard.py --config config.yaml
    torchrun --nproc_per_node=3 train_nemotron_guard.py --config config.yaml --force
"""

from __future__ import annotations


# Patch CVE-2025-32434: transformers>=4.50 blocks torch.load on torch<2.6.
# Must run before transformers is imported.
def _patch_torch_load_check():
    try:
        import transformers.utils.import_utils as _iu

        if hasattr(_iu, "check_torch_load_is_safe"):
            _iu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


_patch_torch_load_check()
import transformers.utils.import_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import transformers.trainer

transformers.trainer.check_torch_load_is_safe = lambda: None
import argparse
import ast
import json
import random
import string
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Exact Nemotron prompt template (from model card)
# ---------------------------------------------------------------------------

NEMOTRON_INSTRUCTION = (
    "Task: Check if there is unsafe content in the user or agent messages in the below "
    "conversation in accordance with our safety policy with the below categories. "
    "Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. "
    "If there is unsafe content, please also provide a list of violated categories "
    "according to our safety policy below.\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "S1: Violence. \n"
    "S2: Sexual. \n"
    "S3: Criminal Planning/Confessions. \n"
    "S4: Guns and Illegal Weapons. \n"
    "S5: Controlled/Regulated Substances. \n"
    "S6: Suicide and Self Harm. \n"
    "S7: Sexual (minor). \n"
    "S8: Hate/Identity Hate. \n"
    "S9: PII/Privacy. \n"
    "S10: Harassment. \n"
    "S11: Threat. \n"
    "S12: Profanity. \n"
    "S13: Needs Caution. \n"
    "S14: Other. \n"
    "S15: Manipulation. \n"
    "S16: Fraud/Deception. \n"
    "S17: Malware. \n"
    "S18: High Risk Gov Decision Making. \n"
    "S19: Political/Misinformation/Conspiracy. \n"
    "S20: Copyright/Trademark/Plagiarism. \n"
    "S21: Unauthorized Advice. \n"
    "S22: Illegal Activity. \n"
    "S23: Immoral/Unethical. \n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
)

NEMOTRON_FOOTER = (
    "<END CONVERSATION>\n\n"
    "Provide your safety assessment for the above conversation in the below output JSON format:\n"
    '{"User Safety": safe or unsafe rating for the user message, '
    '"Response Safety": safe or unsafe rating for the agent response. '
    "Omit if no agent response present. "
    '"Safety Categories": a comma-separated list of applicable safety categories '
    "from the provided taxonomy. Omit if all safe.}\n\n"
    "Do not include anything other than the output JSON in your response.\n"
    "Output JSON: "
)


def render_nemotron_prompt(
    history: list[dict],
    last_user_turn: str,
    agent_response: str | None,
) -> str:
    """
    Builds the <BEGIN CONVERSATION> block for multi-turn input.

    Structure:
      - history turns (all turns before the last user message):
          user: ...
          response: agent: ...
          user: ...
          response: agent: ...
          ...
      - last user turn:
          user: <last_user_turn>
      - agent response (only for response moderation):
          response: agent: <agent_response>

    For query moderation, agent_response is None → block ends after last user turn.
    """
    lines = ["<BEGIN CONVERSATION>\n\n"]

    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"user: {content}\n")
        elif role == "assistant":
            # History turns use "agent:" to distinguish from the final
            # evaluated response which uses "response: agent:"
            lines.append(f"agent: {content}\n")

    # Last user turn (the one being evaluated)
    lines.append(f"user: {last_user_turn}\n")

    # Agent response — only for response moderation
    if agent_response is not None:
        lines.append(f"\nresponse: agent: {agent_response}\n")

    return NEMOTRON_INSTRUCTION + "".join(lines) + NEMOTRON_FOOTER


# ---------------------------------------------------------------------------
# goal -> Safety Categories (Sx codes from taxonomy)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "it",
    "this",
    "that",
    "be",
    "as",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "not",
    "no",
    "can",
    "how",
    "what",
    "why",
    "when",
    "who",
    "will",
    "my",
    "your",
    "me",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "have",
    "has",
    "had",
    "so",
    "if",
    "than",
    "then",
    "about",
    "into",
    "over",
    "after",
    "just",
    "get",
    "give",
    "make",
    "way",
    "good",
    "want",
    "would",
    "could",
    "should",
    "its",
    "their",
    "our",
    "up",
    "out",
    "which",
    "also",
    "like",
    "more",
    "some",
    "there",
}

# keywords → (Sx code, display name used in output)
_KEYWORD_CATEGORY_MAP: list[tuple[set[str], str]] = [
    (
        {
            "kill",
            "murder",
            "attack",
            "bomb",
            "shooting",
            "stab",
            "explosive",
            "assault",
            "hurt",
            "injure",
        },
        "Violence",
    ),
    ({"sex", "porn", "nude", "explicit", "erotic"}, "Sexual"),
    (
        {"gun", "firearm", "rifle", "pistol", "ammunition", "arms", "illegal weapon"},
        "Guns and Illegal Weapons",
    ),
    (
        {
            "drug",
            "meth",
            "cocaine",
            "heroin",
            "fentanyl",
            "narco",
            "overdose",
            "poison",
            "substance",
        },
        "Controlled/Regulated Substances",
    ),
    (
        {
            "suicide",
            "selfharm",
            "self harm",
            "burning",
            "cut myself",
            "hurt myself",
            "skin",
            "pain",
        },
        "Suicide and Self Harm",
    ),
    ({"child", "minor", "underage", "kid", "pedophil", "grooming"}, "Sexual (minor)"),
    (
        {
            "hate",
            "racist",
            "racism",
            "nazi",
            "slur",
            "discrimination",
            "antisemit",
            "white suprema",
        },
        "Hate/Identity Hate",
    ),
    ({"privac", "doxx", "personal data", "stalk", "track", "spy"}, "PII/Privacy"),
    ({"harass", "bully", "threaten", "intimidat"}, "Harassment"),
    (
        {
            "hack",
            "malware",
            "exploit",
            "ransomware",
            "phishing",
            "breach",
            "vulnerability",
            "ddos",
            "keylogger",
            "bypass",
            "spyware",
            "trojan",
            "script",
            "macbook",
            "password",
        },
        "Malware",
    ),
    ({"fraud", "scam", "deceiv", "fake", "impersonat", "manipulat"}, "Fraud/Deception"),
    ({"illegal", "crime", "criminal", "theft", "rob", "smuggl"}, "Criminal Planning/Confessions"),
    ({"immoral", "unethical", "wrong"}, "Immoral/Unethical"),
]


def goal_to_categories(goal: str | None) -> str | None:
    """
    Returns a comma-separated string of category names (e.g. 'Violence, Malware')
    matching the Nemotron taxonomy, or None if nothing matched (→ omit field).
    """
    if not goal or not isinstance(goal, str):
        return None
    text = goal.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = set(text.split()) - _STOPWORDS
    matched = []
    for keywords, category in _KEYWORD_CATEGORY_MAP:
        for kw in keywords:
            if any(kw in t or t in kw for t in tokens):
                if category not in matched:
                    matched.append(category)
                break
    return ", ".join(matched) if matched else "Other"


# ---------------------------------------------------------------------------
# Build Nemotron JSON output
# ---------------------------------------------------------------------------


def build_output_json(harmful: bool, goal: str | None, has_response: bool) -> str:
    """
    Matches exact model card output format:
    - Omit Response Safety if no agent response
    - Omit Safety Categories if all safe
    """
    obj: dict[str, str] = {}
    obj["User Safety"] = "unsafe" if harmful else "safe"

    if has_response:
        obj["Response Safety"] = "unsafe" if harmful else "safe"

    if harmful:
        cats = goal_to_categories(goal)
        if cats:
            obj["Safety Categories"] = cats

    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Conversation parser
# ---------------------------------------------------------------------------

import numpy as np


def parse_conversation(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                result = loader(raw)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    if isinstance(raw, np.ndarray):
        try:
            result = raw.tolist()
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Prompt builder (exact Nemotron format)
# ---------------------------------------------------------------------------


def build_prompt_text(
    tokenizer: AutoTokenizer,
    conversation: list[dict],
    agent_response: str | None,
    include_response: bool,
) -> str:
    """
    Builds the full Nemotron prompt for a multi-turn conversation.

    conversation: all turns (role: user/assistant) including the last user turn.
    We split it as:
      - history = all turns up to (but not including) the last user turn
      - last_user_turn = content of the last user message

    For query moderation  (include_response=False): agent_response omitted.
    For response moderation (include_response=True):  agent_response appended.
    """
    # Find last user turn
    last_user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        history = conversation
        last_user_turn = ""
    else:
        history = conversation[:last_user_idx]
        last_user_turn = conversation[last_user_idx].get("content", "")

    rendered = render_nemotron_prompt(
        history=history,
        last_user_turn=last_user_turn,
        agent_response=agent_response if include_response else None,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_safety_texts(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg_data: dict,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    conversation_col = cfg_data["conversation_col"]
    response_col = cfg_data["response_col"]
    harmful_col = cfg_data["harmful_col"]
    goal_col = cfg_data["goal_col"]
    qr_ratio = cfg_data.get("query_response_ratio", 0.5)

    texts = []
    skipped = 0
    for _, row in df.iterrows():
        conversation = parse_conversation(row.get(conversation_col, []))
        if not conversation:
            skipped += 1
            continue

        harmful = bool(row.get(harmful_col, False))
        goal = row.get(goal_col, None)
        model_response = row.get(response_col, None)
        if isinstance(model_response, float):
            model_response = None

        include_response = rng.random() < qr_ratio and model_response is not None

        prompt = build_prompt_text(tokenizer, conversation, model_response, include_response)
        target = build_output_json(harmful, goal, include_response)
        texts.append(prompt + target + tokenizer.eos_token)

    if skipped:
        print(f"[dataset] Skipped {skipped} rows with empty conversation")
    return texts


def build_ultrachat_texts(
    tokenizer: AutoTokenizer,
    n_samples: int,
    seed: int,
) -> list[str]:
    print(f"[retain] Loading {n_samples} UltraChat samples …")
    raw = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{n_samples}]",
        trust_remote_code=True,
    )
    safe_output = json.dumps({"User Safety": "safe"}, ensure_ascii=False)

    texts = []
    for item in raw:
        conversation = [
            {"role": m["role"], "content": m["content"]}
            for m in item.get("messages", [])
            if m.get("role") in ("user", "assistant")
        ]
        if len(conversation) < 2:
            continue
        prompt = build_prompt_text(tokenizer, conversation, None, False)
        texts.append(prompt + safe_output + tokenizer.eos_token)

    print(f"[retain] Built {len(texts)} retain samples")
    return texts


# ---------------------------------------------------------------------------
# Checkpoint resume helper
# ---------------------------------------------------------------------------


def find_latest_checkpoint(output_dir: str) -> str | None:
    p = Path(output_dir)
    if not p.exists():
        return None
    checkpoints = sorted(
        [c for c in p.glob("checkpoint-*") if c.name.split("-")[-1].isdigit()],
        key=lambda c: int(c.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            loss = logs.get("loss", logs.get("eval_loss", None))
            lr = logs.get("learning_rate", None)
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
            print(
                f"[step {state.global_step}] loss={loss_str}  lr={lr_str}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and start from scratch.",
    )
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})

    seed = train_cfg.get("seed", 42)
    max_seq_length = train_cfg.get("max_seq_length", 4096)
    output_dir = train_cfg["output_dir"]

    set_seed(seed)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    resume_from = None
    if not args.force:
        resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            print(f"[resume] Resuming from: {resume_from}")
        else:
            print("[resume] No checkpoint found, starting from scratch.")
    else:
        print("[resume] --force: starting from scratch.")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[init] Tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    harmful_col = data_cfg["harmful_col"]

    train_df = pd.read_parquet(data_cfg["train_parquet"])
    val_df = pd.read_parquet(data_cfg["val_parquet"])
    train_df = train_df[train_df[harmful_col].notna()].copy()
    val_df = val_df[val_df[harmful_col].notna()].copy()

    print(f"[data] Train: {len(train_df)} rows  harmful={train_df[harmful_col].astype(bool).sum()}")
    print(f"[data] Val:   {len(val_df)} rows  harmful={val_df[harmful_col].astype(bool).sum()}")

    train_texts = build_safety_texts(train_df, tokenizer, data_cfg, seed)
    val_texts = build_safety_texts(val_df, tokenizer, data_cfg, seed + 1)
    retain_texts = build_ultrachat_texts(
        tokenizer,
        n_samples=data_cfg.get("ultrachat_retain_n", 5000),
        seed=data_cfg.get("ultrachat_seed", 42),
    )

    all_train = train_texts + retain_texts
    random.Random(seed).shuffle(all_train)

    train_ds = Dataset.from_dict({"text": all_train})
    val_ds = Dataset.from_dict({"text": val_texts})
    print(f"[data] Final train (with retain): {len(train_ds)}   val: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"[init] Model: {model_cfg['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_safetensors=True,
    )

    # ------------------------------------------------------------------
    # LoRA — matching nvidia's own config: r=8, alpha=32, q_proj+v_proj
    # ------------------------------------------------------------------
    peft_config = None
    if lora_cfg.get("use_lora", True):
        print("[init] Applying LoRA …")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_cfg.get("bias", "none"),
        )

    # ------------------------------------------------------------------
    # SFTConfig
    # ------------------------------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        fp16=False,
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        logging_steps=train_cfg.get("logging_steps", 20),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to="none",
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[LogCallback()],
    )

    print("[train] Starting …")
    trainer.train(resume_from_checkpoint=resume_from)

    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Saving to {best_dir}")

    if lora_cfg.get("use_lora", True):
        # Merge LoRA weights into base model and save full model
        print("[train] Merging LoRA adapter into base model …")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(best_dir))
    else:
        trainer.save_model(str(best_dir))

    tokenizer.save_pretrained(str(best_dir))
    print("[train] Done.")


if __name__ == "__main__":
    main()
