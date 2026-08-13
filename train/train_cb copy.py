"""
train_cb.py
-----------
Fine-tunes GraySwanAI/Llama-3-8B-Instruct-RR using Representation Rerouting (RR).
Faithful port of GraySwanAI/circuit-breakers/src/ adapted to parquet dataset.

Key design decisions matching GraySwanAI:
  - progress = current_step / total_steps (not t/2T)
  - retain_coeff = alpha * progress  (increases over time)
  - cb_coeff    = alpha * (1-progress) (decreases over time)
  - <SEPARATOR> tokenization: request [left-pad, 512] + response [right-pad, 512]
  - layers_to_transform = all layers up to max(target_layers)
  - drop_layers_after = max(target_layers) to save VRAM
  - model_frozen replaces model.disable_adapter() (DDP-safe)
  - HF Trainer with custom compute_loss (no manual training loop)

Run:
    torchrun --nproc_per_node=3 train_cb.py --config config_cb.yaml
    torchrun --nproc_per_node=3 train_cb.py --config config_cb.yaml --force
"""

from __future__ import annotations

import argparse
import ast
import atexit
import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Conversation parser (same as nemotron script)
# ---------------------------------------------------------------------------


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
# Dataset — matches GraySwanAI's <SEPARATOR> tokenization strategy
# ---------------------------------------------------------------------------


class CircuitBreakerDataset(Dataset):
    """
    Each sample returns:
      input_ids_circuit_breaker  [1, 1024]  request[512,left] + response[512,right]
      attention_mask_circuit_breaker [1, 1024]
      input_ids                  [1, 1024]  retain sample
      attention_mask             [1, 1024]
      input_ids_val              [1, 1024]  val sample (cb used as val)
      attention_mask_val         [1, 1024]
    """

    SEP = "<SEPARATOR>"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        df_harmful: pd.DataFrame,
        df_safe: pd.DataFrame,
        conversation_col: str,
        response_col: str,
        harmful_col: str,
        k_last_turns: int,
        ultrachat_n: int,
        max_length: int = 1024,
        half_length: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.half_length = half_length

        user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
        assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag

        rng = random.Random(seed)

        # ---- Circuit Breaker set (D_s) ----
        cb_samples = []
        for _, row in df_harmful.iterrows():
            conversation = parse_conversation(row.get(conversation_col, []))
            if not conversation:
                continue
            model_response = row.get(response_col, None)
            if isinstance(model_response, float):
                model_response = None

            # Build request (everything up to last user turn)
            last_user_idx = None
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                continue

            history = conversation[:last_user_idx]
            last_user_turn = conversation[last_user_idx].get("content", "")

            # Build request string using Llama-3 tags
            request_parts = []
            for turn in history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    request_parts.append(f"{user_tag}{content}")
                elif role == "assistant":
                    request_parts.append(f"{assistant_tag}{content}")
            request_parts.append(f"{user_tag}{last_user_turn}{assistant_tag}")
            request_str = "".join(request_parts)

            # Response string
            response_str = model_response if model_response else ""

            cb_samples.append(f"{request_str}{self.SEP}{response_str}")

        self.circuit_breaker_orig = cb_samples
        rng.shuffle(self.circuit_breaker_orig)
        print(f"[D_s] Circuit breaker samples: {len(self.circuit_breaker_orig)}")

        # ---- Retain set (D_r) ----
        retain_samples = []

        # 1. Safe (correctly refused) conversations
        for _, row in df_safe.iterrows():
            conversation = parse_conversation(row.get(conversation_col, []))
            if not conversation:
                continue
            model_response = row.get(response_col, None)
            if isinstance(model_response, float):
                model_response = None
            text = tokenizer.apply_chat_template(
                [
                    {"role": t.get("role"), "content": t.get("content", "")}
                    for t in conversation
                    if t.get("role") in ("user", "assistant")
                ],
                tokenize=False,
            )
            if model_response:
                text = text.rstrip() + f"\n{assistant_tag}{model_response}"
            retain_samples.append(text)

        # 2. Harmful sub-conversations (safe prefix before turning point)
        for _, row in df_harmful.iterrows():
            conversation = parse_conversation(row.get(conversation_col, []))
            if not conversation:
                continue
            assistant_turns = [t for t in conversation if t.get("role") == "assistant"]
            if len(assistant_turns) < k_last_turns + 1 or len(conversation) < 3:
                continue
            kept_turns, assistant_count = [], 0
            for turn in reversed(conversation):
                if turn.get("role") == "assistant":
                    assistant_count += 1
                    if assistant_count <= k_last_turns:
                        continue
                kept_turns.insert(0, turn)
            while kept_turns and kept_turns[-1].get("role") == "user":
                kept_turns.pop()
            if not kept_turns:
                continue
            text = tokenizer.apply_chat_template(
                [
                    {"role": t.get("role"), "content": t.get("content", "")}
                    for t in kept_turns
                    if t.get("role") in ("user", "assistant")
                ],
                tokenize=False,
            )
            retain_samples.append(text)

        # 3. UltraChat
        print(f"[D_r] Loading {ultrachat_n} UltraChat samples …")
        uc = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"test_sft")
        uc_count = 0
        for item in uc:
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            retain_samples.append(text)
            uc_count += 1
            if uc_count >= ultrachat_n:
                break

        rng.shuffle(retain_samples)
        self.orig_s_retain = retain_samples
        print(f"[D_r] Retain samples: {len(self.orig_s_retain)}")

        # ---- Val set: reuse CB samples ----
        self.val_orig = self.circuit_breaker_orig.copy()
        rng.shuffle(self.val_orig)

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        half = self.half_length
        cb_kwargs = dict(
            max_length=half, padding="max_length", truncation=True, return_tensors="pt"
        )
        retain_kwargs = dict(
            max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # ---- Circuit Breaker ----
        cb_text = self.circuit_breaker_orig[i % len(self.circuit_breaker_orig)]
        if self.SEP in cb_text:
            cb_request, cb_response = cb_text.split(self.SEP, 1)
        else:
            cb_request, cb_response = cb_text, ""

        tokenizer.padding_side = "left"
        tok_request = tokenizer(cb_request, **cb_kwargs)
        tokenizer.padding_side = "right"
        tok_response = tokenizer(cb_response, add_special_tokens=False, **cb_kwargs)
        tokenizer.padding_side = "left"

        cb_input_ids = torch.cat([tok_request["input_ids"], tok_response["input_ids"]], dim=1)
        cb_attn_mask = torch.cat(
            [tok_request["attention_mask"], tok_response["attention_mask"]], dim=1
        )

        # ---- Retain ----
        retain_text = self.orig_s_retain[i % len(self.orig_s_retain)]
        tok_retain = tokenizer(retain_text, **retain_kwargs)

        # ---- Val ----
        val_text = self.val_orig[i % len(self.val_orig)]
        if self.SEP in val_text:
            val_text = val_text.replace(self.SEP, "")
        tok_val = tokenizer(val_text, **retain_kwargs)

        return dict(
            input_ids_circuit_breaker=cb_input_ids,
            attention_mask_circuit_breaker=cb_attn_mask,
            input_ids=tok_retain["input_ids"],
            attention_mask=tok_retain["attention_mask"],
            input_ids_val=tok_val["input_ids"],
            attention_mask_val=tok_val["attention_mask"],
        )


# ---------------------------------------------------------------------------
# Data collator (same as GraySwanAI)
# ---------------------------------------------------------------------------


def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, v in features.items():
            batch_inputs.setdefault(k, []).append(v)
    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            raise ValueError(f"Unexpected type {type(inputs[0])}")
    return batch_inputs


# ---------------------------------------------------------------------------
# Loss (faithful to GraySwanAI — uses model_frozen instead of disable_adapter)
# coefficients: retain_coeff = alpha*progress, cb_coeff = alpha*(1-progress)
# ---------------------------------------------------------------------------


def compute_loss_fn(
    trainer_self,
    model,
    model_frozen,
    inputs,
    target_layers,
    alpha,
    return_outputs=False,
):
    trainer_self.current_training_step += 1
    log_now = trainer_self.current_training_step % 10 == 0

    retain_input_ids = inputs["input_ids"]
    retain_attention_mask = inputs["attention_mask"]
    cb_input_ids = inputs["input_ids_circuit_breaker"]
    cb_attention_mask = inputs["attention_mask_circuit_breaker"]
    val_input_ids = inputs["input_ids_val"]
    val_attention_mask = inputs["attention_mask_val"]

    retain_inputs = dict(
        input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True
    )
    cb_inputs = dict(
        input_ids=cb_input_ids, attention_mask=cb_attention_mask, output_hidden_states=True
    )
    val_inputs = dict(
        input_ids=val_input_ids, attention_mask=val_attention_mask, output_hidden_states=True
    )

    # GraySwanAI coefficients: progress goes 0→1, retain grows, cb shrinks
    progress = trainer_self.get_training_progress()
    cb_coeff = alpha * (1 - progress) ** (0.1)  # slight bias to CB early on
    retain_coeff = alpha - cb_coeff
    if trainer_self.args.local_rank in (-1, 0):
        print(
            f"\nPROGRESS: {progress:.4f}  retain_coeff={retain_coeff:.4f}  cb_coeff={cb_coeff:.4f}"
        )

    layers_cb_attn = cb_attention_mask.repeat(len(target_layers), 1, 1).unsqueeze(-1)

    # ---- Frozen forward (no grad, no adapter issues) ----
    model_frozen.eval()
    with torch.no_grad():
        if retain_coeff > 0:
            orig_retain_hs = model_frozen(**retain_inputs)["hidden_states"]
            orig_retain_hidden = torch.stack(orig_retain_hs).detach()
            layers_retain_attn = retain_attention_mask.repeat(len(orig_retain_hs), 1, 1).unsqueeze(
                -1
            )
            orig_retain_hidden *= layers_retain_attn
            del orig_retain_hs
            gc.collect()

        if cb_coeff >= 0:
            orig_cb_hs = model_frozen(**cb_inputs)["hidden_states"]
            cb_hidden = torch.stack([orig_cb_hs[l].detach() for l in target_layers])
            del orig_cb_hs
            gc.collect()

        if log_now:
            orig_val_hs = model_frozen(**val_inputs)["hidden_states"]
            val_hidden = torch.stack([orig_val_hs[l] for l in target_layers])
            del orig_val_hs
            gc.collect()

    # ---- LoRA forward ----
    model.train()

    retain_loss = torch.tensor(0.0, device=cb_input_ids.device)
    cb_loss = torch.tensor(0.0, device=cb_input_ids.device)

    if retain_coeff > 0:
        lora_retain_hs = model(**retain_inputs)["hidden_states"]
        lora_retain_hidden = torch.stack(lora_retain_hs) * layers_retain_attn
        retain_loss = torch.norm(
            lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float
        ).nanmean()

        if log_now:
            cos_r = cosine_similarity(lora_retain_hidden, orig_retain_hidden, dim=-1)
            cos_r = cos_r * layers_retain_attn.squeeze(-1)
            if trainer_self.args.local_rank in (-1, 0):
                print(f"retain_cos_sim: {(cos_r.sum() / layers_retain_attn.sum()).item():.4f}")

    if cb_coeff >= 0:
        lora_cb_hs = model(**cb_inputs)["hidden_states"]
        lora_cb_hidden = torch.stack([lora_cb_hs[l] for l in target_layers])

        norm_lora = lora_cb_hidden / torch.norm(
            lora_cb_hidden, dim=-1, keepdim=True, dtype=torch.float
        ).clamp(min=1e-8)
        norm_orig = cb_hidden / torch.norm(
            cb_hidden, dim=-1, keepdim=True, dtype=torch.float
        ).clamp(min=1e-8)
        inner = (norm_lora * norm_orig) * layers_cb_attn
        cb_loss = torch.relu(inner.sum(dim=-1)).sum() / layers_cb_attn.sum().clamp(min=1)

        if log_now:
            cos_cb = cosine_similarity(cb_hidden, lora_cb_hidden, dim=-1) * layers_cb_attn.squeeze(
                -1
            )
            if trainer_self.args.local_rank in (-1, 0):
                print(f"cb_cos_sim: {(cos_cb.sum() / layers_cb_attn.sum()).item():.4f}")
                print(
                    f"updated_cb_norm: {torch.mean(lora_cb_hidden.norm(dim=-1).mean(dim=1)).item():.4f}"
                )
                print(
                    f"orig_cb_norm:    {torch.mean(cb_hidden.norm(dim=-1).mean(dim=1)).item():.4f}"
                )

    if log_now:
        with torch.no_grad():
            lora_val_hs = model(**val_inputs)["hidden_states"]
            lora_val_hidden = torch.stack([lora_val_hs[l] for l in target_layers])
            layers_val_attn = val_attention_mask.repeat(len(target_layers), 1, 1).unsqueeze(-1)
            cos_v = cosine_similarity(
                val_hidden, lora_val_hidden, dim=-1
            ) * layers_val_attn.squeeze(-1)
            if trainer_self.args.local_rank in (-1, 0):
                print(f"val_cos_sim: {(cos_v.sum() / layers_val_attn.sum()).item():.4f}")

    loss = retain_coeff * retain_loss + cb_coeff * cb_loss

    if trainer_self.args.local_rank in (-1, 0):
        print(f"retain_loss={retain_loss.item():.4f}  cb_loss={cb_loss.item():.4f}")
        print("=" * 50)

    return (loss,) if return_outputs else loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
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


def save_best(model, tokenizer, output_dir: str, base_model_name: str):
    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] Merging LoRA → {best_dir}")

    # First save just the adapter weights from the truncated model
    raw = model.module if hasattr(model, "module") else model
    tmp_adapter_dir = str(Path(output_dir) / "tmp_adapter")
    raw.save_pretrained(tmp_adapter_dir)

    # Load the FULL base model (no truncated config)
    from peft import PeftModel

    full_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Apply adapter on top of full model and merge
    full_model_with_adapter = PeftModel.from_pretrained(full_base, tmp_adapter_dir)
    merged = full_model_with_adapter.merge_and_unload()
    merged.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Cleanup tmp
    shutil.rmtree(tmp_adapter_dir, ignore_errors=True)
    print("[save] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_cb.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = train_cfg["output_dir"]
    max_length = train_cfg.get("max_seq_length", 1024)
    half_length = max_length // 2
    target_layers = train_cfg.get("repr_layers", [10, 20])
    alpha = train_cfg.get("alpha", 10.0)
    k_last_turns = data_cfg.get("k_last_turns", 2)
    full_layers = train_cfg.get("full_layers", False)

    # drop_layers_after: truncate model after max target layer (saves VRAM)
    drop_layers_after = max(target_layers) if not full_layers else None

    # layers_to_transform: all layers up to max(target_layers) — matching GraySwanAI
    if train_cfg.get("transform_all_layers", True):
        layers_to_transform = list(range(max(target_layers) + 1))
    else:
        layers_to_transform = target_layers

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    resume_from = None
    if not args.force:
        resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            print(f"[resume] Resuming from: {resume_from}")
        else:
            print("[resume] Starting from scratch.")
    else:
        print("[resume] --force: starting from scratch.")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[init] Tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    harmful_col = data_cfg["harmful_col"]
    conversation_col = data_cfg["conversation_col"]
    response_col = data_cfg["response_col"]

    print(f"[data] Loading train: {data_cfg['train_parquet']}")
    df = pd.read_parquet(data_cfg["train_parquet"])
    df = df[df[harmful_col].notna()].copy()
    df[harmful_col] = df[harmful_col].astype(bool)
    df_harmful = df[df[harmful_col]].copy()
    df_safe = df[~df[harmful_col]].copy()
    print(f"[data] Harmful: {len(df_harmful)}  Safe: {len(df_safe)}")

    train_dataset = CircuitBreakerDataset(
        tokenizer=tokenizer,
        df_harmful=df_harmful,
        df_safe=df_safe,
        conversation_col=conversation_col,
        response_col=response_col,
        harmful_col=harmful_col,
        k_last_turns=k_last_turns,
        ultrachat_n=data_cfg.get("ultrachat_retain_n", 5000),
        max_length=max_length,
        half_length=half_length,
        seed=seed,
    )
    print(f"[data] Train dataset len: {len(train_dataset)}")

    # Eval dataset
    eval_dataset = None
    if data_cfg.get("eval_parquet"):
        print(f"[data] Loading eval: {data_cfg['eval_parquet']}")
        df_eval = pd.read_parquet(data_cfg["eval_parquet"])
        df_eval = df_eval[df_eval[harmful_col].notna()].copy()
        df_eval[harmful_col] = df_eval[harmful_col].astype(bool)
        df_eval_harmful = df_eval[df_eval[harmful_col]].copy()
        df_eval_safe = df_eval[~df_eval[harmful_col]].copy()
        print(f"[data] Eval harmful: {len(df_eval_harmful)}  safe: {len(df_eval_safe)}")
        eval_dataset = CircuitBreakerDataset(
            tokenizer=tokenizer,
            df_harmful=df_eval_harmful,
            df_safe=df_eval_safe,
            conversation_col=conversation_col,
            response_col=response_col,
            harmful_col=harmful_col,
            k_last_turns=k_last_turns,
            ultrachat_n=min(500, data_cfg.get("ultrachat_retain_n", 5000)),
            max_length=max_length,
            half_length=half_length,
            seed=seed + 1,
        )
        print(f"[data] Eval dataset len: {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # Model — truncated at drop_layers_after (GraySwanAI trick)
    # ------------------------------------------------------------------
    print(f"[init] Model: {model_cfg['name']}  drop_layers_after={drop_layers_after}")
    config = AutoConfig.from_pretrained(model_cfg["name"])
    if drop_layers_after is not None:
        config.num_hidden_layers = drop_layers_after + 1
        print(f"[init] Truncated to {config.num_hidden_layers} layers")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA on all layers up to max(target_layers) — matching GraySwanAI
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        layers_to_transform=layers_to_transform,
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # Frozen reference model (same truncated config, no LoRA, no grad)
    # ------------------------------------------------------------------
    print(f"[init] Frozen reference model …")
    model_frozen = AutoModelForCausalLM.from_pretrained(
        model_cfg.get("frozen_name", model_cfg["name"]),
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Custom Trainer
    # ------------------------------------------------------------------
    total_steps = train_cfg.get("max_steps", 300)

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_steps=total_steps,
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 100) if eval_dataset is not None else None,
        load_best_model_at_end=False,
        metric_for_best_model=None,  # we use custom loss, not eval metric
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    class CBTrainer(Trainer):
        def __init__(
            self, *args, frozen_model=None, lorra_target_layers=None, lorra_alpha=10.0, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.model_frozen = frozen_model
            self.lorra_target_layers = lorra_target_layers
            self.lorra_alpha = lorra_alpha
            self.current_training_step = 0
            self.total_steps = total_steps

        def get_training_progress(self) -> float:
            """progress 0→1 over training, matching GraySwanAI's step/300."""
            print(f"[progress] Step {self.current_training_step}/{self.total_steps}")
            return min(
                self.current_training_step
                / (self.total_steps * (self.args.gradient_accumulation_steps or 1)),
                1.0,
            )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Move frozen model to same device as trainable model
            device = next(model.parameters()).device
            if next(self.model_frozen.parameters()).device != device:
                self.model_frozen = self.model_frozen.to(device)

            loss = compute_loss_fn(
                trainer_self=self,
                model=model,
                model_frozen=self.model_frozen,
                inputs=inputs,
                target_layers=self.lorra_target_layers,
                alpha=self.lorra_alpha,
                return_outputs=return_outputs,
            )

            return loss

    trainer = CBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        frozen_model=model_frozen,
        lorra_target_layers=target_layers,
        lorra_alpha=alpha,
    )
    model.config.use_cache = False

    # Register final save on exit (matching GraySwanAI's atexit pattern)
    atexit.register(
        save_best,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        base_model_name=model_cfg["name"],
    )
    print(
        f"[train] Starting … total_steps={total_steps}  alpha={alpha}  target_layers={target_layers}"
    )
    trainer.train(resume_from_checkpoint=resume_from)

    # Explicit save at end (atexit is backup)

    if trainer.args.local_rank in (-1, 0):
        save_best(model, tokenizer, output_dir, base_model_name=model_cfg["name"])


if __name__ == "__main__":
    # Patch CVE-2025-32434 for torch < 2.6
    try:
        import transformers.utils.import_utils as _iu

        if hasattr(_iu, "check_torch_load_is_safe"):
            _iu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
    main()
