"""
compute_coverage.py
--------------------
Computes the exact CircuitBreakerDataset sizes (L_cb, L_retain, L_total)
and how many times a given harmful row will be drawn over a full training
run, given config_cb.yaml.

This re-implements the windowing logic from train_cb.py's
CircuitBreakerDataset.__init__ for df_harmful, without needing the model
or tokenizer (uses a dummy tokenizer-free count of windows).

Usage:
    python compute_coverage.py --config config_cb.yaml --num-gpus 3 \
        --row-index 0
"""

import argparse
import ast
import json

import numpy as np
import pandas as pd
import yaml


def parse_conversation(raw):
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


def count_cb_windows_for_row(conversation, window_turns=4, window_stride=2):
    """Mirrors the windowing logic in CircuitBreakerDataset.__init__ and
    returns how many circuit_breaker_orig entries this single row produces."""
    if not conversation:
        return 0

    last_user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return 0

    history = conversation[:last_user_idx]

    if not history:
        return 1  # single sample, no windowing

    if len(history) <= window_turns:
        return 1  # one window = whole history

    windows = []
    start = 0
    while start < len(history):
        end = min(start + window_turns, len(history))
        windows.append((start, end))
        if end == len(history):
            break
        start += window_stride

    last_window = (len(history) - window_turns, len(history))
    if windows[-1] != last_window:
        windows.append(last_window)

    return len(windows)


def count_retain_rows_for_harmful_row(conversation, k_last_turns):
    """Mirrors retain-set source #2: harmful sub-conversation prefixes.
    Returns 1 if this row contributes a retain sample, else 0."""
    assistant_turns = [t for t in conversation if t.get("role") == "assistant"]
    if len(assistant_turns) < k_last_turns + 1 or len(conversation) < 3:
        return 0

    kept_turns = []
    assistant_count = 0
    for turn in reversed(conversation):
        if turn.get("role") == "assistant":
            assistant_count += 1
            if assistant_count <= k_last_turns:
                continue
        kept_turns.insert(0, turn)
    while kept_turns and kept_turns[-1].get("role") == "user":
        kept_turns.pop()

    return 1 if kept_turns else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./train/config_cb.yaml")
    ap.add_argument("--num-gpus", type=int, default=3)
    ap.add_argument("--window-turns", type=int, default=4)
    ap.add_argument("--window-stride", type=int, default=2)
    ap.add_argument("--row-index", type=int, default=None,
                     help="Index (within df_harmful) of a row to report exact hit count for")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    conversation_col = data_cfg["conversation_col"]
    harmful_col = data_cfg["harmful_col"]
    response_col = data_cfg["response_col"]
    k_last_turns = data_cfg.get("k_last_turns", 2)
    ultrachat_n = data_cfg.get("ultrachat_retain_n", 5000)

    df = pd.read_parquet(data_cfg["train_parquet"])
    df = df[df[harmful_col].notna()].copy()
    df[harmful_col] = df[harmful_col].astype(bool)
    df_harmful = df[df[harmful_col]].reset_index(drop=True)
    df_safe = df[~df[harmful_col]].reset_index(drop=True)

    print(f"[data] df_harmful rows: {len(df_harmful)}")
    print(f"[data] df_safe rows:    {len(df_safe)}")

    # ---- L_cb: total circuit_breaker_orig entries ----
    cb_window_counts = []
    for _, row in df_harmful.iterrows():
        conv = parse_conversation(row.get(conversation_col, []))
        n_windows = count_cb_windows_for_row(conv, args.window_turns, args.window_stride)
        cb_window_counts.append(n_windows)

    L_cb = sum(cb_window_counts)
    print(f"[D_s] circuit_breaker_orig size (L_cb): {L_cb}")
    print(f"      (avg windows/row: {L_cb/len(df_harmful):.2f}, "
          f"min: {min(cb_window_counts)}, max: {max(cb_window_counts)})")

    # ---- L_retain: orig_s_retain entries ----
    # source 1: df_safe -> 1 retain row each
    retain_from_safe = len(df_safe)

    # source 2: harmful sub-conversation prefixes -> 0 or 1 per harmful row
    retain_from_harmful_prefix = 0
    for _, row in df_harmful.iterrows():
        conv = parse_conversation(row.get(conversation_col, []))
        retain_from_harmful_prefix += count_retain_rows_for_harmful_row(conv, k_last_turns)

    # source 3: ultrachat (capped at ultrachat_n, but actual count depends on
    # how many items in the test_sft split have >=2 messages -- assume cap is hit)
    retain_from_ultrachat = ultrachat_n

    L_retain = retain_from_safe + retain_from_harmful_prefix + retain_from_ultrachat
    print(f"[D_r] orig_s_retain size (L_retain): {L_retain}")
    print(f"      from df_safe:           {retain_from_safe}")
    print(f"      from harmful prefixes:  {retain_from_harmful_prefix}")
    print(f"      from ultrachat (capped):{retain_from_ultrachat}")

    # ---- L_total ----
    L_total = max(L_cb, L_retain)
    print(f"\n[dataset] __len__ = max(L_retain, L_cb) = {L_total}")

    # ---- total samples drawn during training ----
    max_steps = train_cfg.get("max_steps", 300)
    per_device_bs = train_cfg.get("per_device_train_batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    num_gpus = args.num_gpus

    total_samples = max_steps * per_device_bs * grad_accum * num_gpus
    print(f"\n[training] max_steps={max_steps} per_device_bs={per_device_bs} "
          f"grad_accum={grad_accum} num_gpus={num_gpus}")
    print(f"[training] total samples drawn = {total_samples}")

    # ---- expected hits per cb-set index ----
    expected_hits_per_cb_index = total_samples / L_cb
    print(f"\n[coverage] expected hits per circuit_breaker_orig index "
          f"(uniform draw over {L_total}): {expected_hits_per_cb_index:.2f}")
    print(f"[coverage] expected epochs over dataset: {total_samples / L_total:.2f}")

    # ---- specific row ----
    if args.row_index is not None:
        idx = args.row_index
        if idx < 0 or idx >= len(df_harmful):
            print(f"\n[row] index {idx} out of range (0..{len(df_harmful)-1})")
            return

        # cumulative position of this row's windows in circuit_breaker_orig
        # (before shuffle -- shuffle doesn't change *how many* dataset indices
        # map to this row's windows, just which ones)
        n_windows_this_row = cb_window_counts[idx]
        print(f"\n[row {idx}] produces {n_windows_this_row} window(s) in circuit_breaker_orig")

        # each window occupies one slot in circuit_breaker_orig (size L_cb).
        # dataset index j maps to circuit_breaker_orig[j % L_cb].
        # number of j in [0, L_total) hitting a given cb slot:
        hits_per_slot = L_total // L_cb + (1 if (L_total % L_cb) else 0)  # upper bound
        hits_per_slot_min = L_total // L_cb
        slots_with_extra = L_total % L_cb

        print(f"[row {idx}] per dataset pass (len={L_total}), each of this row's "
              f"{n_windows_this_row} cb slot(s) is hit "
              f"{hits_per_slot_min} or {hits_per_slot_min+1} times "
              f"(slots_with_extra={slots_with_extra} out of {L_cb})")

        min_hits_per_pass = n_windows_this_row * hits_per_slot_min
        max_hits_per_pass = n_windows_this_row * (hits_per_slot_min + 1)
        print(f"[row {idx}] -> total slot-hits per dataset pass: "
              f"{min_hits_per_pass} to {max_hits_per_pass}")

        passes = total_samples / L_total
        print(f"[row {idx}] -> over {passes:.2f} passes (full run): "
              f"~{min_hits_per_pass*passes:.1f} to {max_hits_per_pass*passes:.1f} "
              f"total appearances in training batches")


if __name__ == "__main__":
    main()