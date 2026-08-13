"""
debug_conversation.py
---------------------
Inspects the raw content of conversation_format column to diagnose parse issues.

Usage:
    python debug_conversation.py --parquet ./train/train.parquet
"""

import argparse
import ast
import json
import reprlib
import pandas as pd
import numpy as np


def parse_conversation(raw):
    """Same logic as in training script."""
    if isinstance(raw, list):
        return raw, "already_list"
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result, "json_loads"
        except Exception:
            pass
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return result, "ast_literal_eval"
        except Exception:
            pass
    if isinstance(raw, np.ndarray):
        try:
            result = raw.tolist()
            if isinstance(result, list):
                return result, "numpy_to_list"
        except Exception:
            pass
    return [], f"FAILED (type={type(raw).__name__})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="./train/train.parquet")
    parser.add_argument("--col", default="conversation_format")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Columns: {list(df.columns)}\n")
    print(f"Shape: {df.shape}\n")

    col = args.col
    if col not in df.columns:
        print(f"Column '{col}' NOT FOUND. Available: {list(df.columns)}")
        return

    print(f"dtype of '{col}': {df[col].dtype}\n")
    print("=" * 60)

    for i, raw in enumerate(df[col].iloc[: args.n]):
        parsed, method = parse_conversation(raw)
        print(f"\n--- Row {i} ---")
        print(f"  type      : {type(raw).__name__}")
        print(f"  parse     : {method}")
        if method.startswith("FAILED"):
            # Print raw repr truncated
            print(f"  raw[:500] : {reprlib.repr(str(raw)[:500])}")
        else:
            print(f"  turns     : {len(parsed)}")
            for j, turn in enumerate(parsed[:3]):
                print(f"    turn[{j}] : {turn}")
        print()

    # Count how many fail
    total = len(df)
    failed = sum(1 for raw in df[col] if parse_conversation(raw)[1].startswith("FAILED"))
    print(f"\nSummary: {failed}/{total} rows fail to parse ({100*failed/total:.1f}%)")


if __name__ == "__main__":
    main()
