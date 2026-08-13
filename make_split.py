"""
split_dataset.py
----------------
Splits the safety parquet into train/val sets.

Rules:
  - Keep only rows where attack_type in {fitd, race, goat} (case-insensitive)
  - Drop rows where harmful is None
  - All datasets → 80/20 split by extracted goal ID (sst_NNN, hb_N+, xstest_N+)
  - No goal ID appears in both train and val

Usage:
    python split_dataset.py --input data.parquet --output_dir ./splits
    python split_dataset.py --input data.parquet --output_dir ./splits --seed 42
"""

import re
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ATTACK_TYPES = {"fitd", "race", "goat"}
TEST_ATTACK_TYPES = {"crescendo"}

DATASET_COL = "dataset"
HARMFUL_COL = "harmful"
ATTACK_TYPE_COL = "attack_type"
ROW_ID_COL = "dataset_row_id"

# ---------------------------------------------------------------------------
# Goal ID extraction
# ---------------------------------------------------------------------------


def extract_goal_id(row_id: str) -> str | None:
    """
    Extract the canonical goal ID from a dataset_row_id.
      sst_001_fitd_0   -> sst_001
      hb_42_race_1     -> hb_42
      xstest_7_goat_0  -> xstest_7
    Returns None if no pattern matches.
    """
    match = re.search(r"(sst_\d+|hb_\d+|xstest_safe_\d{3}|xstest_unsafe_\d{3})", str(row_id))
    print(row_id, match)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Load / filter
# ---------------------------------------------------------------------------


def load_and_filter(path: str, attack_types: set) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[load]   Total rows: {len(df)}")

    before = len(df)
    df = df[df[HARMFUL_COL].notna()].copy()
    print(f"[filter] Dropped {before - len(df)} rows with harmful=None → {len(df)} remaining")

    before = len(df)
    df = df[df[ATTACK_TYPE_COL].str.lower().isin(attack_types)].copy()
    print(
        f"[filter] Kept attack_type in {attack_types} → {len(df)} rows "
        f"(dropped {before - len(df)})"
    )

    df[HARMFUL_COL] = df[HARMFUL_COL].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_split(input_path: str, output_dir: str, seed: int = 42):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Crescendo test set ------------------------------------------------
    df_cresc = load_and_filter(input_path, TEST_ATTACK_TYPES)
    if not df_cresc.empty:
        test_out = output_path / "test.parquet"
        df_cresc.to_parquet(test_out, index=False)
        print(f"[saved] {test_out}  ({len(df_cresc)} rows)\n")

    # ---- Train / val -------------------------------------------------------
    df = load_and_filter(input_path, VALID_ATTACK_TYPES)

    # Extract goal ID for every row
    df["_goal_id"] = df[ROW_ID_COL].apply(extract_goal_id)

    # Warn about any rows we couldn't parse
    unparsed = df["_goal_id"].isna().sum()
    if unparsed:
        print(f"[warn]  {unparsed} rows have unparseable row IDs — they will go to train")

    all_trains, all_vals = [], []

    for ds_name in df[DATASET_COL].unique():
        sub = df[df[DATASET_COL] == ds_name].copy()
        print(f"\n[split] Dataset: {ds_name!r}  ({len(sub)} rows)")

        # Rows without a parsed goal ID go entirely to train
        no_goal = sub[sub["_goal_id"].isna()].drop(columns=["_goal_id"])
        has_goal = sub[sub["_goal_id"].notna()].copy()

        unique_goals = has_goal["_goal_id"].unique()
        print(f"        Unique goals: {len(unique_goals)}")

        if len(unique_goals) < 2:
            all_trains.append(pd.concat([no_goal, has_goal.drop(columns=["_goal_id"])]))
            continue

        train_goals, val_goals = train_test_split(unique_goals, test_size=0.2, random_state=seed)
        train_goals_set = set(train_goals)
        val_goals_set = set(val_goals)

        train = has_goal[has_goal["_goal_id"].isin(train_goals_set)].drop(columns=["_goal_id"])
        val = has_goal[has_goal["_goal_id"].isin(val_goals_set)].drop(columns=["_goal_id"])

        # Merge unparsed rows into train
        train = pd.concat([no_goal, train])

        print(f"        Train goals: {len(train_goals)} | Val goals: {len(val_goals)}")
        print(f"        Rows → train={len(train)}  val={len(val)}")

        all_trains.append(train)
        all_vals.append(val)

    train_df = pd.concat(all_trains).reset_index(drop=True)
    val_df = pd.concat(all_vals).reset_index(drop=True)

    # ---- Sanity check: zero overlap on goal IDs ----------------------------
    train_df["_goal_id"] = train_df[ROW_ID_COL].apply(extract_goal_id)
    val_df["_goal_id"] = val_df[ROW_ID_COL].apply(extract_goal_id)

    train_goals_all = set(train_df["_goal_id"].dropna())
    val_goals_all = set(val_df["_goal_id"].dropna())
    overlap = train_goals_all & val_goals_all

    print(f"\n[check] Unique goals in train : {len(train_goals_all)}")
    print(f"[check] Unique goals in val   : {len(val_goals_all)}")
    print(f"[check] Overlap (must be 0)   : {len(overlap)}")
    if overlap:
        print(f"[ERROR] Leaking goals: {sorted(overlap)[:10]}")

    train_df = train_df.drop(columns=["_goal_id"])
    val_df = val_df.drop(columns=["_goal_id"])

    print(f"\n[result] TRAIN total: {len(train_df)}")
    print(f"[result] VAL   total: {len(val_df)}")
    print(f"[result] Harmful in train: {train_df[HARMFUL_COL].sum()}")
    print(f"[result] Harmful in val:   {val_df[HARMFUL_COL].sum()}")

    train_out = output_path / "train.parquet"
    val_out = output_path / "val.parquet"
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(f"\n[saved] {train_out}")
    print(f"[saved] {val_out}")

    return train_df, val_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Split safety dataset into train/val.")
    parser.add_argument("--input", required=True, help="Path to input .parquet file")
    parser.add_argument("--output_dir", default="./splits", help="Directory to save parquets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_split(args.input, args.output_dir, seed=args.seed)
