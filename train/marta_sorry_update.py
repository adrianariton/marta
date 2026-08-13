# XSTest data should have Qwen/Qwen2.5-32B-Instruct as attacker
import pandas as pd
import numpy as np
import os

def prepare_and_shard_for_hf(df, output_dir="hf_dataset_shards", num_shards=10):
    """
    Adds conversation outcome labels, groups conversations seamlessly,
    and saves the DataFrame into sharded Parquet files optimized for Hugging Face.
    """
    print("Step 1: Calculating Conversation Outcomes...")
    # Identify which conversation_ids ever resulted in a harmful outcome
    harmful_convos = set(df.loc[df["harmful"] == True, "conversation_id"].unique())

    # Efficiently assign the outcome label
    df["conversation_outcome"] = np.where(
        df["conversation_id"].isin(harmful_convos), "Harmful Ending", "Unharmful Ending"
    )

    # Create the destination directory
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Step 2: Sharding dataset into {num_shards} chunks while keeping conversations intact..."
    )
    # Extract unique conversation IDs and shuffle them to mix topics evenly across shards
    unique_convos = df["conversation_id"].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_convos)

    # Split conversation IDs into roughly equal sub-arrays
    convo_splits = np.array_split(unique_convos, num_shards)

    for idx, convo_set in enumerate(convo_splits):
        # Gather all corresponding message turns for this subset of conversations
        shard_df = df[df["conversation_id"].isin(convo_set)]

        # Follow HF standard file naming format: train-0000X-of-0000Y.parquet
        shard_filename = os.path.join(output_dir, f"train-{idx:05d}-of-{num_shards:05d}.parquet")

        # Parquet files keep column types intact (bool, float64, objects) perfectly for HF
        shard_df.to_parquet(shard_filename, index=False, compression="snappy")
        print(f" -> Saved shard {idx+1}/{num_shards}: {shard_filename} ({len(shard_df)} rows)")

    print(f"\nSuccess! Your dataset shards are ready in '{output_dir}/'.")


df = pd.read_parquet("sorry/")
# df.loc[df["dataset"] == "walledai/XSTest", "attacker"] = "Qwen/Qwen2.5-32B-Instruct"
# shard save
prepare_and_shard_for_hf(df, output_dir="sorry_corrected", num_shards=16)

