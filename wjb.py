import pandas as pd
import json
import os
from huggingface_hub import hf_hub_download

# 1. Define the repository and the files you want
repo_id = "wicai24/Jailbreak-Datasets"
files = [
    "gemma_autodan.json",
    "llama_autodan.json",
    "gemma_gcg.json",
    "llama_gcg.json",
    "gemma_multi.json",
    "llama_multi.json",
]

all_data = []

# 2. Download and Load
for file_name in files:
    try:
        print(f"Downloading {file_name}...")
        # This downloads the file to your HF cache and returns the local path
        local_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset")

        with open(local_path, "r") as f:
            data = json.load(f)
            # Add metadata so you know which model/attack this row belongs to
            for entry in data:
                entry["source_file"] = file_name
            all_data.extend(data)

    except Exception as e:
        print(f"Could not download/load {file_name}: {e}")

# 3. Create DataFrame
if all_data:
    df = pd.DataFrame(all_data)
    print(f"\n✅ Success! Loaded {len(df)} rows.")
    print(df.head())

    # Save it so you have a clean copy for your thesis work
    df.to_csv("jailbreak_combined.csv", index=False)
    print("Saved to jailbreak_combined.csv")
else:
    print("\n❌ No data was loaded. Check your internet connection or repository name.")
