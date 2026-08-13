from huggingface_hub import HfApi
import yaml

api = HfApi()
repo_id = "aariton/MARTA-turn-by-turn"
# 1. FIX: Call the correct method name 'dataset_info'
info = api.dataset_info(repo_id=repo_id)

# 2. Gather all the files listed in the remote repository
remote_files = [sibling.rfilename for sibling in info.siblings]

# 3. Filter for Parquet files and categorize them dynamically
main_shards = [f for f in remote_files if "train-" in f and "mini-" not in f]
mini_train = any("mini-train" in f for f in remote_files)
mini_test = any("mini-test" in f for f in remote_files)
mini_val = any("mini-val" in f for f in remote_files)

# Build the configuration mapping dict
configs_block = {"configs": []}

# Add Main Config
if main_shards:
    configs_block["configs"].append(
        {"config_name": "default", "data_files": [{"split": "train", "path": "train-*.parquet"}]}
    )

# Add Mini Config pointing directly into your 'data/' directory folder
if mini_train or mini_test or mini_val:
    data_files = []
    if mini_train:
        data_files.append({"split": "mini-train", "path": "data/mini-train-*.parquet"})
    if mini_test:
        data_files.append({"split": "mini-test", "path": "data/mini-test-*.parquet"})
    if mini_val:
        data_files.append({"split": "mini-val", "path": "data/mini-val-*.parquet"})

    configs_block["configs"].append({"config_name": "mini_version", "data_files": data_files})

print("\n--- COPY EVERYTHING BELOW THIS LINE ---")
print(yaml.dump(configs_block, sort_keys=False, default_flow_style=False))
print("--- END OF BLOCK ---")
