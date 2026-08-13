import pandas as pd
from datasets import Dataset, DatasetDict

# 1. Read your existing local Parquet files via Pandas
df_train = pd.read_parquet("train_corrected.parquet")
df_test = pd.read_parquet("test_corrected.parquet")
df_val = pd.read_parquet("val_corrected.parquet")

# 2. Convert them into Hugging Face Dataset structures
dataset_dict = DatasetDict(
    {
        "mini_train": Dataset.from_pandas(df_train),
        "mini_test": Dataset.from_pandas(df_test),
        "mini_val": Dataset.from_pandas(df_val),
    }
)

# 3. Push cleanly to the Hub.
# Because the split names are brand new, your old data shards are safe!
dataset_dict.push_to_hub("aariton/MARTA-turn-by-turn")
