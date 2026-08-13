import pandas as pd

val = pd.read_parquet("val.parquet")
train = pd.read_parquet("train.parquet")

# val = val[val["dataset"] == "Bertievidgen/SimpleSafetyTests"]
# train = train[train["dataset"] == "Bertievidgen/SimpleSafetyTests"]

val_goals = set(val["goal"].unique())
train_goals = set(train["goal"].unique())

print(f"Unique goals in val: {len(val_goals)}")
print(f"Unique goals in train: {len(train_goals)}")
print(f"Goals in val but not in train: {len(val_goals - train_goals)}")
print(f"Goals in train but not in val: {len(train_goals - val_goals)}")
