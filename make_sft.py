import pandas as pd
from core.attacks.interfaces import OneShotConversation
import random
import numpy as np

df = pd.read_parquet("train/alternative_huge_data.parquet")
import re


def strip_apology(text):
    patterns = [
        r"^(?i)i['’]m\s+sorry(?:\s*,\s*|\s+but\s+)?",
        r"^(?i)i\s+apologize(?:\s*,\s*|\s+but\s+)?",
        r"^(?i)unfortunately\s*,\s*",
        r"^(?i)i\s+cannot\s+fulfill\s+this\s+request\s*.?\s+",
    ]

    new_text = text
    for p in patterns:
        new_text = re.sub(p, "", new_text).strip()

    # Capitalize the new first letter
    if new_text:
        new_text = new_text[0].upper() + new_text[1:]

    return new_text


dfs = df.groupby("conversation_id")
all_convos = []
for convo_id, group in dfs:
    final_rows = group[(group["is_final"] == True) | (group["is_refused"] == True)]
    # assert (
    #     len(final_rows) == 1
    # ), f"Expected 1 final row for convo {convo_id}, found {len(final_rows)}"
    for _, final_row in final_rows.iterrows():
        if final_row["conversation"] is not None:
            final_row_convo = OneShotConversation.from_default(final_row["conversation"])
        else:
            final_row_convo = OneShotConversation()
            final_row_convo.add_text_message(final_row["query"], "user")

        final_row_convo.add_text_message(final_row["response"], "assistant")
        old_convo = final_row_convo.copy()

        for _, row in group.iterrows():
            if row["harmful"] == True or row["is_refused"] == True:
                alt_response = row["alternative_harmless_response"]
                resp = final_row["response"]
                final_row_convo.replace(resp, by_y=alt_response, of_role="assistant")

        try:

            all_convos.append(
                {
                    "conversation": final_row_convo.to_auto(),
                    "bad_conversation": old_convo.to_auto(),
                    "raw_final": final_row.to_json(),
                    "raw": group.to_json(),
                    "goal": final_row["goal"],
                    "dataset_row_id": final_row["dataset_row_id"],
                }
            )
        except Exception as e:
            print(e)
            print(f"{final_row_convo._messages}")
            exit(-1)
import json

with open("sft_data_clean.jsonl", "w") as f:
    for entry in all_convos:
        f.write(
            json.dumps(
                {
                    "messages": entry["conversation"],
                    "goal": entry["goal"],
                    "dataset_row_id": entry["dataset_row_id"],
                }
            )
            + "\n"
        )
goals = [entry["goal"] for entry in all_convos]
goals_unique = set(goals)
goals_by_freq = {g: goals.count(g) for g in goals_unique}

# 1. Convertim set-ul în listă pentru NumPy
unique_goals_list = list(goals_unique)
k = int(len(unique_goals_list) * 0.75)

# 2. Calculăm probabilitățile folosind lista ordonată
weights = np.array([goals_by_freq[g] for g in unique_goals_list], dtype=float)
probabilities = weights / weights.sum()

# 3. Folosim unique_goals_list în loc de goals_unique (set)
np.random.seed(42)
train_goals_array = np.random.choice(unique_goals_list, size=k, replace=False)

# 4. Transformăm în set pentru operații de scădere și căutare rapidă
train_goals = set(train_goals_array)
test_goals = goals_unique - train_goals

print(f"Total unique goals: {len(goals_unique)}")
print(f"Train goals: {len(train_goals)}")
print(f"Test goals: {len(test_goals)}")

# Restul codului rămâne la fel
train_entries = [entry for entry in all_convos if entry["goal"] in train_goals]
test_entries = [entry for entry in all_convos if entry["goal"] in test_goals]
with open("sft_data_train.jsonl", "w") as f:
    for entry in train_entries:
        f.write(
            json.dumps(
                {
                    "messages": entry["conversation"],
                    "goal": entry["goal"],
                    "dataset_row_id": entry["dataset_row_id"],
                }
            )
            + "\n"
        )

with open("sft_data_test.jsonl", "w") as f:
    for entry in test_entries:
        f.write(
            json.dumps(
                {
                    "messages": entry["conversation"],
                    "goal": entry["goal"],
                    "dataset_row_id": entry["dataset_row_id"],
                }
            )
            + "\n"
        )
with open("sft_data.jsonl", "w") as f:
    for entry in all_convos:
        f.write(json.dumps(entry) + "\n")
