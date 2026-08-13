import pandas as pd
from core.attacks.interfaces import OneShotConversation

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
        old_convo = final_row_convo.copy()
        final_row_reply = final_row["response"]
        final_row_alternative = final_row["alternative_harmless_response"]

        if final_row["harmful"] == False:
            continue

        for _, row in group.iterrows():
            if row["harmful"] == True or row["is_refused"] == True:
                print(
                    f"Found harmful/refused response in convo {convo_id}, replacing with alternative.",
                    flush=True,
                )
                alt_response = row["alternative_harmless_response"]
                resp = final_row["response"]
                final_row_convo.replace(resp, by_y=alt_response, of_role="assistant")
        try:

            all_convos.append(
                {
                    "prompt": final_row_convo.to_auto(),
                    "chosen": [final_row_alternative],
                    "rejected": [final_row_reply],
                }
            )
        except Exception as e:
            print(e)
            print(f"{final_row_convo._messages}")
            exit(-1)
import json

with open("dpo_data_clean.jsonl", "w") as f:
    for entry in all_convos:
        f.write(json.dumps(entry) + "\n")
