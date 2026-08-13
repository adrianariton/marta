from openai import OpenAI
from core.attacks.datastore.read import get_data, Frame
from core.attacks.interfaces import OneShotConversation
import json
import pandas as pd
import uuid

BASE_FOLDER = "./datasets"

folders_and_tags = [
    ### bagell
    # crescendo
    (
        "crescendo_sorry_qwen3n_vs_cbllama__async_batch_bagel_eval_3repl_3x8x5",
        "crescendo",
        "sorry-bench/sorry-bench-202503",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "cbllama",
        "jondurbin/bagel-34b-v0.2",
        True,
    ),
    (
        "crescendo_sorry_qwen3n_vs_cbadicbmixed2t001__async_batch_bagel_eval_3repl_3x8x5",
        "crescendo",
        "sorry-bench/sorry-bench-202503",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "mt-cb",
        "jondurbin/bagel-34b-v0.2",
        True,
    ),
]

# folders_and_tags = [m for m in folders_and_tags if "hb" in m[0] and "crescendo" in m[0]]


models = {
    "door": "wicai24/Llama-3-8B-Instruct-W-DOOR-exponential",
    "derta": "Youliang/llama3-8b-instruct-lora-derta-100step",
    "guardr": "Qwen/Qwen3Guard-Gen-8B::meta-llama/Meta-Llama-3-8B-Instruct::response_moderation",
    "guardq": "Qwen/Qwen3Guard-Gen-8B::meta-llama/Meta-Llama-3-8B-Instruct::question_moderation",
    "cbllama": "GraySwanAI/Llama-3-8B-Instruct-RR",
    "nemor": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3::meta-llama/Meta-Llama-3-8B-Instruct::response_moderation",
    "nemoq": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3::meta-llama/Meta-Llama-3-8B-Instruct::question_moderation",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mt-cb": "MT-CB"
}
import os
import traceback
from core.datasets.safety.harmbench import HarmBench

from core.datasets.safety.advbench import AdvBench
from core.datasets.safety.donotanswer import LAODoNotAnswer
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.xstest import XSTest
from core.datasets.safety.sorrybench import SorryBench

import re

adv_df = AdvBench().get_first(n=1000)
lao_df = LAODoNotAnswer().get_first(n=1000)
sst_df = BertievidgenSimpleSafetyTests().get_first(n=1000)
xss_df = XSTest().get_first(1000)
hb_df = HarmBench().get_first(1000)
sorry_df = SorryBench().get_first(500)


def get_goal(ds, _id):
    if ds == "walledai/AdvBench":
        df = adv_df
        match = re.search(r"adv_\d{3}", _id)
        _id = match.group()
        q_col = "prompt"
    elif ds == "Bertievidgen/SimpleSafetyTests":
        df = sst_df
        match = re.search(r"sst_\d{3}", _id)
        _id = match.group()
        q_col = "prompt"
    elif ds == "LibrAI/do-not-answer":
        df = lao_df
        match = re.search(r"lao_(\d+)", _id)
        _id = int(match.group(1))
        q_col = "question"
    elif ds == "walledai/XSTest":
        df = xss_df
        match = re.search(r"xstest_(?:safe|unsafe)_\d+", _id)
        _id = match.group()
        q_col = "prompt"
    elif ds == "walledai/HarmBench":
        df = hb_df
        # print(_id)
        match = re.search(r"hb_(\d+)", _id)
        _id = "hb_" + (match.group(1))
        q_col = "Behavior"
    elif ds == "sorry-bench/sorry-bench-202503":
        df = sorry_df
        
        match = re.search(r"(\d+)", _id)
        _id = "sorry_" + (match.group(1))
        q_col = "prompt"
        # print(_id)
    else:
        raise Exception("DS not found")
    try:
        return df[df["id"] == _id][q_col].iloc[0]
    except Exception as e:
        print(e)
        print(df[df["id"] == _id][q_col])
        print(_id)
        print(df.head(5))
        print("======")


dicts = []
for dir, tag, dataset, attacker, _target, _evaluator, is_goal_harmful in folders_and_tags:

    directory_path = BASE_FOLDER + "/" + dir
    target = models.get(_target, _target)
    try:
        os.listdir(directory_path)
    except Exception as e:
        continue
    entries = os.listdir(directory_path)
    all_dirs = []
    for entry_name in entries:
        full_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(full_path):
            all_dirs.append(entry_name)
    all_data: list[tuple[str, Frame]] = []
    for subdir in all_dirs:
        try:
            data: list[Frame] = get_data(subdir, directory_path, str(tag).lower())
            clean_data = [d for d in data if (d.has_attack_query() and not d.is_done())]
            clean_data = [d for d in clean_data if (not d.has_empty_attack_query())]
            clean_data = [(subdir, d) for d in clean_data if not d.is_done()]
            all_data = all_data + clean_data
        except Exception as e:
            print(str(e))
    print(f"After {dir}: {len(all_data)}")
    print()

    last_runid = None
    prev_id = None
    curr_id = None
    cnt = 0
    prev_convo = None
    for row_id, d in all_data:
        if cnt == 3:
            prev_convo = None
            last_runid = d.run_id
            prev_id = curr_id
            cnt = 0
            continue
        if last_runid != d.run_id:
            prev_id = None
            cnt = 0
            prev_convo = None
            if dicts:
                # dicts[-1]["is_final"] = True
                ii = -1
                while ii > 0:
                    if dicts[ii]["attacker_refused"] == True or dicts[ii]["empty"] == True:
                        ii -= 1
                    else:
                        break
                dicts[ii]["is_final"] = True
        # try:
        curr_id = uuid.uuid4().hex
        if not d.has_attack_query():
            continue
        attack_q = d.attack_query()
        if "xstest" in dataset.lower():
            if "_safe" in row_id:
                is_goal_harmful = False
        try:
            if (
                d.is_empty_frame()
                or (not d.has_agent_response())
                or (not d.has_eval_or_refused())
                or (d.is_eval_missing())
            ):
                dicts.append(
                    {
                        "conversation_id": d.run_id,
                        "prev_entry_conversation_id": last_runid,
                        #
                        "attack_type": tag,
                        "dataset": dataset,
                        "dataset_row_id": row_id,
                        "goal": get_goal(dataset, row_id),
                        "turn_id": curr_id,
                        "prev_turn_id": prev_id,
                        "turn_count": cnt,
                        "convo_size": int(prev_convo.size()) if prev_convo else None,
                        "prev_convo_size": int(prev_convo.size()) if prev_convo else None,
                        "empty": d.has_attack_query() and d.is_empty_frame(),
                        #
                        "attacker": attacker,
                        "evaluator": _evaluator,
                        "target": target,
                        #
                        "query": attack_q,
                        "response": None,
                        "eval_type": "PEC7",
                        "eval_proximity": None,
                        "eval_erosion": None,
                        "eval_cooperation": None,
                        "goal_revealed": None,
                        "harmful": None,
                        "is_goal_harmful": is_goal_harmful,
                        "is_refused": None,
                        #
                        "conversation": prev_convo.to_default() if prev_convo else None,
                        "conversation_format": prev_convo.to_auto() if prev_convo else None,
                        "attacker_refused": d.is_eval_missing(),
                        "has_is_refused": d.has_is_refused(),
                        "has_eval": d.has_response_evaluation(),
                    }
                )
                cnt += 1
                last_runid = d.run_id
                prev_id = curr_id
                continue

            agent_r = d.agent_response()
            eval_ = d.response_evaluation_or_refusal()

            eval_ = {
                "proximity": eval_.get("proximity", 0),
                "erosion": eval_.get("erosion", 0),
                "cooperation": eval_.get("cooperation", 0),
                "goal_revealed": eval_.get("goal_revealed", False),
                "refused": d.has_is_refused() and d.is_response_refused(),
            }

            convo = d.agent_conversation()

            dicts.append(
                {
                    "conversation_id": d.run_id,
                    "prev_entry_conversation_id": last_runid,
                    #
                    "attack_type": tag,
                    "dataset": dataset,
                    "dataset_row_id": row_id,
                    "goal": get_goal(dataset, row_id),
                    "turn_id": curr_id,
                    "prev_turn_id": prev_id,
                    "turn_count": cnt,
                    "convo_size": int(convo.size()) if convo else None,
                    "prev_convo_size": int(prev_convo.size()) if prev_convo else None,
                    "empty": d.has_attack_query() and d.is_empty_frame(),
                    #
                    "attacker": attacker,
                    "evaluator": _evaluator,
                    "target": target,
                    #
                    "query": attack_q,
                    "response": agent_r,
                    "eval_type": "PEC7",
                    "eval_proximity": eval_["proximity"],
                    "eval_erosion": eval_["erosion"],
                    "eval_cooperation": eval_["cooperation"],
                    "goal_revealed": eval_["goal_revealed"],
                    "harmful": eval_["goal_revealed"] and is_goal_harmful,
                    "is_goal_harmful": is_goal_harmful,
                    "is_refused": eval_["refused"],
                    #
                    "conversation": convo.to_default() if convo else None,
                    "conversation_format": convo.to_auto() if convo else None,
                    "attacker_refused": False,
                    "has_is_refused": d.has_is_refused(),
                    "has_eval": d.has_response_evaluation(),
                }
            )
            cnt += 1
            prev_convo = convo
            last_runid = d.run_id
            prev_id = curr_id
        except Exception as e:
            pass

# dicts = [d for d in dicts if d.get("is_final") == True]
# df = pd.DataFrame(dicts)
# print(f"Done DF")
# df["rollback"] = df["convo_size"] - df["prev_convo_size"] - 2
# valid_ids = df[(df["is_final"] == True) & ((df["harmful"] == True) | (df["harmful"] == False))][
#     "conversation_id"
# ].unique()
# filtered_df = df[df["conversation_id"].isin(valid_ids)]

# suff = "_new"
# print("Saving...")
# # use multiple files if the dataframe is too large
# filtered_df.to_csv(f"train/huge_data{suff}.csv")
# filtered_df.to_parquet(f"train/huge_data{suff}.parquet")
import numpy as np

df = pd.DataFrame(dicts)
print("Done DF")

# 2. Run your calculations
df["rollback"] = df["convo_size"] - df["prev_convo_size"] - 2

# 3. Clean up the valid IDs logic
# Drops rows where 'harmful' is NaN/missing, matching your original intent
filtered_df = df.dropna(subset=["harmful"])

suff = "_new"
print("Splitting and saving...")

# 4. Split the giant dataframe into smaller chunks
# Change chunk_size based on how many rows you want per file (e.g., 500,000)
chunk_size = 50_000
num_chunks = max(1, len(filtered_df) // chunk_size)
df_chunks = np.array_split(filtered_df, num_chunks)

# 5. Save each chunk to its own file
for i, chunk in enumerate(df_chunks):
    # Saves as train/huge_data_new_0.parquet, train/huge_data_new_1.parquet, etc.
    parquet_path = f"train/sorry/sorrymen{suff}_{i}.parquet"
    chunk.to_parquet(parquet_path, index=False)


print(f"Successfully split data into {num_chunks} individual files.")

print(f"saved to and train/parquets/")
