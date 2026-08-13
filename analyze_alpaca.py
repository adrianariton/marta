from abc import ABC, abstractmethod


class Attack(ABC):
    @abstractmethod
    def get_base_folder(self, index):
        pass

    @abstractmethod
    def get_subfolder(self, index):
        pass

    @abstractmethod
    def attack_id(self):
        pass


def get_against_name(index):
    if index == 9:
        return "cbadimixed2000"
    elif index == 5:
        return "cbllama"
    else:
        raise ValueError("Invalid index")


class A_Crescendo(Attack):
    def get_base_folder(self, index):
        if index == 9:
            return "datasets/crescendo_alpaca_qwen3n_vs_cbadimixed2000__async_batch_bagel_eval_3repl_3x8x5"

        elif index == 5:
            return "datasets/crescendo_alpaca_qwen3n_vs_cbllama__async_batch_bagel_eval_3repl_3x8x5"

    def get_subfolder(self, index):
        return "crescendo_alpaca_" + index

    def attack_id(self):
        return "crescendo"



xstest_group_keys = ["category"]

attacks: list[Attack] = [A_Crescendo()]

from core.datasets.safety.alpaca import AlpacaBench
import matplotlib.pyplot as plt
import os
import ast

df = AlpacaBench().sample_randomly(n=200, random_state=42)

def plot_base_folder(base_folder, target_name, attack: Attack):
    datas = []
    for i, row in df.iterrows():

        # if not ("How" in row["prompt"]):
        #     continue

        output_folder_name = f"{base_folder}/" + attack.get_subfolder(row["id"])
        info = output_folder_name + "/info.txt"

        if os.path.exists(info):
            with open(info, "rt") as f:
                data = f.readline()
                data = data.replace("'", '"')
                data = ast.literal_eval(data)
                if len(list(data.keys())) == 0:
                    datas.append({"jailbroken": False, "data": data, **row.to_dict()})
                else:
                    first_key = sorted(list(data.keys()))[0]
                    dt = {first_key: data[first_key]}
                    datas.append({"jailbroken": True, "data": dt, **row.to_dict()})

        else:
            datas.append({"jailbroken": None, "data": {}, **row.to_dict()})
    assert len(datas) == 200, f"Expected 200 unsafe cases, got {len(datas)}"
    return datas
import pandas as pd
import numpy as np
def plot_base_folder_safe(
    base_folder, target_name, attack: Attack
):
    from core.attacks.datastore.read import Frame, get_data

    datas = []
    for i, row in df.iterrows():

        # if not ("How" in row["prompt"]):
        #     continue

        output_folder_name = f"{base_folder}/" + attack.get_subfolder(row["id"])
        info = output_folder_name + "/info.txt"

        frames: list[Frame] = get_data(
            attack.get_subfolder(row["id"]), base_folder, tag_ATTACK=attack.attack_id()
        )
        entry = {"jailbroken": None, "data": {}, **row.to_dict()}
        if os.path.exists(info):
            with open(info, "rt") as f:
                data = f.readline()
                data = data.replace("'", '"')
                data = ast.literal_eval(data)
                if len(list(data.keys())) == 0:
                    entry["jailbroken"] = False
                    entry["data"] = data
                else:
                    first_key = sorted(list(data.keys()))[0]
                    dt = {first_key: data[first_key]}
                    entry["jailbroken"] = True
                    entry["data"] = dt
        else:
            entry["data"] = {}

        refused = 0
        unrefused = 0
        for frame in frames:
            if frame.has_is_refused() and frame.is_response_refused():
                refused += 1
            else:
                unrefused += 1

        entry["refusal_rate"] = (
            refused / (refused + unrefused) if (refused + unrefused) > 0 else np.NaN
        )
        entry["refused"] = refused
        entry["total"] = refused + unrefused
        datas.append(entry)
    # assert len(datas) == 200, f"Expected 250 safe cases, got {len(datas)}"
    return pd.DataFrame(datas)

df_ref9 = plot_base_folder_safe(attacks[0].get_base_folder(9), get_against_name(9), attacks[0])
df_ref5 = plot_base_folder_safe(attacks[0].get_base_folder(5), get_against_name(5), attacks[0])


print("CBL over-refusal rate:", df_ref5["refusal_rate"].mean())
print("3-MCB over-refusal rate:", df_ref9["refusal_rate"].mean())


