import os
import ast
import argparse
import numpy as np
import pandas as pd

from core.datasets.safety.sorrybench import SorryBench


# =========================
# Args
# =========================

parser = argparse.ArgumentParser()

parser.add_argument("--algo", default="goat")
parser.add_argument("--ds", default="bss")
parser.add_argument("--attacker", default="gemini")
parser.add_argument("--base", default="datasets")

args = parser.parse_args()


# =========================
# Dataset
# =========================

safety_df = SorryBench().get_first(n=500)


all_versus = {
    "cbllama": "CBL",
    "cbadimixed2000": "MT-CB",
}


k_range = [1, 2]


# =========================
# Load results
# =========================

def get_df(versus):

    BASE_FOLDER = (
        f"{args.base}/"
        f"{args.algo}_{args.ds}_{args.attacker}_vs_{versus}"
    )

    print("Loading:", BASE_FOLDER)

    datas = []


    for _, row in safety_df.iterrows():

        for k in k_range:

            suff = f"_i{k}"

            folder = (
                f"{BASE_FOLDER}/"
                f"{args.algo}_sorry_{row['id']}{suff}"
            )

            info = folder + "/info.txt"


            if os.path.exists(info):

                with open(info, "rt") as f:

                    line = f.readline()

                    try:
                        data = ast.literal_eval(line)

                    except:

                        data = {}


                    keys = [1,2]

                    jailbroken = any(
                        data.get(str(x), False) == True
                        or data.get(x, False) == True
                        for x in keys
                    )


                    datas.append({

                        "id": row["id"],

                        "k": k,

                        "jailbroken": jailbroken,

                        "exists": True

                    })


            else:

                datas.append({

                    "id": row["id"],

                    "k": k,

                    "jailbroken": False,

                    "exists": False

                })


    return pd.DataFrame(datas)



# =========================
# DR computation
# =========================
def compute_dr(df):

    results = []


    for conv_id, group in df.groupby("id"):

        group = group.sort_values("k")


        values = (
            group["jailbroken"]
            .astype(int)
            .values
        )


        if len(values) < 2:
            continue


        deltas = np.diff(values)


        # harmfulness decreases
        negative = np.abs(
            deltas[deltas < 0]
        ).sum()


        # harmfulness increases
        positive = (
            deltas[deltas > 0]
        ).sum()


        dr = negative / (
            negative
            +
            positive
            +
            1e-8
        )


        outcome = (
            "Harmful Ending"
            if values[-1] == 1
            else "Unharmful Ending"
        )


        results.append({

            "id": conv_id,

            "DR": dr,

            "conversation_outcome": outcome

        })


    return pd.DataFrame(results)



# =========================
# Run
# =========================

all_results = []


for target, name in all_versus.items():


    df = get_df(target)

    print(df.head())
    dr = compute_dr(df)


    for outcome, group in dr.groupby(
        "conversation_outcome"
    ):

        all_results.append({

            "target": name,

            "conversation_outcome": outcome,

            "mean": group["DR"].mean(),

            "median": group["DR"].median(),

            "std": group["DR"].std(),

            "count": len(group)

        })



# =========================
# Save / print
# =========================


result_df = pd.DataFrame(all_results)


result_df = result_df.sort_values(
    [
        "conversation_outcome",
        "mean"
    ],
    ascending=False
)


print("\n")
print("="*80)
print("DAMPING RESISTANCE SUMMARY")
print("="*80)

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


out = (
    f"graphs/sorry/"
    f"{args.algo}/"
    f"text/"
    f"DR_summary_{args.ds}.csv"
)


os.makedirs(
    os.path.dirname(out),
    exist_ok=True
)


result_df.to_csv(
    out,
    index=False
)


print("\nSaved:", out)