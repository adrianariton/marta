from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.harmbench import HarmBench
from core.datasets.safety.advbench import AdvBench

import os
import json
import ast
import pandas as pd
import argparse
import sys

safety_df = AdvBench().get_first(n=500)
parser = argparse.ArgumentParser(description="Dataset path generator")

parser.add_argument("algo", nargs="?", default="goat", help="Algorithm name (default: goat)")
parser.add_argument("ds", nargs="?", default="bss", help="Dataset name (default: bss)")
parser.add_argument("version", nargs="?", default="", help="Version suffix (optional)")
parser.add_argument("--attacker", "-a", default="gemini", help="Attacker name (default: gemini)")
parser.add_argument("--base", "-b", default="datasets", help="Base folder for datasets")
parsed_args = parser.parse_args()
suffix = f"__{parsed_args.version}" if parsed_args.version else ""

BASE_GRAPHS_FOLDER = f"graphs/tadv/"

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}")

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text")
BASE_GRAPHS_FOLDER = BASE_GRAPHS_FOLDER + parsed_args.algo + "/"
all_versus = {
    "cbllama": "CBL",
    # "dertains": "DER",
    # "door": "DOR",
    # # "guardresp": "GUQ",
    # # "guardllama": "GUA",
    # "llamainstruct": "LLA",
    # "guardr": "GUR",
    # "guardq": "GUQ",
    # "guardrnoh": "GUR",
    # "guardqnoh": "guq",
    # "nemor": "NEMR",
    # "nemoq": "NEMQ",
    "cbadicb3000001": "CB3000",
}

algo = parsed_args.algo
ds = parsed_args.ds


import matplotlib.pyplot as plt


def get_df(versus):
    BASE_FOLDER = f"{parsed_args.base}/{parsed_args.algo}_{parsed_args.ds}_{parsed_args.attacker}_vs_{versus}{suffix}"
    print(f"Base Folder: {BASE_FOLDER}")

    datas = []

    for i, row in safety_df.iterrows():
        for k in range(5):
            suff = "" if k == 0 else f"_i{k}"
            output_folder_name = f"{BASE_FOLDER}/{algo}_bss_{row['id']}{suff}"

            info = output_folder_name + "/info.txt"

            if os.path.exists(info):
                with open(info, "rt") as f:
                    data = f.readline()
                    data = data.replace("'", '"')
                    try:
                        data = ast.literal_eval(data)
                    except:
                        datas.append(
                            {"jailbroken": False, "exists": False, "data": None, **row.to_dict()}
                        )
                        continue
                    if len(list(data.keys())) == 0:
                        datas.append(
                            {"jailbroken": False, "exists": True, "data": data, **row.to_dict()}
                        )
                    else:
                        first_key = sorted(list(data.keys()))[0]
                        dt = {first_key: data[first_key]}
                        datas.append(
                            {"jailbroken": True, "exists": True, "data": dt, **row.to_dict()}
                        )
            else:
                datas.append({"jailbroken": False, "exists": False, "data": None, **row.to_dict()})
                # print(f"no file '{info}'")

    df = pd.DataFrame(datas)
    return df


def plot_grouped(g, c, versus):
    df = get_df(versus)
    pass_rate = df.groupby(g)[c].mean() * 100

    pass_rate.sort_values(ascending=False).plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"{algo.capitalize()} Jailbreak Rate (%) by Category")
    plt.ylabel("Success Rate (%)")

    plt.xlabel("Category")
    plt.xticks(rotation=30, fontsize=5, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{BASE_GRAPHS_FOLDER}{algo}_{ds}_{versus}_{parsed_args.attacker}.png")
    print(f"Saved to '{BASE_GRAPHS_FOLDER}{algo}_{ds}_{versus}_{parsed_args.attacker}.png'")
    # plt.show()


import pandas as pd
from collections import defaultdict


def plot_grouped_2(g, c, data_col, versus):
    df = get_df(versus)
    expanded = df[data_col].apply(pd.Series)
    expanded = expanded.reindex(sorted(expanded.columns), axis=1)
    expanded = expanded.fillna(False).astype(int)
    expanded[g] = df[g]
    grouped = expanded.groupby(g).mean() * 100
    pass_rate = df.groupby(g)[c].mean() * 100
    order = pass_rate.sort_values(ascending=False).index
    grouped = grouped.loc[order]
    ax = grouped.plot(kind="bar", stacked=True, edgecolor="black", colormap="Blues")

    # with open(
    #     f"{BASE_GRAPHS_FOLDER}text/{algo}_{ds}_{parsed_args.version}_{versus}.txt", "w+"
    # ) as f:
    #     grouped.to_csv(f, sep="\t", float_format="%.2f")

    plt.title(f"{algo.capitalize()} Jailbreak Rate vs {all_versus[versus.lower()]} (%) by Category")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Category")
    plt.xticks(rotation=30, fontsize=5, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(
        f"{BASE_GRAPHS_FOLDER}{algo}_{ds}_{parsed_args.version}_{versus}_{parsed_args.attacker}.png"
    )
    print(
        f"Saved to '{BASE_GRAPHS_FOLDER}{algo}_{ds}_{parsed_args.version}_{versus}_{parsed_args.attacker}.png'"
    )


import pandas as pd


def print_simplesafety_latex_tables(df, algo_name, target_name, f, tshort=None):
    """
    Prints LaTeX tables for SimpleSafety results.
    df: The final DataFrame containing 'harm_area', 'category', and 'jailbroken'
    algo_name: e.g., 'Goat'
    target_name: e.g., 'CircuitBreaker'
    """

    if tshort is None:
        tshort = target_name
    # 1. PER-CATEGORY DETAILED TABLE
    # Grouping by both to match your 'plot_grouped_2' logic
    stats = (
        df.groupby(["harm_area", "category"])["jailbroken"]
        .agg(Total="size", Jailbroken=lambda x: (x == True).sum())
        .reset_index()
    )

    stats["JB_Rate"] = (stats["Jailbroken"] / stats["Total"] * 100).round(1)
    stats = stats.sort_values("JB_Rate", ascending=False)

    print(f"\n% --- {algo_name} vs {tshort} (SimpleSafety) ---", file=f)
    print(r"\begin{table}[t]", file=f)
    print(f"\\caption{{Jailbreak Rate by Harm Area and Category: {algo_name} vs {tshort}}}", file=f)
    print(f"\\label{{tab:ss_{algo_name.lower()}_{target_name.lower()}}}", file=f)
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{llc}", file=f)
    print(r"\bf HARM AREA & \bf CATEGORY & \bf SUCCESS RATE (\%) \\", file=f)
    print(r"\hline", file=f)

    for _, row in stats.iterrows():
        area = row["harm_area"].replace("_", " ").title()
        cat = row["category"].replace("_", " ").title()
        print(f"{area:<15} & {cat:<20} & {row['JB_Rate']}\\% \\\\", file=f)

    print(r"\hline", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)
    print(r"\end{table}", file=f)


def print_simplesafety_subtable_format(all_dfs_dict, algo_name, f=None):
    """
    Generates the 'Comprehensive' subtable format where columns are different target models.
    all_dfs_dict: { "TargetName": dataframe_from_that_run }
    """

    targets = list(all_dfs_dict.keys())

    # Get unique categories across all results
    first_df = list(all_dfs_dict.values())[0]
    categories = sorted(first_df["category"].unique())
    categories = list(set(categories))
    print(f"\n% --- Comprehensive Subtable: {algo_name} ---", file=f)
    print(r"\begin{subtable}{1.0\textwidth}", file=f)
    print(r"\centering", file=f)
    print(f"\\caption{{{algo_name} Attack on SimpleSafety}}", file=f)

    # Column definition: Category + 1 for each target
    col_def = "l" + "c" * len(targets)
    print(rf"\begin{{tabular}}{{{col_def}}}", file=f)

    # Header
    header = [r"\bf CATEGORY"] + [rf"\bf {all_versus[t]}" for t in targets]
    print(" & ".join(header) + r" \\", file=f)
    print(r"\hline \\", file=f)

    # Rows
    for cat in categories:
        print(f"cat={cat}\n")
        row_cells = [f"{cat.replace('_', ' ').title():<20}"]
        for t in targets:
            try:
                sub_df = all_dfs_dict[t]
                cat_data = sub_df[sub_df["category"] == cat]["jailbroken"]

                if cat_data.empty:  # or (len(cat_data) < 30 and algo != "crescendo"):
                    val_str = "--"
                else:
                    val = cat_data.mean() * 100
                    val_str = f"{val:.1f}\\%"
                    if val < 15:
                        val_str = f"\\textbf{{{val_str}}}"

                row_cells.append(val_str)
            except Exception as e:
                row_cells.append("-")
        print(" & ".join(row_cells) + r" \\", file=f)

    # Average Row
    print(r"\\ \hline \\", file=f)
    avg_row = [r"\bf Average"]
    for t in targets:
        try:
            avg_val = all_dfs_dict[t]["jailbroken"].mean() * 100
            avg_row.append(f"\\bf {avg_val:.1f}\\%")
        except Exception as e:
            pass

    print(" & ".join(avg_row) + r" \\", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\end{subtable}", file=f)


# --- EXECUTION ---
# To print the standard table for the current run:
import traceback

dd = {}
with open(
    f"{BASE_GRAPHS_FOLDER}text/_latex_tabs_{algo}_{ds}_{parsed_args.version}.txt",
    "w+",
) as f:

    for vs in all_versus.keys():
        try:
            df = get_df(vs)
            if len(df.columns) > 2:
                dd[vs] = df
            # print_simplesafety_latex_tables(df, algo.capitalize(), vs.capitalize(), f)
            plot_grouped_2(
                g=["SemanticCategory", "FunctionalCategory"],
                c="jailbroken",
                data_col="data",
                versus=vs,
            )
        except Exception as e:
            # traceback.print_exc(e)
            print(f"Couldnt do {vs} : {str(e)}")


def ssh(a, l=10):
    if a.lower() == "and":
        return "\&"
    if len(a) > l:
        return a[:l] + "."
    else:
        return a


def shorten(x):
    x = " ".join([ssh(a) for a in x.split(" ")])

    return x


def plc(d):
    ddd = d.copy()
    ddd["category"] = "cat"
    return ddd


dd = {x: plc(y) for x, y in dd.items()}
dcol = {x: y["category"].unique() for x, y in dd.items()}

dcov = {x: len(y["exists"][y["exists"] == True]) for x, y in dd.items()}
dj = {x: len(y["jailbroken"][y["jailbroken"] == True]) for x, y in dd.items()}
import json

with open(
    f"{BASE_GRAPHS_FOLDER}text/_coverage_{algo}_{ds}_{parsed_args.version}.json",
    "w+",
) as f:
    dddd = {"coverage": dcov, "jailbroken": dj}
    json.dump(dddd, fp=f, indent=4)
    # pprint(f"{dcov}", file=f)
with open(
    f"{BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt",
    "w+",
) as f:

    print_simplesafety_subtable_format(dd, algo, f)


print(f"Saved in : {BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt")
