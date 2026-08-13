from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
import os
import json
import ast
import pandas as pd
import argparse
import sys

safety_df = BertievidgenSimpleSafetyTests().get_first(n=80)
parser = argparse.ArgumentParser(description="Dataset path generator")

parser.add_argument("algo", nargs="?", default="goat", help="Algorithm name (default: goat)")
parser.add_argument("ds", nargs="?", default="bss", help="Dataset name (default: bss)")
parser.add_argument("version", nargs="?", default="", help="Version suffix (optional)")
parser.add_argument("--attacker", "-a", default="gemini", help="Attacker name (default: gemini)")
parser.add_argument("--versus", default="circuitbreaker", help="Comparison target")
parser.add_argument("--base", "-b", default="datasets", help="Base folder for datasets")
parsed_args = parser.parse_args()
suffix = f"__{parsed_args.version}" if parsed_args.version else ""
BASE_FOLDER = f"{parsed_args.base}/{parsed_args.algo}_{parsed_args.ds}_{parsed_args.attacker}_vs_{parsed_args.versus}{suffix}"

print(f"Base Folder: {BASE_FOLDER}")

algo = parsed_args.algo
ds = parsed_args.ds

datas = []

for i, row in safety_df.iterrows():
    output_folder_name = f"{BASE_FOLDER}/{algo}_{ds}_{row['id']}"
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
        print(f"no file '{info}'")

df = pd.DataFrame(datas)
print(df[-10:])

import matplotlib.pyplot as plt


def plot_grouped(g, c):
    pass_rate = df.groupby(g)[c].mean() * 100

    pass_rate.sort_values(ascending=False).plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"{algo.capitalize()} Jailbreak Rate (%) by Category")
    plt.ylabel("Success Rate (%)")

    plt.xlabel("Category")
    plt.xticks(rotation=30, fontsize=5, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"graphs/{algo}_{ds}_{parsed_args.versus}_{parsed_args.attacker}.png")
    print(f"Saved to 'graphs/{algo}_{ds}_{parsed_args.versus}_{parsed_args.attacker}.png'")
    # plt.show()


import pandas as pd
from collections import defaultdict

all_versus = {
    "cbllama": "CBL",
    "dertains": "DER",
    "door": "DOR",
    "guardresp": "GUQ",
    "guardllama": "GUA",
    "llamainstruct": "LLA",
}


def plot_grouped_2(g, c, data_col):
    expanded = df[data_col].apply(pd.Series)
    expanded = expanded.reindex(sorted(expanded.columns), axis=1)
    expanded = expanded.fillna(False).astype(int)
    expanded[g] = df[g]
    grouped = expanded.groupby(g).mean() * 100
    pass_rate = df.groupby(g)[c].mean() * 100
    order = pass_rate.sort_values(ascending=False).index
    grouped = grouped.loc[order]
    ax = grouped.plot(kind="bar", stacked=True, edgecolor="black", colormap="Blues")

    with open(f"graphs/text/{algo}_{ds}_{parsed_args.version}_{parsed_args.versus}.txt", "w+") as f:
        grouped.to_csv(f, sep="\t", float_format="%.2f")

    plt.title(
        f"{algo.capitalize()} Jailbreak Rate vs {all_versus[parsed_args.versus.lower()]} (%) by Category"
    )
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Category")
    plt.xticks(rotation=30, fontsize=5, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(
        f"graphs/{algo}_{ds}_{parsed_args.version}_{parsed_args.versus}_{parsed_args.attacker}.png"
    )
    print(
        f"Saved to 'graphs/{algo}_{ds}_{parsed_args.version}_{parsed_args.versus}_{parsed_args.attacker}.png'"
    )


import pandas as pd


def print_simplesafety_latex_tables(df, algo_name, target_name, tshort=None):
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

    print(f"\n% --- {algo_name} vs {tshort} (SimpleSafety) ---")
    print(r"\begin{table}[t]")
    print(f"\\caption{{Jailbreak Rate by Harm Area and Category: {algo_name} vs {tshort}}}")
    print(f"\\label{{tab:ss_{algo_name.lower()}_{target_name.lower()}}}")
    print(r"\begin{center}")
    print(r"\begin{tabular}{llc}")
    print(r"\bf HARM AREA & \bf CATEGORY & \bf SUCCESS RATE (\%) \\")
    print(r"\hline")

    for _, row in stats.iterrows():
        area = row["harm_area"].replace("_", " ").title()
        cat = row["category"].replace("_", " ").title()
        print(f"{area:<15} & {cat:<20} & {row['JB_Rate']}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{center}")
    print(r"\end{table}")


def print_simplesafety_subtable_format(all_dfs_dict, algo_name):
    """
    Generates the 'Comprehensive' subtable format where columns are different target models.
    all_dfs_dict: { "TargetName": dataframe_from_that_run }
    """
    targets = list(all_dfs_dict.keys())

    # Get unique categories across all results
    first_df = list(all_dfs_dict.values())[0]
    categories = sorted(first_df["category"].unique())

    print(f"\n% --- Comprehensive Subtable: {algo_name} ---")
    print(r"\begin{subtable}{1.0\textwidth}")
    print(r"\centering")
    print(f"\\caption{{{algo_name} Attack on SimpleSafety}}")

    # Column definition: Category + 1 for each target
    col_def = "l" + "c" * len(targets)
    print(rf"\begin{{tabular}}{{{col_def}}}")

    # Header
    header = [r"\bf CATEGORY"] + [rf"\bf {t[:3].upper()}" for t in targets]
    print(" & ".join(header) + r" \\")
    print(r"\hline \\")

    # Rows
    for cat in categories:
        row_cells = [f"{cat.replace('_', ' ').title():<20}"]
        for t in targets:
            sub_df = all_dfs_dict[t]
            cat_data = sub_df[sub_df["category"] == cat]["jailbroken"]

            if cat_data.empty:
                val_str = "--"
            else:
                val = cat_data.mean() * 100
                val_str = f"{val:.1f}\\%"
                if val > 50:
                    val_str = f"\\textbf{{{val_str}}}"

            row_cells.append(val_str)
        print(" & ".join(row_cells) + r" \\")

    # Average Row
    print(r"\\ \hline \\")
    avg_row = [r"\bf Average"]
    for t in targets:
        avg_val = all_dfs_dict[t]["jailbroken"].mean() * 100
        avg_row.append(f"\\bf {avg_val:.1f}\\%")

    print(" & ".join(avg_row) + r" \\")
    print(r"\end{tabular}")
    print(r"\end{subtable}")


# --- EXECUTION ---
# To print the standard table for the current run:

print_simplesafety_latex_tables(df, algo.capitalize(), parsed_args.versus.capitalize())
plot_grouped_2(g=["harm_area", "category"], c="jailbroken", data_col="data")

# Note: To use print_simplesafety_subtable_format, you would need to loop
# through your different target models and store the dataframes in a dictionary.
