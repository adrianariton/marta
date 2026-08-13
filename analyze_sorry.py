from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
from core.datasets.safety.harmbench import HarmBench
from core.datasets.safety.advbench import AdvBench
from core.datasets.safety.sorrybench import SorryBench

import os
import json
import ast
import pandas as pd
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
safety_df = SorryBench().get_first(n=500)
parser = argparse.ArgumentParser(description="Dataset path generator")

parser.add_argument("algo", nargs="?", default="goat", help="Algorithm name (default: goat)")
parser.add_argument("ds", nargs="?", default="bss", help="Dataset name (default: bss)")
parser.add_argument("version", nargs="?", default="", help="Version suffix (optional)")
parser.add_argument("--attacker", "-a", default="gemini", help="Attacker name (default: gemini)")
parser.add_argument("--base", "-b", default="datasets", help="Base folder for datasets")
parsed_args = parser.parse_args()
suffix = f"__{parsed_args.version}" if parsed_args.version else ""

BASE_GRAPHS_FOLDER = f"graphs/sorry/"

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}")

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text")
BASE_GRAPHS_FOLDER = BASE_GRAPHS_FOLDER + parsed_args.algo + "/"
all_versus = {
    "cbllama": "CBL",
    # "cbadicb3000001": "2-MCB",
    "cbadimixed2000": "MT-CB",
    # "cbadicbmixed2t001": "MT-CB-3k",
}

algo = parsed_args.algo
ds = parsed_args.ds


import matplotlib.pyplot as plt

k_range = [1, 2] # use 0 for 2-MCB

def get_df(versus):
    BASE_FOLDER = f"{parsed_args.base}/{parsed_args.algo}_{parsed_args.ds}_{parsed_args.attacker}_vs_{versus}{suffix}"
    print(f"Base Folder: {BASE_FOLDER}")

    datas = []

    for i, row in safety_df.iterrows():
        for k in k_range:
            suff = "" if k == 0 else f"_i{k}"
            output_folder_name = f"{BASE_FOLDER}/{algo}_sorry_{row['id']}{suff}"

            info = output_folder_name + "/info.txt"

            if os.path.exists(info):
                with open(info, "rt") as f:
                    data = f.readline()
                    data = data.replace("'", '"')
                    try:
                        data = ast.literal_eval(data)
                    except:
                        datas.append(
                            {"jailbroken": False, "exists": False, "data": None, "k": k, **row.to_dict()}
                        )
                        continue
                    if len(list(data.keys())) == 0:
                        datas.append(
                            {"jailbroken": False, "exists": True, "data": data, "k": k, **row.to_dict()}
                        )
                    else:
                        first_key = sorted(list(data.keys()))[0]
                        dt = {first_key: data[first_key]}
                        keys = [1,2]
                        is_jailbroken = any([data.get(fk, False) == True for fk in keys])
                        datas.append(
                            {"jailbroken": is_jailbroken, "exists": True, "data": dt, "k": k, **row.to_dict()}
                        )
            else:
                datas.append({"jailbroken": False, "exists": False, "data": None, "k": k, **row.to_dict()})
                # print(f"no file '{info}'")

    df = pd.DataFrame(datas)
    return df


def plot_cbl_superiority_total(
    cbl_titl="CBL",
    cbl_versus="cbllama",
    other_versus_list=None,
    algo=None,
    ds=None,
    parsed_args=None,
    k_s=range(15),
    alpha=0.05,
    save_path=None,
):
    """
    For each target model, compute E_k[ASR] and D²_k over k=0..k_s-1
    (ASR(k) = fraction of ALL prompts jailbroken at attempt k).
    Plots a dot + CI chart: x = target model, y = ASR.
    CBL is shown as a horizontal band (mean ± std) for reference.
    Stars mark models where CBL is significantly better (non-overlapping CIs).
    """
    if other_versus_list is None:
        other_versus_list = [v for v in all_versus if v != cbl_versus]
    df["jailbroken"] = df["jailbroken"] & (df["exists"] == True)
    df["exists"] = True  # for ASR calculation, all rows are considered (non-existing = ASR=0)
    def get_asr_series(versus):
        df = get_df(versus)
        df_ex = df[df["exists"] == True].copy()
        asr_by_k = []
        for k in k_s:
            sub = df_ex[df_ex["k"] == k]
            print(len(sub), "rows for", versus, "k=", k)
            print(sub["jailbroken"].mean() * 100 if len(sub) > 0 else "N/A", "% ASR")
            print(sub["jailbroken"].mean() if len(sub) > 0 else "N/A", "% ASR'")
            if len(sub) > 0:
                asr_by_k.append(sub["jailbroken"].sum() / 500 * 100)
        return np.array(asr_by_k)

    # CBL baseline
    cbl_series = get_asr_series(cbl_versus)
    print(f"{cbl_titl} ASR by k:", cbl_series)
    cbl_mean = cbl_series.mean()
    cbl_std  = cbl_series.std(ddof=1)
    cbl_se   = cbl_std / np.sqrt(len(cbl_series))
    z = stats.norm.ppf(1 - alpha / 2)

    # Others
    rows = []
    for vs in other_versus_list:
        try:
            s = get_asr_series(vs)
            if len(s) == 0:
                continue
            mean_asr = s.mean()
            std_asr  = s.std(ddof=1)
            se       = std_asr / np.sqrt(len(s))
            # Welch t-test: CBL vs this target
            t_stat, p_val = stats.ttest_ind_from_stats(
                mean1=cbl_mean, std1=cbl_std, nobs1=len(cbl_series),
                mean2=mean_asr,  std2=std_asr,  nobs2=len(s),
                equal_var=False,
            )
            rows.append({
                "target":   all_versus.get(vs.lower(), vs),
                "mean":     mean_asr,
                "std":      std_asr,
                "se":       se,
                "ci_lo":    mean_asr - z * se,
                "ci_hi":    mean_asr + z * se,
                "p_val":    p_val,
                "sig":      (p_val < alpha) and (mean_asr > cbl_mean),  # other worse = CBL superior
            })
        except Exception as e:
            print(f"Skipping {vs}: {e}")

    plot_df = pd.DataFrame(rows).sort_values("mean", ascending=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.1), 6))

    colors = ["#c0392b" if r["sig"] else "#7f8c8d" for _, r in plot_df.iterrows()]

    # bars
    ax.bar(
        plot_df["target"], plot_df["mean"],
        yerr=plot_df["std"],
        color=colors, alpha=0.75, capsize=5,
        error_kw={"linewidth": 1.3, "ecolor": "#2c3e50"},
        width=0.55,
    )

    # MCB reference band: mean ± std
    ax.axhline(cbl_mean, color="#2980b9", linewidth=2.0, linestyle="--", label=f"{cbl_titl}  μ={cbl_mean:.1f}%")
    ax.axhspan(cbl_mean - cbl_std, cbl_mean + cbl_std,
               color="#2980b9", alpha=0.12, label=f"{cbl_titl} ±σ ({cbl_mean - cbl_std:.1f}–{cbl_mean + cbl_std:.1f}%)")

    # significance stars + p-value annotations
    for i, (_, row) in enumerate(plot_df.iterrows()):
        sig_str = f"p={row['p_val']:.3f}" + (" ★" if row["sig"] else "")
        ax.text(i, row["mean"] + row["std"] + 1.5, sig_str,
                ha="center", va="bottom", fontsize=8,
                color="#c0392b" if row["sig"] else "#7f8c8d")

    ax.set_ylabel(r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $\sigma_k$  (%)", fontsize=11)
    ax.set_xlabel("Target model", fontsize=11)
    ax.set_ylim(0, 25)
    ax.set_title(
        f"{(algo or 'algo').capitalize()} — total ASR: {cbl_titl} superiority\n"
        r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $D_k$,  "
        f"Welch t-test  α={alpha}  |  ★ = {cbl_titl} significantly better",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    sns.despine()
    plt.tight_layout()

    if save_path is None:
        v = f"_{parsed_args.version}" if parsed_args.version else ""
        save_path = f"{BASE_GRAPHS_FOLDER}{algo}_{ds}{v}_{parsed_args.attacker}_cbl_superiority_total.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to '{save_path}'")
    plt.close()
    
    print("Results:")
    for _, r in plot_df.iterrows():
        print(f"{r['target']}: {r['mean']:.1f}% ± {r['std']:.1f}%, p={r['p_val']:.3f}, {f'{cbl_titl} superior' if r['sig'] else 'no sig'}")

    # and MT-CB data for reference
    if cbl_versus in all_versus:
        print(f"\n{cbl_titl} ({all_versus[cbl_versus.lower()]}) ASR: {cbl_mean:.1f}% ± {cbl_std:.1f}% (n={len(cbl_series)})")
        for i, k in enumerate(k_s):
            print(f"  k={k}: {cbl_series[i]:.1f}%")

    # improvement mean MT-CB vs CBL
    if "cbadimixed2000" in all_versus and "cbllama" in all_versus:
        cbl2_series = get_asr_series("cbllama")
        cbl2_mean = cbl2_series.mean()
        improvement = cbl_mean - cbl2_mean
        print(f"\nImprovement of {cbl_titl} over CBL: {improvement:.1f}% (MT-CB mean: {cbl2_mean:.1f}%)")
        percentage_improvement = (improvement / cbl2_mean * 100) if cbl2_mean > 0 else float('inf')
        print(f"Percentage improvement over CBL: {percentage_improvement:.1f}%")
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
    ddd["category"] = ddd["category"].map(lambda x: str(x))
    return ddd


dd = {x: plc(y) for x, y in dd.items()}
dcol = {x: y["category"].unique() for x, y in dd.items()}

dcov = {x: len(y["exists"][y["exists"] == True]) for x, y in dd.items()}
dj = {x: len(y["jailbroken"][y["jailbroken"] == True]) for x, y in dd.items()}

djs = []
for k in k_range:
    dj_k = {x: len(y["jailbroken"][(y["jailbroken"] == True) & (y["k"] == k)]) for x, y in dd.items()}
    djs.append(dj_k)

common_exists_indexes = None

for y in dd.values():
    # Get the indexes where 'exists' is True for the current DataFrame/Series
    true_indexes = set(y[y["exists"] == True].index)

    if common_exists_indexes is None:
        common_exists_indexes = true_indexes
    else:
        # Intersect with the previous keys' True indexes
        common_exists_indexes.intersection_update(true_indexes)

# 2. Filter each dataset by this intersection and count the jailbroken rows
dj_intersection = {
    x: len(
        y.loc[list(common_exists_indexes)][y.loc[list(common_exists_indexes)]["jailbroken"] == True]
    )
    for x, y in dd.items()
}
import json

with open(
    f"{BASE_GRAPHS_FOLDER}text/_coverage_{algo}_{ds}_{parsed_args.version}.json",
    "w+",
) as f:
    dddd = {
        "coverage": dcov,
        "jailbroken": dj,
        "common_jailbroken": dj_intersection,
        "common_total": len(common_exists_indexes),
        "jailbroken_by_k": djs,
    }
    json.dump(dddd, fp=f, indent=4)
    # pprint(f"{dcov}", file=f)
with open(
    f"{BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt",
    "w+",
) as f:

    print_simplesafety_subtable_format(dd, algo, f)


print(f"Saved in : {BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt")

def plot_cbl_superiority_common(
    clb_titl="CBL",
    cbl_versus="cbllama",
    other_versus_list=None,
    algo=None,
    ds=None,
    parsed_args=None,
    k_s=range(15),
    alpha=0.05,
    save_path=None,
):
    if other_versus_list is None:
        other_versus_list = [v for v in all_versus if v != cbl_versus]

    all_versus_keys = [cbl_versus] + other_versus_list
    dfs = {vs: get_df(vs) for vs in all_versus_keys}

    # Build common intersection of indexes where exists==True across ALL models
    common_idx = None
    for vs, df in dfs.items():
        true_idx = set(df[df["exists"] == True].index)
        common_idx = true_idx if common_idx is None else common_idx & true_idx

    print(f"Common existing rows: {len(common_idx)}")

    def get_asr_series_common(versus):
        df = dfs[versus].loc[list(common_idx)]
        asr_by_k = []
        for k in k_s:
            sub = df[df["k"] == k]
            if len(sub) > 0:
                asr_by_k.append(sub["jailbroken"].mean() * 100)
        return np.array(asr_by_k)

    cbl_series = get_asr_series_common(cbl_versus)
    cbl_mean = cbl_series.mean()
    cbl_std  = cbl_series.std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for vs in other_versus_list:
        try:
            s = get_asr_series_common(vs)
            if len(s) == 0:
                continue
            mean_asr = s.mean()
            std_asr  = s.std(ddof=1)
            se       = std_asr / np.sqrt(len(s))
            t_stat, p_val = stats.ttest_ind_from_stats(
                mean1=cbl_mean, std1=cbl_std, nobs1=len(cbl_series),
                mean2=mean_asr,  std2=std_asr,  nobs2=len(s),
                equal_var=False,
            )
            rows.append({
                "target": all_versus.get(vs.lower(), vs),
                "mean":   mean_asr,
                "std":    std_asr,
                "se":     se,
                "ci_lo":  mean_asr - z * se,
                "ci_hi":  mean_asr + z * se,
                "p_val":  p_val,
                "sig":    (p_val < alpha) and (mean_asr > cbl_mean),
            })
        except Exception as e:
            print(f"Skipping {vs}: {e}")

    plot_df = pd.DataFrame(rows).sort_values("mean", ascending=False)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.1), 6))

    colors = ["#c0392b" if r["sig"] else "#7f8c8d" for _, r in plot_df.iterrows()]
    ax.bar(
        plot_df["target"], plot_df["mean"],
        yerr=plot_df["std"],
        color=colors, alpha=0.75, capsize=5,
        error_kw={"linewidth": 1.3, "ecolor": "#2c3e50"},
        width=0.55,
    )

    ax.axhline(cbl_mean, color="#2980b9", linewidth=2.0, linestyle="--", label=f"{clb_titl}  μ={cbl_mean:.1f}%")
    ax.axhspan(cbl_mean - cbl_std, cbl_mean + cbl_std,
               color="#2980b9", alpha=0.12, label=f"{clb_titl} ±σ ({cbl_mean - cbl_std:.1f}–{cbl_mean + cbl_std:.1f}%)")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        sig_str = f"p={row['p_val']:.3f}" + (" ★" if row["sig"] else "")
        ax.text(i, row["mean"] + row["std"] + 1.5, sig_str,
                ha="center", va="bottom", fontsize=8,
                color="#c0392b" if row["sig"] else "#7f8c8d")

    ax.set_ylabel(r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $\sigma_k$  (%)", fontsize=11)
    ax.set_xlabel("Target model", fontsize=11)
    ax.set_ylim(0, 25)
    ax.set_title(
        f"{(algo or 'algo').capitalize()} — common ASR: {clb_titl} superiority  (n={len(common_idx)} common)\n"
        r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $D_k$,  "
        f"Welch t-test  α={alpha}  |  ★ = {clb_titl} significantly better",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    sns.despine()
    plt.tight_layout()

    if save_path is None:
        v = f"_{parsed_args.version}" if parsed_args.version else ""
        save_path = f"{BASE_GRAPHS_FOLDER}{algo}_{ds}{v}_{parsed_args.attacker}_cbl_superiority_common.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to '{save_path}'")
    plt.close()
    
    
    print("Results:")
    for _, r in plot_df.iterrows():
        print(f"{r['target']}: {r['mean']:.1f}% ± {r['std']:.1f}%, p={r['p_val']:.3f}, {f'{clb_titl} superior' if r['sig'] else 'no sig'}")



plot_cbl_superiority_common(
    clb_titl="MT-CB",  # "CBL",
    cbl_versus="cbadimixed2000",
    other_versus_list=[k for k in all_versus if k != "cbadimixed2000"],
    algo=algo, ds=ds,
    parsed_args=parsed_args,
    alpha=0.05,
    k_s=k_range
)
plot_cbl_superiority_total(
    cbl_titl="MT-CB",  # "CBL",
    cbl_versus="cbadimixed2000",
    other_versus_list=[k for k in all_versus if k != "cbadimixed2000"],
    algo=algo, ds=ds,
    parsed_args=parsed_args,
    alpha=0.05,
    k_s=k_range
)