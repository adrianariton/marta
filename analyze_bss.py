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
parser.add_argument("--base", "-b", default="datasets", help="Base folder for datasets")
parsed_args = parser.parse_args()
suffix = f"__{parsed_args.version}" if parsed_args.version else ""

BASE_GRAPHS_FOLDER = f"graphs/inpap/"

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}")

if not os.path.isdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text"):
    os.mkdir(f"{BASE_GRAPHS_FOLDER}{parsed_args.algo}/text")
BASE_GRAPHS_FOLDER = BASE_GRAPHS_FOLDER + parsed_args.algo + "/"
all_versus = {
    "llamainstruct": "LLA",
    "dertains": "DER",
    "door": "DOR",
    # "guardresp": "GUQ",
    # "guardllama": "GUA",
    "guardr": "GUR",
    "guardq": "GUQ",
    "guardrnoh": "GUR",
    "guardqnoh": "GUQ",
    "nemor": "NEMR",
    "nemoq": "NEMQ",
    "cbllama": "CBL",
}

algo = parsed_args.algo
ds = parsed_args.ds


import matplotlib.pyplot as plt

k_s = 15
if parsed_args.algo.lower() == "crescendo":
    k_s = 5


def get_df(versus):
    BASE_FOLDER = f"{parsed_args.base}/{parsed_args.algo}_{parsed_args.ds}_{parsed_args.attacker}_vs_{versus}{suffix}"
    print(f"Base Folder: {BASE_FOLDER}")

    datas = []

    for i, row in safety_df.iterrows():
        for k in range(k_s):
            suff = "" if k == 0 else f"_i{k}"
            output_folder_name = f"{BASE_FOLDER}/{algo}_{ds}_{row['id']}{suff}"
            info = output_folder_name + "/info.txt"

            if os.path.exists(info):
                with open(info, "rt") as f:
                    data = f.readline()
                    data = data.replace("'", '"')
                    try:
                        data = ast.literal_eval(data)
                    except (ValueError, SyntaxError):
                        print(f"Error occurred while evaluating data in file: {info}")
                        datas.append({"jailbroken": False, "exists": False, "data": None, "k": k, **row.to_dict()})
                        continue

                    if len(list(data.keys())) == 0:
                        datas.append(
                            {"jailbroken": False, "exists": True, "data": data, "k": k, **row.to_dict()}
                        )
                    else:
                        first_key = sorted(list(data.keys()))[0]
                        dt = {first_key: data[first_key]}
                        datas.append(
                            {"jailbroken": True, "exists": True, "data": dt, "k": k, **row.to_dict()}
                        )
            else:
                datas.append({"jailbroken": False, "exists": False, "data": None, "k": k, **row.to_dict()})
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

    with open(
        f"{BASE_GRAPHS_FOLDER}text/{algo}_{ds}_{parsed_args.version}_{versus}.txt", "w+"
    ) as f:
        grouped.to_csv(f, sep="\t", float_format="%.2f")

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
            print_simplesafety_latex_tables(df, algo.capitalize(), vs.capitalize(), f)
            plot_grouped_2(g=["harm_area", "category"], c="jailbroken", data_col="data", versus=vs)
        except Exception as e:
            print(f"Couldnt do {vs}")


def ssh(a):
    if a.lower() == "and":
        return "\&"
    if len(a) > 5:
        return a[:5] + "."
    else:
        return a


def shorten(x):
    x = " ".join([ssh(a) for a in x.split(" ")])

    return x


def plc(d):
    ddd = d.copy()
    ddd["category"] = (
        d["harm_area"].map(lambda x: shorten(x)) + ",\\\\" + d["category"].map(lambda x: shorten(x))
    )
    return ddd


# dd = {x: plc(y) for x, y in dd.items()}
# dcol = {x: y["category"].unique() for x, y in dd.items()}

# dcov = {x: len(y["exists"][y["exists"] == True]) for x, y in dd.items()}
# dj = {x: len(y["jailbroken"][y["jailbroken"] == True]) for x, y in dd.items()}
# import json

# with open(
#     f"{BASE_GRAPHS_FOLDER}text/_coverage_{algo}_{ds}_{parsed_args.version}.json",
#     "w+",
# ) as f:
#     dddd = {"coverage": dcov, "jailbroken": dj}
#     json.dump(dddd, fp=f, indent=4)
#     # pprint(f"{dcov}", file=f)
# with open(
#     f"{BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt",
#     "w+",
# ) as f:

#     print_simplesafety_subtable_format(dd, algo, f)
# Note: To use print_simplesafety_subtable_format, you would need to loop
# through your different target models and store the dataframes in a dictionary.
print(f"Saved in : {BASE_GRAPHS_FOLDER}text/_latex_subtabs_{algo}_{ds}_{parsed_args.version}.txt")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_asr_mean_variance(versus, algo, ds, parsed_args, k_s=15, save_path=None):
    """
    For each category, compute:
      - E_k[ASR] = mean over k=0..k_s-1 of ASR(k)   (where ASR(k) = mean jailbroken at attempt k)
      - D²_k     = variance over k of ASR(k)
    Plots a seaborn barplot (mean ± std) sorted by E_k[ASR] descending.
    """
    df = get_df(versus)
    df["jailbroken"] = df["jailbroken"] & (df["exists"] == True)# treat non-existing as non-jailbroken
    df["exists"] = True  # for ASR calculation, all rows are considered (non-existing = ASR=0)
    
    df_ex = df[df["exists"] == True].copy()
    
    if df_ex.empty:
        print(f"No valid data for {versus}. Cannot plot ASR mean/variance.")
        return
    else:
        print(f"Data for '{versus}': {len(df_ex)} rows, cols={df_ex.columns.tolist()}")
    # --- build (category, k) -> ASR table ---
    records = []
    for cat in df_ex["harm_area"].unique():
        cat_df = df_ex[df_ex["harm_area"] == cat]
        for k in range(k_s):
            sub = cat_df[cat_df["k"] == k]
            if len(sub) == 0:
                continue
            asr_k = sub["jailbroken"].mean() * 100          # ASR at this k
            records.append({"category": cat, "k": k, "asr": asr_k})

    asr_df = pd.DataFrame(records)

    # E_k[ASR] and D²_k per category
    stats = (
        asr_df.groupby("category")["asr"]
        .agg(mean_asr="mean", var_asr="var")
        .reset_index()
        .sort_values("mean_asr", ascending=False)
    )
    stats["std_asr"] = np.sqrt(stats["var_asr"])

    target_short = all_versus.get(versus.lower(), versus)

    fig, ax = plt.subplots(figsize=(max(10, len(stats) * 0.9), 6))
    sns.set_theme(style="whitegrid", font_scale=0.95)

    sns.barplot(
        data=asr_df,
        x="category",
        y="asr",
        order=stats["category"],       # sorted by mean desc
        estimator=np.mean,
        errorbar=("sd", 1),            # ±1 std = sqrt(D²_k)
        capsize=0.12,
        err_kws={"linewidth": 1.4},
        palette="mako_r",
        ax=ax,
    )

    # annotate each bar with mean ± std
    for patch, (_, row) in zip(ax.patches, stats.iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + row["std_asr"] + 1.5,
            f"{row['mean_asr']:.1f}\n±{row['std_asr']:.1f}",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
        )

    ax.set_title(
        f"{algo.capitalize()} vs {target_short}\n"
        r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $\sqrt{D^2_k}$ by category",
        fontsize=12,
    )
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel(r"ASR (%)  —  $\mathbb{E}_k \pm \sigma_k$", fontsize=10)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    sns.despine()
    plt.tight_layout()

    if save_path is None:
        v = f"_{parsed_args.version}" if parsed_args.version else ""
        save_path = f"{BASE_GRAPHS_FOLDER}{algo}_{ds}{v}_{versus}_{parsed_args.attacker}_asr_Ek_D2k.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to '{save_path}'")
    plt.close()

def plot_cbl_superiority_total(
    cbl_versus="cbllama",
    other_versus_list=None,
    algo=None,
    ds=None,
    parsed_args=None,
    k_s=15,
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
        for k in range(k_s):
            sub = df_ex[df_ex["k"] == k]
            if len(sub) > 0:
                asr_by_k.append(sub["jailbroken"].mean() * 100)
        return np.array(asr_by_k)

    # CBL baseline
    cbl_series = get_asr_series(cbl_versus)
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

    # CBL reference band: mean ± std
    ax.axhline(cbl_mean, color="#2980b9", linewidth=2.0, linestyle="--", label=f"CBL  μ={cbl_mean:.1f}%")
    ax.axhspan(cbl_mean - cbl_std, cbl_mean + cbl_std,
               color="#2980b9", alpha=0.12, label=f"CBL ±σ ({cbl_mean - cbl_std:.1f}–{cbl_mean + cbl_std:.1f}%)")

    # significance stars + p-value annotations
    for i, (_, row) in enumerate(plot_df.iterrows()):
        sig_str = f"p={row['p_val']:.3f}" + (" ★" if row["sig"] else "")
        ax.text(i, row["mean"] + row["std"] + 1.5, sig_str,
                ha="center", va="bottom", fontsize=8,
                color="#c0392b" if row["sig"] else "#7f8c8d")

    ax.set_ylabel(r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $\sigma_k$  (%)", fontsize=11)
    ax.set_xlabel("Target model", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"{(algo or 'algo').capitalize()} — total ASR: CBL superiority\n"
        r"$\mathbb{E}_k[\mathrm{ASR}]$ ± $D_k$,  "
        f"Welch t-test  α={alpha}  |  ★ = CBL significantly better",
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
    
    
def plot_asr_heatmap(versus, algo, ds, parsed_args, k_s=15, save_path=None):
    """
    For each category and attempt k, compute ASR(k) = mean jailbroken at attempt k.
    Plots a seaborn heatmap of Category vs Attempt k, sorted by E_k[ASR] descending.
    """
    df = get_df(versus)
    df["jailbroken"] = df["jailbroken"] & (df["exists"] == True)  # treat non-existing as non-jailbroken
    df["exists"] = True  # for ASR calculation, all rows are considered (non-existing = ASR=0)
    
    df_ex = df[df["exists"] == True].copy()
    
    if df_ex.empty:
        print(f"No valid data for {versus}. Cannot plot ASR heatmap.")
        return
    else:
        print(f"Data for '{versus}': {len(df_ex)} rows, cols={df_ex.columns.tolist()}")
        
    # --- build (category, k) -> ASR table ---
    records = []
    for cat in df_ex["harm_area"].unique():
        cat_df = df_ex[df_ex["harm_area"] == cat]
        for k in range(k_s):
            sub = cat_df[cat_df["k"] == k]
            if len(sub) == 0:
                continue
            asr_k = sub["jailbroken"].mean() * 100          # ASR at this k
            records.append({"category": cat, "k": k, "asr": asr_k})

    asr_df = pd.DataFrame(records)

    # Compute E_k[ASR] per category to sort the heatmap rows descending
    stats = (
        asr_df.groupby("category")["asr"]
        .mean()
        .reset_index(name="mean_asr")
        .sort_values("mean_asr", ascending=False)
    )

    # Pivot to create a matrix layout: rows = category, columns = k
    heatmap_data = asr_df.pivot(index="category", columns="k", values="asr")
    # Reindex the rows to follow the descending order of mean ASR
    heatmap_data = heatmap_data.loc[stats["category"]]

    target_short = all_versus.get(versus.lower(), versus)

    # Dynamically scale height based on the number of unique categories
    fig, ax = plt.subplots(figsize=(12, max(6, len(stats) * 0.45)))
    sns.set_theme(style="white", font_scale=0.95)

    # Render the heatmap
    sns.heatmap(
        data=heatmap_data,
        annot=True,                 # Display ASR values inside each cell
        fmt=".1f",                  # Format numbers to 1 decimal place
        cmap="mako_r",              # Maintain the original "mako_r" color palette progression
        linewidths=0.5,             # Add slight spacing between cells
        cbar_kws={'label': 'ASR (%)'},
        annot_kws={"size": 8},      # Adjust text size inside cells to ensure clean fit
        ax=ax,
    )

    ax.set_title(
        f"{algo.capitalize()} vs {target_short}\n"
        r"ASR (%) by Category and Attempt $k$ (Sorted by $\mathbb{E}_k[\mathrm{ASR}]$ Descending)",
        fontsize=12,
    )
    ax.set_xlabel(r"Attempt ($k$)", fontsize=10)
    ax.set_ylabel("Category", fontsize=10)
    
    # Heatmap ticks look cleaner unrotated since horizontal space is managed
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    
    plt.tight_layout()

    if save_path is None:
        v = f"_{parsed_args.version}" if parsed_args.version else ""
        save_path = f"{BASE_GRAPHS_FOLDER}{algo}_{ds}{v}_{versus}_{parsed_args.attacker}_asr_heatmap_h.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to '{save_path}'")
    plt.close()
    
def plot_asr_cross_target_heatmap(versus_list, algo, ds, parsed_args, k_s=15, save_path=None):
    """
    Computes the overall mean ASR (averaged over all attempts k=0..k_s-1) 
    for each Category across multiple Target models.
    
    Plots a heatmap with:
      - Y-axis: Harm Categories (sorted by overall average ASR descending)
      - X-axis: Target Models (Versus)
    """
    all_records = []

    # Loop through all targets to gather comparison data
    for versus in versus_list:
        df = get_df(versus)
        df["jailbroken"] = df["jailbroken"] & (df["exists"] == True)
        df["exists"] = True
        
        df_ex = df[df["exists"] == True].copy()
        if df_ex.empty:
            continue
            
        target_short = all_versus.get(versus.lower(), versus)
        
        # Calculate ASR per category and per k for this specific target
        for cat in df_ex["harm_area"].unique():
            cat_df = df_ex[df_ex["harm_area"] == cat]
            for k in range(k_s):
                sub = cat_df[cat_df["k"] == k]
                if len(sub) == 0:
                    continue
                asr_k = sub["jailbroken"].mean() * 100
                all_records.append({
                    "category": cat, 
                    "target": target_short, 
                    "k": k, 
                    "asr": asr_k
                })

    if not all_records:
        print("No valid data found for the provided targets.")
        return

    full_df = pd.DataFrame(all_records)

    # First, compress the "k" dimension by taking the mean across all attempts
    # This gives us the E_k[ASR] for every (category, target) pair
    mean_df = (
        full_df.groupby(["category", "target"])["asr"]
        .mean()
        .reset_index(name="mean_asr")
    )

    # Determine row order: Sort categories by their overall vulnerability across ALL targets
    row_order = (
        mean_df.groupby("category")["mean_asr"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    # Pivot into a 2D matrix: rows = category, columns = target model
    heatmap_data = mean_df.pivot(index="category", columns="target", values="mean_asr")
    heatmap_data = heatmap_data.loc[row_order]  # Apply sorting

    # Dynamic sizing adjusted for number of targets and categories
    fig, ax = plt.subplots(figsize=(max(8, len(heatmap_data.columns) * 1.5), max(6, len(heatmap_data) * 0.45)))
    sns.set_theme(style="white", font_scale=0.95)

    # Plotting the mean ASR values
    sns.heatmap(
        data=heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="mako_r",
        linewidths=0.5,
        cbar_kws={'label': r'Mean ASR (%) over $k$'},
        annot_kws={"size": 9},
        ax=ax,
    )

    ax.set_title(
        f"{algo.capitalize()} Performance Cross-Target Evaluation\n"
        r"$\mathbb{E}_k[\mathrm{ASR}]$ (%) Matrix by Category and Target Model",
        fontsize=12,
    )
    ax.set_xlabel("Target Model", fontsize=10)
    ax.set_ylabel("Harm Category", fontsize=10)

    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)

    plt.tight_layout()

    if save_path is None:
        v = f"_{parsed_args.version}" if parsed_args.version else ""
        save_path = f"{BASE_GRAPHS_FOLDER}{algo}_{ds}{v}_{parsed_args.attacker}_cross_target_heatmap.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved cross-target heatmap to '{save_path}'")
    plt.close()
    
plot_asr_cross_target_heatmap(
        versus_list=list(all_versus.keys()),
        algo=algo, ds=ds,
        parsed_args=parsed_args,
        k_s=k_s,
)

# After the existing loop that builds dd{}
for vs in all_versus.keys():
    plot_asr_mean_variance(
        versus=vs,
        algo=algo, ds=ds,
        parsed_args=parsed_args,
        k_s=k_s,
    )

plot_cbl_superiority_total(
    cbl_versus="cbllama",
    other_versus_list=[k for k in all_versus if k != "cbllama"],
    algo=algo, ds=ds,
    parsed_args=parsed_args,
    alpha=0.05,
)