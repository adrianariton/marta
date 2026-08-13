import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from math import pi


# ── helpers ────────────────────────────────────────────────────────────────────

def read_jsonl(file_path):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: file not found – {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
    return data


def judge_metrics(human, ai, n_total):
    p_human = human.mean()
    p_ai    = ai.mean()
    n       = len(human)

    tp = ((ai == 1) & (human == 1)).sum()
    tn = ((ai == 0) & (human == 0)).sum()
    fp = ((ai == 1) & (human == 0)).sum()
    fn = ((ai == 0) & (human == 1)).sum()

    q1        = tp / (tp + fn)  if (tp + fn) > 0 else 1.0
    q0        = tn / (tn + fp)  if (tn + fp) > 0 else 1.0
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 1.0
    f1        = 2*precision*q1 / (precision+q1) if (precision+q1) > 0 else 0.0

    denom     = q1 - (1 - q0)
    theta_adj = max(0, min(1, (p_ai - (1 - q0)) / denom)) if abs(denom) > 1e-6 else p_ai

    se        = np.sqrt((p_ai*(1-p_ai)/n_total) + (p_human*(1-p_human)/n_total))
    z_score   = (p_ai - p_human) / se if se > 0 else np.nan
    corr      = human.corr(ai)

    return dict(p_human=p_human, p_ai=p_ai, theta_adj=theta_adj,
                q1=q1, q0=q0, q0q1=q0+q1,
                precision=precision, f1=f1,
                z_score=z_score, corr=corr)


def load_ai_series(df, col):
    """Return a 0/1 Series for a judge column (handles raw dict or JSON string)."""
    raw = df[col]
    if raw.apply(lambda x: isinstance(x, dict)).all():
        parsed = pd.json_normalize(raw.tolist())
    else:
        parsed = pd.json_normalize(
            raw.fillna("{}").map(lambda x: json.loads(x) if isinstance(x, str) else x)
        )
    for key in ("harmful", "goal_revealed"):
        if key in parsed.columns:
            return parsed[key].fillna(0).astype(int)
    raise KeyError(f"Neither 'harmful' nor 'goal_revealed' found in {col}")


# ── args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("file1",    help="Judge-1 JSONL")
parser.add_argument("file2",    help="Judge-2 JSONL")
parser.add_argument("file3",    help="AI-judges JSONL (all model columns)")
parser.add_argument("--judge1", default="Human-1")
parser.add_argument("--judge2", default="Human-2")
parser.add_argument("--out",    default="./plots")
args = parser.parse_args()

output_dir = f"{args.out}/judge_ranking"
os.makedirs(output_dir, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────

df1 = pd.DataFrame(read_jsonl(args.file1))
df2 = pd.DataFrame(read_jsonl(args.file2))
df3 = pd.DataFrame(read_jsonl(args.file3))

human1 = pd.json_normalize(df1["human_eval"])["goal_revealed"].astype(int)
human2 = pd.json_normalize(df2["human_eval"])["goal_revealed"].astype(int)

consensus_mask  = human1 == human2
human_consensus = human1[consensus_mask]
n_total         = len(human1)
n_consensus     = consensus_mask.sum()

JUDGES = {
    "ai_eval":                                   "Bagel",
    "eval_PEC7_phi-4-AWQ":                       "Phi-4",
    "eval_PEC7_Mixtral-8x22B-Instruct-v0.1-AWQ": "Mixtral-8×22B",
    "eval_PEC7_Qwen2.5-32B-Instruct-AWQ":        "Qwen2.5-32B",
    "eval_PEC7_Qwen2.5-72B-Instruct-AWQ":        "Qwen2.5-72B",
}

# ── compute metrics ────────────────────────────────────────────────────────────

records = {}
for col, name in JUDGES.items():
    if col not in df3.columns:
        print(f"⚠  column '{col}' not found – skipping {name}")
        continue
    ai = load_ai_series(df3, col)
    records[name] = {
        "vs_j1":   judge_metrics(human1,          ai,                 n_total),
        "vs_j2":   judge_metrics(human2,          ai,                 n_total),
        "vs_cons": judge_metrics(human_consensus, ai[consensus_mask], n_consensus),
    }

names = list(records.keys())

def build_df(variant):
    rows = []
    for name, r in records.items():
        m = r[variant]
        rows.append(dict(judge=name, **m))
    return pd.DataFrame(rows).set_index("judge")

df_j1   = build_df("vs_j1")
df_j2   = build_df("vs_j2")
df_cons = build_df("vs_cons")

# rank by q0+q1 on consensus subset — the bias-corrected discriminability measure
sorted_names = df_cons["q0q1"].sort_values(ascending=False).index.tolist()

print("\nq₀+q₁ ranking (consensus subset):")
for i, name in enumerate(sorted_names, 1):
    print(f"  {i}. {name}: q0+q1={df_cons.loc[name,'q0q1']:.3f}  "
          f"F1={df_cons.loc[name,'f1']:.3f}  "
          f"θ_adj={df_cons.loc[name,'theta_adj']:.1%}")


# ── light-mode colour constants ────────────────────────────────────────────────

BG       = "white"
AX_BG    = "#f8f9fb"
SPINE_C  = "#cccccc"
GRID_C   = "#e0e0e0"
TEXT_C   = "#1a1a2e"
SUBTEXT_C= "#555555"
LEGEND_BG= "#ffffff"
LEGEND_EC= "#cccccc"
REF_LINE = "#d62728"


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Radar chart
# ══════════════════════════════════════════════════════════════════════════════

RADAR_METRICS = ["theta_adj", "q1", "q0", "f1", "precision", "corr"]
RADAR_LABELS  = ["θ_adj", "Sensitivity\n(q₁)", "Specificity\n(q₀)", "F1", "Precision", "Corr (r)"]
N_RAD         = len(RADAR_METRICS)
angles        = [n_ / N_RAD * 2*pi for n_ in range(N_RAD)]
angles       += angles[:1]

PALETTE = sns.color_palette("tab10", len(names))

fig_rad, ax_rad = plt.subplots(figsize=(7, 7),
                                subplot_kw=dict(polar=True),
                                facecolor=BG)
ax_rad.set_facecolor(AX_BG)

for i, name in enumerate(names):
    vals  = [df_cons.loc[name, m] for m in RADAR_METRICS]
    vals += vals[:1]
    ax_rad.plot(angles, vals, color=PALETTE[i], linewidth=2, label=name)
    ax_rad.fill(angles, vals, color=PALETTE[i], alpha=0.10)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(RADAR_LABELS, color=TEXT_C, size=11)
ax_rad.set_ylim(0, 1)
ax_rad.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_rad.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=SUBTEXT_C, size=8)
ax_rad.tick_params(colors=TEXT_C)
ax_rad.spines["polar"].set_color(SPINE_C)
ax_rad.grid(color=GRID_C, linewidth=0.8)

ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
               frameon=True, facecolor=LEGEND_BG, edgecolor=LEGEND_EC,
               labelcolor=TEXT_C, fontsize=10)
ax_rad.set_title("Judge Performance — Radar\n(vs human consensus)",
                  color=TEXT_C, pad=20, size=13, fontweight="bold")

plt.tight_layout()
fig_rad.savefig(f"{output_dir}/radar_chart.png", dpi=160,
                bbox_inches="tight", facecolor=fig_rad.get_facecolor())
print(f"\nSaved → {output_dir}/radar_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Grouped bar chart (ranked by q₀+q₁, y-axis up to 2)
# ══════════════════════════════════════════════════════════════════════════════

BAR_METRICS = ["q0q1", "f1", "theta_adj", "corr"]
BAR_LABELS  = ["q₀+q₁", "F1", "θ_adj", "Corr (r)"]

n_judges  = len(sorted_names)
n_metrics = len(BAR_METRICS)
x         = np.arange(n_judges)
width     = 0.18
offsets   = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * width

fig_bar, ax_bar = plt.subplots(figsize=(11, 5), facecolor=BG)
ax_bar.set_facecolor(AX_BG)

metric_palette = sns.color_palette("muted", n_metrics)

for mi, (metric, label, color) in enumerate(zip(BAR_METRICS, BAR_LABELS, metric_palette)):
    vals = [df_cons.loc[name, metric] for name in sorted_names]
    bars = ax_bar.bar(x + offsets[mi], vals,
                      width=width, label=label,
                      color=color, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}",
                    ha="center", va="bottom",
                    color=TEXT_C, fontsize=7.5, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(
    [f"#{i+1}  {n}" for i, n in enumerate(sorted_names)],
    color=TEXT_C, fontsize=11
)
ax_bar.set_ylim(0, 2.0)
ax_bar.set_yticks(np.arange(0, 2.1, 0.2))
ax_bar.tick_params(colors=TEXT_C)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.spines[["left", "bottom"]].set_color(SPINE_C)
ax_bar.set_ylabel("Score", color=TEXT_C, fontsize=11)
ax_bar.yaxis.label.set_color(TEXT_C)
ax_bar.set_title("AI Judge Ranking — Key Metrics (ranked by q₀+q₁)",
                  color=TEXT_C, fontsize=13, fontweight="bold", pad=12)
ax_bar.grid(axis="y", color=GRID_C, linewidth=0.7, linestyle="--")
ax_bar.legend(facecolor=LEGEND_BG, edgecolor=LEGEND_EC,
               labelcolor=TEXT_C, fontsize=10, loc="upper right")

# reference line at q0+q1 = 1 (random classifier)
ax_bar.axhline(1.0, color=REF_LINE, linewidth=1.2, linestyle="--", alpha=0.7)
ax_bar.text(n_judges - 0.5, 1.02, "random (q₀+q₁=1)",
            color=REF_LINE, fontsize=8, va="bottom", ha="right")

plt.tight_layout()
fig_bar.savefig(f"{output_dir}/ranked_bar_chart.png", dpi=160,
                bbox_inches="tight", facecolor=fig_bar.get_facecolor())
print(f"Saved → {output_dir}/ranked_bar_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – Heatmap
# ══════════════════════════════════════════════════════════════════════════════

HEAT_METRICS = ["q0q1", "theta_adj", "q1", "q0", "f1", "precision", "corr", "z_score"]
HEAT_LABELS  = ["q₀+q₁", "θ_adj", "Sensitivity q₁", "Specificity q₀",
                "F1", "Precision", "Corr (r)", "Z-score"]

heat_data = pd.DataFrame(
    {label: [df_cons.loc[n, m] for n in sorted_names]
     for label, m in zip(HEAT_LABELS, HEAT_METRICS)},
    index=[f"#{i+1} {n}" for i, n in enumerate(sorted_names)]
)

# normalise z-score column to [0,1] for colour scale only
heat_display = heat_data.copy()
zs = heat_display["Z-score"]
heat_display["Z-score"] = (zs - zs.min()) / (zs.max() - zs.min() + 1e-9)
# q0+q1 lives in [0,2] — normalise for colour too
q = heat_display["q₀+q₁"]
heat_display["q₀+q₁"] = (q - 0) / 2.0

fig_heat, ax_heat = plt.subplots(figsize=(13, 4), facecolor=BG)
ax_heat.set_facecolor(BG)

sns.heatmap(heat_display,
            annot=heat_data.round(3),
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0, vmax=1,
            linewidths=0.5, linecolor=BG,
            ax=ax_heat,
            cbar_kws=dict(label="Normalised score"))

ax_heat.set_title("AI Judge Performance Heatmap\n(consensus subset · sorted by q₀+q₁)",
                   color=TEXT_C, fontsize=13, fontweight="bold", pad=12)
ax_heat.tick_params(colors=TEXT_C, labelsize=10)
ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=30, ha="right", color=TEXT_C)
ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0,  color=TEXT_C)
ax_heat.figure.axes[-1].tick_params(colors=TEXT_C)
ax_heat.figure.axes[-1].yaxis.label.set_color(TEXT_C)

plt.tight_layout()
fig_heat.savefig(f"{output_dir}/heatmap.png", dpi=160,
                 bbox_inches="tight", facecolor=fig_heat.get_facecolor())
print(f"Saved → {output_dir}/heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

TEX_METRICS = ["q0q1", "theta_adj", "q1", "q0", "f1", "precision", "corr", "z_score"]
TEX_HEADERS = [r"$q_0+q_1$", r"$\hat\theta$", r"$q_1$", r"$q_0$",
               r"F1", r"Prec.", r"$r$", r"$Z$"]

def fmt_cell(val, metric):
    if metric == "z_score":
        return f"{val:+.2f}"
    if metric == "theta_adj":
        return f"{val:.1%}"
    return f"{val:.3f}"

def best_idx(col_name, values):
    if col_name == "z_score":
        return int(np.argmin(np.abs(values)))
    return int(np.argmax(values))

rows_tex = []
for i, name in enumerate(sorted_names):
    row = [f"\\#{i+1} {name}"]
    for m in TEX_METRICS:
        row.append(fmt_cell(df_cons.loc[name, m], m))
    rows_tex.append(row)

# bold best per column
for mi, m in enumerate(TEX_METRICS):
    vals = [df_cons.loc[n, m] for n in sorted_names]
    br   = best_idx(m, vals)
    rows_tex[br][mi+1] = r"\textbf{" + rows_tex[br][mi+1] + "}"

col_fmt = "l" + "r"*len(TEX_METRICS)
header  = " & ".join([r"\textbf{Judge}"] + TEX_HEADERS) + r" \\"

lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{6pt}",
    r"\caption{AI Judge Performance on Goal-Revealed Classification "
    r"(sorted by $q_0+q_1$; \textbf{bold} = best per column; "
    r"consensus subset $n=" + str(n_consensus) + r"$ of $n=" + str(n_total) + r"$)}",
    r"\label{tab:judge_ranking}",
    r"\begin{tabular}{" + col_fmt + "}",
    r"\toprule",
    header,
    r"\midrule",
]
for row in rows_tex:
    lines.append(" & ".join(row) + r" \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

latex_str = "\n".join(lines)
tex_path  = f"{output_dir}/judge_ranking_table.tex"
with open(tex_path, "w") as f:
    f.write(latex_str)

print(f"\nSaved LaTeX table → {tex_path}")
print("\n" + "─"*60)
print(latex_str)
print("─"*60)
print("\n✅  All done!  Outputs in:", output_dir)