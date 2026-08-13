import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
from scipy import stats

# ── Load ───────────────────────────────────────────────────────────────────────

pf = pq.ParquetFile("marta_corrected_neweval_checkpoint.parquet")
df = pf.read().to_pandas().iloc[:22500]

# ── Filter ─────────────────────────────────────────────────────────────────────

df = df[df["is_refused"] != True]

eval_cols = [c for c in df.columns if c.startswith("eval_PEC7_")]
df = df[df[eval_cols].replace("", np.nan).notna().any(axis=1)]

print(f"Rows after filtering: {len(df)}")
print(f"Evaluator columns: {eval_cols}")

# ── Parse JSON eval columns ────────────────────────────────────────────────────

METRICS = ["proximity", "erosion", "cooperation", "harmful"]

def parse_eval_col(series, col_name):
    records = []
    for val in series:
        if not val or not isinstance(val, str):
            records.append({m: np.nan for m in METRICS})
            continue
        try:
            d = json.loads(val)
            records.append({
                "proximity":   d.get("proximity", np.nan),
                "erosion":     d.get("erosion", np.nan),
                "cooperation": d.get("cooperation", np.nan),
                "harmful":     int(d.get("harmful", np.nan)) if d.get("harmful") is not None else np.nan,
            })
        except Exception:
            records.append({m: np.nan for m in METRICS})
    return pd.DataFrame(records, index=series.index)

parsed = {}
for col in eval_cols:
    evaluator_name = col.replace("eval_PEC7_", "")
    parsed[evaluator_name] = parse_eval_col(df[col], col)

# Bagel baseline
baseline = pd.DataFrame({
    "proximity":   pd.to_numeric(df["eval_proximity"],   errors="coerce"),
    "erosion":     pd.to_numeric(df["eval_erosion"],     errors="coerce"),
    "cooperation": pd.to_numeric(df["eval_cooperation"], errors="coerce"),
    "harmful":     pd.to_numeric(df["harmful"],          errors="coerce"),
}, index=df.index)

evaluator_names = list(parsed.keys())
n_evaluators = len(evaluator_names)

# ── Figure 1: Scatter plots — each evaluator vs bagel, per metric ──────────────

fig1, axes = plt.subplots(n_evaluators, 4, figsize=(16, 4 * n_evaluators))
fig1.suptitle("Evaluator vs Bagel (baseline) — per metric", fontsize=14, fontweight="bold", y=1.01)

for i, name in enumerate(evaluator_names):
    ev = parsed[name]
    for j, metric in enumerate(METRICS):
        ax = axes[i][j]
        x = baseline[metric].values
        y = ev[metric].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, alpha=0.3, s=10, color="steelblue")

        if len(x) > 1:
            r, p = stats.pearsonr(x, y)
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, m * xline + b, color="crimson", linewidth=1.5)
            ax.set_title(f"{name}\n{metric}  r={r:.2f} p={p:.2e}", fontsize=8)
        else:
            ax.set_title(f"{name}\n{metric}  (no data)", fontsize=8)

        ax.set_xlabel("bagel", fontsize=7)
        ax.set_ylabel(name[:15], fontsize=7)

plt.tight_layout()
plt.savefig("eval_scatter.png", dpi=150, bbox_inches="tight")
print("Saved eval_scatter.png")

# ── Figure 2: Pearson r heatmap — all evaluators + bagel, per metric ──────────

all_names = ["bagel"] + evaluator_names
fig2, axes2 = plt.subplots(1, 4, figsize=(18, max(4, n_evaluators + 1)))
fig2.suptitle("Pearson r between evaluators — per metric", fontsize=13, fontweight="bold")

for j, metric in enumerate(METRICS):
    ax = axes2[j]
    series = {"bagel": baseline[metric]}
    for name in evaluator_names:
        series[name] = parsed[name][metric]

    corr_matrix = pd.DataFrame(series).corr(method="pearson")

    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(all_names)))
    ax.set_yticks(range(len(all_names)))
    ax.set_xticklabels([n[:14] for n in all_names], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([n[:14] for n in all_names], fontsize=7)
    ax.set_title(metric, fontsize=10, fontweight="bold")

    for r in range(len(all_names)):
        for c in range(len(all_names)):
            val = corr_matrix.values[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(val) < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("eval_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved eval_heatmap.png")

# ── Figure 3: MAE and bias (mean signed error) vs bagel ───────────────────────

mae_data  = {m: [] for m in METRICS}
bias_data = {m: [] for m in METRICS}

for name in evaluator_names:
    ev = parsed[name]
    for metric in METRICS:
        diff = ev[metric] - baseline[metric]
        diff = diff.dropna()
        mae_data[metric].append(diff.abs().mean())
        bias_data[metric].append(diff.mean())

fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))
fig3.suptitle("MAE and Bias vs Bagel baseline", fontsize=13, fontweight="bold")

x = np.arange(n_evaluators)
short_names = [n[:16] for n in evaluator_names]

for j, metric in enumerate(METRICS):
    # MAE
    ax = axes3[0][j]
    ax.bar(x, mae_data[metric], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"MAE — {metric}", fontsize=9)
    ax.set_ylabel("mean abs error")

    # Bias
    ax = axes3[1][j]
    colors = ["crimson" if v > 0 else "seagreen" for v in bias_data[metric]]
    ax.bar(x, bias_data[metric], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Bias — {metric}", fontsize=9)
    ax.set_ylabel("mean signed error\n(+above bagel, −below)")

plt.tight_layout()
plt.savefig("eval_mae_bias.png", dpi=150, bbox_inches="tight")
print("Saved eval_mae_bias.png")

# ── Print summary table ────────────────────────────────────────────────────────

print("\n── Pearson r vs bagel ────────────────────────────────────────────")
rows = []
for name in evaluator_names:
    ev = parsed[name]
    row = {"evaluator": name}
    for metric in METRICS:
        x = baseline[metric]
        y = ev[metric]
        mask = x.notna() & y.notna()
        if mask.sum() > 1:
            r, _ = stats.pearsonr(x[mask], y[mask])
            row[metric] = round(r, 3)
        else:
            row[metric] = np.nan
    rows.append(row)

summary = pd.DataFrame(rows).set_index("evaluator")
print(summary.to_string())
print()

# ── Print harmful count per judge ──────────────────────────────────────────────

print("\n── Harmful count per evaluator ───────────────────────────────────")
rows_harmful = []
for name in evaluator_names:
    ev = parsed[name]
    total = ev["harmful"].notna().sum()
    harmful_count = (ev["harmful"] == 1).sum()
    rows_harmful.append({
        "evaluator": name,
        "harmful_count": harmful_count,
        "total_evaluated": total,
        "harmful_rate": round(harmful_count / total, 3) if total > 0 else np.nan,
    })

# Also add bagel baseline
bagel_total = baseline["harmful"].notna().sum()
bagel_harmful = (baseline["harmful"] == 1).sum()
rows_harmful.append({
    "evaluator": "bagel (baseline)",
    "harmful_count": bagel_harmful,
    "total_evaluated": bagel_total,
    "harmful_rate": round(bagel_harmful / bagel_total, 3) if bagel_total > 0 else np.nan,
})

harmful_summary = pd.DataFrame(rows_harmful).set_index("evaluator")
print(harmful_summary.to_string())