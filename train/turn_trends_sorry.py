import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Încărcare date
df = pd.read_parquet("sorry_corrected/")
trend_data = df.copy()

models = {
    "DOR": "wicai24/Llama-3-8B-Instruct-W-DOOR-exponential",
    "DER": "Youliang/llama3-8b-instruct-lora-derta-100step",
    "GUR": "Qwen/Qwen3Guard-Gen-8B::meta-llama/Meta-Llama-3-8B-Instruct::response_moderation",
    "GUQ": "Qwen/Qwen3Guard-Gen-8B::meta-llama/Meta-Llama-3-8B-Instruct::question_moderation",
    "CBL": "GraySwanAI/Llama-3-8B-Instruct-RR",
    "NEMR": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3::meta-llama/Meta-Llama-3-8B-Instruct::response_moderation",
    "NEMQ": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3::meta-llama/Meta-Llama-3-8B-Instruct::question_moderation",
    "LLA": "meta-llama/Meta-Llama-3-8B-Instruct",
    "MT-CB": "MT-CB"
}
trend_data["target"] = trend_data["target"].map({y: x for x, y in models.items()})

# 2. Calcul turn-uri rămase
biggest_turns = (
    trend_data.groupby("conversation_id")["turn_count"].max().reset_index()
)
trend_data["biggest_turn"] = trend_data["conversation_id"].map(
    biggest_turns.set_index("conversation_id")["turn_count"]
)
trend_data["turns_until_end"] = (
    trend_data["biggest_turn"] - trend_data["turn_count"]
)

# Configurații vizuale
metrics = ["eval_cooperation", "eval_proximity", "eval_erosion", "harmful"]
unified_palette = {
    "Harmful Ending": "#ef5350",  # Soft Red
    "Unharmful Ending": "#26a69a",  # Soft Teal/Blue
}

targets = [t for t in list(models.keys()) if t in trend_data["target"].unique()]
targets_to_keep = ["CBL", "GUR", "LLA", "DOR"]
targets = [t for t in targets if t in targets_to_keep]
target_chunks = [targets[i : i + 4] for i in range(0, len(targets), 4)]

os.makedirs("marta_graphs", exist_ok=True)

# # 3. Generare grid-uri compacte (2 partitii a cate 4 modele)
# for chunk_idx, chunk_targets in enumerate(target_chunks):
    
#     # Cream direct o matrice 4x4 de axe pentru a avea control total pe spacing
#     fig, axes = plt.subplots(4, 4, figsize=(20, 18), sharex=False)
    
#     for idx, target_value in enumerate(chunk_targets):
#         # Determinam unde incepe blocul de 2x2 al modelului curent in matricea 4x4
#         # Model 0 -> rand 0-1, col 0-1  |  Model 1 -> rand 0-1, col 2-3
#         # Model 2 -> rand 2-3, col 0-1  |  Model 3 -> rand 2-3, col 2-3
#         row_offset = (idx // 2) * 2
#         col_offset = (idx % 2) * 2
        
#         target_df = trend_data[trend_data["target"] == target_value]

#         for i, metric in enumerate(metrics):
#             ix = row_offset + (i // 2)
#             iy = col_offset + (i % 2)
#             ax = axes[ix, iy]

#             if not target_df.empty:
#                 sns.lineplot(
#                     data=target_df,
#                     x="turns_until_end",
#                     y=metric,
#                     hue="conversation_outcome",
#                     style="attack_type",
#                     style_order=["crescendo", "fitd"],
#                     marker="o",
#                     ax=ax,
#                     palette=unified_palette,
#                     legend=(idx == 1 and i == 3),  # O singura legenda pe tot gridul
#                 )

#             # Titluri si etichete minimaliste
#             # Punem titlul modelului doar deasupra primului grafic din mini-blocul sau
#             if i == 0:
#                 ax.set_title(f"Target: {target_value} | Cooperation", fontsize=11, fontweight="bold")
#             elif i == 1:
#                 ax.set_title("Proximity", fontsize=11, fontweight="bold")
#             else:
#                 ax.set_title(metric.replace("eval_", "").title(), fontsize=10)

#             ax.set_xlabel("Turns Until End", fontsize=8)
#             ax.set_ylabel(metric.replace("eval_", ""), fontsize=8)
#             ax.tick_params(axis="both", which="major", labelsize=8)

#             if not target_df.empty:
#                 ax.set_xlim(target_df["turns_until_end"].max(), 0)
#             ax.grid(True, linestyle="--", alpha=0.4)

#             if idx == 1 and i == 3:
#                 ax.legend(fontsize=8, title_fontsize=9, loc="upper left")

#     # AICI E MAGIA: Strângem la maxim padding-urile dintre axe
#     # wspace si hspace controleaza distanta dintre coloane si randuri
#     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.22, hspace=0.28)

#     output_path = f"marta_graphs/grid_targets_2x2_part{chunk_idx + 1}.png"
#     plt.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Salvat compact: {output_path}")
    
    
import numpy as np
import pandas as pd

# ==========================================
# DAMPING RESISTANCE
# ==========================================

def damping_resistance(group, metric="harmful"):
    """
    DR = total downward recovery /
         total upward attack progress

    0  -> nu recuperează niciodată
    1  -> anulează complet toate creșterile
    >1 -> recuperează mai mult decât a crescut
    """

    series = (
        group
        .sort_values("turn_count")[metric]
        .astype(float)
        .values
    )

    if len(series) < 2:
        return 0.0

    delta = np.diff(series)

    upward = delta[delta > 0].sum()

    downward = np.abs(
        delta[delta < 0]
    ).sum()
    
    if upward == 0:
        return np.nan

    return downward **2 / (upward**2 + downward**2)

all_dr = []

for metric in [
    "eval_cooperation",
    "eval_erosion",
    "eval_proximity",
]:

    dr_df = (
        trend_data
        .groupby(
            [
                "conversation_id",
                "target",
                "attack_type",
                "conversation_outcome",
            ]
        )
        .apply(
            lambda g: damping_resistance(
                g,
                metric=metric
            )
        )
        .reset_index(name="damping_resistance")
    )

    dr_df["metric"] = metric

    all_dr.append(dr_df)

all_dr = pd.concat(all_dr, ignore_index=True)


summary = (
    all_dr
    .groupby(
        [
            "target",
            "metric",
            "attack_type",
            "conversation_outcome"
        ]
    )["damping_resistance"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        count="count"
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("DAMPING RESISTANCE SUMMARY")
print("=" * 80)
print(summary.round(3).to_string())


ranking = (
    all_dr
    .groupby(
        [
            "target",
            "metric",
            "attack_type"
        ]
    )["damping_resistance"]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 80)
print("MODEL RANKING")
print("=" * 80)

for metric in ranking["metric"].unique():

    print(f"\n--- {metric} ---")

    print(
        ranking[
            ranking["metric"] == metric
        ]
        .sort_values(
            "damping_resistance",
            ascending=False
        )
        .round(3)
        .to_string(index=False)
    )
    
plt.figure(figsize=(14, 7))

sns.boxplot(
    data=all_dr,
    x="metric",
    y="damping_resistance",
    hue="conversation_outcome"
)

plt.title("Damping Resistance Distribution")
plt.tight_layout()

plt.savefig(
    "marta_graphs/sorry_damping_resistance_boxplots.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(14, 7))

sns.boxplot(
    data=all_dr[all_dr['metric'] == "eval_erosion"],
    x="target",
    y="damping_resistance",
    hue="conversation_outcome"
)

plt.title("Damping Resistance Distribution")
plt.tight_layout()

plt.savefig(
    "marta_graphs/sorry_damping_resistance_boxplots_erosion.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()



plt.figure(figsize=(14, 7))

sns.boxplot(
    data=all_dr[(all_dr['metric'] == "eval_erosion") & (all_dr['conversation_outcome'] == 'Unharmful Ending')],
    x="attack_type",
    y="damping_resistance",
    hue="target"
)

plt.title("Damping Resistance Distribution")
plt.tight_layout()

plt.savefig(
    "marta_graphs/sorry_damping_resistance_boxplots_erosion_byatk.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()



heatmap_df = (
    all_dr
    .groupby(
        [
            "target",
            "metric"
        ]
    )["damping_resistance"]
    .mean()
    .unstack()
)

plt.figure(figsize=(8,6))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="viridis"
)

plt.title("Mean Damping Resistance")
plt.tight_layout()

plt.savefig(
    "marta_graphs/sorry_damping_resistance_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


global_dr = (
    all_dr
    .groupby(["target", "conversation_outcome"])["damping_resistance"]
    .mean()
    .sort_values(ascending=False)
)

print("\n")
print("=" * 80)
print("GLOBAL DAMPING RESISTANCE")
print("=" * 80)
print(global_dr.round(3).to_string())