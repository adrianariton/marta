# read from shards in parquets

import pandas as pd
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("marta/")
# max_turns_to_plot = 23
trend_data = df#[df["turn_count"] <= max_turns_to_plot]
print(df["attack_type"].unique().tolist())
metrics = ["eval_cooperation", "eval_proximity", "eval_erosion", "harmful"]
fig, axes = plt.subplots(2, 2, figsize=(18, 18), sharex=True)

# check biggest turn in a conversation_id equal to conversation_id trend_data["conversation_id"]
biggest_turns = trend_data.groupby("conversation_id")["turn_count"].max().reset_index()

trend_data["biggest_turn"] = trend_data["conversation_id"].map(biggest_turns.set_index("conversation_id")["turn_count"])
trend_data["turns_until_end"] = trend_data["biggest_turn"] - trend_data["turn_count"]

unified_palette = {
    "Harmful Ending": "#ef5350",   # Soft Red
    "Unharmful Ending": "#26a69a"  # Soft Teal/Blue
}

for i, metric in enumerate(metrics):
    # Pass the ENTIRE trend_data at once instead of looping through attacks manually.
    # This allows Seaborn to build a single, intelligent, joint legend automatically.
    
    ix = i // 2
    iy = i % 2
    sns.lineplot(
        data=trend_data,
        x="turns_until_end",
        y=metric,
        hue="conversation_outcome",  
        style="attack_type",         
        style_order=['fitd', 'goat', "race"],
        marker="o",
        ax=axes[ix, iy],
        palette=unified_palette,
    )
    # 1. Main Title
    axes[ix, iy].set_title(
        f'{metric.replace("eval_", "").title()} Trend over Turns', 
        fontsize=34
    )
    
    # 2. X and Y Axis Labels
    axes[ix, iy].set_xlabel("Turns Until End of Conversation", fontsize=34)
    axes[ix, iy].set_ylabel(f"Average {metric.replace('eval_', '')}", fontsize=34)
    
    # 3. X and Y Axis Ticks (The numbers/values on the axes)
    axes[ix, iy].tick_params(axis='both', which='major', labelsize=22)
    
    axes[ix, iy].set_xlim(trend_data["turns_until_end"].max(), 0)  
    axes[ix, iy].grid(True, linestyle="--", alpha=0.6)

    # 4. Legend
    axes[ix, iy].legend(fontsize=29, title_fontsize=34, loc="upper left")
    # Clean up internal axes legends if you have multiple subplots,
    # leaving it only on the last one, or let tight_layout handle it.
    if i < len(metrics) - 1:
        axes[ix, iy].get_legend().remove()

plt.tight_layout()
plt.savefig("marta_graphs/turn_trends.png", dpi=150, bbox_inches="tight")
exit(0)

# Filter to the first few turns where most data exists to avoid tail noise
max_turns_to_plot = 5
trend_data = df[df["turn_count"] <= max_turns_to_plot]

metrics = ["eval_cooperation", "eval_proximity", "eval_erosion"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

for i, metric in enumerate(metrics):
    sns.lineplot(
        data=trend_data,
        x="turn_count",
        y=metric,
        hue="conversation_outcome",
        marker="o",
        ax=axes[i],
        palette={"Harmful Ending": "#ef5350", "Unharmful Ending": "#26a69a"},
    )
    axes[i].set_title(ax_title := f'{metric.replace("eval_", "").title()} Trend over Turns')
    axes[i].set_xlabel("Turn Count")
    axes[i].set_ylabel("Average Score")
    axes[i].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("marta_graphs/turn_trends.png", dpi=150, bbox_inches="tight")


# Aggregate by conversation to avoid counting a single multi-turn attack multiple times
convo_summary = (
    df.groupby("conversation_id")
    .agg(
        {
            "attack_type": "first",
            "dataset": "first",  # assuming 'dataset' represents the category/theme
            "conversation_outcome": "first",
        }
    )
    .reset_index()
)

convo_summary["is_harmful_bool"] = convo_summary["conversation_outcome"] == "Harmful Ending"

# Plot 1: Attack Type Success Rates
plt.figure(figsize=(12, 6))
attack_success = (
    convo_summary.groupby("attack_type")["is_harmful_bool"].mean().sort_values(ascending=False)
    * 100
)
sns.barplot(x=attack_success.values, y=attack_success.index, palette="rocket")
plt.title("Vulnerability Rate by Attack Type (% of Convos Ending Harmfully)")
plt.xlabel("Success Rate (%)")
plt.ylabel("Attack Type")
plt.tight_layout()
plt.savefig("marta_graphs/attack_success_rates.png", dpi=150, bbox_inches="tight")

# Plot 2: Dataset / Category Volume & Outcomes
plt.figure(figsize=(12, 6))
sns.countplot(
    data=convo_summary,
    y="dataset",
    hue="conversation_outcome",
    order=convo_summary["dataset"].value_counts().index,
    palette={"Harmful Ending": "#ef5350", "Unharmful Ending": "#26a69a"},
)
plt.title("Conversation Distribution and Outcomes by Category (Dataset)")
plt.xlabel("Number of Conversations")
plt.ylabel("Dataset Category")
plt.legend(title="Outcome")
plt.tight_layout()
plt.savefig("marta_graphs/dataset_outcomes.png", dpi=150, bbox_inches="tight")
# 1. Deduplicate by conversation_id to get one record per distinct attack campaign
unique_attacks = df.groupby("conversation_id").agg({"target": "first"}).reset_index()

# 2. Count total unique attacks per target model
attack_counts = unique_attacks["target"].value_counts().reset_index()
attack_counts.columns = ["target", "total_attacks"]

# 3. Plotting
plt.figure(figsize=(12, 6))

# Clean, professional deep blue color palette
colors = sns.color_palette("Blues_r", n_colors=len(attack_counts))

ax = sns.barplot(data=attack_counts, x="total_attacks", y="target", palette=colors)

# Add exact integer labels to the end of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=5, fontweight="bold", fontsize=10)

plt.title("Total Unique Attack Volume vs. Target Model", fontsize=14, pad=15, fontweight="bold")
plt.xlabel("Number of Unique Conversations (Attacks)", fontsize=11, labelpad=10)
plt.ylabel("Target Model", fontsize=11)
plt.xlim(0, max(attack_counts["total_attacks"]) * 1.15)  # Give 15% breathing room for labels
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("marta_graphs/target_coverage.png", dpi=150, bbox_inches="tight")


# 1. Deduplicate by conversation_id to get unique attack setups
dataset_coverage = df.groupby("conversation_id").agg({"dataset": "first"}).reset_index()

# 2. Count total unique attacks per dataset category
dataset_counts = dataset_coverage["dataset"].value_counts().reset_index()
dataset_counts.columns = ["dataset", "total_attacks"]

# 3. Plotting
plt.figure(figsize=(12, 6))
colors = sns.color_palette("Purples_r", n_colors=len(dataset_counts))

ax = sns.barplot(data=dataset_counts, x="total_attacks", y="dataset", palette=colors)

# Add exact integer labels to the end of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=5, fontweight="bold", fontsize=10)

plt.title(
    "Dataset Category Coverage (Total Unique Conversations)", fontsize=14, pad=15, fontweight="bold"
)
plt.xlabel("Number of Unique Conversations", fontsize=11, labelpad=10)
plt.ylabel("Dataset / Category", fontsize=11)
plt.xlim(0, max(dataset_counts["total_attacks"]) * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("marta_graphs/ds_coverage.png", dpi=150, bbox_inches="tight")

# 1. Deduplicate by conversation_id to get unique attack setups
attack_type_coverage = df.groupby("conversation_id").agg({"attack_type": "first"}).reset_index()

# 2. Count total unique attacks per attack method
attack_type_counts = attack_type_coverage["attack_type"].value_counts().reset_index()
attack_type_counts.columns = ["attack_type", "total_attacks"]

# 3. Plotting
plt.figure(figsize=(12, 6))
colors = sns.color_palette("YlOrRd_r", n_colors=len(attack_type_counts))

ax = sns.barplot(data=attack_type_counts, x="total_attacks", y="attack_type", palette=colors)

# Add exact integer labels to the end of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=5, fontweight="bold", fontsize=10)

plt.title(
    "Attack Type Coverage (Total Unique Conversations)", fontsize=14, pad=15, fontweight="bold"
)
plt.xlabel("Number of Unique Conversations", fontsize=11, labelpad=10)
plt.ylabel("Attack Type", fontsize=11)
plt.xlim(0, max(attack_type_counts["total_attacks"]) * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("marta_graphs/attack_type_coverage.png", dpi=150, bbox_inches="tight")


df = df[df["is_goal_harmful"]]

# 1. Identify which conversation_ids ever resulted in a harmful outcome
harmful_convos = set(df.loc[df["harmful"] == True, "conversation_id"].unique())

# 2. Group by target model and conversation_id to compress the multi-turn history
# We take 'first' for target assuming a conversation is always with the same model.
convo_models = df.groupby("conversation_id").agg({"target": "first"}).reset_index()

# 3. Tag whether the conversation was successfully breached
convo_models["is_success"] = convo_models["conversation_id"].isin(harmful_convos)
# 4. Calculate Attack Success Rate (ASR) % per target model
asr_data = (
    convo_models.groupby("target")["is_success"]
    .mean()
    .reset_index()
    .rename(columns={"is_success": "ASR"})
)
asr_data["ASR"] = asr_data["ASR"] * 100  # Convert to percentage
asr_data = asr_data.sort_values(ascending=False, by="ASR")

# 5. Plotting
plt.figure(figsize=(12, 6))

# Define custom clean colors (red/orange theme fitting for attack success rates)
colors = sns.color_palette("YlOrRd_r", n_colors=len(asr_data))

ax = sns.barplot(data=asr_data, x="ASR", y="target", palette=colors)

# Add exact percentage data labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=5, fontweight="bold", fontsize=10)

plt.title("Attack Success Rate (ASR) vs. Target Model", fontsize=14, pad=15, fontweight="bold")
plt.xlabel("Attack Success Rate (ASR) %", fontsize=11, labelpad=10)
plt.ylabel("Target Model", fontsize=11)
plt.xlim(0, max(asr_data["ASR"]) + 10)  # Give extra breathing room for labels
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("marta_graphs/asrs.png", dpi=150, bbox_inches="tight")

print("=== CONVERSATION OUTCOME BREAKDOWN ===")
print(convo_summary["conversation_outcome"].value_counts(normalize=True) * 100)

print("\n=== TOP 5 MOST DANGEROUS ATTACK TYPES ===")
print(attack_success.head(5))
