# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # Read from shards in parquets
# df = pd.read_parquet("marta/")

# # Filter for the specific dataset
# df = df[df["dataset"] == "walledai/XSTest"]

# # Group by target and harmful status to compute the mean refusal rate
# df_grouped = df.groupby(["target", "is_goal_harmful"])["is_refused"].mean().reset_index()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # 1. Prepare the Data (Pivot to get both metrics side-by-side per model)
# df_pivot = df_grouped.pivot(index="target", columns="is_goal_harmful", values="is_refused").reset_index()
# df_pivot.columns = ["target", "safe_refusal", "unsafe_refusal"]

# # Calculate the difference: Unsafe Refusal (True Harm) - Over-refusal (Safe Context)
# df_pivot["diff"] = df_pivot["unsafe_refusal"] - df_pivot["safe_refusal"]

# # Sort models by the size of the difference
# df_pivot = df_pivot.sort_values(by="diff", ascending=False).reset_index(drop=True)

# # Find the index of the maximum difference to highlight it
# max_diff_idx = df_pivot["diff"].idxmax()
# max_model = df_pivot.loc[max_diff_idx, "target"]
# max_val = df_pivot.loc[max_diff_idx, "diff"]

# # 2. Plotting
# fig, ax = plt.subplots(figsize=(12, 6))

# # Create a custom color palette where the max model stands out
# colors = ["#1f77b4" if i != max_diff_idx else "#d62728" for i in range(len(df_pivot))]

# sns.barplot(
#     x="target",
#     y="diff",
#     data=df_pivot,
#     palette=colors,
#     ax=ax
# )



# # Add a prominent text callout/annotation for the highlighted model
# ax.annotate(
#     f"Max Gap: {max_val:.2f}\n({max_model})",
#     xy=(max_diff_idx, max_val),
#     xytext=(max_diff_idx + 0.3, max_val - 0.05 if max_val > 0.1 else max_val + 0.05),
#     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
#     fontweight="bold",
#     color="#d62728",
#     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="red")
# )

# # Baseline reference line at 0
# ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

# # Titles & Labels
# ax.set_title("Safety Margin: Unsafe Refusal Rate vs. Over-Refusal Rate", fontsize=14, fontweight="bold")
# ax.set_xlabel("Target Model", fontsize=12)
# ax.set_ylabel("Difference (Unsafe Refusal - Safe Over-Refusal)", fontsize=12)

# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()

# # Save the second graph
# Path("marta_graphs").mkdir(parents=True, exist_ok=True)
# plt.savefig("marta_graphs/refusal_diff_rates.png")
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read from shards in parquets
df = pd.read_parquet("marta/")

# Filter for the specific dataset
df = df[df["dataset"] == "walledai/XSTest"]

# Group by target and harmful status to compute the mean refusal rate
df_grouped = df.groupby(["target", "is_goal_harmful"])["is_refused"].mean().reset_index()

# 1. Prepare the Data
df_pivot = df_grouped.pivot(index="target", columns="is_goal_harmful", values="is_refused").reset_index()
df_pivot.columns = ["target", "safe_refusal", "unsafe_refusal"]

# Calculate the Safety Margin (Difference)
df_pivot["diff"] = df_pivot["unsafe_refusal"] - df_pivot["safe_refusal"]

# Sort models so the plot looks clean
df_pivot = df_pivot.sort_values(by="diff", ascending=False).reset_index(drop=True)

# Find the VISUAL position of the maximum difference (always 0 after sorting descending)
max_model_row = df_pivot.loc[0]
max_model = max_model_row["target"]
max_val = max_model_row["diff"]

# 2. Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Map colors safely using a dictionary or conditional logic tied to the dataframe
colors = ["#d62728" if x == max_model else "#1f77b4" for x in df_pivot["target"]]

sns.barplot(
    x="target",
    y="diff",
    data=df_pivot,
    palette=colors,
    hue="target", # Explicitly pass hue to avoid modern Seaborn warnings
    legend=False,
    ax=ax
)

# Add a prominent text callout/annotation pointing to visual index 0
ax.annotate(
    f"Max Gap: {max_val:.2f}\n({max_model})",
    xy=(0, max_val),  # 0 is the x-position of the first bar now
    xytext=(0.5, max_val - (max_val * 0.1)), # Dynamically offset the text box
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
    fontweight="bold",
    color="#d62728",
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="red")
)

# Baseline reference line at 0
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

# Titles & Labels
ax.set_title("Safety Margin: Unsafe Refusal Rate vs. Over-Refusal Rate (XSTest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Target Model", fontsize=12)
ax.set_ylabel("Difference (Unsafe Refusal - Safe Over-Refusal)", fontsize=12)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save the graph
Path("marta_graphs").mkdir(parents=True, exist_ok=True)
plt.savefig("marta_graphs/refusal_diff_rates.png")
plt.show()