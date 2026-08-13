import json

goals = []
empty_convos = 0


def analyze_refusals(file_path):
    total_conversations = 0
    refusal_count = 0
    global goals

    # Common refusal prefixes
    refusal_triggers = ["I'm sorry", "I’m sorry", "I apologize", "cannot"]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                total_conversations += 1
                data = json.loads(line)
                messages = data.get("messages", [])
                goal = data.get("goal", "")
                row_id = data.get("dataset_row_id", "")
                if goal is not None and goal != "":
                    goals.append(row_id + "::" + goal)
                else:
                    goals.append("")

                # Check each assistant message in the conversation
                is_refusal_conv = False
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "").strip()
                        if content == "":
                            empty_convos += 1
                    elif msg.get("role") == "assistant":
                        content = msg.get("content", "").strip()

                        # Check if the reply starts with a refusal phrase
                        if any((trigger in content) for trigger in refusal_triggers):
                            is_refusal_conv = True
                            break  # Count the conversation once and move on

                if is_refusal_conv:
                    refusal_count += 1

        percentage = (refusal_count / total_conversations) * 100 if total_conversations > 0 else 0

        print(f"--- Dataset Analysis ---")
        print(f"Total Conversations: {total_conversations}")
        print(f"Refusal Conversations: {refusal_count}")
        print(f"Refusal Rate: {percentage:.2f}%")

        if percentage > 20:
            print(
                "\nWarning: High refusal rate detected. This may lead to over-refusal in the fine-tuned model."
            )

    except FileNotFoundError:
        print("File not found. Please check the path.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Ensure your file is in valid JSONL format.")


import argparse

parser = argparse.ArgumentParser(description="Analyze refusal rate in SFT dataset")
parser.add_argument(
    "--file", type=str, default="sft_data_test.jsonl", help="Path to the JSONL dataset file"
)
args = parser.parse_args()
analyze_refusals(args.file)

import pandas as pd
import numpy as np


# Assuming 'goals' is your list of 19,752 goal labels
# (e.g., ['Malware', 'Capybara_Task', 'Hate_Speech', ...])
def analyze_goal_distribution(goals_list):
    # Convert to Series for easy manipulation
    df = pd.Series(goals_list).value_counts().reset_index()
    df.columns = ["Goal", "Count"]

    # Calculate Stats
    mean_val = df["Count"].mean()
    std_dev = df["Count"].std()  # Dispersion
    min_val = df["Count"].min()
    max_val = df["Count"].max()
    median_val = df["Count"].median()

    print("--- Goal Distribution Statistics ---")
    print(f"Unique Goals: {len(df)}")
    print(f"Mean (Average occurrences per goal): {mean_val:.2f}")
    print(f"Dispersion (Std Dev): {std_dev:.2f}")
    print(f"Min (Least frequent goal): {min_val}")
    print(f"Max (Most frequent goal): {max_val}")
    print(f"Median: {median_val}")

    # Show the "Top Heavy" goals
    print("\n--- Top 5 Most Frequent Goals ---")
    print(df.head(5).to_string(index=False))

    # Identify Outliers
    outliers = df[df["Count"] > (mean_val + 2 * std_dev)]
    if not outliers.empty:
        print("\n--- Potential Over-represented Goals (>2 Std Dev) ---")
        print(outliers.to_string(index=False))


goals = [g for g in goals if g is not ""]  # Filter out None values if any
# Example usage:
print(f"Unique goals: {len(set(goals))}/{len(goals)}")
analyze_goal_distribution(goals)


bss_goals = [g for g in goals if "bss" in g]  # Filter out None values if any
print(f"Unique BSS goals: {len(set(bss_goals))}/{len(bss_goals)}")


lao_goals = [g for g in goals if "lao" in g]  # Filter out None values if any
print(f"Unique LAO goals: {len(set(lao_goals))}/{len(lao_goals)}")


adv_goals = [g for g in goals if "adv" in g]  # Filter out None values if any
print(f"Unique ADV goals: {len(set(adv_goals))}/{len(adv_goals)}")


xstest = [g for g in goals if "xstest" in g]  # Filter out None values if any
print(f"Unique XSTEST goals: {len(set(xstest))}/{len(xstest)}")


xstest_safe = [g for g in goals if "xstest" in g and "safe" in g]  # Filter out None values if any
print(f"Unique XSTEST Safe goals: {len(set(xstest_safe))}/{len(xstest_safe)}")

xstest_unsafe = [
    g for g in goals if "xstest" in g and "unsafe" in g
]  # Filter out None values if any
print(f"Unique XSTEST Unsafe goals: {len(set(xstest_unsafe))}/{len(xstest_unsafe)}")

goat_goals = [g for g in goals if "goat" in g]  # Filter out None values if any
print(f"Unique GOAT goals: {len(set(goat_goals))}/{len(goat_goals)}")


crescendo_goals = [g for g in goals if "crescendo" in g]  # Filter out None values if any
print(f"Unique Crescendo goals: {len(set(crescendo_goals))}/{len(crescendo_goals)}")
print(f"Empty conversations: {empty_convos}")
