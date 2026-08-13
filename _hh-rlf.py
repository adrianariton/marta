from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")["train"]
dataset.to_csv("hh-rlhf-red-team-attempts.csv", index=False)
