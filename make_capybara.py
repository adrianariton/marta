from datasets import load_dataset
import json

# ── 1. Load Capybara ──────────────────────────────────────────────────────────
dataset = load_dataset("LDJnr/Capybara", split="train")
print(f"Loaded {len(dataset)} conversations")


# ── 2. Convert to our format ──────────────────────────────────────────────────
def convert(example):
    """
    Capybara has a 'conversation' field with 'input' and 'output' keys.
    Convert to our messages format.
    """
    messages = []
    for turn in example["conversation"]:
        messages.append({"role": "user", "content": turn["input"]})
        messages.append({"role": "assistant", "content": turn["output"]})
    return {"messages": messages}


# ── 3. Filter out any empty or malformed conversations ────────────────────────
dataset = dataset.filter(lambda x: len(x["conversation"]) > 0)

# ── 4. Convert ────────────────────────────────────────────────────────────────
converted = dataset.map(convert, remove_columns=dataset.column_names)
print(f"Converted {len(converted)} conversations")

# ── 5. Take a subset of 20000 ──────────────────────────────────────────────────
subset = converted.shuffle(seed=42).select(range(min(20000, len(converted))))
print(f"Subset: {len(subset)} conversations")

# ── 6. Save to JSONL ──────────────────────────────────────────────────────────
output_file = "capybara_retain.jsonl"
with open(output_file, "w") as f:
    for example in subset:
        f.write(json.dumps(example) + "\n")

print(f"Saved to {output_file}")
