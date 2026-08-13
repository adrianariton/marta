import json

files = [
    "sft_data_clean.jsonl",
    "capybara_retain.jsonl",
]

with open("sft_data_final.jsonl", "w") as out:
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")

print("Merged into sft_data_final.jsonl")


import json

files2 = [
    "sft_data_train.jsonl",
    "capybara_retain.jsonl",
]

with open("sft_data_final_train.jsonl", "w") as out:
    for file in files2:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")

print("Merged into sft_data_final_train.jsonl")
