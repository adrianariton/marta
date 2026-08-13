import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from transformers import TrainerCallback, EarlyStoppingCallback
import json


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = state.epoch
        if "loss" in logs:
            self.train_logs.append({"step": step, "epoch": epoch, "loss": logs["loss"]})
        if "eval_loss" in logs:
            self.eval_logs.append({"step": step, "epoch": epoch, "eval_loss": logs["eval_loss"]})

    def save(self, path="./datasets/dpo_loss_logs.json"):
        with open(path, "w") as f:
            json.dump({"train": self.train_logs, "eval": self.eval_logs}, f, indent=2)
        print(f"Logs saved to {path}")


loss_logger = LossLoggerCallback()

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files="dpo_data_clean.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── 2. Model + Tokenizer ──────────────────────────────────────────────────────
model_id = "GraySwanAI/Mistral-7B-Instruct-RR"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# DPO needs a frozen reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ── 3. Format Dataset ─────────────────────────────────────────────────────────
# DPOTrainer expects "prompt", "chosen", "rejected" as plain strings.
# We apply the chat template to each field separately.
def format_dpo_example(example):
    # Build the prompt string: full conversation up to (not including) the final assistant turn
    prompt_str = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,  # ends with [/INST] so the model knows to continue
    )
    # chosen / rejected are single-turn assistant message lists
    chosen_str = tokenizer.apply_chat_template(
        example["chosen"],
        tokenize=False,
        add_generation_prompt=False,
    )
    rejected_str = tokenizer.apply_chat_template(
        example["rejected"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "prompt": prompt_str,
        "chosen": chosen_str,
        "rejected": rejected_str,
    }


train_dataset = train_dataset.map(format_dpo_example, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_dpo_example, remove_columns=eval_dataset.column_names)

# ── 4. LoRA Config ────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# ── 5. DPO Config ─────────────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir="./mistral-rr-safety-dpo",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    learning_rate=5e-7,  # DPO typically needs a lower LR than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    max_length=4096,  # total prompt + response length
    max_prompt_length=2048,  # max tokens kept from the prompt side
    beta=0.2,  # bumped from 0.1 — small dataset, stay closer to ref
    loss_type="sigmoid",  # standard DPO loss
    eval_strategy="steps",
    eval_steps=30,
    save_strategy="steps",
    save_steps=30,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    bf16=True,
    report_to="none",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[loss_logger, EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# ── 7. Save ───────────────────────────────────────────────────────────────────
if trainer.is_world_process_zero():
    loss_logger.save("./datasets/dpo_loss_logs.json")
trainer.save_model("./mistral-rr-safety-dpo/final")
tokenizer.save_pretrained("./mistral-rr-safety-dpo/final")
print("Done — model saved to ./mistral-rr-safety-dpo/final")
