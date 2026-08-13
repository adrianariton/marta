import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import TrainerCallback
import json

SUFFIX = "_train"  # or "_test"


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

    def save(self, path="loss_logs.json"):
        with open(path, "w") as f:
            json.dump({"train": self.train_logs, "eval": self.eval_logs}, f, indent=2)
        print(f"Logs saved to {path}")


loss_logger = LossLoggerCallback()
# ── 1. Load Dataset ───────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=f"sft_data_final{SUFFIX}.jsonl", split="train")
dataset_test = load_dataset("json", data_files=f"sft_data_test.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"] + dataset_test
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── 2. Model + Tokenizer ──────────────────────────────────────────────────────
model_id = "GraySwanAI/Mistral-7B-Instruct-RR"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ── 3. Format Dataset ─────────────────────────────────────────────────────────
def format_conversations(example):
    messages = [m for m in example["messages"] if m["role"] != "system"]
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    }


train_dataset = train_dataset.map(format_conversations, remove_columns=["messages"])
eval_dataset = eval_dataset.map(format_conversations, remove_columns=["messages"])


if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    lengths = [
        len(tokenizer(ex["text"], add_special_tokens=False)["input_ids"]) for ex in train_dataset
    ]
    print(f"Min:          {min(lengths)}")
    print(f"Mean:         {sum(lengths)/len(lengths):.0f}")
    print(f"Max:          {max(lengths)}")
    print(
        f"Over 1024:    {sum(1 for l in lengths if l > 1024)} ({100*sum(1 for l in lengths if l > 1024)/len(lengths):.1f}%)"
    )
    print(
        f"Over 2048:    {sum(1 for l in lengths if l > 2048)} ({100*sum(1 for l in lengths if l > 2048)/len(lengths):.1f}%)"
    )
    print(
        f"Over 4096:    {sum(1 for l in lengths if l > 4096)} ({100*sum(1 for l in lengths if l > 4096)/len(lengths):.1f}%)"
    )


# ── 4. Data Collator ──────────────────────────────────────────────────────────
collator = DataCollatorForCompletionOnlyLM(
    response_template="[/INST]",
    tokenizer=tokenizer,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# ── 5. SFT Config ─────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./mistral-rr-safety-sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # ← increased from 0.05
    max_grad_norm=1.0,
    max_seq_length=4096,
    packing=False,
    dataset_text_field="text",
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
    gradient_checkpointing=True,  # ← trades compute for memory
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    peft_config=peft_config,  # ← add this back
    callbacks=[loss_logger],
)

trainer.train()

# ── 7. Save ───────────────────────────────────────────────────────────────────
if trainer.is_world_process_zero():  # ← only main process saves
    loss_logger.save(f"./datasets/loss_logs{SUFFIX}.json")
trainer.save_model(f"./mistral-rr-safety-sft{SUFFIX}/final")
tokenizer.save_pretrained(f"./mistral-rr-safety-sft{SUFFIX}/final")
print(f"Done — model saved to ./mistral-rr-safety-sft{SUFFIX}/final")
