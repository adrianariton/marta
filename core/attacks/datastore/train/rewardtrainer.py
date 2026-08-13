import uuid
import gc
import torch.cuda
import json
import uuid
from pathlib import Path
from datetime import datetime
from core.attacks.interfaces import MessageLogger, MessageType, MsgFormat
from core.attacks.interfaces import Tag, Conversation
from typing import Callable, Optional, Literal
from trl import DPOTrainer, DPOConfig
import warnings
from core.attacks.agents import AutoAgent
from core.attacks.datastore.yielder import _Yielder
from datasets import Dataset
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


class RewardWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = inputs.pop("reward").to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss = loss.view(labels.shape)
        loss = loss.mean(dim=1)  # loss per sample
        loss = loss * rewards  # 🔥 weight by reward
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


class TrainerWrapper:
    """
    Dataset format:
    list[{
        "conversation": list[{"role": str, "content": str}],
        "reward": float
    }]
    """

    def __init__(
        self,
        tokenizer,
        model_name: Optional[str],
        dataset: List[Dict],
        lora_config: Optional[LoraConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name

        self.lora_config = lora_config or LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        self.dataset = Dataset.from_list(dataset)

    # -------- formatting --------

    def format_example(self, ex):
        text = self.tokenizer.apply_chat_template(
            ex["conversation"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "text": text,
            "reward": float(ex["reward"]),
        }

    # -------- tokenization --------

    def tokenize(self, ex):
        out = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=2048,
        )
        out["labels"] = out["input_ids"].copy()
        out["reward"] = ex["reward"]
        return out

    # -------- main entry --------

    def run_once(self, output_dir: str = "./lora-weighted-sft"):
        ds = self.dataset.map(self.format_example)

        r = np.array(ds["reward"], dtype=np.float32)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        ds = ds.remove_columns("reward").add_column("reward", r.tolist())

        ds = ds.map(self.tokenize, remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            device_map="auto",
        )
        model = get_peft_model(model, self.lora_config)

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            learning_rate=5e-5,
            num_train_epochs=2,
            fp16=True,
            logging_steps=1,
            save_steps=50,
            report_to="none",
        )

        trainer = RewardWeightedTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        return model

    def run_once_with_model(
        self,
        model,
        output_dir: str = "./lora-weighted-sft-continued",
    ):
        # 1️⃣ format
        ds = self.dataset.map(self.format_example)

        # 2️⃣ normalize reward
        r = np.array(ds["reward"], dtype=np.float32)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        ds = ds.remove_columns("reward").add_column("reward", r.tolist())

        # 3️⃣ tokenize
        ds = ds.map(self.tokenize, remove_columns=["text"])

        # 4️⃣ training args
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            learning_rate=3e-5,  # ⬇️ puțin mai mic
            num_train_epochs=1,  # ⬇️ mai puțin
            fp16=True,
            logging_steps=1,
            save_steps=50,
            report_to="none",
        )

        trainer = RewardWeightedTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        return model
