import io
import json
import base64
import numpy as np
import torch
from pydantic import BaseModel
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.register import seaker_register_function
from dataclasses import dataclass

_model_cache: dict[str, tuple] = {}


@seaker_register_function
def load_hf_model(model_name: str):
    if model_name not in _model_cache:
        print(f"[server] Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        _model_cache[model_name] = (tokenizer, model)
        print(f"[server] Model {model_name} loaded.")
    return HFModelHandle(*(_model_cache[model_name]))


@dataclass
class GenerateOutput:
    text: Any
    attentions: Any


class HFModelHandle:
    """
    Wraps a (tokenizer, model) pair and exposes .generate()
    matching the lazy client's call signature.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer, self.model = tokenizer, model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> "GenerateOutput":
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        new_tokens = outputs.sequences[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerateOutput(text=text, attentions=outputs.attentions)
