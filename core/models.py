from torch import float16
from transformers import (
    DeepseekVLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import Mistral3ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from core.utils import get_device
import warnings


_LOADED_PEFT_MODELS = {}
_PEFT_MODELS = {}

_LOADED_MODELS = {}
_MODELS = {
    "deepseek-community/deepseek-vl-1.3b-chat": (
        lambda: DeepseekVLForConditionalGeneration.from_pretrained(
            "deepseek-community/deepseek-vl-1.3b-chat",
            dtype=float16,
            device_map={"": get_device()},
            attn_implementation="sdpa",
        )
    ),
    "GraySwanAI/Mistral-7B-Instruct-RR": lambda: AutoModelForCausalLM.from_pretrained(
        "GraySwanAI/Mistral-7B-Instruct-RR",
        dtype=float16,
        device_map={"": get_device()},
    ),
    "mistralai/Ministral-3-14B-Instruct-2512": lambda: Mistral3ForConditionalGeneration.from_pretrained(
        "mistralai/Ministral-3-14B-Instruct-2512",
        device_map={"": get_device()},
    ),
}
_LOADED_TOKENIZERS = {}
_TOKENIZERS = {
    "GraySwanAI/Mistral-7B-Instruct-RR": lambda: AutoTokenizer.from_pretrained(
        "GraySwanAI/Mistral-7B-Instruct-RR", use_fast=True
    ),
    # "mistralai/Ministral-3-14B-Instruct-2512": lambda: MistralCommonBackend.from_pretrained(
    #     "mistralai/Ministral-3-14B-Instruct-2512"
    # ),
}
_LOADED_PROCESSORS = {}
_PROCESSORS = {
    "deepseek-community/deepseek-vl-1.3b-chat": (
        lambda: AutoProcessor.from_pretrained(
            "deepseek-community/deepseek-vl-1.3b-chat", use_fast=True
        )
    ),
}


def register_model(key: str, lambda_model):
    _MODELS[key] = lambda_model


def register_tokenizer(key: str, lambda_tokenizer):
    _TOKENIZERS[key] = lambda_tokenizer


def register_processor(key: str, lambda_processor):
    _PROCESSORS[key] = lambda_processor


import threading

_MODEL_LOCK = threading.Lock()


def unload_all_models():
    with _MODEL_LOCK:

        for model in _LOADED_MODELS.values():
            model.cpu()  # move off GPU first
            del model
        _LOADED_MODELS.clear()
        _LOADED_TOKENIZERS.clear()
        _LOADED_PROCESSORS.clear()
        _LOADED_PEFT_MODELS.clear()


def get_or_load_model(name: str, **kwargs):
    with _MODEL_LOCK:  # ← prevent double-load race

        if name not in _MODELS:
            warnings.warn(
                "Unknown model: "
                + name
                + " use one of: "
                + f"{_MODELS}. Loading {name} from AutoModelForCausalLM."
            )
            _MODELS[name] = lambda: AutoModelForCausalLM.from_pretrained(
                name,
                dtype=float16,
                device_map="auto",#{"": get_device()},
                **kwargs,
            )
        if name not in _LOADED_MODELS:
            _LOADED_MODELS[name] = _MODELS[name]()
        return _LOADED_MODELS[name]


def get_or_load_peft_model(name: str, base_model, **kwargs):
    with _MODEL_LOCK:  # ← prevent double-load race
        from peft import PeftModel

        if name not in _PEFT_MODELS:
            warnings.warn(
                "Unknown model: "
                + name
                + " use one of: "
                + f"{_PEFT_MODELS}. Loading {name} from PeftModel."
            )
            _PEFT_MODELS[name] = lambda: PeftModel.from_pretrained(base_model, name, **kwargs)
        if name not in _LOADED_PEFT_MODELS:
            _LOADED_PEFT_MODELS[name] = _PEFT_MODELS[name]()
        return _LOADED_PEFT_MODELS[name]


def get_or_load_processor(name: str, **kwargs):
    with _MODEL_LOCK:  # ← prevent double

        if name not in _PROCESSORS:
            warnings.warn(
                "Unknown processor: "
                + name
                + " use one of: "
                + f"{_PROCESSORS}. Loading {name} from AutoProcessor."
            )
            _PROCESSORS[name] = lambda: AutoProcessor.from_pretrained(name, use_fast=True, **kwargs)
        if name not in _LOADED_PROCESSORS:
            _LOADED_PROCESSORS[name] = _PROCESSORS[name]()
        return _LOADED_PROCESSORS[name]


def get_or_load_tokenizer(name: str, **kwargs):
    with _MODEL_LOCK:
        if name not in _TOKENIZERS:
            warnings.warn(
                "Unknown tokenizer: "
                + name
                + " use one of: "
                + f"{_TOKENIZERS}. Loading {name} from AutoTokenizer."
            )
            _TOKENIZERS[name] = lambda: AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)

        if name not in _LOADED_TOKENIZERS:
            _LOADED_TOKENIZERS[name] = _TOKENIZERS[name]()
        return _LOADED_TOKENIZERS[name]
