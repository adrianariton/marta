from core.attacks.interfaces import TextGenerator
from abc import abstractmethod
from dataclasses import dataclass
import re
from core.attacks.utils import get_faulty_json_data
from typing import Literal
from abc import abstractmethod
from core.attacks.datastore.yielder import YielderRegistry
from core.attacks.utils import Params
from typing import Optional
from core.attacks.agents.dummy import DummyAgent
import random
from core.attacks.datastore.yieldable import Yieldable
from core.attacks.evaluators.PromptResponseEvaluator import PromptResponseEvaluator


class MSMARCOEvaluator(PromptResponseEvaluator):
    def __init__(self):
        super().__init__()

    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__


class HfMSMARCOEvaluator(MSMARCOEvaluator):
    """https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
    MSMARSC PromptResposeEvaluator using Hf ^
    """

    def __init__(self, hf_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.hf_model = hf_model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__

    def evaluate(self, q: str, r: str) -> float:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.set_params(
            Params(
                Params.PType.MSMARCOEVALUATOR,
                q=q,
                r=r,
            )
        )

        features = self.tokenizer([q], [r], padding=True, truncation=True, return_tensors="pt")

        self.model.eval()
        with torch.no_grad():
            scores = self.model(**features).logits

        return scores[0].item()

    def batch_evaluate(self, q: list[str], r: list[str]) -> list[float]:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.set_params(
            Params(
                Params.PType.MSMARCOEVALUATOR,
                q=q,
                r=r,
            )
        )

        model = AutoModelForSequenceClassification.from_pretrained(self.hf_model)
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

        features = tokenizer(q, r, padding=True, truncation=True, return_tensors="pt")

        model.eval()
        with torch.no_grad():
            scores = model(**features).logits

        return scores


class DummyMSMARCOEvaluator(MSMARCOEvaluator):
    def __init__(
        self, _type: Literal["random", "stateful"] = "random", state: Optional[list[int]] = None
    ):
        super().__init__()
        self._type = _type
        self.state = state if state is not None else []
        self.i = 0

    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__

    def evaluate(self, q, r):

        self.set_params(
            Params(
                Params.PType.MSMARCOEVALUATOR,
                q=q,
                r=r,
            )
        )
        if self._type == "random":
            return random.randint(0, 10) / 10

        elif self._type == "stateful":
            if self.i >= len(self.state):
                self.i = 0
            val = self.state[self.i]
            self.i += 1
            return val

    def batch_evaluate(self, q: list[str], r: list[str]):
        return [self.evaluate(q[i], r[i]) for i in range(len(r))]
