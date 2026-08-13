from core.attacks.utils import Params
from typing import Optional


class Yieldable:
    def __init__(self):
        self.last_call_params: Optional[Params] = None
        self.what_to_yield = {}

    def set_params(self, params: Params):
        self.last_call_params = params

    def get_last_call_params(self) -> Optional[Params]:
        return self.last_call_params

    def _reset_yield(self):
        self.what_to_yield = {}

    def yield_or_log(self, **kwargs):
        for k, w in kwargs.items():
            self.what_to_yield[k] = w

    def _get_yield(self) -> dict:
        return self.what_to_yield
