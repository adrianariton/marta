from core.attacks.utils import Params
from typing import Optional
from abc import abstractmethod
from core.attacks.datastore.yielder import YielderRegistry


class RefusalClassifier:
    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__

    def get_last_call_params(self) -> Optional[Params]:
        return None

    @YielderRegistry.register
    @abstractmethod
    def is_refused(self, r: str) -> bool:
        pass

    async def ais_refused(self, r: str) -> bool:
        pass

    def set_agent_history(self, msgs: list[str]):
        pass
