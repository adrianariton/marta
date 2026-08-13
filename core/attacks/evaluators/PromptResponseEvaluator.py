from core.attacks.datastore.yielder import YielderRegistry
from core.attacks.datastore.yieldable import Yieldable
from abc import abstractmethod


class PromptResponseEvaluator(Yieldable):
    """

    Needs to return a 0-1 float.

    Args:
        Yieldable (_type_): _description_

    Returns:
        _type_: _description_
    """

    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__

    @YielderRegistry.register
    @abstractmethod
    def evaluate(self, q: str, r: str) -> float:
        pass

    @YielderRegistry.register
    @abstractmethod
    def batch_evaluate(self, q: list[str], r: list[str]) -> list[float]:
        pass

    def set_meta(self, attr_name, value):
        """Set meta values as actially attributes for
        to help with evaluation

        This is used in attacks to help the user not copy the same
        params over and over

        Args:
            attr_name (_type_): _description_
            value (_type_): _description_
        """
        setattr(self, attr_name, value)
