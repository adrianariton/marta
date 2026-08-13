import random


class QueryModifier:
    def modify(self, q: str, n: int, random_state: random.Random) -> list[str]:
        pass


class DummyQueryModifier(QueryModifier):
    def modify(self, q: str, n: int, random_state: random.Random) -> list[str]:
        return [q] * n
