from core.attacks.interfaces import OneShotStrat


class Default(OneShotStrat):
    def __init__(self, messages):
        super().__init__()
        self._messages = messages

    def messages(self):
        return self._messages
