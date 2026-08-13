from core.attacks.interfaces import OneShotStrat


class Simple(OneShotStrat):
    def __init__(self, message_text):
        super().__init__()
        self.message_text = message_text

    def messages(self):
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.message_text},
                    ],
                }
            ],
        ]
