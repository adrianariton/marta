import os
from together import Together
from core.attacks.interfaces import TextGenerator

from core.message_formatters import (
    default_to_gemini,
)
from core.attacks.conversation import OneShotConversation


class StatelessRemoteTogetherAgent(TextGenerator):
    @staticmethod
    def configure(api_key=None):
        return Together(api_key=os.getenv("TOGETHER_API_KEY") or api_key)

    def __init__(
        self,
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        temperature: int = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        self.client = StatelessRemoteTogetherAgent.configure()
        self.model_name = model_name
        self.system_message = None
        self.kwargs = kwargs
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__()

    def set_system_message(self, message):
        self.system_message = message

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()

        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_gemini()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs
        )

        return [response.choices[0].message.content]
