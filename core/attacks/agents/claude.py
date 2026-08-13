import anthropic
import os

from core.message_formatters import default_to_anthropic  # You may need to implement this
from core.attacks.interfaces import TextGenerator
from core.attacks.conversation import OneShotConversation, Message


class StatelessRemoteClaudeAgent(TextGenerator):
    @staticmethod
    def configure(api_key=None):
        # The Anthropic client reads ANTHROPIC_API_KEY from env by default
        # but we store it for explicit instantiation if provided
        StatelessRemoteClaudeAgent._api_key = os.getenv("ANTHROPIC_API_KEY") or api_key

    def __init__(self, model_name="claude-sonnet-4-20250514", max_tokens: int = 1024):
        StatelessRemoteClaudeAgent.configure()
        self.client = anthropic.Anthropic(
            api_key=getattr(StatelessRemoteClaudeAgent, "_api_key", None)
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=getattr(StatelessRemoteClaudeAgent, "_api_key", None)
        )
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.system_message = None
        super().__init__()

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        # Normalize messages to Anthropic format: list of {"role": "user"/"assistant", "content": str}

        messages.assert_ready_for_completion()

        system_prompt = messages.get_system_message()
        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_anthropic()
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if self.get_system_message():
                kwargs["system"] = self.get_system_message().to_string()

            response = self.client.messages.create(**kwargs)

            # Extract text from the first content block
            if response.content and len(response.content) > 0:
                return [response.content[0].text]
            else:
                print(f"DEBUG: Empty response. Stop reason: {response.stop_reason}")
                return [""]

        except anthropic.APIStatusError as e:
            print(
                f"DEBUG: Request blocked or failed. Status: {e.status_code}, Message: {e.message}"
            )
            return [""]
        except Exception as e:
            print(f"DEBUG: Unexpected error: {e}")
            return [""]

    async def asingle_generate(self, messages: OneShotConversation) -> list[str]:
        # TODO:
        messages.assert_ready_for_completion()

        system_prompt = messages.get_system_message()
        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())
        messages = messages.to_anthropic()
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if self.get_system_message():
                kwargs["system"] = self.get_system_message().to_string()

            response = await self.async_client.messages.create(**kwargs)

            # Extract text from the first content block
            if response.content and len(response.content) > 0:
                return [response.content[0].text]
            else:
                print(f"DEBUG: Empty response. Stop reason: {response.stop_reason}")
                return [""]

        except anthropic.APIStatusError as e:
            print(
                f"DEBUG: Request blocked or failed. Status: {e.status_code}, Message: {e.message}"
            )
            return [""]
        except Exception as e:
            print(f"DEBUG: Unexpected error: {e}")
            return [""]
