from core.attacks.interfaces import TextGenerator, text_generator_gpu_lock
from core.attacks.conversation import OneShotConversation
import requests
import httpx
import asyncio


class HumanInTheLoopAgent(TextGenerator):
    def __init__(self, remote_server="http://localhost:6767/get-input"):
        self.remote_server = remote_server
        super().__init__()

    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__

    def ask_for_input(self, messages: OneShotConversation) -> str:
        if self.remote_server:
            # timeout after 20 minutes to prevent hanging indefinitely
            response = (
                requests.post(
                    self.remote_server,
                    json={"messages": messages.to_auto(), "id": messages._uuid},
                    timeout=1200,
                )
                .json()
                .get("response", "")
            )
        else:
            response = input(f"{messages.to_auto()}\n\nYour response: >")
        return response

    async def _async_ask_for_input(self, messages: OneShotConversation) -> str:
        """The actual async logic."""
        if not self.remote_server:
            # input() is blocking; in a true async app,
            # you'd use aioconsole, but for a simple script:
            return input(f"{messages.to_auto()}\n\nYour response: >")

        # Using a context manager for the client is more efficient
        async with httpx.AsyncClient() as client:
            try:
                # Use timeout=None for truly indefinite waiting,
                # or a high number for your 20-minute limit.
                response = await client.post(
                    self.remote_server,
                    json={"messages": messages.to_auto(), "id": messages._uuid},
                    timeout=1200.0,
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except httpx.TimeoutException:
                return "[Error: Human took too long to respond]"
            except Exception as e:
                return f"[Error: {str(e)}]"

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        response = self.ask_for_input(messages)
        return [response]

    async def asingle_generate(self, messages: OneShotConversation) -> list[str]:
        response = await self._async_ask_for_input(messages)
        return [response]
