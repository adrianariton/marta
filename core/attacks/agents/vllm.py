import google.generativeai as genai
import os
from google import genai as gen2ai
from google.genai import types
import random
import asyncio
from core.message_formatters import (
    default_to_gemini,
)
import re
from core.attacks.interfaces import TextGenerator
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from core.attacks.conversation import Message, OneShotConversation
import asyncio
import aiohttp
import pandas as pd
from typing import Literal
import traceback


class StatelessSelfHostedVLLMAgent(TextGenerator):
    """
    gpt-oss:120b | llama3.3:70b | dolphin-llama3:70b | /models/Qwen3.5-27B
    """
    
    @property
    def identifier(self):
        return super().identifier + "::" + self.model

    def __init__(
        self,
        model: str = "/models/Qwen2.5-72B-Instruct",
        host: str = "localhost",
        port: int = 8001,
        protocol: Literal["http", "https"] = "http",
        session=None,
        think: bool = None,
        max_new_tokens: int = None,
        default_system_message: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.system_message = "You are a helpful assistant" if default_system_message else None
        self.host = host
        self.protocol = protocol
        self.port = port
        self.url = f"{protocol}://{host}:{port}/v1/chat/completions"
        self.model = model
        self.session = session
        self.think = think
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs

    def set_system_message(self, message):
        self.system_message = (
            message if isinstance(message, Message) else Message.from_text(message)
        )

    def single_generate(self, messages) -> list[str]:
        messages.assert_ready_for_completion()
        return asyncio.run(self.asingle_generate(messages))

    # def with_batch(self, batch):
    #     raise Exception("VLLM has native batching. Use normal async generation.")

    async def asingle_generate(self, messages: OneShotConversation) -> list[str]:
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_auto()

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            **self.kwargs,
        }

        if self.max_new_tokens is not None:
            payload["max_tokens"] = self.max_new_tokens

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as resp:
                        result = await resp.json()
                        if not ("choices" in result):
                            print(f"[FAILED] [ERROR] {result=}")
                        return [result["choices"][0]["message"]["content"]]
            except (aiohttp.ClientOSError, aiohttp.ClientConnectionResetError):
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

        return ["(error)"]

    async def abatch_generate(self, convos: list[OneShotConversation]) -> list[str]:
        tasks = [self.asingle_generate(messages) for messages in convos]
        arr = await asyncio.gather(*tasks)
        return [a[0] for a in arr]
