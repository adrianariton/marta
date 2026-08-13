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


class StatelessSelfHostedOllamaAgent(TextGenerator):
    """
    gpt-oss:120b | llama3.3:70b | dolphin-llama3:70b
    """

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        host: str = "localhost",
        port: int = 11434,
        protocol: Literal["http", "https"] = "http",
        session=None,
        think: bool = None,
        ctx: int = 131072,
    ):
        super().__init__()
        self.system_message = "You are a helpful assistant"
        self.host = host
        self.protocol = protocol
        self.port = port
        self.url = f"{protocol}://{host}:{port}/api/chat"
        self.model = model
        self.session = session
        self.think = think
        self.ctx = ctx

    def set_system_message(self, message):
        self.system_message = (
            message if isinstance(message, Message) else Message.from_text(message)
        )

    def single_generate(self, messages) -> list[str]:
        messages.assert_ready_for_completion()
        return asyncio.run(self.asingle_generate(messages))

    def with_batch(self, batch):
        raise Exception("Ollama has native batching. Use normal async generation.")

    async def asingle_generate(self, messages: OneShotConversation) -> list[str]:
        self.session = self.session or aiohttp.ClientSession()
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()
        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_auto()
        # try:
        # print("generate", flush=True)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "num_ctx": self.ctx,  # 128k context for ollama
            },
        }
        if "gpt-oss" in self.model:
            payload["think"] = self.think
        async with self.session.post(
            self.url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            result = await resp.json()
            # print(f"{result=}", flush=True)
            return [result["message"]["content"]]

        # except Exception as e:
        #     traceback.print_exc()
        #     print(f"Error", flush=True)
        #     return [""]
        return ["(error)"]
