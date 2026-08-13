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

# Define the settings to disable all filtering
default_safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    # HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,  # For newer models
}

default_safety_settings__list = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def to_genai_contents(messages: list[dict]) -> list[types.Content]:
    return [
        types.Content(
            role=msg["role"],
            parts=[types.Part(text=p) if isinstance(p, str) else p for p in msg["parts"]],
        )
        for msg in messages
    ]


class StatelessRemoteGeminiAgent(TextGenerator):
    @staticmethod
    def configure(api_key=None):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or api_key)
        return os.getenv("GOOGLE_API_KEY") or api_key

    def __init__(
        self, model_name="gemini-2.5-flash", max_tokens=2048, max_retries=3, base_delay=1.5
    ):
        self.api_key = StatelessRemoteGeminiAgent.configure()
        self.model = genai.GenerativeModel(model_name, safety_settings=default_safety_settings)
        self.model_name = model_name
        self.system_message = None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_tokens = max_tokens
        super().__init__()
        self.client = gen2ai.Client(api_key=self.api_key)

    def set_system_message(self, message):
        self.system_message = (
            message if isinstance(message, Message) else Message.from_text(message)
        )
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=self.system_message.to_string(),
            safety_settings=default_safety_settings,
        )

    def single_generate(self, messages) -> list[str]:
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()
        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_gemini()
        response = self.model.generate_content(messages)
        if response.candidates and response.candidates[0].content.parts:
            return [response.text]
        else:
            if not response.candidates:
                print(
                    "DEBUG: No candidates returned. The prompt or response was likely blocked by safety filters."
                )
                return [""]
            # Log the finish reason and safety ratings for your research data
            print(f"DEBUG: Response blocked. Reason: {response.candidates[0].finish_reason}")
            print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
            return [""]
        # return [response.text]

    async def asingle_generate(self, messages) -> list[str]:
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()
        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_gemini()
        sysins = (
            self.get_system_message().to_string()
            if self.get_system_message()
            else "You are a helpful research assistant."
        )
        max_retries = self.max_retries
        base_delay = self.base_delay
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=to_genai_contents(messages),
                    config=types.GenerateContentConfig(
                        system_instruction=sysins,
                        safety_settings=default_safety_settings__list,
                        max_output_tokens=self.max_tokens,
                    ),
                )

                candidate = response.candidates[0] if response.candidates else None

                if candidate and candidate.content and candidate.content.parts:
                    return [response.text]
                elif not candidate:
                    print("DEBUG: No candidates returned. Likely blocked by safety filters.")
                else:
                    print(f"DEBUG: Response blocked. Reason: {candidate.finish_reason}")
                    print(f"Safety Ratings: {candidate.safety_ratings}")

                return [""]

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        wait_match = re.search(r"retry in ([\d\.]+)s", err_str)
                        suggested_delay = float(wait_match.group(1)) if wait_match else 25.0
                        delay = suggested_delay + random.uniform(0.5, 1.5)

                        print(
                            f"[Gemini 429] Quota exceeded. Retrying in {delay:.2f}s...", flush=True
                        )
                        await asyncio.sleep(delay)
                        continue
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"[Gemini 503] attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"[Gemini 503] all {max_retries} attempts exhausted")
                else:
                    print(f"DEBUG: Unexpected error: {e}")
                return [""]
        return [""]
