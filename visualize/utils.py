from openai import OpenAI
import os


class Aider:
    def __init__(self, model="gemini-3.1-flash", gkey=None):
        if (gkey or os.getenv("GOOGLE_API_KEY_FREE")) is not None:
            self.client = OpenAI(
                api_key=gkey or os.getenv("GOOGLE_API_KEY_FREE"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            self.client = None
        self.model = model

    def aid_me_please(self, system, query, model=None):
        if self.client is None:
            return "Unable to aid!"
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
        )

        return response.choices[0].message.content
