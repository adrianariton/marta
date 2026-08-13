import os
from together import Together

client = Together(api_key="db41e98442edac99aacf020347865d1e6ff9ed7079fa2519abce5d1c9f10a7a2")

response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[{"role": "user", "content": "Tell me fun things to do in New York"}],
)
print(response.choices[0].message.content)
