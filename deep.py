from google import genai
import os

client = genai.Client(api_key="AIzaSyDF13a-bPfjhpLXP0w2IMnXCfJ8PCDUNbc")
print(client.models.list().page)
response = client.models.generate_content(
    model="gemini-1.5-pro", contents="Explain post-dominance in compiler control-flow graphs."
)

print(response.text)
