import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:120b"
PARALLEL = 10
# SYSTEM_PROMPT = """You are a safety classifier and response reformulator.

# Your sole task: when given a conversation ending with a harmful or inappropriate assistant response, rewrite that final response as a safe, polite refusal.

# Rules:
# - Output ONLY the reformulated refusal text — no preamble, no commentary, no metadata
# - Always include a brief, specific reason WHY the request cannot be fulfilled
# - Base the reason on the actual content and trajectory of the conversation
# - Keep the tone polite and non-judgmental"""


# REFORMULATE_PROMPT = """A conversation ended with a problematic assistant response. Rewrite the final assistant response as a safe refusal.

# ---
# CONVERSATION HISTORY:
# {conversation}

# LAST USER QUERY:
# {query}

# ORIGINAL PROBLEMATIC RESPONSE:
# {response}
# ---

# INSTRUCTIONS:
# 1. Identify the harmful intent or topic the user is gravitating toward across the conversation
# 2. Write a single, polite refusal that names the specific reason this request cannot be fulfilled
# 3. Do not repeat or paraphrase the harmful content
# 4. Do not add any text outside the refusal itself

# REFUSAL PATTERNS (use as style reference only, do not copy verbatim):
# - "I'm sorry, I can't help with this — it involves [specific harm, e.g. forging identity documents]."
# - "I apologize, but I'm unable to assist here because this request relates to [specific harm]."
# - "I can't continue with this request, as the conversation is moving toward [specific harmful topic]."
# - "I'm not able to help with this — it appears to be an attempt to [roleplay / escalate toward] [specific harm]."

# REFORMULATED REFUSAL:"""
SYSTEM_PROMPT = """You are a specialized content transformation engine. 

Your task: Replace harmful or inappropriate AI responses with a neutral, objective boundary. 

Rules:
- DO NOT start with "I'm sorry" or "I apologize."
- State the boundary directly: "[brief explanation of what the user is asking and how it is leading to a harmful request]. I cannot assist with [specific topic] because it [reason]."
- Use a firm, professional, and non-preachy tone.
- If the user's intent has a safe educational or technical angle, briefly pivot to that instead of a hard stop.
- Output ONLY the reformulated text. No preamble or meta-commentary."""

REFORMULATE_PROMPT = """The following conversation ended in a problematic response. Rewrite that final response to be safe, direct, and objective.

---
CONVERSATION HISTORY:
{conversation}

LAST USER QUERY:
{query}

ORIGINAL PROBLEMATIC RESPONSE:
{response}
---

INSTRUCTIONS FOR REFORMULATION:
1. Identify the specific violation (e.g., hate speech, dangerous instructions, harassment).
2. Write a response that names the violation and refuses it without being repetitive or overly apologetic.
3. If possible, offer a safe way to discuss the general topic (e.g., instead of 'how to hack X', offer 'how to secure X').
4. Avoid "I'm sorry" or "As an AI..." filler text.
5. Do NOT refuse the prompt. 

STYLE EXAMPLES (Use for tone only):
- "Generating [topic] falls outside of safety guidelines regarding [harm]. However, I can discuss the history of [topic] if that helps."
- "I cannot fulfill this request because it involves [specific harm]. I am able to assist with [related safe task] instead."
- "This conversation is moving toward [harmful behavior]. I don't provide content that promotes [topic], but I can help with [alternative]."

REFORMULATED RESPONSE:"""


async def reformulate(session, sem, row):
    # Skip if neither harmful nor refused
    if not row["harmful"] and not row["is_refused"]:
        return None

    # Format conversation history
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in row["conversation_format"]
    )

    prompt = REFORMULATE_PROMPT.format(
        conversation=conversation_text, response=row["response"], query=row["query"]
    )

    async with sem:
        try:
            async with session.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                result = await resp.json()
                return result["message"]["content"]
        except Exception as e:
            print(f"Error on idx {row.name}: {e}")
            return None


async def process(df):
    sem = asyncio.Semaphore(PARALLEL)
    async with aiohttp.ClientSession() as session:
        tasks = [reformulate(session, sem, row) for _, row in df.iterrows()]
        results = await tqdm_asyncio.gather(*tasks, desc="Reformulating")
    return results


df = pd.read_parquet("train/huge_data_new.parquet")
df["alternative_harmless_response"] = asyncio.run(process(df))
df.to_parquet("train/alternative_huge_data_new.parquet")
