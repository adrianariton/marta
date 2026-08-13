import asyncio
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tempfile
from core.attacks.agents.batch import Batch
from core.attacks.agents import StatelessSelfHostedVLLMAgent
from core.attacks.evaluators import AgentPEC7Evaluator, PEC7, DummyPEC7Evaluator
from core.attacks.interfaces import LinearHistory, OneShotConversation, Conversation


EVALUATOR_DEFS = {
    "phi-4-AWQ":                       (8001, "/models/phi-4-AWQ",                       ["<|end|>", "<|endoftext|>"]),
    "Mixtral-8x22B-Instruct-v0.1-AWQ":(8002, "/models/Mixtral-8x22B-Instruct-v0.1-AWQ",["</s>"]),
    "Qwen2.5-32B-Instruct-AWQ":       (8003, "/models/Qwen2.5-32B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
    "Qwen2.5-72B-Instruct-AWQ":       (8014, "/models/Qwen2.5-72B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
}


mixtral  = StatelessSelfHostedVLLMAgent(
            host="localhost",
            port=8002,
            model="/models/Mixtral-8x22B-Instruct-v0.1-AWQ",
            temperature=0.7,
            max_tokens=1024,
            stop=["</s>"] + ["User:", "Assistant:"],
            presence_penalty=0.0,
            repetition_penalty=1.1,
            frequency_penalty=0.0,
            top_p=0.9,
            default_system_message=False
            # structured_outputs={"json": PEC7.PEC7.schema()},
            # structured_outputs=structured_outputs,

        )

convo = Conversation.from_system_and_user("Hello bro!")

print(convo.latest_oneshot().to_auto())

print("===================")


async def main():
    response = await mixtral.agenerate(convo.latest_oneshot())
    print(response)


if __name__ == "__main__":
    asyncio.run(main())