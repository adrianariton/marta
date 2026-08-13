"""
server.py — GPU server that executes lazy computation programs.

POST /api/solve_program
    body: { instructions: [...], result_id: "abc123" }
    returns: { result: <serialized output> }
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import core.function
from core.register import function_registry
from core.deserialize import execute_program, ProgramRequest, _serialize_result
import traceback

print(f"{function_registry=}")


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8888)
args = parser.parse_args()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
models = {}  # cache loaded models


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/solve_program")
async def solve_program(body: ProgramRequest):
    try:
        result = execute_program(body.instructions, body.result_id)
        return {"result": _serialize_result(result)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/hf/generate")
async def generate(modelname: str, body: GenerateRequest):
    if modelname not in models:
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
        model.to("cuda")
        models[modelname] = (tokenizer, model)

    tokenizer, model = models[modelname]

    # Use chat template if available (TinyLlama, Llama, etc.)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        # Parse history from raw prompt back into messages
        messages = []
        for line in body.prompt.strip().split("\n"):
            if line.startswith("User: "):
                messages.append({"role": "user", "content": line[6:]})
            elif line.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": line[11:]})

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = body.prompt

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=body.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the NEW tokens, not the echoed prompt
    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"generated": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
