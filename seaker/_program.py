"""
client.py — Lazy computation graph client.

Usage (sync):
    with Program() as program:
        model = get_hf_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        output = model.generate(prompt="Hello!", max_new_tokens=50)
        attentions = output.attentions
        pca = PCA(n_components=2)
        attentions_pca = pca(attentions)
        result = program.get(attentions_pca)

Usage (async):
    async with Program() as program:
        model = get_hf_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        output = model.generate(prompt="Hello!", max_new_tokens=50)
        attentions = output.attentions
        pca = PCA(n_components=2)
        attentions_pca = pca(attentions)
        result = await program.aget(attentions_pca)
"""

import json
import uuid
import asyncio
import requests
import aiohttp
from typing import Any, Optional

SERVER_URL = "http://localhost:8888"

# ── Active program context ─────────────────────────────────────────────────────

_active_program: Optional["Program"] = None


def _get_program() -> "Program":
    if _active_program is None:
        raise RuntimeError("Must be used inside `with Program() as program:`")
    return _active_program


# ── Variable ───────────────────────────────────────────────────────────────────


class Variable:
    """
    A lazy node in the computation graph.
    Does not hold a value — just records operations.
    """

    def __init__(self, node_id: str, program: "Program"):
        object.__setattr__(self, "_id", node_id)
        object.__setattr__(self, "_program", program)

    def __getattr__(self, name: str) -> "Variable":
        if name.startswith("_"):
            raise AttributeError(name)
        program = object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "getattr",
                "var": node_id,
                "attr": name,
            }
        )
        return Variable(new_id, program)

    def __call__(self, *args, **kwargs) -> "Variable":
        program = object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "call",
                "var": node_id,
                "args": [_ser(a) for a in args],
                "kwargs": {k: _ser(v) for k, v in kwargs.items()},
            }
        )
        return Variable(new_id, program)

    def __repr__(self):
        node_id = object.__getattribute__(self, "_id")
        return f"Variable(id={node_id})"


# ── Expression ─────────────────────────────────────────────────────────────────


class Expression:
    """Wrap a plain python value to pass explicitly as a lazy arg."""

    def __init__(self, value: Any):
        self.value = value


# ── Program ───────────────────────────────────────────────────────────────────


class Program:
    """
    Collects instructions lazily.
    Supports both sync (get) and async (aget) execution.
    Also works as both sync and async context manager.
    """

    def __init__(self, server_url: str = SERVER_URL):
        self._instructions: list[dict] = []
        self._server_url = server_url

    # ── sync context manager ──
    def __enter__(self) -> "Program":
        global _active_program
        self._instructions = []
        _active_program = self
        return self

    def __exit__(self, *args):
        global _active_program
        _active_program = None

    # ── async context manager ──
    async def __aenter__(self) -> "Program":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    # ── instruction graph ──
    def _add(self, instruction: dict) -> str:
        node_id = uuid.uuid4().hex[:8]
        self._instructions.append({"id": node_id, **instruction})
        return node_id

    def _build_payload(self, var: Variable) -> dict:
        node_id = object.__getattribute__(var, "_id")
        return {
            "instructions": self._instructions,
            "result_id": node_id,
        }

    @staticmethod
    def _parse_response(data: dict) -> Any:
        if "error" in data:
            raise RuntimeError(f"Server error: {data['error']}")
        return data["result"]

    # ── sync execution ──
    def get(self, var: Variable, timeout: int = 300) -> Any:
        """Blocking — sends program to server and returns result."""
        payload = self._build_payload(var)
        print(f"[client] Sending {len(self._instructions)} instructions (sync)...")

        resp = requests.post(
            f"{self._server_url}/api/solve_program",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    # ── async execution ──
    async def aget(self, var: Variable, timeout: int = 300) -> Any:
        """Non-blocking — sends program to server and returns result."""
        payload = self._build_payload(var)
        print(f"[client] Sending {len(self._instructions)} instructions (async)...")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._server_url}/api/solve_program",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return self._parse_response(data)

    def debug(self) -> str:
        return json.dumps(self._instructions, indent=2)


# ── Lazy constructors ─────────────────────────────────────────────────────────


def get_hf_model(model_name: str) -> Variable:
    program = _get_program()
    node_id = program._add(
        {
            "op": "load_hf_model",
            "model_name": model_name,
        }
    )
    return Variable(node_id, program)


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit_transform(self, var: Variable) -> Variable:
        program = _get_program()
        node_id = object.__getattribute__(var, "_id")
        new_id = program._add(
            {
                "op": "pca_fit_transform",
                "var": node_id,
                "n_components": self.n_components,
            }
        )
        return Variable(new_id, program)

    def __call__(self, var: Variable) -> Variable:
        return self.fit_transform(var)


# ── Serialization helpers ─────────────────────────────────────────────────────


def _ser(arg: Any) -> Any:
    if isinstance(arg, Variable):
        return {"__type__": "var", "id": object.__getattribute__(arg, "_id")}
    if isinstance(arg, Expression):
        return {"__type__": "expr", "value": arg.value}
    return {"__type__": "val", "value": arg}


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── sync ──
    print("=== SYNC ===")
    with Program() as program:
        model = get_hf_model("gpt2")
        output = model.generate(
            prompt="What is the capital of France?",
            max_new_tokens=50,
            temperature=0.7,
        )
        text = output.text
        print(program.debug())
        result = program.get(text)
        print("Result:", result)
