import io
import json
import base64
import numpy as np
import torch
from pydantic import BaseModel
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.register import get_seaker_registered_function


def _deser_key(key, env):
    """Reconstruct a getitem key — int, slice, ellipsis, None, tuple, or Variable ref."""
    if isinstance(key, dict):
        if "__type__" not in key:
            raise Exception(f"key {key} has no __type__ set")
        match key["__type__"]:
            case "tuple":
                return tuple(_deser_key(k, env) for k in key["items"])
            case "slice":
                return slice(
                    _deser_key(key["start"], env) if key["start"] is not None else None,
                    _deser_key(key["stop"], env) if key["stop"] is not None else None,
                    _deser_key(key["step"], env) if key["step"] is not None else None,
                )
            case "ellipsis":
                return ...
            case "none":
                return None
            case "unpack":
                return tuple(_deser(key["var"], env))
            case _:
                # fall through to _deser for variable refs etc.
                return _deser(key, env)
    return _deser(key, env)


def _deser(arg: Any, env: dict) -> Any:
    """Deserialize an argument from the instruction payload."""
    if isinstance(arg, dict):
        t = arg.get("__type__")
        if t == "var":
            return env[arg["id"]]
        if t == "val":
            return arg["value"]
        if t == "expr":
            return arg["value"]
    return arg


def solve_math(operator_str, a, b=None, reflected=False):
    if reflected and b is not None:
        a, b = b, a

    match operator_str:
        # Arithmetic
        case "+":
            return a + b
        case "-":
            return a - b
        case "*":
            return a * b
        case "/":
            return a / b
        case "//":
            return a // b
        case "%":
            return a % b
        case "**":
            return a**b

        # Unary
        case "neg":
            return -a
        case "abs":
            return abs(a)

        # Bitwise
        case "&":
            return a & b
        case "|":
            return a | b
        case "^":
            return a ^ b
        case "<<":
            return a << b
        case ">>":
            return a >> b

        # Comparison
        case "==":
            return a == b
        case "!=":
            return a != b
        case "<":
            return a < b
        case "<=":
            return a <= b
        case ">":
            return a > b
        case ">=":
            return a >= b

        case _:
            raise ValueError(f"Unknown operator: {operator_str!r}")


def get_package_by_name(pkg):
    match pkg:
        case "torch":
            import torch

            return torch
        case "transformers":
            import transformers

            return transformers
        case "sentence_transformers":
            import sentence_transformers

            return sentence_transformers
        case "spacy":
            import spacy

            return spacy
        case "sklearn":
            import sklearn

            return sklearn
        case "pydantic":
            import pydantic

            return pydantic

        case _:
            raise Exception(f"Package {pkg} not available in this runtime!")


class Scope:
    def __init__(self, parent_scope: "Scope" = None):
        self.parent_scope = parent_scope
        self.env: dict[str, Any] = {}

    def look_inparent_scope(self, vid, vname):
        if vid not in self.env:
            if self.parent_scope is None:
                return False
            else:
                return self.parent_scope.look_inparent_scope(vid, vname)
        else:
            self.env[vid] = vname
            return True

    def add(self, vid, vname):
        if self.look_inparent_scope(vid, vname):
            return
        self.env[vid] = vname

    def child_scope(self):
        return Scope(self)

    def get(self, vid):
        if vid not in self.env:
            if self.parent_scope is None:
                raise Exception(f"Variable {vid} not found.")
            else:
                return self.parent_scope.get(vid)
        else:
            return self.env.get(vid)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.add(key, value)

    def log(self):
        print(f"Scope({self.env})")


def eval_and_store_instr(instr, env: Scope = None, all_instrs: list[dict] = None):
    env: Scope = env or Scope()
    print(f"DEBUG: type of instr is {type(instr)}, value is: {instr}")
    node_id = instr["id"]
    op = instr["op"]

    print(f"[server] Executing op={op} id={node_id}")
    if op == "getattr":
        obj = env[instr["var"]]
        val = getattr(obj, instr["args"][0])
        env[node_id] = val
    elif op == "getitem":
        print(f"{instr['var']=}")
        obj = env[instr["var"]]
        key = _deser_key(instr["args"][0], env)
        env[node_id] = obj[key]

    elif op == "setitem":
        obj = env[instr["var"]]
        key = _deser_key(instr["args"][0], env)
        val = _deser(instr["args"][1], env)
        obj[key] = val
        env[node_id] = obj
    # ── call ──
    elif op == "call":
        fn = env[instr["var"]]
        args = [_deser(a, env) for a in instr.get("args", [])]
        kwargs = {k: _deser(v, env) for k, v in instr.get("kwargs", {}).items()}
        result = fn(*args, **kwargs)
        env[node_id] = result
    elif op == "registered_function_call":
        fn = get_seaker_registered_function(instr["var"])
        args = [_deser(a, env) for a in instr.get("args", [])]
        kwargs = {k: _deser(v, env) for k, v in instr.get("kwargs", {}).items()}
        result = fn(*args, **kwargs)
        env[node_id] = result
    elif op == "operator":
        operator = instr.get("operator")
        me = env[instr["var"]]
        args = [_deser(a, env) for a in instr.get("args", [])]
        kwargs = {k: _deser(v, env) for k, v in instr.get("kwargs", {}).items()}
        result = solve_math(operator, me, *args, **kwargs)
        env[node_id] = result
    elif op == "assign":
        me = instr["var"]
        args = [_deser(a, env) for a in instr.get("args", [])]
        kwargs = {k: _deser(v, env) for k, v in instr.get("kwargs", {}).items()}
        r = _deser(args[0], env)
        print(f"[server] assigning {node_id} to {r}")
        env[me] = r
    elif op == "package_import":
        package = instr["var"]
        env[node_id] = get_package_by_name(package)
    elif op == "if":
        body = instr["body"]
        conditions = body["conditions"]
        _instructions_groups = body["instructions"]
        instructions_groups = [
            _instructions_groups[f"{i}"] for i in range(len(_instructions_groups))
        ]
        assert len(conditions) == len(
            instructions_groups
        ), f"Improper if/else at: {conditions=} {instructions_groups=}"

        for cond, instrs in zip(conditions, instructions_groups):
            cond_eval = _deser(cond, env)
            print(f"[server] cond_eval for {cond} evaluated to {cond_eval}")
            assert isinstance(
                cond_eval, bool
            ), f"{cond} did not evaluate to bool but to {cond_eval}"
            if cond_eval:
                execute_program(instrs, result_id=None, env=env.child_scope())
                break
    elif op == "while":
        body = instr["body"]
        condition = body["condition"]
        instructions = body["instructions"]
        # here
        MAX_REC_DEPTH = 10000
        i = 0
        assert all_instrs is not None, "While error!"
        to_reeval = get_non_floating_frontier(cond_var=condition, instructions=all_instrs)
        print("==============reeval===============")
        print(json.dumps(to_reeval, indent=4))
        print("=============/reeval===============")

        for i in range(20):
            execute_program(to_reeval, result_id=None, env=env)
            cond_eval = _deser(condition, env)
            assert isinstance(
                cond_eval, bool
            ), f"{cond} did not evaluate to bool but to {cond_eval}"
            if cond_eval == True:
                execute_program(instructions, result_id=None, env=env.child_scope())
            else:
                break
        if MAX_REC_DEPTH - 1 == i:
            raise ValueError("Max recursion depth!")
    else:
        raise ValueError(f"Unknown op: {op}")


def is_untied(var_id: str):
    return not (":" in var_id)


def varname(var_id: str):
    if is_untied(var_id):
        return None
    return var_id.split(":")[0]


def is_cond(var_id: str):
    return varname(var_id) == "0cond"


def get_non_floating_frontier(cond_var: dict, instructions: list[dict]) -> list[dict]:
    frontier = []
    cond_var_id = cond_var["id"]
    assert is_cond(cond_var_id), "Use this function on condition variables only."
    instr_index = len(instructions) - 1
    while True:
        if instructions[instr_index].get("id") != cond_var_id:
            instr_index -= 1
        else:
            break
    var_ids_to_explore = [cond_var_id]
    while instr_index >= 0:
        current_instr = instructions[instr_index]
        if current_instr["id"] in var_ids_to_explore:
            frontier.append(current_instr)
            # i need to explore it
            args = current_instr["args"]
            main_arg_id = current_instr["var"]
            if is_cond(main_arg_id) or is_untied(main_arg_id):
                var_ids_to_explore.append(main_arg_id)
            for arg in args:
                if arg["__type__"] == "var":
                    if is_cond(arg["id"]) or is_untied(arg["id"]):
                        var_ids_to_explore.append(arg["id"])
        instr_index -= 1

    return list(reversed(frontier))


def execute_program(instructions: list[dict], result_id: str, env: Scope = None) -> Any:
    """
    Walk the instruction list, execute each op, store results in env by node id.
    Return the value at result_id.
    """
    env: Scope = env or Scope()

    for instr in instructions:
        eval_and_store_instr(instr, env, instructions)
        env.log()

    if result_id is not None:
        return env[result_id]
    return None


# ── API ───────────────────────────────────────────────────────────────────────


class ProgramRequest(BaseModel):
    instructions: list[dict]
    result_id: str


import torch
import io
import base64
import dataclasses


def _serialize_result(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{
                f.name: _serialize_result(getattr(value, f.name)) for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {k: _serialize_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_result(v) for v in value]
    if isinstance(value, torch.Tensor):
        buf = io.BytesIO()
        np.save(buf, value.cpu().float().numpy())
        return {
            "__type__": "tensor",
            "shape": list(value.shape),
            "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # fallback
    return str(value)
