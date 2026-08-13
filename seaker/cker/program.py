from typing import Any

import json
import uuid
import asyncio
import requests
import aiohttp
from typing import Any, Optional

import builtins

_active_programs: list["Program"] = None


def _get_program() -> "Program":
    global _active_programs
    if _active_programs is None or _active_programs == []:
        raise RuntimeError("Must be used inside `with Program() as program:`")
    return _active_programs[-1]


def push_active_program(prog: "Program"):
    print(f"Pushing: {prog}")
    global _active_programs
    if _active_programs is None:
        _active_programs = [prog]
    else:
        _active_programs.append(prog)


def pop_active_program(prog: "Program"):
    print(f"Popping: {prog}")
    if _active_programs is None or _active_programs == []:
        raise Exception("No program is active and 'pop_active_program' was called")
    else:
        p = _active_programs.pop()
        assert p == prog
        return p


class UnpackArgs:
    """Wraps a Variable to signal *unpacking"""

    def __init__(self, var):
        self.var = var


class UnpackIndex:
    """Wraps a Variable to signal unpacking inside [...] — x[*variable]"""

    def __init__(self, var):
        self.var = var


class UnpackKwargs:
    """Wraps a Variable to signal **unpacking"""

    def __init__(self, var):
        self.var = var


class Constant:
    """Wrap a plain python value to pass explicitly as a lazy arg."""

    def __init__(self, value: Any):
        self.value = value


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
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add({"op": "getattr", "var": node_id, "args": [name], "kwargs": {}})
        return Variable(new_id, program)

    def __getitem__(self, key) -> "Variable":
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "getitem",
                "var": node_id,
                "args": [_ser_key(key)],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)

    def __setitem__(self, key, value) -> "Variable":
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        program._add(
            {
                "op": "setitem",
                "var": node_id,
                "args": [_ser_key(key), _ser(value)],
                "kwargs": {},
            }
        )

    def __call__(self, *args, **kwargs) -> "Variable":
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "call",
                "var": node_id,
                "args": [_ser(a) for a in args],
                "kwargs": {k: _ser(v) for k, v in kwargs.items()},
                "unpack_args": any(isinstance(a, UnpackArgs) for a in args),  # *args
                "unpack_kwargs": any(
                    isinstance(v, UnpackKwargs) for v in kwargs.values()
                ),  # **kwargs
            }
        )
        return Variable(new_id, program)

    def _math_op(self, other, operator: str, **kwargs) -> "Variable":
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        # print(f"{program=}")
        new_id = program._add(
            {
                "op": "operator",
                "operator": operator,
                "var": node_id,
                "args": [_ser(other)],
                "kwargs": {},
            },
            **kwargs,
        )
        return Variable(new_id, program)

    def _rmath_op(self, other, operator: str) -> "Variable":
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": operator,
                "var": node_id,
                "args": [_ser(other)],
                "kwargs": {"reflected": True},
            }
        )
        return Variable(new_id, program)

    def __repr__(self):
        node_id = object.__getattribute__(self, "_id")
        return f"Variable(id={node_id})"

    # Arithmetic
    def __add__(self, other):
        return self._math_op(other, "+")

    def __radd__(self, other):
        return self._rmath_op(other, "+")

    def __iadd__(self, other):
        self <<= self + other
        return self

    def __sub__(self, other):
        return self._math_op(other, "-")

    def __rsub__(self, other):
        return self._rmath_op(other, "-")

    def __isub__(self, other):
        self <<= self - other
        return self

    def __mul__(self, other):
        return self._math_op(other, "*")

    def __rmul__(self, other):
        return self._rmath_op(other, "*")

    def __imul__(self, other):
        self <<= self * other
        return self

    def __truediv__(self, other):
        return self._math_op(other, "/")

    def __rtruediv__(self, other):
        return self._rmath_op(other, "/")

    def __itruediv__(self, other):
        self <<= self / other
        return self

    def __floordiv__(self, other):
        return self._math_op(other, "//")

    def __rfloordiv__(self, other):
        return self._rmath_op(other, "//")

    def __ifloordiv__(self, other):
        self <<= self // other
        return self

    def __mod__(self, other):
        return self._math_op(other, "%")

    def __rmod__(self, other):
        return self._rmath_op(other, "%")

    def __imod__(self, other):
        self <<= self % other
        return self

    def __pow__(self, other):
        return self._math_op(other, "**")

    def __rpow__(self, other):
        return self._rmath_op(other, "**")

    def __contains__(self, item):
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": "contains",
                "var": node_id,
                "args": [_ser(item)],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)

    # Unary
    def __neg__(self):
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": "neg",
                "var": node_id,
                "args": [],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)

    def __abs__(self):
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(self, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": "abs",
                "var": node_id,
                "args": [],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)

    # Bitwise
    def __and__(self, other):
        return self._math_op(other, "&")

    def __or__(self, other):
        return self._math_op(other, "|")

    def __xor__(self, other):
        return self._math_op(other, "^")

    def __lshift__(self, other):
        return self._math_op(other, "<<")

    def __rshift__(self, other):
        return self._math_op(other, ">>")

    # Comparison
    def __eq__(self, other):
        return self._math_op(other, "==")

    def __ne__(self, other):
        return self._math_op(other, "!=")

    def __lt__(self, other):
        return self._math_op(other, "<")

    def __le__(self, other):
        return self._math_op(other, "<=")

    def __gt__(self, other):
        return self._math_op(other, ">")

    def __ge__(self, other):
        return self._math_op(other, ">=")

    def __ilshift__(self, other):
        assign(self, other)
        return self


def new_var(name: str = None):
    name = name or ""
    node_id = uuid.uuid4().hex
    if name:
        node_id = name + ":" + node_id
    v = Variable(node_id, _get_program())
    v <<= None
    return v


def new_cond():
    return new_var("0cond")


def check(variable: Variable, other: Variable | bool = True):
    if not isinstance(variable, Variable):
        return variable == True
    return variable._math_op(
        other,
        "==",
    )


def len(v, /):
    if isinstance(v, Variable):
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(v, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": "len",
                "var": node_id,
                "args": [],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)
    return builtins.len(v)


def next(v, /):
    if isinstance(v, Variable):
        program = _get_program()  # object.__getattribute__(self, "_program")
        node_id = object.__getattribute__(v, "_id")
        new_id = program._add(
            {
                "op": "operator",
                "operator": "next",
                "var": node_id,
                "args": [],
                "kwargs": {},
            }
        )
        return Variable(new_id, program)
    return builtins.next(v)


def assign_and_return(variable, other):
    node_id = object.__getattribute__(variable, "_id")
    # print(node_id)
    program = _get_program()  # object.__getattribute__(self, "_program")
    new_id = program._add(
        {
            "op": "assign",
            "var": node_id,
            "args": [_ser(other)],
            "kwargs": {},
        }
    )
    return variable


def assign(variable, other):
    node_id = object.__getattribute__(variable, "_id")
    # print(node_id)
    program = _get_program()  # object.__getattribute__(self, "_program")
    new_id = program._add(
        {
            "op": "assign",
            "var": node_id,
            "args": [_ser(other)],
            "kwargs": {},
        }
    )
    return Variable(new_id, program)


def _ser(arg: Any) -> Any:
    if isinstance(arg, Variable):
        return {"__type__": "var", "id": object.__getattribute__(arg, "_id")}
    if isinstance(arg, Constant):
        return {"__type__": "expr", "value": arg.value}
    return {"__type__": "val", "value": arg}


def _ser_key(key):
    if isinstance(key, UnpackIndex):
        return {"__type__": "unpack", "var": _ser(key.var)}
    elif isinstance(key, tuple):
        return {"__type__": "tuple", "items": [_ser_key(k) for k in key]}
    elif isinstance(key, slice):
        return {
            "__type__": "slice",
            "start": _ser(key.start),
            "stop": _ser(key.stop),
            "step": _ser(key.step),
        }
    elif key is Ellipsis:
        return {"__type__": "ellipsis"}
    elif key is None:
        return {"__type__": "none"}
    else:
        return _ser(key)


import io
import numpy as np
import torch
import base64


def _deserialize_result(value: Any, env: dict = None) -> Any:
    if isinstance(value, dict):
        t = value.get("__type__")

        if t == "tensor":
            buf = io.BytesIO(base64.b64decode(value["data"]))
            arr = np.load(buf)
            return torch.from_numpy(arr)

        if t is not None:
            # dataclass reconstruction — look up the class by name
            # from core.types import DATACLASS_REGISTRY  # dict of name -> class

            # cls = DATACLASS_REGISTRY.get(t)
            # if cls is None:
            #     raise ValueError(f"Unknown dataclass type: {t!r}")
            # fields = {k: _deserialize_result(v, env) for k, v in value.items() if k != "__type__"}
            # return cls(**fields)
            return value

        # plain dict
        return {k: _deserialize_result(v, env) for k, v in value.items()}

    if isinstance(value, list):
        return [_deserialize_result(v, env) for v in value]

    # primitives
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return value


def upsert_by_key(lst: list[dict], item: dict, key: str) -> None:
    for i, d in enumerate(lst):
        if d.get(key) == item[key]:
            lst.pop(i)
            lst.append(item)
            return
    lst.append(item)


class Program:
    """
    Collects instructions lazily.
    Supports both sync (get) and async (aget) execution.
    Also works as both sync and async context manager.
    """

    def __init__(self, server_url: str):
        self._instructions: dict[int, list[dict]] = {}
        self.active_index = 0
        self._server_url = server_url
        if self.active_index not in self._instructions:
            self._instructions[self.active_index] = []
        self.active_child: "Program" = None
        self.evaluating_condition = False

    @property
    def instructions(self):
        if self.active_index not in self._instructions:
            self._instructions[self.active_index] = []
        return self._instructions[self.active_index]

    # ── sync context manager ──
    def __enter__(self) -> "Program":
        self._instructions = {}
        push_active_program(self)
        return self

    def __exit__(self, *args):
        pop_active_program(self)

    async def __aenter__(self) -> "Program":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    # ── instruction graph ──
    def _add(self, instruction: dict) -> str:
        node_id = uuid.uuid4().hex
        assert not isinstance(
            self, Variable
        ), f"Assign Hook went wrong! Make sure you are only Hooking Variables"
        if self.active_index not in self._instructions:
            self._instructions[self.active_index] = []
        if instruction.get("op") == "assign":
            self._instructions[self.active_index].append({"id": instruction["var"], **instruction})
        else:
            self._instructions[self.active_index].append({"id": node_id, **instruction})
        return node_id

    def _add_or_replace(self, instruction: dict, property: str):
        node_id = uuid.uuid4().hex
        if self.active_index not in self._instructions:
            self._instructions[self.active_index] = []
        upsert_by_key(
            self._instructions[self.active_index], {"id": node_id, **instruction}, property
        )
        return node_id

    # ── instruction graph ──
    def execute(self, seaker_function_name, *args, **kwargs) -> Variable:
        node_id = self._add(
            {
                "op": "registered_function_call",
                "var": seaker_function_name,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return Variable(node_id, self)

    def package(self, package_name) -> Variable:
        node_id = self._add(
            {
                "op": "package_import",
                "var": package_name,
                "args": [],
                "kwargs": {},
            }
        )
        return Variable(node_id, self)

    def _build_payload(self, var: Variable) -> dict:
        node_id = object.__getattribute__(var, "_id")
        return {
            "instructions": self.instructions,
            "result_id": node_id,
        }

    def _build_payload_no_output(self) -> dict:
        return {
            "instructions": self.instructions,
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
        print(f"[client] Sending {len(self.instructions)} instructions (sync)...")
        print(json.dumps(self.instructions, indent="\t"))
        print(f"----send---- {requests}")
        resp = requests.post(
            f"{self._server_url}/api/solve_program",
            json=payload,
            timeout=timeout,
        )
        print(f"{resp=}")
        resp.raise_for_status()
        return _deserialize_result(self._parse_response(resp.json()))

    # ── async execution ──
    async def aget(self, var: Variable, timeout: int = 300) -> Any:
        """Non-blocking — sends program to server and returns result."""
        payload = self._build_payload(var)
        print(f"[client] Sending {len(self.instructions)} instructions (async)...")

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
        return json.dumps(self.instructions, indent=2)

    def set_active_index(self, index: int):
        self.active_index = index

    def next_active_index(self):
        self.set_active_index(self.active_index + 1)

    def reset_active_index(self):
        self.active_index = 0

    def if_(self, condition: dict | bool) -> "If":
        if_ = If(father_program=self, condition=condition, server_url=self._server_url)
        return if_

    def complete(self):
        pass

    def _has_active_child(self):
        return self.active_child is not None


class If(Program):
    def __init__(
        self, condition: Variable | bool, father_program: Program = None, server_url: str = None
    ):
        self.father_program = father_program or _get_program()
        server_url = server_url or self.father_program._server_url
        super().__init__(server_url)
        self.conditions = [condition]
        self.father_program.active_child = self
        self.child_id = uuid.uuid4().hex
        self.cond_ids = []
        self._built_cond_id = {}
        self.cond_id_idx = 0

    def built_cond_id(self, cond_int_index: int):
        return self._built_cond_id.get(cond_int_index, False)

    def set_built_cond_id(self, cond_id: str):
        self._built_cond_id[self.cond_id_idx] = True
        self.cond_id_idx += 1
        self.cond_ids.append(cond_id)
        assert len(self.cond_ids) == self.cond_id_idx, "Something went wrong with if conditions"

    def get_cond_id(self, cond_idx: int):
        return self.cond_ids[cond_idx]

    def __enter__(self) -> "If":
        self._instructions = {}
        push_active_program(self)
        return self

    async def __aenter__(self) -> "If":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    def _build_payload_no_output(self) -> dict:
        # evaluate wrt father program
        var_ids = []
        # print("==============================")
        # print(f"{_active_programs}")
        pop_active_program(self)
        for i in range(len(self.conditions)):
            # print(f"{self._built_cond_id=}")
            if self.built_cond_id(i):
                var_ids.append(self.get_cond_id(i))
                # print(f"Already built id {i} -> {self.get_cond_id(i)}.")
            else:
                # print(f"Bulding id {i}.")
                condition = self.conditions[i]
                _local_var = new_cond()
                cond_id = _ser(_local_var)
                var_ids.append(cond_id)
                assign(
                    _local_var,
                    check(condition),
                )
                self.set_built_cond_id(cond_id)
                # print(f"\t{cond_id=}")
        push_active_program(self)

        result = {
            "instructions": self._instructions,
            "conditions": var_ids,
        }
        tab = "\t"
        # print(f"[client] if sending {json.dumps(result, indent=tab)}")
        return result

    def __exit__(self, *args):
        # print("out!")
        self.complete()
        pop_active_program(self)

    def complete(self):

        result = self._build_payload_no_output()
        self.father_program._add_or_replace(
            {
                "op": "if",
                "child-id": self.child_id,
                "body": result,
            },
            property="child-id",
        )

    def else_if(self, condition: Variable | bool):
        self.next_active_index()
        self.conditions.append(condition)

    def else_(self):
        self.next_active_index()
        self.conditions.append(True)


class ElseIf(Program):
    def __init__(self, condition: Variable | bool):
        self.father_program = _get_program()
        self.branch_program: If = self.father_program.active_child
        assert isinstance(
            self.branch_program, If
        ), f"Invalid active child for elseif: {self.branch_program}, If expected."
        self.condition = condition

    def __enter__(self) -> "ElseIf":
        # print(self)
        # print("sssssssssssss")
        push_active_program(self.branch_program)
        self.branch_program.else_if(self.condition)
        return self

    async def __aenter__(self) -> "ElseIf":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    def __exit__(self, *args):
        self.branch_program.complete()
        pop_active_program(self.branch_program)


class Else(Program):
    def __init__(self):
        self.father_program = _get_program()
        self.branch_program: If = self.father_program.active_child
        assert isinstance(self.branch_program, If)
        self.condition = True

    def __enter__(self) -> "Else":
        # print(self)
        push_active_program(self.branch_program)
        self.branch_program.else_()
        return self

    async def __aenter__(self) -> "Else":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    def __exit__(self, *args):
        self.branch_program.complete()
        pop_active_program(self.branch_program)


class While(Program):
    def __init__(
        self, condition: Variable | bool, father_program: Program = None, server_url: str = None
    ):
        self.father_program = father_program or _get_program()
        server_url = server_url or self.father_program._server_url
        super().__init__(server_url)
        self.conditions = condition
        self.father_program.active_child = self
        self.child_id = uuid.uuid4().hex

    def __enter__(self) -> "While":
        self._instructions = {}
        push_active_program(self)
        return self

    async def __aenter__(self) -> "While":
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__()

    def __exit__(self, *args):
        self.complete()
        pop_active_program(self)

    def _build_payload_no_output(self) -> dict:
        # evaluate wrt father program
        var_ids = []
        # print("==============================")
        # print(f"{_active_programs}")
        pop_active_program(self)
        condition = self.conditions
        _local_var = new_cond()
        cond_id = _ser(_local_var)
        var_ids.append(cond_id)
        assign(
            _local_var,
            check(condition),
        )
        push_active_program(self)

        result = {
            "instructions": self.instructions,
            "condition": var_ids[0],
        }
        return result

    def complete(self):
        result = self._build_payload_no_output()
        self.father_program._add_or_replace(
            {
                "op": "while",
                "child-id": self.child_id,
                "body": result,
            },
            property="child-id",
        )
