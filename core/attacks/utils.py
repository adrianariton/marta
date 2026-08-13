import re

from typing import Literal
from enum import Enum
from enum import Enum, EnumMeta


class Stateful:
    def __init__(self, n, type_: Literal[">= n", "< n"] = ">= n"):
        self.i = 0
        self.n = n
        self.type_ = type_

    def execute(self, f, *args, **kwargs):
        if self.type_ == ">= n" and self.i >= self.n:
            self.i += 1
            return f(*args, **kwargs)
        elif self.type_ == "< n" and self.i < self.n:
            self.i += 1
            return f(*args, **kwargs)
        else:
            self.i += 1
            return None

    def reset(self):
        self.i = 0


class AutoRegisterMeta(EnumMeta):
    def __getattr__(cls, name):
        # Handles Tag.NEW_NAME
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_"):
                raise
            return cls._generate_new_member(name, name)

    def __call__(cls, value, names=None, *args, **kwargs):
        # Handles Tag("NEW_VALUE")
        # If 'names' is present, it's the functional API (Tag('Tag', ['A', 'B']))
        if names is None:
            try:
                return super().__call__(value, *args, **kwargs)
            except ValueError:
                return cls._generate_new_member(value, value)
        return super().__call__(value, names, *args, **kwargs)

    def _generate_new_member(cls, name, value):
        # Logic to inject the new member into the Enum registry
        new_member = obj = object.__new__(cls)
        obj._name_ = name
        obj._value_ = value
        cls._member_map_[name] = new_member
        cls._value2member_map_[value] = new_member
        return new_member


class Params:
    class PType(Enum, metaclass=AutoRegisterMeta):
        TEXTGENERATOR = 1
        REFUSALCLASSIFIER = 2
        BRIDGER = 3
        REALIGNER = 4
        PEC7EVALUATOR = 5
        MIDQUERY = 6
        UNKNOWN = 7
        ALIGNMENTTESTER = 8

    def __init__(self, param_type: PType, **kwargs):
        self._dict: dict = dict(**kwargs)
        self.param_type = param_type


def params_encoder(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Params):
        return {"param_type": obj.param_type.value, **obj._dict}
    else:
        return {"class": obj.__class__.__name__, **obj.__dict__}
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# def get_faulty_json_data(json_str: str, prop: str):
#     """
#     Extracts metrics and metadata from a potentially malformed
#     JSON string using flexible regex.
#     """
#     # Pattern explanation:
#     # 1. Look for the property name in quotes
#     # 2. Match the colon and optional whitespace
#     # 3. Capture content that is either inside quotes OR ends at a comma/bracket
#     pattern = rf'"{prop}"\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s}}]+))'

#     match = re.search(pattern, json_str)

#     if not match:
#         return None

#     # The value could be in group 1 (double quotes), 2 (single quotes), or 3 (unquoted)
#     val = match.group(1) or match.group(2) or match.group(3)

#     if val is None:
#         return None

#     clean_val = val.strip()

#     # 1. Handle Booleans
#     if clean_val.lower() == "true":
#         return True
#     if clean_val.lower() == "false":
#         return False

#     # 2. Handle Numbers (Int & Float)
#     try:
#         if "." in clean_val:
#             return float(clean_val)
#         return int(clean_val)
#     except ValueError:
#         # 3. Handle Strings (Fallback)
#         # Removes residual quotes if the regex missed them
#         return clean_val.replace('"', "").replace("'", "")

import re
import json


def get_faulty_json_data(json_str: str, prop: str):
    """
    Extrage valori din JSON-uri malformate, suportând:
    - Chei cu ghilimele duble: "prop"
    - Chei cu ghilimele simple: 'prop'
    - Chei fără ghilimele: prop
    """
    did = 0
    try:
        jsjs = json.loads(json_str)
        if prop in jsjs:
            return jsjs[prop]
    except json.JSONDecodeError:
        did = 1

    # EXPLICAȚIE PATTERN:
    # (['"])?           <- Grupul 1: Caută o ghilimele simplă sau dublă (opțional)
    # {re.escape(prop)} <- Numele proprietății (escaped pentru siguranță)
    # \1                <- Backreference: Dacă a găsit o ghilimele la început,
    #                      trebuie să găsească aceeași ghilimele la finalul cheii
    # \s*:\s* <- Separatorul colon cu spații opționale
    # (?:               <- Grup de captură pentru VALOARE (non-greedy):
    #    "([^"]*)"      <- Grupul 2: Valoare între ghilimele duble
    #    |'([^']*)'     <- Grupul 3: Valoare între ghilimele simple
    #    |([^,\s}}]+)   <- Grupul 4: Valoare neîncapsulată (numere, bool, etc.)
    # )

    pattern = rf'([\'"])?{re.escape(prop)}\1\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s}}]+))'

    match = re.search(pattern, json_str)

    if not match:
        return None

    # Extragem valoarea din grupul care a făcut match (2, 3 sau 4)
    val = match.group(2) or match.group(3) or match.group(4)

    if val is None:
        return None

    clean_val = val.strip()

    # Conversii de tip
    low_val = clean_val.lower()
    if low_val == "true":
        return True
    if low_val == "false":
        return False
    if low_val == "null" or low_val == "none":
        return None

    try:
        if "." in clean_val:
            return float(clean_val)
        return int(clean_val)
    except ValueError:
        # Curățăm resturile de ghilimele dacă regex-ul a fost forțat
        return clean_val.strip("'\"")


# Example of how this handles the "Stack":
# raw = '{"proximity": 7, "comment": "Jailbreak detected", "goal-revealed": true}'
# p = get_pec7_data(raw, "proximity")      # returns int: 7
# c = get_pec7_data(raw, "comment")        # returns str: "Jailbreak detected"
# g = get_pec7_data(raw, "goal-revealed") # returns bool: True

import re


def parse_simple(string: str, markers: list[str]) -> dict[str, str]:
    """

    ```py
    data = "ID: 123 Name: Alice Status: Active"
    markers = ["ID:", "Name:", "Status:"]

    print(parse_simple(data, markers))
    ```

    ```sh
    {
        "ID:": "123",
        "Name:": "Alice",
        "Status:": "Active"
    }
    ```

    Args:
        string (str): _description_
        markers (list[str]): _description_

    Returns:
        _type_: _description_
    """
    escaped_markers = [re.escape(m) for m in markers]

    pattern = "(" + "|".join(escaped_markers) + ")"

    # Split the string by the markers while keeping them
    parts = re.split(pattern, string)

    result = {}
    # Iterate through the parts: [pre-match, marker1, content1, marker2, content2...]
    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i + 1].strip()
        result[marker] = content

    return result
