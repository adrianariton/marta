import re
from enum import Enum, EnumMeta

import re


def parse_structure(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    text = text.strip()

    # 1. Funcție cu << >>
    fn_match = re.match(r"#?\[([^\]]+)\]\(([^)]+)\)\s*\{\s*\}\s*<<", text)
    if fn_match:
        f, g = fn_match.groups()
        content, _ = extract_nested(text, fn_match.end())
        raw_params = split_top_level(content)
        parsed_params = [parse_structure(p.strip()) for p in raw_params]
        return f"{g}_{f}({', '.join(parsed_params)})"

    # 2. Tag simplu - match complet
    tag_match = re.match(r"#?\[([^\]]+)\]\(([^)]+)\)\s*\{.*\}\s*$", text, re.DOTALL)
    if tag_match:
        name, tag = tag_match.groups()
        return f"{tag}{name}"

    # 3. Fallback
    return text.strip()


def extract_nested(text, start_index):
    """Extrage tot ce e între prima pereche de << >> întâlnită."""
    count = 1
    for j in range(start_index, len(text)):
        if text[j : j + 2] == "<<":
            count += 1
        if text[j : j + 2] == ">>":
            count -= 1
        if count == 0:
            return text[start_index:j], j + 2
    return text[start_index:], len(text)


def split_top_level(text):
    """Split la virgulă DOAR dacă nu suntem în interiorul altui << >> sau {...}."""
    parts = []
    current = []
    angle_depth = 0  # Pentru << >>
    brace_depth = 0  # Pentru { }
    i = 0

    while i < len(text):
        char = text[i]

        # Verifică << (double angle bracket)
        if text[i : i + 2] == "<<":
            angle_depth += 1
            current.append("<<")
            i += 2
            continue

        # Verifică >> (double angle bracket)
        if text[i : i + 2] == ">>":
            angle_depth -= 1
            current.append(">>")
            i += 2
            continue

        # Verifică { (brace)
        if char == "{":
            brace_depth += 1
            current.append(char)
            i += 1
            continue

        # Verifică } (brace)
        if char == "}":
            brace_depth -= 1
            current.append(char)
            i += 1
            continue

        # Split la virgulă DOAR dacă suntem la nivel 0 pentru ambele
        if char == "," and angle_depth == 0 and brace_depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

        i += 1

    # Adaugă ultima parte
    parts.append("".join(current))

    return [p.strip() for p in parts if p.strip()]


def simplify_conversation(conversation):
    if not isinstance(conversation, list):
        return []
    simplified = []
    for entry in conversation:
        role = entry.get("role", "unknown")
        content = entry.get("content", [])

        # Scoatem textul (string sau listă)
        raw_text = ""
        for c in content:
            t = c.get("text", "")
            if isinstance(t, list):
                raw_text += " ".join([str(x) for x in t])
            else:
                raw_text += str(t)

        if role == "user" and "(done)" in raw_text:
            break

        # Parsăm string-ul rezultat
        parsed = parse_structure(raw_text)
        simplified.append(parsed)
    return simplified


# --- 3. Dictionary and Conversation Logic ---


def get_dict(obj):
    """Helper to safely get dictionary from Params or dict objects."""
    if hasattr(obj, "._dict"):
        return obj._dict
    return obj if isinstance(obj, dict) else {}


def process_any(data):
    """Recursively traverses data, unwrapping Params objects along the way."""
    # Handle Params classes by unwrapping first
    if hasattr(data, "._dict"):
        return process_any(data._dict)

    if isinstance(data, dict):
        return {k: process_any(v) for k, v in data.items()}
    if isinstance(data, list):
        return [process_any(v) for v in data]
    if isinstance(data, str):
        return parse_structure(data)
    return data


def simplify_latest_oneshot_for_testing(agent_latest_oneshot_0):
    return simplify_conversation(agent_latest_oneshot_0)


def simplify_logger_data_for_testing(logger_data):
    simplified = []
    for entry in logger_data:
        # Ensure entry is treated as a dict
        d_entry = get_dict(entry)

        if d_entry.get("mtype") == "CONVERSATION":
            text_field = d_entry.get("text", [{}, ""])

            # Safely unwrap the params (the first element of 'text')
            params = get_dict(text_field[0])

            # Create a copy to avoid mutating original data
            params_copy = params.copy()

            if "conversation" in params_copy:
                params_copy["conversation"] = simplify_conversation(params_copy["conversation"])

            simplified_params = process_any(params_copy)
            result = process_any(text_field[1]) if len(text_field) > 1 else ""

            simplified.append(
                {
                    "tag": d_entry.get("tag", None),
                    "result": result,
                    "params": simplified_params,
                }
            )
        elif d_entry.get("mtype") == "EXTRA":
            simplified.append(
                {
                    "tag": "EXTRA_" + d_entry.get("tag", None),
                    "result": d_entry.get("text", None),
                }
            )

    return simplified


def cut_after_done(agent_latest_oneshot_0):
    cut = []
    for entry in agent_latest_oneshot_0:
        cut.append(entry)
        if entry["role"] == "user" and entry["content"][0]["text"] == "(done)":
            break
    return cut
