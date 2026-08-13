from typing import TypedDict, Literal, List, Any
from enum import Enum

Role = Literal["user", "assistant", "system"]


class DefaultMessage(TypedDict):
    role: Role
    content: str


DefaultConversation = List[DefaultMessage]

from enum import Enum


class MsgFormat(Enum):
    DEFAULT = "default"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    VLLM = "vllm"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    JSONLOG = "jsonlog"
    GEMINI = "gemini"
    AUTO = "auto"


def gemini_to_default(messages) -> DefaultConversation:
    """
    Convert Gemini messages to Default format.
    - Gemini messages have 'role' and 'parts' (list of strings)
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        parts = m.get("parts", [])
        if not isinstance(parts, list):
            raise TypeError(f"Invalid parts in Gemini message: {parts}")
        content = "".join(str(p) for p in parts)
        out.append({"role": role, "content": content})
    return out


def openai_to_default(messages) -> DefaultConversation:
    """
    Convert OpenAI messages to Default format.
    - Content should be string; fallback to empty string if missing.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # Flatten chunks if any
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def anthropic_to_default(payload) -> DefaultConversation:
    """
    Convert Anthropic messages to Default format.
    - Anthropic messages have 'messages' list with 'role' and 'content' (list of dicts with 'text').
    """
    out = []
    for m in payload.get("messages", []):
        role = m.get("role", "user")
        content_list = m.get("content", [])
        if not isinstance(content_list, list):
            raise TypeError(f"Invalid content in Anthropic message: {content_list}")
        text = "".join(c.get("text", "") for c in content_list if c.get("type") == "text")
        out.append({"role": role, "content": text})
    return out


def hf_to_default(messages) -> DefaultConversation:
    """
    Convert HuggingFace / LLaVA messages to Default format.
    - Content can be a string or list of chunks.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text = "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            text = str(content)
        out.append({"role": role, "content": text})
    return out


def langchain_to_default(messages) -> DefaultConversation:
    """
    Convert LangChain messages to Default format.
    - Messages have .type and .content attributes.
    """
    map_role = {"human": "user", "ai": "assistant", "system": "system"}
    out = []
    for m in messages:
        role = map_role.get(getattr(m, "type", "user"), getattr(m, "type", "user"))
        content = getattr(m, "content", "")
        if isinstance(content, list):
            # Flatten chunks if any
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported LangChain content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def llamaindex_to_default(messages) -> DefaultConversation:
    """
    Convert LlamaIndex messages to Default format.
    - Messages have .content attribute.
    - Role inferred from type name ('ai' or 'human').
    """
    out = []
    for m in messages:
        t = type(m).__name__.lower()
        role = "assistant" if "ai" in t else "user"
        content = getattr(m, "content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported LlamaIndex content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def vllm_to_default(payload) -> DefaultConversation:
    """
    Convert vLLM payload to Default format.
    - Only a single 'prompt' string as user message.
    """
    prompt = payload.get("prompt", "")
    return [{"role": "user", "content": str(prompt)}]


def ollama_to_default(messages) -> DefaultConversation:
    """
    Convert Ollama messages to Default format.
    - Similar to OpenAI: 'role' and 'content'.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported Ollama content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def mistral_to_default(payload) -> DefaultConversation:
    """
    Convert Mistral payload to Default format.
    - Payload has 'messages' list of dicts with 'role' and 'content'.
    - Content can be string or list of chunks.
    """
    out = []
    for m in payload.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported Mistral content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def jsonlog_to_default(log) -> DefaultConversation:
    """
    Convert JSON log messages to Default format.
    - Each log entry has 'role' and 'msg'.
    """
    out = []
    for m in log:
        role = m.get("role", "user")
        content = str(m.get("msg", ""))
        out.append({"role": role, "content": content})
    return out


def gemini_to_default(messages) -> DefaultConversation:
    """
    Convert Gemini messages to Default format.
    - Gemini messages have 'role' and 'parts' (list of strings)
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        parts = m.get("parts", [])
        if not isinstance(parts, list):
            raise TypeError(f"Invalid parts in Gemini message: {parts}")
        content = "".join(str(p) for p in parts)
        out.append({"role": role, "content": content})
    return out


def openai_to_default(messages) -> DefaultConversation:
    """
    Convert OpenAI messages to Default format.
    - Content should be string; fallback to empty string if missing.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # Flatten chunks if any
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def anthropic_to_default(payload) -> DefaultConversation:
    """
    Convert Anthropic messages to Default format.
    - Anthropic messages have 'messages' list with 'role' and 'content' (list of dicts with 'text').
    """
    out = []
    for m in payload.get("messages", []):
        role = m.get("role", "user")
        content_list = m.get("content", [])
        if not isinstance(content_list, list):
            raise TypeError(f"Invalid content in Anthropic message: {content_list}")
        text = "".join(c.get("text", "") for c in content_list if c.get("type") == "text")
        out.append({"role": role, "content": text})
    return out


def hf_to_default(messages) -> DefaultConversation:
    """
    Convert HuggingFace / LLaVA messages to Default format.
    - Content can be a string or list of chunks.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text = "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            text = str(content)
        out.append({"role": role, "content": text})
    return out


def langchain_to_default(messages) -> DefaultConversation:
    """
    Convert LangChain messages to Default format.
    - Messages have .type and .content attributes.
    """
    map_role = {"human": "user", "ai": "assistant", "system": "system"}
    out = []
    for m in messages:
        role = map_role.get(getattr(m, "type", "user"), getattr(m, "type", "user"))
        content = getattr(m, "content", "")
        if isinstance(content, list):
            # Flatten chunks if any
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported LangChain content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def llamaindex_to_default(messages) -> DefaultConversation:
    """
    Convert LlamaIndex messages to Default format.
    - Messages have .content attribute.
    - Role inferred from type name ('ai' or 'human').
    """
    out = []
    for m in messages:
        t = type(m).__name__.lower()
        role = "assistant" if "ai" in t else "user"
        content = getattr(m, "content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported LlamaIndex content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def vllm_to_default(payload) -> DefaultConversation:
    """
    Convert vLLM payload to Default format.
    - Only a single 'prompt' string as user message.
    """
    prompt = payload.get("prompt", "")
    return [{"role": "user", "content": str(prompt)}]


def ollama_to_default(messages) -> DefaultConversation:
    """
    Convert Ollama messages to Default format.
    - Similar to OpenAI: 'role' and 'content'.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported Ollama content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def mistral_to_default(payload) -> DefaultConversation:
    """
    Convert Mistral payload to Default format.
    - Payload has 'messages' list of dicts with 'role' and 'content'.
    - Content can be string or list of chunks.
    """
    out = []
    for m in payload.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported Mistral content chunk: {c}")
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def auto_to_default(payload) -> DefaultConversation:
    return mistral_to_default(payload)


def jsonlog_to_default(log) -> DefaultConversation:
    """
    Convert JSON log messages to Default format.
    - Each log entry has 'role' and 'msg'.
    """
    out = []
    for m in log:
        role = m.get("role", "user")
        content = str(m.get("msg", ""))
        out.append({"role": role, "content": content})
    return out


FROM_DEFAULT_REGISTRY = {
    MsgFormat.OPENAI: openai_to_default,
    MsgFormat.ANTHROPIC: anthropic_to_default,
    MsgFormat.HUGGINGFACE: hf_to_default,
    MsgFormat.LANGCHAIN: langchain_to_default,
    MsgFormat.LLAMAINDEX: llamaindex_to_default,
    MsgFormat.VLLM: vllm_to_default,
    MsgFormat.OLLAMA: ollama_to_default,
    MsgFormat.MISTRAL: mistral_to_default,
    MsgFormat.JSONLOG: jsonlog_to_default,
    MsgFormat.AUTO: auto_to_default,
}


def to_default(fmt: MsgFormat, payload) -> DefaultConversation:
    if fmt == MsgFormat.DEFAULT:
        return payload
    return FROM_DEFAULT_REGISTRY[fmt](payload)


def default_to_openai(messages: DefaultConversation):
    """
    Convert Default messages to OpenAI format.
    - Each message is a dict with 'role' and 'content' as a string.
    - Flatten any chunked content into a single string.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        # Flatten chunked content
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
            content = "".join(text_parts)
        # Ensure string
        content = str(content)
        out.append({"role": role, "content": content})
    return out


def default_to_anthropic(messages: DefaultConversation):
    """
    Convert Default messages to Anthropic format.
    - Each message becomes {'role': role, 'content': [{'type': 'text', 'text': ...}]}
    - Chunked content is preserved as separate text entries if present.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        parts = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif isinstance(c, str):
                    parts.append({"type": "text", "text": c})
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
        else:
            parts.append({"type": "text", "text": str(content)})
        out.append({"role": role, "content": parts})
    return out


def default_to_hf(messages: DefaultConversation):
    """
    Convert Default messages to HuggingFace (LLaVA-style) format.
    - Each message becomes {'role': role, 'content': [{'type': 'text', 'text': ...}]}
    - Chunked content is preserved as separate text entries if present.
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        parts = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif isinstance(c, str):
                    parts.append({"type": "text", "text": c})
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
        else:
            parts.append({"type": "text", "text": str(content)})
        out.append({"role": role, "content": parts})
    return out


def default_to_gemini(messages: DefaultConversation):
    """
    Convert Default messages to Gemini format.
    - 'role' is mapped: 'assistant' -> 'model'
    - 'content' is always wrapped into a list called 'parts'
    - If content is already a list of chunks, each chunk's text is used
    """
    gemini_msgs = []
    for m in messages:
        role = m.get("role", "user")
        if role == "assistant":
            role = "model"  # Gemini uses 'model' instead of 'assistant'

        content = m.get("content")
        # If content is a list of chunks (like [{"type":"text","text":"..."}])
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    raise TypeError(f"Unsupported content chunk: {c}")
        else:
            # Single string content -> wrap in list
            parts = [str(content)]

        gemini_msgs.append({"role": role, "parts": parts})

    return gemini_msgs


def default_to_mistral(messages: DefaultConversation):
    """
    Convert Default messages to Mistral format.
    - Flatten any content chunks into a single string.
    - Keep roles unchanged.
    """
    mistral_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        # Flatten if content is a list of chunks
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
                else:
                    raise TypeError(f"Unsupported content type: {c}")
            content = "".join(text_parts)

        # Ensure content is a string
        if not isinstance(content, str):
            raise TypeError(f"Invalid content: {content}")

        mistral_msgs.append({"role": role, "content": content})

    return mistral_msgs


def default_to_auto(messages: DefaultConversation):
    """
    Convert Default messages to Mistral format.
    - Flatten any content chunks into a single string.
    - Keep roles unchanged.
    """
    return default_to_mistral(messages)


TO_DEFAULT_REGISTRY = {
    MsgFormat.OPENAI: default_to_openai,
    MsgFormat.ANTHROPIC: default_to_anthropic,
    MsgFormat.HUGGINGFACE: default_to_hf,
    MsgFormat.MISTRAL: default_to_mistral,
    MsgFormat.GEMINI: default_to_gemini,
    MsgFormat.AUTO: default_to_auto,
}


def from_default(fmt: MsgFormat, messages: DefaultConversation):
    if fmt == MsgFormat.DEFAULT:
        return messages
    return TO_DEFAULT_REGISTRY[fmt](messages)


def convert_messages(src_format: MsgFormat, tgt_format: MsgFormat, payload):
    default = to_default(src_format, payload)
    return from_default(tgt_format, default)


def detect_format(payload: Any) -> MsgFormat:
    """
    Detect the format of a chat payload.
    Returns a MsgFormat enum.
    """

    # ------------------- Check for Default format -------------------
    if isinstance(payload, list) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in payload
    ):
        return MsgFormat.DEFAULT

    # ------------------- OpenAI format -------------------
    if isinstance(payload, list) and all(
        isinstance(m, dict) and "role" in m and isinstance(m.get("content"), str) for m in payload
    ):
        return MsgFormat.OPENAI

    # ------------------- HuggingFace / LLaVA chunks -------------------
    if isinstance(payload, list) and all(
        isinstance(m, dict)
        and "role" in m
        and (
            isinstance(m.get("content"), str)
            or (
                isinstance(m.get("content"), list)
                and all(isinstance(c, dict) and "type" in c and "text" in c for c in m["content"])
            )
        )
        for m in payload
    ):
        return MsgFormat.HUGGINGFACE

    # ------------------- Anthropic -------------------
    if (
        isinstance(payload, dict)
        and "messages" in payload
        and isinstance(payload["messages"], list)
        and all("role" in m and isinstance(m["content"], list) for m in payload["messages"])
    ):
        return MsgFormat.ANTHROPIC

    # ------------------- LangChain -------------------
    if isinstance(payload, list) and all(
        hasattr(m, "type") and hasattr(m, "content") for m in payload
    ):
        return MsgFormat.LANGCHAIN

    # ------------------- LlamaIndex -------------------
    if isinstance(payload, list) and all(
        hasattr(m, "content")
        and "ai" in type(m).__name__.lower()
        or "human" in type(m).__name__.lower()
        for m in payload
    ):
        return MsgFormat.LLAMAINDEX

    # ------------------- vLLM -------------------
    if isinstance(payload, dict) and "prompt" in payload:
        return MsgFormat.VLLM

    # ------------------- Gemini -------------------
    if isinstance(payload, list) and all("role" in m and "parts" in m for m in payload):
        return MsgFormat.GEMINI

    # ------------------- Ollama -------------------
    if isinstance(payload, list) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in payload
    ):
        return MsgFormat.OLLAMA

    # ------------------- Mistral -------------------
    if isinstance(payload, dict) and "messages" in payload:
        return MsgFormat.MISTRAL

    # ------------------- JSON log -------------------
    if isinstance(payload, list) and all("role" in m and "msg" in m for m in payload):
        return MsgFormat.JSONLOG

    # ------------------- Unknown -------------------
    return None
