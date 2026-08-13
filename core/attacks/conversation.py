from abc import abstractmethod, ABC
from copy import deepcopy
from enum import Enum
from typing import Callable
from collections import Counter
import numpy as np
from typing import List, Dict, Union, Literal
from copy import deepcopy
from core.attacks.datastore.yielder_registry import YielderRegistry
from core.attacks.datastore.yieldable import Yieldable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.attacks.classifiers.refusal.base import RefusalClassifier
from core.attacks.utils import Params
from typing import Optional, TypeVar, Type
from core.message_formatters import MsgFormat


def negate_role(role: Literal["user", "assistant"]):
    assert role in ["user", "assistant", "system"], "Role must be eithe ruser or assistant"
    if role in ["user", "assistant"]:
        return "user" if role == "assistant" else "assistant"
    return "system"


class Content:
    def __init__(self):
        pass

    def get(self):
        pass

    def to_string(self) -> str:
        pass

    @classmethod
    def from_dict(cls, d):
        if d["type"] == "text":
            return TextContent.from_dict(d)


class TextContent(Content):
    def __init__(self, text: str):
        self.text = text

    def get(self):
        return {
            "type": "text",
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(text=d["text"])

    def to_string(self) -> str:
        return self.text


class Message:
    def __init__(self, content: list[Content]):
        self.content = content  # [c for c in content if c is not None]

    def to_json(self):
        return [c.get() for c in self.content]

    @classmethod
    def from_text(cls, text: str):
        assert isinstance(text, str)
        return cls([TextContent(text=text)])

    def to_string(self) -> str:
        return "\n".join([c.to_string() for c in self.content])

    def to_parts(self) -> list[str]:
        return [c.to_string() for c in self.content]

    def get(self) -> list[dict]:
        return [c.get() for c in self.content]

    @classmethod
    def from_jsonl(cls, jsonl: list[dict]):

        return cls([Content.from_dict(d) for d in jsonl])


import uuid


class OneShotConversation:
    def __init__(self, _uuid=None):
        self._messages: list[tuple[Literal["user", "assistant", "system"], Message]] = []
        self._uuid = _uuid if _uuid is not None else str(uuid.uuid4())

    def __getitem__(self, key: int) -> tuple[Literal["user", "assistant", "system"], str]:
        return self._messages[key]

    def size(self):
        return len(self._messages)

    def empty(self):
        return len(self._messages) == 0

    def last_has_role(self, role: Literal["user", "assistant", "system"]) -> bool:
        return self._messages[len(self._messages) - 1][0] == role

    def get_last_with_role(self, role: Literal["user", "assistant", "system"]) -> Optional[Message]:
        n = len(self._messages) - 1
        while True:
            if n >= 0 and self._messages[n][0] == role:
                return self._messages[n][1]
            if n == 0:
                return None
            n -= 1
        return None

    def get_last(self) -> tuple[Literal["user", "assistant", "system"], Message]:
        return self._messages[len(self._messages) - 1]

    def roles(self) -> list[str]:
        return list([m[0] for m in self._messages])

    def is_ready_for_completion(self) -> tuple[bool, str]:
        roles = [role for role, mess in self.messages()]
        if len(roles) == 0:
            return False, "Empty conversation not ready for completion."
        if roles[0] == "system":
            roles = roles[1:]

        if "system" in roles:
            return False, "Duplicate system messages =>  not ready for completion."

        if len(roles) % 2 == 0:
            return (
                False,
                f"Conversation - System msg must have odd length =>  not ready for completion. {roles=}",
            )

        for i in range(len(roles) // 2):
            if roles[2 * i] != "user" and roles[2 * i + 1] != "assistant":
                return (
                    False,
                    "Conversation - System msg must alternate user/assistant =>  not ready for completion.",
                )

        if roles[-1] != "user":
            return (
                False,
                "Conversation must end in 'user' message role  =>  not ready for completion. "
                + f"{roles=}",
            )
        return True, "ok"

    def assert_ready_for_completion(self):
        isready, err = self.is_ready_for_completion()
        if not isready:
            raise Exception(f"{err}")

    @classmethod
    def from_system_and_user(cls, user, system=None):
        c = cls()
        if system is not None:
            c.set_system_message(system)
        c.add_text_message(user, role="user")
        return c

    def get_system_message(self) -> Optional[Message]:
        if len(self._messages) == 0:
            return None
        if self._messages[0][0] == "system":
            return self._messages[0][1]
        return None

    def set_system_message(self, message: Message | str):
        message = message if isinstance(message, Message) else Message.from_text(message)
        if len(self._messages) == 0:
            self._messages.append(("system", message))
        elif self._messages[0][0] == "system":
            self._messages[0][1] = message
        else:
            self._messages = [("system", message)] + self._messages

    def pop_system_message(self) -> Optional[Message]:
        if len(self._messages) == 0:
            return None
        if self._messages[0][0] == "system":
            return self._messages.pop(0)[1]
        return None

    def add_text_message(self, message: Message | str, role: Literal["user", "assistant"]):
        self.add_message(
            Message.from_text(message) if isinstance(message, str) else message, role=role
        )

    def add_message(self, message: Message, role: Literal["user", "assistant"]):
        assert role in ["user", "assistant"]
        self._messages.append((role, message))

    def flipped(self, keep_system=False, remove_first_non_system=False) -> "OneShotConversation":
        convo = OneShotConversation(_uuid=self._uuid)
        msgs = []
        for role, message in self._messages:
            if role in ["assistant", "user"] or (
                keep_system and role in ["assistant", "user", "system"]
            ):
                msgs.append((negate_role(role), message))
                # convo.add_message(message, negate_role(role))

        if remove_first_non_system:
            if msgs[0][0] == "system" and len(msgs) > 1:
                msgs = msgs[0] + msgs[2:]
            else:
                msgs = msgs[1:]
        for role, message in msgs:
            convo.add_message(message, role)  # TODODODOD
        return convo

    def copy(self) -> "OneShotConversation":
        return deepcopy(self)

    def replace(self, x: Message | str, by_y: Message | str, of_role: Literal["user", "assistant"]):
        """
        Replace x by y in convo

        Args:
            x (Message | str): _description_
            by_y (Message | str): _description_
            of_role (Literal[&quot;user&quot;, &quot;assistant&quot;]): _description_

        Returns:
            _type_: _description_
        """
        x = x if isinstance(x, Message) else Message.from_text(x)
        by_y = by_y if isinstance(by_y, Message) else Message.from_text(by_y)
        assert isinstance(x, Message)
        assert isinstance(by_y, Message)

        def repl(role, mess: Message):
            if role == of_role:
                if mess.to_string() == x.to_string():
                    return role, by_y
                return role, mess
            return role, mess

        self._messages = [repl(*m) for m in self._messages]

    @classmethod
    def from_default(cls, default_format: list[dict[str, str | dict[str, Any]]]):
        """format default:
        ```
        [
            {
                'role': 'user'/'assistant'
                'content': [
                    {
                        'type': 'text',
                        'text': 'blabla'
                    },
                    {
                        'type': 'text',
                        'text': 'blabla2'
                    },
                    ...
                ]
            },
            ...
        ]
        ```
        """
        c = cls()
        for msg in default_format:
            role = msg["role"]
            message = Message.from_jsonl(msg["content"])
            c._messages.append((role, message))
        return c

    def to_default(self) -> list[dict[str, str | dict[str, Any]]]:
        """
        Yields as format default:
        ```
        [
            {
                'role': 'user'/'assistant'
                'content': [
                    {
                        'type': 'text',
                        'text': 'blabla'
                    },
                    {
                        'type': 'text',
                        'text': 'blabla2'
                    },
                    ...
                ]
            },
            ...
        ]
        ```
        """
        return [{"role": m[0], "content": m[1].to_json()} for m in self._messages]

    def to_gemini(self, ignore_system=True):
        """
        Convert Default messages to Gemini format.
        - 'role' is mapped: 'assistant' -> 'model'
        - 'content' is always wrapped into a list called 'parts'
        - If content is already a list of chunks, each chunk's text is used

        ```
        [
            {"role": "user", "parts": ["Hello, how are you?"]},
            {"role": "model", "parts": ["I am doing well. How can I help?"]},
            {"role": "user", "parts": ["What is the weather?"]}
        ]
        ```
        """
        gemini_msgs = []
        for role, message in self._messages:
            if role == "assistant":
                role = "model"  # Gemini uses 'model' instead of 'assistant'
            parts = message.to_parts()
            if role in ["user", "model"] or ignore_system == False:
                gemini_msgs.append({"role": role, "parts": parts})
        return gemini_msgs

    def to_mistral(self):
        """
        Convert Default messages to Mistral format.
        - Flatten any content chunks into a single string.
        - Keep roles unchanged.
        ```
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well. How can I help?"},
            {"role": "user", "content": "What is the weather?"}
        ]
        ```
        """
        mistral_msgs = []
        for role, message in self._messages:
            content = message.to_string()
            mistral_msgs.append({"role": role, "content": content})
        return mistral_msgs

    def to_auto(self):
        """
        Convert Default messages to Mistral format.
        - Flatten any content chunks into a single string.
        - Keep roles unchanged.
        ```
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well. How can I help?"},
            {"role": "user", "content": "What is the weather?"}
        ]
        ```
        """
        return self.to_mistral()

    def to_openai(self):
        """
        Convert Default messages to Mistral format.
        - Flatten any content chunks into a single string.
        - Keep roles unchanged.
        ```
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well. How can I help?"},
            {"role": "user", "content": "What is the weather?"}
        ]
        ```
        """
        return self.to_mistral()

    def to_anthropic(self, ignore_system=True):
        """
        Convert Default messages to Anthropic format.
        - Each message becomes {'role': role, 'content': [{'type': 'text', 'text': ...}]}
        - Chunked content is preserved as separate text entries if present.
        """
        out = []
        for role, message in self._messages:
            parts = message.get()
            if role in ["user", "assistant"] or ignore_system is False:
                out.append({"role": role, "content": parts})
        return out

    def to_format(self, format_: MsgFormat):
        """
        Check documentation of each function for info on output
        ```
        if format_ == MsgFormat.ANTHROPIC:
            return self.to_anthropic()
        elif format_ == MsgFormat.OPENAI:
            return self.to_openai()
        elif format_ == MsgFormat.AUTO:
            return self.to_auto()
        elif format_ == MsgFormat.MISTRAL:
            return self.to_mistral()
        elif format_ == MsgFormat.DEFAULT:
            return self.to_default()
        elif format_ == MsgFormat.GEMINI:
            return self.to_gemini()
        ```
        """
        if format_ == MsgFormat.ANTHROPIC:
            return self.to_anthropic()
        elif format_ == MsgFormat.OPENAI:
            return self.to_openai()
        elif format_ == MsgFormat.AUTO:
            return self.to_auto()
        elif format_ == MsgFormat.MISTRAL:
            return self.to_mistral()
        elif format_ == MsgFormat.DEFAULT:
            return self.to_default()
        elif format_ == MsgFormat.GEMINI:
            return self.to_gemini()
        else:
            raise Exception(f"Unsupported format {format_}")

    def messages(self):
        return self._messages

    def last_user_message(self) -> Optional[Message]:
        return self.get_last_with_role("user")

    def last_assistant_message(self) -> Optional[Message]:
        return self.get_last_with_role("assistant")
