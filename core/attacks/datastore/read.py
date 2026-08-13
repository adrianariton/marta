import os
import re
import json
from pathlib import Path
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Literal
from core.attacks.interfaces import LinearHistory, OneShotConversation
from typing_extensions import TypeVar, Generic

V = TypeVar("V")


@dataclass
class Info(Generic[V]):
    conversation: Optional[OneShotConversation]
    reply: V
    raw: dict
    goal: str

    def __repr__(self):
        if isinstance(self.reply, list):
            return self.reply[0][:50]
        return f"{self.reply}"


@dataclass
class Frame:
    run_id: str
    batch_idx: str
    attack: Optional[Info[list[str]]] = None
    agent: Optional[Info[list[str]]] = None
    is_attack_refused: Optional[Info[bool]] = None
    eval: Optional[Info[str]] = None
    other_: Optional[dict[str, Info]] = None

    def __post_init__(self):
        self.other_ = {}

    def history_pre_agent_reply(self):
        return self.agent.conversation

    def set_tag(self, tag: str, info_: Info):
        match tag.lower():
            case "attack":
                self.attack = info_
            case "agent":
                self.agent = info_
            case "is_attack_refused":
                self.is_attack_refused = info_
            case "eval":
                self.eval = info_

    def set_other_tag(self, tag: str, info_: Info):
        self.other_[tag] = info_

    def has_attack_query(self) -> bool:
        return self.attack is not None

    def has_empty_attack_query(self) -> bool:
        return self.attack_query() == ""

    def has_agent_response(self) -> bool:
        return self.agent is not None

    def agent_conversation(self) -> OneShotConversation:
        return self.agent.conversation

    def has_response_evaluation(self) -> bool:
        return self.eval is not None

    def has_is_refused(self):
        return self.is_attack_refused is not None

    def has_eval_or_refused(self):
        return self.has_is_refused() or self.has_response_evaluation()

    def is_eval_missing(self):
        return (self.has_is_refused() and not self.is_response_refused()) and (
            not self.has_response_evaluation()
        )

    def is_done(self) -> bool:
        if self.attack is None:
            return False
        return self.attack.reply[0] == "(done)" or self.attack.reply == "(done)"

    def attack_query(self) -> str:
        return self.attack.reply[0]

    def is_empty_frame(self) -> bool:
        return self.attack.reply[0] == "(nothing)"

    def agent_response(self) -> str:
        return self.agent.reply[0]

    def response_evaluation(self, format: Literal["dict"] = "dict") -> dict:
        if format == "dict":
            return json.loads(self.eval.reply)
        raise ValueError(f"Unsuported format {format}.")

    def response_evaluation_or_refusal(
        self,
        format: Literal["dict", "json"] = "dict",
        default: Literal["PEC7Default"] = "PEC7Default",
    ) -> dict:
        if self.has_is_refused() and self.is_response_refused():
            if default == "PEC7Default":
                return {"erosion": 0, "proximity": 0, "cooperation": 0, "goal_revealed": False}
            raise ValueError(f"Unsuported default {default}.")
        if format == "json":
            return json.loads(self.eval.reply)
        elif format == "dict":
            return self.eval.reply
        raise ValueError(f"Unsuported format {format}.")

    def is_response_refused(self) -> bool:
        return self.is_attack_refused.reply

    def __repr__(self):
        return f"Frame(attack={self.attack}, agent={self.agent}, eval={self.eval}, is_attack_refused={self.is_attack_refused})"


def get_data(folder_name, base_folder, tag_ATTACK: str, split_at_agent=False) -> list[Frame]:
    """Get all JSONL data from a folder, organized by run_id and batched by tag sequences"""
    folder_path = os.path.join(base_folder, folder_name)

    if not os.path.exists(folder_path):
        return []

    def get_shard(f):
        m = re.search(r"shard=(\d+)", f)
        return int(m.group(1)) if m else None

    # Get all JSONL files
    jsonl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".jsonl")], key=lambda f: get_shard(f)
    )

    # Organize data by run_id
    runs = {}

    for file in jsonl_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    run_id = entry.get("run_id")

                    if run_id not in runs:
                        runs[run_id] = []

                    runs[run_id].append(entry)
                except json.JSONDecodeError:
                    continue

    # Convert to frames format - each run_id becomes multiple frames (batches)
    frames = []
    for run_id, entries in runs.items():
        # print(f"{run_id=} {len(entries)=}")
        # Group entries into batches based on ATTACK tags (each ATTACK starts a new batch)
        batches = []
        current_batch = []

        for entry in entries:
            tag = entry.get("tag", "")
            # ATTACK tag starts a new batch
            if (tag.lower() == tag_ATTACK.lower()) and split_at_agent == False:

                if current_batch:  # Save previous batch if it exists
                    batches.append(current_batch)
                current_batch = [entry]
            elif tag.lower() == "agent" and split_at_agent == True:
                current_batch.append(entry)
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
            else:
                current_batch.append(entry)

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        # Convert each batch to a frame
        for batch_idx, batch in enumerate(batches):
            frame = Frame(run_id=run_id, batch_idx=batch_idx)

            for entry in batch:
                tag = entry.get("tag", "").lower()
                text = entry.get("text", "")

                # Handle the text field (which contains tuple: conversation list and reply)
                if isinstance(text, list) and len(text) == 2:
                    conversation, reply = text
                    # if tag.lower().strip() == "goat":
                    #     print(f"{reply=}")
                    #     exit(-1)
                    goal = "unspecified"
                    if isinstance(conversation, dict):
                        if "goal" in conversation:
                            goal = conversation["goal"]
                            # print(f"{goal}")
                        if "conversation" in conversation:
                            conversation = conversation["conversation"]

                        if (
                            isinstance(conversation, list)
                            and len(conversation) > 0
                            and isinstance(conversation[0], list)
                        ):
                            conversation = conversation[0]
                    else:
                        warnings.warn(
                            "Old or invalid format detected. Proceeding anyway. Consider using new format."
                        )
                    # print(f"{conversation=}")
                    if isinstance(conversation, dict):
                        if "conversation" in conversation:
                            conversation = conversation["conversation"]
                        if (
                            isinstance(conversation, list)
                            and len(conversation) > 0
                            and isinstance(conversation[0], list)
                        ):
                            conversation = conversation[0]
                        else:
                            conversation = None
                    conversation = (
                        OneShotConversation.from_default(conversation) if conversation else None
                    )
                    info_ = Info(
                        conversation=(conversation),
                        reply=reply,
                        raw=entry,
                        goal=goal,
                    )
                    if tag in ["is_attack_refused", "eval", "agent", tag_ATTACK]:
                        if tag.lower() == tag_ATTACK.lower():
                            tag = "attack"
                        frame.set_tag(tag, info_)
                    else:
                        frame.set_other_tag(tag, info_)
                else:
                    conversation = (
                        OneShotConversation.from_default(conversation) if conversation else None
                    )
                    info_ = Info(
                        conversation=(conversation),
                        reply=str(text),
                        raw=entry,
                        goal=goal,
                    )
                    if tag in ["is_attack_refused", "eval", "agent", tag_ATTACK]:
                        if tag.lower() == tag_ATTACK.lower():
                            tag = "attack"
                        frame.set_tag(tag, info_)
                    else:
                        frame.set_other_tag(tag, info_)
            frames.append(frame)
    return frames
