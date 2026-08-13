import uuid
import json
import uuid
from pathlib import Path
from datetime import datetime
from core.attacks.utils import params_encoder
from typing import Callable
from enum import Enum
from core.attacks.utils import AutoRegisterMeta


class Tag(Enum, metaclass=AutoRegisterMeta):
    EVAL = "EVAL"
    ATTACK = "ATTACK"
    AGENT = "AGENT"
    IS_ATTACK_REFUSED = "IS_ATTACK_REFUSED"
    UNSPECIFIED = "UNSPECIFIED"


class MessageType(Enum):
    RESPONSE = "RESPONSE"
    SYSTEM = "SYSTEM"
    QUERY = "QUERY"
    CONVERSATION = "CONVERSATION"
    EXTRA = "EXTRA"


class MessageLogger:
    def __init__(self, on_tag: Callable[[int, str, list[str], Tag | str, MessageType], None]):
        self.on_tag = on_tag

    def on(
        self,
        i: int,
        gen_id: str,
        text: list[str],
        tag: Tag | str = None,
        mtype: MessageType = MessageType.RESPONSE,
    ):
        self.on_tag(i, gen_id, text, tag, mtype)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always flush remaining data
        self.close()
        # False → propagate exception if any
        return False


class MessageArrayLogger(MessageLogger):
    def __init__(self):
        self.data = []

    def on(
        self,
        i: int,
        gen_id: str,
        text: list[str],
        tag=None,
        mtype: MessageType = MessageType.RESPONSE,
    ):
        if mtype == MessageType.CONVERSATION or mtype == MessageType.EXTRA:
            self.data.append(
                {
                    "modelwise_index": i,
                    "gen_id": gen_id,
                    "tag": tag if isinstance(tag, str) else tag.name,
                    "text": text,
                    "mtype": mtype.name,
                }
            )

    def getData(self):
        return self.data


class MessageFileLogger(MessageLogger):
    def __init__(
        self, output_folder_name: str, buffer_max_size: int = 10, file_prefix: str = "log"
    ):
        self.output_dir = Path(output_folder_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = []
        self.buffer_max_size = buffer_max_size
        self.file_prefix = file_prefix

        self.file_index = 0
        self.run_id = uuid.uuid4()

        self.singular_id = uuid.uuid4()

        # optional but very useful
        self.created_at = datetime.utcnow().isoformat()

    def new_id(self):
        """Start a new logical run (new UUID)"""
        self._flush()
        self.run_id = uuid.uuid4()

    def _flush(self):
        if not self.buffer:
            return

        filename = (
            f"{self.file_prefix}_"
            f"i={self.singular_id}"
            f"run={self.run_id}_"
            f"shard={self.file_index:05d}.jsonl"
        )

        path = self.output_dir / filename

        with open(path, "w+") as f:
            for record in self.buffer:
                f.write(json.dumps(record, ensure_ascii=False, default=params_encoder) + "\n")

        self.buffer.clear()
        self.file_index += 1

    def on(
        self,
        i: int,
        gen_id: str,
        text: list[str],
        tag=None,
        mtype: MessageType = MessageType.RESPONSE,
    ):
        if mtype == MessageType.CONVERSATION:
            self.buffer.append(
                {
                    "run_id": str(self.run_id),
                    "modelwise_index": i,
                    "gen_id": gen_id,
                    "tag": tag if isinstance(tag, str) else tag.name,
                    "text": text,
                    "created_at": self.created_at,
                    "mtype": mtype.name,
                }
            )

            if len(self.buffer) >= self.buffer_max_size:
                self._flush()

    def close(self):
        """Call at program exit"""
        self._flush()

    def already_saved(self):
        if (self.output_dir / "info.txt").exists():
            print(f"[{self.output_dir.name}] already saved, skipping...", flush=True)
            return True
        else:
            folder_path = self.output_dir
            if folder_path.exists() and folder_path.is_dir():
                for file in folder_path.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil

                        shutil.rmtree(file)
            return False

    def save_info(self, info):
        with open(self.output_dir / "info.txt", "w+") as f:
            f.write(f"{info}")
