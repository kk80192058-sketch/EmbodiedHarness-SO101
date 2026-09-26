"""Append-only, JSONL event recording and read-only replay."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from harness.core import RuntimeEvent, new_episode_id


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class EpisodeRecorder:
    def __init__(self, root: Path, episode_id: str | None = None) -> None:
        self.episode_id = episode_id or new_episode_id()
        self.path = root / self.episode_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def append(self, event_type: str, payload: dict[str, Any]) -> RuntimeEvent:
        event = RuntimeEvent(self._sequence, event_type, payload)
        self._sequence += 1
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), default=_json_default, sort_keys=True))
            stream.write("\n")
        return event

    def replay(self) -> Iterator[dict[str, Any]]:
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

