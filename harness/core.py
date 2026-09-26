"""Typed, hardware-independent runtime contracts.

These records are JSON-friendly so that an episode can be audited without
requiring a particular simulator, LLM, or robot driver to be installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class FailureType(str, Enum):
    PERCEPTION_MISS = "perception_miss"
    PRECONDITION_FAILED = "precondition_failed"
    GRASP_FAIL = "grasp_fail"
    PLACE_FAIL = "place_fail"
    SAFETY_REJECT = "safety_reject"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    label: str
    pose: Pose2D
    visible: bool = True
    grasped: bool = False
    container_id: str | None = None


@dataclass(frozen=True)
class RobotObservation:
    timestamp: float
    connected: bool
    gripper_open: bool
    held_object_id: str | None
    objects: tuple[ObjectState, ...]


@dataclass(frozen=True)
class SkillRequest:
    name: str
    object_id: str
    target_id: str | None = None


@dataclass(frozen=True)
class Evidence:
    criterion: str
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class SkillResult:
    request: SkillRequest
    success: bool
    evidence: tuple[Evidence, ...]
    failure: FailureType | None = None


@dataclass(frozen=True)
class ExecutionReceipt:
    accepted: bool
    timestamp: float
    note: str


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_episode_id() -> str:
    return f"ep_{uuid.uuid4().hex}"

