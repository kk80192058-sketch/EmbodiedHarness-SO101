"""Hardware-independent safety checks applied before any robot action."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


class SafetyRejected(ValueError):
    """An action was rejected before it could reach a hardware adapter."""


@dataclass(frozen=True)
class SafetyConfig:
    robot_id: str
    action_timeout_s: float
    max_step_counts: int
    default_soft_margin_counts: int
    joint_limits: dict[str, tuple[int, int]]

    @classmethod
    def from_json(cls, path: Path) -> "SafetyConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            robot_id=raw["robot_id"],
            action_timeout_s=float(raw["action_timeout_s"]),
            max_step_counts=int(raw["max_step_counts"]),
            default_soft_margin_counts=int(raw["default_soft_margin_counts"]),
            joint_limits={name: tuple(bounds) for name, bounds in raw["joint_limits"].items()},
        )


class SafetyGate:
    """Reject unsafe raw joint goals before they are sent to a robot adapter."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def validate_raw_joint_goals(
        self,
        current: Mapping[str, int],
        proposed: Mapping[str, int],
        *,
        timeout_s: float,
    ) -> dict[str, int]:
        if timeout_s <= 0 or timeout_s > self.config.action_timeout_s:
            raise SafetyRejected(
                f"timeout {timeout_s}s is outside allowed range (0, {self.config.action_timeout_s}]"
            )

        unknown = set(proposed) - set(self.config.joint_limits)
        missing_current = set(proposed) - set(current)
        if unknown:
            raise SafetyRejected(f"unknown joints: {sorted(unknown)}")
        if missing_current:
            raise SafetyRejected(f"missing current positions: {sorted(missing_current)}")

        validated: dict[str, int] = {}
        for joint, target in proposed.items():
            start = current[joint]
            low, high = self.config.joint_limits[joint]
            soft_low = low + self.config.default_soft_margin_counts
            soft_high = high - self.config.default_soft_margin_counts
            if not low <= start <= high:
                raise SafetyRejected(f"{joint} current position {start} is outside calibrated bounds")
            if abs(target - start) > self.config.max_step_counts:
                raise SafetyRejected(
                    f"{joint} step {target - start} exceeds {self.config.max_step_counts} counts"
                )
            if not soft_low <= target <= soft_high:
                raise SafetyRejected(
                    f"{joint} target {target} is outside soft limits [{soft_low}, {soft_high}]"
                )
            validated[joint] = target
        return validated
