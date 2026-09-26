"""Hardware-independent task agent with explicit recovery and evidence gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol

from harness.core import Evidence, FailureType, RobotObservation, SkillRequest, SkillResult
from harness.recording import EpisodeRecorder
from harness.sim import TabletopSimAdapter


class SkillBackend(Protocol):
    name: str

    def run(self, request: SkillRequest, robot: TabletopSimAdapter) -> SkillResult: ...


class RecoveryAction(str, Enum):
    RETRY = "retry"
    OPEN_AND_RETRY = "open_and_retry"
    ABORT = "abort"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str


@dataclass(frozen=True)
class TaskVerification:
    success: bool
    evidence: tuple[Evidence, ...]


class RecoveryPolicy:
    """Conservative recovery: bounded retries only; no invented motion plans."""

    def decide(self, result: SkillResult, attempts_used: int, max_attempts: int) -> RecoveryDecision:
        if attempts_used >= max_attempts:
            return RecoveryDecision(RecoveryAction.ABORT, "retry budget exhausted")
        if result.failure == FailureType.GRASP_FAIL:
            return RecoveryDecision(RecoveryAction.OPEN_AND_RETRY, "re-open gripper before bounded retry")
        if result.failure in {FailureType.PERCEPTION_MISS, FailureType.PLACE_FAIL}:
            return RecoveryDecision(RecoveryAction.RETRY, "bounded retry after fresh observation")
        return RecoveryDecision(RecoveryAction.ABORT, f"non-recoverable failure: {result.failure}")


class EvidenceVerifier:
    def verify_pick_place(self, observation: RobotObservation, object_id: str, target_id: str) -> TaskVerification:
        objects = {obj.object_id: obj for obj in observation.objects}
        obj = objects.get(object_id)
        target = objects.get(target_id)
        checks = (
            Evidence("robot_connected", observation.connected, {}),
            Evidence("object_exists", obj is not None, {"object_id": object_id}),
            Evidence("target_exists", target is not None, {"target_id": target_id}),
            Evidence(
                "object_inside_target",
                obj is not None and obj.container_id == target_id and not obj.grasped,
                {"object_id": object_id, "target_id": target_id},
            ),
            Evidence("gripper_released", observation.held_object_id is None and observation.gripper_open, {}),
        )
        return TaskVerification(all(check.passed for check in checks), checks)


class TaskAgent:
    """Runs a deterministic skill plan and records every decision for replay."""

    def __init__(
        self,
        backend: SkillBackend,
        recorder: EpisodeRecorder,
        *,
        recovery: RecoveryPolicy | None = None,
        verifier: EvidenceVerifier | None = None,
        max_attempts_per_skill: int = 2,
    ) -> None:
        if max_attempts_per_skill < 1:
            raise ValueError("max_attempts_per_skill must be positive")
        self.backend = backend
        self.recorder = recorder
        self.recovery = recovery or RecoveryPolicy()
        self.verifier = verifier or EvidenceVerifier()
        self.max_attempts_per_skill = max_attempts_per_skill

    def run_pick_place(self, robot: TabletopSimAdapter, object_id: str, target_id: str) -> TaskVerification:
        requests = (SkillRequest("pick", object_id), SkillRequest("place", object_id, target_id))
        self.recorder.append("episode_started", {"backend": self.backend.name, "plan": [asdict(item) for item in requests]})
        for request in requests:
            for attempt in range(1, self.max_attempts_per_skill + 1):
                self.recorder.append("observation", {"state": robot.observe(), "request": request, "attempt": attempt})
                self.recorder.append("skill_requested", {"request": request, "backend": self.backend.name, "attempt": attempt})
                result = self.backend.run(request, robot)
                self.recorder.append("skill_result", {"result": result, "attempt": attempt})
                if result.success:
                    break
                decision = self.recovery.decide(result, attempt, self.max_attempts_per_skill)
                self.recorder.append("recovery_decision", {"request": request, "decision": decision, "attempt": attempt})
                if decision.action == RecoveryAction.OPEN_AND_RETRY:
                    receipt = robot.execute("open_gripper")
                    self.recorder.append("recovery_action", {"action": "open_gripper", "receipt": receipt})
                if decision.action == RecoveryAction.ABORT:
                    verification = TaskVerification(False, (Evidence("episode_aborted", False, {"reason": decision.reason}),))
                    self.recorder.append("episode_completed", {"verification": verification})
                    return verification
            else:  # Defensive: normal policy exits through the abort branch above.
                raise RuntimeError("skill loop exhausted without result")
        verification = self.verifier.verify_pick_place(robot.observe(), object_id, target_id)
        self.recorder.append("episode_completed", {"verification": verification, "final_state": robot.observe()})
        return verification
