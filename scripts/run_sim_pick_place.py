"""Run one replayable, evidence-gated zero-demonstration simulation episode."""

from pathlib import Path

from harness.agent import TaskAgent
from harness.core import ObjectState, Pose2D
from harness.data_gate import EpisodeDataGate
from harness.recording import EpisodeRecorder
from harness.sim import TabletopSimAdapter
from harness.skills import GeometricSkillBackend


def main() -> None:
    robot = TabletopSimAdapter([
        ObjectState("red_block", "red block", Pose2D(0.10, 0.15)),
        ObjectState("left_box", "left box", Pose2D(-0.15, 0.15)),
    ])
    robot.connect()
    recorder = EpisodeRecorder(Path("artifacts/episodes"))
    verification = TaskAgent(GeometricSkillBackend(), recorder).run_pick_place(robot, "red_block", "left_box")
    decision = EpisodeDataGate().evaluate(recorder.replay())
    if not verification.success or not decision.accepted:
        raise SystemExit(f"episode rejected: verification={verification.success}; reasons={decision.reasons}")
    print(f"accepted simulation episode; replayable trace: {recorder.path}")


if __name__ == "__main__":
    main()
