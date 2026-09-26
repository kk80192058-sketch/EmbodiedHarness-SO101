from pathlib import Path
import tempfile
import unittest

from harness.agent import TaskAgent
from harness.core import FailureType, ObjectState, Pose2D, SkillRequest, SkillResult
from harness.data_gate import EpisodeDataGate
from harness.recording import EpisodeRecorder
from harness.sim import TabletopSimAdapter
from harness.skills import GeometricSkillBackend


def robot() -> TabletopSimAdapter:
    adapter = TabletopSimAdapter([
        ObjectState("red", "red block", Pose2D(0, 0)),
        ObjectState("box", "blue box", Pose2D(1, 1)),
    ])
    adapter.connect()
    return adapter


class OneShotGraspFailureBackend:
    name = "one-shot-failure"

    def __init__(self) -> None:
        self.calls = 0
        self.real = GeometricSkillBackend()

    def run(self, request: SkillRequest, adapter: TabletopSimAdapter) -> SkillResult:
        self.calls += 1
        if self.calls == 1:
            return SkillResult(request, False, (), FailureType.GRASP_FAIL)
        return self.real.run(request, adapter)


class AgentRuntimeTests(unittest.TestCase):
    def test_clean_episode_is_verified_and_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(Path(directory), "clean")
            verification = TaskAgent(GeometricSkillBackend(), recorder).run_pick_place(robot(), "red", "box")
            decision = EpisodeDataGate().evaluate(recorder.replay())
        self.assertTrue(verification.success)
        self.assertTrue(decision.accepted, decision.reasons)

    def test_recovered_episode_is_not_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(Path(directory), "recovered")
            verification = TaskAgent(OneShotGraspFailureBackend(), recorder).run_pick_place(robot(), "red", "box")
            events = list(recorder.replay())
            decision = EpisodeDataGate().evaluate(events)
        self.assertTrue(verification.success)
        self.assertFalse(decision.accepted)
        self.assertIn("trace includes failed skill attempts; keep for debugging, exclude from training", decision.reasons)
        self.assertIn("recovery_action", [event["event_type"] for event in events])

    def test_unknown_skill_aborts_without_unsafe_retry(self) -> None:
        class UnknownBackend:
            name = "unknown"
            def run(self, request: SkillRequest, adapter: TabletopSimAdapter) -> SkillResult:
                return SkillResult(request, False, (), FailureType.PRECONDITION_FAILED)
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(Path(directory), "abort")
            verification = TaskAgent(UnknownBackend(), recorder).run_pick_place(robot(), "red", "box")
            events = list(recorder.replay())
        self.assertFalse(verification.success)
        self.assertEqual([event["event_type"] for event in events].count("skill_result"), 1)
        self.assertEqual(events[-1]["event_type"], "episode_completed")
