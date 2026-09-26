from pathlib import Path
import tempfile
import unittest

from harness.core import ObjectState, Pose2D, SkillRequest
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


class W1ContractTests(unittest.TestCase):
    def test_pick_and_place_produce_verifiable_state(self) -> None:
        adapter = robot()
        backend = GeometricSkillBackend()
        self.assertTrue(backend.run(SkillRequest("pick", "red"), adapter).success)
        result = backend.run(SkillRequest("place", "red", "box"), adapter)
        self.assertTrue(result.success)
        red = next(obj for obj in adapter.observe().objects if obj.object_id == "red")
        self.assertEqual(red.container_id, "box")
        self.assertTrue(result.evidence[0].passed)

    def test_place_is_rejected_without_grasp_precondition(self) -> None:
        result = GeometricSkillBackend().run(SkillRequest("place", "red", "box"), robot())
        self.assertFalse(result.success)
        self.assertEqual(result.failure.value, "precondition_failed")

    def test_episode_recorder_is_append_only_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(Path(directory), "known_episode")
            recorder.append("first", {"value": 1})
            recorder.append("second", {"value": 2})
            replay = list(recorder.replay())
        self.assertEqual([event["sequence"] for event in replay], [0, 1])
        self.assertEqual([event["event_type"] for event in replay], ["first", "second"])
