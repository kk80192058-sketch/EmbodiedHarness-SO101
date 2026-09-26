import unittest
from pathlib import Path

from harness.safety import SafetyConfig, SafetyGate, SafetyRejected


class WatchdogContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = SafetyGate(SafetyConfig.from_json(Path("configs/so101_safety.json")))
        self.safe_state = {
            "shoulder_pan": 1894,
            "shoulder_lift": 2134,
            "elbow_flex": 1967,
            "wrist_flex": 2154,
            "wrist_roll": 2120,
            "gripper": 2049,
        }

    def test_hold_can_latch_a_safe_current_state(self) -> None:
        self.assertEqual(
            self.gate.validate_raw_joint_goals(self.safe_state, self.safe_state, timeout_s=3.0),
            self.safe_state,
        )

    def test_watchdog_rejects_an_out_of_bounds_observation(self) -> None:
        unsafe = {**self.safe_state, "elbow_flex": 3027}
        with self.assertRaises(SafetyRejected):
            self.gate.validate_raw_joint_goals(unsafe, unsafe, timeout_s=3.0)
