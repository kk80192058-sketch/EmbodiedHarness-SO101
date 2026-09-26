from pathlib import Path
import unittest

from harness.safety import SafetyConfig, SafetyGate, SafetyRejected


CONFIG = Path("configs/so101_safety.json")


class SafetyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = SafetyGate(SafetyConfig.from_json(CONFIG))
        self.current = {
            "shoulder_pan": 1894,
            "shoulder_lift": 2134,
            "elbow_flex": 1967,
            "wrist_flex": 2154,
            "wrist_roll": 2120,
            "gripper": 2049,
        }

    def test_accepts_the_verified_first_step(self) -> None:
        accepted = self.gate.validate_raw_joint_goals(
            self.current, {"shoulder_pan": 1914}, timeout_s=3.0
        )
        self.assertEqual(accepted, {"shoulder_pan": 1914})

    def test_rejects_a_large_step(self) -> None:
        with self.assertRaises(SafetyRejected):
            self.gate.validate_raw_joint_goals(
                self.current, {"shoulder_pan": 1930}, timeout_s=3.0
            )

    def test_rejects_a_target_near_the_hard_limit(self) -> None:
        near_upper = self.gate.config.joint_limits["shoulder_pan"][1] - 10
        with self.assertRaises(SafetyRejected):
            self.gate.validate_raw_joint_goals(
                {**self.current, "shoulder_pan": near_upper - 10},
                {"shoulder_pan": near_upper},
                timeout_s=3.0,
            )

    def test_rejects_an_invalid_timeout(self) -> None:
        with self.assertRaises(SafetyRejected):
            self.gate.validate_raw_joint_goals(
                self.current, {"shoulder_pan": 1914}, timeout_s=3.1
            )
