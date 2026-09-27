import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from harness.smolvla_policy import raw_to_policy, policy_to_raw, assess_proposal, evidence_manifest
from harness.so101_step import NAMES
from harness.safety import SafetyConfig


class PolicyContractTests(unittest.TestCase):
    def setUp(self):
        limits = [(745,3443),(889,3262),(825,3026),(952,3222),(0,4095),(1849,3307)]
        self.cal = {n: {'id': i, 'drive_mode': 0, 'homing_offset': 0,
                        'range_min': low, 'range_max': high}
                    for i, (n, (low, high)) in enumerate(zip(NAMES, limits, strict=True), 1)}
        self.raw = dict(zip(NAMES, [2416, 2413, 2345, 1783, 2140, 2077], strict=True))

    @unittest.skipUnless(importlib.util.find_spec('lerobot') is not None,
                         'requires optional so101-policy dependency')
    def test_raw_units_match_installed_lerobot_without_hardware(self):
        from lerobot.motors import Motor, MotorNormMode, MotorCalibration
        from lerobot.motors.feetech import FeetechMotorsBus
        for units, mode in [('degrees', MotorNormMode.DEGREES), ('range_m100_100', MotorNormMode.RANGE_M100_100)]:
            bus = FeetechMotorsBus('/unused', {n: Motor(self.cal[n]['id'], 'sts3215', MotorNormMode.RANGE_0_100 if n == 'gripper' else mode) for n in NAMES},
                                   calibration={n: MotorCalibration(**self.cal[n]) for n in NAMES})
            expected = bus._normalize({self.cal[n]['id']: self.raw[n] for n in NAMES})
            actual = raw_to_policy(self.raw, self.cal, body_units=units)
            for name, value in zip(NAMES, actual, strict=True):
                self.assertAlmostEqual(value, expected[self.cal[name]['id']])
            self.assertEqual(policy_to_raw(actual, self.cal, body_units=units), self.raw)

    def test_invalid_values_are_rejected_without_silent_clamping(self):
        values = raw_to_policy(self.raw, self.cal, body_units='degrees')
        for bad in (float('nan'), float('inf'), -1000, 1000):
            altered = values.copy(); altered[-1] = bad
            with self.assertRaises(ValueError):
                policy_to_raw(altered, self.cal, body_units='degrees')
        with self.assertRaises(ValueError):
            policy_to_raw(values[:-1], self.cal, body_units='degrees')

    def test_real_shadow_proposal_is_never_made_executable_by_clipping(self):
        registers = {k: dict.fromkeys(NAMES, v) for k, v in [('Torque_Enable', 1), ('Status', 0), ('Present_Load', 20), ('Present_Temperature', 30)]}
        registers.update(Present_Position=self.raw, Goal_Position=self.raw)
        prediction = {'raw_proposals': [{'raw_target': dict(zip(NAMES, [2282,2173,2270,2126,2232,1856], strict=True))}],
                      'normalized_state': [[1.29,1.45,-0.34,-3.07,-0.32,-5.12]]}
        report = assess_proposal(prediction, {'registers': registers}, self.cal, SafetyConfig.from_json(Path('configs/so101_safety.json')))
        self.assertFalse(report['hardware_execution_permitted'])
        self.assertEqual(report['out_of_distribution_joints'], ['wrist_flex', 'gripper'])
        self.assertTrue(all(not v['within_step_gate'] for v in report['joint_checks'].values()))

    def test_evidence_manifest_binds_json_and_both_saved_camera_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrist, global_frame = root / 'wrist.png', root / 'global.png'
            wrist.write_bytes(b'wrist-image'); global_frame.write_bytes(b'global-image')
            observation = root / 'observation.json'
            record = {'cameras': {'wrist': {'path': str(wrist)}, 'global': {'path': str(global_frame)}}}
            observation.write_text(json.dumps(record))
            manifest = evidence_manifest(observation, record)
            self.assertEqual(manifest['observation_json']['path'], str(observation.resolve()))
            self.assertNotEqual(manifest['camera_frames']['wrist']['sha256'],
                                manifest['camera_frames']['global']['sha256'])
            wrist.unlink()
            with self.assertRaisesRegex(ValueError, 'wrist'):
                evidence_manifest(observation, record)


if __name__ == '__main__':
    unittest.main()
