import unittest

from harness.so101_readiness import assess_readiness
from harness.so101_step import NAMES


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {name: {'range_min': 1000, 'range_max': 3000} for name in NAMES}
        self.record = {
            'registers': {key: dict.fromkeys(NAMES, value) for key, value in (
                ('Present_Position', 2000), ('Goal_Position', 2000), ('Present_Load', 0),
                ('Torque_Enable', 1), ('Status', 0), ('Present_Temperature', 30),
            )},
            'cameras': {role: {'age_at_read_s': 0.1} for role in ('wrist', 'global')},
            'red_block': {role: {'status': 'visible_unique', 'target': {'center_px': [1, 2]}}
                          for role in ('wrist', 'global')},
        }
        self.prediction = {'mode': 'shadow_only', 'hardware_writes': 0,
                           'execution_assessment': {'hardware_execution_permitted': True, 'reasons': []}}

    def test_healthy_evidence_is_not_an_action_authorization(self):
        report = assess_readiness(self.record, self.prediction, self.calibration)
        self.assertTrue(report['observation_healthy'])
        self.assertTrue(report['policy_proposal_compatible'])
        self.assertFalse(report['motion_authorized'])
        self.assertFalse(report['ready_for_supervised_execution'])

    def test_stale_or_ambiguous_camera_and_rejected_policy_are_reported(self):
        self.record['cameras']['wrist']['age_at_read_s'] = 0.8
        self.record['red_block']['global'] = {'status': 'unknown', 'target': None}
        self.prediction['execution_assessment'] = {'hardware_execution_permitted': False,
                                                   'reasons': ['out of range']}
        report = assess_readiness(self.record, self.prediction, self.calibration)
        self.assertFalse(report['observation_healthy'])
        self.assertFalse(report['policy_proposal_compatible'])
        self.assertTrue(any('wrist camera' in reason for reason in report['reasons']))
        self.assertTrue(any('out of range' in reason for reason in report['reasons']))
