import unittest

from harness.contact_coverage import assess_contact_pose_coverage
from harness.contact_evidence import JOINTS


def sample(name, value):
    return {'name': name, 'raw_encoder_counts': dict.fromkeys(JOINTS, value)}


class ContactCoverageTests(unittest.TestCase):
    def test_reports_inside_envelope_and_nearest_sample(self):
        report = assess_contact_pose_coverage([sample('low', 1000), sample('high', 2000)], dict.fromkeys(JOINTS, 1500))
        self.assertTrue(report['inside_all_observed_joint_envelopes'])
        self.assertEqual(report['outside_joints'], [])
        self.assertEqual(report['nearest_sample']['max_abs_joint_delta_counts'], 500)

    def test_reports_exact_joints_outside_historical_contact_envelope(self):
        current = dict.fromkeys(JOINTS, 1500)
        current['wrist_flex'] = 900
        current['gripper'] = 2100
        report = assess_contact_pose_coverage([sample('low', 1000), sample('high', 2000)], current)
        self.assertFalse(report['inside_all_observed_joint_envelopes'])
        self.assertEqual(report['outside_joints'], ['wrist_flex', 'gripper'])
        self.assertEqual(report['joint_ranges']['wrist_flex']['current'], 900)
