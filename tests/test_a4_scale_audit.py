import unittest

import cv2
import numpy as np

from scripts.audit_a4_scale import EXPECTED_MM, detect_scale_endpoints, evaluate


class A4ScaleAuditTests(unittest.TestCase):
    def test_length_alone_does_not_accept_shifted_mapping(self):
        shifted = np.array([[1, 0, 20], [0, 1, -30], [0, 0, 1]], dtype=float)
        result = evaluate(shifted, EXPECTED_MM)
        self.assertAlmostEqual(result['measured_length_mm'], 100)
        self.assertFalse(result['scale_location_check_passed'])

    def test_known_scale_location_is_accepted_without_motion_permission(self):
        result = evaluate(np.eye(3), EXPECTED_MM)
        self.assertTrue(result['scale_location_check_passed'])
        self.assertFalse(result['motion_ready'])

    def test_empty_roi_cannot_supply_evidence(self):
        with self.assertRaises(ValueError):
            detect_scale_endpoints(np.full((300, 100, 3), 255, np.uint8), [0, 0, 100, 300])

    def test_detects_an_independent_printed_line(self):
        image = np.full((300, 100, 3), 255, np.uint8)
        cv2.line(image, (30, 40), (38, 240), (0, 0, 0), 2)
        endpoints = detect_scale_endpoints(image, [10, 20, 60, 260])
        self.assertLess(np.linalg.norm(endpoints[0] - [30, 40]), 4)
        self.assertLess(np.linalg.norm(endpoints[1] - [38, 240]), 4)


if __name__ == '__main__':
    unittest.main()
