import unittest

import numpy as np

from scripts.propose_contact_pose_from_samples import barycentric, select_triangle


class EmpiricalContactPoseTests(unittest.TestCase):
    def test_selects_containing_triangle_and_weights_sum_to_one(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [50.0, 50.0]])
        indices, weights = select_triangle(points, np.array([30.0, 60.0]))
        self.assertEqual(len(indices), 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreaterEqual(float(weights.min()), 0.0)

    def test_rejects_target_outside_convex_coverage(self) -> None:
        points = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        with self.assertRaises(ValueError):
            select_triangle(points, np.array([200.0, 10.0]))

    def test_barycentric_rejects_degenerate_triangle(self) -> None:
        self.assertIsNone(barycentric(np.array([0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])))
