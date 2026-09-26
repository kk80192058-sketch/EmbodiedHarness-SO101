import unittest

import numpy as np

from scripts.calibrate_so101_jaw_and_base import fit_joint_alignment, predict


class JointJawAlignmentTests(unittest.TestCase):
    def test_recovers_joint_offset_and_planar_alignment(self) -> None:
        # Four distinct gripper orientations/positions, in millimetres.
        angles = [0.0, 0.5, -0.8, 1.1, -1.4]
        translations = [[30, 10, 120], [80, -20, 115], [-40, 60, 125], [10, 110, 118], [130, 70, 123]]
        poses = []
        for angle, translation in zip(angles, translations):
            c, s = np.cos(angle), np.sin(angle)
            poses.append([[c, -s, 0, translation[0]], [s, c, 0, translation[1]], [0, 0, 1, translation[2]]])
        poses = np.asarray(poses, dtype=float)
        expected = np.array([0.37, 41.0, -22.0, 14.0, -9.0, 26.0])
        board = predict(expected, poses)

        observed, rms, _ = fit_joint_alignment(poses, board)

        np.testing.assert_allclose(predict(observed, poses), board, atol=1e-5)
        self.assertLess(rms, 1e-6)

    def test_requires_four_contacts(self) -> None:
        with self.assertRaises(ValueError):
            fit_joint_alignment(np.zeros((3, 3, 4)), np.zeros((3, 2)))
