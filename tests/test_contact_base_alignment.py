import unittest

import numpy as np

from scripts.calibrate_robot_base_from_contacts import board_point_from_pixel, fit_rigid_transform, fit_board_transform


class ContactBaseAlignmentTests(unittest.TestCase):
    def test_fits_a_known_rigid_transform_without_scale(self) -> None:
        robot = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 50.0], [40.0, 60.0]])
        angle = np.deg2rad(30.0)
        expected_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        expected_translation = np.array([15.0, -22.0])
        board = (expected_rotation @ robot.T).T + expected_translation

        rotation, translation, rms = fit_rigid_transform(robot, board)

        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-9)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-9)
        self.assertLess(rms, 1e-9)

    def test_projects_a_pixel_with_a_homography(self) -> None:
        homography = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(board_point_from_pixel(homography, [4.0, 5.0]), [9.0, 13.0])

    def test_rejects_too_few_contacts(self) -> None:
        with self.assertRaises(ValueError):
            fit_rigid_transform(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_printed_y_down_board_requires_a_basis_reflection(self) -> None:
        robot = np.array([[0., 0.], [100., 0.], [0., 70.], [40., 30.]])
        board = robot * [1, -1] + [10, 25]
        linear, translation, rms = fit_board_transform(robot, board, 'down')
        np.testing.assert_allclose(robot @ linear.T + translation, board, atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(linear), -1)
        self.assertLess(rms, 1e-9)
        _, _, wrong_rms = fit_board_transform(robot, board, 'up')
        self.assertGreater(wrong_rms, 10)
