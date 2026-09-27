import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.calibrate_so101_jaw_and_base import fit_joint_alignment, load_audited_samples, predict, validate_held_out
from scripts.calibrate_robot_base_from_contacts import board_basis
from harness.contact_evidence import JOINTS


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

    def test_y_down_conversion_and_independent_predictions(self) -> None:
        poses = []
        for i, angle in enumerate([0., .5, -.8, 1.1, -1.4, .9]):
            c, s = np.cos(angle), np.sin(angle)
            poses.append([[c, -s, 0, i * i * 13.], [s, c, 0, i * 21.], [0, 0, 1, 120.]])
        poses = np.asarray(poses)
        parameters = np.array([.37, 41., -22., 14., -9., 26.])
        printed = predict(parameters, poses) @ board_basis('down')
        fitting = printed @ board_basis('down')
        fitted, rms, _ = fit_joint_alignment(poses, fitting)
        np.testing.assert_allclose(predict(fitted, poses) @ board_basis('down'), printed, atol=1e-5)
        self.assertLess(max(validate_held_out(poses, fitting)), 1e-4)

    def test_four_fitted_contacts_are_not_a_held_out_validation(self) -> None:
        with self.assertRaises(ValueError):
            validate_held_out(np.zeros((4, 3, 4)), np.zeros((4, 2)))

    def test_loader_requires_audited_contact_evidence_before_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(5):
                image = root / f'{index}.png'
                image.write_bytes(b'image')
                sample = root / f'{index}.json'
                sample.write_text(json.dumps({
                    'name': f'p{index}', 'board_xy_mm': [float(index), 0.0],
                    'raw_encoder_counts': dict.fromkeys(JOINTS, 2000), 'camera_evidence': str(image),
                }))
                paths.append(sample)
            manifest = root / 'samples.json'
            manifest.write_text(json.dumps({'samples': [str(path) for path in paths]}))
            samples, audit = load_audited_samples(manifest)
            self.assertEqual(len(samples), 5)
            self.assertTrue(audit['accepted'])
            paths[0].unlink()
            with self.assertRaisesRegex(ValueError, 'contact evidence audit failed'):
                load_audited_samples(manifest)
