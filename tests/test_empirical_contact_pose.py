import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from harness.contact_evidence import JOINTS
from scripts.propose_contact_pose_from_samples import barycentric, load_audited_samples, select_triangle


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

    def test_empirical_proposal_requires_audited_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, point in enumerate(([0, 0], [0, 100], [100, 100])):
                image = root / f"{index}.png"
                image.write_bytes(b"image")
                sample = root / f"{index}.json"
                sample.write_text(json.dumps({
                    "name": f"p{index}", "board_xy_mm": point,
                    "raw_encoder_counts": dict.fromkeys(JOINTS, 2000), "camera_evidence": str(image),
                }))
                paths.append(sample)
            manifest = root / "samples.json"
            manifest.write_text(json.dumps({"samples": [str(path) for path in paths]}))
            samples, audit = load_audited_samples(manifest)
            self.assertEqual(len(samples), 3)
            self.assertTrue(audit["accepted"])
            paths[0].unlink()
            with self.assertRaisesRegex(ValueError, "contact evidence audit failed"):
                load_audited_samples(manifest)
