import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.derive_contact_coordinate_corrections import derive_sample


class ContactCoordinateCorrectionTests(unittest.TestCase):
    def test_derives_new_coordinates_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps({"name": "old", "board_xy_mm": [0, 297]}))
            derived = derive_sample(source, "inner_bl", [0, 172], "operator clarified inner board frame")
            self.assertEqual(json.loads(source.read_text())["board_xy_mm"], [0, 297])
            self.assertEqual(derived["name"], "inner_bl")
            self.assertEqual(derived["board_xy_mm"], [0.0, 172.0])
            correction = derived["coordinate_correction"]
            self.assertEqual(correction["original_board_xy_mm"], [0, 297])
            self.assertEqual(correction["source_sample_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_rejects_invalid_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps({"name": "old", "board_xy_mm": [0, 297]}))
            with self.assertRaisesRegex(ValueError, "two numeric"):
                derive_sample(source, "bad", [0], "reason")
