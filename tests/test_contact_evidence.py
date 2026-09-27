import json
from pathlib import Path
import tempfile
import unittest

from harness.contact_evidence import JOINTS, audit_contact_manifest


class ContactEvidenceTests(unittest.TestCase):
    def make_sample(self, root: Path, name: str, point: list[float], *, evidence=True):
        image = root / f'{name}.png'
        if evidence:
            image.write_bytes(name.encode())
        sample = root / f'{name}.json'
        sample.write_text(json.dumps({'name': name, 'board_xy_mm': point,
                                      'raw_encoder_counts': dict.fromkeys(JOINTS, 2000),
                                      'camera_evidence': str(image)}))
        return sample

    def test_hashes_complete_valid_evidence_without_claiming_motion_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [self.make_sample(root, f'p{index}', [float(index), 0.0]) for index in range(5)]
            manifest = root / 'samples.json'
            manifest.write_text(json.dumps({'samples': [str(path) for path in samples]}))
            audit = audit_contact_manifest(manifest)
            self.assertTrue(audit['accepted'])
            self.assertEqual(audit['valid_sample_count'], 5)
            self.assertEqual(len(audit['source_sha256']), 11)  # manifest + five JSON + five images
            self.assertTrue(audit['limitations'])

    def test_missing_evidence_or_duplicate_board_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [self.make_sample(root, f'p{index}', [0.0, 0.0], evidence=index != 0) for index in range(5)]
            manifest = root / 'samples.json'
            manifest.write_text(json.dumps({'samples': [str(path) for path in samples]}))
            audit = audit_contact_manifest(manifest)
            self.assertFalse(audit['accepted'])
            self.assertTrue(any('camera evidence is missing' in error for error in audit['errors']))
            self.assertTrue(any('duplicate board point' in error for error in audit['errors']))
