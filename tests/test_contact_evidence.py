import json
import hashlib
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

    def test_v2_capture_provenance_is_hashed_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration, urdf = root / 'calibration.json', root / 'robot.urdf'
            calibration.write_text('{"version": 1}')
            urdf.write_text('<robot/>')
            samples = [self.make_sample(root, f'p{index}', [float(index), 0.0]) for index in range(5)]
            payload = json.loads(samples[0].read_text())
            payload['capture_provenance'] = {
                'schema_version': 2,
                'operator_confirmed_jaw_centre': True,
                'camera_index': 1,
                'calibration': {'path': str(calibration), 'sha256': hashlib.sha256(calibration.read_bytes()).hexdigest()},
                'urdf': {'path': str(urdf), 'sha256': hashlib.sha256(urdf.read_bytes()).hexdigest()},
            }
            samples[0].write_text(json.dumps(payload))
            manifest = root / 'samples.json'
            manifest.write_text(json.dumps({'samples': [str(path) for path in samples]}))
            audit = audit_contact_manifest(manifest)
            self.assertTrue(audit['accepted'])
            self.assertIn(str(calibration.resolve()), audit['source_sha256'])
            calibration.write_text('{"version": 2}')
            audit = audit_contact_manifest(manifest)
            self.assertFalse(audit['accepted'])
            self.assertTrue(any('calibration sha256 mismatch' in error for error in audit['errors']))
