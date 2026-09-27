"""Integrity checks for human-positioned SO-101 contact calibration samples.

The audit is offline.  It establishes provenance and schema validity only; it
does not assert that an operator positioned a jaw centre correctly and it does
not make a fitted calibration motion-ready.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_captured_provenance(sample: dict[str, Any], root: Path, hashes: dict[str, str]) -> None:
    """Validate the stricter v2 metadata emitted by the capture utility.

    Legacy human-contact evidence predates this metadata, so it remains
    auditable.  When a sample claims v2 capture provenance, however, its
    calibration and URDF inputs must both be present and content-addressed.
    """
    provenance = sample.get('capture_provenance')
    if provenance is None:
        return
    if not isinstance(provenance, dict) or provenance.get('schema_version') != 2:
        raise ValueError('capture_provenance must use schema_version 2')
    if provenance.get('operator_confirmed_jaw_centre') is not True:
        raise ValueError('capture_provenance lacks operator jaw-centre confirmation')
    if not isinstance(provenance.get('camera_index'), int) or provenance['camera_index'] < 0:
        raise ValueError('capture_provenance camera_index must be a non-negative integer')
    for source_name in ('calibration', 'urdf'):
        source = provenance.get(source_name)
        if not isinstance(source, dict) or not isinstance(source.get('path'), str) or not isinstance(source.get('sha256'), str):
            raise ValueError(f'capture_provenance {source_name} needs path and sha256')
        path = _resolve(root, source['path']).resolve()
        if not path.is_file():
            raise ValueError(f'capture_provenance {source_name} file is missing')
        digest = _sha256(path)
        if digest != source['sha256']:
            raise ValueError(f'capture_provenance {source_name} sha256 mismatch')
        hashes[str(path)] = digest


def audit_contact_manifest(manifest_path: Path, *, root: Path = Path('.'), min_samples: int = 5) -> dict[str, Any]:
    """Read a manifest and return a deterministic, evidence-only audit.

    ``root`` gives meaning to repository-relative paths inside the manifest and
    sample JSON.  Every referenced input is hashed in the output so a later
    fitting step can verify it used the same evidence.
    """
    if min_samples < 3:
        raise ValueError('at least three samples are required for a contact audit')
    root = Path(root).resolve()
    manifest_path = _resolve(root, str(manifest_path)).resolve()
    result: dict[str, Any] = {'mode': 'offline_contact_evidence_audit', 'hardware_access': False,
                              'accepted': False, 'errors': [], 'samples': [], 'source_sha256': {}}
    if not manifest_path.is_file():
        result['errors'].append('contact manifest is missing')
        return result
    result['source_sha256'][str(manifest_path)] = _sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
        listed = manifest['samples']
        if not isinstance(listed, list):
            raise ValueError('samples is not a list')
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        result['errors'].append(f'invalid contact manifest: {error}')
        return result
    if len(listed) < min_samples:
        result['errors'].append(f'need at least {min_samples} sample paths')
    names, board_points = set(), set()
    for entry in listed:
        if not isinstance(entry, str):
            result['errors'].append('sample path is not a string')
            continue
        path = _resolve(root, entry).resolve()
        sample_result: dict[str, Any] = {'path': str(path), 'valid': False}
        result['samples'].append(sample_result)
        if not path.is_file():
            sample_result['error'] = 'sample JSON is missing'
            result['errors'].append(f'missing sample: {entry}')
            continue
        result['source_sha256'][str(path)] = _sha256(path)
        try:
            sample = json.loads(path.read_text())
            name = sample['name']
            board = sample['board_xy_mm']
            raw = sample['raw_encoder_counts']
            evidence = _resolve(root, sample['camera_evidence']).resolve()
            if not isinstance(name, str) or not name:
                raise ValueError('name is missing')
            if name in names:
                raise ValueError(f'duplicate sample name: {name}')
            if not isinstance(board, list) or len(board) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in board):
                raise ValueError('board_xy_mm must contain two finite values')
            if tuple(board) in board_points:
                raise ValueError(f'duplicate board point: {board}')
            if set(raw) != set(JOINTS) or not all(type(raw[joint]) is int and 0 <= raw[joint] <= 4095 for joint in JOINTS):
                raise ValueError('raw encoder counts must cover every joint with integer 0..4095 values')
            if not evidence.is_file():
                raise ValueError('camera evidence is missing')
            _validate_captured_provenance(sample, root, result['source_sha256'])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            sample_result['error'] = str(error)
            result['errors'].append(f'{entry}: {error}')
            continue
        names.add(name); board_points.add(tuple(board))
        result['source_sha256'][str(evidence)] = _sha256(evidence)
        sample_result.update(name=name, board_xy_mm=board, camera_evidence=str(evidence), valid=True)
    result['sample_count'] = len(result['samples'])
    result['valid_sample_count'] = sum(sample['valid'] for sample in result['samples'])
    result['accepted'] = not result['errors'] and result['valid_sample_count'] >= min_samples
    result['limitations'] = [
        'Hashes prove saved input identity, not the physical truth of a human-positioned contact.',
        'An accepted audit does not validate robot-base alignment, jaw offset, clearance, or grasp force.',
    ]
    return result
