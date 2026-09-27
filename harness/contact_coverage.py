"""Compare a live/saved joint state with audited contact-sample coverage.

This is a diagnostic only.  Being inside a per-joint envelope is not a proof
of reachability, collision clearance, or a valid grasp trajectory; being
outside it does prove that the current state cannot be described as an already
observed contact configuration.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from harness.contact_evidence import JOINTS


def assess_contact_pose_coverage(samples: list[dict[str, Any]], current_raw: dict[str, int]) -> dict[str, Any]:
    """Return transparent per-joint coverage information for one raw state."""
    if not samples:
        raise ValueError('at least one contact sample is required')
    if set(current_raw) != set(JOINTS) or not all(type(current_raw[joint]) is int for joint in JOINTS):
        raise ValueError('current state must contain integer raw counts for every joint')
    try:
        matrix = np.asarray([[sample['raw_encoder_counts'][joint] for joint in JOINTS] for sample in samples], dtype=int)
        names = [sample['name'] for sample in samples]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'invalid audited contact sample: {error}') from error
    current = np.asarray([current_raw[joint] for joint in JOINTS], dtype=int)
    lower, upper = matrix.min(axis=0), matrix.max(axis=0)
    inside = (lower <= current) & (current <= upper)
    deltas = matrix - current
    nearest_index = int(np.argmin(np.abs(deltas).max(axis=1)))
    return {
        'mode': 'offline_contact_joint_coverage',
        'hardware_access': False,
        'sample_count': len(samples),
        'current_raw_encoder_counts': dict(current_raw),
        'joint_ranges': {joint: {'min': int(low), 'max': int(high), 'current': int(value),
                                 'inside_observed_envelope': bool(ok)}
                         for joint, low, high, value, ok in zip(JOINTS, lower, upper, current, inside, strict=True)},
        'inside_all_observed_joint_envelopes': bool(inside.all()),
        'outside_joints': [joint for joint, ok in zip(JOINTS, inside, strict=True) if not ok],
        'nearest_sample': {
            'name': names[nearest_index],
            'joint_deltas_counts': {joint: int(value) for joint, value in zip(JOINTS, deltas[nearest_index], strict=True)},
            'max_abs_joint_delta_counts': int(np.abs(deltas[nearest_index]).max()),
        },
        'limitations': [
            'The envelope ignores coupled joint geometry and is not an interpolation, IK, collision, or force check.',
            'Inside coverage cannot authorise motion; outside coverage only rejects the claim that this is a previously observed contact pose.',
        ],
    }
