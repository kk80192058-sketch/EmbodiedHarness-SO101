"""Evidence-only readiness assessment for an SO-101 policy observation.

This is deliberately not an action gate and never produces a motor command.
It tells an operator or higher-level runtime why a captured observation is or
is not suitable for *further validation*.  A separate validated executor is
required before any physical action can be authorised.
"""
from __future__ import annotations

from typing import Any

from harness.so101_step import NAMES, state_fault


def assess_readiness(record: dict[str, Any], prediction: dict[str, Any], calibration: dict[str, Any], *,
                     max_camera_age_s: float = 0.5) -> dict[str, Any]:
    """Assess one saved observation and one shadow-policy receipt.

    ``motion_authorized`` is always false: passing this report only proves that
    the evidence is complete enough to investigate a supervised executor.
    """
    if max_camera_age_s <= 0:
        raise ValueError('max_camera_age_s must be positive')
    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    registers = record.get('registers')
    try:
        fault = state_fault(registers, calibration) if isinstance(registers, dict) else 'missing registers'
    except (KeyError, TypeError):
        fault = 'incomplete registers'
    checks['robot_state'] = {'passed': fault is None, 'detail': fault}
    if fault is not None:
        reasons.append(f'robot state gate failed: {fault}')

    cameras = record.get('cameras')
    vision_ok = True
    if not isinstance(cameras, dict):
        cameras = {}
    for role in ('wrist', 'global'):
        camera = cameras.get(role)
        target = record.get('red_block', {}).get(role, {}) if isinstance(record.get('red_block'), dict) else {}
        camera_ok = isinstance(camera, dict) and isinstance(camera.get('age_at_read_s'), (int, float)) \
            and 0 <= camera['age_at_read_s'] <= max_camera_age_s
        visible = target.get('status') == 'visible_unique' and target.get('target') is not None
        passed = camera_ok and visible
        checks[f'{role}_observation'] = {'passed': passed, 'camera_fresh': camera_ok,
                                         'target_visible_unique': visible}
        if not passed:
            vision_ok = False
            reasons.append(f'{role} camera needs a fresh, unambiguous target observation')

    shadow_only = prediction.get('mode') == 'shadow_only' and prediction.get('hardware_writes') == 0
    assessment = prediction.get('execution_assessment')
    compatible = isinstance(assessment, dict) and assessment.get('hardware_execution_permitted') is True
    checks['policy_receipt'] = {'passed': shadow_only, 'shadow_only': shadow_only,
                                'proposal_compatible': compatible}
    if not shadow_only:
        reasons.append('policy receipt is not a verified zero-write shadow result')
    if not compatible:
        detail = assessment.get('reasons') if isinstance(assessment, dict) else None
        reasons.append('policy proposal is not compatible with current safety gates' +
                       (f': {detail}' if detail else ''))

    observation_healthy = checks['robot_state']['passed'] and vision_ok
    return {
        'mode': 'evidence_only_readiness',
        'motion_authorized': False,
        'observation_healthy': observation_healthy,
        'policy_proposal_compatible': compatible,
        'ready_for_supervised_execution': False,
        'checks': checks,
        'reasons': reasons + ['No validated multi-joint real-hardware executor is installed'],
        'required_next_gate': 'validated geometry, scene adaptation, and supervised executor acceptance',
    }
