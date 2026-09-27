"""Offline SmolVLA inference and explicit encoder/action unit conversion.

This module has no motor I/O. A prediction is a proposal, never an execution
receipt. Keep checkpoint statistics and the locally calibrated joint mapping
separate: policy mean/std normalization belongs to the checkpoint processors.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from harness.so101_step import NAMES


def evidence_manifest(observation_path, record):
    """Hash the exact saved observation and camera frames used by a prediction.

    Prediction receipts are diagnostic evidence, not commands. Binding them to
    content hashes prevents a later analysis from accidentally treating a
    different frame at the same path as the policy input.
    """
    observation_path = Path(observation_path).resolve()
    if not observation_path.is_file():
        raise ValueError('Saved observation JSON is missing')
    cameras = record.get('cameras')
    if not isinstance(cameras, dict):
        raise ValueError('Observation has no camera metadata')
    frames = {}
    for role in ('wrist', 'global'):
        metadata = cameras.get(role)
        path = Path(metadata['path']).resolve() if isinstance(metadata, dict) and 'path' in metadata else None
        if path is None or not path.is_file():
            raise ValueError(f'Observation is missing saved {role} frame')
        with path.open('rb') as stream:
            frames[role] = {'path': str(path), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    with observation_path.open('rb') as stream:
        observation_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'observation_json': {'path': str(observation_path), 'sha256': observation_hash},
            'camera_frames': frames}


def raw_to_policy(raw, calibration, *, body_units):
    if body_units not in ('degrees', 'range_m100_100'):
        raise ValueError('Specify checkpoint body units explicitly')
    result = []
    for name in NAMES:
        value = raw[name]
        low, high = (calibration[name][k] for k in ('range_min', 'range_max'))
        if not math.isfinite(value) or not low <= value <= high or low >= high:
            raise ValueError(f'Invalid calibrated position: {name}')
        if calibration[name]['drive_mode'] != 0:
            raise ValueError('Nonzero drive mode needs explicit device-specific mapping')
        if name == 'gripper':
            result.append((value - low) * 100 / (high - low))
        elif body_units == 'degrees':
            result.append((value - (low + high) / 2) * 360 / 4095)
        else:
            result.append((value - low) * 200 / (high - low) - 100)
    return result


def policy_to_raw(action, calibration, *, body_units):
    if body_units not in ('degrees', 'range_m100_100') or len(action) != 6:
        raise ValueError('Explicit units and six ordered actions required')
    result = {}
    for name, value in zip(NAMES, action, strict=True):
        if not math.isfinite(value):
            raise ValueError('Nonfinite policy action')
        entry = calibration[name]
        low, high = entry['range_min'], entry['range_max']
        if low >= high or entry['drive_mode'] != 0:
            raise ValueError('Unsupported calibration')
        if name == 'gripper':
            raw = low + value / 100 * (high - low)
        elif body_units == 'degrees':
            raw = (low + high) / 2 + value * 4095 / 360
        else:
            raw = low + (value + 100) / 200 * (high - low)
        # Never silently clamp a model's out-of-range proposal.
        if not low <= raw <= high:
            raise ValueError(f'Policy target outside calibration: {name}={raw}')
        result[name] = round(raw)
    return result


def verify_checkpoint(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'hub_metadata.json').read_text())
    weight_info = next(s for s in metadata['siblings'] if s['rfilename'] == 'model.safetensors')
    path = folder / 'model.safetensors'
    if path.stat().st_size != weight_info['size']:
        raise ValueError('Incomplete model download')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != weight_info['lfs']['sha256']:
        raise ValueError('Model checksum does not match pinned Hub metadata')
    dataset = json.loads((folder / 'dataset_info.json').read_text())
    for feature in ('action', 'observation.state'):
        if dataset['features'][feature]['names'] != [f'{n}.pos' for n in NAMES]:
            raise ValueError('Checkpoint dataset joint order mismatch')
    return {'repo_id': metadata['id'], 'revision': metadata['sha'], 'weight_sha256': digest}


def assess_proposal(result, record, calibration, config):
    """Diagnose a shadow proposal; never promote it into a motor command."""
    from harness.so101_step import validate
    state = record['registers']
    first = result['raw_proposals'][0]
    report = {'hardware_execution_permitted': False, 'reasons': [], 'joint_checks': {}}
    normalized = result['normalized_state'][0]
    report['out_of_distribution_joints'] = [name for name, z in zip(NAMES, normalized, strict=True) if abs(z) > 3]
    if report['out_of_distribution_joints']:
        report['reasons'].append('State exceeds three training standard deviations; calibration/scene adaptation required')
    if first['raw_target'] is None:
        report['reasons'].append(first['conversion_error'])
        return report
    deltas = {n: first['raw_target'][n] - state['Present_Position'][n] for n in NAMES}
    if sum(v != 0 for v in deltas.values()) > 1:
        report['reasons'].append('Policy requests simultaneous joints; active executor allows one joint per segment')
    for name, delta in deltas.items():
        try:
            if delta:
                validate(state, calibration, config, name, delta)
            report['joint_checks'][name] = {'delta': delta, 'within_step_gate': True}
        except ValueError as error:
            report['joint_checks'][name] = {'delta': delta, 'within_step_gate': False, 'reason': str(error)}
    report['reasons'].append('Checkpoint has not passed scene/closed-loop execution validation')
    return report


class SmolVLAShadowPolicy:
    def __init__(self, folder, vlm_folder, *, device='mps', body_units='degrees'):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch = torch
        self.device = device
        self.body_units = body_units
        self.identity = verify_checkpoint(folder)
        self.identity['vlm'] = json.loads((Path(vlm_folder) / 'source.json').read_text())
        config = PreTrainedConfig.from_pretrained(str(folder), local_files_only=True)
        if config.type != 'smolvla' or tuple(config.output_features['action'].shape) != (6,):
            raise ValueError('Unsupported policy type/action shape')
        expected = {'observation.state', 'observation.images.fixed', 'observation.images.handy'}
        if set(config.input_features) != expected:
            raise ValueError('Unexpected policy camera/state contract')
        config.device = device
        config.vlm_model_name = str(Path(vlm_folder).resolve())
        config.load_vlm_weights = False
        start = time.monotonic()
        self.policy = SmolVLAPolicy.from_pretrained(str(folder), config=config,
                                                   local_files_only=True, strict=True)
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(folder),
            preprocessor_overrides={'device_processor': {'device': device},
                                    'tokenizer_processor': {'tokenizer_name': config.vlm_model_name}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.load_seconds = time.monotonic() - start
        self.config = config

    def predict(self, record, images, calibration, task, *, seed=0):
        import cv2
        import numpy as np
        torch = self.torch
        state = raw_to_policy(record['registers']['Present_Position'], calibration,
                              body_units=self.body_units)
        batch = {'observation.state': torch.tensor(state, dtype=torch.float32), 'task': task}
        for role, key in [('global', 'fixed'), ('wrist', 'handy')]:
            feature = f'observation.images.{key}'
            rgb = cv2.cvtColor(images[role], cv2.COLOR_BGR2RGB)
            # Preserve source geometry. LeRobot's policy performs its own
            # aspect-preserving resize/pad to config.resize_imgs_with_padding.
            batch[feature] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        torch.manual_seed(seed)
        start = time.monotonic()
        self.policy.reset()
        with torch.inference_mode():
            processed = self.pre(batch)
            normalized = self.policy.predict_action_chunk(processed)
            actions = self.post(normalized).detach().cpu()
        if self.device == 'mps':
            torch.mps.synchronize()
        latency = time.monotonic() - start
        if actions.shape != (1, self.config.chunk_size, 6) or not torch.isfinite(actions).all():
            raise ValueError(f'Invalid predicted action chunk: {tuple(actions.shape)}')
        proposals = []
        for action in actions[0].tolist():
            try:
                raw = policy_to_raw(action, calibration, body_units=self.body_units)
                proposals.append({'raw_target': raw, 'conversion_error': None})
            except ValueError as error:
                proposals.append({'raw_target': None, 'conversion_error': str(error)})
        return {'mode': 'shadow_only', 'hardware_writes': 0, 'model': self.identity,
                'task': task, 'body_units': self.body_units, 'unit_status': 'dataset statistics support degrees; original recording config unavailable',
                'state': state, 'seed': seed, 'device': self.device, 'load_s': self.load_seconds,
                'inference_s': latency, 'actions': actions[0].tolist(),
                'raw_proposals': proposals, 'grasp_status': 'unknown',
                'normalized_state': processed['observation.state'].detach().cpu().tolist()}
