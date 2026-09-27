"""One explicitly requested, monitored raw step; preserve existing torque state.

Default is read-only. Never changes calibration, torque, gains or other joints.
Every invocation writes a new evidence directory, including images and telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from harness.safety import SafetyConfig
from harness.so101_step import (NAMES, REGISTERS, execute_step, LOAD_STOP_RAW,
                               LOAD_PEAK_STOP_RAW, LOAD_MAX_ELEVATED_S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', choices=NAMES)
    parser.add_argument('--delta', type=int)
    args = parser.parse_args()
    if (args.joint is None) != (args.delta is None):
        parser.error('Supply both joint and delta, or neither for read-only')
    config_raw = json.loads(Path('configs/so101_safety.json').read_text())
    config = SafetyConfig.from_json(Path('configs/so101_safety.json'))
    calibration = json.loads(Path(config_raw['calibration_file']).read_text())
    directory = Path('artifacts/bringup/bounded_steps') / (time.strftime('%Y%m%dT%H%M%S') + f'_{time.time_ns() % 1000000:06d}')
    directory.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 2, 'mode': 'read_only' if args.joint is None else 'single_joint_bounded_step',
              'joint': args.joint, 'delta': args.delta, 'samples': [], 'writes': []}
    record['load_policy'] = {'preflight_max_raw_exclusive': LOAD_STOP_RAW,
                            'immediate_stop_raw': LOAD_PEAK_STOP_RAW,
                            'sustained_stop_raw': LOAD_STOP_RAW,
                            'maximum_elevated_seconds': LOAD_MAX_ELEVATED_S,
                            'note': 'Session supervision limits, not manufacturer overload ratings; no motor parameter writes.'}
    bus = FeetechMotorsBus(config_raw['port'], {
        name: Motor(calibration[name]['id'], 'sts3215',
                    MotorNormMode.DEGREES if name != 'gripper' else MotorNormMode.RANGE_0_100)
        for name in NAMES})
    camera = cv2.VideoCapture(1)

    def save():
        temporary = directory / 'state.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(directory / 'state.json')

    def emit(event, payload):
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': event, 'time': time.time(), 'payload': payload}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if event == 'sample':
            record['samples'].append(payload)
        elif event == 'before':
            record['before'] = payload
        elif event == 'result':
            record.update(payload)
        save()

    def frame(label):
        for _ in range(5):
            ok, image = camera.read()
            if not ok:
                raise RuntimeError('Camera read failed')
        if not cv2.imwrite(str(directory / f'{label}.png'), image):
            raise RuntimeError('Image save failed')

    def snapshot(registers=REGISTERS):
        return {reg: {name: bus.read(reg, name, normalize=False, num_retry=2)
                      for name in NAMES} for reg in registers}

    def write(goal, reason):
        intent = {'joint': args.joint, 'goal': goal, 'reason': reason, 'time': time.time()}
        record['writes'].append(intent)
        emit('write_intent', intent)
        bus.write('Goal_Position', args.joint, goal, normalize=False)
        emit('write_acknowledged', intent)

    try:
        if not camera.isOpened():
            raise RuntimeError('Camera 1 unavailable')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frame('before')
        bus.connect()
        if args.joint is not None:
            execute_step(snapshot, write, emit, calibration, config, args.joint, args.delta,
                         clock=time.monotonic, sleep=time.sleep)
        else:
            record['before'] = snapshot()
            emit('before', record['before'])
            record['after'] = record['before']
            record['outcome'] = 'read_only'
            record['success'] = None
            emit('read_only_result', record['after'])
        frame('after')
        record['capture_complete'] = True
    except BaseException as error:
        record['error'] = repr(error)
        record['success'] = False
        record['outcome'] = 'evidence_or_io_error'
        raise
    finally:
        save()
        camera.release()
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        print('evidence_dir=' + str(directory.resolve()), flush=True)
        print(json.dumps({k: v for k, v in record.items() if k != 'samples'}, indent=2), flush=True)
    if args.joint is not None and not record.get('success'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
