"""Persistent SO-101 I/O without configuration, calibration, or torque writes.

Only ``step`` may command hardware; it delegates to the existing bounded
single-joint supervisor. Camera frames carry acquisition receipt timestamps,
not sensor exposure timestamps. No network or policy inference runs here.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from harness.safety import SafetyConfig
from harness.so101_step import NAMES, REGISTERS, execute_step


class CameraStream:
    def __init__(self, index):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f'Camera {index} unavailable')
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError('Camera acquisition failed')
                with self.lock:
                    self.latest = (time.monotonic(), time.time(), frame)
        except Exception as error:
            self.error = error
        finally:
            self.capture.release()

    def read(self, max_age=0.5):
        if self.error:
            raise RuntimeError('Camera stream failed') from self.error
        with self.lock:
            if self.latest is None:
                raise RuntimeError('Camera has no frame')
            mono, wall, frame = self.latest
            age = time.monotonic() - mono
            if age > max_age:
                raise RuntimeError(f'Stale camera frame: {age:.3f}s')
            return frame.copy(), {'received_monotonic_s': mono, 'received_unix_s': wall,
                                  'age_at_read_s': age, 'shape': list(frame.shape)}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


class SO101Session:
    def __init__(self, output: Path, *, config_path=Path('configs/so101_safety.json'),
                 cameras=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.raw_config = json.loads(config_path.read_text())
        self.config = SafetyConfig.from_json(config_path)
        self.calibration_path = Path(self.raw_config['calibration_file'])
        self.calibration = json.loads(self.calibration_path.read_text())
        self.source_paths = [config_path, self.calibration_path]
        self.source_hashes = self._hashes()
        self.camera_indices = cameras or {'wrist': 0, 'global': 1}
        self.cameras = {}
        self.bus = None
        self.lock_file = None
        self.sequence = 0

    def _hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_paths}

    def event(self, kind, payload):
        with (self.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': kind, 'unix_s': time.time(),
                                     'monotonic_s': time.monotonic(), 'payload': payload}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def __enter__(self):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        try:
            self.lock_file = Path('artifacts/so101_serial.lock').open('a')
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.bus = FeetechMotorsBus(self.raw_config['port'], {
                name: Motor(self.calibration[name]['id'], 'sts3215', MotorNormMode.DEGREES)
                for name in NAMES})
            self.bus.connect()
            for role, index in self.camera_indices.items():
                self.cameras[role] = CameraStream(index)
            deadline = time.monotonic() + 10
            while any(cam.latest is None for cam in self.cameras.values()):
                if time.monotonic() > deadline or any(cam.error for cam in self.cameras.values()):
                    raise RuntimeError('Dual camera startup failed')
                time.sleep(0.02)
            self.event('connected_without_configuration', {
                'source_hashes': self.source_hashes, 'camera_indices': self.camera_indices,
                'camera_role_basis': '20260927 visual inventory; verify saved frames after reconnect'})
            return self
        except BaseException:
            self.close()
            raise

    def read(self, registers=REGISTERS):
        return {reg: {name: self.bus.read(reg, name, normalize=False, num_retry=2)
                      for name in NAMES} for reg in registers}

    def observe(self, label='observation'):
        import cv2
        from harness.red_block_tracking import detect_red_block
        self.sequence += 1
        folder = self.output / f'{self.sequence:04d}_{label}'
        folder.mkdir()
        started = time.monotonic()
        state = self.read()
        finished = time.monotonic()
        images, metadata = {}, {}
        for role, cam in self.cameras.items():
            images[role], metadata[role] = cam.read()
            path = folder / f'{role}.png'
            if not cv2.imwrite(str(path), images[role]):
                raise RuntimeError('Image persistence failed')
            metadata[role]['path'] = str(path.resolve())
        record = {'registers': state, 'state_read_interval_monotonic_s': [started, finished],
                  'cameras': metadata, 'unix_s': time.time(), 'grasp_status': 'unknown',
                  'red_block': {role: detect_red_block(frame, role) for role, frame in images.items()}}
        (folder / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
        self.event('observation', record)
        return record, images

    def step(self, joint, delta):
        self.observe('before_step')
        if (self.output / 'STOP').exists():
            raise RuntimeError('STOP requested')

        def monitored_read():
            for camera in self.cameras.values():
                camera.read()
            if (self.output / 'STOP').exists():
                raise RuntimeError('STOP requested')
            return self.read()

        def write(goal, reason):
            # Recovery reads are also fresh. For a camera/STOP error the existing
            # supervisor must still be able to latch the measured motor position.
            intent = {'joint': joint, 'goal': goal, 'reason': reason}
            self.event('write_intent', intent)
            self.bus.write('Goal_Position', joint, goal, normalize=False)
            self.event('write_acknowledged', intent)

        # Camera freshness is checked before motion; read remains available during
        # recovery. A monitor failure raises once, allowing subsequent hold read.
        monitor_failed = False
        def read():
            nonlocal monitor_failed
            if monitor_failed:
                return self.read()
            try:
                return monitored_read()
            except BaseException:
                monitor_failed = True
                raise

        result = execute_step(read, write, self.event, self.calibration, self.config,
                              joint, delta, clock=time.monotonic, sleep=time.sleep)
        self.observe('after_step')
        return result

    def close(self):
        try:
            if self.bus is not None and self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
        finally:
            for camera in self.cameras.values():
                camera.close()
            if self.lock_file is not None:
                self.lock_file.close()
            self.event('closed', {'torque_disable_requested': False,
                                 'sources_unchanged': self._hashes() == self.source_hashes})

    def __exit__(self, *exc):
        self.close()
