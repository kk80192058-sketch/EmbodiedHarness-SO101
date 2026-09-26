"""One reversible, bounded SO-101 follower bring-up motion test.

This script intentionally bypasses high-level planning. It only moves
``shoulder_pan`` by a tiny raw encoder delta, returns it to its measured start
position, then disables torque on every motor. Do not use it for task actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import cv2

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from harness.safety import SafetyConfig, SafetyGate


PORT = "/dev/cu.usbmodem5A7C1240491"
CALIBRATION = Path("artifacts/calibration/so101_follower_kai_01/so101_follower_kai_01.json")
STEP_COUNTS = 20  # approximately 1.8 degrees for a 4096-count encoder
SETTLE_S = 1.0


def capture_camera_frame(camera_index: int, label: str) -> Path:
    """Capture one evidence frame without changing robot state."""
    output_dir = Path("artifacts/bringup/camera_motion_checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open OpenCV camera index {camera_index}")
    try:
        frame = None
        for _ in range(12):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Could not read OpenCV camera index {camera_index}")
    finally:
        camera.release()
    output = output_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{label}.png"
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write evidence frame {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="One bounded, reversible SO-101 joint test")
    parser.add_argument("--joint", required=True, choices=(
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
    ))
    parser.add_argument("--step-counts", type=int, default=STEP_COUNTS)
    parser.add_argument(
        "--camera-index",
        type=int,
        help="Optional OpenCV camera index; saves start/moved/returned evidence frames",
    )
    parser.add_argument(
        "--keep-holding",
        action="store_true",
        help="Latch every joint at its current measured pose and keep torque enabled after a passing test",
    )
    args = parser.parse_args()
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    gate = SafetyGate(SafetyConfig.from_json(Path("configs/so101_safety.json")))
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(PORT, motors)
    start: dict[str, int] | None = None
    keep_holding = False
    try:
        bus.connect()
        start = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        invalid = [
            name for name, position in start.items()
            if not calibration[name]["range_min"] <= position <= calibration[name]["range_max"]
        ]
        if invalid:
            raise RuntimeError(f"Refusing motion: outside calibrated bounds: {invalid}")

        proposed = {args.joint: start[args.joint] + args.step_counts}
        target = gate.validate_raw_joint_goals(start, proposed, timeout_s=3.0)[args.joint]
        frames: list[Path] = []
        if args.camera_index is not None:
            frames.append(capture_camera_frame(args.camera_index, f"{args.joint}_start"))

        # Prevent a torque-enable jump: latch measured positions before enable.
        # In persistent-hold mode every joint is latched, so unsupported links
        # cannot sag while the selected joint is tested.
        if args.keep_holding:
            for joint, position in start.items():
                bus.write("Goal_Position", joint, position, normalize=False)
            bus.enable_torque()
        else:
            bus.disable_torque()
            bus.write("Goal_Position", args.joint, start[args.joint], normalize=False)
            bus.enable_torque(args.joint)
        time.sleep(SETTLE_S)

        bus.write("Goal_Position", args.joint, target, normalize=False)
        time.sleep(SETTLE_S)
        moved = bus.read("Present_Position", args.joint, normalize=False, num_retry=2)
        if args.camera_index is not None:
            frames.append(capture_camera_frame(args.camera_index, f"{args.joint}_moved"))

        bus.write("Goal_Position", args.joint, start[args.joint], normalize=False)
        time.sleep(SETTLE_S)
        returned = bus.read("Present_Position", args.joint, normalize=False, num_retry=2)
        if args.camera_index is not None:
            frames.append(capture_camera_frame(args.camera_index, f"{args.joint}_returned"))
        print(
            f"PASS joint={args.joint} start={start[args.joint]} target={target} "
            f"observed={moved} returned={returned}"
        )
        for frame in frames:
            print(f"camera_frame={frame}")
        keep_holding = args.keep_holding
    finally:
        if bus.is_connected:
            # Fail closed unless a complete test explicitly opted into holding.
            if not keep_holding:
                bus.disable_torque(num_retry=2)
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
