"""Record one human-positioned SO-101 tabletop-contact calibration sample.

Run only after an operator has safely positioned the physical jaw-centre over a
known board point.  The program takes no motion action: it reads encoders once,
uses the official SO-101 URDF to derive the gripper-frame XY position, captures
one camera frame as evidence, and writes a contact JSON snippet for
``calibrate_robot_base_from_contacts.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from scripts.preview_so101_fk import forward_kinematics, parse_urdf, raw_to_degrees


PORT = "/dev/cu.usbmodem5A7C1240491"
CALIBRATION = Path("artifacts/calibration/so101_follower_kai_01/so101_follower_kai_01.json")
URDF = Path("artifacts/models/so101_new_calib.urdf")


def parse_pair(value: str) -> list[float]:
    try:
        result = [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be X,Y") from error
    if len(result) != 2:
        raise argparse.ArgumentTypeError("point must contain exactly X,Y")
    return result


def make_bus() -> FeetechMotorsBus:
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    return FeetechMotorsBus(PORT, motors)


def capture_frame(camera_index: int, path: Path) -> None:
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    try:
        for _ in range(12):
            ok, frame = camera.read()
        if not ok:
            raise RuntimeError("camera frame capture failed")
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"could not write camera evidence: {path}")
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--board-point-mm", required=True, type=parse_pair)
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    bus = make_bus()
    try:
        bus.connect()
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    finally:
        if bus.is_connected:
            # This diagnostic must preserve the current torque state.
            bus.disconnect(disable_torque=False)
    pose = forward_kinematics(raw_to_degrees(raw, calibration), parse_urdf(URDF))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    image = args.output.parent / f"{timestamp}_{args.name}_contact_evidence.png"
    capture_frame(args.camera_index, image)
    sample = {
        "name": args.name,
        "board_xy_mm": [round(value, 3) for value in args.board_point_mm],
        "robot_tool_xy_mm": [round(float(value * 1000), 3) for value in pose[:2, 3]],
        "raw_encoder_counts": raw,
        "gripper_frame_z_mm": round(float(pose[2, 3] * 1000), 3),
        "camera_evidence": str(image),
        "measurement_note": "Operator must have placed the physical jaw centre at board_xy_mm before capture.",
    }
    args.output.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
