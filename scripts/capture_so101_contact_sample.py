"""Record one human-positioned SO-101 tabletop-contact calibration sample.

Run only after an operator has safely positioned the physical jaw-centre over a
known board point.  The program takes no motion action: it reads encoders once,
uses the official SO-101 URDF to derive the gripper-frame XY position, captures
one camera frame as evidence, and writes a contact JSON snippet for
``calibrate_robot_base_from_contacts.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import time

from scripts.preview_so101_fk import forward_kinematics, parse_urdf, raw_to_degrees
from harness.so101_step import NAMES, TEMPERATURE_STOP


PORT = "/dev/cu.usbmodem5A7C1240491"
CALIBRATION = Path("artifacts/calibration/so101_follower_kai_01/so101_follower_kai_01.json")
URDF = Path("artifacts/models/so101_new_calib.urdf")
CAPTURE_REGISTERS = ("Present_Position", "Torque_Enable", "Present_Load", "Present_Temperature", "Status")


def parse_pair(value: str) -> list[float]:
    try:
        result = [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be X,Y") from error
    if len(result) != 2:
        raise argparse.ArgumentTypeError("point must contain exactly X,Y")
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_capture_telemetry(bus) -> dict:
    return {register: bus.sync_read(register, normalize=False, num_retry=2)
            for register in CAPTURE_REGISTERS}


def validate_capture_telemetry(first: dict, second: dict, calibration: dict, *, max_settle_delta_counts: int) -> dict:
    """Reject a sample whose read-only hardware receipts are not stable/healthy.

    Contact sampling does not command an actuator.  It is nevertheless unsafe
    to turn a changing, faulted, overheated, or out-of-calibration state into a
    geometric measurement.  Load and torque are recorded rather than gated:
    a lightly supported tabletop contact can legitimately affect them.
    """
    if type(max_settle_delta_counts) is not int or max_settle_delta_counts < 0:
        raise ValueError('max settle delta must be a non-negative integer')
    for register in CAPTURE_REGISTERS:
        if set(first.get(register, ())) != set(NAMES) or set(second.get(register, ())) != set(NAMES):
            raise ValueError(f'capture telemetry missing complete {register}')
    deltas = {}
    for name in NAMES:
        position = second['Present_Position'][name]
        if not calibration[name]['range_min'] <= position <= calibration[name]['range_max']:
            raise ValueError(f'{name}: outside calibration during capture')
        if second['Status'][name] != 0:
            raise ValueError(f'{name}: nonzero status during capture')
        if second['Present_Temperature'][name] >= TEMPERATURE_STOP:
            raise ValueError(f'{name}: temperature stop during capture')
        delta = second['Present_Position'][name] - first['Present_Position'][name]
        deltas[name] = delta
        if abs(delta) > max_settle_delta_counts:
            raise ValueError(f'{name}: moved {delta} counts during contact capture')
    return {
        'registers': second,
        'inter_read_position_delta_counts': deltas,
        'max_accepted_settle_delta_counts': max_settle_delta_counts,
    }


def make_bus():
    """Load the optional hardware driver only for an actual capture."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
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
    import cv2
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
    parser.add_argument("--settle-seconds", type=float, default=0.25,
                        help="time between read-only receipts used to confirm the contact pose is still")
    parser.add_argument("--max-settle-delta-counts", type=int, default=2)
    parser.add_argument(
        "--confirm-jaw-centre",
        action="store_true",
        help="required acknowledgement that an operator physically placed the jaw centre at --board-point-mm",
    )
    args = parser.parse_args()

    if not args.confirm_jaw_centre:
        parser.error("refusing capture without --confirm-jaw-centre; this tool cannot verify physical contact itself")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    bus = make_bus()
    try:
        bus.connect()
        first_telemetry = read_capture_telemetry(bus)
        time.sleep(args.settle_seconds)
        second_telemetry = read_capture_telemetry(bus)
    finally:
        if bus.is_connected:
            # This diagnostic must preserve the current torque state.
            bus.disconnect(disable_torque=False)
    try:
        telemetry = validate_capture_telemetry(
            first_telemetry, second_telemetry, calibration,
            max_settle_delta_counts=args.max_settle_delta_counts,
        )
    except ValueError as error:
        raise SystemExit(f"REFUSED: {error}") from error
    raw = telemetry['registers']['Present_Position']
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
        "capture_provenance": {
            "schema_version": 3,
            "operator_confirmed_jaw_centre": True,
            "camera_index": args.camera_index,
            "settle_seconds": args.settle_seconds,
            "calibration": {"path": str(CALIBRATION), "sha256": sha256(CALIBRATION)},
            "urdf": {"path": str(URDF), "sha256": sha256(URDF)},
        },
        "hardware_telemetry": telemetry,
        "measurement_note": "Operator confirmed the physical jaw centre at board_xy_mm before capture; the software records, but cannot independently prove, that placement.",
    }
    args.output.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
