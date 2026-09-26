"""Capture one synchronized, no-motion SO-101 kinematic/visual proxy sample.

This diagnostic records the actual encoder state, the official-URDF gripper
frame pose, and a template-tracked red-star/wrist-housing proxy in one camera
frame.  It never writes Goal_Position, enables/disables torque, or otherwise
commands hardware.  A collection of these samples supports later automatic
camera-to-robot fitting from bounded, reversible motion probes.
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
from scripts.track_gripper_star_template import ensure_in_bounds, load_gray, parse_box, track
from scripts.detect_visual_proxies import detect, red_mask


PORT = "/dev/cu.usbmodem5A7C1240491"
CALIBRATION = Path("artifacts/calibration/so101_follower_kai_01/so101_follower_kai_01.json")
URDF = Path("artifacts/models/so101_new_calib.urdf")


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
            raise RuntimeError(f"could not write frame: {path}")
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--template-box", type=parse_box, required=True)
    parser.add_argument("--search-box", type=parse_box, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.95)
    parser.add_argument("--star-config", type=Path, default=Path("configs/vision_proxies.json"))
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_score <= 1.0:
        parser.error("--minimum-score must be between 0 and 1")

    bus = make_bus()
    try:
        bus.connect()
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    finally:
        if bus.is_connected:
            # Preserve the exact torque state observed at entry.
            bus.disconnect(disable_torque=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    frame_path = args.output.parent / f"{timestamp}_{args.label}_proxy_pose.png"
    capture_frame(args.camera_index, frame_path)
    reference = load_gray(args.reference_image)
    ensure_in_bounds(args.template_box, reference, "template")
    x1, y1, x2, y2 = args.template_box
    template = reference[y1:y2, x1:x2]
    frame_gray = load_gray(frame_path)
    score, center = track(frame_gray, template, args.search_box)
    tracking_method = "wrist_housing_template"
    fallback = None
    if score < args.minimum_score:
        # A pure template legitimately changes under perspective and roll.  The
        # red star is a second, spatially constrained physical cue, never a
        # global 'any red pixel' fallback.
        star_config = json.loads(args.star_config.read_text(encoding="utf-8"))["gripper_star"]
        color = cv2.imread(str(frame_path))
        fallback = detect(
            red_mask(color), star_config["roi_px"], star_config["minimum_area_px"],
            tuple(star_config["expected_center_px"]),
        )
        center = fallback["center_px"]
        tracking_method = "roi_constrained_red_star_fallback"
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    degrees = raw_to_degrees(raw, calibration)
    pose = forward_kinematics(degrees, parse_urdf(URDF))
    result = {
        "schema_version": 1,
        "mode": "read_only_synchronized_proxy_pose",
        "hardware_motion": False,
        "label": args.label,
        "captured_at": datetime.now(UTC).isoformat(),
        "frame": str(frame_path),
        "reference_image": str(args.reference_image),
        "proxy_center_px": center,
        "proxy_match_score": round(score, 5),
        "proxy_tracking_method": tracking_method,
        "red_star_fallback": fallback,
        "raw_encoder_counts": raw,
        "joint_degrees": {name: round(value, 5) for name, value in degrees.items()},
        "urdf_gripper_frame_position_mm": [round(float(value * 1000), 4) for value in pose[:3, 3]],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
