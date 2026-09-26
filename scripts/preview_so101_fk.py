"""Compute an SO-101 gripper-frame pose from saved raw encoder readings.

This tool is deliberately offline: it never opens a serial port and never
sends motor commands.  It parses the official SO-101 URDF and applies the
same calibrated-midpoint degree conversion used by LeRobot's motor bus.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ENCODER_MAX = 4095


def rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]],
        dtype=float,
    )


def transform(xyz: list[float], rpy: list[float], angle_rad: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = xyz
    z_rotation = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
         [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float,
    )
    return result @ z_rotation


def parse_urdf(urdf: Path) -> dict[str, tuple[list[float], list[float]]]:
    root = ET.parse(urdf).getroot()
    parsed = {}
    for element in root.findall("joint"):
        name = element.attrib["name"]
        if name not in {*JOINTS, "gripper_frame_joint"}:
            continue
        origin = element.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        parsed[name] = (xyz, rpy)
    missing = (set(JOINTS) | {"gripper_frame_joint"}) - set(parsed)
    if missing:
        raise ValueError(f"URDF lacks required joints: {sorted(missing)}")
    return parsed


def raw_to_degrees(raw: dict[str, int], calibration: dict[str, dict[str, int]]) -> dict[str, float]:
    return {
        name: (raw[name] - (calibration[name]["range_min"] + calibration[name]["range_max"]) / 2)
        * 360 / ENCODER_MAX
        for name in JOINTS
    }


def forward_kinematics(degrees: dict[str, float], urdf_joints: dict[str, tuple[list[float], list[float]]]) -> np.ndarray:
    pose = np.eye(4)
    for name in JOINTS:
        xyz, rpy = urdf_joints[name]
        pose = pose @ transform(xyz, rpy, np.deg2rad(degrees[name]))
    xyz, rpy = urdf_joints["gripper_frame_joint"]
    return pose @ transform(xyz, rpy)


def latest_raw_reading(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no samples in {path}")
    return {name: int(rows[-1][name]) for name in JOINTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--joint-log", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    raw = latest_raw_reading(args.joint_log)
    degrees = raw_to_degrees(raw, calibration)
    pose = forward_kinematics(degrees, parse_urdf(args.urdf))
    print(json.dumps({
        "mode": "offline_forward_kinematics_only",
        "raw_encoder_counts": raw,
        "joint_degrees": {key: round(value, 3) for key, value in degrees.items()},
        "gripper_frame_position_m": [round(float(value), 4) for value in pose[:3, 3]],
    }, indent=2))


if __name__ == "__main__":
    main()
