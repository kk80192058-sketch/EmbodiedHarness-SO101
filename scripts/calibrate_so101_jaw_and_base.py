"""Fit the physical jaw-centre offset and SO-101 base-to-board alignment.

Unlike a naive gripper-frame fit, this solver models the fact that the URDF
``gripper_frame`` is not necessarily the midpoint between the two physical
jaws.  From four or more human-confirmed tabletop contacts, it jointly fits:

* a fixed 3-D offset from the URDF gripper frame to the jaw centre; and
* a rigid 2-D transform from robot base XY to A4 board XY.

It is offline and never accesses motors or a camera.  A solution is rejected
unless it has an independent residual check (four or more contacts) and meets
the requested RMS bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.preview_so101_fk import forward_kinematics, parse_urdf, raw_to_degrees


def predict(parameters: np.ndarray, poses_mm: np.ndarray) -> np.ndarray:
    """Map gripper-frame poses to predicted board XY for parameters.

    ``poses_mm`` is shape (N, 3, 4): rotation then translation in millimetres.
    Parameters are [theta_rad, tx_mm, ty_mm, offset_x_mm, offset_y_mm, offset_z_mm].
    """
    theta, tx, ty, *offset = parameters
    rotation_2d = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset_array = np.asarray(offset, dtype=float)
    jaw_robot = np.einsum("nij,j->ni", poses_mm[:, :, :3], offset_array) + poses_mm[:, :, 3]
    return (rotation_2d @ jaw_robot[:, :2].T).T + np.array([tx, ty])


def residual_vector(parameters: np.ndarray, poses_mm: np.ndarray, board_xy_mm: np.ndarray) -> np.ndarray:
    return (predict(parameters, poses_mm) - board_xy_mm).reshape(-1)


def finite_difference_jacobian(parameters: np.ndarray, poses_mm: np.ndarray, board_xy_mm: np.ndarray) -> np.ndarray:
    base = residual_vector(parameters, poses_mm, board_xy_mm)
    steps = np.array([1e-6, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    columns = []
    for index, step in enumerate(steps):
        shifted = parameters.copy()
        shifted[index] += step
        columns.append((residual_vector(shifted, poses_mm, board_xy_mm) - base) / step)
    return np.column_stack(columns)


def fit_joint_alignment(poses_mm: np.ndarray, board_xy_mm: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Damped Gauss-Newton fit with conservative, finite-difference derivatives."""
    if poses_mm.ndim != 3 or poses_mm.shape[1:] != (3, 4):
        raise ValueError("poses_mm must have shape (N, 3, 4)")
    if board_xy_mm.shape != (len(poses_mm), 2):
        raise ValueError("board_xy_mm must have shape (N, 2)")
    if len(poses_mm) < 4:
        raise ValueError("at least four contacts are required for a verified jaw-centre fit")
    # Start with a zero jaw offset and align centroids.  Multi-start theta avoids
    # a local orientation ambiguity when the arm samples form a narrow arc.
    candidates = []
    for theta in np.linspace(-np.pi, np.pi, 25, endpoint=False):
        jaw_xy = poses_mm[:, :2, 3]
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        translation = board_xy_mm.mean(axis=0) - (r @ jaw_xy.mean(axis=0))
        parameters = np.array([theta, translation[0], translation[1], 0.0, 0.0, 0.0])
        damping = 1e-3
        for iteration in range(120):
            residual = residual_vector(parameters, poses_mm, board_xy_mm)
            jacobian = finite_difference_jacobian(parameters, poses_mm, board_xy_mm)
            normal = jacobian.T @ jacobian + damping * np.eye(6)
            update = np.linalg.solve(normal, jacobian.T @ residual)
            trial = parameters - update
            if np.linalg.norm(residual_vector(trial, poses_mm, board_xy_mm)) < np.linalg.norm(residual):
                parameters = trial
                damping = max(damping / 2, 1e-9)
                if np.linalg.norm(update) < 1e-5:
                    break
            else:
                damping *= 10
        rms = float(np.sqrt(np.mean(residual_vector(parameters, poses_mm, board_xy_mm) ** 2)))
        candidates.append((rms, parameters, iteration + 1))
    rms, parameters, iterations = min(candidates, key=lambda item: item[0])
    return parameters, rms, iterations


def pose_for_sample(sample: dict, calibration: dict, urdf_joints: dict) -> np.ndarray:
    raw = sample["raw_encoder_counts"]
    pose_m = forward_kinematics(raw_to_degrees(raw, calibration), urdf_joints)
    pose_mm = pose_m[:3, :].copy()
    pose_mm[:, 3] *= 1000
    return pose_mm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="JSON list: {\"samples\": [\"sample.json\", ...]}")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--max-rms-error-mm", type=float, default=3.0)
    parser.add_argument("--max-jaw-offset-mm", type=float, default=150.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.samples.read_text(encoding="utf-8"))
    paths = [Path(item) for item in manifest.get("samples", [])]
    if len(paths) < 4:
        parser.error("sample manifest needs at least four contact sample paths")
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    urdf_joints = parse_urdf(args.urdf)
    poses = np.asarray([pose_for_sample(sample, calibration, urdf_joints) for sample in samples])
    board = np.asarray([sample["board_xy_mm"] for sample in samples], dtype=float)
    parameters, rms, iterations = fit_joint_alignment(poses, board)
    theta, tx, ty, ox, oy, oz = parameters
    offset_norm = float(np.linalg.norm([ox, oy, oz]))
    accepted = rms <= args.max_rms_error_mm and offset_norm <= args.max_jaw_offset_mm
    result = {
        "schema_version": 1,
        "mode": "offline_joint_jaw_offset_and_base_alignment",
        "hardware_access": False,
        "sample_count": len(samples),
        "sample_names": [sample["name"] for sample in samples],
        "jaw_centre_offset_from_urdf_gripper_frame_mm": [round(float(value), 5) for value in (ox, oy, oz)],
        "jaw_offset_norm_mm": round(offset_norm, 5),
        "robot_xy_mm_to_board_xy_mm": {
            "rotation_rad": round(float(theta), 9),
            "translation_mm": [round(float(tx), 5), round(float(ty), 5)],
        },
        "fit_rms_error_mm": round(rms, 5),
        "iterations": iterations,
        "maximum_accepted_rms_error_mm": args.max_rms_error_mm,
        "maximum_accepted_jaw_offset_mm": args.max_jaw_offset_mm,
        "accepted": accepted,
        "limits": [
            "Acceptance aligns the physical jaw centre to the board plane only.",
            "It does not prove collision clearance, grasp force, or object height.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise SystemExit("FAIL: jaw/base calibration failed its evidence gates")


if __name__ == "__main__":
    main()
