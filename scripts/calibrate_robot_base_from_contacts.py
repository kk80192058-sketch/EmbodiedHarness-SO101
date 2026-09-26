"""Fit a robot-base-to-A4 transform from verified tabletop contact points.

The A4 calibration establishes camera pixels to table-plane millimetres.  This
tool supplies the missing *robot* side without pretending that a marker in the
air lies on the table.  Each input contact must be a deliberate, verified
touch of the physical jaw-centre point on a printed board point; it is a
calibration measurement, not a task demonstration.

This program is offline.  It does not open a camera or motor serial port and
never commands the SO-101.  It fails closed if the three or more contact pairs
cannot be explained by one rigid 2-D transform within the requested residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def board_point_from_pixel(homography: np.ndarray, pixel: list[float]) -> np.ndarray:
    homogeneous = homography @ np.array([pixel[0], pixel[1], 1.0])
    if abs(homogeneous[2]) < 1e-12:
        raise ValueError("pixel projects to infinity under tabletop homography")
    return homogeneous[:2] / homogeneous[2]


def fit_rigid_transform(robot_points: np.ndarray, board_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return R, t, rms for board ~= R @ robot + t, with no scale fitting."""
    if robot_points.shape != board_points.shape or robot_points.ndim != 2 or robot_points.shape[1] != 2:
        raise ValueError("robot and board points must both have shape (N, 2)")
    if len(robot_points) < 3:
        raise ValueError("at least three contact points are required for a verified fit")
    robot_center = robot_points.mean(axis=0)
    board_center = board_points.mean(axis=0)
    covariance = (robot_points - robot_center).T @ (board_points - board_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    translation = board_center - rotation @ robot_center
    residuals = (rotation @ robot_points.T).T + translation - board_points
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return rotation, translation, rms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-calibration", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rms-error-mm", type=float, default=3.0)
    args = parser.parse_args()
    if args.max_rms_error_mm <= 0:
        parser.error("--max-rms-error-mm must be positive")

    table = json.loads(args.table_calibration.read_text(encoding="utf-8"))
    contacts = json.loads(args.contacts.read_text(encoding="utf-8"))
    entries = contacts.get("contacts")
    if not isinstance(entries, list) or len(entries) < 3:
        parser.error("contacts JSON must contain at least three entries in a 'contacts' list")
    homography = np.asarray(table["homography_pixel_to_board_mm"], dtype=float)
    robot_points, board_points, used = [], [], []
    for entry in entries:
        try:
            robot = [float(value) for value in entry["robot_tool_xy_mm"]]
            pixel = [float(value) for value in entry["camera_px"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each contact needs numeric robot_tool_xy_mm and camera_px pairs") from error
        if len(robot) != 2 or len(pixel) != 2:
            raise ValueError("contact coordinate pairs must each contain exactly two values")
        board = board_point_from_pixel(homography, pixel)
        robot_points.append(robot)
        board_points.append(board.tolist())
        used.append({
            "name": str(entry.get("name", f"contact_{len(used) + 1}")),
            "robot_tool_xy_mm": [round(value, 3) for value in robot],
            "camera_px": [round(value, 3) for value in pixel],
            "board_xy_mm": [round(float(value), 3) for value in board],
        })
    rotation, translation, rms = fit_rigid_transform(np.asarray(robot_points), np.asarray(board_points))
    accepted = rms <= args.max_rms_error_mm
    result = {
        "schema_version": 1,
        "mode": "offline_contact_based_robot_base_alignment",
        "hardware_access": False,
        "contacts": used,
        "robot_xy_mm_to_board_xy_mm": {
            "rotation": rotation.round(9).tolist(),
            "translation_mm": translation.round(6).tolist(),
        },
        "fit_rms_error_mm": round(rms, 4),
        "maximum_accepted_rms_error_mm": args.max_rms_error_mm,
        "accepted": accepted,
        "scope": {
            "complete_when_accepted": ["robot tool centre on table plane to A4 board frame"],
            "not_complete": ["collision geometry", "grasp force", "object-height estimation"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise SystemExit("FAIL: contact residual exceeds threshold; do not use this transform for motion")


if __name__ == "__main__":
    main()
