"""Detect one red tabletop block inside the calibrated A4 workspace.

This is perception only: it reads an image and writes an observation record.
It deliberately has no motor, serial, or robot-control dependency.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np


def board_polygon_px(h_pixel_to_board: np.ndarray) -> np.ndarray:
    board = np.float32([[[0, 0], [277, 0], [277, 172], [0, 172]]])
    return cv2.perspectiveTransform(board, np.linalg.inv(h_pixel_to_board))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    homography = np.asarray(calibration["homography_pixel_to_board_mm"], dtype=np.float32)
    polygon = board_polygon_px(homography).astype(np.int32)

    workspace_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(workspace_mask, polygon, 255)
    # Exclude the 15 mm border; reference marks must never be mistaken for a target.
    inner_mm = np.float32([[[15, 15], [262, 15], [262, 157], [15, 157]]])
    inner_px = cv2.perspectiveTransform(inner_mm, np.linalg.inv(homography))[0].astype(np.int32)
    inner_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(inner_mask, inner_px, 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 60), (12, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))
    red &= inner_mask
    components, _, stats, centroids = cv2.connectedComponentsWithStats(red)
    candidates = []
    for index in range(1, components):
        x, y, width, height, area = (int(v) for v in stats[index])
        aspect = width / max(height, 1)
        if area >= 150 and 0.45 <= aspect <= 2.2:
            candidates.append((area, x, y, width, height, centroids[index]))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one red target inside workspace; found {len(candidates)}")

    area, x, y, width, height, center = candidates[0]
    center_px = np.float32([[[center[0], center[1]]]])
    center_mm = cv2.perspectiveTransform(center_px, homography)[0, 0]
    observation = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "perception_only",
        "source_image": str(args.image),
        "calibration": str(args.calibration),
        "target": {
            "class": "red_block",
            "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
            "center_board_mm": [round(float(center_mm[0]), 3), round(float(center_mm[1]), 3)],
            "bbox_px": [x, y, width, height],
            "red_pixel_area": area,
        },
        "workspace_gate": {
            "inside_inner_15mm_margin": True,
            "board_mm_bounds": [0, 0, 277, 172],
            "note": "Observation is not a robot motion target until robot-base alignment is complete.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    args.review.parent.mkdir(parents=True, exist_ok=True)
    review = image.copy()
    cv2.polylines(review, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(review, (x, y), (x + width, y + height), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(review, (round(center[0]), round(center[1])), (0, 0, 255), cv2.MARKER_CROSS, 25, 2, cv2.LINE_AA)
    label = f"red block: ({center_mm[0]:.1f}, {center_mm[1]:.1f}) mm"
    cv2.putText(review, label, (x - 20, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(args.review), review):
        raise RuntimeError(f"could not write review: {args.review}")
    print(json.dumps(observation, indent=2))


if __name__ == "__main__":
    main()
