"""Save a metric tabletop-plane calibration from the printed A4 reference board.

This deliberately calibrates only the *table plane*.  It does not invent a
camera-to-robot transform: that will be a separate, physical measurement once
we have a safe way to identify the SO-101 tool point.

The four image points are the centres of the thick L corners, in camera-pixel
coordinates.  Give them in the visual order TL, TR, BR, BL as printed on page
one of the reference kit (not necessarily the order they appear in the camera
when the sheet has been rotated).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np


BOARD_MM = np.float32(
    [
        [0.0, 0.0],
        [277.0, 0.0],
        [277.0, 172.0],
        [0.0, 172.0],
    ]
)


def parse_corner(value: str) -> tuple[float, float]:
    try:
        x, y = (float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("a corner must be X,Y, for example 630,315") from exc
    return x, y


def px_per_mm(points_px: np.ndarray) -> dict[str, float]:
    """Report the local pixel density along the printed board edges."""
    edges = {
        "top": (points_px[0], points_px[1], 277.0),
        "right": (points_px[1], points_px[2], 172.0),
        "bottom": (points_px[2], points_px[3], 277.0),
        "left": (points_px[3], points_px[0], 172.0),
    }
    return {
        name: round(float(np.linalg.norm(end - start) / length_mm), 5)
        for name, (start, end, length_mm) in edges.items()
    }


def draw_review(image: np.ndarray, points_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Draw source corners and a 50 mm metric grid for a human visual check."""
    review = image.copy()
    cv2.polylines(review, [points_px.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    for name, point in zip(("TL (0,0)", "TR (277,0)", "BR (277,172)", "BL (0,172)"), points_px):
        x, y = (int(round(v)) for v in point)
        cv2.circle(review, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(review, name, (x + 10, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    inverse = np.linalg.inv(homography)
    for x_mm in range(0, 278, 50):
        line = cv2.perspectiveTransform(np.float32([[[x_mm, 0], [x_mm, 172]]]), inverse)[0]
        cv2.line(review, tuple(np.int32(line[0])), tuple(np.int32(line[1])), (255, 150, 0), 1, cv2.LINE_AA)
    for y_mm in range(0, 173, 50):
        line = cv2.perspectiveTransform(np.float32([[[0, y_mm], [277, y_mm]]]), inverse)[0]
        cv2.line(review, tuple(np.int32(line[0])), tuple(np.int32(line[1])), (255, 150, 0), 1, cv2.LINE_AA)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Saved camera frame containing the complete board")
    parser.add_argument("--output", type=Path, required=True, help="JSON calibration output")
    parser.add_argument("--review", type=Path, required=True, help="Annotated PNG output")
    parser.add_argument(
        "--orientation",
        choices=("upright", "rotated_90_cw", "rotated_180", "rotated_90_ccw"),
        required=True,
    )
    parser.add_argument(
        "--corner",
        type=parse_corner,
        action="append",
        required=True,
        metavar="X,Y",
        help="Printed TL, TR, BR, BL image coordinate, in that exact order (repeat four times)",
    )
    args = parser.parse_args()
    if len(args.corner) != 4:
        parser.error("pass exactly four --corner values in printed TL, TR, BR, BL order")

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    points_px = np.float32(args.corner)
    homography = cv2.getPerspectiveTransform(points_px, BOARD_MM)
    projected = cv2.perspectiveTransform(points_px.reshape(1, -1, 2), homography)[0]
    reprojection_error_mm = float(np.max(np.linalg.norm(projected - BOARD_MM, axis=1)))
    density = px_per_mm(points_px)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "image": str(args.image),
        "image_size_px": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "camera_index": 1,
        "reference_board": {
            "paper": "A4 landscape printed at 100%",
            "printed_board_size_mm": {"width": 277.0, "height": 172.0},
            "orientation_in_camera": args.orientation,
            "image_corners_px_in_print_order": {
                "TL": points_px[0].tolist(), "TR": points_px[1].tolist(),
                "BR": points_px[2].tolist(), "BL": points_px[3].tolist(),
            },
        },
        "homography_pixel_to_board_mm": homography.tolist(),
        "quality": {
            "corner_selection": "operator-reviewed centres of printed L corners",
            "max_corner_reprojection_error_mm": reprojection_error_mm,
            "edge_pixel_density_px_per_mm": density,
            "print_scale_check": "100 mm confirmed by user",
        },
        "scope": {
            "complete": ["camera pixel to tabletop board millimetres"],
            "not_yet_calibrated": ["tabletop board to robot base", "camera height / 3D object height"],
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not cv2.imwrite(str(args.review), draw_review(image, points_px, homography)):
        raise RuntimeError(f"could not write review image: {args.review}")
    print(json.dumps({"output": str(args.output), "review": str(args.review), "quality": payload["quality"]}, indent=2))


if __name__ == "__main__":
    main()
