"""Verify a rigid gripper-side visual proxy across saved probe frames.

This is deliberately offline: it reads PNGs captured by a bounded motion
probe, and never opens a serial port, camera, or motor bus.  It tracks a
template containing the red star *and* surrounding black wrist housing so
that the result is not dependent on the star's small red pixel area alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_box(value: str) -> tuple[int, int, int, int]:
    """Parse an x1,y1,x2,y2 pixel box, rejecting degenerate rectangles."""
    try:
        box = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2") from error
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise argparse.ArgumentTypeError("box must have x2>x1 and y2>y1")
    return box


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def ensure_in_bounds(box: tuple[int, int, int, int], image: np.ndarray, label: str) -> None:
    x1, y1, x2, y2 = box
    height, width = image.shape
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise RuntimeError(f"{label} box {box} is outside {width}x{height} image")


def track(
    image: np.ndarray, template: np.ndarray, search_box: tuple[int, int, int, int]
) -> tuple[float, list[float]]:
    ensure_in_bounds(search_box, image, "search")
    x1, y1, x2, y2 = search_box
    search = image[y1:y2, x1:x2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        raise RuntimeError("search box must be at least as large as template box")
    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
    center = [
        round(x1 + location[0] + template.shape[1] / 2, 3),
        round(y1 + location[1] + template.shape[0] / 2, 3),
    ]
    return float(score), center


def delta(first: list[float], second: list[float]) -> list[float]:
    return [round(second[index] - first[index], 3) for index in range(2)]


def distance(vector: list[float]) -> float:
    return round(float(np.hypot(vector[0], vector[1])), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--moved", type=Path, required=True)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--template-box", type=parse_box, required=True)
    parser.add_argument("--search-box", type=parse_box, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.95)
    parser.add_argument("--minimum-motion-px", type=float, default=0.5)
    parser.add_argument("--maximum-return-error-px", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_score <= 1.0:
        parser.error("--minimum-score must be between 0 and 1")
    if args.minimum_motion_px < 0 or args.maximum_return_error_px < 0:
        parser.error("pixel thresholds must be non-negative")

    images = [("start", args.start), ("moved", args.moved), ("returned", args.returned)]
    start_image = load_gray(args.start)
    ensure_in_bounds(args.template_box, start_image, "template")
    x1, y1, x2, y2 = args.template_box
    template = start_image[y1:y2, x1:x2]
    tracked = []
    for label, path in images:
        score, center = track(load_gray(path), template, args.search_box)
        tracked.append({"label": label, "image": str(path), "score": round(score, 5), "center_px": center})

    start_center, moved_center, returned_center = (item["center_px"] for item in tracked)
    start_to_moved = delta(start_center, moved_center)
    start_to_returned = delta(start_center, returned_center)
    motion_px = distance(start_to_moved)
    return_error_px = distance(start_to_returned)
    scores = [item["score"] for item in tracked]
    passed = (
        min(scores) >= args.minimum_score
        and motion_px >= args.minimum_motion_px
        and return_error_px <= args.maximum_return_error_px
    )
    result = {
        "schema_version": 1,
        "mode": "offline_template_proxy_verification",
        "hardware_access": False,
        "template_box_px": list(args.template_box),
        "search_box_px": list(args.search_box),
        "frames": tracked,
        "start_to_moved_delta_px": start_to_moved,
        "start_to_moved_distance_px": motion_px,
        "start_to_returned_delta_px": start_to_returned,
        "start_to_returned_distance_px": return_error_px,
        "thresholds": {
            "minimum_score": args.minimum_score,
            "minimum_motion_px": args.minimum_motion_px,
            "maximum_return_error_px": args.maximum_return_error_px,
        },
        "passed": passed,
        "limits": [
            "A passed result establishes repeatable 2D tracking of the proxy only.",
            "It does not establish the physical jaw centre or a camera-to-robot transform.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("FAIL: proxy tracking did not satisfy its declared thresholds")


if __name__ == "__main__":
    main()
