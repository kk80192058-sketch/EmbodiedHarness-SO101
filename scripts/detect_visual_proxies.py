"""Read red gripper-star and red-block proxies from one saved camera image.

This is perception-only. It never opens a motor serial port or sends actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 80, 35), (15, 255, 255)) | cv2.inRange(hsv, (165, 80, 35), (180, 255, 255))


def dark_mask(image: np.ndarray) -> np.ndarray:
    """Pick dark mechanical features against the light tabletop."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 0), (180, 255, 75))


def detect(mask: np.ndarray, roi: list[int], minimum_area: int, expected: tuple[float, float] | None = None) -> dict[str, object]:
    left, top, right, bottom = roi
    cropped = mask[top:bottom, left:right]
    count, _, stats, centers = cv2.connectedComponentsWithStats(cropped)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < minimum_area:
            continue
        center = (float(centers[index][0] + left), float(centers[index][1] + top))
        distance = np.linalg.norm(np.asarray(center) - np.asarray(expected)) if expected else -area
        candidates.append((distance, area, center, [x + left, y + top, width, height]))
    if not candidates:
        raise RuntimeError(f"no red component meets area threshold {minimum_area} in ROI {roi}")
    _, area, center, bbox = min(candidates, key=lambda item: item[0])
    return {"center_px": [round(center[0], 3), round(center[1], 3)], "bbox_px": bbox, "area_px": area}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vision_proxies.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read {args.image}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mask = red_mask(image)
    star_config, tool_config, block_config = (
        config["gripper_star"],
        config["gripper_tool_proxy"],
        config["red_block"],
    )
    result = {
        "image": str(args.image),
        "gripper_star": detect(mask, star_config["roi_px"], star_config["minimum_area_px"], tuple(star_config["expected_center_px"])),
        "gripper_tool_proxy": detect(
            dark_mask(image),
            tool_config["roi_px"],
            tool_config["minimum_area_px"],
            tuple(tool_config["expected_center_px"]),
        ),
        "red_block": detect(mask, block_config["roi_px"], block_config["minimum_area_px"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
