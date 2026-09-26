"""Turn saved SO-101 camera micro-motion frames into local visual signatures.

This utility is deliberately offline: it reads already-captured PNGs and never
opens a serial port or camera.  The signatures are evidence for local planning,
not a replacement for a full robot-base kinematic calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np


def largest_change(start: np.ndarray, moved: np.ndarray) -> dict[str, object]:
    gray_start = cv2.cvtColor(start, cv2.COLOR_BGR2GRAY)
    gray_moved = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray_start, gray_moved)
    # Suppress sensor noise and small exposure changes; keep coherent motion only.
    _, binary = cv2.threshold(difference, 24, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)]
    if not candidates:
        return {"detected": False, "mean_abs_pixel_change": round(float(difference.mean()), 4)}
    area, index = max(candidates)
    x, y, width, height, _ = (int(v) for v in stats[index])
    centroid = centroids[index]
    return {
        "detected": True,
        "mean_abs_pixel_change": round(float(difference.mean()), 4),
        "largest_change_bbox_px": [x, y, width, height],
        "largest_change_centroid_px": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
        "largest_change_area_px": area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=Path("artifacts/bringup/camera_motion_checks"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--since", default="", help="Optional inclusive YYYYMMDDTHHMMSS capture timestamp")
    args = parser.parse_args()
    results: list[dict[str, object]] = []
    pattern = re.compile(r"(?P<timestamp>\d{8}T\d{6})_(?P<joint>.+)_(?P<label>start|moved|returned)\.png$")
    parsed = []
    for path in args.frames_dir.glob("*.png"):
        match = pattern.match(path.name)
        if match:
            parsed.append((match.group("timestamp"), match.group("joint"), match.group("label"), path))
    for timestamp, joint, _, moved in sorted(item for item in parsed if item[2] == "moved"):
        if args.since and timestamp < args.since:
            continue
        starts = [item for item in parsed if item[1] == joint and item[2] == "start" and item[0] <= timestamp]
        returns = [item for item in parsed if item[1] == joint and item[2] == "returned" and item[0] >= timestamp]
        if not starts or not returns:
            continue
        start = max(starts, key=lambda item: item[0])[3]
        returned = min(returns, key=lambda item: item[0])[3]
        start_image, moved_image, returned_image = (cv2.imread(str(path)) for path in (start, moved, returned))
        if any(image is None for image in (start_image, moved_image, returned_image)):
            continue
        results.append({
            "probe": f"{timestamp}_{joint}",
            "joint": joint,
            "start_to_moved": largest_change(start_image, moved_image),
            "moved_to_returned": largest_change(moved_image, returned_image),
            "frames": {"start": str(start), "moved": str(moved), "returned": str(returned)},
        })
    payload = {
        "schema_version": 1,
        "mode": "offline_visual_motion_signatures",
        "probe_count": len(results),
        "probes": results,
        "limits": [
            "Signatures describe image change, not full 3D tool pose.",
            "No motion is planned or executed by this script.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
