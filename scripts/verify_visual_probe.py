"""Fail closed unless a visual micro-motion probe is internally consistent.

Reads three perception-only proxy observations: start, bounded motion, and
return.  It never opens a camera or a motor serial port.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def center(payload: dict, key: str) -> np.ndarray:
    return np.asarray(payload[key]["center_px"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--moved", type=Path, required=True)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, moved, returned = (read(path) for path in (args.start, args.moved, args.returned))
    tool_return_error = float(np.linalg.norm(center(start, "gripper_tool_proxy") - center(returned, "gripper_tool_proxy")))
    tool_motion = float(np.linalg.norm(center(start, "gripper_tool_proxy") - center(moved, "gripper_tool_proxy")))
    block_drift = max(
        float(np.linalg.norm(center(start, "red_block") - center(moved, "red_block"))),
        float(np.linalg.norm(center(start, "red_block") - center(returned, "red_block"))),
    )
    start_area = float(start["gripper_tool_proxy"]["area_px"])
    return_area = float(returned["gripper_tool_proxy"]["area_px"])
    area_ratio = return_area / start_area if start_area else 0.0
    gates = {
        "returned_near_start": tool_return_error <= 8.0,
        "static_target": block_drift <= 3.0,
        "tool_shape_consistent": 0.65 <= area_ratio <= 1.35,
        "motion_was_observed": tool_motion >= 0.5,
    }
    result = {
        "mode": "perception_only_visual_probe_gate",
        "tool_motion_px": round(tool_motion, 3),
        "tool_return_error_px": round(tool_return_error, 3),
        "target_drift_px": round(block_drift, 3),
        "tool_return_area_ratio": round(area_ratio, 3),
        "gates": gates,
        "accepted": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit("visual probe rejected; do not plan another physical motion from this evidence")


if __name__ == "__main__":
    main()
