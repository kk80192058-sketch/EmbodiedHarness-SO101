"""Derive corrected contact samples without overwriting original evidence.

Use only when an operator has explicitly clarified a coordinate-frame mistake.
The derived JSON records the original file digest, old board coordinate, and
new board coordinate so later audit can distinguish a metadata correction from
a fresh physical contact measurement.  This script never accesses hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_sample(source_path: Path, name: str, board_xy_mm: list[float], reason: str) -> dict:
    source_path = source_path.resolve()
    sample = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(name, str) or not name:
        raise ValueError("derived name is required")
    if not isinstance(board_xy_mm, list) or len(board_xy_mm) != 2 or not all(isinstance(v, (int, float)) for v in board_xy_mm):
        raise ValueError("board_xy_mm must contain two numeric values")
    original = sample.get("board_xy_mm")
    sample["name"] = name
    sample["board_xy_mm"] = [float(value) for value in board_xy_mm]
    sample["coordinate_correction"] = {
        "schema_version": 1,
        "source_sample": str(source_path),
        "source_sample_sha256": sha256(source_path),
        "original_board_xy_mm": original,
        "corrected_board_xy_mm": sample["board_xy_mm"],
        "reason": reason,
        "hardware_access": False,
    }
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, required=True,
                        help='JSON: {"reason": str, "samples": [{"source": str, "name": str, "board_xy_mm": [x,y]}]}')
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.corrections.read_text(encoding="utf-8"))
    reason, entries = payload.get("reason"), payload.get("samples")
    if not isinstance(reason, str) or not reason or not isinstance(entries, list) or not entries:
        parser.error("corrections needs a non-empty reason and samples list")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths, names, points = [], set(), set()
    for entry in entries:
        try:
            sample = derive_sample(Path(entry["source"]), entry["name"], entry["board_xy_mm"], reason)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid correction: {error}")
        if sample["name"] in names or tuple(sample["board_xy_mm"]) in points:
            parser.error("corrected names and board points must be unique")
        names.add(sample["name"]); points.add(tuple(sample["board_xy_mm"]))
        path = args.output_dir / f"{sample['name']}.json"
        path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        paths.append(str(path))
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": paths}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "sample_count": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
