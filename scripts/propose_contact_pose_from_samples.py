"""Propose an empirical SO-101 contact pose from verified board contacts.

This intentionally does *not* use a URDF or command hardware.  It performs
piecewise affine interpolation in observed joint space over a triangle of
human-verified physical jaw-centre contacts.  The result is a table-contact
candidate only: callers must add a separately verified clearance strategy
before it may become a motion target.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from harness.contact_evidence import audit_contact_manifest


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def parse_pair(value: str) -> np.ndarray:
    try:
        result = np.asarray([float(part) for part in value.split(",")])
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be X,Y") from error
    if result.shape != (2,):
        raise argparse.ArgumentTypeError("point must contain exactly X,Y")
    return result


def barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    matrix = np.column_stack((triangle[0] - triangle[2], triangle[1] - triangle[2]))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-8:
        return None
    first_two = np.linalg.solve(matrix, point - triangle[2])
    return np.array([first_two[0], first_two[1], 1 - first_two.sum()])


def select_triangle(points: np.ndarray, target: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray]:
    candidates = []
    for indices in itertools.combinations(range(len(points)), 3):
        weights = barycentric(target, points[list(indices)])
        if weights is None or np.min(weights) < -1e-7:
            continue
        # Prefer a well-conditioned, local triangle: maximise its smallest
        # barycentric weight, then minimise area only as a tiebreaker.
        first_edge = points[indices[1]] - points[indices[0]]
        second_edge = points[indices[2]] - points[indices[0]]
        area = abs(float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])) / 2
        candidates.append((float(np.min(weights)), -area, indices, weights))
    if not candidates:
        raise ValueError("target lies outside every non-degenerate contact triangle")
    _, _, indices, weights = max(candidates, key=lambda item: (item[0], item[1]))
    return indices, weights


def load_audited_samples(manifest_path: Path) -> tuple[list[dict], dict]:
    """Load interpolation inputs only after the contact-evidence audit passes.

    This establishes saved-input integrity, not a clearance or motion claim.
    """
    audit = audit_contact_manifest(manifest_path, min_samples=3)
    if not audit["accepted"]:
        raise ValueError("contact evidence audit failed: " + "; ".join(audit["errors"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in manifest["samples"]], audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--target-board-mm", type=parse_pair, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        samples, audit = load_audited_samples(args.samples)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    points = np.asarray([sample["board_xy_mm"] for sample in samples], dtype=float)
    raw = np.asarray([[sample["raw_encoder_counts"][joint] for joint in JOINTS] for sample in samples], dtype=float)
    indices, weights = select_triangle(points, args.target_board_mm)
    candidate = weights @ raw[list(indices)]
    result = {
        "schema_version": 1,
        "mode": "offline_empirical_contact_pose_proposal",
        "hardware_access": False,
        "contact_evidence_sha256": audit["source_sha256"],
        "contact_evidence_audit_mode": audit["mode"],
        "target_board_mm": args.target_board_mm.round(4).tolist(),
        "triangle_samples": [samples[index]["name"] for index in indices],
        "barycentric_weights": weights.round(6).tolist(),
        "proposed_raw_contact_pose": {joint: int(round(value)) for joint, value in zip(JOINTS, candidate)},
        "limits": [
            "This is a table-contact pose candidate derived from physical contact samples.",
            "It must not be commanded as a hover or grasp pose without a clearance verification.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
