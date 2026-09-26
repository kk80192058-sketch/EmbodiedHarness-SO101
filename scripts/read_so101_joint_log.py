"""Read SO-101 joints without sending any motion command.

This is a bring-up diagnostic only.  It opens the Feetech bus, repeatedly reads
the Present_Position register, checks the saved calibration bounds, then closes
the port without changing torque state.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


DEFAULT_PORT = "/dev/cu.usbmodem5A7C1240491"
DEFAULT_ID = "so101_follower_kai_01"
DEFAULT_CALIBRATION_DIR = Path("artifacts/calibration") / DEFAULT_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SO-101 joint logger")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--port", default=DEFAULT_PORT)
    args = parser.parse_args()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=DEFAULT_ID,
            calibration_dir=DEFAULT_CALIBRATION_DIR,
        )
    )
    output_dir = Path("artifacts/bringup")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"joint_read_{time.strftime('%Y%m%dT%H%M%S')}.csv"
    fields = ["timestamp", *robot.bus.motors, "in_calibrated_range"]
    samples = 0
    violations = 0
    deadline = time.monotonic() + args.duration_s

    try:
        robot.bus.connect()
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while time.monotonic() < deadline:
                positions = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                in_range = all(
                    robot.calibration[name].range_min <= value <= robot.calibration[name].range_max
                    for name, value in positions.items()
                )
                violations += int(not in_range)
                writer.writerow({"timestamp": time.time(), **positions, "in_calibrated_range": in_range})
                stream.flush()
                samples += 1
                time.sleep(args.interval_s)
    finally:
        if robot.bus.is_connected:
            # Reads never enable torque; preserve that state on disconnect.
            robot.bus.disconnect(disable_torque=False)

    print(f"samples={samples} out_of_range_samples={violations} log={output}")
    if violations:
        raise SystemExit("A joint left its calibrated range during read-only logging.")


if __name__ == "__main__":
    main()
