"""Verify SO-101 hold and fail-closed watchdog behavior.

The script latches all joints to their *measured* positions before torque is
enabled. It then watches present positions for a bounded interval. By default
it disables torque before closing; ``--keep-holding`` is the explicit opt-in
used when an operator needs the arm to remain supported after the check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from harness.safety import SafetyConfig, SafetyGate


PORT = "/dev/cu.usbmodem5A7C1240491"


def make_bus() -> FeetechMotorsBus:
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    return FeetechMotorsBus(PORT, motors)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 bounded hold/watchdog test")
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--poll-s", type=float, default=0.05)
    parser.add_argument(
        "--keep-holding",
        action="store_true",
        help="Leave torque enabled only after a complete, passing watchdog interval",
    )
    args = parser.parse_args()
    if args.hold_s <= 0 or args.poll_s <= 0:
        raise ValueError("hold and poll intervals must be positive")

    gate = SafetyGate(SafetyConfig.from_json(Path("configs/so101_safety.json")))
    bus = make_bus()
    samples = 0
    keep_holding = False
    try:
        bus.connect()
        present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        # A target equal to present position must still satisfy all soft limits.
        hold_targets = gate.validate_raw_joint_goals(present, present, timeout_s=3.0)

        bus.disable_torque()
        for joint, position in hold_targets.items():
            bus.write("Goal_Position", joint, position, normalize=False)
        bus.enable_torque()

        import time

        deadline = time.monotonic() + args.hold_s
        while time.monotonic() < deadline:
            observed = bus.sync_read("Present_Position", normalize=False, num_retry=2)
            gate.validate_raw_joint_goals(observed, observed, timeout_s=3.0)
            samples += 1
            time.sleep(args.poll_s)
        keep_holding = args.keep_holding
        terminal_state = "torque_enabled" if keep_holding else "torque_disabled"
        print(f"PASS hold_s={args.hold_s} samples={samples} watchdog=armed terminal_state={terminal_state}")
    finally:
        if bus.is_connected:
            # Fail closed on every failure; persistence is an explicit successful opt-in.
            if not keep_holding:
                bus.disable_torque(num_retry=2)
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
