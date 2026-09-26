"""Release all SO-101 follower joints without sending a position command."""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main() -> None:
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus("/dev/cu.usbmodem5A7C1240491", motors)
    try:
        bus.connect()
        bus.disable_torque(num_retry=2)
        print("PASS torque disabled for all six joints; no goal positions written")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
