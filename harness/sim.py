"""A deterministic, minimal tabletop implementation of the RobotAdapter contract."""

from __future__ import annotations

from dataclasses import replace
import time

from harness.core import ExecutionReceipt, ObjectState, Pose2D, RobotObservation


class TabletopSimAdapter:
    """Simple stateful adapter; deliberately no physics engine dependency in W1."""

    def __init__(self, objects: list[ObjectState]) -> None:
        self._objects = {obj.object_id: obj for obj in objects}
        self._connected = False
        self._gripper_open = True
        self._held_object_id: str | None = None

    def connect(self) -> None:
        self._connected = True

    def observe(self) -> RobotObservation:
        return RobotObservation(
            timestamp=time.time(), connected=self._connected,
            gripper_open=self._gripper_open, held_object_id=self._held_object_id,
            objects=tuple(self._objects.values()),
        )

    def execute(self, action: str) -> ExecutionReceipt:
        if not self._connected:
            return ExecutionReceipt(False, time.time(), "adapter is disconnected")
        if action not in {"open_gripper", "close_gripper"}:
            return ExecutionReceipt(False, time.time(), f"unsupported action: {action}")
        self._gripper_open = action == "open_gripper"
        return ExecutionReceipt(True, time.time(), action)

    def pick(self, object_id: str) -> bool:
        obj = self._objects.get(object_id)
        if not self._connected or not self._gripper_open or obj is None or not obj.visible:
            return False
        self._gripper_open = False
        self._held_object_id = object_id
        self._objects[object_id] = replace(obj, grasped=True, container_id=None)
        return True

    def place(self, target_id: str) -> bool:
        if not self._connected or self._held_object_id is None or target_id not in self._objects:
            return False
        held_id = self._held_object_id
        target = self._objects[target_id]
        held = self._objects[held_id]
        self._objects[held_id] = replace(held, pose=target.pose, grasped=False, container_id=target_id)
        self._held_object_id = None
        self._gripper_open = True
        return True

    def hold(self) -> None:
        """Contract placeholder for later real hardware stop/hold behavior."""

    def safe_home(self) -> None:
        """Contract placeholder; no motion is modeled in this W1 simulator."""

