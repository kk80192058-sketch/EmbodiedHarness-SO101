"""Deterministic geometric skill semantics for the W1 simulation slice."""

from __future__ import annotations

from harness.core import Evidence, FailureType, SkillRequest, SkillResult
from harness.sim import TabletopSimAdapter


class GeometricSkillBackend:
    name = "geometric"

    def run(self, request: SkillRequest, robot: TabletopSimAdapter) -> SkillResult:
        before = robot.observe()
        objects = {obj.object_id: obj for obj in before.objects}
        if request.object_id not in objects or not objects[request.object_id].visible:
            return SkillResult(request, False, (Evidence("object_visible", False, {}),), FailureType.PERCEPTION_MISS)

        if request.name == "pick":
            ok = robot.pick(request.object_id)
            return SkillResult(
                request, ok,
                (Evidence("object_grasped", ok, {"object_id": request.object_id}),),
                None if ok else FailureType.GRASP_FAIL,
            )

        if request.name == "place":
            if request.target_id is None or request.target_id not in objects:
                return SkillResult(request, False, (Evidence("target_visible", False, {}),), FailureType.PRECONDITION_FAILED)
            if before.held_object_id != request.object_id:
                return SkillResult(request, False, (Evidence("object_held", False, {}),), FailureType.PRECONDITION_FAILED)
            ok = robot.place(request.target_id)
            after = robot.observe()
            placed = next(obj for obj in after.objects if obj.object_id == request.object_id)
            evidence = Evidence("object_inside_target", ok and placed.container_id == request.target_id, {"target_id": request.target_id})
            return SkillResult(request, evidence.passed, (evidence,), None if evidence.passed else FailureType.PLACE_FAIL)

        return SkillResult(request, False, (Evidence("known_skill", False, {"name": request.name}),), FailureType.PRECONDITION_FAILED)

