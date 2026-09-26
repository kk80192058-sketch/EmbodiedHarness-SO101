# Agent Runtime: Hardware-Independent Slice

This document describes the portion of EmbodiedHarness that can be developed
and verified while no robot is connected.

## Execution flow

```text
task request
  -> deterministic skill backend
  -> append-only event trace
  -> final-state evidence verifier
  -> clean-data admission gate
```

`TaskAgent` executes the zero-demonstration `pick -> place` plan against the
adapter contract. Every request, observation, result, and recovery decision is
written to JSONL before the next decision is made.

## Failure policy

| Failure | Agent response | Data-gate outcome |
| --- | --- | --- |
| perception miss | one bounded retry | excluded if any attempt fails |
| grasp fail | open gripper, then one bounded retry | excluded if any attempt fails |
| place fail | one bounded retry | excluded if any attempt fails |
| precondition/safety/unknown | abort | excluded |

The purpose of a recovery trace is diagnosis, not automatic self-training. A
recovered episode may be useful to inspect, but it is not a clean policy
example.

## Acceptance invariant

An episode is eligible for self-improvement data only when all conditions hold:

1. Its event sequences are contiguous and append-only.
2. It has exactly one completion event.
3. The final verifier proves that the requested object is released inside the
   requested target and the gripper is open.
4. No `skill_result` event reports failure.

The deterministic simulator tests all of these conditions, including successful
execution, bounded recovery, and an unsafe/precondition abort.

## Explicitly deferred to SO-101 bring-up

The following are intentionally **not** claimed by this slice: camera-to-base
extrinsics, 3D tool pose, collision geometry, force-sensitive grasping,
hardware action execution, or learning from a real robot trace. They remain
behind the SO-101 safety gate and require the physical calibration steps.
