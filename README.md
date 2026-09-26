# EmbodiedHarness × SO-101

An evidence-gated, model-agnostic runtime for **zero-demonstration tabletop
manipulation**. The project separates high-level task reasoning from physical
execution so that every real-world action is bounded, logged, and independently
verifiable.

> Status: simulation agent runtime and SO-101 bring-up utilities are working.
> Autonomous real-world pick-and-place is intentionally not claimed yet.

## Why this project

Most tabletop-robot demos jump directly from vision to actuation. This project
focuses on the pieces that make that handoff auditable:

- typed observation, skill, result, and evidence contracts;
- deterministic geometric skills that work before demonstrations exist;
- bounded recovery instead of open-ended retries;
- append-only episode traces and final-state verification;
- a data gate that keeps failed or recovered traces out of self-improvement
  training data;
- a hardware Safety Gate that rejects unsafe motor targets before they reach an
  SO-101 follower arm.

## Architecture

```text
task request
  -> deterministic skill plan
  -> simulator or future robot adapter
  -> append-only JSONL trace
  -> final-state evidence verifier
  -> clean-data admission gate
```

The task-agent and data-admission invariants are documented in
[`docs/agent_runtime.md`](docs/agent_runtime.md).

## What is implemented

| Layer | Implementation | Verification |
| --- | --- | --- |
| Typed contracts | `harness/core.py` | unit tests |
| Deterministic tabletop simulation | `harness/sim.py` | replayable pick/place episode |
| Geometric skills | `harness/skills.py` | precondition and final-state checks |
| Agent + recovery | `harness/agent.py` | bounded retry and abort tests |
| Data-quality gate | `harness/data_gate.py` | clean vs. recovered-trace tests |
| Motor Safety Gate | `harness/safety.py` | hard/soft limit and timeout tests |
| SO-101 bring-up tools | `scripts/safe_so101_step_test.py`, `scripts/so101_hold_watchdog_test.py` | used only after physical calibration |
| Visual workspace tools | `scripts/calibrate_tabletop_from_a4.py`, `scripts/detect_visual_proxies.py` | local, hardware-adjacent tooling |

## Quick start

Requires Python 3.11 or newer. The simulation path uses only the standard
library.

```bash
git clone <your-repository-url>
cd harness
python3 -m scripts.run_sim_pick_place
python3 -m unittest discover -s tests -v
```

The simulation command prints `accepted simulation episode` only after final
state verification and the clean-data gate both pass.

## Safety boundary

Hardware is not an extension of the simulator. SO-101 utilities require a
calibrated follower arm and apply the configuration in
[`configs/so101_safety.json`](configs/so101_safety.json). The gate rejects
unknown joints, stale or out-of-range states, targets beyond hard/soft limits,
steps above the configured maximum, and actions exceeding the time budget.

The scripts are deliberately conservative: they latch measured positions before
enabling torque, record readback, and fail closed by disabling torque unless a
successful operation explicitly requests a temporary hold. Do not run them on
an uncalibrated arm.

## What remains

The next hardware milestone is visual end-effector alignment followed by
closed-loop, approach-only motion above a target. Collision geometry, 3D tool
pose, force-sensitive grasping, and autonomous grasp execution remain gated on
that validation; the repository does not present them as finished.

## Repository hygiene

The repository intentionally excludes calibration images, serial recordings,
generated episode data, virtual environments, and local PDFs. This keeps the
public project focused on reproducible source, tests, and design decisions.
