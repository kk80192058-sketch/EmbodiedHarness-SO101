# Contributing

Thank you for improving EmbodiedHarness × SO-101. The repository separates
hardware-independent contracts from physical-arm tooling; contributions should
preserve that boundary.

## Local checks

Use Python 3.11 or newer and run the portable test suite before opening a pull
request:

```bash
python3 -m pip install -e ".[test]"
python3 -m unittest discover -s tests -v
git diff --check
```

The optional `.[so101-policy]` extra is for local SmolVLA shadow inference. Do
not make the normal test path depend on model weights, a serial port, cameras,
or a real arm.

## Hardware changes

Hardware-facing changes must make their safety boundary explicit. In
particular, do not add a code path that silently changes calibration, PID,
torque settings, current limits, or a motor goal while collecting evidence.
Keep before/after observations and the exact write intent in the session log.
Tests must cover every newly introduced reject/stop path without accessing
hardware.

Do not commit camera images, serial logs, model weights, calibration captures,
or generated episodes; these are intentionally ignored under `artifacts/`.
When a change relies on real evidence, describe the collection procedure and
the evidence limitations in documentation instead.

## Pull requests

Keep pull requests focused. Explain the contract being changed, list commands
run, and distinguish simulated coverage from physical validation. Never label
a physical pick/place result successful without independent final-state
evidence.
