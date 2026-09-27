## Summary

Describe the contract, documentation, or behavior changed.

## Validation

- [ ] `python3 -m unittest discover -s tests -v`
- [ ] `git diff --check`
- [ ] New reject/stop paths are covered without hardware access.

## Hardware and evidence boundary

- [ ] This change does not silently change calibration, PID, torque/current limits, or motor targets.
- [ ] I distinguished simulation/fixture coverage from physical validation.
- [ ] I did not add ignored captures, serial logs, model weights, or credentials.
- [ ] If physical evidence is relevant, the documentation states what it proves and what it does not prove.
