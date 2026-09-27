# Security policy

## Scope

This project controls or prepares commands for physical robotics hardware.
Potentially unsafe command generation, bypasses of safety limits, unexpected
calibration/torque writes, unsafe deserialization, credential exposure, and
dependency supply-chain concerns are in scope.

## Reporting a vulnerability

Please do **not** publish a proof of concept that can move hardware or bypass
safety gates in a public issue. Use GitHub's private security-advisory reporting
for this repository when available, or contact the repository owner privately
through the GitHub profile associated with the repository.

Include affected revision, environment, reproduction steps, expected versus
actual behavior, and whether a connected arm can receive a write. We will
acknowledge a report as soon as practical, work on a fix, and coordinate public
disclosure after users can mitigate the issue.

## Operational safety

No software report substitutes for physical supervision. Keep the arm's
workspace clear and use the documented hardware gates. Do not rely on an
unverified issue reproduction to control a real robot.
