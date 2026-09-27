"""Capture both SO-101 cameras and telemetry on one persistent connection."""
import argparse
import json
from pathlib import Path
import time

from harness.so101_session import SO101Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=int, default=3)
    parser.add_argument('--interval', type=float, default=0.25)
    parser.add_argument('--step', nargs=2, metavar=('JOINT', 'DELTA'))
    args = parser.parse_args()
    if not 1 <= args.observations <= 1000 or not 0 <= args.interval <= 60:
        parser.error('Invalid capture duration')
    output = Path('artifacts/bringup/sessions') / str(time.time_ns())
    print(f'evidence_dir={output.resolve()}', flush=True)
    with SO101Session(output) as session:
        for _ in range(args.observations):
            record, _ = session.observe()
            print(json.dumps(record), flush=True)
            time.sleep(args.interval)
        if args.step:
            result = session.step(args.step[0], int(args.step[1]))
            print(json.dumps(result), flush=True)
            if not result['success']:
                raise SystemExit(2)


if __name__ == '__main__':
    main()
