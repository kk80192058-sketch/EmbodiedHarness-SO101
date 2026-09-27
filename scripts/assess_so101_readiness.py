"""Create an evidence-only SO-101 readiness report from saved shadow evidence."""
import argparse
import json
from pathlib import Path
import time

from harness.so101_readiness import assess_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    record = json.loads(args.observation.read_text())
    prediction = json.loads(args.prediction.read_text())
    config = json.loads(Path('configs/so101_safety.json').read_text())
    calibration = json.loads(Path(config['calibration_file']).read_text())
    result = assess_readiness(record, prediction, calibration)
    output = args.output or Path('artifacts/policy_readiness') / f'{time.time_ns()}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**result, 'source_observation': str(args.observation.resolve()),
                                  'source_prediction': str(args.prediction.resolve())}, indent=2) + '\n')
    print('report=' + str(output.resolve()))
    print(json.dumps({'motion_authorized': result['motion_authorized'],
                      'observation_healthy': result['observation_healthy'],
                      'policy_proposal_compatible': result['policy_proposal_compatible']}))


if __name__ == '__main__':
    main()
