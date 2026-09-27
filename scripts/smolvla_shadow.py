"""Run a pinned SmolVLA checkpoint on saved dual-camera evidence; no hardware."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vlm', type=Path, required=True)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--body-units', choices=['degrees', 'range_m100_100'], required=True)
    parser.add_argument('--task', default='Pick up the red cube.')
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()
    import cv2
    from harness.smolvla_policy import SmolVLAShadowPolicy, assess_proposal
    from harness.safety import SafetyConfig
    output = Path('artifacts/policy_shadow') / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    print('evidence_dir=' + str(output.resolve()), flush=True)
    record = json.loads(args.observation.read_text())
    config = json.loads(Path('configs/so101_safety.json').read_text())
    calibration = json.loads(Path(config['calibration_file']).read_text())
    images = {role: cv2.imread(meta['path']) for role, meta in record['cameras'].items()}
    if any(image is None for image in images.values()):
        raise ValueError('Missing source image')
    try:
        policy = SmolVLAShadowPolicy(args.model, args.vlm, device=args.device, body_units=args.body_units)
        for seed in range(args.runs):
            result = policy.predict(record, images, calibration, args.task, seed=seed)
            result['execution_assessment'] = assess_proposal(
                result, record, calibration, SafetyConfig.from_json(Path('configs/so101_safety.json')))
            result['source_observation'] = str(args.observation.resolve())
            (output / f'prediction_{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ('inference_s', 'state', 'normalized_state')}), flush=True)
            print('first_proposal=' + json.dumps(result['raw_proposals'][0]), flush=True)
    except BaseException as error:
        (output / 'error.json').write_text(json.dumps({'error': repr(error), 'hardware_writes': 0}, indent=2))
        raise


if __name__ == '__main__':
    main()
