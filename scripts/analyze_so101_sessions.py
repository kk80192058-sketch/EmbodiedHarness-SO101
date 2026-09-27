"""Retain all physical step outcomes and estimate local visual response offline."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from harness.red_block_tracking import detect_red_block
from harness.session_analysis import pair_completed_steps


def main():
    output = Path('artifacts/experience') / str(time.time_ns())
    output.mkdir(parents=True)
    steps = []
    for folder in sorted(Path('artifacts/bringup/sessions').iterdir()):
        events = folder / 'events.jsonl'
        if not events.exists(): continue
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        for pair in pair_completed_steps(rows):
            result, before, after, intent = (pair[key] for key in ('result', 'before', 'after', 'intent'))
            detections = {}
            for label, obs in [('before', before), ('after', after)]:
                detections[label] = {role: detect_red_block(cv2.imread(meta['path']), role)
                                     for role, meta in obs['cameras'].items()}
            entry = {'session': str(folder.resolve()), 'result': result, 'detections': detections,
                     'grasp_outcome': 'unknown', 'eligible_for_success_imitation': False,
                     'data_route': 'system_identification_and_outcome_review'}
            deltas = {n: result['after']['Present_Position'][n] - result['before']['Present_Position'][n]
                      for n in result['before']['Present_Position']}
            entry['actual_joint_deltas'] = deltas
            entry['commanded_joint'] = intent['joint']
            for role in ('wrist', 'global'):
                a, b = detections['before'][role]['target'], detections['after'][role]['target']
                if a and b:
                    entry[role + '_pixel_delta'] = (np.array(b['center_px']) - a['center_px']).tolist()
            steps.append(entry)
    (output / 'steps.json').write_text(json.dumps(steps, indent=2) + '\n')
    # A local camera response estimate, not a trained VLA or a grasp policy.
    useful = [s for s in steps if s['commanded_joint'] == 'wrist_flex'
              and abs(s['actual_joint_deltas']['wrist_flex']) >= 5
              and 'wrist_pixel_delta' in s and 'global_pixel_delta' in s
              and np.linalg.norm(s['global_pixel_delta']) < 3
              and max(abs(v) for k,v in s['actual_joint_deltas'].items() if k != 'wrist_flex') <= 2]
    report = {'mode': 'offline_local_visual_response', 'samples': len(useful), 'all_steps_retained': len(steps),
              'hardware_execution': False, 'grasp_success_verified': False}
    if len(useful) >= 3:
        x = np.array([s['actual_joint_deltas']['wrist_flex'] for s in useful])
        y = np.array([s['wrist_pixel_delta'] for s in useful])
        slope = (x @ y) / (x @ x)
        errors = []
        for i in range(len(x)):
            train_x, train_y = np.delete(x,i), np.delete(y,i,axis=0)
            estimate = (train_x @ train_y) / (train_x @ train_x)
            errors.append(float(np.linalg.norm(estimate*x[i]-y[i])))
        report.update(wrist_pixels_per_actual_count=slope.tolist(), leave_one_out_errors_px=errors,
                      observed_wrist_range=[min(s['result']['after']['Present_Position']['wrist_flex'] for s in useful),
                                            max(s['result']['before']['Present_Position']['wrist_flex'] for s in useful)],
                      limits='Local orientation/image response only; does not estimate grasp height or contact.')
    (output / 'visual_response.json').write_text(json.dumps(report, indent=2) + '\n')
    print('evidence_dir=' + str(output.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
