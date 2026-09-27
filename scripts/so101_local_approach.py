"""Bounded wrist-only visual approach using the measured local image response.

Re-observes and supervises every segment on one connection. This is a scene
specific orientation skill, not a grasp-success detector or an IK solution.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from harness.so101_session import SO101Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--target-y', type=float, required=True)
    parser.add_argument('--max-steps', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 4 or not 800 <= args.target_y <= 900:
        parser.error('Only the inspected near-gripper image region is supported')
    response = json.loads(args.response.read_text())
    slope = response['wrist_pixels_per_actual_count'][1]
    if response['samples'] < 3 or not -2.5 < slope < -1.0 or max(response['leave_one_out_errors_px']) > 8:
        parser.error('Local response estimate is insufficient')
    output = Path('artifacts/bringup/sessions') / str(time.time_ns())
    print('evidence_dir=' + str(output.resolve()), flush=True)
    with SO101Session(output) as session:
        initial, _ = session.observe('approach_initial')
        original_q = initial['registers']['Present_Position']['wrist_flex']
        baseline_target = initial['red_block']['global']['target']
        if baseline_target is None:
            raise RuntimeError('Global target is ambiguous')
        baseline = np.array(baseline_target['center_px'])
        summary = {'grasp_status': 'unknown', 'steps': [], 'outcome': 'step_budget'}
        for _ in range(args.max_steps):
            observation, _ = session.observe('approach_feedback')
            wrist = observation['red_block']['wrist']['target']
            global_target = observation['red_block']['global']['target']
            if wrist is None or global_target is None:
                summary['outcome'] = 'target_visibility_stop'; break
            if np.linalg.norm(np.array(global_target['center_px']) - baseline) > 3:
                summary['outcome'] = 'object_moved_requires_review'; break
            if not 940 < wrist['center_px'][0] < 1040:
                summary['outcome'] = 'lateral_alignment_stop'; break
            error = args.target_y - wrist['center_px'][1]
            if abs(error) <= 10:
                summary['outcome'] = 'image_target_reached'; break
            # Only continue the empirically tested negative wrist direction.
            if error < 0:
                summary['outcome'] = 'overshoot_stop'; break
            desired = error / slope
            delta = -min(40, max(18, round(abs(desired) + 14)))
            current = observation['registers']['Present_Position']['wrist_flex']
            if current + delta < original_q - 100:
                summary['outcome'] = 'episode_travel_stop'; break
            session.event('visual_proposal', {'pixel_error_y': error, 'delta': delta,
                                             'response_source': str(args.response.resolve())})
            result = session.step('wrist_flex', delta)
            summary['steps'].append(result)
            if result['outcome'] not in ('target_reached', 'stalled') or not result.get('recovery_state_within_limits', True):
                summary['outcome'] = 'supervisor_stop'; break
            if result['actual_delta'] >= -5:
                summary['outcome'] = 'no_physical_progress'; break
        final, _ = session.observe('approach_final')
        summary['final_observation'] = final
        session.event('approach_summary', summary)
        (output / 'approach_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'outcome': summary['outcome'], 'segments': len(summary['steps']),
                          'position': final['registers']['Present_Position'], 'red_block': final['red_block']}), flush=True)


if __name__ == '__main__':
    main()
