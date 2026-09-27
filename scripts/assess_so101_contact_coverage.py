"""Compare a saved SO-101 observation with audited contact calibration poses."""
import argparse
import json
from pathlib import Path

from harness.contact_coverage import assess_contact_pose_coverage
from harness.contact_evidence import audit_contact_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = audit_contact_manifest(args.samples)
    if not audit['accepted']:
        raise SystemExit('Refusing unaudited contact evidence: ' + '; '.join(audit['errors']))
    manifest = json.loads(args.samples.read_text())
    samples = [json.loads(Path(path).read_text()) for path in manifest['samples']]
    observation = json.loads(args.observation.read_text())
    result = assess_contact_pose_coverage(samples, observation['registers']['Present_Position'])
    result['contact_evidence_sha256'] = audit['source_sha256']
    result['source_observation'] = str(args.observation.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'inside_all_observed_joint_envelopes': result['inside_all_observed_joint_envelopes'],
                      'outside_joints': result['outside_joints'], 'nearest_sample': result['nearest_sample']}, indent=2))


if __name__ == '__main__':
    main()
