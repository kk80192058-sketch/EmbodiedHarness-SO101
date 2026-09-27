"""Audit the provenance and schema of SO-101 human-positioned contact samples."""
import argparse
import json
from pathlib import Path

from harness.contact_evidence import audit_contact_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True, help='JSON manifest containing sample paths')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-samples', type=int, default=5)
    args = parser.parse_args()
    result = audit_contact_manifest(args.samples, min_samples=args.min_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'accepted': result['accepted'], 'valid_sample_count': result.get('valid_sample_count'),
                      'errors': result['errors']}, indent=2))
    if not result['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
