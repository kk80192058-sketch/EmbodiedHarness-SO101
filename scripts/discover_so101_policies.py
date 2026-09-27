"""Discover pinned SO-101 SmolVLA candidates. This never activates a policy."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error('limit must be 1..50')
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    output = Path('artifacts/policy_discovery') / str(time.time_ns())
    output.mkdir(parents=True)
    candidates = []
    print('evidence_dir=' + str(output.resolve()), flush=True)
    ids = ['nota-gmbh/so101_pick_place_smolvla']
    ids += [m.id for m in api.list_models(search='so101', sort='downloads', limit=100)
            if 'smolvla' in m.id.lower()]
    for repo in list(dict.fromkeys(ids))[:args.limit]:
        candidate = {'repo_id': repo, 'activation': 'not_validated'}
        try:
            metadata = api.model_info(repo, files_metadata=True)
            config = json.loads(Path(hf_hub_download(repo, 'config.json', revision=metadata.sha)).read_text())
            files = {s.rfilename: s.size for s in metadata.siblings}
            candidate.update(revision=metadata.sha, config=config, files=files,
                             card_data=metadata.card_data.to_dict() if metadata.card_data else {})
            reasons = []
            if config.get('type') != 'smolvla': reasons.append('not SmolVLA')
            if config.get('output_features', {}).get('action', {}).get('shape') != [6]: reasons.append('not six actions')
            if config.get('input_features', {}).get('observation.state', {}).get('shape') != [6]: reasons.append('not six state values')
            cameras = [k for k in config.get('input_features', {}) if k.startswith('observation.images.')]
            if len(cameras) != 2: reasons.append('does not match two-camera input count')
            for required in ('model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                if required not in files: reasons.append(f'missing {required}')
            candidate.update(structural_match=not reasons, rejection_reasons=reasons,
                             requires_review=['joint order and units', 'robot/calibration conventions',
                                              'camera view and task distribution', 'real vs simulation provenance'])
        except Exception as error:
            candidate.update(structural_match=False, error=repr(error))
        candidates.append(candidate)
        (output / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
        print(json.dumps({k: candidate[k] for k in ('repo_id', 'structural_match')}), flush=True)


if __name__ == '__main__':
    main()
