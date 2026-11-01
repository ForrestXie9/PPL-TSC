#!/usr/bin/env python3
"""Export an explicit formula-only cross-network variant without changing its source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from evaluate import validate_deployment
from ppl_tsc_current.artifact_io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.output.resolve() == args.deployment.resolve():
        parser.error('Output must be a new file')
    source_bytes = args.deployment.read_bytes()
    deployment = json.loads(source_bytes)
    try:
        if not isinstance(deployment, dict):
            raise ValueError('Deployment must be a JSON object')
        if not isinstance(deployment.get('function_code'), str) or not deployment['function_code'].strip():
            raise ValueError('Deployment must contain function_code')
        config = deployment.get('run_config')
        if not isinstance(config, dict) or config.get('teacher_model') not in {
            'CoLight', 'EcoLight', 'FRAP', 'MPLight', 'TinyLight'
        }:
            raise ValueError('Deployment must record a supported source teacher')
        validate_deployment(deployment, config.get('source_roadnet_sha256') or '')
        if 'transfer_export' in deployment:
            raise ValueError('Deployment is already a formula-transfer export; use the original source policy')
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    removed = deployment.get('per_intersection_bias', {})
    deployment['per_intersection_bias'] = {}
    # Source metrics describe the calibrated policy, not this unevaluated variant.
    source_metrics = deployment.pop('metrics', None)
    source_score = deployment.pop('score', None)
    deployment['transfer_export'] = {
        'protocol': 'formula_only_source_statistics_v1',
        'evaluation_status': 'not_evaluated',
        'source_deployment_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'source_policy_metrics': source_metrics,
        'source_policy_score': source_score,
        'removed_source_bias': removed,
        'description': 'Fixed formula and source feature statistics; source-local biases explicitly removed. '
                       'No target search, calibration or teacher rollout. '
                       'This is not transfer of the full calibrated policy.',
    }
    atomic_json(args.output, deployment, overwrite=False)
    print(args.output)


if __name__ == '__main__':
    main()
