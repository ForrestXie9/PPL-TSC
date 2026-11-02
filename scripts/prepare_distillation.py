#!/usr/bin/env python3
"""Validate a source trajectory and create its distillation configuration offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ppl_tsc.artifacts import atomic_json, new_gzip_jsonl


def phase_projection(source, destination, *, loader=None):
    """Preserve physical teacher actions; remove native-only supervision fields."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists() or destination.is_symlink():
        raise ValueError('Projection output must be new and different from the source')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if loader is None:
        from ppl_tsc.data.loader import TeacherDataLoader
        loader = TeacherDataLoader(str(source))

    def rows():
        if not loader.timesteps:
            raise ValueError('Trajectory must contain a header and observations')
        header = loader.get_header()
        header['meta'].update(action_interface='physical_phase_selection_projection',
                              native_trajectory_sha256=digest)
        header['meta'].pop('binary_supervision', None)
        header['schema']['binary_interface'] = False
        yield header
        for original in loader.timesteps:
            row = copy.deepcopy(original)
            phases, action = row['phases'], row['teacher_action']
            if [phase['phase_id'] for phase in phases if phase.get('selected') is True] != [action]:
                raise ValueError('Physical phase labels are inconsistent')
            for key in ('teacher_binary_action', 'teacher_binary_q_values'):
                row.pop(key, None)
            for phase in phases:
                for key in ('binary_features', 'exact_binary_features'):
                    phase['state'].pop(key, None)
            yield row

    new_gzip_jsonl(destination, rows())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True,
                        choices=sorted(path.stem for path in (ROOT / 'configs/examples').glob('*.json')))
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=2026103101)
    parser.add_argument('--endpoint', default='api.n1n.ai')
    parser.add_argument('--model', default='gemini-3.1-flash-lite')
    args = parser.parse_args()
    from ppl_tsc.data.loader import TeacherDataLoader
    from ppl_tsc.data.preparation import prepare_source

    trajectory = args.trajectory.resolve()
    config = json.loads((ROOT / 'configs/examples' / f'{args.variant}.json').read_text())
    traffic_path = ROOT / 'configs/traffic.json'
    from ppl_tsc.config import source_environment
    traffic = source_environment(traffic_path)
    teacher = config['data']['teacher_model']
    try:
        loader = TeacherDataLoader(str(trajectory))
        meta = loader.get_meta()
        if meta.get('model') != teacher:
            raise ValueError('Trajectory teacher does not match variant')
        if meta.get('direction_feature_semantics') != traffic['direction_feature_semantics']:
            raise ValueError('Trajectory direction feature semantics do not match the current method')
        if meta.get('environment_semantics') != traffic.get('waiting_time_semantics', 'per_decision'):
            raise ValueError('Trajectory environment semantics do not match the current method')
        if type(meta.get('seed')) is not int:
            raise ValueError('Trajectory must record an integer simulation seed')
        # Validate native EcoLight before projecting its physical phase labels.
        source_mode = 'binary_keep_switch' if args.variant.startswith('EcoLight') and any(
            'teacher_binary_action' in row for row in loader.timesteps
        ) else config['search']['action_mode']
        prepare_source(loader, teacher, traffic[teacher]['observation_feature_list'], traffic,
                       source_mode, project_root=ROOT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    output = args.output_dir.resolve()
    destination = output / 'config.json'
    if destination.exists() or destination.is_symlink():
        parser.error('Config already exists; choose a new output directory')
    if args.variant in {'EcoLight-phase-selection', 'EcoLight-original'} and source_mode == 'binary_keep_switch':
        projected = output / 'phase_trajectory.jsonl.gz'
        phase_projection(trajectory, projected, loader=loader)
        trajectory = projected
    config['experiment'].update(
        name=f"{args.variant}__{meta['dataset']}__flow{meta['flow_idx']}",
        output_dir=str(output), random_seed=args.seed)
    config['data'].update(teacher_data=str(trajectory), traffic_config=str(traffic_path))
    config['evaluation']['simulation_seed'] = meta['seed']
    config['llm'].update(endpoint=args.endpoint, model=args.model, cache_dir=str(output / 'llm_cache'))
    atomic_json(destination, config, overwrite=False)
    print(destination)


if __name__ == '__main__':
    main()
