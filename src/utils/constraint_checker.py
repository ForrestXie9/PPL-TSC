"""Legacy diagnostics for teacher observations and generated phase utility expressions."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import re


def _load_timesteps_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """Load timesteps from a teacher JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    timesteps = data.get('data', {}).get('timesteps', [])

    if not timesteps:

        timesteps = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'intersection_id' in record and 'phases' in record:
                        timesteps.append(record)
                except:
                    pass

    return timesteps


def check_phase_utility_constraints(
    timesteps: List[Dict[str, Any]],
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Deprecated: inspect constraints on teacher Q-values/logits, not generated policies."""

    if intersection_id is not None:
        timesteps = [t for t in timesteps if t.get('intersection_id') == intersection_id]

    if len(timesteps) == 0:
        return {
            'boundary_violations': [],
            'trend_violations': [],
            'summary': {'total_timesteps': 0, 'error': 'No valid timesteps found'}
        }

    boundary_violations = []
    trend_violations = []


    for timestep_idx, timestep in enumerate(timesteps):
        phases = timestep.get('phases', [])
        if not phases:
            continue


        q_values = []
        logits = []
        selected_phase_id = None
        selected_q = None

        for phase in phases:
            q_val = phase.get('Q_value')
            logit = phase.get('logit')

            if q_val is not None:
                q_values.append((phase['phase_id'], q_val))
                if phase.get('selected', False):
                    selected_phase_id = phase['phase_id']
                    selected_q = q_val

            if logit is not None:
                logits.append((phase['phase_id'], logit))


        for phase_id, q_val in q_values:
            if not np.isfinite(q_val):
                violation = {
                    'type': 'invalid_q_value',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'q_value': q_val,
                    'message': f'Nonfinite Q-value: {q_val} (NaN or Inf)'
                }
                boundary_violations.append(violation)


        for phase_id, logit in logits:
            if not np.isfinite(logit):
                violation = {
                    'type': 'invalid_logit',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'logit': logit,
                    'message': f'Nonfinite logit: {logit} (NaN or Inf)'
                }
                boundary_violations.append(violation)


        if selected_phase_id is not None and len(q_values) > 1:
            max_q_phase_id, max_q = max(q_values, key=lambda x: x[1])
            if selected_phase_id != max_q_phase_id:

                q_diff = max_q - selected_q
                if q_diff > 0.1:
                    violation = {
                        'type': 'selected_phase_not_max_q',
                        'intersection_id': timestep.get('intersection_id'),
                        'timestep': timestep.get('timestep'),
                        'selected_phase_id': selected_phase_id,
                        'selected_q': selected_q,
                        'max_q_phase_id': max_q_phase_id,
                        'max_q': max_q,
                        'q_diff': q_diff,
                        'message': f'Selected phase {selected_phase_id} has Q-value ({selected_q:.4f}), below the maximum at phase {max_q_phase_id}({max_q:.4f})'
                    }
                    boundary_violations.append(violation)


        for phase_id, q_val in q_values:
            if abs(q_val) > 1000:
                violation = {
                    'type': 'q_value_out_of_range',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'q_value': q_val,
                    'message': f'Q-value outside the diagnostic range: {q_val}'
                }
                boundary_violations.append(violation)



    for timestep_idx in range(len(timesteps) - 1):
        current_timestep = timesteps[timestep_idx]
        next_timestep = timesteps[timestep_idx + 1]


        if current_timestep.get('intersection_id') != next_timestep.get('intersection_id'):
            continue

        current_phases = {p['phase_id']: p for p in current_timestep.get('phases', [])}
        next_phases = {p['phase_id']: p for p in next_timestep.get('phases', [])}


        for phase_id in current_phases.keys():
            if phase_id not in next_phases:
                continue

            current_phase = current_phases[phase_id]
            next_phase = next_phases[phase_id]

            current_q = current_phase.get('Q_value')
            next_q = next_phase.get('Q_value')

            if current_q is None or next_q is None:
                continue


            current_state = current_phase.get('state', {})
            next_state = next_phase.get('state', {})

            controlled_directions = current_state.get('controlled_directions', [])
            if not controlled_directions:
                continue


            current_features = current_state.get('features_by_direction', {})
            next_features = next_state.get('features_by_direction', {})


            for direction in controlled_directions:
                if direction not in current_features or direction not in next_features:
                    continue

                current_dir_features = current_features[direction]
                next_dir_features = next_features[direction]


                if 'num_vehicle' in current_dir_features and 'num_vehicle' in next_dir_features:
                    current_num = current_dir_features['num_vehicle']
                    next_num = next_dir_features['num_vehicle']
                    num_change = next_num - current_num
                    q_change = next_q - current_q


                    if num_change > 0.1 and q_change < -0.1:
                        violation = {
                            'type': 'trend_violation_vehicle_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'num_vehicle_change': num_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'Direction {direction} vehicle count increased by {num_change:.2f}, but phase {phase_id} Q-value decreased by {q_change:.4f}'
                        }
                        trend_violations.append(violation)


                if 'pressure' in current_dir_features and 'pressure' in next_dir_features:
                    current_pressure = current_dir_features['pressure']
                    next_pressure = next_dir_features['pressure']
                    pressure_change = next_pressure - current_pressure
                    q_change = next_q - current_q


                    if pressure_change > 0.1 and q_change < -0.1:
                        violation = {
                            'type': 'trend_violation_pressure_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'pressure_change': pressure_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'Direction {direction} pressure increased by {pressure_change:.2f}, but phase {phase_id} Q-value decreased by {q_change:.4f}'
                        }
                        trend_violations.append(violation)


                if 'waiting_time' in current_dir_features and 'waiting_time' in next_dir_features:
                    current_wait = current_dir_features['waiting_time']
                    next_wait = next_dir_features['waiting_time']
                    wait_change = next_wait - current_wait
                    q_change = next_q - current_q


                    if wait_change > 0.1 and q_change < -0.1:
                        violation = {
                            'type': 'trend_violation_waiting_time_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'waiting_time_change': wait_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'Direction {direction} waiting time increased by {wait_change:.2f}, but phase {phase_id} Q-value decreased by {q_change:.4f}'
                        }
                        trend_violations.append(violation)



            all_directions = set()
            for phase in current_phases.values():
                all_directions.update(phase.get('state', {}).get('controlled_directions', []))

            for direction in all_directions:
                if direction in controlled_directions:
                    continue


                for other_phase_id, other_phase in current_phases.items():
                    if other_phase_id == phase_id:
                        continue

                    other_controlled = other_phase.get('state', {}).get('controlled_directions', [])
                    if direction in other_controlled:

                        continue


                    if direction in current_features and direction in next_features:
                        current_dir_features = current_features[direction]
                        next_dir_features = next_features[direction]


                        if 'num_vehicle' in current_dir_features and 'num_vehicle' in next_dir_features:
                            num_change = next_dir_features['num_vehicle'] - current_dir_features['num_vehicle']
                            q_change = next_q - current_q


                            if num_change > 0.1 and q_change > 0.2:
                                violation = {
                                    'type': 'trend_violation_uncontrolled_direction',
                                    'intersection_id': current_timestep.get('intersection_id'),
                                    'timestep_from': current_timestep.get('timestep'),
                                    'timestep_to': next_timestep.get('timestep'),
                                    'phase_id': phase_id,
                                    'direction': direction,
                                    'num_vehicle_change': num_change,
                                    'q_value_change': q_change,
                                    'message': f'Direction {direction} vehicle count increased by {num_change:.2f}, but phase {phase_id} does not serve this direction, yet its Q-value increased by {q_change:.4f} (potential inconsistency)'
                                }
                                trend_violations.append(violation)


    summary = {
        'total_timesteps': len(timesteps),
        'total_boundary_violations': len(boundary_violations),
        'total_trend_violations': len(trend_violations),
        'boundary_violation_types': {},
        'trend_violation_types': {}
    }


    for violation in boundary_violations:
        v_type = violation['type']
        summary['boundary_violation_types'][v_type] = summary['boundary_violation_types'].get(v_type, 0) + 1


    for violation in trend_violations:
        v_type = violation['type']
        summary['trend_violation_types'][v_type] = summary['trend_violation_types'].get(v_type, 0) + 1


    if verbose:
        print("=" * 80)
        print("Constraint report")
        print("=" * 80)
        print(f"Total timesteps: {summary['total_timesteps']}")
        print(f"Boundary violations: {summary['total_boundary_violations']}")
        print(f"Trend violations: {summary['total_trend_violations']}")

        if summary['boundary_violation_types']:
            print("\nBoundary violation types:")
            for v_type, count in summary['boundary_violation_types'].items():
                print(f"  - {v_type}: {count} occurrences")

        if summary['trend_violation_types']:
            print("\nTrend violation types:")
            for v_type, count in summary['trend_violation_types'].items():
                print(f"  - {v_type}: {count} occurrences")


        if boundary_violations:
            print("\nBoundary violation examples (first 5):")
            for i, violation in enumerate(boundary_violations[:5]):
                print(f"  {i+1}. {violation['message']}")

        if trend_violations:
            print("\nTrend violation examples (first 5):")
            for i, violation in enumerate(trend_violations[:5]):
                print(f"  {i+1}. {violation['message']}")

        print("=" * 80)

    return {
        'boundary_violations': boundary_violations,
        'trend_violations': trend_violations,
        'summary': summary
    }


def check_generated_function_constraints(
    expression_template: str,
    timesteps: List[Dict[str, Any]],
    feature_names: List[str],
    feature_mapping: Dict[str, Callable],
    intersection_id: Optional[int] = None,
    num_phases: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Check a generated expression against boundary and trend diagnostics."""
    from agent.imitation_based.llm_unified_compiler import UnifiedExpressionCompiler


    if intersection_id is not None:
        timesteps = [t for t in timesteps if t.get('intersection_id') == intersection_id]

    if len(timesteps) == 0:
        return {
            'boundary_violations': [],
            'trend_violations': [],
            'expression_valid': False,
            'summary': {'total_timesteps': 0, 'error': 'No valid timesteps found'}
        }


    if num_phases is None:
        if timesteps and timesteps[0].get('phases'):
            num_phases = len(timesteps[0]['phases'])
        else:
            num_phases = 8

    boundary_violations = []
    trend_violations = []
    expression_valid = True


    compiler = UnifiedExpressionCompiler()


    compiled_functions = {}
    for phase_idx in range(num_phases):
        try:
            func, coeff_names = compiler.compile_unified_expression(expression_template, phase_idx)
            compiled_functions[phase_idx] = (func, coeff_names)
        except Exception as e:
            expression_valid = False
            violation = {
                'type': 'expression_compile_error',
                'phase_idx': phase_idx,
                'error': str(e),
                'message': f'Phase {phase_idx} expression failed to compile: {str(e)}'
            }
            boundary_violations.append(violation)
            if verbose:
                print(f"Warning: phase {phase_idx} expression failed to compile: {e}")

    if not expression_valid:
        return {
            'boundary_violations': boundary_violations,
            'trend_violations': [],
            'expression_valid': False,
            'summary': {'total_timesteps': len(timesteps), 'error': 'Expression compilation failed'}
        }



    for timestep_idx, timestep in enumerate(timesteps[:100]):
        phases = timestep.get('phases', [])
        if not phases:
            continue

        for phase in phases:
            phase_id = phase.get('phase_id')
            if phase_id not in compiled_functions:
                continue

            func, coeff_names = compiled_functions[phase_id]
            state = phase.get('state', {})


            try:
                feature_values = []
                for feat_name in feature_names:
                    if feat_name in feature_mapping:
                        feat_value = feature_mapping[feat_name](state, phase)
                        feature_values.append(feat_value)
                    else:

                        feature_values.append(0.0)

                X = np.array([feature_values])


                num_coeffs = len(coeff_names)
                test_coeffs = np.random.randn(num_coeffs) * 0.1


                try:
                    q_value = func(X, *test_coeffs)[0]


                    if not np.isfinite(q_value):
                        violation = {
                            'type': 'invalid_q_value',
                            'intersection_id': timestep.get('intersection_id'),
                            'timestep': timestep.get('timestep'),
                            'phase_id': phase_id,
                            'q_value': q_value,
                            'message': f'Expression returned a nonfinite Q-value: {q_value} (NaN or Inf)'
                        }
                        boundary_violations.append(violation)


                    if abs(q_value) > 1000:
                        violation = {
                            'type': 'q_value_out_of_range',
                            'intersection_id': timestep.get('intersection_id'),
                            'timestep': timestep.get('timestep'),
                            'phase_id': phase_id,
                            'q_value': q_value,
                            'message': f'Expression Q-value outside the diagnostic range: {q_value}'
                        }
                        boundary_violations.append(violation)

                except Exception as e:
                    violation = {
                        'type': 'expression_evaluation_error',
                        'intersection_id': timestep.get('intersection_id'),
                        'timestep': timestep.get('timestep'),
                        'phase_id': phase_id,
                        'error': str(e),
                        'message': f'Expression evaluation failed: {str(e)}'
                    }
                    boundary_violations.append(violation)

            except Exception as e:
                if verbose:
                    print(f"Warning: feature extraction failed (timestep={timestep_idx}, phase={phase_id}): {e}")



    for timestep_idx in range(min(len(timesteps) - 1, 100)):
        current_timestep = timesteps[timestep_idx]
        next_timestep = timesteps[timestep_idx + 1]

        if current_timestep.get('intersection_id') != next_timestep.get('intersection_id'):
            continue

        current_phases = {p['phase_id']: p for p in current_timestep.get('phases', [])}
        next_phases = {p['phase_id']: p for p in next_timestep.get('phases', [])}

        for phase_id in current_phases.keys():
            if phase_id not in next_phases or phase_id not in compiled_functions:
                continue

            current_phase = current_phases[phase_id]
            next_phase = next_phases[phase_id]

            func, coeff_names = compiled_functions[phase_id]
            num_coeffs = len(coeff_names)
            test_coeffs = np.random.randn(num_coeffs) * 0.1


            try:
                current_features = []
                next_features = []

                for feat_name in feature_names:
                    if feat_name in feature_mapping:
                        current_features.append(feature_mapping[feat_name](current_phase.get('state', {}), current_phase))
                        next_features.append(feature_mapping[feat_name](next_phase.get('state', {}), next_phase))
                    else:
                        current_features.append(0.0)
                        next_features.append(0.0)

                current_X = np.array([current_features])
                next_X = np.array([next_features])

                current_q = func(current_X, *test_coeffs)[0]
                next_q = func(next_X, *test_coeffs)[0]



                current_state = current_phase.get('state', {})
                next_state = next_phase.get('state', {})

                controlled_directions = current_state.get('controlled_directions', [])
                current_features_by_dir = current_state.get('features_by_direction', {})
                next_features_by_dir = next_state.get('features_by_direction', {})

                for direction in controlled_directions:
                    if direction not in current_features_by_dir or direction not in next_features_by_dir:
                        continue

                    current_dir_feat = current_features_by_dir[direction]
                    next_dir_feat = next_features_by_dir[direction]


                    if 'num_vehicle' in current_dir_feat and 'num_vehicle' in next_dir_feat:
                        num_change = next_dir_feat['num_vehicle'] - current_dir_feat['num_vehicle']
                        q_change = next_q - current_q

                        if num_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_vehicle_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'num_vehicle_change': num_change,
                                'q_value_change': q_change,
                                'message': f'Direction {direction} vehicle count increased by {num_change:.2f}, but expression Q-value decreased by {q_change:.4f} (unexpected trend)'
                            }
                            trend_violations.append(violation)


                    if 'pressure' in current_dir_feat and 'pressure' in next_dir_feat:
                        pressure_change = next_dir_feat['pressure'] - current_dir_feat['pressure']
                        q_change = next_q - current_q

                        if pressure_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_pressure_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'pressure_change': pressure_change,
                                'q_value_change': q_change,
                                'message': f'Direction {direction} pressure increased by {pressure_change:.2f}, but expression Q-value decreased by {q_change:.4f} (unexpected trend)'
                            }
                            trend_violations.append(violation)


                    if 'num_waiting_vehicle' in current_dir_feat and 'num_waiting_vehicle' in next_dir_feat:
                        waiting_change = next_dir_feat['num_waiting_vehicle'] - current_dir_feat['num_waiting_vehicle']
                        q_change = next_q - current_q

                        if waiting_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_waiting_vehicle_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'waiting_vehicle_change': waiting_change,
                                'q_value_change': q_change,
                                'message': f'Direction {direction} waiting-vehicle count increased by {waiting_change:.2f}, but expression Q-value decreased by {q_change:.4f} (unexpected trend)'
                            }
                            trend_violations.append(violation)

            except Exception as e:
                if verbose:
                    print(f"Warning: trend check failed (timestep={timestep_idx}, phase={phase_id}): {e}")


    summary = {
        'total_timesteps': len(timesteps),
        'expression_valid': expression_valid,
        'total_boundary_violations': len(boundary_violations),
        'total_trend_violations': len(trend_violations),
        'boundary_violation_types': {},
        'trend_violation_types': {}
    }

    for violation in boundary_violations:
        v_type = violation['type']
        summary['boundary_violation_types'][v_type] = summary['boundary_violation_types'].get(v_type, 0) + 1

    for violation in trend_violations:
        v_type = violation['type']
        summary['trend_violation_types'][v_type] = summary['trend_violation_types'].get(v_type, 0) + 1

    if verbose:
        print("=" * 80)
        print("Generated function constraint report")
        print("=" * 80)
        print(f"Expression template: {expression_template[:100]}...")
        print(f"Expression valid: {expression_valid}")
        print(f"Total timesteps: {summary['total_timesteps']}")
        print(f"Boundary violations: {summary['total_boundary_violations']}")
        print(f"Trend violations: {summary['total_trend_violations']}")

        if summary['boundary_violation_types']:
            print("\nBoundary violation types:")
            for v_type, count in summary['boundary_violation_types'].items():
                print(f"  - {v_type}: {count} occurrences")

        if summary['trend_violation_types']:
            print("\nTrend violation types:")
            for v_type, count in summary['trend_violation_types'].items():
                print(f"  - {v_type}: {count} occurrences")

        if boundary_violations:
            print("\nBoundary violation examples (first 5):")
            for i, violation in enumerate(boundary_violations[:5]):
                print(f"  {i+1}. {violation['message']}")

        if trend_violations:
            print("\nTrend violation examples (first 5):")
            for i, violation in enumerate(trend_violations[:5]):
                print(f"  {i+1}. {violation['message']}")

        print("=" * 80)

    return {
        'boundary_violations': boundary_violations,
        'trend_violations': trend_violations,
        'expression_valid': expression_valid,
        'summary': summary
    }


def build_feature_mapping_for_teacher_model(
    teacher_model: str,
    observation_feature_list: List[str]
) -> Dict[str, Callable]:
    """Build observation-feature extractors for the selected teacher interface."""
    feature_mapping = {}

    for feat_name in observation_feature_list:
        if feat_name == 'inter_2_current_phase':
            def extract_current_phase(state, phase, feat_name=feat_name):
                return float(state.get('current_phase', 0))
            feature_mapping[feat_name] = extract_current_phase

        elif feat_name == 'lanelink_2_pressure':

            def extract_lanelink_pressure(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_pressure = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        pressure = features_by_dir[direction].get('pressure', 0.0)
                        total_pressure += pressure
                        count += 1
                return total_pressure / max(count, 1)
            feature_mapping[feat_name] = extract_lanelink_pressure

        elif feat_name == 'lanelink_2_num_vehicle':

            def extract_lanelink_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_lanelink_num_vehicle

        elif feat_name == 'lane_2_num_vehicle':

            def extract_lane_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_lane_num_vehicle

        elif feat_name == 'inlane_2_num_waiting_vehicle':

            def extract_inlane_waiting_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_waiting = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        waiting = features_by_dir[direction].get('num_waiting_vehicle', 0.0)
                        total_waiting += waiting
                        count += 1
                return total_waiting / max(count, 1)
            feature_mapping[feat_name] = extract_inlane_waiting_vehicle

        elif feat_name == 'inlane_2_num_vehicle':

            def extract_inlane_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_inlane_num_vehicle

        elif feat_name == 'outlane_2_num_vehicle':

            def extract_outlane_num_vehicle(state, phase, feat_name=feat_name):


                other_features = state.get('other_features', {})
                if feat_name in other_features:
                    if isinstance(other_features[feat_name], dict):
                        return other_features[feat_name].get('mean', 0.0)
                    return float(other_features[feat_name])
                return 0.0
            feature_mapping[feat_name] = extract_outlane_num_vehicle

        else:

            def extract_default(state, phase, feat_name=feat_name):
                other_features = state.get('other_features', {})
                if feat_name in other_features:
                    if isinstance(other_features[feat_name], dict):
                        return other_features[feat_name].get('mean', 0.0)
                    return float(other_features[feat_name])
                return 0.0
            feature_mapping[feat_name] = extract_default

    return feature_mapping


def build_default_feature_mapping(feature_names: List[str]) -> Dict[str, Callable]:
    """Build default extractors for the supplied feature names."""
    feature_mapping = {}

    for feat_name in feature_names:

        if feat_name in ('phase_active', 'x0') or 'current_phase' in feat_name.lower():

            def extract_current_phase(state, phase, feat_name=feat_name):
                return float(state.get('current_phase', 0))
            feature_mapping[feat_name] = extract_current_phase

        elif 'pressure' in feat_name.lower() or 'x1' in feat_name:

            def extract_pressure(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_pressure = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        pressure = features_by_dir[direction].get('pressure', 0.0)
                        total_pressure += pressure
                        count += 1
                return total_pressure / max(count, 1)
            feature_mapping[feat_name] = extract_pressure

        elif 'vehicle' in feat_name.lower() or 'num_vehicle' in feat_name.lower():

            def extract_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_num_vehicle

        elif 'waiting' in feat_name.lower() or 'wait_time' in feat_name.lower():

            def extract_waiting_time(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_wait = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        wait_time = features_by_dir[direction].get('waiting_time', 0.0)
                        total_wait += wait_time
                        count += 1
                return total_wait / max(count, 1)
            feature_mapping[feat_name] = extract_waiting_time

        else:

            def extract_default(state, phase, feat_name=feat_name):
                return 0.0
            feature_mapping[feat_name] = extract_default

    return feature_mapping


def check_generated_function_for_teacher_model(
    expression_template: str,
    json_file_path: str,
    teacher_model: str,
    observation_feature_list: List[str],
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Check a generated expression using the selected teacher feature mapping."""

    feature_mapping = build_feature_mapping_for_teacher_model(teacher_model, observation_feature_list)


    from agent.imitation_based.feature_descriptions import get_feature_names_mapping
    feature_name_to_var = get_feature_names_mapping(observation_feature_list)
    feature_names = [feature_name_to_var[feat] for feat in observation_feature_list]


    var_feature_mapping = {}
    for feat_name, var_name in feature_name_to_var.items():
        if feat_name in feature_mapping:
            var_feature_mapping[var_name] = feature_mapping[feat_name]

    return check_generated_function_from_json(
        expression_template, json_file_path, feature_names, var_feature_mapping, intersection_id, verbose
    )


def check_generated_function_from_json(
    expression_template: str,
    json_file_path: str,
    feature_names: List[str],
    feature_mapping: Optional[Dict[str, Callable]] = None,
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Load source JSON and check a generated expression using the supplied or default mapping."""
    timesteps = _load_timesteps_from_json(json_file_path)


    if feature_mapping is None:
        feature_mapping = build_default_feature_mapping(feature_names)
        if verbose:
            print(f"Using the default mapping ({len(feature_mapping)} features)")

    return check_generated_function_constraints(
        expression_template, timesteps, feature_names, feature_mapping, intersection_id, None, verbose
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python constraint_checker.py <json_file_path> [intersection_id]")
        sys.exit(1)

    json_file = sys.argv[1]
    intersection_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    results = check_constraints_from_json_file(json_file, intersection_id, verbose=True)


    output_file = json_file.replace('.json', '_constraint_check.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {output_file}")

