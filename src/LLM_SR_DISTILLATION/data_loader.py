"""Load phase-level observations collected from a teacher policy."""
import gzip
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class TeacherDataLoader:
    """Load and manage teacher trajectories."""

    def __init__(self, json_file_path: str, *, verbose: bool = True):
        """Initialize the loader from a teacher trajectory JSON file."""
        self.json_file_path = json_file_path
        self.verbose = verbose
        self.timesteps = []
        self.intersection_ids = set()
        self.meta = {}
        self.config = {}
        self.stats = {}
        self.schema = {}
        self._load_data()

    def _load_data(self):
        """Load complete JSON or JSONL data without discarding malformed records."""
        try:
            if self.json_file_path.lower().endswith((".jsonl", ".jsonl.gz")):
                self._load_jsonl()
            else:
                try:
                    with self._open_text() as stream:
                        data = json.load(stream)
                except json.JSONDecodeError:
                    # Legacy rollouts sometimes use .json for JSONL content.
                    # Apply exactly the same strict parser as for .jsonl.
                    self._load_jsonl()
                else:
                    self._read_header(data, 'JSON root')
                    container = data.get('data', data)
                    if not isinstance(container, dict):
                        raise ValueError('JSON data must be an object')
                    if 'timesteps' in container:
                        if not isinstance(container['timesteps'], list):
                            raise ValueError('JSON timesteps must be a list')
                        for index, row in enumerate(container['timesteps']):
                            self._read_timestep(row, f'JSON timestep {index}')
                    elif 'intersection_id' in data or 'phases' in data:
                        self._read_timestep(data, 'JSON root')
            self._finish_index()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Failed to load data: {exc}") from exc

    def _read_header(self, record, location):
        if not isinstance(record, dict):
            raise ValueError(f'{location}: expected a JSON object')
        for field in ('meta', 'config', 'stats', 'schema'):
            if field not in record:
                continue
            value = record[field] if record[field] is not None else {}
            if not isinstance(value, dict):
                raise ValueError(f'{location}: {field} must be an object')
            existing = getattr(self, field)
            if existing and existing != value:
                raise ValueError(f'{location}: conflicting {field} records')
            setattr(self, field, value)

    def _read_timestep(self, row, location):
        if not isinstance(row, dict) or 'intersection_id' not in row or not isinstance(row.get('phases'), list):
            raise ValueError(f'{location}: expected an intersection_id and phases list')
        self.timesteps.append(row)

    def _load_jsonl(self) -> None:
        """Stream a plain or gzip-compressed JSONL rollout exactly once."""
        with self._open_text() as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number}: {exc}"
                    ) from exc
                location = f'JSONL line {line_number}'
                self._read_header(record, location)
                if 'intersection_id' in record or 'phases' in record:
                    self._read_timestep(record, location)
                elif not any(field in record for field in ('meta', 'config', 'stats', 'schema')):
                    raise ValueError(f'{location}: expected a trajectory header or timestep')

    def _open_text(self):
        if self.json_file_path.lower().endswith(".gz"):
            return gzip.open(self.json_file_path, "rt", encoding="utf-8")
        return open(self.json_file_path, "r", encoding="utf-8")

    def _finish_index(self) -> None:
        self.intersection_ids = {
            timestep["intersection_id"]
            for timestep in self.timesteps
            if "intersection_id" in timestep
        }
        if self.verbose:
            print(f"Successfully loaded {len(self.timesteps)} timesteps")
            print(
                f"Contains {len(self.intersection_ids)} intersections: "
                f"{sorted(self.intersection_ids)}"
            )

    def get_meta(self) -> Dict[str, Any]:
        """Return meta section from the teacher data file."""
        return dict(self.meta) if isinstance(self.meta, dict) else {}

    def get_collector_config(self) -> Dict[str, Any]:
        """Return config section from the teacher data file."""
        return dict(self.config) if isinstance(self.config, dict) else {}

    def get_header(self) -> Dict[str, Any]:
        """Use the same parsed metadata for preparation and format conversion."""
        return {field: dict(getattr(self, field)) for field in ('meta', 'config', 'stats', 'schema')}

    def get_timesteps(
        self,
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return timesteps, optionally restricted to one intersection and a maximum count."""
        timesteps = self.timesteps

        if intersection_id is not None:
            timesteps = [t for t in timesteps if t.get('intersection_id') == intersection_id]

        if max_timesteps is not None:
            timesteps = timesteps[:max_timesteps]

        return timesteps

    def get_phase_data(
        self,
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Return phase-level (state, teacher_info) pairs for the selected observations."""
        timesteps = self.get_timesteps(intersection_id, max_timesteps)
        phase_data = []

        for timestep in timesteps:
            phases = timestep.get('phases', [])
            for phase in phases:
                state = phase.get('state', {})
                teacher_info = {
                    'Q_value': phase.get('Q_value'),
                    'logit': phase.get('logit'),
                    'selected': phase.get('selected', False),
                    'phase_id': phase.get('phase_id'),
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep')
                }
                phase_data.append((state, teacher_info))

        return phase_data

    def get_action_consistency_data(
        self,
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        include_teacher_q_values: bool = True,
        include_timestep: bool = False
    ) -> List[Tuple[Dict[str, Any], int, Optional[List[float]]]]:
        """Return (state_dict, teacher_action, teacher_q_values) for fidelity evaluation.

        Q-values, when available, follow sorted physical phase IDs; otherwise None
        is returned. include_teacher_q_values controls optional soft targets."""
        timesteps = self.get_timesteps(intersection_id, max_timesteps)
        action_data = []

        for timestep in timesteps:
            phases = timestep.get('phases', [])
            if not phases:
                continue

            # Index source phase states and Q-values by physical phase ID.
            state_dict = {}
            teacher_action = None
            teacher_q_dict = {}  # {phase_id: Q_value}
            teacher_binary_q_values = timestep.get(
                "teacher_binary_q_values"
            )

            for phase in phases:
                phase_id = phase.get('phase_id')
                state = phase.get('state', {})
                state_dict[phase_id] = state


                if include_teacher_q_values:
                    q_value = phase.get('Q_value')
                    teacher_q_dict[phase_id] = q_value if q_value is not None else 0.0

                if phase.get('selected', False):
                    teacher_action = phase_id

            if teacher_action is not None:
                # Match the sorted physical phase order used by the evaluator.
                if (
                    include_teacher_q_values
                    and isinstance(teacher_binary_q_values, list)
                    and len(teacher_binary_q_values) == 2
                ):
                    teacher_q_values_list = [
                        float(teacher_binary_q_values[0]),
                        float(teacher_binary_q_values[1]),
                    ]
                elif include_teacher_q_values and teacher_q_dict:
                    sorted_phase_ids = sorted(teacher_q_dict.keys())
                    teacher_q_values_list = [teacher_q_dict[pid] for pid in sorted_phase_ids]
                else:
                    teacher_q_values_list = None

                # Debug: Check if teacher_q_values is None when it shouldn't be
                if include_teacher_q_values and teacher_q_values_list is None:
                    timestep_id = timestep.get('timestep', 'unknown')
                    print(f"      DEBUG: teacher_q_values_list is None for timestep {timestep_id}, "
                          f"teacher_q_dict={teacher_q_dict}, state_dict keys={list(state_dict.keys())}")

                if include_timestep:
                    action_data.append((state_dict, teacher_action, teacher_q_values_list, timestep.get('timestep')))
                else:
                    action_data.append((state_dict, teacher_action, teacher_q_values_list))

        return action_data

    def compute_binary_feature_stats(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """Return statistics for exact EcoLight green/red vehicle inputs."""
        values = {
            "green_vehicle": [],
            "red_vehicle": [],
        }
        for timestep in self.timesteps:
            phases = timestep.get("phases") or []
            if not phases:
                continue
            state = phases[0].get("state") or {}
            binary = state.get("binary_features") or {}
            for key in values:
                try:
                    values[key].append(float(binary[key]))
                except (KeyError, TypeError, ValueError):
                    continue

        stats: Dict[str, Dict[str, float]] = {}
        for key, raw_values in values.items():
            if not raw_values:
                continue
            array = np.asarray(raw_values, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite source binary measurements: {key}')
            std = float(np.std(array))
            stats[key] = {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
                "std": std,
                "count": int(array.size),
            }
        from stats_extension import extend
        return extend(stats, values)

    def get_teacher_metrics(
        self,
        intersection_id: Optional[int] = None
    ) -> Dict[str, float]:
        """Extract teacher performance metrics from trajectory statistics or timestep records."""
        metrics = {
            'average_delay': None,
            'average_travel_time': None,
            'average_queue_length': None,
            'average_throughput': None
        }

        # Prefer recorded trajectory statistics.
        if hasattr(self, 'stats') and self.stats:

            stats_mapping = {
                'average_delay': ['avg_delay', 'average_delay', 'avg_AD'],
                'average_travel_time': ['avg_travel_time', 'average_travel_time', 'avg_ATT'],
                'average_queue_length': ['avg_queue_length', 'average_queue_length', 'avg_AQL'],
                'average_throughput': ['avg_throughput', 'average_throughput']
            }

            for key, possible_names in stats_mapping.items():
                for name in possible_names:
                    if name in self.stats and metrics[key] is None:
                        metrics[key] = float(self.stats[name])
                        break

        # Fallback to per-timestep metrics when statistics are absent.
        if metrics['average_delay'] is None:
            timesteps = self.get_timesteps(intersection_id)
            for timestep in timesteps:
                if 'metrics' in timestep:
                    metrics_data = timestep['metrics']
                    for key in metrics:
                        if key in metrics_data and metrics[key] is None:
                            metrics[key] = float(metrics_data[key])

        return metrics

    def compute_feature_stats(self, observation_features: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute min, max, mean, std for normalization from teacher data.

        Args:
            observation_features: List of observation feature names

        Returns:
            Dictionary {feature_name: {'min', 'max', 'mean', 'std'}}
            - min/max: for Min-Max normalization using source values
            - mean/std: for Z-score normalization (more robust to outliers)
        """
        feature_stats = {}

        # Map observation feature names to aggregation kinds
        feature_kinds_map = {
            'pressure': 'pressure_sum',
            'num_vehicle': 'num_vehicle_sum',
            'num_waiting_vehicle': 'num_waiting_vehicle_sum',
            'waiting_time': 'waiting_time_sum',
        }

        # Collect all values for each feature kind
        # State can be at timestep level or inside phases[i].state (collect_teacher_data writes per-phase state)
        values_by_kind = {}

        def collect_from_fbd(fbd):
            if not isinstance(fbd, dict):
                return
            for direction_name, direction_features in fbd.items():
                if not isinstance(direction_features, dict):
                    continue

                collected_kinds = set()
                for obs_feat in observation_features or []:
                    obs_feat_lower = (obs_feat or "").lower()

                    # Skip current_phase-like globals here; they are handled separately as 'current_phase'
                    if "current_phase" in obs_feat_lower:
                        continue

                    # Map to aggregation kind
                    kind = None
                    for key, mapped_kind in feature_kinds_map.items():
                        if key in obs_feat_lower:
                            kind = mapped_kind
                            break

                    if kind is None:
                        kind = obs_feat_lower

                    # Decide preferred raw keys based on observation feature name
                    preferred_keys = []
                    if "num_waiting_vehicle" in obs_feat_lower:
                        preferred_keys = ["num_waiting_vehicle", "waiting_vehicle"]
                    elif "num_vehicle" in obs_feat_lower:
                        preferred_keys = ["num_vehicle"]
                    elif "waiting_time" in obs_feat_lower or "sum_waiting_time" in obs_feat_lower:
                        preferred_keys = ["waiting_time", "sum_waiting_time"]
                    elif "pressure" in obs_feat_lower:
                        preferred_keys = ["pressure"]
                    elif "delay" in obs_feat_lower:
                        preferred_keys = ["delay"]

                    # Try to find the raw feature in direction_features
                    raw_value = None
                    # Never substitute another physical quantity for missing data.
                    for raw_key in [obs_feat, obs_feat_lower] + preferred_keys:
                        if raw_key in direction_features:
                            try:
                                raw_value = float(direction_features[raw_key])
                                break
                            except (TypeError, ValueError):
                                continue

                    if raw_value is not None and kind not in collected_kinds:
                        collected_kinds.add(kind)
                        if kind not in values_by_kind:
                            values_by_kind[kind] = []
                        values_by_kind[kind].append(raw_value)

        for timestep in self.timesteps:
            # Top-level (legacy)
            if 'features_by_direction' in timestep:
                collect_from_fbd(timestep.get('features_by_direction', {}))
            # Per-phase state (collect_teacher_data format)
            for phase in timestep.get('phases', []):
                state = phase.get('state') or {}
                if 'features_by_direction' in state:
                    collect_from_fbd(state.get('features_by_direction', {}))

        # Compute min, max, mean, std for each feature kind
        # Return with kind names (will be remapped to actual feature names in main.py)
        for kind, values in values_by_kind.items():
            if values:
                arr = np.array(values, dtype=float)
                if not np.isfinite(arr).all():
                    raise ValueError(f'Nonfinite source measurements: {kind}')
                mu = float(np.mean(arr))
                sigma = float(np.std(arr))
                feature_stats[kind] = {
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'mean': mu,
                    'std': sigma,
                    'count': int(arr.size),
                }

        # current_phase/phase_active: binary 0 or 1 only; min/max/mean/std for compatibility
        feature_stats['current_phase'] = {'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.5}
        from stats_extension import extend
        feature_stats = extend(feature_stats, values_by_kind)

        return feature_stats

    def compute_phase_feature_stats(self, extractor) -> Dict[str, Dict[str, float]]:
        """Source-rollout statistics in the same phase aggregation used at runtime."""
        values = {spec.name: [] for spec in extractor.specs}
        for timestep in self.timesteps:
            phases = timestep.get('phases') or []
            for phase_index, phase in enumerate(phases):
                state = phase.get('state') or {}
                if not isinstance(state, dict):
                    continue
                phase_id = phase.get('phase_id', phase_index)
                for spec in extractor.specs:
                    value = float(extractor._extract_one(spec, state, int(phase_id)))
                    if np.isfinite(value):
                        values[spec.name].append(value)
        result = {}
        for name, samples in values.items():
            if not samples:
                raise ValueError(f'No source phase samples for {name}')
            arr = np.asarray(samples, dtype=float)
            result[name] = {
                'min': float(arr.min()), 'max': float(arr.max()),
                'mean': float(arr.mean()), 'std': max(float(arr.std()), 1e-8),
            }
        return result
