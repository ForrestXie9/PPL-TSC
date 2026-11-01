"""
数据加载模块
加载teacher模型收集的phase-level数据
"""
import gzip
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class TeacherDataLoader:
    """加载和管理teacher模型数据"""
    
    def __init__(self, json_file_path: str):
        """
        初始化数据加载器
        
        Args:
            json_file_path: teacher数据JSON文件路径
        """
        self.json_file_path = json_file_path
        self.timesteps = []
        self.intersection_ids = set()
        self.meta = {}
        self.config = {}
        self.stats = {}
        self._load_data()
    
    def _load_data(self):
        """从JSON文件加载数据"""
        try:
            # JSONL is streamed directly. The legacy loader first attempted
            # json.load(), which read a large JSONL file twice.
            if self.json_file_path.lower().endswith((".jsonl", ".jsonl.gz")):
                self._load_jsonl()
                self._finish_index()
                return

            # First try to load as a single JSON object
            try:
                with self._open_text() as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: treat as JSONL (one JSON object per line)
                self.meta = {}
                self.config = {}
                self.stats = {}
                self.timesteps = []
                with self._open_text() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # If this line looks like a timestep record, keep it.
                        if 'intersection_id' in record and 'phases' in record:
                            self.timesteps.append(record)
                        # Optionally pick up meta/config/stats if present on any line.
                        if isinstance(record, dict):
                            if 'meta' in record and not self.meta:
                                self.meta = record.get('meta') or {}
                            if 'config' in record and not self.config:
                                self.config = record.get('config') or {}
                            if 'stats' in record and not self.stats:
                                self.stats = record.get('stats') or {}
            else:
                # Save meta/config/stats if present
                self.meta = data.get('meta', {}) or {}
                self.config = data.get('config', {}) or {}
                self.stats = data.get('stats', {}) or {}
                
                # 尝试从标准格式加载
                if 'data' in data and 'timesteps' in data['data']:
                    self.timesteps = data['data']['timesteps']
                elif 'timesteps' in data:
                    self.timesteps = data['timesteps']
                else:
                    # 尝试JSONL格式（data 是一些非结构化字段）
                    self.timesteps = []
                    with self._open_text() as f:
                        for line in f:
                            line = strip_line = line.strip()
                            if not strip_line:
                                continue
                            try:
                                record = json.loads(strip_line)
                            except json.JSONDecodeError:
                                continue
                            if 'intersection_id' in record and 'phases' in record:
                                self.timesteps.append(record)
            
            self._finish_index()
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

    def _load_jsonl(self) -> None:
        """Stream a plain or gzip-compressed JSONL rollout exactly once."""
        self.meta = {}
        self.config = {}
        self.stats = {}
        self.timesteps = []
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
                if not isinstance(record, dict):
                    continue
                if "intersection_id" in record and "phases" in record:
                    self.timesteps.append(record)
                if "meta" in record and not self.meta:
                    self.meta = record.get("meta") or {}
                if "config" in record and not self.config:
                    self.config = record.get("config") or {}
                if "stats" in record and not self.stats:
                    self.stats = record.get("stats") or {}

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
    
    def get_timesteps(
        self, 
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取时间步数据
        
        Args:
            intersection_id: 如果指定，只返回该路口的数据
            max_timesteps: 最大返回时间步数（用于快速测试）
        
        Returns:
            时间步数据列表
        """
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
        """
        获取phase-level数据对 (state, teacher_info)
        
        Args:
            intersection_id: 如果指定，只返回该路口的数据
            max_timesteps: 最大返回时间步数
        
        Returns:
            [(state, teacher_info), ...] 列表
            state: phase的state特征
            teacher_info: {Q_value, logit, selected, phase_id}
        """
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
        """
        获取用于计算动作一致率的数据
        
        Args:
            intersection_id: 如果指定，只返回该路口的数据
            max_timesteps: 最大返回时间步数
            include_teacher_q_values: 是否包含教师的Q值用于软目标蒸馏（默认True）
        
        Returns:
            [(state_dict, teacher_action, teacher_q_values), ...] 列表
            state_dict: 包含所有phase的state的字典 {phase_id: state}
            teacher_action: teacher选择的phase_id
            teacher_q_values: 教师对每个phase的Q值 (按phase_id排序)，若不存在则为None
        """
        timesteps = self.get_timesteps(intersection_id, max_timesteps)
        action_data = []
        
        for timestep in timesteps:
            phases = timestep.get('phases', [])
            if not phases:
                continue
            
            # 构建state字典和Q值字典（按phase_id索引）
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
                
                # 收集教师的Q值（用于软目标蒸馏）
                if include_teacher_q_values:
                    q_value = phase.get('Q_value')
                    teacher_q_dict[phase_id] = q_value if q_value is not None else 0.0
                
                if phase.get('selected', False):
                    teacher_action = phase_id
            
            if teacher_action is not None:
                # 按phase_id排序后转为列表（与evaluator中sorted(state_dict.keys())顺序一致）
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
        """
        获取teacher模型的性能指标（从数据中提取）
        
        Args:
            intersection_id: 如果指定，只返回该路口的数据
        
        Returns:
            包含性能指标的字典
        """
        metrics = {
            'average_delay': None,
            'average_travel_time': None,
            'average_queue_length': None,
            'average_throughput': None
        }
        
        # 首先尝试从stats中提取（这是主要来源）
        if hasattr(self, 'stats') and self.stats:
            # 映射stats中的字段名
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
        
        # 如果stats中没有，尝试从timestep中提取（备用方法）
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
