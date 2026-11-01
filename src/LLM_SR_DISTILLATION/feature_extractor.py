"""
Feature Extractor Module
Build a teacher-model-aware mapping from structured phase-level state to feature variables (phase_active, pressure_norm, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sum_keys_from_features_by_direction(state: Dict[str, Any], keys: List[str]) -> float:
    total = 0.0
    fbd = state.get("features_by_direction") or {}
    if not isinstance(fbd, dict):
        return 0.0
    for _, dir_feats in fbd.items():
        if not isinstance(dir_feats, dict):
            continue
        for k in keys:
            v = _safe_float(dir_feats.get(k))
            if v is not None:
                total += v
                break
    return total


def _sum_keys_from_other_features(state: Dict[str, Any], keys: List[str]) -> float:
    other = state.get("other_features") or {}
    if not isinstance(other, dict):
        return 0.0
    for key in keys:
        v = other.get(key)
        vf = _safe_float(v)
        if vf is not None:
            return vf
        # If it's a list/array-like, sum numeric items.
        if isinstance(v, (list, tuple)):
            s = 0.0
            for item in v:
                it = _safe_float(item)
                if it is not None:
                    s += it
            return s
    return 0.0


@dataclass(frozen=True)
class FeatureSpec:
    """
    One feature specification.
    """

    name: str  # e.g., "phase_active" or "pressure_norm"
    kind: str  # e.g., "current_phase", "pressure_sum", ...
    source: str = "features_by_direction"  # "features_by_direction" or "other_features"
    keys: Optional[List[str]] = None  # candidate raw keys in the source dict
    normalized: bool = False  # whether this variable is the min-max normalized view


class FeatureExtractor:
    """
    Extract feature variables from the structured phase-level state produced by collect_teacher_data.py.
    Supports Min-Max normalization for stable feature ranges.
    """

    def __init__(
        self,
        specs: List[FeatureSpec],
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        normalize_features: bool = False,
        clip_normalized: bool = False,
    ):
        """
        Initialize feature extractor with optional normalization statistics.
        
        Args:
            specs: List of FeatureSpec objects
            feature_stats: Optional dict {feature_name: {'min','max','mean','std'}}
                          If provided, source features map to [0, 1]
            normalize_features: Whether to apply normalization in this extractor
        """
        self.specs = list(specs)
        self.feature_names = [s.name for s in self.specs]
        self.feature_stats = feature_stats or {}
        self.enable_normalization = normalize_features
        self.clip_normalized = clip_normalized

    def extract(self, state: Dict[str, Any], phase_id: int) -> Dict[str, float]:
        """
        Extract features from state and optionally normalize them.
        
        Args:
            state: State dictionary
            phase_id: Phase ID for current phase indicator
        
        Returns:
            Dictionary of feature values, normalized if stats are available
        """
        out: Dict[str, float] = {}
        for spec in self.specs:
            raw_value = self._extract_one(spec, state, phase_id)

            if spec.normalized:
                out[spec.name] = self._normalize_minmax(raw_value, spec.name)
            else:
                out[spec.name] = raw_value

        return out
    
    def _normalize_minmax(self, value: float, feature_name: str) -> float:
        """
        Apply Min-Max normalization: (x - min) / (max - min)
        
        Args:
            value: Raw feature value
            feature_name: Feature name to look up stats
        
        Returns:
            Source values map to [0, 1]; target values may exceed these bounds.
        """
        if not self.enable_normalization or feature_name not in self.feature_stats:
            return value
        
        stats = self.feature_stats[feature_name]
        min_val = stats.get('min', 0.0)
        max_val = stats.get('max', 1.0)
        
        # Handle edge case where min == max
        if max_val - min_val < 1e-8:
            return 0.5 if self.clip_normalized else 0.0  # Preserve archived v1 semantics.
        
        scaled_value = np.clip(value, min_val, max_val) if self.clip_normalized else value
        normalized = (scaled_value - min_val) / (max_val - min_val)
        
        return float(normalized)

    def _extract_one(self, spec: FeatureSpec, state: Dict[str, Any], phase_id: int) -> float:
        if spec.kind == "current_phase":
            # phase_active is now a binary indicator: 1 if phase_id is the currently executing phase, 0 otherwise
            current_executing_phase = _safe_float(state.get("current_phase"))
            if current_executing_phase is not None:
                return 1.0 if int(current_executing_phase) == phase_id else 0.0
            return 0.0

        # Prefer explicit key candidates if provided.
        keys = spec.keys or []
        if spec.source == "features_by_direction":
            if keys:
                return _sum_keys_from_features_by_direction(state, keys)
            # Backward-compatible defaults.
            if spec.kind == "pressure_sum":
                return _sum_keys_from_features_by_direction(state, ["pressure"])
            if spec.kind == "num_vehicle_sum":
                return _sum_keys_from_features_by_direction(state, ["num_vehicle"])
            if spec.kind == "num_waiting_vehicle_sum":
                return _sum_keys_from_features_by_direction(state, ["num_waiting_vehicle"])
            if spec.kind == "waiting_time_sum":
                return _sum_keys_from_features_by_direction(state, ["waiting_time", "sum_waiting_time"])
            return 0.0

        if spec.source == "other_features":
            if keys:
                return _sum_keys_from_other_features(state, keys)
            if spec.kind.startswith("other:"):
                k = spec.kind.split("other:", 1)[1]
                return _sum_keys_from_other_features(state, [k])
            return 0.0

        # Unknown kind -> 0.0
        return 0.0


MODEL_KEY_OVERRIDES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    # Values are candidate raw keys to try, in priority order.
    # The collector writes per-direction values into state["features_by_direction"][dir][<raw_key>].
    "MPLight": {
        "features_by_direction": {
            "pressure_sum": ["pressure"],
        }
    },
    "FRAP": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
        }
    },
    "MaxPressure": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
            "pressure_sum": ["pressure"],
        }
    },
    "SOTL": {
        "features_by_direction": {
            "num_waiting_vehicle_sum": ["num_waiting_vehicle", "waiting_vehicle"],
        }
    },
    "PiLight": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
            "num_waiting_vehicle_sum": ["num_waiting_vehicle", "waiting_vehicle"],
            "waiting_time_sum": ["waiting_time", "sum_waiting_time"],
            "pressure_sum": ["pressure"],
        }
    },
    # Fallback models: keep broad candidates.
    "CoLight": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
        }
    },
    "EcoLight": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
        }
    },
    "RandomPath": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
            "num_waiting_vehicle_sum": ["num_waiting_vehicle", "waiting_vehicle"],
            "waiting_time_sum": ["waiting_time", "sum_waiting_time"],
            "pressure_sum": ["pressure"],
        }
    },
    "TinyLight": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
            "num_waiting_vehicle_sum": ["num_waiting_vehicle", "waiting_vehicle"],
            "waiting_time_sum": ["waiting_time", "sum_waiting_time"],
            "pressure_sum": ["pressure"],
        }
    },
    "TinyLightQuan": {
        "features_by_direction": {
            "num_vehicle_sum": ["num_vehicle"],
            "num_waiting_vehicle_sum": ["num_waiting_vehicle", "waiting_vehicle"],
            "waiting_time_sum": ["waiting_time", "sum_waiting_time"],
            "pressure_sum": ["pressure"],
        }
    },
}


def build_feature_extractor(
    teacher_model: str, 
    observation_feature_list: List[str],
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
    normalize_features: bool = False,
    clip_normalized: bool = False,
) -> FeatureExtractor:
    """
    Build a feature extractor and a stable feature variable layout for a given teacher model.

    Design:
    - phase_active is always current phase if available.
    - Each "semantic group" in observation_feature_list becomes one aggregated feature, so the
      meaning is consistent across models (pressure/vehicle/waiting/etc.) while remaining
      compatible with the teacher's state.
    - Optional Min-Max normalization based on feature_stats

    Args:
        teacher_model: Teacher model name
        observation_feature_list: List of observation feature names
        feature_stats: Optional dict {feature_name: {'min','max','mean','std'}}
                  If provided, features can be normalized (Min-Max or Z-score)
        normalize_features: Whether to apply normalization in this extractor
    
    Returns:
        FeatureExtractor instance with optional normalization
    """

    specs: List[FeatureSpec] = []

    # Always include phase_active as current phase (it exists in structured_state for all models that include it).
    # If current phase is not meaningful for a teacher, the expression can simply ignore phase_active.
    specs.append(FeatureSpec(name="phase_active", kind="current_phase"))

    # Map observation feature names to semantic kinds.
    # We intentionally keep this minimal but extensible.
    semantic_kinds: List[str] = []
    feature_name_map = {}  # kind -> short_name
    
    for feat in observation_feature_list or []:
        f = (feat or "").lower()
        # current phase is already represented by phase_active
        if "current_phase" in f:
            continue
        if "pressure" in f:
            if "pressure_sum" not in semantic_kinds:
                semantic_kinds.append("pressure_sum")
                feature_name_map["pressure_sum"] = "pressure"
        elif "num_waiting_vehicle" in f:
            if "num_waiting_vehicle_sum" not in semantic_kinds:
                semantic_kinds.append("num_waiting_vehicle_sum")
                feature_name_map["num_waiting_vehicle_sum"] = "waiting_veh"
        elif "sum_waiting_time" in f or "waiting_time" in f:
            if "waiting_time_sum" not in semantic_kinds:
                semantic_kinds.append("waiting_time_sum")
                feature_name_map["waiting_time_sum"] = "waiting_time"
        elif "num_vehicle" in f:
            if "num_vehicle_sum" not in semantic_kinds:
                semantic_kinds.append("num_vehicle_sum")
                feature_name_map["num_vehicle_sum"] = "vehicles"
        else:
            # For any unsupported feature, keep a placeholder mapped from other_features by name.
            kind = f"other:{feat}"
            semantic_kinds.append(kind)
            # Use simplified name from original feature name
            feature_name_map[kind] = feat.replace("_", "")[:12]

    overrides = MODEL_KEY_OVERRIDES.get(teacher_model, {})
    fbd_overrides = overrides.get("features_by_direction", {})
    other_overrides = overrides.get("other_features", {})

    # Assign meaningful names in order of semantic kinds.
    for kind in semantic_kinds:
        base_name = feature_name_map.get(kind, f"feature{len(specs)}")
        if kind.startswith("other:"):
            raw_name = kind.split("other:", 1)[1]
            keys = other_overrides.get(kind, [raw_name])
            source = "other_features"
        else:
            keys = fbd_overrides.get(kind)
            source = "features_by_direction"

        key_list = list(keys) if keys else None
        specs.append(
            FeatureSpec(name=f"{base_name}_raw", kind=kind, source=source, keys=key_list, normalized=False)
        )
        specs.append(
            FeatureSpec(name=f"{base_name}_norm", kind=kind, source=source, keys=key_list, normalized=True)
        )

    # Ensure at least phase_active/pressure_raw/pressure_norm exists to avoid degenerate prompts.
    if len(specs) == 1:
        fallback_kind = "pressure_sum" if teacher_model == "MPLight" else "num_vehicle_sum"
        fallback_base = "pressure" if teacher_model == "MPLight" else "vehicles"
        keys = fbd_overrides.get(fallback_kind)
        key_list = list(keys) if keys else None
        specs.append(
            FeatureSpec(name=f"{fallback_base}_raw", kind=fallback_kind, source="features_by_direction", keys=key_list, normalized=False)
        )
        specs.append(
            FeatureSpec(name=f"{fallback_base}_norm", kind=fallback_kind, source="features_by_direction", keys=key_list, normalized=True)
        )

    return FeatureExtractor(specs, feature_stats=feature_stats,
                            normalize_features=normalize_features,
                            clip_normalized=clip_normalized)
