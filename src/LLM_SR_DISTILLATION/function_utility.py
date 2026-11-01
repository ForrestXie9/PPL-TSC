"""
Function Utility Module
Execute dynamically generated Python utility functions
"""
import numpy as np
from typing import Dict, Any, Optional, Callable, Union
from .policy_process import PolicyProcess


class FunctionUtility:
    """
    Wrapper for dynamically generated Python utility functions.
    Executes a function code string as a callable utility function.
    """
    
    def __init__(
        self,
        function_code: str,
        feature_extractor=None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        action_mode: str = "phase_argmax",
    ):
        """
        Initialize function utility from code string.
        
        Args:
            function_code: Complete Python function code (def compute_utility(...): ...)
            feature_extractor: Optional FeatureExtractor instance (for backward compatibility)
            feature_stats: Optional feature statistics for normalization
        """
        self.function_code = function_code.strip()
        self.feature_extractor = feature_extractor
        self.feature_stats = feature_stats or {}
        self.action_mode = str(action_mode or "phase_argmax")
        if self.action_mode not in {"phase_argmax", "keep_switch", "binary_keep_switch"}:
            raise ValueError(f"Unsupported action mode: {self.action_mode}")
        
        # Compile and load the function
        self._compiled_function = None
        self._load_function()
    
    def _load_function(self):
        """Compile and execute generated code in a bounded isolated interpreter."""
        try:
            self._compiled_function = PolicyProcess(self.function_code)
        except Exception as e:
            raise ValueError(f"Failed to compile function code: {e}")

    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """
        Get parameter bounds for ConstraintValidator compatibility.
        FunctionUtility has no tuneable parameters (logic is in the code).
        Returns empty dict.
        """
        return {}

    def compute(
        self,
        state: Dict[str, Any],
        phase_id: Optional[int] = None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        raise_on_error: bool = False,
    ) -> float:
        """
        Compute utility value for a given state and phase.
        
        Args:
            state: Raw state dictionary for the phase (or flat features dict for ConstraintValidator)
            phase_id: Phase ID being evaluated. If a dict is passed (validator interface), treated as 0.
            feature_stats: Optional feature statistics (overrides self.feature_stats)
            raise_on_error: If True, propagate exceptions (TypeError, KeyError, etc.) instead of
                returning 0.0. Use True for constraint validation so real errors are not masked.
        
        Returns:
            Utility Q value
        """
        if self._compiled_function is None:
            raise ValueError("Function not properly compiled")
        # ConstraintValidator compatibility: compute(features_dict, parameters_dict)
        if isinstance(phase_id, dict):
            phase_id = 0
        if phase_id is None:
            phase_id = 0
        # Use provided feature_stats or fall back to instance default
        stats = feature_stats if feature_stats is not None else self.feature_stats
        
        try:
            if (self.feature_extractor is not None
                    and self.feature_extractor.enable_normalization
                    and isinstance(state, dict)
                    and "features_by_direction" in state):
                state = dict(state)
                state["phase_features"] = self.feature_extractor.extract(state, phase_id)
            # Call the compiled function
            Q = self._compiled_function(state, phase_id=phase_id, feature_stats=stats)
            
            # A missing return is an invalid policy, including during review.
            if Q is None:
                raise ValueError("compute_utility returned None; expected a finite scalar")
            
            # Reject complex (should be caught at syntax stage; fallback if edge case)
            if isinstance(Q, (complex, np.complexfloating)):
                raise ValueError(f"Function returned complex (reject at syntax: sqrt/negative**frac)")
            if not isinstance(Q, (int, float, np.integer, np.floating)):
                try:
                    Q = float(Q)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Function returned non-numeric value: {type(Q).__name__} = {Q}")
            
            Q = float(Q)
            
            # Check for invalid values
            if np.isnan(Q) or np.isinf(Q):
                raise ValueError(f"Function returned invalid value: {Q}")
            
            return Q
            
        except ValueError:
            # Re-raise ValueError as-is (these are our validation errors)
            raise
        except (TypeError, AttributeError, KeyError) as e:
            # Common errors from bad LLM code: float(None), missing key, None.x
            if raise_on_error:
                raise
            # Return 0.0 so loss computation continues (individual gets bad score for this phase)
            return 0.0
        except Exception as e:
            if raise_on_error:
                raise
            # Return 0.0 so evaluation continues; individual gets bad score
            return 0.0
    
    def compute_all_phases(
        self,
        state_dict: Dict[int, Dict[str, Any]],
        feature_extractor: Optional[Callable] = None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        raise_on_error: bool = False,
    ) -> Dict[int, float]:
        """
        Compute utility for all phases. Optionally add per-phase bias (e.g. per-intersection tuning).
        
        Args:
            state_dict: Dictionary mapping phase_id to state
            feature_extractor: Optional feature extractor (for backward compatibility)
            feature_stats: Optional feature statistics
            phase_bias: Optional dict phase_id -> bias; Q_adjusted[phase] = Q_raw[phase] + phase_bias.get(phase, 0.0)
        
        Returns:
            Dictionary mapping phase_id to Q value
        """
        q_values = {}
        
        for phase_id, state in state_dict.items():
            try:
                q_value = self.compute(
                    state, phase_id=phase_id, feature_stats=feature_stats,
                    raise_on_error=raise_on_error
                )
                bias = (phase_bias.get(phase_id, 0.0) if phase_bias else 0.0)
                q_values[phase_id] = q_value + bias
            except Exception:
                if raise_on_error:
                    raise
                q_values[phase_id] = float('-inf')
        
        return q_values

    @staticmethod
    def _sorted_phase_ids(state_dict: Dict[int, Dict[str, Any]]) -> list:
        try:
            return sorted(state_dict.keys())
        except Exception:
            return list(state_dict.keys())

    def _select_action_keep_switch(
        self,
        state_dict: Dict[int, Dict[str, Any]],
        feature_extractor: Optional[Callable] = None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        raise_on_error: bool = False,
    ) -> int:
        """
        EcoLight-style binary decision:
        - keep: stay on current_phase
        - switch: move to the cyclic next phase

        The returned value is still a phase_id so the rest of the pipeline can
        continue to work unchanged.
        """
        if not state_dict:
            return 0

        phase_ids = self._sorted_phase_ids(state_dict)
        sample_state = state_dict[phase_ids[0]]
        current_phase = sample_state.get("current_phase", phase_ids[0])
        if current_phase not in state_dict:
            current_phase = phase_ids[0]

        try:
            current_pos = phase_ids.index(current_phase)
        except ValueError:
            current_pos = 0
            current_phase = phase_ids[0]
        next_phase = phase_ids[(current_pos + 1) % len(phase_ids)]

        keep_q = self.compute(
            state_dict[current_phase],
            phase_id=current_phase,
            feature_stats=feature_stats,
            raise_on_error=raise_on_error,
        )
        switch_q = self.compute(
            state_dict[next_phase],
            phase_id=next_phase,
            feature_stats=feature_stats,
            raise_on_error=raise_on_error,
        )

        if phase_bias:
            keep_q += float(phase_bias.get(current_phase, 0.0))
            switch_q += float(phase_bias.get(next_phase, 0.0))

        return next_phase if switch_q > keep_q else current_phase

    @staticmethod
    def _native_binary_context(
        state_dict: Dict[int, Dict[str, Any]],
    ) -> tuple:
        """Build the EcoLight-native binary state and its reachable phases.

        EcoLight predicts two action values from one global state:
        action 0 keeps the current phase and action 1 advances to the cyclic
        successor.  The aligned symbolic controller therefore receives the
        same global green/red vehicle counts and evaluates binary action IDs,
        rather than assigning independent utilities to traffic phases.
        """
        if not state_dict:
            empty_state = {
                "current_phase": 0,
                "next_phase": 0,
                "binary_features": {
                    "green_vehicle": 0.0,
                    "red_vehicle": 0.0,
                },
            }
            return empty_state, 0, 0

        phase_ids = FunctionUtility._sorted_phase_ids(state_dict)
        sample_state = state_dict[phase_ids[0]]
        current_phase = sample_state.get("current_phase", phase_ids[0])
        if current_phase not in state_dict:
            current_phase = phase_ids[0]
        try:
            current_pos = phase_ids.index(current_phase)
        except ValueError:
            current_pos = 0
            current_phase = phase_ids[0]
        next_phase = phase_ids[(current_pos + 1) % len(phase_ids)]

        # New diagnosis rollouts and the online simulator provide the exact
        # two EcoLight inputs.  Retain a deterministic fallback so saved
        # legacy rollouts can still be inspected, but the controlled
        # experiment validates that exact_binary_features=True.
        exact = state_dict[current_phase].get("binary_features")
        exact_binary_features = isinstance(exact, dict)
        if exact_binary_features:
            green_vehicle = float(exact.get("green_vehicle", 0.0))
            red_vehicle = float(exact.get("red_vehicle", 0.0))
        else:
            merged_features: Dict[str, Dict[str, Any]] = {}
            for state in state_dict.values():
                for direction, features in (
                    state.get("features_by_direction") or {}
                ).items():
                    if direction not in merged_features and isinstance(
                        features, dict
                    ):
                        merged_features[direction] = features

            green_directions = set(
                state_dict[current_phase].get("controlled_directions") or []
            )
            green_vehicle = 0.0
            red_vehicle = 0.0
            for direction, features in merged_features.items():
                value = float(features.get("num_vehicle", 0.0))
                if direction in green_directions:
                    green_vehicle += value
                else:
                    red_vehicle += value

        binary_state = {
            "current_phase": int(current_phase),
            "next_phase": int(next_phase),
            "binary_features": {
                "green_vehicle": float(green_vehicle),
                "red_vehicle": float(red_vehicle),
            },
            "exact_binary_features": bool(exact_binary_features),
        }
        return binary_state, current_phase, next_phase

    def compute_binary_action_values(
        self,
        state_dict: Dict[int, Dict[str, Any]],
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        raise_on_error: bool = False,
    ) -> tuple:
        """Return ``(current_phase, next_phase, Q_keep, Q_switch)``."""
        binary_state, current_phase, next_phase = self._native_binary_context(
            state_dict
        )
        keep_q = self.compute(
            binary_state,
            phase_id=0,
            feature_stats=feature_stats,
            raise_on_error=raise_on_error,
        )
        switch_q = self.compute(
            binary_state,
            phase_id=1,
            feature_stats=feature_stats,
            raise_on_error=raise_on_error,
        )
        if phase_bias:
            keep_q += float(phase_bias.get(current_phase, 0.0))
            switch_q += float(phase_bias.get(next_phase, 0.0))
        return current_phase, next_phase, float(keep_q), float(switch_q)

    def _select_action_binary_keep_switch(
        self,
        state_dict: Dict[int, Dict[str, Any]],
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        raise_on_error: bool = False,
    ) -> int:
        current_phase, next_phase, keep_q, switch_q = (
            self.compute_binary_action_values(
                state_dict,
                feature_stats=feature_stats,
                phase_bias=phase_bias,
                raise_on_error=raise_on_error,
            )
        )
        return next_phase if switch_q > keep_q else current_phase
    
    def select_action(
        self,
        state_dict: Dict[int, Dict[str, Any]],
        feature_extractor: Optional[Callable] = None,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        raise_on_error: bool = False,
    ) -> int:
        """
        Select action (phase) with highest utility. Optionally use per-phase bias.
        
        Args:
            state_dict: Dictionary mapping phase_id to state
            feature_extractor: Optional (for backward compatibility)
            feature_stats: Optional feature statistics
            phase_bias: Optional per-phase bias (e.g. from per-intersection tuning)
        
        Returns:
            Selected phase_id
        """
        if self.action_mode == "keep_switch":
            return self._select_action_keep_switch(
                state_dict,
                feature_extractor=feature_extractor,
                feature_stats=feature_stats,
                phase_bias=phase_bias,
                raise_on_error=raise_on_error,
            )
        if self.action_mode == "binary_keep_switch":
            return self._select_action_binary_keep_switch(
                state_dict,
                feature_stats=feature_stats,
                phase_bias=phase_bias,
                raise_on_error=raise_on_error,
            )

        q_values = self.compute_all_phases(
            state_dict, feature_extractor, feature_stats, phase_bias=phase_bias,
            raise_on_error=raise_on_error
        )
        
        if not q_values:
            return 0
        
        # Select phase with highest Q value
        best_phase = max(q_values.items(), key=lambda x: x[1])
        return best_phase[0]
    
    def __str__(self):
        return f"FunctionUtility(code_length={len(self.function_code)}, action_mode={self.action_mode})"
    
    def __repr__(self):
        return self.__str__()
