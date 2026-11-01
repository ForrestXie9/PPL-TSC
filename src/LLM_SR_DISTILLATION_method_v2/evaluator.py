"""
Evaluator Module
Evaluate utility functions based on behavioral fidelity and control performance
"""
import os
import hashlib
import json
import numpy as np
import time
from typing import Dict, Callable, Optional, Any

from LLM_SR_DISTILLATION.function_utility import FunctionUtility
from LLM_SR_DISTILLATION.feature_extractor import FeatureExtractor
from LLM_SR_DISTILLATION.simulator import StudentSimulator, SimulationResult


class Evaluator:
    """Evaluate a student utility function after review/calibration."""
    
    def __init__(
        self,
        data_loader,
        teacher_travel_time: float,
        feature_extractor: FeatureExtractor,
        constraint_validator=None,
        simulator: Optional[StudentSimulator] = None,
        verbose: bool = False,
        epsilon: float = 1e-6,
        action_weight: float = 1.0,
        performance_weight: float = 1.0,
        random_seed: int = 0,
        native_liveness_guard: bool = False,
    ):
        """
        Initialize evaluator
        
        Args:
            data_loader: TeacherDataLoader instance
            teacher_travel_time: Teacher model average travel time (seconds)
            feature_extractor: FeatureExtractor instance
            constraint_validator: ConstraintValidator instance (optional)
            simulator: StudentSimulator instance (optional)
            epsilon: Small constant to avoid numerical instability
        """
        self.data_loader = data_loader
        self.teacher_travel_time = teacher_travel_time
        self.feature_extractor = feature_extractor
        self.constraint_validator = constraint_validator
        self.simulator = simulator
        self.verbose = verbose
        self.epsilon = epsilon
        self.action_weight = float(action_weight)
        self.performance_weight = float(performance_weight)
        self.random_seed = int(random_seed)
        self.native_liveness_guard = bool(native_liveness_guard)
        self._last_policy_decisions = 0
        self._last_policy_switches = 0
        if not np.isfinite(teacher_travel_time) or teacher_travel_time <= 0:
            raise ValueError("Teacher ATT must be finite and positive")
        if (not all(np.isfinite(w) and w >= 0 for w in (self.action_weight, self.performance_weight))
                or self.action_weight + self.performance_weight <= 0):
            raise ValueError("Objective weights must be finite, nonnegative, and not both zero")
        self._runtime_checked_codes: set[str] = set()

    def assert_utility_runtime(
        self,
        utility_function: FunctionUtility,
        *,
        max_seconds_per_action: float = 0.05,
        sample_count: int = 4,
    ) -> None:
        """Check sustained per-state latency after warm-up; retain the 50ms limit.

        A single pipe/scheduler stall is not a stable policy runtime estimate.
        Three measured calls per state give a median; a slow state cannot hide
        behind fast states. Isolated-worker hard timeouts remain in force.
        """
        code = str(getattr(utility_function, "function_code", ""))
        if code in self._runtime_checked_codes:
            return
        intersection_ids = sorted(
            getattr(self.data_loader, "intersection_ids", [])
        )
        if not intersection_ids:
            raise ValueError('Runtime check requires source intersections')
        action_data = self.data_loader.get_action_consistency_data(
            intersection_ids[0],
            max_timesteps=max(1, int(sample_count)),
        )
        feature_stats = (
            self.feature_extractor.feature_stats
            if hasattr(self.feature_extractor, "feature_stats")
            else None
        )
        if not action_data:
            raise ValueError('Runtime check requires source decision states')
        if not np.isfinite(max_seconds_per_action) or max_seconds_per_action <= 0:
            raise ValueError('Runtime threshold must be finite and positive')
        utility_function.select_action(
            action_data[0][0], feature_extractor=None, feature_stats=feature_stats,
            phase_bias=None, raise_on_error=True,
        )
        checked = 0
        self._last_runtime_check = dict(threshold_seconds=float(max_seconds_per_action),
            warmup_calls=1, measured_calls_per_state=3, statistic='per_state_median', states=[])
        for sample_tuple in action_data[: max(1, int(sample_count))]:
            state_dict = sample_tuple[0]
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                utility_function.select_action(
                    state_dict, feature_extractor=None, feature_stats=feature_stats,
                    phase_bias=None, raise_on_error=True,
                )
                timings.append(time.perf_counter() - started)
            elapsed = float(np.median(timings))
            self._last_runtime_check['states'].append(dict(
                source_index=checked, seconds=timings, median_seconds=elapsed))
            checked += 1
            if elapsed > float(max_seconds_per_action):
                raise RuntimeError(
                    "Generated utility is too slow for real-time control: "
                    f"{elapsed:.3f}s median for one source-state action exceeds "
                    f"{float(max_seconds_per_action):.3f}s"
                )
        if checked:
            self._runtime_checked_codes.add(code)
    
    def _compute_performance_score(self, delta: float) -> float:
        """
        Control performance score P(U) = exp(min(Delta, 0)), capped at 1.

        Nonnegative relative ATT improvement receives the capped score 1.
        Positive teacher ATT and nonnegative student ATT imply Delta <= 1.
        """
        delta = float(delta)
        if not np.isfinite(delta) or delta > 1.0:
            raise ValueError("Relative ATT improvement must be finite and <= 1")
        return float(np.exp(min(delta, 0.0)))
    
    def _is_intersection_specific_params(self, parameters: Dict) -> bool:
        """Check whether parameters are stored per-intersection."""
        return isinstance(parameters, dict) and len(parameters) > 0 and all(
            isinstance(v, dict) for v in parameters.values()
        )
    
    def _get_params_for_intersection(self, parameters: Dict, intersection_id: int) -> Dict[str, float]:
        """Fetch parameters for a specific intersection (fallback to first available)."""
        if self._is_intersection_specific_params(parameters):
            if intersection_id in parameters:
                return parameters[intersection_id]
            if str(intersection_id) in parameters:
                return parameters[str(intersection_id)]
            # Fallback to any available set
            return next(iter(parameters.values()))
        return parameters
    
    def _compute_q_value(
        self,
        utility_function: FunctionUtility,
        state: Dict[str, Any],
        phase_id: int,
        parameters: Dict[str, float],
        feature_extractor: Optional[Callable] = None
    ) -> float:
        """
        Compute Q value for FunctionUtility (unified single function type).
        
        Args:
            utility_function: FunctionUtility instance
            state: Raw state dictionary for the phase
            phase_id: Phase ID
            parameters: Parameter dictionary (not used for FunctionUtility)
            feature_extractor: Not used (kept for backward compatibility)
        
        Returns:
            Q value
        
        Raises:
            ValueError: If function execution fails or returns invalid value
        """
        try:
            # FunctionUtility accepts raw state directly
            return utility_function.compute(
                state,
                phase_id=phase_id,
                feature_stats=self.feature_extractor.feature_stats if hasattr(self.feature_extractor, 'feature_stats') else None,
                raise_on_error=True,
            )
        except ValueError as e:
            msg = str(e)
            raise ValueError(f"Q computation failed for phase {phase_id}: {e}")
        except Exception as e:
            # Wrap other exceptions
            raise ValueError(f"Unexpected error computing Q for phase {phase_id}: {type(e).__name__}: {e}")
    
    def evaluate_all_intersections(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        max_timesteps: Optional[int] = None,
        feature_extractor: Optional[Callable] = None,
        per_intersection_bias: Optional[Dict[int, Dict[int, float]]] = None,
        raise_on_error: bool = True,
        evaluate_constraints: bool = True,
        use_constraint_score: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a calibrated utility function on all intersections.

        Review/constraint diagnostics are retained for reporting, but the
        evaluation score is driven by:
        - Behavioral Fidelity F(U)
        - Control Performance P(U)
        
        Returns:
            Dictionary containing:
            - 'average': Dict with averaged metrics across all intersections
            - 'per_intersection': Dict mapping intersection_id to its metrics
        """
        if self.simulator is None:
            raise ValueError('Control-performance evaluation requires a simulator and measured student ATT')
        intersection_ids = sorted(self.data_loader.intersection_ids)
        
        if feature_extractor is None:
            feature_extractor = self.feature_extractor.extract
        
        # Step 1: Run simulation ONCE to get network-wide average travel time
        sim_metrics = self._compute_student_travel_time_simulation(
            utility_function, parameters, per_intersection_bias=per_intersection_bias,
            raise_on_error=raise_on_error
        )
        student_travel_time = float(sim_metrics.avg_travel_time)
        if not np.isfinite(student_travel_time) or student_travel_time <= 0:
            raise ValueError("Student ATT must be finite and positive")
        if (
            self.native_liveness_guard
            and getattr(utility_function, "action_mode", None) == "binary_keep_switch"
            and self._last_policy_decisions > 0
        ):
            switch_rate = self._last_policy_switches / self._last_policy_decisions
            att_ratio = student_travel_time / self.teacher_travel_time
            if switch_rate < 0.05 and att_ratio > 1.20:
                raise ValueError(
                    "Native closed-loop liveness check failed: "
                    f"switch_rate={switch_rate:.3f} < 0.05 and "
                    f"source_ATT/teacher_ATT={att_ratio:.3f} > 1.20"
                )
        if self.verbose:
            print(
                f"    Network average travel time: {sim_metrics.avg_travel_time:.4f} | "
                f"Teacher travel time: {self.teacher_travel_time:.4f}"
            )
        
        # Compute Delta and exponential control performance (capped at 1).
        delta = self._compute_relative_performance_factor(student_travel_time)
        performance_score = self._compute_performance_score(delta)

        # Behavioral fidelity F(U): agreement with teacher rollout actions.
        fidelity_score = self._compute_action_consistency(
            utility_function,
            parameters,
            intersection_id=None,
            max_timesteps=max_timesteps,
            feature_extractor=feature_extractor,
            per_intersection_bias=per_intersection_bias,
            raise_on_error=raise_on_error,
        )
        binary_q_diagnostics = None
        if (
            getattr(utility_function, "action_mode", None)
            == "binary_keep_switch"
        ):
            binary_q_diagnostics = self._compute_binary_q_diagnostics(
                utility_function,
                max_timesteps=max_timesteps,
                per_intersection_bias=per_intersection_bias,
                raise_on_error=raise_on_error,
            )
        
        # Step 2: Constraint status (evaluate explicitly when validator is available)
        constraint_result = None
        constraint_satisfied: Optional[bool] = None
        constraint_penalty: float = 0.0
        constraint_evaluated = False
        if evaluate_constraints and self.constraint_validator is not None:
            constraint_evaluated = True
            try:
                expr_for_check = getattr(utility_function, "function_code", "")
                constraint_result = self.constraint_validator.validate_hybrid(
                    expression_template=expr_for_check,
                    utility_function=utility_function,
                    verbose=False,
                )
                constraint_satisfied = bool(constraint_result.get("valid", False))
                constraint_penalty = 0.0 if constraint_satisfied else 1.0
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: constraint validation failed: {type(e).__name__}: {e}")
                constraint_result = {
                    "valid": False,
                    "message": f"constraint evaluation error: {type(e).__name__}: {e}",
                }
                constraint_satisfied = False
                constraint_penalty = 1.0
        
        per_intersection = {}

        per_inter_avg_tt = None
        if sim_metrics is not None:
            per_inter_avg_tt = sim_metrics.per_intersection_avg_travel_time
        
        for inter_id in intersection_ids:
            eval_params = self._get_params_for_intersection(parameters, inter_id)
            
            # Load data for this intersection
            action_data = self.data_loader.get_action_consistency_data(
                inter_id, max_timesteps
            )
            
            # Compute per-intersection behavioral fidelity for reporting.
            if action_data:
                try:
                    phase_bias = (per_intersection_bias.get(inter_id) if per_intersection_bias else None)
                    inter_fidelity_score = self._compute_action_consistency_with_data(
                        utility_function, eval_params, action_data, feature_extractor,
                        phase_bias=phase_bias,
                        raise_on_error=raise_on_error,
                    )
                    if not np.isfinite(inter_fidelity_score):
                        inter_fidelity_score = 0.0
                except Exception as e:
                    if self.verbose:
                        print(f"    Error computing behavioral fidelity for intersection {inter_id}: {e}")
                    inter_fidelity_score = 0.0
            else:
                inter_fidelity_score = 0.0
            
            # Store per-intersection metrics.
            # Per-intersection score: loss-based, 0~1 (higher = lower loss). Overall fitness remains function_label.
            inter_avg_tt = None
            if per_inter_avg_tt is not None:
                inter_avg_tt = per_inter_avg_tt.get(inter_id)
                if inter_avg_tt is None:
                    try:
                        inter_avg_tt = per_inter_avg_tt.get(int(inter_id))
                    except Exception:
                        inter_avg_tt = None

            per_inter_score = float(
                float(self.action_weight) * float(inter_fidelity_score)
                + float(self.performance_weight) * float(performance_score)
            )
            per_intersection[inter_id] = {
                'score': per_inter_score,
                'behavioral_fidelity': float(inter_fidelity_score),
                'fidelity_score': float(inter_fidelity_score),
                'student_travel_time': student_travel_time,
                'inter_avg_travel_time': inter_avg_tt,
                'teacher_travel_time': self.teacher_travel_time,
                'delta': delta,
                'delta_d': delta,
                'control_performance': float(performance_score),
                'performance_score': float(performance_score),
                'constraint_evaluated': constraint_evaluated,
                'constraint_satisfied': constraint_satisfied,
                'constraint_penalty': constraint_penalty,
                'constraint_result': constraint_result,
            }
        
        num_intersections = len(per_intersection)
        constraint_score = 1.0 if constraint_satisfied is True else (0.0 if constraint_satisfied is False else 1.0)
        base_score = (
            float(self.action_weight) * float(fidelity_score)
            + float(self.performance_weight) * float(performance_score)
        )
        function_label = float(constraint_score) + base_score
        avg_score = float(function_label if use_constraint_score else base_score)
        
        avg_metrics = {
            'score': avg_score,
            'score_mode': 'review_plus_eval' if use_constraint_score else 'eval_only',
            'base_score': float(base_score),
            'full_score': float(function_label),
            'evaluation_score': float(base_score),
            'student_travel_time': student_travel_time,
            'teacher_travel_time': self.teacher_travel_time,
            'delta': delta,
            'delta_d': delta,
            'behavioral_fidelity': float(fidelity_score),
            'fidelity_score': float(fidelity_score),
            'action_consistency': float(fidelity_score),
            'control_performance': float(performance_score),
            'performance_score': float(performance_score),
            'action_weight': float(self.action_weight),
            'performance_weight': float(self.performance_weight),
            'constraint_score': float(constraint_score),
            'function_label': float(function_label),
            'constraint_evaluated': constraint_evaluated,
            'constraint_penalty': constraint_penalty,
            'constraint_satisfied': constraint_satisfied,
            'sim_avg_travel_time': None if sim_metrics is None else sim_metrics.avg_travel_time,
            'sim_avg_queue_length': None if sim_metrics is None else sim_metrics.avg_queue_length,
            'sim_avg_throughput': None if sim_metrics is None else sim_metrics.avg_throughput,
            'per_intersection_avg_travel_time': None if sim_metrics is None else sim_metrics.per_intersection_avg_travel_time,
        }
        if sim_metrics is not None and os.environ.get("PPL_METHOD_PROFILE")=="full_100_256_v1":
            avg_metrics["source_rollout_feedback"] = self._source_rollout_feedback
        if binary_q_diagnostics is not None:
            avg_metrics["binary_q_diagnostics"] = binary_q_diagnostics
        if sim_metrics is not None and getattr(utility_function, "action_mode", None) == "binary_keep_switch":
            decisions = self._last_policy_decisions
            avg_metrics["on_policy_switching"] = {
                "decisions": decisions,
                "switches": self._last_policy_switches,
                "switch_rate": self._last_policy_switches / decisions if decisions else None,
            }
        
        return {
            'average': avg_metrics,
            'per_intersection': per_intersection,
            'num_intersections': num_intersections
        }
    
    def _compute_student_travel_time_simulation(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        per_intersection_bias: Optional[Dict[int, Dict[int, float]]] = None,
        raise_on_error: bool = True,
    ) -> SimulationResult:
        """
        Run one simulation episode and return average travel time for the student policy.
        """

        assert self.simulator is not None
        self._last_policy_decisions = 0
        self._last_policy_switches = 0
        from source_rollout import SourceRolloutRecorder
        recorder = SourceRolloutRecorder(self.simulator.env_config.get("action_interval", 20)) if os.environ.get("PPL_METHOD_PROFILE")=="full_100_256_v1" else None

        def _policy(env, inter_idx: int) -> int:
            state_dict = self.simulator.build_phase_state_dict(env, inter_idx)
            phase_bias = None
            if per_intersection_bias is not None:
                phase_bias = per_intersection_bias.get(inter_idx)
                if phase_bias is None:
                    phase_bias = per_intersection_bias.get(str(inter_idx))
            selected = utility_function.select_action(
                state_dict, 
                feature_extractor=None,
                feature_stats=self.feature_extractor.feature_stats if hasattr(self.feature_extractor, 'feature_stats') else None,
                phase_bias=phase_bias,
                raise_on_error=raise_on_error,
            )
            if getattr(utility_function, "action_mode", None) == "binary_keep_switch":
                current = int(next(iter(state_dict.values()))["current_phase"])
                self._last_policy_decisions += 1
                self._last_policy_switches += int(selected != current)
            if recorder is not None:
                recorder.observe(inter_idx, state_dict, selected)
            return selected

        result = self.simulator.run_episode(_policy)
        if recorder is not None:
            self._source_rollout_feedback = recorder.finish(self, utility_function, per_intersection_bias)
        return result
    
    def _compute_action_consistency(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        feature_extractor: Optional[Callable] = None,
        per_intersection_bias: Optional[Dict[int, Dict[int, float]]] = None,
        raise_on_error: bool = True,
    ) -> float:
        """
        Compute action consistency rate (0 to 1). Optionally use per-intersection phase bias.
        
        Args:
            utility_function: FunctionUtility instance
            parameters: Parameter dictionary
            intersection_id: If specified, only evaluate this intersection
            max_timesteps: Maximum number of timesteps
            feature_extractor: Function to extract features from state
            per_intersection_bias: Optional {inter_id: {phase_id: bias}} for Q_adjusted = Q_raw + bias
        
        Returns:
            Action consistency rate (0 to 1)
        """
        if feature_extractor is None:
            feature_extractor = self.feature_extractor.extract
        intersection_ids = (
            [intersection_id] if intersection_id is not None
            else sorted(self.data_loader.intersection_ids)
        )
        total_actions = 0
        weighted_correct = 0.0
        for inter_id in intersection_ids:
            action_data = self.data_loader.get_action_consistency_data(
                inter_id, max_timesteps
            )
            if not action_data:
                continue
            phase_bias = per_intersection_bias.get(inter_id) if per_intersection_bias else None
            ratio = self._compute_action_consistency_with_data(
                utility_function,
                self._get_params_for_intersection(parameters, inter_id),
                action_data,
                feature_extractor,
                phase_bias=phase_bias,
                raise_on_error=raise_on_error,
            )
            if not np.isfinite(ratio):
                if raise_on_error:
                    raise ValueError(f"Non-finite action fidelity at intersection {inter_id}")
                ratio = 0.0
            total_actions += len(action_data)
            weighted_correct += ratio * len(action_data)
        return weighted_correct / total_actions if total_actions else 0.0

    def _compute_action_consistency_with_data(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        action_data: list,
        feature_extractor: Optional[Callable] = None,
        phase_bias: Optional[Dict[int, float]] = None,
        debug_counts: bool = False,
        raise_on_error: bool = True,
    ) -> float:
        """
        Compute action consistency rate using pre-loaded data. Optionally use per-phase bias.
        
        Args:
            utility_function: FunctionUtility instance
            parameters: Parameter dictionary (ignored for FunctionUtility)
            action_data: Pre-loaded action consistency data (list of (state_dict, teacher_action[, teacher_q_values]))
            feature_extractor: Function to extract features from state (used for FunctionUtility only)
            phase_bias: Optional {phase_id: bias} for FunctionUtility (Q_adjusted = Q_raw + bias)
        
        Returns:
            Action consistency rate (0 to 1), or NaN if critical errors occur
        """
        if feature_extractor is None:
            feature_extractor = self.feature_extractor.extract
        
        if not action_data:
            return 0.0
        
        correct_actions = 0
        total_actions = 0
        error_count = 0  # Count numerical errors (NaN/Inf)
        
        matched_timesteps = []

        for sample_tuple in action_data:
            try:
                timestep_id = None
                if len(sample_tuple) == 4:
                    state_dict, teacher_action, _teacher_q_values, timestep_id = sample_tuple
                elif len(sample_tuple) == 3:
                    state_dict, teacher_action, _teacher_q_values = sample_tuple
                else:
                    state_dict, teacher_action = sample_tuple
                
                # Select action using utility function (FunctionUtility only)
                student_action = utility_function.select_action(
                    state_dict, feature_extractor=None, feature_stats=self.feature_extractor.feature_stats if hasattr(self.feature_extractor, 'feature_stats') else None,
                    phase_bias=phase_bias,
                    raise_on_error=raise_on_error,
                )
                
                # Check if actions match
                if student_action == teacher_action:
                    correct_actions += 1
                    if debug_counts and timestep_id is not None and len(matched_timesteps) < 50:
                        matched_timesteps.append(timestep_id)
                total_actions += 1
            except Exception as e:
                if raise_on_error:
                    raise
                # If evaluation fails, count as incorrect
                if "NaN" in str(e) or "Inf" in str(e) or "invalid value" in str(e):
                    error_count += 1
                total_actions += 1
        
        if total_actions == 0:
            return 0.0
        
        # If >20% of samples produced numerical errors, params are invalid
        if error_count > total_actions * 0.2:
            return np.nan  # Signal optimizer that these params are invalid

        if debug_counts and self.verbose:
            print(
                f"      Consistency computed over {total_actions} samples, "
                f"consistent_steps={correct_actions}, error_count={error_count}"
            )
            if matched_timesteps:
                joined = ", ".join([str(t) for t in matched_timesteps])
                print(f"      First {len(matched_timesteps)} matched timesteps: {joined}")

        return correct_actions / total_actions

    def _compute_binary_q_diagnostics(
        self,
        utility_function: FunctionUtility,
        max_timesteps: Optional[int] = None,
        per_intersection_bias: Optional[
            Dict[int, Dict[int, float]]
        ] = None,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Compare student and teacher binary margins without scale matching."""
        teacher_margins = []
        student_margins = []
        teacher_confidences = []
        agreements = []
        label_matches = []
        exact_feature_flags = []

        feature_stats = (
            self.feature_extractor.feature_stats
            if hasattr(self.feature_extractor, "feature_stats")
            else None
        )
        for inter_id in sorted(self.data_loader.intersection_ids):
            action_data = self.data_loader.get_action_consistency_data(
                inter_id,
                max_timesteps=max_timesteps,
            )
            phase_bias = None
            if per_intersection_bias:
                phase_bias = per_intersection_bias.get(inter_id)
                if phase_bias is None:
                    phase_bias = per_intersection_bias.get(str(inter_id))
            for sample in action_data:
                state_dict, teacher_action = sample[:2]
                teacher_q = sample[2] if len(sample) >= 3 else None
                if not (
                    isinstance(teacher_q, (list, tuple, np.ndarray))
                    and len(teacher_q) == 2
                ):
                    continue
                try:
                    teacher_keep = float(teacher_q[0])
                    teacher_switch = float(teacher_q[1])
                    current, next_phase, student_keep, student_switch = (
                        utility_function.compute_binary_action_values(
                            state_dict,
                            feature_stats=feature_stats,
                            phase_bias=phase_bias,
                            raise_on_error=raise_on_error,
                        )
                    )
                    binary_state, _, _ = (
                        utility_function._native_binary_context(state_dict)
                    )
                except Exception:
                    if raise_on_error:
                        raise
                    continue

                teacher_binary_action = (
                    1 if teacher_switch > teacher_keep else 0
                )
                teacher_phase_from_q = (
                    next_phase if teacher_binary_action else current
                )
                student_phase = (
                    next_phase
                    if student_switch > student_keep
                    else current
                )
                teacher_margin = teacher_switch - teacher_keep
                student_margin = student_switch - student_keep
                teacher_margins.append(float(teacher_margin))
                student_margins.append(float(student_margin))
                teacher_confidences.append(abs(float(teacher_margin)))
                agreements.append(
                    1.0 if student_phase == teacher_action else 0.0
                )
                label_matches.append(
                    1.0
                    if teacher_phase_from_q == teacher_action
                    else 0.0
                )
                exact_feature_flags.append(
                    1.0
                    if binary_state.get("exact_binary_features", False)
                    else 0.0
                )

        if not agreements:
            return {
                "sample_count": 0,
                "available": False,
            }

        confidences = np.asarray(teacher_confidences, dtype=float)
        agreement_array = np.asarray(agreements, dtype=float)
        weight_sum = float(np.sum(confidences))
        weighted_abf = (
            float(np.sum(confidences * agreement_array) / weight_sum)
            if weight_sum > 1e-12
            else float(np.mean(agreement_array))
        )
        teacher_array = np.asarray(teacher_margins, dtype=float)
        student_array = np.asarray(student_margins, dtype=float)
        if (
            teacher_array.size >= 2
            and float(np.std(teacher_array)) > 1e-12
            and float(np.std(student_array)) > 1e-12
        ):
            margin_correlation = float(
                np.corrcoef(teacher_array, student_array)[0, 1]
            )
        else:
            margin_correlation = None

        quantile_edges = np.quantile(confidences, [0.25, 0.5, 0.75])
        buckets = np.digitize(
            confidences,
            quantile_edges,
            right=True,
        )
        quartile_abf = {}
        for bucket in range(4):
            mask = buckets == bucket
            quartile_abf[f"q{bucket + 1}"] = (
                float(np.mean(agreement_array[mask]))
                if np.any(mask)
                else None
            )

        return {
            "available": True,
            "sample_count": int(len(agreements)),
            "teacher_q_label_consistency": float(
                np.mean(label_matches)
            ),
            "exact_binary_feature_rate": float(
                np.mean(exact_feature_flags)
            ),
            "hard_abf": float(np.mean(agreement_array)),
            "teacher_margin_weighted_abf": weighted_abf,
            "teacher_student_margin_pearson": margin_correlation,
            "abf_by_teacher_margin_quartile": quartile_abf,
        }

    def _build_cached_bias_consistency(
        self, utility_function: FunctionUtility, action_data: list,
    ) -> Callable[[Dict[int, float]], float]:
        """Cache finite utilities only; failed samples invalidate calibration.

        Bias cannot repair a broken utility evaluation. Never convert an error
        to an argmax tie or remove failed samples from the agreement denominator.
        """
        feature_stats = getattr(self.feature_extractor, "feature_stats", None)
        mode = getattr(utility_function, "action_mode", "phase_argmax") or "phase_argmax"
        if mode not in {"phase_argmax", "keep_switch", "binary_keep_switch"}:
            raise ValueError(f"Unsupported calibration action mode: {mode}")
        samples = []
        for sample in action_data:
            states, teacher_action = sample[:2]
            if not states:
                raise ValueError("Calibration sample has no legal phases")
            validated=getattr(utility_function,"_validated_source_utilities",None)
            if validated is not None:
                from ppl_tsc_current.source_execution import state_key
            if validated is not None and validated['feature_stats_key']==state_key(feature_stats):
                existing=validated['values'].get(state_key(states))
                if existing is not None and existing[0]==mode:
                    samples.append((mode,teacher_action,existing[1]))
                    continue
            if mode == "phase_argmax":
                values = utility_function.compute_all_phases(
                    states, feature_extractor=None, feature_stats=feature_stats,
                    phase_bias=None, raise_on_error=True,
                )
                if set(values) != set(states):
                    raise ValueError("Calibration utilities do not cover every legal phase")
                items = tuple((p, float(q)) for p, q in values.items())
                if not all(np.isfinite(q) for _, q in items):
                    raise ValueError("Nonfinite raw utility during calibration")
                samples.append((mode, teacher_action, items))
            else:
                if mode == "binary_keep_switch":
                    current, following, keep_q, switch_q = utility_function.compute_binary_action_values(
                        states, feature_stats=feature_stats, phase_bias=None, raise_on_error=True,
                    )
                else:
                    ids = utility_function._sorted_phase_ids(states)
                    current = states[ids[0]].get("current_phase", ids[0])
                    if current not in states:
                        current = ids[0]
                    following = ids[(ids.index(current) + 1) % len(ids)]
                    keep_q = utility_function.compute(states[current], phase_id=current,
                        feature_stats=feature_stats, raise_on_error=True)
                    switch_q = utility_function.compute(states[following], phase_id=following,
                        feature_stats=feature_stats, raise_on_error=True)
                if not np.isfinite(keep_q) or not np.isfinite(switch_q):
                    raise ValueError("Nonfinite binary utility during calibration")
                samples.append((mode, teacher_action,
                                (current, following, float(keep_q), float(switch_q))))

        def consistency(phase_bias: Dict[int, float]) -> float:
            if not all(np.isfinite(float(b)) for b in phase_bias.values()):
                raise ValueError("Nonfinite calibration bias")
            correct = 0
            for mode, teacher_action, values in samples:
                if mode == "phase_argmax":
                    adjusted = [(p, q + float(phase_bias.get(p, 0.0))) for p, q in values]
                    if not all(np.isfinite(q) for _, q in adjusted):
                        raise ValueError("Nonfinite adjusted calibration utility")
                    action = max(adjusted, key=lambda item: item[1])[0]
                else:
                    current, following, keep_q, switch_q = values
                    keep_q += float(phase_bias.get(current, 0.0))
                    switch_q += float(phase_bias.get(following, 0.0))
                    if not np.isfinite(keep_q) or not np.isfinite(switch_q):
                        raise ValueError("Nonfinite adjusted binary utility")
                    action = following if switch_q > keep_q else current
                correct += int(action == teacher_action)
            return correct / len(samples) if samples else 0.0
        consistency.samples = samples
        return consistency
    
    def _compute_relative_performance_factor(self, student_travel_time: float) -> float:
        """
        Compute Delta = (TT_teacher - TT_student) / (|TT_teacher| + epsilon).
        
        Args:
            student_travel_time: Student model average travel time (seconds)
        
        Returns:
            Relative performance improvement Delta
        """
        denominator = abs(self.teacher_travel_time) + self.epsilon
        delta = (self.teacher_travel_time - student_travel_time) / denominator
        return delta
    
    def fit_per_intersection_bias(
        self,
        utility_function: "FunctionUtility",
        max_timesteps: Optional[int] = None,
        max_iter: int = 20,
        bias_bound: float = 1.0,
        verbose: bool = True,
    ) -> Dict[int, Dict[int, float]]:
        """
        Calibration stage: fit per-intersection, per-phase additive bias to maximize behavioral fidelity.
        
        For each intersection, optimize independent bias values for each traffic signal phase.
        The shared utility function remains unchanged; only per-phase biases are tuned.
        Final Q-value: Q_adjusted = Q_raw(from utility function) + bias[phase_id]
        
        Optimization objective: maximize behavioral fidelity (fraction of correct argmax matches)
        This directly optimizes which action is selected, not just probability distributions.
        
        Args:
            utility_function: FunctionUtility instance (shared compute_utility function)
            max_timesteps: Max timesteps for action data (None = use all)
            max_iter: Maximum CMA-ES iterations per intersection
            bias_bound: Bound for each bias value (e.g. 5.0 -> bias in [-5, 5])
            verbose: Whether to print progress
        
        Returns:
            per_intersection_bias: {inter_id: {phase_id: bias_float}}
                - Outer dict: one entry per intersection
                - Inner dict: one bias value per phase (typically 4 phases: 0, 1, 2, 3)
        """
        if not np.isfinite(bias_bound) or bias_bound <= 0:
            raise ValueError("bias_bound must be finite and positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        import cma  # type: ignore
        
        # Require a controller interface that can apply phase_bias during action selection.
        if not callable(getattr(utility_function, "select_action", None)):
            if verbose:
                print("    fit_per_intersection_bias: controller has no select_action, returning empty bias")
            return {}
        self.assert_utility_runtime(utility_function)
        feature_extractor = self.feature_extractor.extract if hasattr(self.feature_extractor, "extract") else None
        per_intersection_bias: Dict[int, Dict[int, float]] = {}

        intersection_ids = sorted(self.data_loader.intersection_ids)
        verbose_samples = False
        self.calibration_details = {}

        for inter_id in intersection_ids:
            action_data = self.data_loader.get_action_consistency_data(
                inter_id, max_timesteps=max_timesteps, include_timestep=True
            )
            if not action_data:
                per_intersection_bias[inter_id] = {}
                if verbose:
                    print(f"    Intersection {inter_id}: no action_data available, skipping")
                continue
            
            # Phase IDs from first sample
            state_dict_0 = action_data[0][0]
            phase_ids = sorted(state_dict_0.keys())
            n_phases = len(phase_ids)
            if n_phases == 0:
                per_intersection_bias[inter_id] = {}
                if verbose:
                    print(f"    Intersection {inter_id}: no phases in state_dict, skipping")
                continue
            
            if verbose:
                print(f"    Intersection {inter_id}: {len(action_data)} samples, {n_phases} phases")
                try:
                    unmodified_consistency = self._compute_action_consistency_with_data(
                        utility_function, {}, action_data, feature_extractor,
                        phase_bias=None, debug_counts=False, raise_on_error=False,
                    )
                    print(f"      Unmodified consistency: {unmodified_consistency:.4f}")
                except Exception as e:
                    print(f"      Unmodified consistency: failed ({type(e).__name__}: {e})")

            if verbose and verbose_samples and action_data and hasattr(utility_function, "compute_all_phases") and callable(utility_function.compute_all_phases):
                    state_dict, teacher_action = action_data[0][:2]
                    q_values = utility_function.compute_all_phases(state_dict, feature_extractor)
                    student_action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else 0
                    q_display = ', '.join([f'{pid}:{q:.3f}' for pid, q in sorted(q_values.items())])
                    print(f"      All Q values: {{{q_display}}}")
                    print(f"      Sample: teacher={teacher_action}, student={student_action}, match={teacher_action==student_action}")
                    
                    # Show actual state structure to diagnose function issues
                    print(f"      State structure for phase 0:")
                    phase0_state = state_dict.get(0, {})
                    if isinstance(phase0_state, dict):
                        for key in list(phase0_state.keys())[:5]:  # First 5 keys
                            val = phase0_state[key]
                            if isinstance(val, dict):
                                print(f"        {key}: {{...}} (dict with {len(val)} items)")
                            else:
                                print(f"        {key}: {val} ({type(val).__name__})")
                    print(f"      Current phase: {state_dict.get(0, {}).get('current_phase', '?')}")

            cached_bias_consistency = self._build_cached_bias_consistency(
                utility_function,
                action_data,
            )
            
            def _consistency_from_x(x: np.ndarray) -> float:
                phase_bias = dict(zip(phase_ids, x.tolist()))
                try:
                    consistency = cached_bias_consistency(phase_bias)
                    if np.isfinite(consistency):
                        return float(consistency)
                except Exception:
                    pass
                return 0.0

            evaluated = []

            def objective(x: np.ndarray) -> float:
                phase_bias = dict(zip(phase_ids, x.tolist()))
                try:
                    # Optimize action_consistency directly (negative because minimize)
                    # This optimizes argmax matching, not just probability distribution
                    consistency = cached_bias_consistency(phase_bias)
                    # Return negative (scipy minimizes, we want to maximize consistency)
                    obj_value = -float(consistency) if np.isfinite(consistency) else 1e10
                    evaluated.append((np.asarray(x, dtype=float).copy(), obj_value))
                    return obj_value
                except Exception as e:
                    if verbose:
                        print(f"      objective error: {e}")
                    return 1e10
            
            x0 = np.zeros(n_phases)
            bounds = [(-bias_bound, bias_bound)] * n_phases
            try:
                # Evaluate objective at initial point
                initial_obj = objective(x0)
                initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
                if verbose:
                    print(f"      Initial consistency: {initial_consistency:.4f}")
                popsize = 20

                # CMA-ES optimization (more stable for non-smooth objectives)
                x0 = np.zeros(n_phases, dtype=float)
                sigma0 = max(1e-6, bias_bound * float(os.environ.get("PPL_CMA_SIGMA_RATIO", str(1.0/3.0))))
                cma_opts = {
                    "bounds": [[-bias_bound] * n_phases, [bias_bound] * n_phases],
                    "popsize": popsize,
                    "maxiter": max_iter,
                    "verbose": -9,
                    # CMA-ES otherwise seeds itself from wall-clock entropy.
                    "seed": (self.random_seed + int(inter_id)) % 2147483646 + 1,
                }

                es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
                iter_state = {"i": 0, "last": None, "best": float("-inf")}

                while not es.stop() and iter_state["i"] < max_iter:
                    solutions = es.ask()
                    values = [objective(x) for x in solutions]
                    es.tell(solutions, values)

                    iter_state["i"] += 1
                    best_idx = int(np.argmin(values))
                    consistency = -float(values[best_idx]) if np.isfinite(values[best_idx]) else 0.0
                    consistent_steps = int(round(consistency * len(action_data)))
                    if consistency > iter_state["best"]:
                        iter_state["best"] = consistency

                    # Per-iteration output disabled for consistency (was only for first_intersection)
                    iter_state["last"] = consistency

                optimized_bias = es.result.xbest if es.result.xbest is not None else x0
                optimized_bias = np.clip(np.array(optimized_bias, dtype=float), -bias_bound, bias_bound).tolist()

                from calibration_ties import choose_tied_bias
                trace_digest = hashlib.sha256(json.dumps([(x.tolist(),v) for x,v in evaluated],allow_nan=False).encode()).hexdigest()
                if os.environ.get("PPL_CMA_TIE_RULE", "first") == "least_changes":
                    optimized_bias = choose_tied_bias(evaluated, cached_bias_consistency.samples, phase_ids).tolist()
                chosen_abf = cached_bias_consistency(dict(zip(phase_ids,optimized_bias)))
                best_observed_abf = max(-value for _,value in evaluated)
                self.calibration_details[str(inter_id)] = dict(population=20,maxiter=max_iter,
                    iterations=iter_state["i"],cma_evaluations=int(es.result.evaluations),
                    sigma0=sigma0,bound=bias_bound,trace_sha256=trace_digest,
                    best_observed_abf=best_observed_abf,chosen_abf=chosen_abf,
                    tie_rule=os.environ.get("PPL_CMA_TIE_RULE", "first"))

                class _ResShim:
                    def __init__(self, xbest, fbest, nfev, success):
                        self.x = xbest
                        self.fun = fbest
                        self.nfev = nfev
                        self.success = success

                stop_reasons = es.stop()
                success = not (stop_reasons and ("maxiter" in stop_reasons or "maxfevals" in stop_reasons))
                res = _ResShim(
                    xbest=np.array(optimized_bias, dtype=float),
                    fbest=-chosen_abf,
                    nfev=int(es.result.evaluations) if es.result.evaluations is not None else 0,
                    success=success,
                )

                # Extract optimized bias values (ALWAYS use res.x, not x0)
                optimized_bias = res.x.tolist()

                if verbose:
                    # Detailed diagnostic: show optimization progress
                    final_consistency = -res.fun if res.fun < 1e9 else 0.0
                    initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
                    total_actions = len(action_data)
                    initial_steps = int(round(initial_consistency * total_actions)) if total_actions > 0 else 0
                    final_steps = int(round(final_consistency * total_actions)) if total_actions > 0 else 0
                    bias_values = ", ".join([f"P{pid}:{val:+.3f}" for pid, val in zip(phase_ids, optimized_bias)])
                    nfev = getattr(res, 'nfev', '?')
                    print(f"    Intersection {inter_id}: "
                          f"consistency {initial_consistency:.4f}({initial_steps}/{total_actions})->"
                          f"{final_consistency:.4f}({final_steps}/{total_actions}), "
                          f"bias=[{bias_values}], success={res.success}, nfev={nfev}")

                # A tie provides no fidelity benefit: retain the original policy.
                final_consistency = -res.fun if res.fun < 1e9 else 0.0
                initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
                if final_consistency <= initial_consistency:
                    if verbose:
                        print(
                            f"    Intersection {inter_id}: consistency did not improve "
                            f"({initial_consistency:.4f}->{final_consistency:.4f}), revert bias to zeros"
                        )
                    per_intersection_bias[inter_id] = dict(zip(phase_ids, [0.0] * n_phases))
                else:
                    per_intersection_bias[inter_id] = dict(zip(phase_ids, optimized_bias))
            except Exception as e:
                raise RuntimeError(f"Calibration failed at {inter_id}: {e}") from e

        return per_intersection_bias

    # NOTE: The old default feature extractor was removed because feature variables must be
    # teacher-model-aware and consistent. Use FeatureExtractor.extract instead.
