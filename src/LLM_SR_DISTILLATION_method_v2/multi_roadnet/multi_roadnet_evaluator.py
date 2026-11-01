"""
Multi-Roadnet Evaluator

Wraps multiple single-roadnet Evaluators; aggregates action_consistency, student_travel_time,
and constraint_score across all road networks to produce a unified fitness for joint distillation.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator
from LLM_SR_DISTILLATION_method_v2.function_utility import FunctionUtility


def sigmoid(x: float, beta: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-beta * np.clip(x, -100, 100)))


class MultiRoadnetEvaluator:
    """
    Evaluator for one-teacher multi-roadnet joint distillation.

    Holds N (Evaluator, scenario_id, teacher_travel_time) tuples.
    Aggregates metrics across all road networks:
    - action_consistency: weighted average by sample count (or simple mean)
    - student_travel_time: mean of per-roadnet sim results
    - teacher_travel_time: mean of per-roadnet teacher ATT
    - per_intersection_bias: {(scenario_id, inter_id): {phase_id: bias}}
    """

    def __init__(
        self,
        roadnet_evaluators: List[Tuple[Evaluator, str, float]],
        verbose: bool = False,
        epsilon: float = 1e-6,
        action_weight: float = 1.0,
        performance_weight: float = 1.0,
    ):
        """
        Args:
            roadnet_evaluators: [(Evaluator, scenario_id, teacher_travel_time), ...]
            verbose: Whether to print progress
            epsilon: Small constant for delta_d denominator
        """
        self.roadnet_evaluators = roadnet_evaluators
        self.verbose = verbose
        self.epsilon = epsilon
        # Primary evaluator (first) for feature_extractor, constraint_validator
        self.primary_evaluator = roadnet_evaluators[0][0]
        self.feature_extractor = self.primary_evaluator.feature_extractor
        self.constraint_validator = self.primary_evaluator.constraint_validator
        self.data_loader = self.primary_evaluator.data_loader  # For constraint validation
        self.action_weight = float(action_weight)
        self.performance_weight = float(performance_weight)

    def _extract_bias_for_roadnet(
        self,
        per_intersection_bias: Optional[Dict],
        scenario_id: str,
        intersection_ids,
    ) -> Optional[Dict[int, Dict[int, float]]]:
        """Convert multi-roadnet bias format to single-roadnet format."""
        if per_intersection_bias is None:
            return None
        result = {}
        for inter_id in intersection_ids:
            for key in [(scenario_id, inter_id), f"{scenario_id}_{inter_id}"]:
                if key in per_intersection_bias:
                    result[inter_id] = per_intersection_bias[key]
                    break
        return result if result else None

    def evaluate_all_intersections(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        max_timesteps: Optional[int] = None,
        feature_extractor: Optional[Any] = None,
        per_intersection_bias: Optional[Dict] = None,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate utility function across all road networks and aggregate metrics.

        per_intersection_bias format: {f"{scenario_id}_{inter_id}": {phase_id: bias}}
        or {(scenario_id, inter_id): {phase_id: bias}}
        """
        student_tt_list = []
        teacher_tt_list = []
        acc_list = []
        sample_counts = []
        all_per_intersection = {}
        constraint_evaluated_list = []
        constraint_satisfied_list = []

        for evaluator, scenario_id, teacher_tt in self.roadnet_evaluators:
            bias_for_roadnet = self._extract_bias_for_roadnet(
                per_intersection_bias, scenario_id, evaluator.data_loader.intersection_ids
            )
            if bias_for_roadnet is None and per_intersection_bias:
                # Try string key format "Jinan_0"
                bias_for_roadnet = {}
                for inter_id in evaluator.data_loader.intersection_ids:
                    sk = f"{scenario_id}_{inter_id}"
                    if sk in per_intersection_bias:
                        bias_for_roadnet[inter_id] = per_intersection_bias[sk]
                bias_for_roadnet = bias_for_roadnet if bias_for_roadnet else None

            result = evaluator.evaluate_all_intersections(
                utility_function=utility_function,
                parameters=parameters,
                max_timesteps=max_timesteps,
                feature_extractor=feature_extractor,
                per_intersection_bias=bias_for_roadnet,
                raise_on_error=raise_on_error,
            )

            student_tt = result["average"].get("student_travel_time") or result["average"].get("sim_avg_travel_time") or teacher_tt
            student_tt_list.append(student_tt)
            teacher_tt_list.append(teacher_tt)
            acc_list.append(result["average"]["action_consistency"])
            # Use real sample count for action consistency weighting
            action_data = evaluator.data_loader.get_action_consistency_data(
                intersection_id=None, max_timesteps=max_timesteps
            )
            n_actions = len(action_data) if action_data else 0
            sample_counts.append(n_actions)

            for inter_id, metrics in result.get("per_intersection", {}).items():
                key = f"{scenario_id}_{inter_id}"
                all_per_intersection[key] = {**metrics, "scenario": scenario_id}

            avg = result.get("average", {})
            if avg.get("constraint_evaluated", False):
                constraint_evaluated_list.append(True)
                constraint_satisfied_list.append(avg.get("constraint_satisfied", False))

            if self.verbose:
                print(f"    [{scenario_id}] student_tt={student_tt:.2f}, teacher_tt={teacher_tt:.2f}, acc={result['average']['action_consistency']:.4f}")

        avg_student_tt = float(np.mean(student_tt_list))
        avg_teacher_tt = float(np.mean(teacher_tt_list))
        delta_d = (avg_student_tt - avg_teacher_tt) / (abs(avg_teacher_tt) + self.epsilon)
        performance_score = float(sigmoid(-delta_d, beta=5.0))

        total_samples = sum(sample_counts)
        if total_samples > 0:
            action_consistency = sum(a * c for a, c in zip(acc_list, sample_counts)) / total_samples
        else:
            action_consistency = float(np.mean(acc_list)) if acc_list else 0.0

        # constraint: any roadnet evaluated -> use AND of all satisfaction (all must pass)
        constraint_evaluated = len(constraint_evaluated_list) > 0
        constraint_satisfied = all(constraint_satisfied_list) if constraint_satisfied_list else None
        constraint_score = 1.0 if constraint_satisfied is True else (0.0 if constraint_satisfied is False else 1.0)
        base_score = (
            float(self.action_weight) * float(action_consistency)
            + float(self.performance_weight) * float(performance_score)
        )
        function_label = float(constraint_score) + float(base_score)

        avg_metrics = {
            "score": float(function_label),
            "score_mode": "review_plus_eval",
            "base_score": float(base_score),
            "full_score": float(function_label),
            "evaluation_score": float(base_score),
            "student_travel_time": avg_student_tt,
            "teacher_travel_time": avg_teacher_tt,
            "delta_d": float(delta_d),
            "action_consistency": float(action_consistency),
            "performance_score": float(performance_score),
            "action_weight": float(self.action_weight),
            "performance_weight": float(self.performance_weight),
            "constraint_score": float(constraint_score),
            "function_label": float(function_label),
            "constraint_evaluated": constraint_evaluated,
            "constraint_penalty": 0.0 if constraint_satisfied else 1.0,
            "constraint_satisfied": constraint_satisfied,
            "num_roadnets": len(self.roadnet_evaluators),
            "per_roadnet_student_tt": student_tt_list,
            "per_roadnet_teacher_tt": teacher_tt_list,
            "sim_avg_travel_time": avg_student_tt,
            "sim_avg_queue_length": None,
            "sim_avg_throughput": None,
            "per_intersection_avg_travel_time": None,
        }

        return {
            "average": avg_metrics,
            "per_intersection": all_per_intersection,
            "num_intersections": len(all_per_intersection),
        }

    def fit_per_intersection_bias(
        self,
        utility_function: FunctionUtility,
        max_timesteps: Optional[int] = None,
        max_iter: int = 100,
        bias_bound: float = 5.0,
        verbose: bool = True,
    ) -> Dict[str, Dict[int, float]]:
        """
        Fit per-intersection bias for each road network; merge into composite key format.

        Returns:
            {f"{scenario_id}_{inter_id}": {phase_id: bias}}
        """
        combined: Dict[str, Dict[int, float]] = {}
        for evaluator, scenario_id, _ in self.roadnet_evaluators:
            if verbose:
                print(f"    Fitting bias for roadnet: {scenario_id}")
            single_bias = evaluator.fit_per_intersection_bias(
                utility_function=utility_function,
                max_timesteps=max_timesteps,
                max_iter=max_iter,
                bias_bound=bias_bound,
                verbose=verbose and self.verbose,
            )
            for inter_id, phase_bias in single_bias.items():
                key = f"{scenario_id}_{inter_id}"
                combined[key] = phase_bias
        return combined

    def _compute_action_consistency(
        self,
        utility_function: FunctionUtility,
        parameters: Dict[str, float],
        intersection_id: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        feature_extractor: Optional[Any] = None,
        per_intersection_bias: Optional[Dict] = None,
        raise_on_error: bool = True,
    ) -> float:
        """Aggregate action consistency across all road networks."""
        total_actions = 0
        weighted_correct = 0.0
        for evaluator, scenario_id, _ in self.roadnet_evaluators:
            bias_for_roadnet = self._extract_bias_for_roadnet(
                per_intersection_bias, scenario_id, evaluator.data_loader.intersection_ids
            )
            if bias_for_roadnet is None and per_intersection_bias:
                bias_for_roadnet = {}
                for inter_id in evaluator.data_loader.intersection_ids:
                    sk = f"{scenario_id}_{inter_id}"
                    if sk in per_intersection_bias:
                        bias_for_roadnet[inter_id] = per_intersection_bias[sk]
                bias_for_roadnet = bias_for_roadnet if bias_for_roadnet else None

            acc = evaluator._compute_action_consistency(
                utility_function, parameters,
                intersection_id=intersection_id,
                max_timesteps=max_timesteps,
                feature_extractor=feature_extractor,
                per_intersection_bias=bias_for_roadnet,
                raise_on_error=raise_on_error,
            )
            action_data = evaluator.data_loader.get_action_consistency_data(
                intersection_id=intersection_id, max_timesteps=max_timesteps
            )
            n = len(action_data) if action_data else 0
            total_actions += n
            weighted_correct += acc * n
        if total_actions == 0:
            return 0.0
        return weighted_correct / total_actions

