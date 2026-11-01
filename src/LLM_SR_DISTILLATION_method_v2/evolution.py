"""
Evolution Module
Manage population and evolution process
"""
import copy
import time
import traceback
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from LLM_SR_DISTILLATION_method_v2.function_utility import FunctionUtility
from LLM_SR_DISTILLATION_method_v2.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION_method_v2.llm_generator import LLMGenerator
from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator


class Individual:
    """Individual in the population"""
    
    def __init__(
        self,
        function_code: str,
        utility_function: FunctionUtility,
        score: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        explanation: Optional[str] = None,
    ):
        """
        Initialize individual
        
        Args:
            function_code: Complete Python function code string (def compute_utility(...): ...)
            utility_function: FunctionUtility instance (executable function)
            score: Fitness score
            metrics: Evaluation metrics dictionary
            explanation: Text explanation of the function (design motivation)
        """
        self.function_code = function_code
        self.utility_function = utility_function
        self.score = score
        self.metrics = metrics or {}
        self.explanation = explanation or ""
        # For backward compatibility (if needed)
        self.expression_template = function_code[:100] + "..." if len(function_code) > 100 else function_code
        self.parameters = {}  # FunctionUtility doesn't use separate parameters
        self.per_intersection_bias: Optional[Dict[int, Dict[int, float]]] = None  # Fitted per-intersection phase bias
    
    def __repr__(self):
        return f"Individual(score={self.score:.4f}, code_length={len(self.function_code)})"


class Evolution:
    """Evolution algorithm manager"""
    
    def __init__(
        self,
        data_loader,
        constraint_validator: ConstraintValidator,
        llm_generator: LLMGenerator,
        evaluator: Evaluator,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        population_size: int = 20,
        n_generations: int = 10,
        mutation_rate: float = 1.0,
        use_intersection_specific_params: bool = True,
        max_constraint_retries: int = 3,
        use_slot_based_parent: bool = False,
        fit_bias_during_evolution: bool = True,
        bias_fit_max_iter: int = 20,
        bias_fit_bound: float = 1.0,
        use_constraint_validation: bool = False,
        max_init_attempts: Optional[int] = 30,
        max_timesteps: Optional[int] = None,
        time_budget_seconds: Optional[float] = None,
        repair_rejected_candidates: bool = False,
        evaluation_error_feedback: bool = False,
    ):
        """
        Initialize evolution algorithm

        Args:
            data_loader: TeacherDataLoader instance
            constraint_validator: ConstraintValidator instance
            llm_generator: LLMGenerator instance
            evaluator: Evaluator instance
            feature_stats: Feature statistics (for feature normalization in FunctionUtility)
            population_size: Population size
            n_generations: Number of generations
            mutation_rate: Mutation rate (probability of mutation)
            use_intersection_specific_params: Whether to use intersection-specific parameters
            max_constraint_retries: Maximum number of constraint validation retries with LLM feedback
            use_slot_based_parent: If True, for slot i use individual at position i as parent (so each position gets one mutation attempt per generation). If False, use tournament selection (best individual is selected often).
            use_constraint_validation: If True, run _validate_with_retries on each new function (init + evolution). Default False for backward compatibility.
            max_init_attempts: Max LLM attempts per individual during initial population (default 30).
        """
        self.data_loader = data_loader
        self.constraint_validator = constraint_validator
        self.llm_generator = llm_generator
        self.evaluator = evaluator
        self.feature_stats = feature_stats or {}
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.use_intersection_specific_params = use_intersection_specific_params
        self.max_constraint_retries = max_constraint_retries
        self.use_slot_based_parent = use_slot_based_parent
        self.fit_bias_during_evolution = fit_bias_during_evolution
        self.bias_fit_max_iter = bias_fit_max_iter
        self.bias_fit_bound = bias_fit_bound
        self.use_constraint_validation = use_constraint_validation
        self.repair_rejected_candidates = bool(repair_rejected_candidates)
        self.evaluation_error_feedback = bool(evaluation_error_feedback)
        self.max_init_attempts = max_init_attempts
        self.max_timesteps = max_timesteps
        self.time_budget_seconds = (
            None
            if time_budget_seconds is None
            else float(time_budget_seconds)
        )
        if (
            self.time_budget_seconds is not None
            and self.time_budget_seconds <= 0.0
        ):
            raise ValueError("time_budget_seconds must be positive")
        self.action_mode = self.llm_generator._decision_mode()
        
        self.population: List[Individual] = []
        self.generation_history: List[List[Individual]] = []
        self.current_generation_index: int = 0
        # A deterministic function receives the same calibration and CityFlow
        # result within one scenario. Keep population and offspring entries
        # separate because the legacy paths intentionally store different
        # metric detail.
        self._evaluation_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Raw evaluator output has one schema in both paths. Sharing it across
        # those paths avoids recalibrating and resimulating an offspring when
        # it appears in the next generation.
        self._raw_evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.search_started_monotonic: Optional[float] = None
        self.search_finished_monotonic: Optional[float] = None
        self.deadline_exhausted: bool = False
        self.candidate_starts: int = 0

    def _elapsed_search_seconds(self) -> float:
        if self.search_started_monotonic is None:
            return 0.0
        end = (
            self.search_finished_monotonic
            if self.search_finished_monotonic is not None
            else time.perf_counter()
        )
        return max(0.0, float(end - self.search_started_monotonic))

    def _can_start_candidate(self) -> bool:
        """Whether a new complete generate/review/evaluate unit may start.

        This implements a soft wall-clock deadline: a candidate that starts
        before the deadline is allowed to finish, but no later candidate is
        started.  The rule avoids censoring slow CityFlow rollouts midway.
        """
        if self.time_budget_seconds is None:
            return True
        allowed = self._elapsed_search_seconds() < self.time_budget_seconds
        if not allowed:
            self.deadline_exhausted = True
        return allowed

    def time_budget_summary(self) -> Dict[str, Any]:
        elapsed = self._elapsed_search_seconds()
        budget = self.time_budget_seconds
        return {
            "unit": "wall_clock_seconds",
            "maximum": budget,
            "stopping_rule": (
                "soft_candidate_boundary"
                if budget is not None
                else "generation_count"
            ),
            "candidate_starts": int(self.candidate_starts),
            "completed_unique_evaluations": int(
                len(self._raw_evaluation_cache)
            ),
            "elapsed_seconds": float(elapsed),
            "overrun_seconds": (
                max(0.0, elapsed - budget) if budget is not None else 0.0
            ),
            "deadline_exhausted": bool(self.deadline_exhausted),
        }

    @staticmethod
    def _stage_log(verbose: bool, generation_label: str, stage: str, detail: Optional[str] = None):
        """Print a consistent stage marker for generation/review/calibration/evaluation."""
        if not verbose:
            return
        msg = f"[{generation_label}] {stage}"
        if detail:
            msg += f" | {detail}"
        print(msg)

    def _restore_cached_evaluation(
        self,
        context: str,
        individual: Individual,
        verbose: bool,
        generation_label: str,
        candidate_label: str,
    ) -> bool:
        """Restore an exact-code evaluation previously computed in this scenario."""
        cached = self._evaluation_cache.get((context, individual.function_code))
        if cached is None:
            return False

        individual.score = copy.deepcopy(cached["score"])
        individual.metrics = copy.deepcopy(cached["metrics"])
        individual.per_intersection_bias = copy.deepcopy(
            cached["per_intersection_bias"]
        )
        self._stage_log(
            verbose,
            generation_label,
            "Evaluation reuse",
            f"{candidate_label} | identical function code",
        )
        return True

    def _cache_evaluation(self, context: str, individual: Individual) -> None:
        """Save only successful evaluations; cached state is never shared by reference."""
        if not np.isfinite(individual.score):
            raise ValueError("Non-finite candidate evaluation score")
        if individual.score == -1.0 or "error" in individual.metrics:
            return
        self._evaluation_cache[(context, individual.function_code)] = {
            "score": copy.deepcopy(individual.score),
            "metrics": copy.deepcopy(individual.metrics),
            "per_intersection_bias": copy.deepcopy(
                individual.per_intersection_bias
            ),
        }

    def _restore_raw_evaluation(
        self,
        function_code: str,
    ) -> Optional[Tuple[Dict[str, Any], Optional[Dict[int, Dict[int, float]]]]]:
        """Return deep-copied calibration and evaluator output for exact code."""
        cached = self._raw_evaluation_cache.get(function_code)
        if cached is None:
            return None
        return (
            copy.deepcopy(cached["all_results"]),
            copy.deepcopy(cached["per_intersection_bias"]),
        )

    def _cache_raw_evaluation(
        self,
        function_code: str,
        all_results: Dict[str, Any],
        per_intersection_bias: Optional[Dict[int, Dict[int, float]]],
    ) -> None:
        """Cache successful raw output without coupling caller-specific metrics."""
        if not np.isfinite(all_results['average']['score']):
            raise ValueError("Non-finite candidate evaluation score")
        self._raw_evaluation_cache[function_code] = {
            "all_results": copy.deepcopy(all_results),
            "per_intersection_bias": copy.deepcopy(per_intersection_bias),
        }

    def _build_parent_info(self, parent: Individual) -> str:
        """
        Build detailed parent information to guide LLM mutation
        
        Args:
            parent: Parent Individual instance
        
        Returns:
            Formatted string containing parent's key metrics and constraints
        """
        info_parts = []
        
        # Metrics (avoid opaque score; show meaningful metrics instead)
        metrics = dict(parent.metrics or {})
        if getattr(getattr(self, "evaluator", None), "action_weight", 1.) == 0:
            for key in ('behavioral_fidelity', 'action_consistency', 'fidelity_score'):
                metrics.pop(key, None)
        if getattr(self, 'evaluation_error_feedback', False) and metrics.get('error'):
            info_parts.append(f"Evaluation Failure: {metrics['error']}")
        if metrics.get('behavioral_fidelity') is not None:
            info_parts.append(f"Behavioral Fidelity: {metrics['behavioral_fidelity']:.4f}")
        elif metrics.get('action_consistency') is not None:
            info_parts.append(f"Behavioral Fidelity: {metrics['action_consistency']:.4f}")
        if metrics.get('control_performance') is not None:
            info_parts.append(f"Control Performance: {metrics['control_performance']:.4f}")
        elif metrics.get('performance_score') is not None:
            info_parts.append(f"Control Performance: {metrics['performance_score']:.4f}")
        elif metrics.get('performance_factor') is not None:
            info_parts.append(f"Performance Factor: {metrics['performance_factor']:.4f}")
        tt_teacher = metrics.get('teacher_travel_time') or metrics.get('teacher_delay')
        if tt_teacher is not None:
            info_parts.append(f"Teacher Avg Travel Time: {tt_teacher:.2f}s")
        tt_student = metrics.get('student_travel_time') or metrics.get('student_delay')
        if tt_student is not None:
            info_parts.append(f"Student Avg Travel Time: {tt_student:.2f}s")

        # Include constraint feedback so mutation can actually fix failures.
        c_sat = metrics.get('constraint_satisfied', None)
        c_result = metrics.get('constraint_result', None)
        if c_sat is True:
            info_parts.append("Constraint Status: satisfied")
        elif c_sat is False:
            info_parts.append("Constraint Status: violated")
        else:
            info_parts.append("Constraint Status: not_evaluated")

        if isinstance(c_result, dict):
            violations = c_result.get('violations', [])
            if isinstance(violations, list) and violations:
                preview = []
                for item in violations[:3]:
                    preview.append(str(item))
                info_parts.append("Constraint Violations:")
                for item in preview:
                    info_parts.append(f"  - {item}")
            msg = c_result.get('message')
            if msg:
                info_parts.append(f"Constraint Message: {msg}")
        
        # Suggestions based on metrics
        suggestions = []
        if metrics.get('behavioral_fidelity', metrics.get('action_consistency', 1.0)) < 0.5:
            suggestions.append("Improve expression to better match teacher actions")
        perf = metrics.get('control_performance', metrics.get('performance_score', metrics.get('performance_factor', 1.0)))
        if perf < 0.8:
            suggestions.append("Optimize parameters or expression structure for better performance")
        
        if suggestions:
            info_parts.append("Suggestions for improvement:")
            for suggestion in suggestions:
                info_parts.append(f"  - {suggestion}")
        
        population = getattr(self, 'population', [])
        if population:
            best = max(population, key=lambda x: x.score)
            bm = best.metrics or {}
            info_parts.append(f"Parent fitness: {parent.score:.6f}; global best fitness: {best.score:.6f}; gap: {best.score-parent.score:.6f}")
            info_parts.append(f"Global-best control score: {bm.get('control_performance')}; source ATT: {bm.get('student_travel_time')}")
            if getattr(getattr(self, "evaluator", None), "action_weight", 1.) > 0:
                info_parts.append(f"Global-best behavioral fidelity: {bm.get('behavioral_fidelity')}")
            history = [max(x.score for x in generation) for generation in getattr(self, 'generation_history', []) if generation]
            stagnant = 0
            for previous, current in reversed(list(zip(history, history[1:]))):
                if current > previous + 1e-10:
                    break
                stagnant += 1
            info_parts.append(f"Consecutive completed generations without strict global-best improvement: {stagnant}")
            from ppl_tsc_current.profile import control_score_description
            info_parts.append("The search compares total fitness, not parent replacement counts. " + control_score_description() + " Preserve legality and finite arithmetic. Explain useful structural changes; the number of changes is not limited.")
            if getattr(getattr(self, "evaluator", None), "action_weight", 1.) > 0:
                info_parts.append("Improve the fidelity/control tradeoff; increasing offline teacher agreement alone can still worsen closed-loop ATT.")
        from source_feedback import feedback
        extra_feedback = feedback(self, parent, 'combined')
        if extra_feedback:
            info_parts.append(extra_feedback)
        return "\n".join(info_parts)

    def _build_constraint_feedback(self, result: Dict[str, Any]) -> str:
        """
        Build clear, actionable feedback from constraint validation result.
        Covers feature keys, numerical guards, real states and signed boundary inputs.
        """
        constraints = result.get('constraints', {}) or {}
        violations = result.get('violations', [])
        max_items = 3

        lines = ["Constraint violations (please fix):"]
        idx = 0

        for key, info in constraints.items():
            if info.get('satisfied', False):
                continue
            idx += 1
            name = info.get('name', key)
            details = info.get('details') or {}
            if isinstance(details, list):
                details = {'violations': details}
            elif not isinstance(details, dict):
                details = {}

            if key == 'feature_keys':
                vios = details.get('violations', [])
                actual = details.get('actual_keys', [])
                lines.append(f"\n{idx}. {name}:")
                for v in vios[:max_items]:
                    lines.append(f"   - {v}")
                if actual:
                    lines.append(f"   Available keys: {', '.join(actual)}")

            elif key == 'real_state':
                vios = info.get('violations', []) or violations
                lines.append(f"\n{idx}. Real-state validation (runtime error):")
                for v in vios[:max_items]:
                    lines.append(f"   - {v}")
                if len(vios) > max_items:
                    lines.append(f"   ... and {len(vios) - max_items} more")

            else:
                vios = details.get('violations', []) if isinstance(details, dict) else []
                lines.append(f"\n{idx}. {name}:")
                for v in (vios if isinstance(vios, list) else [])[:max_items]:
                    txt = v.get('message', v.get('case', v.get('feature', str(v)))) if isinstance(v, dict) else str(v)
                    lines.append(f"   - {txt}")
                if not vios and violations:
                    for v in violations[:max_items]:
                        lines.append(f"   - {v}")

        if len(lines) == 1:
            lines.append("\n" + (violations[0] if violations else "Unknown error"))
        feedback = "\n".join(lines)
        if self.repair_rejected_candidates:
            feedback = self._with_targeted_repair_guidance(feedback)
        return feedback

    @staticmethod
    def _with_targeted_repair_guidance(feedback: str) -> str:
        """Translate validator errors into a concise repair instruction.

        The validator diagnosis remains verbatim so the model can address the
        actual line and expression. No code is silently rewritten or admitted.
        """
        diagnosis = []
        lowered = feedback.casefold()
        if "denominator" in lowered or "zerodivision" in lowered:
            diagnosis.append(
                "For each cited division, guard the entire denominator. "
                "If its sign matters, divide only when abs(d) > 1e-8 and use "
                "0.0 otherwise; use max(abs(d), 1e-8) only for magnitude ratios. "
                "Adding a constant to a signed pressure is not a guard."
            )
        if "feature" in lowered or "keyerror" in lowered or " key " in lowered:
            diagnosis.append(
                "Remove unavailable state keys and read only fields in the "
                "DATA CONTRACT. In binary mode, green_vehicle and red_vehicle "
                "are inside state['binary_features']."
            )
        if "action" in lowered or "phase" in lowered:
            diagnosis.append(
                "Keep the configured action mode: phase_id is a binary action "
                "ID only in binary_keep_switch mode."
            )
        if not diagnosis:
            diagnosis.append("Fix the concrete failing check while preserving the controller interface.")
        return "TARGETED REPAIR INSTRUCTIONS:\n" + "\n".join(diagnosis) + "\nVALIDATOR DIAGNOSTIC:\n" + feedback

    def _repair_rejected_candidate(self, code: str, error: Exception) -> Optional[str]:
        """Spend one existing proposal attempt on a diagnosed repair.

        The rejected code is never executed or scored as valid. This optional
        path changes only the proposal supplied to the existing review gate.
        """
        feedback = f"{type(error).__name__}: {error}"
        if len(feedback) > 1800:
            feedback = feedback[:1800] + " [truncated]"
        return self.llm_generator.mutate_expression(
            code,
            feedback=("Fix the reported validation failure at its source. "
                      "Keep the action interface and the available feature keys unchanged.\n"
                      + self._with_targeted_repair_guidance(feedback)),
            parent_explanation=self.llm_generator.expression_explanations.get(code),
            mutation_context="constraint_fix",
            max_retries=1,
        )

    def _validate_with_retries(
        self,
        function_code: str,
        utility_function: FunctionUtility,
        verbose: bool,
        max_attempts: Optional[int] = None,
        mutation_phase: str = "initialization"
    ) -> Tuple[str, FunctionUtility, Dict[str, Any]]:
        """
        Validate FunctionUtility with retries and LLM-guided mutation.
        Designed for full Python function code (FunctionUtility), not expression templates.

        Args:
            function_code: Full Python function code (def compute_utility(...): ...)
            utility_function: Current FunctionUtility instance
            verbose: Whether to print progress
            max_attempts: Maximum retry attempts (uses self.max_constraint_retries if None)
            mutation_phase: Phase context for LLM mutation - "initialization", "retry", or "evolution"

        Returns:
            Tuple of (function_code, utility_function, validation_result)
        """
        if max_attempts is None:
            max_attempts = self.max_constraint_retries

        current_code = function_code
        current_uf = utility_function
        last_result: Dict[str, Any] = {}
        attempt = 0

        for attempt in range(max_attempts):
            if verbose:
                print(f"    Review attempt {attempt + 1}/{max_attempts} | phase={mutation_phase}")
            # ConstraintValidator expects expression_template for feature key check; pass function_code
            last_result = self.constraint_validator.validate_hybrid(
                expression_template=current_code,
                utility_function=current_uf,
                verbose=False
            )
            if last_result.get('valid', False):
                if verbose:
                    if attempt == 0:
                        print(f"    Review passed")
                    else:
                        print(f"    Repair succeeded")
                        print(f"    Review passed")
                    print(f"    Function code length: {len(current_code)} chars")
                    constraints = last_result.get("constraints", {}) or {}
                    if constraints:
                        checked = ", ".join(constraints.keys())
                        print(f"    Constraints checked: {checked}")
                return current_code, current_uf, last_result

            if verbose:
                print(f"    Review failed -> send feedback to LLM")
                print(f"    Function code length: {len(current_code)} chars")

            if attempt == max_attempts - 1:
                break

            # Build constraint violation feedback for LLM (clear, concise, actionable)
            feedback = self._build_constraint_feedback(last_result)
            
            # Determine mutation context for LLM
            if mutation_phase == "initialization":
                context = "constraint_fix"
            elif mutation_phase == "retry":
                context = "constraint_retry"
            else:
                context = "evolution"
            
            mutated = self.llm_generator.mutate_expression(
                current_code,
                feedback=feedback,
                mutation_context=context
            )
            if not mutated:
                break

            current_code = mutated
            current_uf = FunctionUtility(
                function_code=mutated,
                feature_extractor=self.evaluator.feature_extractor if hasattr(self.evaluator, 'feature_extractor') else None,
                feature_stats=self.feature_stats,
                action_mode=self.action_mode,
            )

        return current_code, current_uf, last_result

    def run(self, verbose: bool = True, print_functions: bool = False) -> List[Individual]:
        """
        Run evolution algorithm
        
        Args:
            verbose: Whether to print progress
            print_functions: If True and verbose, print every generated function (preview) for each individual
        
        Returns:
            Final population
        """
        if verbose:
            print("=" * 80)
            print("Starting Evolution Algorithm")
            print("=" * 80)
            print(f"Action mode: {self.action_mode}")
            if self.time_budget_seconds is not None:
                print(
                    "Search budget: "
                    f"{self.time_budget_seconds:.3f}s wall clock "
                    "(soft candidate boundary)"
                )

        self.search_started_monotonic = time.perf_counter()
        self.search_finished_monotonic = None
        self.deadline_exhausted = False
        self.candidate_starts = 0
        
        # Initialize population
        if verbose:
            print(f"\nGeneration 0: Initializing population of size {self.population_size}...")
            print(f"  Generating {self.population_size} expressions via LLM (may take 1-2 min)...")
        self.current_generation_index = 0
        if self.time_budget_seconds is None:
            self._initialize_population(
                verbose=verbose,
                print_functions=print_functions,
            )
            self.candidate_starts += len(self.population)
        else:
            self._initialize_population_time_limited(
                verbose=verbose,
                print_functions=print_functions,
            )
        self._assert_full_population()
        if not self.population:
            self.search_finished_monotonic = time.perf_counter()
            raise RuntimeError(
                "No valid initial candidate; inspect generation, review, and calibration errors"
                if self.time_budget_seconds is None else
                "Search time budget expired before one valid initial candidate could be evaluated"
            )
        
        # Evolution loop
        for generation in range(self.n_generations):
            self.current_generation_index = generation + 1
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Generation {generation + 1}/{self.n_generations}")
                print(f"{'=' * 80}")
            
            # Evaluate population
            self._evaluate_population(verbose=verbose)
            
            # Store generation history
            self.generation_history.append(self.population.copy())
            
            # Print best individual
            if verbose:
                best = self._get_best_individual()
                print(f"\nBest individual in generation {generation + 1}:")
                if best.score == -1.0 and "error" in best.metrics:
                    print(f"  [WARNING] Evaluation failed for all individuals in this generation!")
                    print(f"  Error: {best.metrics.get('error_type', 'Unknown')}: {best.metrics.get('error', '')}")
                print(f"  Function Code Length: {len(best.function_code)} chars")
                if print_functions:
                    self._print_function_preview(
                        best.function_code,
                        title="Best function (preview)",
                        explanation=best.explanation,
                    )
                print(f"  Evaluation Score: {best.score:.4f}")
                print(f"      * Behavioral Fidelity: {best.metrics.get('behavioral_fidelity', best.metrics.get('action_consistency', 0)):.4f}")
                print(f"      * Control Performance: {best.metrics.get('control_performance', best.metrics.get('performance_score', 0)):.4f}")
                st = best.metrics.get('student_travel_time') or best.metrics.get('student_delay', 0)
                tt = best.metrics.get('teacher_travel_time') or best.metrics.get('teacher_delay', 0)
                print(f"    - Avg Travel Time: {st:.4f} (Teacher: {tt:.4f})")
                print(f"    - Delta: {best.metrics.get('delta_d', 0):.4f}")
                
                # Show per-intersection details
                per_inter = best.metrics.get('per_intersection', {})
                if per_inter:
                    print(f"\n  Per-Intersection Details ({len(per_inter)} intersections):")
                    for inter_id in sorted(per_inter.keys()):
                        m = per_inter[inter_id]
                        print(f"    Intersection {inter_id}:")
                        print(
                            f"      Score: {m['score']:.4f} | "
                            f"Behavioral Fidelity: {m.get('behavioral_fidelity', m.get('fidelity_score', 0.0)):.4f} | "
                            f"Control Performance: {m.get('control_performance', m.get('performance_score', 0.0)):.4f}"
                        )
            
            # Generate next generation (except last generation)
            if generation < self.n_generations - 1:
                if not self._can_start_candidate():
                    if verbose:
                        print(
                            "\nSearch deadline reached; no new candidate "
                            "will be started."
                        )
                    break
                self._evolve_population(verbose=verbose, print_functions=print_functions)
        
        # Do not repeat calibration and CityFlow evaluation after the final
        # generation. Reusing the already evaluated population is faster and
        # prevents stochastic calibration from changing the reported result.
        if self.n_generations <= 0:
            self._evaluate_population(verbose=verbose)

        self.search_finished_monotonic = time.perf_counter()
        self.require_valid_best()
        
        if verbose:
            print("\n" + "=" * 80)
            print("Evolution Complete")
            print("=" * 80)
            best = self._get_best_individual()
            print(f"\nBest individual overall:")
            print(f"  Expression: {best.expression_template}")
            print(f"  Evaluation Score: {best.score:.4f}")
            print(f"    - Behavioral Fidelity: {best.metrics.get('behavioral_fidelity', best.metrics.get('action_consistency', 0)):.4f}")
            print(f"    - Control Performance: {best.metrics.get('control_performance', best.metrics.get('performance_score', 0)):.4f}")
            st = best.metrics.get('student_travel_time') or best.metrics.get('student_delay', 0)
            tt = best.metrics.get('teacher_travel_time') or best.metrics.get('teacher_delay', 0)
            print(f"    - Avg Travel Time: {st:.4f}")
            print(f"    - Teacher Travel Time: {tt:.4f}")
            print(f"    - Delta: {best.metrics.get('delta_d', 0):.4f}")
            if print_functions:
                self._print_function_preview(
                    best.function_code,
                    title="Best individual overall (preview)",
                    explanation=best.explanation,
                )
            
            # Show per-intersection details
            per_inter = best.metrics.get('per_intersection', {})
            if per_inter:
                print(f"\n  Per-Intersection Details ({len(per_inter)} intersections):")
                for inter_id in sorted(per_inter.keys()):
                    m = per_inter[inter_id]
                    print(f"    Intersection {inter_id}:")
                    print(
                        f"      Score: {m['score']:.4f} | "
                        f"Behavioral Fidelity: {m.get('behavioral_fidelity', m.get('fidelity_score', 0.0)):.4f} | "
                        f"Control Performance: {m.get('control_performance', m.get('performance_score', 0.0)):.4f}"
                    )
        
        return self.population

    def _initialize_population_time_limited(
        self,
        verbose: bool = False,
        print_functions: bool = False,
    ) -> None:
        """Generate and evaluate initial candidates one at a time.

        The legacy initializer first generates the entire population and only
        then evaluates it.  Under a time budget that would make the boundary
        ambiguous and could spend the whole budget on unevaluated LLM output.
        This path reserves one complete candidate at a time and is used only
        when ``time_budget_seconds`` is explicitly configured.
        """
        self.population = []
        target_size = int(self.population_size)

        for candidate_index in range(target_size):
            if not self._can_start_candidate():
                break
            self.candidate_starts += 1
            generation_label = f"Generation {self.current_generation_index}"
            candidate_label = (
                f"individual {candidate_index + 1}/{target_size}"
            )
            last_error: Optional[Exception] = None
            individual: Optional[Individual] = None

            for review_attempt in range(6):
                try:
                    function_code = self.llm_generator.generate_initial_population(
                        1,
                        max_total_attempts_per_individual=self.max_init_attempts,
                        verbose=verbose,
                    )[0]
                    self._stage_log(
                        verbose,
                        generation_label,
                        "Generation stage",
                        f"{candidate_label} | initial candidate created",
                    )
                    reviewed_code, utility_function, _review_result = (
                        self._compile_and_review_candidate(
                            function_code=function_code,
                            verbose=verbose,
                            mutation_phase=(
                                "initialization"
                                if review_attempt == 0
                                else "retry"
                            ),
                            generation_label=generation_label,
                            candidate_label=candidate_label,
                        )
                    )
                    explanation = self.llm_generator.expression_explanations.get(
                        reviewed_code,
                        self.llm_generator.expression_explanations.get(
                            function_code, ""
                        ),
                    )
                    individual = Individual(
                        function_code=reviewed_code,
                        utility_function=utility_function,
                        score=0.0,
                        explanation=explanation,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if verbose:
                        print(
                            f"  [FAIL] {candidate_label} review attempt "
                            f"{review_attempt + 1}/6: "
                            f"{type(exc).__name__}: {exc}"
                        )

            if individual is None:
                if verbose:
                    print(
                        f"  [FAIL] {candidate_label} could not be initialized"
                        + (
                            f": {type(last_error).__name__}: {last_error}"
                            if last_error is not None
                            else ""
                        )
                    )
                continue

            self.population.append(individual)
            if verbose and print_functions:
                self._print_function_preview(
                    individual.function_code,
                    title=f"{candidate_label} (time-budgeted initial)",
                    explanation=individual.explanation,
                )

            # The candidate was reserved before its LLM call, so its
            # calibration and CityFlow rollout may finish after the deadline.
            self._evaluate_population(verbose=verbose)

        if verbose:
            print(
                f"Successfully initialized {len(self.population)}/"
                f"{target_size} individuals under the time budget"
            )

    @staticmethod
    def _print_function_preview(
        function_code: str,
        title: str = "Function",
        max_lines: int = 30,
        explanation: Optional[str] = None,
    ) -> None:
        """Print function code and optional explanation for debugging."""
        code = function_code.strip()
        # Strip trailing ### Explanation line (redundant when we show explanation separately)
        if code.endswith("### Explanation") or "\n### Explanation" in code:
            code = code.split("\n### Explanation")[0].rstrip()
        lines = code.split("\n")
        preview = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += f"\n... ({len(lines) - max_lines} more lines)"
        print(f"\n--- {title} ---\n{preview}")
        if explanation and explanation.strip():
            print(f"\nExplanation: {explanation.strip()}")
        print("---")

    def _compile_and_review_candidate(
        self,
        function_code: str,
        verbose: bool,
        mutation_phase: str,
        generation_label: str = "Generation",
        candidate_label: str = "candidate",
    ) -> Tuple[str, FunctionUtility, Dict[str, Any]]:
        """
        Compile a candidate and run the configured review pipeline.

        Returns:
            (final_code, utility_function, review_result)
        """
        utility_function = FunctionUtility(
            function_code=function_code,
            feature_extractor=self.evaluator.feature_extractor if hasattr(self.evaluator, 'feature_extractor') else None,
            feature_stats=self.feature_stats,
            action_mode=self.action_mode,
        )

        self._stage_log(
            verbose,
            generation_label,
            "Review stage",
            f"{candidate_label} | phase={mutation_phase} | compile"
        )

        if self.use_constraint_validation:
            self._stage_log(
                verbose,
                generation_label,
                "Review stage",
                f"{candidate_label} | phase={mutation_phase} | iterative review+repair"
            )
            final_code, final_uf, review_result = self._validate_with_retries(
                function_code=function_code,
                utility_function=utility_function,
                verbose=verbose,
                mutation_phase=mutation_phase,
            )
            if not review_result.get("valid", False):
                raise RuntimeError(
                    "Constraint review failed after repair attempts: "
                    f"{review_result.get('message', review_result.get('violations', 'unknown review error'))}"
                )
            return final_code, final_uf, review_result

        self._stage_log(
            verbose,
            generation_label,
            "Review stage",
            f"{candidate_label} | phase={mutation_phase} | single-pass review"
        )
        fk = self.constraint_validator.check_feature_key_constraints(function_code)
        if not fk.get("valid", True):
            raise RuntimeError(f"Feature keys: {fk.get('violations', [])}")
        numeric = self.constraint_validator.check_numerical_safety(function_code)
        if not numeric['valid']:
            raise RuntimeError(f"Numerical safety: {numeric['violations']}")
        if self.constraint_validator.data_loader and self.constraint_validator.feature_stats:
            rs_valid, rs_msg = self.constraint_validator._validate_on_real_states(utility_function, max_samples=25)
            if not rs_valid:
                raise RuntimeError(f"Real-state validation failed: {rs_msg}")
        return function_code, utility_function, {"valid": True, "mode": "single_pass_review"}

    def _assert_full_population(self):
        """Never report a reduced population as a fixed-budget experiment."""
        if self.time_budget_seconds is None and len(self.population) != self.population_size:
            raise RuntimeError(
                f"Incomplete initial population: {len(self.population)}/{self.population_size}; "
                "fixed-population search must not silently run a smaller experiment"
            )

    def _initialize_population(self, verbose: bool = False, print_functions: bool = False):
        """Initialize population with LLM-generated Python functions"""
        function_codes = self.llm_generator.generate_initial_population(
            self.population_size,
            max_total_attempts_per_individual=self.max_init_attempts,
            verbose=verbose,
        )
        
        self.population = []
        for i, function_code in enumerate(function_codes):
            generation_label = f"Generation {self.current_generation_index}"
            candidate_label = f"individual {i + 1}/{self.population_size}"
            if verbose and (i + 1) % 5 == 0:
                print(f"  Initializing individual {i + 1}/{self.population_size}...")
            
            try:
                self._stage_log(verbose, generation_label, "Generation stage", f"{candidate_label} | initial candidate created")
                reviewed_code, utility_function, _review_result = self._compile_and_review_candidate(
                    function_code=function_code,
                    verbose=verbose,
                    mutation_phase="initialization",
                    generation_label=generation_label,
                    candidate_label=candidate_label,
                )
                explanation = self.llm_generator.expression_explanations.get(
                    reviewed_code,
                    self.llm_generator.expression_explanations.get(function_code, "")
                )
                
                individual = Individual(
                    function_code=reviewed_code,
                    utility_function=utility_function,
                    score=0.0,
                    explanation=explanation,
                )
                
                self.population.append(individual)
                if verbose:
                    print(f"  [OK] Successfully generated individual {i + 1}/{self.population_size}")
                if verbose and print_functions:
                    self._print_function_preview(
                        reviewed_code,
                        title=f"Individual {i + 1}/{self.population_size} (initial)",
                        explanation=explanation,
                    )
                
            except Exception as e:
                if verbose:
                    print(f"  [FAIL] Individual {i + 1} failed (compilation error): {type(e).__name__}: {e}")
                    print(f"    Regenerating...")
                
                retry_count = 0
                rejected_code = function_code
                rejection = e
                while len(self.population) < i + 1 and retry_count < 5:
                    new_code = None
                    try:
                        if self.repair_rejected_candidates and rejected_code:
                            new_code = self._repair_rejected_candidate(rejected_code, rejection)
                        else:
                            new_code = self.llm_generator._generate_single_expression()
                        if not new_code:
                            retry_count += 1
                            rejected_code = None
                            continue
                        self._stage_log(verbose, generation_label, "Generation stage", f"{candidate_label} | regenerated candidate {retry_count + 1}/5")
                        reviewed_code, utility_function, _review_result = self._compile_and_review_candidate(
                            function_code=new_code,
                            verbose=verbose,
                            mutation_phase="retry",
                            generation_label=generation_label,
                            candidate_label=candidate_label,
                        )
                        explanation = self.llm_generator.expression_explanations.get(
                            reviewed_code,
                            self.llm_generator.expression_explanations.get(new_code, "")
                        )
                        
                        individual = Individual(
                            function_code=reviewed_code,
                            utility_function=utility_function,
                            score=0.0,
                            explanation=explanation,
                        )
                        
                        self.population.append(individual)
                        if verbose:
                            print(f"    [OK] Successfully regenerated individual {i + 1}/{self.population_size}")
                        if verbose and print_functions:
                            self._print_function_preview(
                                reviewed_code,
                                title=f"Individual {i + 1}/{self.population_size} (regenerated)",
                                explanation=explanation,
                            )
                        break
                        
                    except Exception as retry_e:
                        rejected_code = new_code if isinstance(new_code, str) else None
                        rejection = retry_e
                        retry_count += 1
                        if verbose and retry_count < 5:
                            print(f"    Retry {retry_count}/5...")
                        continue
        
        if verbose:
            print(f"Successfully initialized {len(self.population)}/{self.population_size} individuals")
    
    def _evaluate_population(self, verbose: bool = False):
        """Evaluate all individuals in population. Optionally run calibration before evaluation."""
        for i, individual in enumerate(self.population):
            generation_label = f"Generation {self.current_generation_index}"
            candidate_label = f"individual {i + 1}/{len(self.population)}"
            if verbose and (i + 1) % 5 == 0:
                print(f"  Evaluating individual {i + 1}/{len(self.population)}...")
            
            # Calibration stage: fit per-intersection phase bias before evaluation.
            per_intersection_bias = None
            base_action_consistency = None
            bias_action_consistency = None
            try:
                if self._restore_cached_evaluation(
                    "population",
                    individual,
                    verbose,
                    generation_label,
                    candidate_label,
                ):
                    if verbose:
                        num_intersections = individual.metrics.get(
                            "num_intersections", "cached"
                        )
                        print(
                            f"  [OK] Individual {i + 1} avg score: "
                            f"{individual.score:.6f} "
                            f"(reused; {num_intersections} intersections)"
                        )
                    continue

                raw_evaluation = self._restore_raw_evaluation(
                    individual.function_code
                )
                if raw_evaluation is None:
                    all_results = None
                else:
                    all_results, per_intersection_bias = raw_evaluation

                if self.fit_bias_during_evolution and isinstance(individual.utility_function, FunctionUtility):
                    self._stage_log(verbose, generation_label, "Calibration stage", candidate_label)
                    if verbose and (i + 1) % 5 == 0:
                        print(f"    Calibrating per-intersection phase bias for individual {i + 1}...")
                    self.evaluator.assert_utility_runtime(
                        individual.utility_function
                    )
                    base_action_consistency = self.evaluator._compute_action_consistency(
                        individual.utility_function,
                        {},
                        intersection_id=None,
                        max_timesteps=self.max_timesteps,
                        feature_extractor=None,
                        per_intersection_bias=None,
                        raise_on_error=True,
                    )
                
                    # Debug: check if base_action_consistency is too low or no data
                    if verbose:
                        if base_action_consistency == 0.0:
                            print(f"    WARNING: base behavioral fidelity = 0.0 (no data or all wrong)")
                        elif base_action_consistency < 0.05:
                            print(f"    [WARNING] CRITICAL: base behavioral fidelity = {base_action_consistency:.4f} (very low!)")
                            # Show complete generated function to diagnose
                            print(f"    ===== Generated function (full code) =====")
                            print(individual.utility_function.function_code)
                            print(f"    ===== End of function =====")
                    
                    if raw_evaluation is None:
                        per_intersection_bias = self.evaluator.fit_per_intersection_bias(
                            individual.utility_function,
                            max_timesteps=self.max_timesteps,
                            max_iter=self.bias_fit_max_iter,
                            bias_bound=self.bias_fit_bound,
                            verbose=True,  # Show bias fit details to diagnose issues
                        )
                    else:
                        self._stage_log(
                            verbose,
                            generation_label,
                            "Calibration reuse",
                            f"{candidate_label} | evaluated as offspring",
                        )
                    bias_action_consistency = self.evaluator._compute_action_consistency(
                        individual.utility_function,
                        {},
                        intersection_id=None,
                        max_timesteps=self.max_timesteps,
                        feature_extractor=None,
                        per_intersection_bias=per_intersection_bias,
                        raise_on_error=True,
                    )
                    individual.per_intersection_bias = per_intersection_bias
                    if verbose and base_action_consistency is not None and bias_action_consistency is not None:
                        delta = bias_action_consistency - base_action_consistency
                        
                        # Show sample bias values to diagnose if they're actually non-zero
                        sample_inter = list(per_intersection_bias.keys())[0] if per_intersection_bias else None
                        sample_bias = per_intersection_bias.get(sample_inter, {}) if sample_inter is not None else {}
                        if sample_bias:
                            # Only show non-zero biases for clarity
                            nonzero_bias = [(pid, val) for pid, val in sorted(sample_bias.items()) if abs(val) > 0.001]
                            if nonzero_bias:
                                bias_str = ", ".join([f"P{pid}:{val:+.3f}" for pid, val in nonzero_bias])
                                print(f"    Sample bias (Inter {sample_inter}): [{bias_str}]")
                            else:
                                print(f"    Sample bias (Inter {sample_inter}): [all zeros] (no calibration improvement found)")
                        
                        print(
                            f"    Calibrated behavioral fidelity: {base_action_consistency:.4f} -> "
                            f"{bias_action_consistency:.4f} (delta {delta:+.4f})"
                        )
                
                # Evaluate across all intersections (with fitted bias if available).
                if all_results is None:
                    self._stage_log(verbose, generation_label, "Evaluation stage", candidate_label)
                    all_results = self.evaluator.evaluate_all_intersections(
                        individual.utility_function,
                        {},  # Empty parameters dict for FunctionUtility
                        max_timesteps=self.max_timesteps,
                        per_intersection_bias=per_intersection_bias,
                    )
                    self._cache_raw_evaluation(
                        individual.function_code,
                        all_results,
                        per_intersection_bias,
                    )
                else:
                    self._stage_log(
                        verbose,
                        generation_label,
                        "Evaluation reuse",
                        f"{candidate_label} | evaluated as offspring",
                    )
                
                # Store average metrics for scoring
                individual.score = all_results['average']['score']
                if not np.isfinite(individual.score):
                    raise ValueError("Non-finite candidate evaluation score")
                individual.metrics = all_results['average'].copy()
                if base_action_consistency is not None and bias_action_consistency is not None:
                    individual.metrics['bias_action_consistency_before'] = float(base_action_consistency)
                    individual.metrics['bias_action_consistency_after'] = float(bias_action_consistency)
                    individual.metrics['bias_action_consistency_delta'] = float(
                        bias_action_consistency - base_action_consistency
                    )
                    individual.metrics['calibration_fidelity_before'] = float(base_action_consistency)
                    individual.metrics['calibration_fidelity_after'] = float(bias_action_consistency)
                    individual.metrics['calibration_fidelity_delta'] = float(
                        bias_action_consistency - base_action_consistency
                    )
                
                # Also store per-intersection details
                individual.metrics['per_intersection'] = all_results['per_intersection']
                individual.metrics['num_intersections'] = all_results['num_intersections']
                self._cache_evaluation("population", individual)
                
                if verbose:
                    print(f"  [OK] Individual {i + 1} avg score: {individual.score:.6f} (across {all_results['num_intersections']} intersections)")
            except Exception as e:
                if verbose:
                    print(f"  Evaluation error (individual {i + 1}): {type(e).__name__}: {e}")
                    traceback.print_exc()
                individual.score = -1.0
                individual.metrics = {'error': str(e), 'error_type': type(e).__name__}
        
        # Sort population by score (descending)
        self.population.sort(key=lambda x: x.score, reverse=True)
    
    def _evolve_population(self, verbose: bool = False, print_functions: bool = False):
        """Evolve population to next generation (maintain population size)"""
        # Update each population slot: select parent -> mutate -> evaluate -> replace
        slot_count = len(self.population)
        for i in range(slot_count):
            if not self._can_start_candidate():
                if verbose:
                    print(
                        "    Search deadline reached at a candidate "
                        "boundary; remaining slots are skipped."
                    )
                break
            self.candidate_starts += 1
            generation_label = f"Generation {self.current_generation_index}"
            candidate_label = f"slot {i + 1}/{slot_count}"
            # Select parent: slot-based (position i) or tournament (best among 3 random)
            if self.use_slot_based_parent:
                parent_idx = i
                parent = self.population[parent_idx]
            else:
                parent = self._tournament_selection()
                parent_idx = self.population.index(parent)
            
            # Mutate to create offspring
            if np.random.random() < self.mutation_rate:
                # Build detailed parent information for LLM
                parent_info = self._build_parent_info(parent)
                
                # Ask the LLM for a mutation, then compile and review it on
                # real teacher states before accepting the offspring.
                mutated_code = None
                utility_function = None
                max_parse_retries = 4
                last_review_error = None
                rejected_mutation = None
                for retry_idx in range(max_parse_retries):
                    self._stage_log(
                        verbose,
                        generation_label,
                        "Generation stage",
                        f"{candidate_label} | mutate parent position {parent_idx + 1} | attempt {retry_idx + 1}/{max_parse_retries}"
                    )
                    if self.repair_rejected_candidates and rejected_mutation and last_review_error:
                        mutated_code = self._repair_rejected_candidate(
                            rejected_mutation, last_review_error,
                        )
                    else:
                        mutated_code = self.llm_generator.mutate_expression(
                            parent.function_code,
                            feedback=parent_info,
                            parent_explanation=parent.explanation,
                            mutation_context="evolution",
                            parent_metrics=parent.metrics if parent.metrics else None,
                        )
                    if not mutated_code:
                        continue
                    try:
                        reviewed_code, utility_function, _review_result = self._compile_and_review_candidate(
                            function_code=mutated_code,
                            verbose=verbose,
                            mutation_phase="evolution",
                            generation_label=generation_label,
                            candidate_label=candidate_label,
                        )
                        mutated_code = reviewed_code
                        if verbose and retry_idx > 0:
                            print(
                                f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): "
                                f"Mutation succeeded after {retry_idx + 1} LLM attempts"
                            )
                        break
                    except Exception as e:
                        last_review_error = e
                        rejected_mutation = mutated_code
                        mutated_code = None
                        utility_function = None
                        continue

                # If mutation candidates still fail review after repair attempts,
                # regenerate a fresh candidate instead of keeping the failed one.
                if (not mutated_code or utility_function is None):
                    regen_retries = 3
                    for regen_idx in range(regen_retries):
                        try:
                            self._stage_log(
                                verbose,
                                generation_label,
                                "Generation stage",
                                f"{candidate_label} | regenerate fresh candidate {regen_idx + 1}/{regen_retries}"
                            )
                            regenerated_code = self.llm_generator._generate_single_expression()
                            if not regenerated_code:
                                continue
                            reviewed_code, utility_function, _review_result = self._compile_and_review_candidate(
                                function_code=regenerated_code,
                                verbose=verbose,
                                mutation_phase="retry",
                                generation_label=generation_label,
                                candidate_label=candidate_label,
                            )
                            mutated_code = reviewed_code
                            if verbose:
                                print(
                                    f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): "
                                    f"mutation review failed, regenerated a fresh reviewed candidate "
                                    f"({regen_idx + 1}/{regen_retries})"
                                )
                            break
                        except Exception as regen_e:
                            last_review_error = regen_e
                            mutated_code = None
                            utility_function = None
                            continue
                
                if mutated_code and utility_function:
                    offspring = Individual(
                        function_code=mutated_code,
                        utility_function=utility_function,
                        score=0.0,
                        explanation=self.llm_generator.expression_explanations.get(mutated_code, ""),
                    )
                    
                    # Calibration stage before evaluation.
                    per_intersection_bias = None
                    try:
                        reused_evaluation = self._restore_cached_evaluation(
                            "offspring",
                            offspring,
                            verbose,
                            generation_label,
                            candidate_label,
                        )
                        if not reused_evaluation:
                            raw_evaluation = self._restore_raw_evaluation(
                                offspring.function_code
                            )
                            if raw_evaluation is None:
                                if self.fit_bias_during_evolution and isinstance(offspring.utility_function, FunctionUtility):
                                    self._stage_log(verbose, generation_label, "Calibration stage", candidate_label)
                                    if verbose:
                                        print(f"    Slot {i + 1}/{self.population_size}: Calibrating phase bias for offspring...")
                                    per_intersection_bias = self.evaluator.fit_per_intersection_bias(
                                        offspring.utility_function,
                                        max_timesteps=self.max_timesteps,
                                        max_iter=self.bias_fit_max_iter,
                                        bias_bound=self.bias_fit_bound,
                                        verbose=False,
                                    )
                                    offspring.per_intersection_bias = per_intersection_bias
                                
                                # Evaluate offspring
                                self._stage_log(verbose, generation_label, "Evaluation stage", candidate_label)
                                all_results = self.evaluator.evaluate_all_intersections(
                                    offspring.utility_function,
                                    {},  # Empty parameters for FunctionUtility
                                    max_timesteps=self.max_timesteps,
                                    per_intersection_bias=per_intersection_bias,
                                )
                                self._cache_raw_evaluation(
                                    offspring.function_code,
                                    all_results,
                                    per_intersection_bias,
                                )
                            else:
                                all_results, per_intersection_bias = raw_evaluation
                                offspring.per_intersection_bias = per_intersection_bias
                                self._stage_log(
                                    verbose,
                                    generation_label,
                                    "Evaluation reuse",
                                    f"{candidate_label} | evaluated in population",
                                )
                            offspring.score = all_results['average']['score']
                            offspring.metrics = all_results['average'].copy()
                            self._cache_evaluation("offspring", offspring)
                    except (
                        ValueError, TypeError, KeyError, AttributeError,
                        ZeroDivisionError, ArithmeticError, NameError,
                        UnboundLocalError, RuntimeError, IndexError,
                    ) as e:
                        if verbose:
                            print(f"    Slot {i + 1}/{self.population_size}: Offspring invalid (runtime error: {type(e).__name__}), keeping parent")
                        continue
                    
                    if verbose and print_functions:
                        self._print_function_preview(
                            offspring.function_code,
                            title=f"Slot {i + 1}/{self.population_size} - Parent at position {parent_idx + 1} (offspring)",
                            explanation=self.llm_generator.expression_explanations.get(offspring.function_code, ""),
                        )
                    # Compare offspring vs parent based on overall score
                    if offspring.score > parent.score:
                        self.population[parent_idx] = offspring
                        if verbose:
                            print(
                                f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): "
                                f"mutation succeeded, new individual better ({offspring.score:.4f} > {parent.score:.4f}), replaced"
                            )
                    else:
                        if verbose:
                            print(
                                f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): "
                                f"mutation succeeded, but parent better ({parent.score:.4f} > {offspring.score:.4f}), kept"
                            )
                else:
                    # No valid reviewed candidate after mutation and regeneration; keep parent.
                    if verbose:
                        reason = getattr(
                            self.llm_generator, "_last_mutation_failure_reason", None
                        )
                        msg = f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): No valid mutation, kept parent"
                        if reason:
                            msg += f" - reason: {reason[:80]}"
                        if last_review_error is not None:
                            msg += f" - review error: {type(last_review_error).__name__}"
                        print(msg)
                    continue
            else:
                # No mutation, keep parent
                if verbose:
                    print(f"    Slot {i + 1}/{self.population_size} (position {parent_idx + 1}): No mutation, kept parent")
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        tournament = np.random.choice(
            self.population,
            size=min(tournament_size, len(self.population)),
            replace=False
        )
        return max(tournament, key=lambda x: x.score)
    
    def _get_best_individual(self) -> Individual:
        """Get best individual in current population"""
        return max(self.population, key=lambda x: x.score)

    def require_valid_best(self) -> Individual:
        """Do not report or export a run whose candidates all failed evaluation."""
        if not self.population:
            raise RuntimeError("Search produced no candidate")
        best = self._get_best_individual()
        if (not np.isfinite(best.score) or best.score < 0
                or best.metrics.get("error")):
            raise RuntimeError("Search produced no successfully evaluated candidate")
        return best

    def save_best_model(
        self,
        save_path: str,
        feature_extractor=None,
        per_intersection_bias: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Save complete configuration of the best model (for deployment).
        
        Args:
            save_path: Save path (JSON file)
            feature_extractor: FeatureExtractor instance, used to get statistics
            per_intersection_bias: Optional {inter_id: {phase_id: bias}} for deployment (Q_adjusted = Q_raw + bias).
                                   If None, uses best.per_intersection_bias if available.
        
        Returns:
            Saved configuration dictionary
        """
        import json
        from pathlib import Path
        
        best = self.require_valid_best()
        # Use individual's fitted bias if available, otherwise use provided bias
        if per_intersection_bias is None and hasattr(best, 'per_intersection_bias'):
            per_intersection_bias = best.per_intersection_bias
        
        # Build complete configuration
        run_config = getattr(self, "run_config", {}) or {}
        config = {
            "function_code": best.function_code,
            "score": best.score,
            "metrics": best.metrics,
            "explanation": best.explanation,
            "run_config": run_config,
            # Feature statistics required during deployment.
            "feature_stats": {},
            "feature_names": self.llm_generator.feature_names,
            "action_mode": self.action_mode,
            
            # Per-intersection phase bias (optional): use in deployment as Q_adjusted[phase] = Q_raw[phase] + bias[inter_id][phase]
            "per_intersection_bias": {} if per_intersection_bias is None else {
                str(k): {str(p): float(v) for p, v in vb.items()}
                for k, vb in per_intersection_bias.items()
            },
            
            # Deployment instructions
            "deployment_info": {
                "warning": "This is a complete Python function - import and use directly",
                "usage": "from best_student import compute_utility; Q = compute_utility(state, phase_id, feature_stats)",
                "per_intersection_bias": "If present, add bias[inter_id][phase_id] to Q when selecting action for that intersection.",
                "action_mode": self.action_mode,
            }
        }
        
        # Get feature statistics
        if feature_extractor is not None and hasattr(feature_extractor, 'feature_stats'):
            config["feature_stats"] = feature_extractor.feature_stats
        else:
            # If not provided, use default range (normalized)
            config["feature_stats"] = {
                name: {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5}
                for name in self.llm_generator.feature_names
            }
        
        # Save JSON config
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_obj, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Save a directly importable Python utility next to the JSON artifact.
        py_path = save_path_obj.with_suffix('.py')
        
        # Save generated Python utility function code.
        py_code_lines = [
            '"""',
            "Auto-generated student utility function (evolved via function evolution).",
            "Inputs:",
            "  - state: dict, raw state dictionary for the evaluated phase",
            "  - phase_id: int, current phase ID being evaluated (optional)",
            "  - feature_stats: dict, optional {'min','max','mean','std'} per feature for normalization",
            "Output:",
            "  - utility Q value (larger value means the phase is more preferable)",
            '"""',
            "",
            "import numpy as np",
            "",
            best.function_code,
            ""
        ]

        with open(py_path, "w", encoding="utf-8") as f_py:
            f_py.write("\n".join(py_code_lines))

        print(f"\n[OK] Model saved to: {save_path_obj}")  # JSON
        print(f"  Function code length: {len(best.function_code)} characters")
        print(f"  Score: {best.score:.4f}")
        print(f"  Feature statistics: {len(config['feature_stats'])} features")
        print(f"  Python utility function saved to: {py_path}")

        return config
