"""Generate and mutate executable utilities using one explicit prompt implementation."""

import re
import time
import copy
from typing import Optional, List, Dict, Tuple, Callable
from ppl_tsc.llm.client import run_llm
from ppl_tsc.llm.prompts import render
from ppl_tsc.config import CALIBRATION


class LLMGenerator:
    """LLM utility generator using the declared action and feature interface."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model: str,
        feature_names: List[str],
        num_phases: int,
        teacher_model: str,
        observation_features: Optional[List[str]] = None,
        debug_mode: bool = False,
        action_mode: str = "auto",
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        actual_feature_keys: Optional[List[str]] = None,
        request: Optional[Callable] = None,
        fit_phase_bias: bool = True,
        calibration_bias_bound: float = CALIBRATION.bias_bound,
    ):
        """
        Initialize LLM generator

        Args:
            api_endpoint: LLM API endpoint
            api_key: LLM API key
            model: LLM model name
            feature_names: List of feature names
            num_phases: Number of phases
            teacher_model: Teacher model name
            observation_features: Raw observation feature names from config.json
            debug_mode: Debug mode flag
            action_mode: Policy decision rule. ``auto`` uses keep/switch for EcoLight
                and all-phase argmax for other teachers. Otherwise choose
                ``phase_argmax``, ``keep_switch``, or
                ``binary_keep_switch``.
        """
        if action_mode not in {
            "auto",
            "phase_argmax",
            "keep_switch",
            "binary_keep_switch",
        }:
            raise ValueError(
                "action_mode must be one of: auto, phase_argmax, " "keep_switch, binary_keep_switch"
            )
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.feature_names = feature_names
        self.num_phases = num_phases
        self.teacher_model = teacher_model
        self.observation_features = list(observation_features or [])
        self.actual_feature_keys = (
            list(actual_feature_keys) if actual_feature_keys is not None else None
        )
        self.debug_mode = debug_mode
        self.action_mode = action_mode
        self.fit_phase_bias = bool(fit_phase_bias)
        self.calibration_bias_bound = float(calibration_bias_bound)
        self.request = request if request is not None else run_llm
        # Snapshot the exact source statistics used by validation and inference.
        self.feature_stats = copy.deepcopy(feature_stats) if feature_stats is not None else None
        # Map from expression string to its explanation text
        self.expression_explanations: Dict[str, str] = {}
        self._last_generation_failure_reason: Optional[str] = None

    def _decision_mode(self) -> str:
        if self.action_mode != "auto":
            return self.action_mode
        return "keep_switch" if str(self.teacher_model).lower() == "ecolight" else "phase_argmax"

    def generate_initial_population(
        self,
        population_size: int,
        max_retries: int = 3,
        max_total_attempts_per_individual: Optional[int] = 30,
        retry_sleep_seconds: float = 1.0,
        verbose: bool = False,
    ) -> List[str]:
        """
        Generate initial population of utility function code strings.

        Args:
            population_size: Size of initial population
            max_retries: Maximum retry attempts per batch
            max_total_attempts_per_individual: Cap on total attempts per individual. None or <=0 means unlimited.
            retry_sleep_seconds: Sleep between retries
            verbose: Whether to print progress

        Returns:
            List of full Python function code strings (def compute_utility(...): ...)
        """
        expressions = []

        for i in range(population_size):
            if verbose:
                print(f"  [{i + 1}/{population_size}] Calling LLM...", flush=True)
            total_attempts = 0
            last_reason: Optional[str] = None
            while True:
                if (
                    max_total_attempts_per_individual is not None
                    and max_total_attempts_per_individual > 0
                    and total_attempts >= max_total_attempts_per_individual
                ):
                    detail = f" Last failure: {last_reason}" if last_reason else ""
                    raise RuntimeError(
                        f"Failed to generate a valid expression for individual {i + 1}/{population_size} "
                        f"after {max_total_attempts_per_individual} total attempts.{detail}"
                    )

                # Try a small batch of retries, then loop again
                for attempt in range(max_retries):
                    total_attempts += 1
                    try:
                        expression = self._generate_single_expression()
                        if expression:
                            expressions.append(expression)
                            if verbose:
                                print(f"  [{len(expressions)}/{population_size}] OK", flush=True)
                            break
                        last_reason = (
                            self._last_generation_failure_reason
                            or "LLM response could not be parsed into a valid compute_utility function"
                        )
                    except Exception as e:
                        last_reason = str(e)
                        if self.debug_mode:
                            print(
                                f"Generation attempt {total_attempts} "
                                f"(batch attempt {attempt + 1}/{max_retries}) failed: {e}"
                            )
                        time.sleep(retry_sleep_seconds)

                if len(expressions) == i + 1:
                    break

        return expressions

    def mutate_expression(
        self,
        parent_function_code: str,
        feedback: Optional[str] = None,
        parent_explanation: Optional[str] = None,
        mutation_context: str = "evolution",
        max_retries: int = 10,
        parent_metrics: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Mutate a parent function to generate offspring.

        Args:
            parent_function_code: Full Python function code (def compute_utility(...): ...)
            feedback: Feedback from constraint validation or parent performance info
            parent_explanation: Parent function design motivation
            mutation_context: Type of mutation - "constraint_fix", "evolution", or "constraint_retry"
                - "constraint_fix": Fixing constraint violations from initial generation
                - "constraint_retry": Retrying after constraint failure in retry loop
                - "evolution": Generating offspring based on parent performance metrics
            max_retries: Maximum retry attempts
            parent_metrics: Optional dict with keys like 'action_consistency', 'performance_score'; used to add targeted guidance when action_consistency is low

        Returns:
            Mutated function code string or None if failed
        """
        self._last_mutation_failure_reason: Optional[str] = None  # for verbose logging
        for attempt in range(max_retries):
            try:
                self._last_mutation_failure_reason = None
                if mutation_context == "constraint_fix":
                    prompt = self._build_constraint_fix_prompt(
                        parent_function_code, feedback, parent_explanation=parent_explanation
                    )
                elif mutation_context == "constraint_retry":
                    prompt = self._build_constraint_retry_prompt(
                        parent_function_code, feedback, parent_explanation=parent_explanation
                    )
                elif mutation_context == "evolution":
                    prompt = self._build_evolution_mutation_prompt(
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation,
                        parent_metrics=parent_metrics,
                    )
                else:
                    # Fallback to evolution mutation
                    prompt = self._build_evolution_mutation_prompt(
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation,
                        parent_metrics=parent_metrics,
                    )

                response = self.request(
                    prompt,
                    api_endpoint=self.api_endpoint,
                    api_key=self.api_key,
                    model=self.model,
                    max_retries=3,
                )
                expr, expl = self._parse_expression_and_explanation(response)
                if expr:
                    if expl is not None:
                        self.expression_explanations[expr] = expl
                    return expr
                self._last_mutation_failure_reason = "parse returned no valid function (missing def compute_utility / return / or syntax error)"
            except Exception as e:
                self._last_mutation_failure_reason = str(e)
                if self.debug_mode:
                    print(f"Mutation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        return None

    def _generate_single_expression(self) -> Optional[str]:
        """Generate a single utility function expression"""
        self._last_generation_failure_reason = None
        prompt = self._build_initialization_prompt()
        response = self.request(
            prompt,
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            model=self.model,
            max_retries=3,
        )
        expr, expl = self._parse_expression_and_explanation(response)
        if expr is not None and expl is not None:
            self.expression_explanations[expr] = expl
        if expr is None:
            self._last_generation_failure_reason = "parse returned no valid function (missing def compute_utility / return / or syntax error)"
        return expr

    @staticmethod
    def _extract_function_only(code: str) -> str:
        """
        Keep only the compute_utility function body; strip ### Explanation and any
        trailing non-Python text that would cause 'invalid syntax' when exec() runs.
        """
        if "def compute_utility" not in code:
            return code
        lines = code.split("\n")
        def_idx = None
        for i, line in enumerate(lines):
            if "def compute_utility" in line:
                def_idx = i
                break
        if def_idx is None:
            return code
        base_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())
        end_idx = def_idx
        for i in range(def_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            # Stop before ### Explanation / ### Function (markdown sections, not Python)
            if stripped.startswith("### "):
                break
            if not stripped:
                end_idx = i
                continue
            line_indent = len(line) - len(line.lstrip())
            # Stop at first top-level (or same-indent) line that is not a comment
            if line_indent <= base_indent and not stripped.startswith("#"):
                break
            end_idx = i
        return "\n".join(lines[def_idx : end_idx + 1]).rstrip()

    def _parse_expression_and_explanation(
        self, response: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse Python function code and explanation from LLM response.

        Returns:
            Tuple of (function_code_string, explanation_string) or (None, None) if parsing fails
        """
        # Try to extract function code from markdown code blocks
        # Look for ```python ... ``` or ``` ... ```
        code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        function_code = None
        if matches:
            # Try each code block (mutation responses may have multiple; prefer one with def compute_utility)
            for candidate_raw in matches:
                candidate = candidate_raw.strip()
                if "def compute_utility" not in candidate:
                    continue
                extracted = self._extract_function_only(candidate)
                if extracted and len(extracted) >= 50 and "return" in extracted:
                    if re.search(r"\breturn\s+Q\b", extracted) and not re.search(
                        r"\bQ\s*=", extracted
                    ):
                        continue
                    try:
                        compile(extracted, "<string>", "exec")
                        function_code = extracted
                        break
                    except SyntaxError:
                        continue
            if function_code is None and matches:
                # Fallback: first block that has def compute_utility (parser will validate later)
                candidate = matches[0].strip()
                if "def compute_utility" in candidate:
                    function_code = self._extract_function_only(candidate)

        # If no fenced code block was found, extract the full unfenced function.
        # Stopping at the first blank line silently discards ordinary Python
        # functions with blank lines between statements and the final return.
        if function_code is None:
            match = re.search(r"(?m)^def compute_utility\s*\(", response)
            if match:
                function_code = self._extract_function_only(response[match.start() :])

        # Extract explanation (look for ### Explanation section)
        explanation = None
        expl_pattern = r"###\s*Explanation\s*\n(.*?)(?=\n###|\Z)"
        expl_match = re.search(expl_pattern, response, re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1).strip()
        elif function_code:
            # If no explicit explanation section, use text after the function
            parts = response.split("```", 2)
            if len(parts) >= 3:
                explanation = parts[2].strip()[:500]  # Limit length

        if function_code is None or len(function_code) < 50:
            return None, None

        # Basic validation: function should contain 'return'
        if "return" not in function_code:
            return None, None
        # If function returns Q, it must define Q (e.g. Q = ...) to avoid NameError at runtime
        if re.search(r"\breturn\s+Q\b", function_code) and not re.search(r"\bQ\s*=", function_code):
            return None, None
        # Reject if Python syntax is invalid so we don't pass bad code to FunctionUtility
        try:
            compile(function_code, "<string>", "exec")
        except SyntaxError:
            return None, None

        return function_code, explanation

    def _get_state_keys_for_teacher(self) -> List[str]:
        """
        Return the list of raw keys that will be present in state for this teacher.
        State has state['current_phase'], state['controlled_directions'], and state['features_by_direction'][direction][key].
        Only keys corresponding to observation_feature_list are populated.
        """
        if self.actual_feature_keys is not None:
            return ["current_phase"] + [k for k in self.actual_feature_keys if k != "current_phase"]
        keys_in_all: List[str] = []
        for f in self.observation_features or []:
            f_lower = (f or "").lower()
            if "pressure" in f_lower and "pressure" not in keys_in_all:
                keys_in_all.append("pressure")
            elif "num_vehicle" in f_lower and "num_vehicle" not in keys_in_all:
                keys_in_all.append("num_vehicle")
            elif "num_waiting_vehicle" in f_lower and "num_waiting_vehicle" not in keys_in_all:
                keys_in_all.append("num_waiting_vehicle")
            elif "sum_waiting_time" in f_lower or "waiting_time" in f_lower:
                if "waiting_time" not in keys_in_all:
                    keys_in_all.append("waiting_time")
                if "sum_waiting_time" not in keys_in_all:
                    keys_in_all.append("sum_waiting_time")
        # current_phase is at state top level
        return ["current_phase"] + keys_in_all

    def _build_initialization_prompt(self):
        return render(self, "initialization")

    def _build_constraint_fix_prompt(
        self, parent_function_code, constraint_violations, parent_explanation=None
    ):
        return render(
            self, "repair", parent_function_code, parent_explanation, constraint_violations
        )

    def _build_constraint_retry_prompt(
        self, parent_function_code, constraint_violations, parent_explanation=None
    ):
        return render(
            self, "repair", parent_function_code, parent_explanation, constraint_violations
        )

    def _build_evolution_mutation_prompt(
        self,
        parent_function_code,
        parent_performance_info=None,
        parent_explanation=None,
        parent_metrics=None,
    ):
        return render(
            self, "evolution", parent_function_code, parent_explanation, parent_performance_info
        )
