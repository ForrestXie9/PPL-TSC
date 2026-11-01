"""
LLM Generator Module
Generate and mutate utility function expressions using LLM
"""
import re
import time
import copy
import json
from typing import Optional, List, Dict, Tuple
from agent.imitation_based.llm_integration import run_llm


from LLM_SR_DISTILLATION.semantic_checks import contract_text


class LLMGenerator:
    """LLM-based utility function generator"""
    
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
        binary_margin_guidance: bool = False,
        calibration_bias_bound: float = 10.0,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        actual_feature_keys: Optional[List[str]] = None,
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
            action_mode: Policy decision rule. ``auto`` preserves the legacy
                EcoLight keep/switch behavior; otherwise explicitly choose
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
                "action_mode must be one of: auto, phase_argmax, "
                "keep_switch, binary_keep_switch"
            )
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.feature_names = feature_names
        self.num_phases = num_phases
        self.teacher_model = teacher_model
        self.observation_features = list(observation_features or [])
        self.actual_feature_keys = list(actual_feature_keys) if actual_feature_keys is not None else None
        self.debug_mode = debug_mode
        self.action_mode = action_mode
        self.binary_margin_guidance = bool(binary_margin_guidance)
        self.calibration_bias_bound = float(calibration_bias_bound)
        # Snapshot the exact source statistics used by validation and inference.
        self.feature_stats = copy.deepcopy(feature_stats) if feature_stats is not None else None
        # Map from expression string to its explanation text
        self.expression_explanations: Dict[str, str] = {}
        self._last_generation_failure_reason: Optional[str] = None

    def _decision_mode(self) -> str:
        if self.action_mode != "auto":
            return self.action_mode
        return "keep_switch" if str(self.teacher_model).lower() == "ecolight" else "phase_argmax"

    def _decision_mode_text(self) -> str:
        if self._decision_mode() == "binary_keep_switch":
            return """DECISION MODE:
- EcoLight-native aligned binary action values.
- The controller evaluates the SAME global traffic state twice:
  1) phase_id=0 means the binary action KEEP
  2) phase_id=1 means the binary action SWITCH to the cyclic next phase
- compute_utility returns Q_keep(s) or Q_switch(s), not a traffic-phase score.
- During inference, the controller switches iff Q_switch(s) > Q_keep(s)."""
        if self._decision_mode() == "keep_switch":
            return """DECISION MODE:
- EcoLight-style binary decision.
- The controller does NOT rank all phases globally.
- Instead, it only compares:
  1) keep: stay on the current active phase
  2) switch: move to the cyclic next phase
- Concretely, if current_phase = c, then switch target is the next phase in cyclic order.
- Therefore, compute_utility should produce meaningful values especially for:
  - the current phase (keep)
  - the next phase (switch)
- During inference, if Q(next_phase) > Q(current_phase), the controller switches; otherwise it keeps the current phase."""
        return """DECISION MODE:
- Standard all-phases argmax.
- The controller evaluates every candidate phase and selects the phase with the highest utility Q."""
    
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
                if (max_total_attempts_per_individual is not None
                        and max_total_attempts_per_individual > 0
                        and total_attempts >= max_total_attempts_per_individual):
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
                        last_reason = self._last_generation_failure_reason or "LLM response could not be parsed into a valid compute_utility function"
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
        parent_metrics: Optional[Dict] = None
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
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation
                    )
                elif mutation_context == "constraint_retry":
                    prompt = self._build_constraint_retry_prompt(
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation
                    )
                elif mutation_context == "evolution":
                    prompt = self._build_evolution_mutation_prompt(
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation,
                        parent_metrics=parent_metrics
                    )
                else:
                    # Fallback to evolution mutation
                    prompt = self._build_evolution_mutation_prompt(
                        parent_function_code,
                        feedback,
                        parent_explanation=parent_explanation,
                        parent_metrics=parent_metrics
                    )
                
                response = run_llm(
                    prompt,
                    api_endpoint=self.api_endpoint,
                    api_key=self.api_key,
                    model=self.model,
                    max_retries=3
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
        response = run_llm(
            prompt,
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            model=self.model,
            max_retries=3
        )
        expr, expl = self._parse_expression_and_explanation(response)
        if expr is not None and expl is not None:
            self.expression_explanations[expr] = expl
        if expr is None:
            self._last_generation_failure_reason = (
                "parse returned no valid function (missing def compute_utility / return / or syntax error)"
            )
        return expr

    def _build_data_contract_text(self) -> str:
        """Build a shared DATA CONTRACT block to keep prompts consistent."""
        if self._decision_mode() == "binary_keep_switch":
            return self._build_binary_data_contract_text()

        feature_list = ", ".join(self.feature_names)
        statistics_keys = ", ".join(self.feature_stats) if self.feature_stats is not None else feature_list
        feature_semantics = self._build_feature_semantics_text()
        allowed_state_keys = self._get_state_keys_for_teacher()
        allowed_raw_feature_keys = [k for k in allowed_state_keys if k != "current_phase"]
        allowed_raw_feature_keys_text = ", ".join(allowed_raw_feature_keys) if allowed_raw_feature_keys else "(none)"

        return f"""DATA CONTRACT:

{self._decision_mode_text()}

INPUT PARAMETERS (compute_utility):
- state: dict for the phase being evaluated. Keys: current_phase (int), controlled_directions (list), features_by_direction (dict).
- phase_id: int or None. The phase ID being evaluated. Use for phase_active: 1 if state.get("current_phase")==phase_id else 0. Do not hard-gate (non-current phases can still get non-zero Q).
- feature_stats: dict or None. {{name: {{"min","max","mean","std"}}}} for normalization. Check "if feature_stats and 'X' in feature_stats" before use.

STATE LAYOUT:
- Top-level state keys (always available): current_phase, controlled_directions, features_by_direction
- Top-level state layout:
  - state["current_phase"]: int
  - state["controlled_directions"]: list[str] of directions controlled by the evaluated phase
  - state["features_by_direction"]: dict[str, dict[str, float]] of per-direction measurements
- Per-direction raw feature keys from teacher (STRICT): {allowed_raw_feature_keys_text}
- Per-direction access pattern: state["features_by_direction"][<direction>][<raw_key>]
- Teacher raw key summary: {", ".join(allowed_state_keys)}
- Statistics aliases (not state fields): {feature_list}
- Runtime types: current_phase and phase_id are int; controlled_directions is list[str]; features_by_direction is dict[str, dict[str, float]].
- Per-direction num_vehicle, num_waiting_vehicle, waiting_time and sum_waiting_time are nonnegative floats; pressure is a signed float. Values are averaged within each direction before a phase aggregates them.
- No derived phase_features field is injected. Normalization is optional and, if useful, must be implemented inside compute_utility by the generated program.
- feature_stats contains frozen SOURCE per-direction measurement statistics: min/max are extrema, mean is the arithmetic mean, std is the standard deviation. These are NOT statistics of summed phase aggregates. Applying them to an aggregate or transformed value is reference scaling, not exact normalization of that distribution.
- Names ending in _raw and _norm are legacy statistics aliases for the SAME underlying raw measurement distribution. The _norm suffix does not supply a normalized input or describe an already normalized distribution.
- Source min/max do not bound unseen target traffic. With fixed source Min-Max statistics, target values can map below 0 or above 1. Do not recompute statistics from the target scene.
- The available feature_stats keys are exactly: {statistics_keys or '(none)'}. Raw state keys such as num_vehicle are a different namespace; never use an absent statistics key.
{self._source_statistics_text()}
- Equality and saturation are allowed by weak monotonicity; use source performance evidence to assess their consequences. No minimum response magnitude or forced switching is required.
- feature_stats (optional): dict {{name: {{\"min\",\"max\",\"mean\",\"std\"}}}}. NOTE: phase_active is binary (0 or 1), do NOT normalize it.
{self._normalization_example_text(allowed_raw_feature_keys)}
{contract_text()}
- Feature semantics:
{feature_semantics}

DIVISION SAFETY (MANDATORY - ZeroDivisionError will fail validation):
- Never use division with an unproven denominator; it can be 0 for some states (e.g., empty lanes, zero std).
- A statically proven nonzero denominator is allowed. Otherwise use a guard, for example:
  1) Guarded: if denom > 1e-8: x = a / denom else: x = 0.0
  2) Magnitude normalization: x = a / max(abs(denom), 1e-8)
  3) Max: x = a / max(denom, 1e-8)  or  x = a / np.maximum(denom, 1e-8)
- Pressure and derived sums may be NEGATIVE. Adding 1 or epsilon does not ensure a nonzero denominator.
- Preserve signed ratios with: x = a / denom if abs(denom) > 1e-8 else 0.0.
- Use magnitude normalization only when intended; max(abs(denom),1e-8) changes signed ratios.
- These guards also apply to floor division, remainder, reciprocal and negative powers, including NumPy calls. A finite final fallback does not excuse invalid intermediate arithmetic.
- sqrt requires a nonnegative argument; log requires a positive argument; fractional powers of negative values are invalid in real-valued utility. Guard the complete transformed expression, including out-of-source normalized values.
- Examples: (val - p_min) / max(abs(p_max - p_min), 1e-8); x / max(abs(std), 1e-8); total / max(cnt, 1e-8)"""

    def _source_statistics_text(self) -> str:
        if self.feature_stats is None:
            return '- Source statistic values were not supplied to this generator; do not invent observed ranges. Guard optional statistics access.'
        import os
        from stats_extension import DESCRIPTION, BINARY_DESCRIPTION
        from ppl_tsc_current.profile import SHARED_LANE_DESCRIPTION
        description = ''
        if os.environ.get('PPL_METHOD_PROFILE') == 'full_100_256_v1':
            description = (BINARY_DESCRIPTION if self._decision_mode() == 'binary_keep_switch' else DESCRIPTION) + '\n' + SHARED_LANE_DESCRIPTION + '\n'
        return description + '- Frozen source feature_stats values (no target data):\n' + json.dumps(
            self.feature_stats, sort_keys=True, indent=2, allow_nan=False)

    def _normalization_example_text(self, raw_keys) -> str:
        aliases = [('num_vehicle', 'vehicles_raw'), ('num_waiting_vehicle', 'waiting_veh_raw'),
                   ('waiting_time', 'waiting_time_raw'), ('sum_waiting_time', 'waiting_time_raw'),
                   ('pressure', 'pressure_raw')]
        available = self.feature_stats if self.feature_stats is not None else self.feature_names
        for raw, alias in aliases:
            if raw in raw_keys and alias in available:
                return f'''- Optional normalization example for raw field {raw} and statistics alias {alias}:
  stat = feature_stats.get("{alias}", {{}}) if feature_stats else {{}}
  for direction in state.get("controlled_directions", []):
      val = float(state.get("features_by_direction", {{}}).get(direction, {{}}).get("{raw}", 0.0))
      norm = val  # Explicit raw fallback for missing statistics or constant source range.
      if "min" in stat and "max" in stat:
          span = float(stat["max"]) - float(stat["min"])
          if span > 1e-8:
              norm = (val - float(stat["min"])) / span
  Source-range normalization may produce negative values or values above 1. This example addresses numerical access only; the full utility must also pass semantic constraints.'''
        return '- No matching raw-feature/statistics pair is available for a normalization example. Use the documented raw inputs; do not normalize phase_active.'

    def _build_binary_data_contract_text(self) -> str:
        """Data contract for an EcoLight-aligned two-action symbolic head."""
        return f"""DATA CONTRACT:

{self._decision_mode_text()}

INPUT PARAMETERS (compute_utility):
- state: one global EcoLight state shared by both binary actions.
- phase_id: the BINARY ACTION ID, not a traffic-phase ID:
  - 0 = KEEP the current traffic phase
  - 1 = SWITCH to the cyclic next traffic phase
- feature_stats: dict[str, dict[str, float]] or None, with optional source-derived statistics keyed by green_vehicle and red_vehicle.
- Each statistics entry contains min, max, mean and std of the corresponding exact source binary input. Normalization is optional and is implemented by the generated function if useful. Keep these source statistics frozen during transfer; unseen values can exceed the source range.
{self._source_statistics_text()}

STATE LAYOUT (STRICT):
- state["current_phase"]: int current traffic-phase ID (context only).
- state["next_phase"]: int cyclic successor traffic-phase ID (context only).
- state["binary_features"]["green_vehicle"]: nonnegative float, exact total number of vehicles
  on lanes currently receiving green, matching the EcoLight teacher input.
- state["binary_features"]["red_vehicle"]: nonnegative float, exact total number of vehicles
  on lanes currently receiving red, matching the EcoLight teacher input.

Use only the keys listed above. In particular, do not treat phase_id as a
traffic phase and do not access phase-local controlled_directions.

{contract_text(binary=True)}

PHYSICAL CONSISTENCY:
- Increasing green_vehicle should not reduce the preference for KEEP,
  Q_keep - Q_switch.
- Increasing red_vehicle should not reduce the preference for SWITCH,
  Q_switch - Q_keep.

DIVISION SAFETY:
- Never use an unguarded denominator.
- Use a/max(abs(b),1e-8), a/max(b,1e-8), or an explicit positive-denominator guard.
- Guard sqrt/log/power inputs and always return a finite float."""

    def _build_binary_prompt(
        self,
        *,
        stage: str,
        parent_function_code: Optional[str] = None,
        feedback: Optional[str] = None,
        parent_explanation: Optional[str] = None,
    ) -> str:
        """Build initialization, repair, retry, and evolution prompts."""
        stage_text = {
            "initialization": (
                "Generate an initial EcoLight-native binary student function. "
                "Explore a clear, non-trivial symbolic structure."
            ),
            "constraint_fix": (
                "Rewrite the function to fix every reported validation "
                "failure while preserving the aligned binary semantics."
            ),
            "constraint_retry": (
                "Produce a structurally different corrected binary function; "
                "the previous repair still failed."
            ),
            "evolution": (
                "Generate a structurally different offspring that improves "
                "the joint behavioral-fidelity and traffic-control score."
            ),
        }[stage]
        parent_block = ""
        if parent_function_code:
            parent_block = f"""
CURRENT/PARENT FUNCTION:
```python
{parent_function_code}
```

CURRENT/PARENT EXPLANATION:
{parent_explanation or "No explanation available"}
"""
        feedback_block = ""
        if feedback:
            feedback_block = f"""
VALIDATION OR PERFORMANCE FEEDBACK:
{feedback}
"""
        margin_guidance = ""
        if self.binary_margin_guidance:
            margin_guidance = f"""
CALIBRATION SCALE (use the existing action and calibration interface):
- After this function is generated, a separate calibration may add a
  phase-specific bias in [-{self.calibration_bias_bound:g},
  {self.calibration_bias_bound:g}] to each action value. The selected action
  depends on Q_switch - Q_keep plus the difference of the two phase biases.
- Keep the difference between Q_switch and Q_keep responsive to meaningful
  changes in red and green vehicle counts. Avoid normalizing both scores by
  the same total into a tiny fixed interval that phase biases can dominate.
- Preserve numerical safety and physical consistency. Do not simply multiply
  both scores by an arbitrary large constant; give the score scale a clear
  traffic interpretation.
"""
        return f"""{stage_text}

{parent_block}
{feedback_block}
{self._build_binary_data_contract_text()}
{margin_guidance}

OBJECTIVE:
- Implement the two EcoLight action values Q_keep(s) and Q_switch(s) in one
  readable function. Larger Q means the binary action is preferred.
- Use both exact green/red vehicle counts and make phase_id=0 and phase_id=1
  meaningfully different.

HARD REQUIREMENTS:
1. Signature exactly:
   def compute_utility(state, phase_id=None, feature_stats=None):
2. Read traffic only from state.get("binary_features", {{}}), using only
   green_vehicle and red_vehicle with numeric defaults.
3. phase_id is the action ID: 0=KEEP and 1=SWITCH. Branching on it is required.
4. Preserve physical consistency: more green demand must not lower KEEP
   preference; more red demand must not lower SWITCH preference.
5. Use safe normalization and numeric operations. Never divide by an
   unguarded denominator.
6. Set Q explicitly and return:
   float(Q) if np.isfinite(Q) else 0.0
7. Keep the code concise and human-readable.

OUTPUT FORMAT (STRICT):
- Reply with exactly one Python code block containing ### Function, the full
  function, then ### Explanation. Do not add text outside it.

```python
### Function
def compute_utility(state, phase_id=None, feature_stats=None):
    import numpy as np
    binary = state.get("binary_features", {{}})
    green = float(binary.get("green_vehicle", 0.0))
    red = float(binary.get("red_vehicle", 0.0))
    action_id = 0 if phase_id is None else int(phase_id)
    Q = 0.0
    # Compute Q_keep when action_id == 0 and Q_switch otherwise.
    return float(Q) if np.isfinite(Q) else 0.0

### Explanation
<brief explanation of the KEEP/SWITCH logic and traffic meaning>
```"""

    def _build_initialization_prompt(self) -> str:
        """Build a concise initialization prompt for compute_utility."""
        if self._decision_mode() == "binary_keep_switch":
            return self._build_binary_prompt(stage="initialization")

        data_contract = self._build_data_contract_text()

        return f"""Generate an initial STUDENT utility function for traffic signal control, distilled from teacher model `{self.teacher_model}`.

{data_contract}

OBJECTIVE:
- Implement `compute_utility(state, phase_id=None, feature_stats=None)` that returns scalar Q for one phase state.
- Controller action selection follows the DECISION MODE above. Larger Q means more preferred.
- Prioritize exploratory structure design in initialization to increase search-space coverage.

HARD REQUIREMENTS:
1. Signature must be exactly: `def compute_utility(state, phase_id=None, feature_stats=None):`
2. Use safe access (`.get`) with numeric defaults; initialize `Q = 0.0`.
3. Aggregate traffic only over `state["controlled_directions"]` for this evaluated phase.
4. STRICT KEY CONSTRAINT: use only raw keys from the allowed list above. Do not invent keys (e.g., queue/delay if not listed).
5. Do not hard-gate by current phase (do not force non-current phases to zero). A current-phase bonus may express a holding preference; the verifier imposes no arbitrary bonus-size limit.
6. If normalization is needed, use `feature_stats` safely. Division: prove the denominator nonzero or guard it. Use `a/max(abs(b),1e-8)` or `if b>1e-8: x=a/b else: x=0`.
7. Safe numeric ops: sqrt use `np.sqrt(np.maximum(x, 0.0))`; log use `np.log(np.maximum(x, 1e-8))`; power avoid negative**0.5 (use abs/max(0,x) first).
8. Always return `float(Q) if np.isfinite(Q) else 0.0`; never return None.
9. Q must be phase-discriminative (different phases can get different values under same global situation).
   - For EcoLight/keep-switch mode, this especially means separating the current phase from the cyclic next phase.
10. Weak monotonicity: when traffic load of a phase increases (e.g., pressure / num_vehicle / num_waiting_vehicle / waiting_time / sum_waiting_time on its controlled lanes), Q for that phase should not significantly decrease under otherwise similar conditions.
11. Prioritize exploration coverage: prefer non-trivial feature combinations and avoid overly simplistic single-signal heuristics.
12. Code style: write clear, concise, and readable code structure (simple control flow, no unnecessary complexity).

OUTPUT FORMAT (STRICT):
- Reply with exactly ONE code block.
- The block must contain `### Function`, full function, then `### Explanation`.

```python
### Function
def compute_utility(state, phase_id=None, feature_stats=None):
    import numpy as np
    Q = 0.0
    fbd = state.get('features_by_direction', dict())
    controlled = state.get('controlled_directions', [])
    # Division: prove a nonzero denominator or guard it; preserve its intended sign.
    # compute Q using only allowed raw keys from fbd[d][key]
    return float(Q) if np.isfinite(Q) else 0.0

### Explanation
<brief 2-3 sentence explanation of signals used and aggregation logic>
```

Do not output any extra text outside the code block."""

    def _build_constraint_fix_prompt(
        self,
        parent_function_code: str,
        constraint_violations: Optional[str] = None,
        parent_explanation: Optional[str] = None
    ) -> str:
        """
        Build prompt for fixing constraint violations during initial generation.
        Used when a function fails constraints during initialization phase.
        """
        if self._decision_mode() == "binary_keep_switch":
            return self._build_binary_prompt(
                stage="constraint_fix",
                parent_function_code=parent_function_code,
                feedback=constraint_violations,
                parent_explanation=parent_explanation,
            )

        data_contract = self._build_data_contract_text()

        prompt = f"""Your initial function failed constraint validation. Rewrite it to satisfy constraints.

Current function:
```python
{parent_function_code}
```

Current explanation:
{parent_explanation or "No explanation available"}

{data_contract}

Function I/O:
- Signature: `def compute_utility(state, phase_id=None, feature_stats=None):`
- Inputs:
  - state: dict, raw teacher state for the evaluated phase
  - phase_id: optional phase identifier (do not use as hard gate)
  - feature_stats: dict, optional per-feature min/max/mean/std for Min-Max or Z-score normalization
- Output: float Q (must be finite)

CONSTRAINT VIOLATIONS TO FIX:
{constraint_violations or "Generic: Q must be finite; use safe numeric ops; guard or prove nonzero denominators; avoid log(<=0)."}

        HARD REQUIREMENTS:
1. Keep signature exactly: `def compute_utility(state, phase_id=None, feature_stats=None):`
2. Use safe access (`.get`) with numeric defaults; initialize `Q = 0.0`.
3. Aggregate traffic only over `state["controlled_directions"]` for the evaluated phase.
4. STRICT KEY CONSTRAINT: use only raw keys listed in DATA CONTRACT; do not invent keys.
5. Do not hard-gate by current phase (do not force non-current phases to zero).
6. Safe numeric ops: sqrt use `np.sqrt(np.maximum(x, 0.0))`; log use `np.log(np.maximum(x, 1e-8))`; division must have a nonzero denominator; possible guards include `a/max(abs(b),1e-8)` or `max(b,1e-8)`.
7. Weak monotonicity: when traffic load of a phase increases (e.g., pressure / num_vehicle / num_waiting_vehicle / waiting_time / sum_waiting_time), Q for that phase should not significantly decrease under otherwise similar conditions.
8. Always return `float(Q) if np.isfinite(Q) else 0.0`; never return None.
9. Keep explanation non-empty and describe what was changed to fix violations.
10. Code style: write clear, concise, and readable code structure (simple control flow, no unnecessary complexity).

OUTPUT FORMAT (STRICT):
- Reply with exactly ONE code block.
- The block must contain `### Function`, full function, then `### Explanation`.

```python
### Function
def compute_utility(state, phase_id=None, feature_stats=None):
    import numpy as np
    Q = 0.0
    fbd = state.get('features_by_direction', dict())
    controlled = state.get('controlled_directions', [])
    # Fix violations while using only allowed raw keys from fbd[d][key]
    return float(Q) if np.isfinite(Q) else 0.0

### Explanation
<brief explanation of changes made to satisfy constraint violations (1-3 sentences, non-empty)>
```

Do not output any extra text outside the code block."""

        return prompt
    
    def _build_constraint_retry_prompt(
        self,
        parent_function_code: str,
        constraint_violations: Optional[str] = None,
        parent_explanation: Optional[str] = None
    ) -> str:
        """
        Build prompt for retrying constraint fixes (up to max_constraint_retries).
        Used when a function fails constraints during the retry loop.
        """
        if self._decision_mode() == "binary_keep_switch":
            return self._build_binary_prompt(
                stage="constraint_retry",
                parent_function_code=parent_function_code,
                feedback=constraint_violations,
                parent_explanation=parent_explanation,
            )

        data_contract = self._build_data_contract_text()

        prompt = f"""Your function still failed constraint validation. This is a retry attempt; produce a different corrected variant.

Previous function:
```python
{parent_function_code}
```

Previous explanation:
{parent_explanation or "No explanation available"}

{data_contract}

Function I/O:
- Signature: `def compute_utility(state, phase_id=None, feature_stats=None):`
- Inputs: state (raw teacher state), phase_id (optional), feature_stats (optional)
- Output: float Q (must be finite)

STILL-FAILING CONSTRAINTS (MUST FIX):
{constraint_violations or "Generic constraints: boundary conditions, numerical stability, extreme values"}

HARD REQUIREMENTS:
1. Keep signature exactly: `def compute_utility(state, phase_id=None, feature_stats=None):`
2. This retry must be structurally different from previous function (not only coefficient tweaks).
3. Use safe access (`.get`) with numeric defaults; initialize `Q = 0.0`.
4. STRICT KEY CONSTRAINT: use only raw keys listed in DATA CONTRACT.
5. Use meaningful combinations of available state signals (avoid trivial one-key heuristic if possible).
6. Safe numeric ops: sqrt use `np.sqrt(np.maximum(x, 0.0))`; log use `np.log(np.maximum(x, 1e-8))`; division must have a nonzero denominator; possible guards include `a/max(abs(b),1e-8)` or `max(b,1e-8)`.
7. Weak monotonicity: when traffic load of a phase increases (e.g., pressure / num_vehicle / num_waiting_vehicle / waiting_time / sum_waiting_time), Q for that phase should not significantly decrease under otherwise similar conditions.
8. Do not hard-gate by current phase (do not force non-current phases to zero).
9. Always return `float(Q) if np.isfinite(Q) else 0.0`; never return None.
10. Keep explanation non-empty and clearly state what changed vs previous retry.
11. Code style: write clear, concise, and readable code structure (simple control flow, no unnecessary complexity).

OUTPUT FORMAT (STRICT):
- Reply with exactly ONE code block.
- The block must contain `### Function`, full function, then `### Explanation`.

```python
### Function
def compute_utility(state, phase_id=None, feature_stats=None):
    import numpy as np
    Q = 0.0
    fbd = state.get('features_by_direction', dict())
    controlled = state.get('controlled_directions', [])
    # Different structure from previous while satisfying all constraints
    return float(Q) if np.isfinite(Q) else 0.0

### Explanation
<brief explanation of what changed and why it addresses the failing constraints (1-3 sentences, non-empty)>
```

Do not output any extra text outside the code block."""

        return prompt
    
    def _build_evolution_mutation_prompt(
        self,
        parent_function_code: str,
        parent_performance_info: Optional[str] = None,
        parent_explanation: Optional[str] = None,
        parent_metrics: Optional[Dict] = None
    ) -> str:
        """
        Build prompt for evolution-phase mutation based on parent performance metrics.
        Used when generating offspring in the evolution loop.
        """
        if self._decision_mode() == "binary_keep_switch":
            return self._build_binary_prompt(
                stage="evolution",
                parent_function_code=parent_function_code,
                feedback=parent_performance_info,
                parent_explanation=parent_explanation,
            )

        data_contract = self._build_data_contract_text()

        prompt = f"""You are evolving a utility function for traffic signal control.

Parent function (selected from high-performing population):
```python
{parent_function_code}
```

Parent design motivation:
{parent_explanation or "No explanation available"}

Parent performance information:
{parent_performance_info or "No performance metrics available"}

{data_contract}

OBJECTIVE:
Generate an improved offspring function that:
1. Prioritizes exploring better scoring structures using the provided state signals.
2. Uses meaningful combinations of available raw features (not only trivial single-feature heuristics).
3. Fixes listed constraint violations if present in parent performance information.
4. Is structurally different from parent (not trivial parameter tweaks).
5. Respects the DECISION MODE above.

HARD REQUIREMENTS:
1. Signature must be: def compute_utility(state, phase_id=None, feature_stats=None)
2. Use only the documented raw keys in features_by_direction.
3. Use safe access (.get), initialize Q = 0.0, and always return float(Q) if np.isfinite(Q) else 0.0.
4. Normalization is optional. Raw direction values may be used directly; if scaling is useful, implement it inside the function using matching source per-direction feature_stats keys. Do not assume normalized values or phase_features are supplied.
5. Safe numeric ops: sqrt use `np.sqrt(np.maximum(x, 0.0))`; log use `np.log(np.maximum(x, 1e-8))`; division must have a nonzero denominator; possible guards include `a/max(abs(b),1e-8)` or `max(b,1e-8)`.
6. Weak monotonicity: when traffic load of a phase increases (e.g., pressure / num_vehicle / num_waiting_vehicle / waiting_time / sum_waiting_time), Q for that phase should not significantly decrease under otherwise similar conditions.
7. Do not hard-gate by current phase (do not force non-current phases to zero).
8. Code style: write clear, concise, and readable code structure (simple control flow, no unnecessary complexity).

Output format (strict):
- Reply with exactly ONE code block.
- The block must contain ### Function, the full compute_utility, then ### Explanation.

```python
### Function
def compute_utility(state, phase_id=None, feature_stats=None):
    import numpy as np
    Q = 0.0
    # Use only allowed raw keys and controlled_directions. Division: a/max(abs(b),1e-8) or max(b,1e-8).
    # ... your implementation ...
    return float(Q) if np.isfinite(Q) else 0.0

### Explanation
<brief explanation of how this variant improves upon or differs from the parent (1-3 sentences, non-empty)>
```

Do not add any extra sections. Do not output the parent code again."""
        return prompt

    def _build_mutation_prompt(
        self,
        parent_function_code: str,
        constraint_feedback: Optional[str] = None,
        parent_explanation: Optional[str] = None
    ) -> str:
        """
        (Deprecated) Build generic prompt for function mutation.

        Prefer: _build_constraint_fix_prompt, _build_constraint_retry_prompt,
        or _build_evolution_mutation_prompt depending on context.
        """
        return self._build_evolution_mutation_prompt(
            parent_function_code,
            parent_performance_info=constraint_feedback,
            parent_explanation=parent_explanation
        )
    
    @staticmethod
    def _extract_function_only(code: str) -> str:
        """
        Keep only the compute_utility function body; strip ### Explanation and any
        trailing non-Python text that would cause 'invalid syntax' when exec() runs.
        """
        if 'def compute_utility' not in code:
            return code
        lines = code.split('\n')
        def_idx = None
        for i, line in enumerate(lines):
            if 'def compute_utility' in line:
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
            if stripped.startswith('### '):
                break
            if not stripped:
                end_idx = i
                continue
            line_indent = len(line) - len(line.lstrip())
            # Stop at first top-level (or same-indent) line that is not a comment
            if line_indent <= base_indent and not stripped.startswith('#'):
                break
            end_idx = i
        return '\n'.join(lines[def_idx:end_idx + 1]).rstrip()
    
    def _parse_expression_and_explanation(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse Python function code and explanation from LLM response.
        
        Returns:
            Tuple of (function_code_string, explanation_string) or (None, None) if parsing fails
        """
        # Try to extract function code from markdown code blocks
        # Look for ```python ... ``` or ``` ... ```
        code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        function_code = None
        if matches:
            # Try each code block (mutation responses may have multiple; prefer one with def compute_utility)
            for candidate_raw in matches:
                candidate = candidate_raw.strip()
                if 'def compute_utility' not in candidate:
                    continue
                extracted = self._extract_function_only(candidate)
                if extracted and len(extracted) >= 50 and 'return' in extracted:
                    if re.search(r'\breturn\s+Q\b', extracted) and not re.search(r'\bQ\s*=', extracted):
                        continue
                    try:
                        compile(extracted, '<string>', 'exec')
                        function_code = extracted
                        break
                    except SyntaxError:
                        continue
            if function_code is None and matches:
                # Fallback: first block that has def compute_utility (parser will validate later)
                candidate = matches[0].strip()
                if 'def compute_utility' in candidate:
                    function_code = self._extract_function_only(candidate)
        
        # If no fenced code block was found, extract the full unfenced function.
        # Stopping at the first blank line silently discards ordinary Python
        # functions with blank lines between statements and the final return.
        if function_code is None:
            match = re.search(r'(?m)^def compute_utility\s*\(', response)
            if match:
                function_code = self._extract_function_only(response[match.start():])
        
        # Extract explanation (look for ### Explanation section)
        explanation = None
        expl_pattern = r'###\s*Explanation\s*\n(.*?)(?=\n###|\Z)'
        expl_match = re.search(expl_pattern, response, re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1).strip()
        elif function_code:
            # If no explicit explanation section, use text after the function
            parts = response.split('```', 2)
            if len(parts) >= 3:
                explanation = parts[2].strip()[:500]  # Limit length
        
        if function_code is None or len(function_code) < 50:
            return None, None
        
        # Basic validation: function should contain 'return'
        if 'return' not in function_code:
            return None, None
        # If function returns Q, it must define Q (e.g. Q = ...) to avoid NameError at runtime
        if re.search(r'\breturn\s+Q\b', function_code) and not re.search(r'\bQ\s*=', function_code):
            return None, None
        # Reject if Python syntax is invalid so we don't pass bad code to FunctionUtility
        try:
            compile(function_code, '<string>', 'exec')
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
            return ['current_phase'] + [k for k in self.actual_feature_keys if k != 'current_phase']
        keys_in_all: List[str] = []
        for f in self.observation_features or []:
            f_lower = (f or "").lower()
            if "pressure" in f_lower and "pressure" not in keys_in_all:
                keys_in_all.append("pressure")
            elif "num_vehicle" in f_lower and "num_vehicle" not in keys_in_all:
                keys_in_all.append("num_vehicle")
            elif "num_waiting_vehicle" in f_lower and "num_waiting_vehicle" not in keys_in_all:
                keys_in_all.append("num_waiting_vehicle")
            elif ("sum_waiting_time" in f_lower or "waiting_time" in f_lower):
                if "waiting_time" not in keys_in_all:
                    keys_in_all.append("waiting_time")
                if "sum_waiting_time" not in keys_in_all:
                    keys_in_all.append("sum_waiting_time")
        # current_phase is at state top level
        return ["current_phase"] + keys_in_all

    @staticmethod
    def _truncate_code_at_line(code: str, max_chars: int = 600) -> str:
        """Truncate code at a line boundary so the snippet is valid and readable."""
        if len(code) <= max_chars:
            return code
        head = code[:max_chars]
        last_nl = head.rfind("\n")
        if last_nl > max_chars // 2:
            return head[: last_nl + 1]
        return head

    def _build_feature_semantics_text(self) -> str:
        """
        Build human-readable semantics for feature variables based on observation_features.

        We mirror the logic in feature_extractor.build_feature_extractor:
        - phase_active is a BINARY indicator (0 or 1): 1 if the evaluated phase is currently executing, 0 otherwise.
        - Other features (pressure_norm, vehicles_norm, ...) correspond to aggregated traffic features.
        - Note: phase_active is NOT a phase index. It's a boolean flag for the evaluated phase.
        """
        meanings = {
            'pressure': 'signed traffic pressure per direction; larger pressure must not lower utility',
            'vehicles': 'number of vehicles per direction',
            'waiting_veh': 'number of waiting vehicles per direction',
            'waiting_time': 'waiting time per direction',
        }
        lines = []
        for name in self.feature_names:
            if name == 'phase_active':
                lines.append('- phase_active: binary indicator of phase_id == current_phase; do not normalize it. Its statistics describe declared binary support, not observed phase frequencies.')
                continue
            base = name.removesuffix('_raw').removesuffix('_norm')
            meaning = meanings.get(base)
            if meaning:
                lines.append(f'- {name}: {meaning}; statistics alias for raw per-direction measurements, not a supplied state field.')
            else:
                lines.append(f'- {name}: available source-statistics alias; not a raw state key. Use only documented measured fields.')
        return '\n'.join(lines)
