"""Generate a shared symbolic Q-value template with separately fitted per-phase coefficients."""
import re
from typing import Tuple, Optional, List, Dict
from agent.imitation_based.llm_integration import run_llm
from agent.imitation_based.feature_descriptions import (
    get_feature_descriptions_for_agent,
    format_features_for_prompt
)


def build_unified_q_value_prompt(
    feature_names: List[str],
    feature_descriptions: Dict[str, str],
    num_phase: int,
    agent_name: str = "RL Agent"
) -> str:
    """Build an LLM prompt for a unified Q-value expression template."""

    feature_text = format_features_for_prompt(feature_names, feature_descriptions)

    return f"""
You are a reinforcement learning expert tasked with creating a UNIVERSAL symbolic expression template for Q-value prediction in traffic signal control.

Context:
- Agent type: {agent_name}
- This expression will be used for ALL {num_phase} phases
- Each phase will have its own set of coefficients (a1_i, a2_i, ...) fitted from data
- Q-value represents the expected future cumulative reward for selecting a phase
- Higher Q-value means better long-term performance (lower travel time, less delay)

Available observation features:
{feature_text}

Your task: Create a UNIVERSAL symbolic mathematical expression template for Q-value that satisfies:

REQUIREMENTS:
1. The expression should be a template that works for all phases
2. Use coefficients a1, a2, a3, etc. (these will be optimized separately for each phase)
3. The expression should include phase-specific terms using SMOOTH/CONTINUOUS functions:
   - Use exp(- (x0 - phase_idx) * (x0 - phase_idx) ) for phase preference (peak at current phase)
   - Or use 1/(1 + (x0 - phase_idx) * (x0 - phase_idx) ) as soft phase indicator
   - Where phase_idx will be replaced by the actual phase index (0, 1, 2, ...)
4. The expression should reflect traffic control principles:
   - Higher vehicle count → generally higher Q-value (more vehicles to serve)
   - Higher pressure → generally higher Q-value (more urgent to clear queue)
   - Higher waiting time → generally higher Q-value (need to reduce delay)
   - Current phase information should influence Q-value appropriately
5. Expression must be mathematically valid:
   - No division by zero
   - No negative values in log()
   - Handle edge cases gracefully
6. Use operations: +, -, *, /, **, log, exp
7. Keep expression reasonably simple and interpretable
8. Expression should output a real number

CRITICAL SYNTAX RULES:
- DO NOT use C-style ternary operators like (condition ? val1 : val2)
- DO NOT use == comparison for discrete indicators
- Do NOT use abs, sqrt, max, min
- Use ONLY smooth/continuous functions for all conditional logic
- All expressions must be valid Python/NumPy syntax

IMPORTANT: The expression should be written as a template where:
- Coefficients a1, a2, a3, ... will be fitted separately for each phase
- Use 'phase_idx' as a placeholder for the phase index (will be replaced with 0, 1, 2, ...)
- Example: a1 * x1 + a2 * log(x1 + 1) + a3 * exp(- (x0 - phase_idx) * (x0 - phase_idx) )

Output EXACTLY in this format (no extra text before or after):
```
### Expression
Q = <your universal expression template with coefficients a1, a2, etc. and phase_idx placeholder>
### Explanation
<Brief explanation of how the template works for all phases>
```

Example format:
```
### Expression
Q = a1 * x1 + a2 * log(x0 + 1) + a3 * exp(- (x0 - phase_idx) * (x0 - phase_idx) ) + a4 / (1 + (x0 - phase_idx) * (x0 - phase_idx) )
### Explanation
The template combines pressure (x1), current phase (x0), and smooth phase-specific indicators. exp(- (x0 - phase_idx) * (x0 - phase_idx) ) peaks when current phase matches target, while 1/(1+(x0-phase_idx)*(x0-phase_idx)) provides a softer preference. Each phase will have its own coefficients fitted.
```

Output:
"""


def parse_unified_llm_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the unified expression template and explanation from an LLM response."""


    patterns = [
        r'### Expression\s*\n\s*Q\s*=\s*(.+?)(?:\n|###|$)',
        r'Q\s*=\s*(.+?)(?:\n|###|$)',
    ]

    expression = None
    for pattern in patterns:
        expr_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if expr_match:
            expression = expr_match.group(1).strip()
            break

    if expression is None:
        return None, None


    expression = re.sub(r'```.*?```', '', expression, flags=re.DOTALL)
    expression = expression.split('\n')[0].strip()
    expression = expression.replace('```', '').replace('`', '').strip()


    expression = re.sub(r'#.*$', '', expression, flags=re.MULTILINE).strip()


    expl_patterns = [
        r'### Explanation\s*\n\s*(.+?)(?:###|$)',
        r'Explanation\s*:\s*(.+?)(?:###|$)',
    ]

    explanation = None
    for pattern in expl_patterns:
        expl_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1).strip()
            break

    if explanation is None:
        explanation = "No explanation provided"

    return expression, explanation


def generate_unified_q_expression_template(
    feature_names: List[str],
    feature_descriptions: Dict[str, str],
    num_phase: int,
    agent_name: str = "RL Agent",
    max_retries: int = 3,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> Tuple[str, str]:
    """Generate a unified Q-value template for the supplied features, phase count and agent."""

    for attempt in range(max_retries):
        try:

            prompt = build_unified_q_value_prompt(
                feature_names, feature_descriptions, num_phase, agent_name
            )


            response = run_llm(
                prompt,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model=model,
                max_retries=3
            )


            expression, explanation = parse_unified_llm_response(response)

            if expression is None:
                print(f"  Attempt {attempt + 1}: Failed to parse LLM response")
                if max_retries > 1:
                    print(f"  Response preview: {response[:200]}...")
                continue

            # Disallow unsupported operators
            forbidden_tokens = ['abs', 'sqrt', 'max', 'min']
            if any(re.search(rf'\b{tok}\b', expression) for tok in forbidden_tokens):
                print(f"  Attempt {attempt + 1}: Expression contains forbidden operators")
                continue


            if len(expression) < 5:
                print(f"  Attempt {attempt + 1}: Expression too short: {expression}")
                continue


            if not any(coef in expression for coef in ['a1', 'a2', 'a3', 'a4', 'a5']):
                print(f"  Attempt {attempt + 1}: Expression missing coefficients")
                continue


            if 'phase_idx' not in expression:
                print(f"  Attempt {attempt + 1}: Expression missing phase_idx placeholder")

                print(f"  Warning: Expression does not contain phase_idx, will add phase-specific terms automatically")

            return expression, explanation

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise

    raise RuntimeError(f"Failed to generate valid unified expression template after {max_retries} attempts")

