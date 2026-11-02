#!/usr/bin/env python3
"""Check dependencies and core method interfaces locally, without an API request."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    import cityflow  # Required simulator extension.
    import cma
    import gymnasium
    import numpy as np
    import torch
    from ppl_tsc.config import CALIBRATION, validate_config
    from ppl_tsc.evaluation.evaluator import Evaluator
    from ppl_tsc.llm.generator import LLMGenerator
    from ppl_tsc.runtime.utility import FunctionUtility
    from ppl_tsc.verification.bounds import check_division_safety

    for path in (ROOT / "configs/examples").glob("*.json"):
        validate_config(json.loads(path.read_text()))

    code = (
        "def compute_utility(state, phase_id=None, feature_stats=None):\n"
        "    return float(sum(state['features_by_direction'][d]['pressure'] "
        "for d in state['controlled_directions']))\n"
    )
    requests = []

    def fixed_response(prompt, **settings):
        requests.append(prompt)
        return "```python\n" + code + "```\nLocal interface check."

    generator = LLMGenerator(
        api_endpoint="offline.invalid",
        api_key="",
        model="local-check",
        feature_names=["pressure_raw"],
        num_phases=2,
        teacher_model="MPLight",
        observation_features=["lane_link_2_pressure"],
        actual_feature_keys=["pressure"],
        action_mode="phase_argmax",
        request=fixed_response,
    )
    generated = generator._generate_single_expression()
    require(generated is not None and len(requests) == 1, "Candidate parsing failed")
    require(check_division_safety(generated)["valid"], "Valid arithmetic rejected")
    unsafe = "def compute_utility(state):\n    return 1.0 / state['x']\n"
    require(not check_division_safety(unsafe)["valid"], "Unguarded division accepted")

    traffic = {"north": {"pressure": 0.1}, "south": {"pressure": 0.0}}
    states = {
        phase: {
            "current_phase": 0,
            "controlled_directions": [direction],
            "features_by_direction": traffic,
        }
        for phase, direction in enumerate(traffic)
    }
    loader = SimpleNamespace(
        intersection_ids={0}, get_action_consistency_data=lambda *args, **kwargs: [(states, 1)] * 8
    )
    evaluator = Evaluator(loader, 100.0, SimpleNamespace(feature_stats=None), random_seed=42)
    utility = FunctionUtility(generated)
    try:
        require(utility.select_action(states, raise_on_error=True) == 0, "Raw action mismatch")
        require(
            utility.select_action(states, phase_bias={0: 0.0, 1: 0.2}, raise_on_error=True) == 1,
            "Offsets not applied",
        )
        biases = evaluator.fit_per_intersection_bias(utility, verbose=False)
        require(
            all(abs(value) <= CALIBRATION.bias_bound for value in biases[0].values()),
            "Calibration exceeded its bound",
        )
        require(
            evaluator.calibration_details["0"]["chosen_abf"] == 1.0,
            "Calibration fixture did not match its teacher labels",
        )
        require(
            evaluator._compute_performance_score(0.1) == 1.0
            and evaluator._compute_performance_score(-0.1) < 1.0,
            "Control-score mismatch",
        )
    finally:
        utility._compiled_function.close()

    print(
        json.dumps(
            {
                "status": "passed",
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cma": cma.__version__,
                "gymnasium": gymnasium.__version__,
                "simulator": cityflow.__name__,
                "checks": [
                    "dependencies",
                    "example_configs",
                    "candidate_parsing",
                    "numeric_guard",
                    "isolated_utility",
                    "phase_offsets",
                    "CMA_calibration",
                    "capped_control_score",
                ],
                "API_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
