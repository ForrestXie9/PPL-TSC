"""Authoritative method settings and validation; no runtime class replacement."""

from dataclasses import dataclass
import json
from pathlib import Path

PROFILE = "full"
PROMPT_TEMPLATE = "paper_figure_a2"
DIRECTION_SEMANTICS = "compatible_shared_lane_mean"


@dataclass(frozen=True)
class CalibrationSettings:
    population_size: int = 20
    max_iterations: int = 20
    bias_bound: float = 0.5
    sigma_ratio: float = 1.0 / 3.0
    tie_rule: str = "least_changes"


CALIBRATION = CalibrationSettings()


def source_run_settings(entry, *, method_profile=None):
    """Use the same path, seed and override rules in preflight and execution."""
    if not isinstance(entry, dict):
        raise ValueError("Each source run must be a JSON object")
    result = dict(entry)
    result["teacher_data_path"] = entry.get("teacher_data_path") or entry.get("teacher_data")
    result["simulation_seed"] = next(
        (entry[key] for key in ("simulation_seed", "sim_seed") if entry.get(key) is not None),
        None,
    )
    result["search_time_budget_seconds"] = next(
        (entry[key] for key in ("search_time_budget_seconds", "time_budget_seconds") if entry.get(key) is not None),
        None,
    )
    if method_profile == PROFILE:
        if result["search_time_budget_seconds"] is not None:
            raise ValueError("The full method uses its fixed candidate budget; per-source time limits require method.profile=custom")
        if result.get("teacher_delay") is not None:
            raise ValueError("The full method uses recorded source ATT; a manual teacher baseline requires method.profile=custom")
    return result


def validate_config(config):
    """Validate the stable full method separately from explicit custom settings."""
    method = config.get("method", {})
    unknown_method = sorted(set(method) - {"profile", "prompt_template"})
    if unknown_method:
        raise ValueError("Unsupported method options: " + ", ".join(unknown_method))
    if method.get("prompt_template", PROMPT_TEMPLATE) != PROMPT_TEMPLATE:
        raise ValueError("Supported prompt_template=" + PROMPT_TEMPLATE)
    search = config.get("search", {})
    allowed = {
        "population_size",
        "generations",
        "mutation_rate",
        "constraint_retries",
        "max_initial_attempts",
        "fit_phase_bias",
        "bias_fit_iterations",
        "bias_bound",
        "slot_based_parent",
        "action_mode",
        "constraint_validation",
        "time_budget_seconds",
    }
    unknown = sorted(set(search) - allowed)
    if unknown:
        raise ValueError("Unsupported search options: " + ", ".join(unknown))
    profile = config.get("method", {}).get("profile")
    if profile not in {None, "custom", PROFILE}:
        raise ValueError("Unknown method profile: " + str(profile))
    if profile != PROFILE:
        return
    expected = {
        "population_size": 5,
        "generations": 5,
        "mutation_rate": 1.0,
        "constraint_retries": 3,
        "max_initial_attempts": 30,
        "fit_phase_bias": True,
        "bias_fit_iterations": CALIBRATION.max_iterations,
        "bias_bound": CALIBRATION.bias_bound,
        "slot_based_parent": True,
    }
    for key, value in expected.items():
        if search.get(key) != value:
            raise ValueError(f"{PROFILE} requires search.{key}={value!r}")
    if search.get("constraint_validation", True) is not True:
        raise ValueError("The full method requires complete iterative constraint review")
    if search.get("time_budget_seconds") is not None:
        raise ValueError(
            "The full method uses its fixed candidate budget; choose method.profile=custom for a time limit"
        )
    evaluation = config["evaluation"]
    if (
        evaluation.get("action_weight") != 1.0
        or evaluation.get("performance_weight") != 1.0
        or evaluation.get("score_mode") != "action_plus_perf"
    ):
        raise ValueError("The full method requires ABF plus capped exponential control performance")
    if (
        evaluation.get("simulation_steps") is not None
        or evaluation.get("max_timesteps") is not None
    ):
        raise ValueError("The full method inherits the complete source horizon and observations")


def configure(config):
    """Validate explicit settings; configuration does not change class methods."""
    validate_config(config)


def source_environment(path):
    """Use the shared traffic settings with the declared student feature semantics."""
    config = json.loads(Path(path).read_text())
    config["direction_feature_semantics"] = DIRECTION_SEMANTICS
    return config
