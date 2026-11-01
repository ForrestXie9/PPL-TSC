#!/usr/bin/env python3
"""Evaluate one frozen deployment in CityFlow, without a teacher or an LLM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_key(value):
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 0:
        raise ValueError("Bias indices must be canonical nonnegative integers")
    return int(value)


def validate_bias_support(bias, phase_counts):
    for intersection, phases in bias.items():
        if intersection >= len(phase_counts):
            raise ValueError("Calibration refers to an unknown target intersection")
        if any(phase >= phase_counts[intersection] for phase in phases):
            raise ValueError("Calibration refers to an unknown target phase")


def validate_deployment(deployment: dict, roadnet_hash: str) -> dict[int, dict[int, float]]:
    """Preserve source calibration and reject unmapped intersection biases."""
    if not isinstance(deployment, dict):
        raise ValueError("Deployment must be a JSON object")
    if deployment.get("action_mode") not in {"phase_argmax", "keep_switch", "binary_keep_switch"}:
        raise ValueError("Deployment must specify a supported action_mode")
    if not isinstance(deployment.get("feature_stats"), dict):
        raise ValueError("Deployment must contain frozen feature_stats")
    run_config = deployment.get("run_config", {})
    if not isinstance(run_config, dict):
        raise ValueError("Deployment run_config must be a JSON object")
    if run_config.get("feature_contract_version") not in {None, "raw_direction_features_v1", "normalized_phase_features_v1", "normalized_phase_features_v2"}:
        raise ValueError("Unknown feature contract; refusing silent normalization fallback")
    from ppl_tsc_current.shared_lane import restore_contract
    restore_contract({}, run_config)
    raw = deployment.get("per_intersection_bias", {})
    if not isinstance(raw, dict) or any(not isinstance(v, dict) for v in raw.values()):
        raise ValueError("Malformed per_intersection_bias")
    bias = {index_key(i): {index_key(p): float(v) for p, v in phases.items()} for i, phases in raw.items()}
    if any(not math.isfinite(v) for phases in bias.values() for v in phases.values()):
        raise ValueError("Calibration biases must be finite")
    if any(v != 0 for phases in bias.values() for v in phases.values()):
        source_hash = deployment.get("run_config", {}).get("source_roadnet_sha256")
        if source_hash != roadnet_hash:
            raise ValueError(
                "Per-intersection calibration requires the identical source roadnet. "
                "Cross-topology evaluation needs a policy prepared with a portable "
                "calibration protocol; this runner does not drop or remap biases."
            )
    return bias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--scenario", required=True, choices=sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir()))
    parser.add_argument("--flow", type=int, choices=range(5), required=True)
    parser.add_argument("--steps", type=int, help="Required when changing scenario or using a legacy artifact without a saved horizon")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Compatibility flag: action diagnostics are now recorded by default")
    parser.add_argument("--environment-semantics", choices=["per_decision", "legacy_per_read"],
                        help="Required for legacy TinyLight artifacts; legacy_per_read reproduces the historical feature bug")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.deployment.resolve():
        parser.error("Output must not overwrite deployment")
    from ppl_tsc_current.artifact_io import evaluation_record
    with evaluation_record(args.output, {"deployment_path": str(args.deployment.resolve()), "scenario": args.scenario, "flow": args.flow}) as commit_result:
        deployment = json.loads(args.deployment.read_text())
        scenario_dir = ROOT / "data" / args.scenario
        roadnet = scenario_dir / "roadnet.json"
        flow = scenario_dir / f"flow_{args.flow}.json"
        bias = validate_deployment(deployment, file_hash(roadnet))
        run_config = deployment.get("run_config", {})
        if args.steps is None and run_config.get("dataset") != args.scenario:
            parser.error("--steps is required for a different target scenario")
        steps = args.steps if args.steps is not None else run_config.get("simulation_steps")
        if steps is None:
            parser.error("--steps is required: deployment does not record its horizon")

        from utilities.utils import get_config
        from LLM_SR_DISTILLATION.function_utility import FunctionUtility
        from LLM_SR_DISTILLATION.feature_extractor import build_feature_extractor
        from LLM_SR_DISTILLATION.policy_diagnostics import PolicyDecisionDiagnostics
        from LLM_SR_DISTILLATION.simulator import StudentSimulator

        config = get_config(str(ROOT / "configs/traffic.json"))
        teacher = run_config.get("teacher_model")
        if teacher not in config:
            raise ValueError("Deployment run_config.teacher_model is missing or unsupported")
        features = run_config.get("observation_feature_list", config[teacher]["observation_feature_list"])
        semantics = args.environment_semantics or run_config.get("environment_semantics")
        if semantics is None and any("sum_waiting_time" in feature for feature in features):
            parser.error("Artifact uses waiting time but records no semantics; explicitly select --environment-semantics")
        config["waiting_time_semantics"] = semantics or "per_decision"
        if deployment["action_mode"] == "binary_keep_switch" and "inlane_2_num_vehicle" not in features:
            raise ValueError("Native EcoLight needs the exact inlane_2_num_vehicle feature interface")
        interval = int(config.get("action_interval", 20))
        if steps <= 0 or steps % interval:
            parser.error(f"--steps must be a positive multiple of {interval}")
        seed = args.seed if args.seed is not None else int(run_config.get("sim_seed", 992832))
        config.update({
            "direction_feature_semantics": run_config.get("direction_feature_semantics", "legacy_first_match"),
            "cur_agent": teacher, "save_result": False, "render": False,
            "inter_name": args.scenario, "flow_idx": args.flow,
            "dir": str(scenario_dir) + "/", "roadnetFile": "roadnet.json",
            "flowFile": flow.name, "seed": seed, "num_step": steps,
            "current_episode_idx": 0, "current_episode_step_idx": 0,
        })
        simulator = StudentSimulator(config, features)
        feature_contract = run_config.get("feature_contract_version")
        extractor = build_feature_extractor(
            teacher, features, deployment["feature_stats"],
            normalize_features=feature_contract in {"normalized_phase_features_v1", "normalized_phase_features_v2"},
            clip_normalized=feature_contract == "normalized_phase_features_v1",
        )
        utility = FunctionUtility(deployment["function_code"],
                                  feature_extractor=extractor,
                                  feature_stats=deployment["feature_stats"],
                                  action_mode=deployment["action_mode"])
        diagnostics = PolicyDecisionDiagnostics()

        bias_checked = False

        def policy(env, intersection):
            nonlocal bias_checked
            if not bias_checked:
                validate_bias_support(bias, [len(inter.n_phase) for inter in env.n_intersection])
                bias_checked = True
            states = simulator.build_phase_state_dict(env, intersection)
            phase_bias = bias.get(intersection)
            if phase_bias and not set(phase_bias).issubset(states):
                raise ValueError("Calibration refers to an unknown target phase")
            selected = utility.select_action(states, phase_bias=phase_bias, raise_on_error=True)
            if diagnostics is not None:
                current = int(next(iter(states.values()))["current_phase"])
                raw = (utility.select_action(states, phase_bias=None, raise_on_error=True)
                       if phase_bias and any(value != 0 for value in phase_bias.values())
                       else selected)
                diagnostics.record(intersection, current, selected, raw)
            return selected

        try:
            metrics = asdict(simulator.run_episode(policy))
        finally:
            if utility._compiled_function is not None:
                utility._compiled_function.close()
        if not math.isfinite(metrics["avg_travel_time"]) or metrics["avg_travel_time"] <= 0:
            raise ValueError("Evaluation produced invalid ATT")
        result = {
            "status": "complete", "scenario": args.scenario, "flow": args.flow,
            "simulation_seed": seed, "simulation_steps": steps,
            "action_mode": deployment["action_mode"], "policy_state": "frozen",
            "environment_semantics": config["waiting_time_semantics"],
            "method_profile":run_config.get("method_profile","legacy_configuration"),
            "direction_feature_semantics":config["direction_feature_semantics"],
            "target_search": False, "target_calibration": False,
            "target_feature_stat_recomputation": False,
            "deployment_sha256": file_hash(args.deployment),
            "roadnet_sha256": file_hash(roadnet), "flow_sha256": file_hash(flow),
            "traffic_config_sha256": file_hash(ROOT / "configs/traffic.json"),
            "metrics": metrics,
        }
        if diagnostics is not None:
            result["policy_diagnostics"] = diagnostics.summary()
        commit_result(result)
        print(f"Frozen policy ATT: {metrics['avg_travel_time']:.6f}; saved {args.output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
