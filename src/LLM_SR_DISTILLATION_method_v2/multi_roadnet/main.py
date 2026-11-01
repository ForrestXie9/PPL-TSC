# -*- coding: utf-8 -*-
"""
Multi-Roadnet Joint Distillation

One teacher, multiple road networks: trains a single utility function that generalizes
across Jinan, Atlanta, Hangzhou, etc. Does not modify any original LLM_SR_DISTILLATION_method_v2 files.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Any

from LLM_SR_DISTILLATION_method_v2.data_loader import TeacherDataLoader
from LLM_SR_DISTILLATION.feature_statistics import merge_feature_statistics
from LLM_SR_DISTILLATION_method_v2.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION_method_v2.llm_generator import LLMGenerator
from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator
from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
from LLM_SR_DISTILLATION_method_v2.feature_extractor import build_feature_extractor
from LLM_SR_DISTILLATION_method_v2.simulator import StudentSimulator
from LLM_SR_DISTILLATION_method_v2.multi_roadnet.multi_roadnet_evaluator import MultiRoadnetEvaluator
from utilities.utils import (
    get_config,
    infer_dataset_name_from_teacher_path,
    validate_cityflow_roadnet_file,
    resolve_llm_connection_settings,
    apply_llm_backend_cli_choice,
    llm_api_key_missing_message,
    set_seed,
)


def run_multi_roadnet_distillation(
    roadnet_configs: List[Dict[str, Any]],
    teacher_model: str,
    args,
    config: Dict,
    llm_api_endpoint: str,
    llm_api_key: str,
    llm_model: str,
) -> None:
    """Run joint distillation across multiple road networks."""
    if teacher_model not in config:
        raise ValueError(f"Teacher model {teacher_model} not found in config.json")

    teacher_config = config[teacher_model]
    observation_features = teacher_config.get("observation_feature_list", [])

    scenario_names = [c.get("scenario", f"roadnet_{i}") for i, c in enumerate(roadnet_configs)]
    scenario_str = "+".join(scenario_names[:5])
    if len(scenario_names) > 5:
        scenario_str += f"_+{len(scenario_names)-5}more"

    print("=" * 80)
    print("LLM-based Multi-Roadnet Joint Distillation")
    print("=" * 80)
    print(f"Teacher Model: {teacher_model}")
    print(f"Road Networks: {scenario_str}")
    print(f"Observation Features: {observation_features}")
    print(f"Population Size: {args.population_size}")
    print(f"Generations: {args.n_generations}")
    print("=" * 80)

    data_loaders = []
    teacher_tt_list = []
    feature_stats_list = []

    for cfg in roadnet_configs:
        path = cfg["teacher_data_path"]
        scenario_id = cfg.get("scenario", "unknown")
        loader = TeacherDataLoader(path)
        data_loaders.append((loader, scenario_id))

        teacher_delay = cfg.get("teacher_delay")
        if teacher_delay is not None:
            tt = float(teacher_delay)
        else:
            metrics = loader.get_teacher_metrics()
            tt = metrics.get("average_travel_time") or metrics.get("average_delay")
            if tt is None:
                raise ValueError(
                    f"Cannot extract teacher travel time from {path}. "
                    "Provide teacher_delay in config or ensure data has stats.avg_travel_time."
                )
            tt = float(tt)
        teacher_tt_list.append(tt)
        feature_stats_list.append(loader.compute_feature_stats(observation_features))

    merged_stats_by_kind = merge_feature_statistics(feature_stats_list)

    feature_extractor = build_feature_extractor(
        teacher_model=teacher_model,
        observation_feature_list=observation_features,
        feature_stats=None,
        normalize_features=False,
    )
    feature_stats = {}
    for spec in feature_extractor.specs:
        if spec.kind in merged_stats_by_kind:
            s = merged_stats_by_kind[spec.kind]
            feature_stats[spec.name] = dict(s)
        else:
            feature_stats[spec.name] = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 1.0}
    feature_extractor.feature_stats = feature_stats

    actual_feature_keys = []
    for feat in observation_features or []:
        f_lower = (feat or "").lower()
        if "pressure" in f_lower and "pressure" not in actual_feature_keys:
            actual_feature_keys.append("pressure")
        elif "num_waiting_vehicle" in f_lower and "num_waiting_vehicle" not in actual_feature_keys:
            actual_feature_keys.append("num_waiting_vehicle")
        elif "num_vehicle" in f_lower and "num_vehicle" not in actual_feature_keys:
            actual_feature_keys.append("num_vehicle")
        elif "delay" in f_lower and "delay" not in actual_feature_keys:
            actual_feature_keys.append("delay")
        elif "waiting_time" in f_lower or "sum_waiting_time" in f_lower:
            if "waiting_time" not in actual_feature_keys:
                actual_feature_keys.append("waiting_time")
            if "sum_waiting_time" not in actual_feature_keys:
                actual_feature_keys.append("sum_waiting_time")
    allowed_keys = list(set(actual_feature_keys) | set(feature_extractor.feature_names or []))
    feature_kinds = {spec.name: spec.kind for spec in feature_extractor.specs}
    feature_ranges = {n: (float(s["min"]), float(s["max"])) for n, s in feature_stats.items()}

    first_path = roadnet_configs[0]["teacher_data_path"]
    first_loader = data_loaders[0][0]

    constraint_validator = ConstraintValidator(
        teacher_model=teacher_model,
        observation_feature_list=observation_features,
        json_file_path=first_path,
        intersection_id=None,
        feature_ranges=feature_ranges,
        feature_kinds=feature_kinds,
        actual_feature_keys=allowed_keys,
        data_loader=first_loader,
        feature_stats=feature_stats,
    )

    num_phases = 8
    first_ts = first_loader.get_timesteps(max_timesteps=1)
    if first_ts and isinstance(first_ts[0].get("phases"), list) and len(first_ts[0]["phases"]) > 0:
        num_phases = len(first_ts[0]["phases"])

    llm_generator = LLMGenerator(
        api_endpoint=llm_api_endpoint,
        api_key=llm_api_key,
        model=llm_model,
        feature_names=feature_extractor.feature_names,
        num_phases=num_phases,
        teacher_model=teacher_model,
        observation_features=observation_features,
        debug_mode=args.verbose,
        feature_stats=feature_stats,
    )

    project_root = os.getenv("METHOD_PROJECT_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    base_config = get_config(args.config_path)
    base_config.update(config)

    roadnet_evaluators = []
    for i, (loader, scenario_id) in enumerate(data_loaders):
        teacher_tt = teacher_tt_list[i]
        cfg_i = roadnet_configs[i]
        meta = loader.get_meta()
        collector_cfg = loader.get_collector_config()
        dataset = meta.get("dataset") or cfg_i.get("dataset")
        if not dataset:
            dataset = infer_dataset_name_from_teacher_path(cfg_i.get("teacher_data_path"))
        if not dataset:
            dataset = cfg_i.get("scenario") or teacher_model
        flow_idx = meta.get("flow_idx") if "flow_idx" in meta else cfg_i.get("flow_idx", 0)
        seed = args.sim_seed if args.sim_seed is not None else meta.get("seed", config.get("seed", 0))
        num_step_sim = args.sim_num_step or collector_cfg.get("num_step") or config.get("num_step", 3600)

        data_dir = os.path.join(project_root, "data", dataset)
        if not getattr(args, "enable_simulation", True):
            simulator = None
        elif not os.path.exists(data_dir):
            print(f"  [WARN] Data dir not found for {scenario_id}: {data_dir}, simulation disabled")
            simulator = None
        else:
            validate_cityflow_roadnet_file(data_dir)
            sim_config = dict(base_config)
            sim_config.update({
                "cur_agent": teacher_model,
                "save_result": False,
                "render": False,
                "inter_name": dataset,
                "flow_idx": int(flow_idx),
                "dir": data_dir + os.sep,
                "roadnetFile": "roadnet.json",
                "flowFile": f"flow_{int(flow_idx)}.json",
                "seed": int(seed),
                "num_step": int(num_step_sim),
            })
            simulator = StudentSimulator(sim_config, observation_feature_list=observation_features)

        evaluator = Evaluator(
            data_loader=loader,
            teacher_travel_time=teacher_tt,
            feature_extractor=feature_extractor,
            constraint_validator=constraint_validator if i == 0 else None,
            simulator=simulator,
            verbose=args.verbose,
            action_weight=float(args.action_weight),
            performance_weight=float(args.performance_weight),
            random_seed=int(args.random_seed) + i * 100000,
        )
        roadnet_evaluators.append((evaluator, scenario_id, teacher_tt))

    multi_evaluator = MultiRoadnetEvaluator(
        roadnet_evaluators=roadnet_evaluators,
        verbose=args.verbose,
        action_weight=float(args.action_weight),
        performance_weight=float(args.performance_weight),
    )

    evolution = Evolution(
        data_loader=first_loader,
        constraint_validator=constraint_validator,
        llm_generator=llm_generator,
        evaluator=multi_evaluator,
        feature_stats=feature_stats,
        population_size=args.population_size,
        n_generations=args.n_generations,
        mutation_rate=args.mutation_rate,
        use_intersection_specific_params=args.use_intersection_specific_params,
        max_constraint_retries=args.max_constraint_retries,
        use_slot_based_parent=args.use_slot_based_parent,
        fit_bias_during_evolution=args.fit_bias_during_evolution,
        bias_fit_max_iter=args.bias_fit_max_iter,
        bias_fit_bound=args.bias_fit_bound,
        use_constraint_validation=args.use_constraint_validation,
        max_init_attempts=args.max_init_attempts if args.max_init_attempts > 0 else None,
        max_timesteps=args.max_timesteps,
    )

    evolution.run_config = {
        "feature_contract_version": "raw_direction_features_v1",
        "teacher_model": teacher_model,
        "dataset": f"multi_{scenario_str}",
        "scenario": f"multi_{scenario_str}",
        "population_size": int(args.population_size),
        "n_generations": int(args.n_generations),
        "action_weight": float(args.action_weight),
        "performance_weight": float(args.performance_weight),
    }

    print("\nStarting multi-roadnet evolution...")
    evolution.run(verbose=args.verbose)

    output_dir_basename = os.path.basename(os.path.normpath(args.output_dir))
    if output_dir_basename.startswith("flow") and output_dir_basename[4:].isdigit():
        out_dir = os.path.join(args.output_dir, scenario_str.replace("+", "_"))
    else:
        out_dir = os.path.join(args.output_dir, teacher_model, "multi", scenario_str.replace("+", "_"))
    os.makedirs(out_dir, exist_ok=True)

    best = evolution._get_best_individual()
    per_intersection_bias = best.per_intersection_bias

    first_meta = data_loaders[0][0].get_meta()
    seed = args.sim_seed if args.sim_seed is not None else first_meta.get("seed", config.get("seed", 0))
    run_suffix = f"seed{int(seed)}_pop{args.population_size}_gen{args.n_generations}"

    results = {
        "run_config": {
            "teacher_model": teacher_model,
            "dataset": f"multi_{scenario_str}",
            "scenario": f"multi_{scenario_str}",
            "roadnets": scenario_names,
            "population_size": int(args.population_size),
            "n_generations": int(args.n_generations),
            "action_weight": float(args.action_weight),
            "performance_weight": float(args.performance_weight),
        },
        "best_individual": {
            "function_code": best.function_code[:500] + "..." if len(best.function_code) > 500 else best.function_code,
            "function_code_length": len(best.function_code),
            "score": best.score,
            "metrics": best.metrics,
            "explanation": best.explanation,
        },
        "per_intersection_bias": {
            str(k): {str(p): float(v) for p, v in vb.items()}
            for k, vb in (per_intersection_bias or {}).items()
        },
    }

    results_path = os.path.join(out_dir, f"results_{run_suffix}.json")
    deployment_path = os.path.join(out_dir, f"deployment_model_{run_suffix}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    evolution.save_best_model(
        save_path=deployment_path,
        feature_extractor=feature_extractor,
        per_intersection_bias=per_intersection_bias,
    )

    print(f"\nResults saved to {results_path}")
    print(f"Deployment config saved to {deployment_path}")
    print(f"\nBest score: {best.score:.4f} (across {len(roadnet_configs)} road networks)")


def main():
    parser = argparse.ArgumentParser(
        description="One-teacher multi-roadnet joint distillation",
        epilog="""
Example:
  python -m LLM_SR_DISTILLATION_method_v2.multi_roadnet.main --config multi_roadnet/teachers_config_multi_roadnet_example.json
""",
    )
    parser.add_argument("--config", type=str, required=True, help="JSON config path (teacher_model + roadnet_configs)")
    parser.add_argument("--config_path", type=str, default="configs/traffic.json", help="Traffic config path")
    parser.add_argument("--output_dir", type=str, default="./llm_sr_results_multi")
    parser.add_argument("--llm_api_endpoint", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument(
        "--llm_backend",
        type=str,
        default="1",
        choices=["1", "2", "3", "4"],
        help=(
            "Default 1: general gateway (optional general_llm_embedded.py); "
            "2=COMP（comp_llm_embedded.py）；"
            "3: Jiekou.AI (optional jiekou_llm_embedded.py). Omission selects 1 and ignores a stale LLM_BACKEND. "
            "4: Yunwu (yunwu_llm_embedded.py). "
        ),
    )
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--n_generations", type=int, default=5)
    parser.add_argument("--mutation_rate", type=float, default=1.0)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="action_plus_perf",
        choices=["action_plus_perf", "action_only", "performance_only"],
        help="Scoring objective for evolution.",
    )
    parser.add_argument(
        "--action_weight",
        type=float,
        default=1.0,
        help="Weight for action_consistency term in base score.",
    )
    parser.add_argument(
        "--performance_weight",
        type=float,
        default=1.0,
        help="Weight for performance_score term in base score.",
    )
    parser.add_argument("--use_slot_based_parent", action="store_true", default=True)
    parser.add_argument("--no_slot_based_parent", action="store_false", dest="use_slot_based_parent")
    parser.add_argument("--fit_bias_during_evolution", action="store_true", default=True)
    parser.add_argument("--no_fit_bias_during_evolution", action="store_false", dest="fit_bias_during_evolution")
    parser.add_argument("--bias_fit_max_iter", type=int, default=30)
    parser.add_argument("--bias_fit_bound", type=float, default=10.0)
    parser.add_argument("--use_intersection_specific_params", action="store_true", default=True)
    parser.add_argument("--max_constraint_retries", type=int, default=3)
    parser.add_argument("--max_init_attempts", type=int, default=30, help="Max LLM attempts per individual during init; 0=unlimited")
    parser.add_argument("--use_constraint_validation", action="store_true", default=True)
    parser.add_argument("--enable_simulation", action="store_true", default=True)
    parser.add_argument("--no_enable_simulation", action="store_false", dest="enable_simulation")
    parser.add_argument("--sim_num_step", type=int, default=None)
    parser.add_argument("--sim_seed", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    apply_llm_backend_cli_choice(getattr(args, "llm_backend", None))
    set_seed(int(args.random_seed), use_torch=False)

    if args.score_mode == "action_only":
        args.action_weight = 1.0
        args.performance_weight = 0.0
    elif args.score_mode == "performance_only":
        args.action_weight = 0.0
        args.performance_weight = 1.0
    else:
        args.action_weight = float(args.action_weight)
        args.performance_weight = float(args.performance_weight)

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(args.config, "r", encoding="utf-8") as f:
        multi_cfg = json.load(f)

    teacher_model = multi_cfg["teacher_model"]
    roadnet_configs = multi_cfg["roadnet_configs"]
    if not roadnet_configs:
        raise ValueError("roadnet_configs must be a non-empty list")

    args.enable_simulation = True

    llm_api_endpoint, llm_api_key, llm_model = resolve_llm_connection_settings(
        args.llm_api_endpoint,
        args.llm_api_key,
        args.llm_model,
    )
    if not llm_api_key:
        raise ValueError(llm_api_key_missing_message())

    from LLM_SR_DISTILLATION_method_v2.llm_runtime_info import print_llm_startup_banner

    print_llm_startup_banner(
        llm_api_endpoint,
        llm_model,
        title="LLM configuration (multiple road networks)",
        cli_llm_model=args.llm_model,
    )

    if "output_dir" in multi_cfg:
        args.output_dir = multi_cfg["output_dir"]

    run_multi_roadnet_distillation(
        roadnet_configs=roadnet_configs,
        teacher_model=teacher_model,
        args=args,
        config=config,
        llm_api_endpoint=llm_api_endpoint,
        llm_api_key=llm_api_key,
        llm_model=multi_cfg.get("llm_model") or llm_model,
    )


if __name__ == "__main__":
    main()
