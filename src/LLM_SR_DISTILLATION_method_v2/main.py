# -*- coding: utf-8 -*-
"""

Main Entry Point

LLM-based Symbolic Regression Distillation

"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from LLM_SR_DISTILLATION_method_v2.data_loader import TeacherDataLoader
from LLM_SR_DISTILLATION.source_preparation import prepare_source, check_prepared_statistics
from LLM_SR_DISTILLATION_method_v2.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION_method_v2.llm_generator import LLMGenerator
from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator
from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
from LLM_SR_DISTILLATION_method_v2.feature_extractor import build_feature_extractor
from LLM_SR_DISTILLATION_method_v2.simulator import StudentSimulator
from LLM_SR_DISTILLATION_method_v2.multi_roadnet.main import run_multi_roadnet_distillation
from utilities.utils import (
    get_config,
    infer_dataset_name_from_teacher_path,
    validate_cityflow_roadnet_file,
    resolve_llm_connection_settings,
    apply_llm_backend_cli_choice,
    llm_api_key_missing_message,
    set_seed,
)

def load_config(config_path: str = "config.json") -> Dict:

    """Load configuration from JSON file"""

    with open(config_path, 'r', encoding='utf-8') as f:

        return json.load(f)


def require_ecolight_action_contract(action_mode: str, first_timestep: Dict) -> None:
    """Also protect direct internal entry points from Naive/Native mixing."""
    mode = "keep_switch" if action_mode == "auto" else action_mode
    if mode not in {"keep_switch", "binary_keep_switch", "phase_argmax"}:
        raise ValueError(f"Unsupported EcoLight action mode {action_mode!r}")
    phases = first_timestep.get("phases") or []
    if not phases:
        raise ValueError("EcoLight rollout has no phase observations")
    binary = phases[0].get("state", {}).get("binary_features")
    aligned = "teacher_binary_action" in first_timestep or isinstance(binary, dict)
    if mode == "binary_keep_switch":
        if ("teacher_binary_action" not in first_timestep or not isinstance(binary, dict)
                or not {"green_vehicle", "red_vehicle"} <= set(binary)):
            raise ValueError("binary_keep_switch requires exact aligned EcoLight binary inputs")
    elif aligned:
        raise ValueError("Aligned EcoLight binary rollout requires binary_keep_switch")


def run_distillation_single(
    teacher_data_path: str,
    teacher_model: str,
    teacher_delay: Optional[float],
    args,
    config: Dict,
    llm_api_endpoint: str,
    llm_api_key: str,
    llm_model: str,
    scenario: Optional[str] = None,
    simulation_seed_override: Optional[int] = None,
    random_seed_override: Optional[int] = None,
    derive_seed_from_rollout: bool = False,
    time_budget_seconds_override: Optional[float] = None,
) -> None:
    """Run distillation for a single teacher model. Saves results to args.output_dir."""
    if teacher_model not in config:
        raise ValueError(f"Teacher model {teacher_model} not found in config.json")

    if os.environ.get("PPL_METHOD_PROFILE") == "full_100_256_v1":
        config["direction_feature_semantics"]="compatible_shared_lane_mean_v1"
    teacher_config = config[teacher_model]
    observation_features = teacher_config.get('observation_feature_list', [])

    print("=" * 80)
    print("LLM-based Symbolic Regression Distillation")
    print("=" * 80)
    print(f"Teacher Model: {teacher_model}")
    print(f"Teacher Data: {teacher_data_path}")
    print(f"Observation Features: {observation_features}")
    print(f"Population Size: {args.population_size}")
    print(f"Generations: {args.n_generations}")
    print("=" * 80)

    print("\nInitializing components...")
    data_loader = TeacherDataLoader(teacher_data_path)
    source_preparation = prepare_source(
        data_loader, teacher_model, observation_features, config, args.action_mode)
    meta = data_loader.get_meta()
    first_ts = data_loader.get_timesteps(intersection_id=args.intersection_id, max_timesteps=1)
    if teacher_model.casefold() == "ecolight":
        if not first_ts:
            raise ValueError("EcoLight rollout has no timesteps")
        require_ecolight_action_contract(args.action_mode, first_ts[0])
    rollout_seed = meta.get("seed") if derive_seed_from_rollout else None
    simulation_seed = (
        simulation_seed_override
        if simulation_seed_override is not None
        else rollout_seed
        if rollout_seed is not None
        else args.sim_seed
        if args.sim_seed is not None
        else config.get("seed", 0)
    )
    random_seed = (
        random_seed_override
        if random_seed_override is not None
        else rollout_seed
        if rollout_seed is not None
        else args.random_seed
    )
    simulation_seed = int(simulation_seed)
    random_seed = int(random_seed)
    from agent.imitation_based.llm_integration import reset_llm_cache_sequence
    cache_context = json.dumps({
        "teacher": teacher_model,
        "trajectory_sha256": hashlib.sha256(Path(teacher_data_path).read_bytes()).hexdigest(),
        "random_seed": random_seed, "simulation_seed": simulation_seed,
        "traffic_config": config,
    }, sort_keys=True, default=str)
    reset_llm_cache_sequence(hashlib.sha256(cache_context.encode()).hexdigest())
    time_budget_seconds = (
        float(time_budget_seconds_override)
        if time_budget_seconds_override is not None
        else (
            float(args.time_budget_seconds)
            if args.time_budget_seconds is not None
            else None
        )
    )

    # Make a scenario reproducible both on its own and at any position in a batch.
    set_seed(random_seed, use_torch=False)
    seed_source = (
        "manifest override"
        if simulation_seed_override is not None or random_seed_override is not None
        else "rollout metadata"
        if rollout_seed is not None
        else "experiment defaults"
    )
    print(
        f"Effective seeds: simulation={simulation_seed}, random={random_seed} "
        f"({seed_source})"
    )

    num_phases = 8
    if first_ts and isinstance(first_ts[0].get("phases"), list) and len(first_ts[0]["phases"]) > 0:
        num_phases = len(first_ts[0]["phases"])

    teacher_travel_time = teacher_delay
    if teacher_travel_time is None:
        teacher_metrics = data_loader.get_teacher_metrics(args.intersection_id)
        if teacher_metrics.get('average_travel_time') is not None:
            teacher_travel_time = teacher_metrics['average_travel_time']
            print(f"Automatically extracted teacher travel time from data: {teacher_travel_time:.2f} seconds")
        elif teacher_metrics.get('average_delay') is not None:
            teacher_travel_time = teacher_metrics['average_delay']
            print(f"Automatically extracted teacher delay from data (used as travel time proxy): {teacher_travel_time:.2f} seconds")
        else:
            raise ValueError(
                "Cannot extract teacher travel time from data. "
                "Please ensure data contains 'stats.avg_travel_time' (or avg_ATT) field or provide teacher_delay."
            )
    else:
        print(f"Using provided teacher travel time: {teacher_travel_time:.2f} seconds")

    print("\nComputing feature statistics for normalization...")
    feature_extractor = build_feature_extractor(
        teacher_model=teacher_model,
        observation_feature_list=observation_features,
        feature_stats=None,
        normalize_features=False
    )
    feature_names = feature_extractor.feature_names

    feature_stats_by_kind = data_loader.compute_feature_stats(observation_features)
    from LLM_SR_DISTILLATION.feature_statistics import source_feature_contract
    feature_stats, actual_feature_keys = source_feature_contract(
        feature_extractor, feature_stats_by_kind, data_loader)
    feature_names = feature_extractor.feature_names
    if args.action_mode == "binary_keep_switch":
        binary_stats = data_loader.compute_binary_feature_stats()
        missing_binary = sorted(
            {"green_vehicle", "red_vehicle"} - set(binary_stats)
        )
        if missing_binary:
            raise ValueError(
                "binary_keep_switch requires exact EcoLight binary features "
                f"in the rollout; missing: {missing_binary}"
            )
        feature_stats.update(binary_stats)
    feature_extractor.feature_stats = feature_stats
    source_preparation['statistics_check'] = check_prepared_statistics(
        source_preparation, feature_extractor, feature_stats)

    allowed_keys_for_check = actual_feature_keys
    feature_kinds = {spec.name: spec.kind for spec in feature_extractor.specs}
    feature_ranges = {
        name: (float(stats.get('min', 0.0)), float(stats.get('max', 1.0)))
        for name, stats in feature_stats.items()
    }

    constraint_validator = ConstraintValidator(
        teacher_model=teacher_model,
        observation_feature_list=observation_features,
        json_file_path=teacher_data_path,
        intersection_id=args.intersection_id,
        feature_ranges=feature_ranges,
        feature_kinds=feature_kinds,
        actual_feature_keys=allowed_keys_for_check,
        data_loader=data_loader,
        feature_stats=feature_stats,
        action_mode=args.action_mode,
        binary_min_margin_span=(
            0.5 * args.bias_fit_bound if args.binary_margin_span_guard else 0.0
        ),
    )

    constraint_validator.prepare_source_checks()
    source_preparation['verifier_revision'] = 'source_preparation_active_coverage_v1'
    source_preparation['active_phase_preparation'] = {
        'status': constraint_validator._active_phase_prepared['status'],
        'planned_pairs': len(constraint_validator._active_phase_prepared['probes']),
    }
    preparation_path = Path(args.output_dir) / 'source_preparation.json'
    preparation_path.parent.mkdir(parents=True, exist_ok=True)
    preparation_path.write_text(json.dumps(source_preparation, indent=2, allow_nan=False) + '\n')

    llm_generator = LLMGenerator(
        api_endpoint=llm_api_endpoint,
        api_key=llm_api_key,
        model=llm_model,
        feature_names=feature_names,
        num_phases=num_phases,
        teacher_model=teacher_model,
        observation_features=observation_features,
        debug_mode=args.verbose,
        action_mode=args.action_mode,
        binary_margin_guidance=args.binary_margin_guidance,
        calibration_bias_bound=args.bias_fit_bound,
        feature_stats=feature_stats,
        actual_feature_keys=actual_feature_keys,
    )

    simulator = None
    if args.enable_simulation:
        collector_cfg = data_loader.get_collector_config()
        dataset = meta.get("dataset") or infer_dataset_name_from_teacher_path(teacher_data_path)
        if not dataset:
            dataset = teacher_model
        flow_idx = meta.get("flow_idx", args.intersection_id or 0)
        num_step_sim = args.sim_num_step or collector_cfg.get("num_step", None) or config.get("num_step", 3600)

        current_file = os.path.abspath(__file__)
        llm_sr_dir = os.path.dirname(current_file)
        project_root = os.getenv("METHOD_PROJECT_ROOT") or os.path.dirname(
            os.path.dirname(llm_sr_dir)
        )
        data_dir = os.path.join(project_root, "data", dataset)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please ensure data/{dataset}/ exists in the project root."
            )

        validate_cityflow_roadnet_file(data_dir)

        sim_config = get_config(args.config_path)
        sim_config.update(config)
        sim_config.update({
            "cur_agent": teacher_model,
            "save_result": False,
            "render": False,
            "inter_name": dataset,
            "flow_idx": int(flow_idx),
            "dir": data_dir + os.sep,
            "roadnetFile": "roadnet.json",
            "flowFile": f"flow_{int(flow_idx)}.json",
            "seed": simulation_seed,
            "num_step": int(num_step_sim),
            "current_episode_idx": 0,
            "current_episode_step_idx": 0,
        })
        simulator = StudentSimulator(sim_config, observation_feature_list=observation_features)

    evaluator = Evaluator(
        data_loader=data_loader,
        teacher_travel_time=teacher_travel_time,
        feature_extractor=feature_extractor,
        constraint_validator=constraint_validator,
        simulator=simulator,
        verbose=args.verbose,
        action_weight=float(args.action_weight),
        performance_weight=float(args.performance_weight),
        random_seed=random_seed,
        native_liveness_guard=args.native_liveness_guard,
    )

    evolution = Evolution(
        data_loader=data_loader,
        constraint_validator=constraint_validator,
        llm_generator=llm_generator,
        evaluator=evaluator,
        feature_stats=feature_extractor.feature_stats,
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
        time_budget_seconds=time_budget_seconds,
        repair_rejected_candidates=args.repair_rejected_candidates,
        evaluation_error_feedback=args.evaluation_error_feedback,
    )

    dataset = meta.get("dataset") or teacher_model
    evolution.run_config = {
        "algorithm_version": "0.3.2-raw-state-optional-normalization",
        "feature_contract_version": "raw_direction_features_v1",
        "source_verifier_revision": "source_preparation_active_coverage_v1",
        "teacher_model": teacher_model,
        "environment_semantics": get_config(args.config_path).get("waiting_time_semantics", "per_decision"),
        "dataset": dataset,
        "flow_idx": int(meta.get("flow_idx", 0)),
        "observation_feature_list": observation_features,
        "simulation_steps": int(num_step_sim) if args.enable_simulation else None,
        "teacher_simulation_steps": data_loader.get_collector_config().get("num_step"),
        "performance_horizon_matched": (
            args.enable_simulation and int(num_step_sim) == data_loader.get_collector_config().get("num_step")
        ),
        "max_timesteps_per_intersection": args.max_timesteps,
        "source_roadnet_sha256": (
            hashlib.sha256(Path(data_dir, "roadnet.json").read_bytes()).hexdigest()
            if args.enable_simulation else None
        ),
        "sim_seed": simulation_seed,
        "random_seed": random_seed,
        "population_size": int(args.population_size),
        "n_generations": int(args.n_generations),
        "calibration": {
            "enabled": bool(args.fit_bias_during_evolution),
            "population_size": 20,
            "max_iterations": int(args.bias_fit_max_iter),
            "bias_bound": float(args.bias_fit_bound),
            "initial_sigma": max(1e-6, float(args.bias_fit_bound) / 3.0),
            "early_stopping": "native CMA-ES stop conditions",
        },
        "scenario": scenario,
        "action_weight": float(args.action_weight),
        "performance_weight": float(args.performance_weight),
        "control_score": {
            "version": "capped_exp_relative_att_v1",
            "formula": "exp(min((teacher_ATT-student_ATT)/(abs(teacher_ATT)+epsilon),0))",
            "epsilon": float(evaluator.epsilon),
            "clipped_at_one": True,
        },
        "time_budget_seconds": time_budget_seconds,
        "repair_rejected_candidates": bool(args.repair_rejected_candidates),
        "binary_margin_guidance": bool(args.binary_margin_guidance),
        "binary_margin_span_guard": bool(args.binary_margin_span_guard),
        "native_liveness_guard": bool(args.native_liveness_guard),
        "evaluation_error_feedback": bool(args.evaluation_error_feedback),
    }

    if os.environ.get("PPL_METHOD_PROFILE") == "full_100_256_v1":
        from ppl_tsc_current.revision import implementation_identity
        evolution.run_config.update(algorithm_version="0.4.5-scale-reminder",method_profile="full_100_256_v1",prompt_template="paper_figure_a2_v1",direction_feature_semantics="compatible_shared_lane_mean_v1")
        evolution.run_config.update(implementation_identity())
        evolution.run_config["verifier"]={"contract_version":"source_schema_sparse_discrimination_v2","source_bundles":100,"coordinates":256,"direction_order_comparisons":40,"phase_label_comparisons":40,"original_source_execution":"all"}
        evolution.run_config["calibration"].update(bias_range_policy="fixed_absolute",tie_rule="least_changes",initial_sigma=float(args.bias_fit_bound)/3.0)

    print("\nStarting evolution...")
    evolution.run(verbose=args.verbose, print_functions=False)

    def _derive_output_dir(
        base_output_dir: str,
        teacher_model_name: str,
        scenario_name: Optional[str],
        teacher_path: str,
        dataset_name: str,
        flow_idx_val: Optional[int],
    ) -> str:
        """Return ``output_dir/<teacher>/<city>/flow<K>``.

        Scenario labels take priority. The teacher-data path and rollout
        metadata provide deterministic fallbacks.
        """
        safe_teacher = str(teacher_model_name).replace(os.sep, "_")

        city: Optional[str] = None
        flow: Optional[int] = None

        # 1) Parse scenario first if available.
        if scenario_name:
            s = str(scenario_name).strip()
            m = re.search(r"(?:^|_)flow(?P<flow>\d+)$", s)
            if m:
                flow = int(m.group("flow"))
                prefix = s[: m.start()]  # remove the trailing "_flowK"
                if prefix.startswith(f"{teacher_model_name}_"):
                    prefix = prefix[len(teacher_model_name) + 1 :]
                city = prefix or None

        # 2) Parse paths such as data_cache/FRAP/Atlanta/flow3.jsonl(.gz).
        if city is None or flow is None:
            norm = str(teacher_path).replace("\\", "/")
            m = re.search(
                r"/(?P<city>[^/]+)/flow(?P<flow>\d+)\.(jsonl|json)(?:\.gz)?$",
                norm,
            )
            if m:
                city = city or m.group("city")
                flow = flow if flow is not None else int(m.group("flow"))

        # 3) fallback
        if city is None:
            city = str(dataset_name or "unknown")
        if flow is None:
            flow = int(flow_idx_val) if flow_idx_val is not None else 0

        safe_city = str(city).replace(os.sep, "_")
        return os.path.join(base_output_dir, safe_teacher, safe_city, f"flow{int(flow)}")

    meta = data_loader.get_meta()
    dataset = meta.get("dataset") or teacher_model
    flow_idx_meta = meta.get("flow_idx", None)
    out_dir = _derive_output_dir(
        base_output_dir=args.output_dir,
        teacher_model_name=teacher_model,
        scenario_name=scenario,
        teacher_path=teacher_data_path,
        dataset_name=str(dataset),
        flow_idx_val=int(flow_idx_meta) if isinstance(flow_idx_meta, (int, float)) else None,
    )
    os.makedirs(out_dir, exist_ok=True)
    best_individual = evolution._get_best_individual()
    per_intersection_bias = best_individual.per_intersection_bias

    collector_cfg = data_loader.get_collector_config()
    dataset = meta.get("dataset") or teacher_model

    results = {
        'run_config': dict(evolution.run_config),
        'search_budget': evolution.time_budget_summary(),
        'best_individual': {
            'function_code': best_individual.function_code[:500] + "..." if len(best_individual.function_code) > 500 else best_individual.function_code,
            'function_code_length': len(best_individual.function_code),
            'score': best_individual.score,
            'metrics': best_individual.metrics,
            'explanation': best_individual.explanation
        },
        'per_intersection_bias': {str(k): {str(p): float(v) for p, v in vb.items()} for k, vb in (per_intersection_bias or {}).items()},
        'generation_history': [
            {
                'generation': i,
                'best_score': max(ind.score for ind in gen_pop),
                'best_function_length': len(max(gen_pop, key=lambda x: x.score).function_code),
                'best_function_preview': max(gen_pop, key=lambda x: x.score).function_code[:200] + "..."
            }
            for i, gen_pop in enumerate(evolution.generation_history)
        ]
    }

    # Build a compact run suffix; teacher/scenario/flow already appear in the output directory.
    run_suffix = f"seed{simulation_seed}_pop{int(args.population_size)}_gen{int(args.n_generations)}"
    results_path = os.path.join(out_dir, f"results_{run_suffix}.json")
    deployment_config_path = os.path.join(out_dir, f"deployment_model_{run_suffix}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")

    evolution.save_best_model(
        save_path=deployment_config_path,
        feature_extractor=feature_extractor,
        per_intersection_bias=per_intersection_bias,
    )
    print(f"\nDeployment config saved to: {deployment_config_path}")

    print("\nBest individual:")
    print(f"  Function Code Length: {len(best_individual.function_code)} characters")
    print(f"  Evaluation Score: {best_individual.score:.4f}")
    print(f"    - Behavioral Fidelity: {best_individual.metrics.get('behavioral_fidelity', best_individual.metrics.get('action_consistency', 0)):.4f}")
    print(f"    - Control Performance: {best_individual.metrics.get('control_performance', best_individual.metrics.get('performance_score', 0)):.4f}")
    st = best_individual.metrics.get('student_travel_time') or best_individual.metrics.get('student_delay', 0)
    tt = best_individual.metrics.get('teacher_travel_time') or best_individual.metrics.get('teacher_delay', 0)
    print(f"    - Avg Travel Time: {st:.4f}")
    print(f"    - Teacher Travel Time: {tt:.4f}")
    print(f"    - Delta: {best_individual.metrics.get('delta_d', 0):.4f}")
    per_inter = best_individual.metrics.get('per_intersection', {})
    if per_inter:
        print(f"\n  Per-Intersection Details ({len(per_inter)} intersections):")
        for inter_id in sorted(per_inter.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            m = per_inter[inter_id]
            score_val = m.get('score', 0)
            fidelity_val = m.get('behavioral_fidelity', m.get('fidelity_score', 0.0))
            perf_val = m.get('control_performance', m.get('performance_score', 0.0))
            print(
                f"    Intersection {inter_id}: "
                f"score={score_val:.4f} | "
                f"Behavioral Fidelity={fidelity_val:.4f} | "
                f"Control Performance={perf_val:.4f}"
            )


def main():

    parser = argparse.ArgumentParser(

        description="LLM-based Symbolic Regression Distillation for Traffic Signal Control",
        epilog="""
Examples:
  # Single teacher (default data path)
  python -m LLM_SR_DISTILLATION_method_v2.main

  # Single teacher with explicit paths
  python -m LLM_SR_DISTILLATION_method_v2.main --teacher_data_path data_cache/MPLight/Jinan/flow0.json --teacher_model MPLight

  # Single teacher with custom output dir and generations
  python -m LLM_SR_DISTILLATION_method_v2.main --teacher_data_path data_cache/MPLight/Jinan/flow0.json --teacher_model MPLight --output_dir ./my_results --n_generations 5 --population_size 3

  # Multiple teachers / scenarios (batch run)
  python -m LLM_SR_DISTILLATION_method_v2.main --teachers_config LLM_SR_DISTILLATION_method_v2/teachers_config_example.json

  # MPLight all scenarios (Atlanta, Hangzhou1/2/3, Jinan, LosAngeles, Manhattan)
  python -m LLM_SR_DISTILLATION_method_v2.main --teachers_config LLM_SR_DISTILLATION_method_v2/teachers_config_MPLight_all.json

  # Joint mode (one teacher, all roadnets, one model):
  python -m LLM_SR_DISTILLATION_method_v2.main --teachers_config LLM_SR_DISTILLATION_method_v2/multi_roadnet/teachers_config_multi_roadnet_example.json

  # teachers_config formats:
  # Sequential (list): one model per entry
  # [{"teacher_data_path": "...", "teacher_model": "MPLight", "scenario": "Jinan_flow"}, ...]
  # Joint (dict): one model for all roadnets
  # {"teacher_model": "MPLight", "roadnet_configs": [{"teacher_data_path": "...", "scenario": "Jinan"}, ...]}
"""
    )
    # Data arguments

    # Quick data input defaults (edit here for local runs)

    DEFAULT_TEACHER_DATA_PATH = "data_cache/MPLight/Jinan/flow0.jsonl"

    DEFAULT_TEACHER_MODEL = "MPLight"

    DEFAULT_TEACHER_DELAY = None

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for candidate selection and phase-bias calibration.",
    )

    parser.add_argument(

        "--teacher_data_path",

        type=str,

        default=DEFAULT_TEACHER_DATA_PATH,

        help="Path to teacher model data JSON file"

    )

    parser.add_argument(

        "--teacher_model",

        type=str,

        default=DEFAULT_TEACHER_MODEL,

        help="Teacher model name (e.g., MPLight, FRAP)"

    )

    parser.add_argument(

        "--teacher_delay",

        type=float,

        default=DEFAULT_TEACHER_DELAY,

        help="Teacher model average travel time (seconds). If not provided, will automatically extract from teacher data stats."

    )

    parser.add_argument(

        "--teachers_config",

        type=str,

        default=None,

        help="Path to JSON config. Two formats supported: (1) List format: one teacher per entry, run sequentially (one model per roadnet). "
             "(2) Joint format: {\"teacher_model\": \"MPLight\", \"roadnet_configs\": [{...}, ...]}, one model for all roadnets. Auto-detected by structure."

    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue a batch after a failed scenario. By default batches fail fast.",
    )



    # LLM arguments

    parser.add_argument(

        "--llm_api_endpoint",

        type=str,

        default=None,

        help="LLM API hostname only (e.g. comp.azure-api.net, api.jiekou.ai, yunwu.ai). Default: env LLM_API_ENDPOINT, else embedded/built-in host per --llm_backend (1/2/3/4).",

    )

    parser.add_argument(

        "--llm_api_key",

        type=str,

        default=None,

        help="LLM API key (or set LLM_API_KEY env var)"

    )

    parser.add_argument(

        "--llm_model",

        type=str,

        default=None,

        help="Model / Azure deployment name (or LLM_MODEL env). Default depends on LLM_BACKEND.",

    )

    parser.add_argument(
        "--llm_backend",
        type=str,
        default="1",
        choices=["1", "2", "3", "4"],
        help=(
            "Default 1: general OpenAI-compatible API (optional defaults in general_llm_embedded.py); "
            "2：COMP Azure APIM（comp_llm_embedded.py）；"
            "3: Jiekou.AI (optional jiekou_llm_embedded.py; api.jiekou.ai + /openai/v1/chat/completions). "
            "4: Yunwu (yunwu_llm_embedded.py; yunwu.ai + /v1/chat/completions). "
            "Omitting this option also selects 1 and ignores a stale shell LLM_BACKEND."
        ),
    )

    parser.add_argument(
        "--action_mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "phase_argmax",
            "keep_switch",
            "binary_keep_switch",
        ],
        help=(
            "Student decision rule. auto preserves the legacy behavior "
            "(EcoLight keep/switch; all other teachers all-phase argmax)."
        ),
    )



    # Evolution arguments

    parser.add_argument(

        "--population_size",

        type=int,

        default=5,

        help="Population size"

    )

    parser.add_argument(

        "--n_generations",

        type=int,

        default=5,

        help="Number of generations"

    )

    parser.add_argument(
        "--time_budget_seconds",
        type=float,
        default=None,
        help=(
            "Optional wall-clock search budget. A complete candidate that "
            "starts before the deadline may finish; no later candidate starts."
        ),
    )

    parser.add_argument(

        "--mutation_rate",

        type=float,

        default=1,

        help="Mutation rate"

    )



    # Constraint validation arguments

    parser.add_argument(

        "--max_constraint_retries",

        type=int,

        default=3,

        help="Maximum number of constraint validation retries with LLM feedback"

    )

    parser.add_argument(
        "--max_init_attempts",
        type=int,
        default=0,
        help="Max LLM attempts per individual during init (default 30). Use 0 for unlimited retries."
    )

    # Simulation arguments (for student_travel_time)

    parser.add_argument(

        "--enable_simulation",

        action="store_true",

        default=True,

        help="Enable simulation evaluation to compute student average travel time. Always enabled by default."

    )

    parser.add_argument(

        "--sim_num_step",

        type=int,

        default=None,

        help="Override simulation horizon (num_step). If not set, will use value from teacher data or config.json."

    )

    parser.add_argument(

        "--sim_seed",

        type=int,

        default=None,

        help="Override simulation seed. If not set, will use teacher data meta.seed if available."

    )



    # Other arguments

    parser.add_argument(
        "--score_mode",
        type=str,
        default="action_plus_perf",
        choices=["action_plus_perf", "action_only", "performance_only"],
        help="Scoring objective for evolution: action_plus_perf, action_only, or performance_only.",
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

    parser.add_argument(

        "--intersection_id",

        type=int,

        default=None,

        help="If specified, only optimize for this intersection"

    )

    parser.add_argument(

        "--max_timesteps",

        type=int,

        default=None,

        help="Maximum number of timesteps to use for evaluation"

    )

    parser.add_argument(

        "--use_intersection_specific_params",

        action="store_true",

        default=True,

        help="Use intersection-specific parameters (default: independent per intersection)"

    )

    parser.add_argument(

        "--use_slot_based_parent",

        action="store_true",

        default=True,

        help="Use slot-based parent: for slot i mutate individual at position i (default: True). Use --no_slot_based_parent to disable and use tournament selection instead."

    )

    parser.add_argument(

        "--no_slot_based_parent",

        action="store_false",

        dest="use_slot_based_parent",

        help="Disable slot-based parent; use tournament selection (best individual selected often)."

    )

    parser.add_argument(

        "--fit_bias_during_evolution",

        action="store_true",

        default=True,

        help="Fit per-intersection bias before evaluating each individual during evolution (default: True). Slower but better scores."

    )

    parser.add_argument(

        "--no_fit_bias_during_evolution",

        action="store_false",

        dest="fit_bias_during_evolution",

        help="Disable bias fitting during evolution (evaluate without bias, faster but lower scores)."

    )

    parser.add_argument(

        "--use_constraint_validation",

        action="store_true",

        default=True,

        help="Enable constraint validation with LLM retries for each new function (init + evolution). Default: True."

    )

    parser.add_argument(

        "--no_constraint_validation",

        action="store_false",

        dest="use_constraint_validation",

        help="Disable iterative TSC constraint review/repair; retain single-pass feature-key and real-state checks."

    )

    parser.add_argument(
        "--repair_rejected_candidates", action="store_true", default=False,
        help="Experimental: use an existing retry to repair the previous rejected program with its diagnostic.",
    )
    parser.add_argument(
        "--binary_margin_guidance", action="store_true", default=False,
        help="Experimental: explain calibrated action-value scale in EcoLight Native prompts.",
    )
    parser.add_argument(
        "--binary_margin_span_guard", action="store_true", default=False,
        help="Experimental: require the binary action margin to vary across source feature extremes.",
    )
    parser.add_argument(
        "--native_liveness_guard", action="store_true", default=False,
        help="Experimental: reject Native source rollouts with near-zero switching and severe ATT regression.",
    )
    parser.add_argument(
        "--evaluation_error_feedback", action="store_true", default=False,
        help="Experimental: include failed source evaluation reason in the next mutation prompt.",
    )

    parser.add_argument(

        "--bias_fit_max_iter",

        type=int,

        default=20,

        help="Maximum CMA-ES iterations for bias calibration (default: 20)."

    )

    parser.add_argument(

        "--bias_fit_bound",

        type=float,

        default=1.0,

        help="Bound for bias values during fitting (default: 1.0, bias in [-1, 1])."

    )

    parser.add_argument(

        "--fit_per_intersection_bias",

        action="store_true",

        default=False,

        help="Deprecated: post-evolution bias fitting is disabled; flag has no effect."

    )

    parser.add_argument(

        "--no_fit_per_intersection_bias",

        action="store_false",

        dest="fit_per_intersection_bias",

        help="Deprecated: post-evolution bias fitting is disabled (default)."

    )

    parser.add_argument(

        "--config_path",

        type=str,

        default="configs/traffic.json",

        help="Path to config.json file"

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default="./llm_sr_results",

        help="Output directory for results"

    )

    parser.add_argument(

        "--verbose",

        action="store_true",

        default=True,

        help="Verbose output"

    )



    args = parser.parse_args()

    apply_llm_backend_cli_choice(getattr(args, "llm_backend", None))
    set_seed(int(args.random_seed), use_torch=False)

    # score_mode has priority and rewrites weights for reproducible ablation setups.
    if args.score_mode == "action_only":
        args.action_weight = 1.0
        args.performance_weight = 0.0
    elif args.score_mode == "performance_only":
        args.action_weight = 0.0
        args.performance_weight = 1.0
    else:
        # Keep user-provided weights when action_plus_perf is selected.
        args.action_weight = float(args.action_weight)
        args.performance_weight = float(args.performance_weight)

    args.enable_simulation = True

    config = load_config(args.config_path)

    llm_api_endpoint, llm_api_key, llm_model = resolve_llm_connection_settings(
        args.llm_api_endpoint,
        args.llm_api_key,
        args.llm_model,
    )

    if not llm_api_key:
        raise ValueError(llm_api_key_missing_message())

    from LLM_SR_DISTILLATION_method_v2.llm_runtime_info import print_llm_startup_banner

    print_llm_startup_banner(llm_api_endpoint, llm_model, cli_llm_model=args.llm_model)

    if args.teachers_config:
        with open(args.teachers_config, 'r', encoding='utf-8') as f:
            teachers_cfg = json.load(f)

        def _run_joint_from_dict(joint_cfg: Dict, idx: Optional[int] = None, total: Optional[int] = None) -> None:
            roadnet_configs = joint_cfg.get("roadnet_configs", [])
            if not joint_cfg.get("teacher_model") or not isinstance(roadnet_configs, list) or not roadnet_configs:
                raise ValueError("Joint config must have 'teacher_model' and non-empty 'roadnet_configs'")
            if "output_dir" in joint_cfg:
                args.output_dir = joint_cfg["output_dir"]
            prefix = ""
            if idx is not None and total is not None:
                prefix = f"\n>>> Joint Run {idx + 1}/{total}: "
            print(prefix + f"[Mode: Joint] Teacher={joint_cfg['teacher_model']}, roadnets={len(roadnet_configs)}\n")
            run_multi_roadnet_distillation(
                roadnet_configs=roadnet_configs,
                teacher_model=joint_cfg["teacher_model"],
                args=args,
                config=config,
                llm_api_endpoint=llm_api_endpoint,
                llm_api_key=llm_api_key,
                llm_model=joint_cfg.get("llm_model") or llm_model,
            )

        # Joint mode: one teacher, all roadnets, one model
        if isinstance(teachers_cfg, dict) and "roadnet_configs" in teachers_cfg and "teacher_model" in teachers_cfg:
            _run_joint_from_dict(teachers_cfg)
        else:
            teacher_runs = teachers_cfg if isinstance(teachers_cfg, list) else []
            if not teacher_runs:
                raise ValueError(
                    "teachers_config must be either: (1) a non-empty JSON array, "
                    "or (2) a dict with 'teacher_model' and 'roadnet_configs'."
                )

            # List-of-joint: multiple teachers, each teacher -> one unified student across all roadnets
            if all(isinstance(x, dict) and ("roadnet_configs" in x) and ("teacher_model" in x) for x in teacher_runs):
                print(f"\n[Mode: Joint x{len(teacher_runs)}] Multiple teachers, each trains one unified model across its roadnets.\n")
                for idx, joint_cfg in enumerate(teacher_runs):
                    _run_joint_from_dict(joint_cfg, idx=idx, total=len(teacher_runs))
                print(f"\nAll {len(teacher_runs)} joint teacher distillation runs completed.")
            else:
                # Sequential mode: one teacher per entry, one model per roadnet
                print(f"\n[Mode: Sequential] Running distillation for {len(teacher_runs)} teacher(s), one model per roadnet.\n")
                failures = []
                for idx, run_cfg in enumerate(teacher_runs):
                    teacher_data_path = run_cfg.get("teacher_data_path")
                    teacher_model = run_cfg.get("teacher_model")
                    teacher_delay = run_cfg.get("teacher_delay")
                    if not teacher_data_path or not teacher_model:
                        raise ValueError(
                            f"Entry {idx} in teachers_config must have 'teacher_data_path' and 'teacher_model'"
                        )
                    scenario = run_cfg.get("scenario")
                    print(f"\n>>> Run {idx + 1}/{len(teacher_runs)}: {teacher_model}" + (f" [{scenario}]" if scenario else "") + f" ({teacher_data_path})\n")
                    try:
                        run_distillation_single(
                            teacher_data_path=teacher_data_path,
                            teacher_model=teacher_model,
                            teacher_delay=teacher_delay,
                            args=args,
                            config=config,
                            llm_api_endpoint=llm_api_endpoint,
                            llm_api_key=llm_api_key,
                            llm_model=llm_model,
                            scenario=scenario,
                            simulation_seed_override=run_cfg.get(
                                "simulation_seed",
                                run_cfg.get("sim_seed"),
                            ),
                            random_seed_override=run_cfg.get("random_seed"),
                            derive_seed_from_rollout=True,
                            time_budget_seconds_override=run_cfg.get(
                                "search_time_budget_seconds",
                                run_cfg.get("time_budget_seconds"),
                            ),
                        )
                    except Exception as e:
                        if not args.continue_on_error:
                            raise
                        failures.append(
                            {
                                "index": idx,
                                "teacher_model": teacher_model,
                                "scenario": scenario,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        print(
                            f"\n[FAILED] Run {idx + 1}/{len(teacher_runs)} failed "
                            f"({teacher_model}" + (f" [{scenario}]" if scenario else "") + f"): "
                            f"{type(e).__name__}: {e}\n"
                        )
                        continue
                if failures:
                    summary = "; ".join(
                        f"#{item['index'] + 1} {item['teacher_model']}"
                        + (f"/{item['scenario']}" if item["scenario"] else "")
                        + f": {item['error']}"
                        for item in failures
                    )
                    raise RuntimeError(
                        f"{len(failures)}/{len(teacher_runs)} batch runs failed: {summary}"
                    )
                print(f"\nAll {len(teacher_runs)} teacher distillation runs completed.")
    else:
        # Single teacher: original behavior
        run_distillation_single(
            teacher_data_path=args.teacher_data_path,
            teacher_model=args.teacher_model,
            teacher_delay=args.teacher_delay,
            args=args,
            config=config,
            llm_api_endpoint=llm_api_endpoint,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )


if __name__ == "__main__":

    main()
