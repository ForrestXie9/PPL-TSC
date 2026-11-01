"""Paired frozen-policy evaluation for one teacher/scenario/flow/condition."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


# All imports resolve inside this standalone artifact.
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from LLM_SR_DISTILLATION_method_v2.feature_extractor import (  # noqa: E402
    build_feature_extractor,
)
from LLM_SR_DISTILLATION_method_v2.function_utility import (  # noqa: E402
    FunctionUtility,
)
from LLM_SR_DISTILLATION_method_v2.simulator import (  # noqa: E402
    StudentSimulator,
)
from utilities.utils import (  # noqa: E402
    copy_model_params,
    get_agent,
    get_config,
    set_seed,
    set_thread,
)

from ood_env import OODTSCEnv  # noqa: E402



def normalize_bias(raw: Any) -> dict[int, dict[int, float]]:
    if not isinstance(raw, dict):
        return {}
    return {
        int(intersection): {
            int(phase): float(value) for phase, value in phases.items()
        }
        for intersection, phases in raw.items()
        if isinstance(phases, dict)
    }


def extract_metrics(info: dict[str, Any], env: OODTSCEnv) -> dict[str, float]:
    def value(key: str) -> float:
        raw = info.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            return float(raw[0])
        if isinstance(raw, np.ndarray) and raw.size:
            return float(raw.reshape(-1)[0])
        return float(raw or 0.0)

    return {
        "ATT": value("world_2_average_travel_time"),
        "AQL": value("world_2_average_queue_length"),
        "AD": value("world_2_average_delay"),
        "Throughput": value("world_2_average_throughput"),
        "finished_vehicle_count": float(
            env.eng.get_finished_vehicle_cnt()
        ),
        "running_vehicle_count": float(env.eng.get_vehicle_count()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    set_thread()


def build_config(
    traffic_config_path: Path,
    teacher: str,
    scenario: str,
    flow_idx: int,
    seed: int,
    num_step: int,
    data_dir: Path,
) -> dict[str, Any]:
    config = get_config(str(traffic_config_path))
    config.update(
        {
            "cur_agent": teacher,
            # These helpers replay the historical rebuttal protocol. New
            # training/collection explicitly use the corrected semantics.
            "waiting_time_semantics": "legacy_per_read",
            "save_result": False,
            "saveReplay": False,
            "render": False,
            "device": "cpu",
            "inter_name": scenario,
            "flow_idx": int(flow_idx),
            "dir": str(data_dir.resolve()) + "/",
            "roadnetFile": "roadnet.json",
            "flowFile": f"flow_{int(flow_idx)}.json",
            "seed": int(seed),
            "num_step": int(num_step),
            "current_episode_idx": 0,
            "current_episode_step_idx": 0,
        }
    )
    return config


def run_episode(
    config: dict[str, Any],
    incident: dict[str, Any] | None,
    action_builder: Callable[
        [OODTSCEnv, Any, int], list[int]
    ],
    setup: Callable[[OODTSCEnv], Any] | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    env = OODTSCEnv(copy.deepcopy(config), incident=incident)
    controller = setup(env) if setup is not None else None
    observations = env.reset()
    info: dict[str, Any] = {}
    action_steps = 0
    started = time.time()
    for current_step in range(
        0,
        int(config["num_step"]),
        int(config["action_interval"]),
    ):
        env.config["current_episode_step_idx"] = current_step
        actions = action_builder(
            env, controller if controller is not None else observations,
            current_step,
        )
        observations, _reward, done, info = env.step(actions)
        if controller is None:
            # Teacher actions consume observations directly. Student actions
            # ignore this value and rebuild structured phase states from env.
            pass
        action_steps += 1
        if all(done):
            break
        if controller is not None and isinstance(controller, dict):
            controller["latest_observations"] = observations
    if not info:
        info = env._get_info()
    return {
        "metrics": extract_metrics(info, env),
        "action_steps": action_steps,
        "runtime_seconds": time.time() - started,
        "incident": env.incident_report(),
    }


def load_teacher_agents(
    env: OODTSCEnv,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint type {type(checkpoint)}: "
            f"{checkpoint_path}"
        )
    if checkpoint.get('parameter_sharing') is True or 'shared_local' in checkpoint:
        raise ValueError('Experimental Shared checkpoints are not accepted by the original-family loader')
    agent_class = get_agent(config["cur_agent"])
    agents: list[Any] = []
    if config["cur_agent"] in {"CoLight", "MPLight"}:
        env.n_agent = [None] * env.n
    for idx in range(env.n):
        agent = agent_class(config, env, idx)
        agents.append(agent)
        if config["cur_agent"] in {"CoLight", "MPLight"}:
            env.n_agent[idx] = agent
    for idx, agent in enumerate(agents):
        local_key = f"agent_{idx}_local"
        target_key = f"agent_{idx}_target"
        if not hasattr(agent, "network_local"):
            raise TypeError(
                f"{config['cur_agent']} agent {idx} lacks network_local"
            )
        if local_key not in checkpoint:
            raise KeyError(
                f"{local_key} missing from {checkpoint_path}"
            )
        agent.network_local.load_state_dict(checkpoint[local_key])
        agent.network_local.eval()
        if hasattr(agent, "network_target"):
            if target_key in checkpoint:
                agent.network_target.load_state_dict(
                    checkpoint[target_key]
                )
            else:
                copy_model_params(
                    agent.network_local, agent.network_target
                )
            agent.network_target.eval()
    return {"agents": agents, "latest_observations": None}


def teacher_action_builder(
    env: OODTSCEnv,
    controller: dict[str, Any],
    _current_step: int,
) -> list[int]:
    observations = controller.get("latest_observations")
    if observations is None:
        observations = env.n_obs
    return [
        int(agent.pick_action(observations, on_training=False))
        for agent in controller["agents"]
    ]


def build_student_controller(
    config: dict[str, Any],
    teacher: str,
    deployment: dict[str, Any],
    expected_action_mode: str | None = None,
) -> dict[str, Any]:
    from ppl_tsc_current.shared_lane import restore_contract
    config=restore_contract(copy.deepcopy(config),deployment.get("run_config",{}))
    if deployment.get("run_config",{}).get("method_profile")=="full_100_256_v1":
        import hashlib
        raw_bias=deployment.get("per_intersection_bias") or {}
        if any(float(v)!=0. for phases in raw_bias.values() for v in phases.values()):
            roadnet=Path(config['dir'])/config.get('roadnetFile','roadnet.json')
            if hashlib.sha256(roadnet.read_bytes()).hexdigest()!=deployment['run_config'].get('source_roadnet_sha256'):
                raise ValueError('Current source biases require the identical roadnet')
    teacher_cfg = config.get(teacher)
    if not isinstance(teacher_cfg, dict):
        raise KeyError(f"Teacher {teacher} missing from traffic config")
    observation_features = list(
        teacher_cfg.get("observation_feature_list", [])
    )
    feature_stats = deployment.get("feature_stats") or {}
    feature_contract = deployment.get("run_config", {}).get("feature_contract_version")
    extractor = build_feature_extractor(
        teacher_model=teacher,
        observation_feature_list=observation_features,
        feature_stats=feature_stats,
        normalize_features=feature_contract in {"normalized_phase_features_v1", "normalized_phase_features_v2"},
        clip_normalized=feature_contract == "normalized_phase_features_v1",
    )
    extractor.feature_stats = feature_stats
    saved_action_mode = str(
        deployment.get("action_mode") or "phase_argmax"
    )
    if (
        expected_action_mode is not None
        and saved_action_mode != expected_action_mode
    ):
        raise ValueError(
            f"Deployment action mode is {saved_action_mode!r}, expected "
            f"{expected_action_mode!r}"
        )
    utility = FunctionUtility(
        deployment["function_code"],
        feature_extractor=extractor,
        feature_stats=feature_stats,
        action_mode=saved_action_mode,
    )
    return {
        "helper": StudentSimulator(
            copy.deepcopy(config),
            observation_feature_list=observation_features,
        ),
        "utility": utility,
        "feature_stats": feature_stats,
        "bias": normalize_bias(
            deployment.get("per_intersection_bias")
        ),
        "latest_observations": None,
    }


def student_action_builder(
    env: OODTSCEnv,
    controller: dict[str, Any],
    _current_step: int,
) -> list[int]:
    actions: list[int] = []
    for intersection_idx in range(env.n):
        state_dict = controller["helper"].build_phase_state_dict(
            env, intersection_idx
        )
        actions.append(
            int(
                controller["utility"].select_action(
                    state_dict,
                    feature_stats=controller["feature_stats"],
                    phase_bias=controller["bias"].get(
                        intersection_idx
                    ),
                    raise_on_error=True,
                )
            )
        )
    return actions
