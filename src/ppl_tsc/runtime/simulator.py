"""
Simulation Module
Run a student policy in CityFlow and return travel time and supporting traffic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from math import pi

import numpy as np

from env import TSCEnv


def _get_direction_code_from_roadlink(roadlink, inter, env) -> Optional[str]:
    """
    Match collect_teacher.py: map a roadlink to a direction code such as 'WT', 'WL', 'ER'.
    """
    start_road_obj = env.id2road.get(roadlink.startroad_id)
    if start_road_obj is None:
        return None

    road_dir = start_road_obj.road_direction

    # Convert angle to cardinal direction of start road
    if road_dir >= 7 * pi / 4 or road_dir < pi / 4:
        start_direction = "E"
    elif road_dir >= pi / 4 and road_dir < 3 * pi / 4:
        start_direction = "N"
    elif road_dir >= 3 * pi / 4 and road_dir < 5 * pi / 4:
        start_direction = "W"
    elif road_dir >= 5 * pi / 4 and road_dir < 7 * pi / 4:
        start_direction = "S"
    else:
        start_direction = "E"

    link_type = None
    for r_link in inter.inter_dict.get("roadLinks", []):
        if (
            r_link.get("startRoad") == roadlink.startroad_id
            and r_link.get("endRoad") == roadlink.endroad_id
        ):
            link_type = r_link.get("type", "")
            break

    if link_type == "go_straight":
        turn_type = "T"
    elif link_type == "turn_left":
        turn_type = "L"
    elif link_type == "turn_right":
        turn_type = "R"
    else:
        turn_type = "T"

    return f"{start_direction}{turn_type}"


@dataclass
class SimulationResult:
    """
    Aggregate KPIs from one simulation episode.

    The evaluator uses avg_travel_time for the configured control score;
    behavioral fidelity is evaluated separately on source teacher observations.
    """

    avg_travel_time: float
    avg_delay: Optional[float] = None
    avg_queue_length: Optional[float] = None
    avg_throughput: Optional[float] = None
    per_intersection_avg_travel_time: Optional[Dict[int, float]] = None


class StudentSimulator:
    """
    Run a UtilityFunction-based policy in TSCEnv.

    Notes:
    - TSCEnv does not use a done signal; we step for a fixed horizon using config['num_step'] and config['action_interval'].
    - We compute average delay using env._info_functions['world_2_average_delay']() at the end of the episode.
    """

    def __init__(self, env_config: Dict[str, Any], observation_feature_list: List[str]):
        self.env_config = dict(env_config)
        self.observation_feature_list = list(observation_feature_list or [])

    def run_episode(self, policy, *, on_intersection_step=None) -> SimulationResult:
        """
        Run one episode and return aggregate metrics.

        policy: callable(env, inter_idx) -> action(int)
        """
        env = TSCEnv(self.env_config)
        env.intersection_step_observer = on_intersection_step
        env.reset()

        num_step = int(self.env_config.get("num_step", 3600))
        action_interval = int(self.env_config.get("action_interval", 20))
        # Ensure the key exists for metric update logic in env.step()
        self.env_config["current_episode_step_idx"] = 0

        per_inter_sum_tt = np.zeros(env.n, dtype=float)
        per_inter_count = np.zeros(env.n, dtype=float)

        for t in range(0, num_step, action_interval):
            self.env_config["current_episode_step_idx"] = t
            actions = []
            for inter_idx in range(env.n):
                actions.append(int(policy(env, inter_idx)))
            env.step(actions)

            for inter_idx in range(env.n):
                inter = env.n_intersection[inter_idx]
                sum_tt = env._info_functions["inter_2_sum_travel_time_since_last_action"](inter)
                num_passed = env._info_functions["inter_2_num_passed_vehicle_since_last_action"](
                    inter
                )
                per_inter_sum_tt[inter_idx] += float(np.asarray(sum_tt))
                per_inter_count[inter_idx] += float(np.asarray(num_passed))

        # Read metrics directly from env info functions
        info = env._get_info()
        tt_mean = float(info.get("world_2_average_travel_time", (0.0, 0.0))[0])
        delay_mean = float(info.get("world_2_average_delay", (0.0, 0.0))[0])
        ql_mean = float(info.get("world_2_average_queue_length", (0.0, 0.0))[0])
        th_mean = float(info.get("world_2_average_throughput", (0.0, 0.0))[0])

        per_inter_avg_tt = {
            int(i): (per_inter_sum_tt[i] / per_inter_count[i] if per_inter_count[i] > 0 else 0.0)
            for i in range(env.n)
        }

        return SimulationResult(
            avg_travel_time=tt_mean,
            avg_delay=delay_mean,
            avg_queue_length=ql_mean,
            avg_throughput=th_mean,
            per_intersection_avg_travel_time=per_inter_avg_tt,
        )

    def build_phase_state_dict(self, env: TSCEnv, inter_idx: int) -> Dict[int, Dict[str, Any]]:
        """
        Build a minimal structured state_dict compatible with FeatureExtractor for each phase.

        We populate:
        - current_phase: int
        - controlled_directions: list of direction codes (e.g., ["WT", "WL"])
        - features_by_direction: {direction_code: {raw_key: value}}

        Supported raw keys:
        - pressure: aggregated by direction from lanelink_2_pressure
        - num_vehicle: aggregated by direction from inlane_2_num_vehicle
        - num_waiting_vehicle: aggregated by direction from inlane_2_num_waiting_vehicle
        - waiting_time / sum_waiting_time: aggregated by direction from inlane_2_sum_waiting_time
        """
        inter = env.n_intersection[inter_idx]
        num_phases = len(inter.n_phase)

        # Precompute feature vectors if needed
        need_pressure = any("pressure" in f for f in self.observation_feature_list)
        need_num_vehicle = any("num_vehicle" in f for f in self.observation_feature_list)
        need_num_waiting = any("num_waiting_vehicle" in f for f in self.observation_feature_list)
        need_waiting_time = any(
            ("sum_waiting_time" in f) or ("waiting_time" in f)
            for f in self.observation_feature_list
        )

        lanelink_pressure = env._lanelink_2_pressure(inter) if need_pressure else None
        inlane_num_vehicle = env._inlane_2_num_vehicle(inter) if need_num_vehicle else None
        inlane_num_waiting_vehicle = (
            env._inlane_2_num_waiting_vehicle(inter) if need_num_waiting else None
        )
        inlane_sum_waiting_time = (
            env._inlane_2_sum_waiting_time(inter) if need_waiting_time else None
        )

        # Map roadlink -> direction code (WT/WL/ET/...)
        roadlink_direction_codes: List[Optional[str]] = []
        for roadlink in inter.n_roadlink:
            roadlink_direction_codes.append(_get_direction_code_from_roadlink(roadlink, inter, env))

        # Map lanelink index -> direction
        lanelink_to_direction: Dict[int, str] = {}
        lanelink_idx = 0
        for rl_idx, roadlink in enumerate(inter.n_roadlink):
            direction_code = roadlink_direction_codes[rl_idx] or "Unknown"
            for _ in roadlink.n_lanelink_id:
                lanelink_to_direction[lanelink_idx] = direction_code
                lanelink_idx += 1

        # Map inlane index -> direction (first matched roadlink)
        lane_id_to_direction: Dict[str, str] = {}
        for rl_idx, roadlink in enumerate(inter.n_roadlink):
            direction_code = roadlink_direction_codes[rl_idx]
            if not direction_code:
                continue
            for startlane_id in roadlink.n_startlane_id:
                if startlane_id not in lane_id_to_direction:
                    lane_id_to_direction[startlane_id] = direction_code

        inlane_to_direction: Dict[int, str] = {}
        for idx, lane_id in enumerate(inter.n_in_lane_id):
            inlane_to_direction[idx] = lane_id_to_direction.get(lane_id, "Unknown")

        if self.env_config.get("direction_feature_semantics") == "compatible_shared_lane_mean":
            from ppl_tsc.data.lanes import lane_directions

            mapping = lane_directions(inter.n_roadlink, roadlink_direction_codes)
            inlane_to_directions = {
                idx: mapping.get(lane, []) for idx, lane in enumerate(inter.n_in_lane_id)
            }
        else:
            inlane_to_directions = {
                idx: [direction] for idx, direction in inlane_to_direction.items()
            }

        # Build features_by_direction (direction -> {feature: value})
        features_by_direction: Dict[str, Dict[str, List[float]]] = {}

        def _append(direction: str, key: str, value: float) -> None:
            if direction == "Unknown":
                return
            if direction not in features_by_direction:
                features_by_direction[direction] = {}
            if key not in features_by_direction[direction]:
                features_by_direction[direction][key] = []
            features_by_direction[direction][key].append(float(value))

        if lanelink_pressure is not None:
            for idx, val in enumerate(np.asarray(lanelink_pressure).reshape(-1)):
                _append(lanelink_to_direction.get(idx, "Unknown"), "pressure", val)

        if inlane_num_vehicle is not None:
            for idx, val in enumerate(np.asarray(inlane_num_vehicle).reshape(-1)):
                for direction in inlane_to_directions.get(idx, []):
                    _append(direction, "num_vehicle", val)

        if inlane_num_waiting_vehicle is not None:
            for idx, val in enumerate(np.asarray(inlane_num_waiting_vehicle).reshape(-1)):
                for direction in inlane_to_directions.get(idx, []):
                    _append(direction, "num_waiting_vehicle", val)

        if inlane_sum_waiting_time is not None:
            for idx, val in enumerate(np.asarray(inlane_sum_waiting_time).reshape(-1)):
                for direction in inlane_to_directions.get(idx, []):
                    _append(direction, "waiting_time", val)
                for direction in inlane_to_directions.get(idx, []):
                    _append(direction, "sum_waiting_time", val)

        # Reduce lists to mean values
        features_by_direction_mean: Dict[str, Dict[str, float]] = {}
        for direction, feats in features_by_direction.items():
            features_by_direction_mean[direction] = {}
            for key, values in feats.items():
                if values:
                    features_by_direction_mean[direction][key] = float(np.mean(values))

        # Build controlled_directions per phase
        phase_controlled_directions: Dict[int, List[str]] = {}
        for phase_id in range(num_phases):
            controlled = set()
            for roadlink_idx in inter.n_phase[phase_id].n_available_roadlink_idx:
                direction_code = roadlink_direction_codes[roadlink_idx]
                if direction_code:
                    controlled.add(direction_code)
            phase_controlled_directions[phase_id] = sorted(list(controlled))

        # EcoLight's native network does not score traffic phases.  It scores
        # two binary actions from the exact global counts currently facing a
        # green or red signal.  Preserve those two inputs in every phase state
        # so the aligned symbolic binary head sees the same information.
        current_phase = int(inter.current_phase)
        green_vehicle = 0.0
        red_vehicle = 0.0
        if inlane_num_vehicle is not None:
            lane_values = np.asarray(inlane_num_vehicle, dtype=float).reshape(-1)
            phase_lane_mask = np.asarray(
                inter.phase_2_passable_lane_idx[current_phase],
                dtype=float,
            ).reshape(-1)
            if lane_values.shape == phase_lane_mask.shape:
                green_vehicle = float(np.sum(phase_lane_mask * lane_values))
                red_vehicle = float(np.sum((1.0 - phase_lane_mask) * lane_values))

        binary_features = {
            "green_vehicle": green_vehicle,
            "red_vehicle": red_vehicle,
        }

        state_dict: Dict[int, Dict[str, Any]] = {}
        for phase_id in range(num_phases):
            controlled = phase_controlled_directions.get(phase_id, [])
            phase_features_by_direction = {}
            for direction in controlled:
                if direction in features_by_direction_mean:
                    phase_features_by_direction[direction] = features_by_direction_mean[direction]
            state_dict[phase_id] = {
                "current_phase": current_phase,
                "controlled_directions": controlled,
                "features_by_direction": phase_features_by_direction,
                "binary_features": dict(binary_features),
            }

        return state_dict
