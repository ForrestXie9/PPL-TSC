"""CityFlow TSC environment with an auditable temporary approach closure."""

from __future__ import annotations

from typing import Any

from env import TSCEnv


class OODTSCEnv(TSCEnv):
    """Preserve the original environment logic and add a per-tick incident."""

    def __init__(
        self,
        config: dict[str, Any],
        incident: dict[str, Any] | None = None,
    ):
        self.incident = dict(incident or {})
        self._incident_unique_vehicles: set[str] = set()
        self._incident_speed_set_calls = 0
        self._incident_active_ticks = 0
        self._incident_first_applied_time: float | None = None
        self._incident_last_applied_time: float | None = None
        super().__init__(config)

    def reset(self):
        self._incident_unique_vehicles = set()
        self._incident_speed_set_calls = 0
        self._incident_active_ticks = 0
        self._incident_first_applied_time = None
        self._incident_last_applied_time = None
        return super().reset()

    def _apply_incident(self) -> None:
        if not self.incident:
            return
        current_time = float(self.eng.get_current_time())
        start_time = float(self.incident["start_time"])
        end_time = float(self.incident["end_time"])
        if not start_time <= current_time < end_time:
            return
        lane_ids = self.incident.get("lane_ids")
        if not lane_ids:
            lane_ids = [self.incident["lane_id"]]
        lane_ids = [str(lane_id) for lane_id in lane_ids]
        target_speed = float(self.incident.get("target_speed", 0.0))
        lane_vehicle_map = self.eng.get_lane_vehicles()
        self._incident_active_ticks += 1
        for lane_id in lane_ids:
            for vehicle_id in lane_vehicle_map.get(lane_id, []):
                self.eng.set_vehicle_speed(vehicle_id, target_speed)
                self._incident_unique_vehicles.add(str(vehicle_id))
                self._incident_speed_set_calls += 1
                if self._incident_first_applied_time is None:
                    self._incident_first_applied_time = current_time
                self._incident_last_applied_time = current_time

    def step(self, n_action):
        self.last_action, self.current_action = self.current_action, n_action
        for _ in range(self.config["action_interval"]):
            for inter, action in zip(self.n_intersection, n_action):
                inter.step(action, self.interval)
            self._apply_incident()
            self.eng.next_step()
            if self.config["render"]:
                self.render(inter_idx=0)

        n_next_obs = self._get_n_obs()
        n_reward = self._get_n_reward()
        n_done = self._get_n_done()
        info = self._get_info()
        self.n_obs = n_next_obs

        if (
            self.config["current_episode_step_idx"] % 60 == 0
            and self.config["current_episode_step_idx"] > 0
        ):
            metric_features = self.config[
                self.config["cur_agent"]
            ]["metric_feature_list"]
            if "world_2_average_queue_length" in metric_features:
                self._update_average_queue_length()
            if "world_2_average_delay" in metric_features:
                self._update_average_delay()
            if "world_2_average_throughput" in metric_features:
                self._update_average_throughput()
        return n_next_obs, n_reward, n_done, info

    def incident_report(self) -> dict[str, Any]:
        if not self.incident:
            return {
                "configured": False,
                "active_ticks": 0,
                "speed_set_calls": 0,
                "unique_affected_vehicles": 0,
            }
        return {
            "configured": True,
            "road_id": self.incident.get("road_id"),
            "lane_id": str(self.incident["lane_id"]),
            "lane_ids": [
                str(lane_id)
                for lane_id in self.incident.get(
                    "lane_ids", [self.incident["lane_id"]]
                )
            ],
            "start_time": float(self.incident["start_time"]),
            "end_time": float(self.incident["end_time"]),
            "target_speed": float(
                self.incident.get("target_speed", 0.0)
            ),
            "active_ticks": int(self._incident_active_ticks),
            "speed_set_calls": int(self._incident_speed_set_calls),
            "unique_affected_vehicles": len(
                self._incident_unique_vehicles
            ),
            "first_applied_time": self._incident_first_applied_time,
            "last_applied_time": self._incident_last_applied_time,
        }
