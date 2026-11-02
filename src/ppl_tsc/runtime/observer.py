"""Observe existing source decisions through explicit simulator callbacks."""

import copy
from ppl_tsc.runtime.lanes import Recorder


class SourceObserver:
    def __init__(self):
        self.recorder = Recorder()
        self.last = {}
        self.snapshots = {}

    def observe(self, intersection, states, selected):
        self.last[intersection] = {
            "states": copy.deepcopy(states),
            "requested_phase": int(selected),
        }

    def tick(self, inter, interval):
        self.recorder.tick(inter, interval)
        for lane in inter.n_in_lane_id:
            clock = self.recorder.lanes.get((inter.inter_idx, lane))
            if clock is None:
                continue
            if clock["without_green"] <= 0 or inter.inter_idx not in self.last:
                continue
            key = (inter.inter_idx, lane)
            old = self.snapshots.get(key)
            if old is None or clock["without_green"] > old["continuous_wait_without_green_seconds"]:
                self.snapshots[key] = dict(
                    intersection=inter.inter_idx,
                    lane=lane,
                    continuous_wait_without_green_seconds=clock["without_green"],
                    source_tick_time=float(inter.eng.get_current_time()),
                    **self.last[inter.inter_idx],
                )
            if len(self.snapshots) > 12:
                del self.snapshots[
                    min(
                        self.snapshots,
                        key=lambda k: self.snapshots[k]["continuous_wait_without_green_seconds"],
                    )
                ]

    def finish(self, evaluator, utility_function, per_intersection_bias):
        summary = self.recorder.summarize()
        examples = []
        stats = getattr(evaluator.feature_extractor, "feature_stats", None)
        for row in sorted(
            self.snapshots.values(),
            key=lambda r: r["continuous_wait_without_green_seconds"],
            reverse=True,
        )[:3]:
            states = row.pop("states")
            phase_bias = (per_intersection_bias or {}).get(
                row["intersection"], (per_intersection_bias or {}).get(str(row["intersection"]), {})
            ) or {}
            from ppl_tsc.runtime.decision_values import decision_values

            raw = decision_values(utility_function, states, stats)
            if utility_function.action_mode == "binary_keep_switch":
                cur = next(iter(states.values()))["current_phase"]
                nxt = next((p for p in raw if p != cur), cur)
                view = {
                    "current_phase": cur,
                    "next_phase": nxt,
                    "binary_features": next(iter(states.values())).get("binary_features"),
                }
            else:
                view = states
            adjusted = {
                p: q + phase_bias.get(p, phase_bias.get(str(p), 0.0)) for p, q in raw.items()
            }
            from ppl_tsc.runtime.response_probes import inspect_response

            try:
                row["joint_response"] = inspect_response(
                    utility_function, states, stats, phase_bias
                )
            except Exception as exc:
                row["joint_response"] = {
                    "status": "incomplete",
                    "error": type(exc).__name__ + ": " + str(exc),
                    "no_hard_rejection": True,
                }
            row.update(
                source_decision_state=view,
                raw_utilities=raw,
                adjusted_utilities=adjusted,
                phase_bias=phase_bias,
                raw_choice=max(raw, key=raw.get),
                adjusted_choice=max(adjusted, key=adjusted.get),
            )
            if row["adjusted_choice"] != row["requested_phase"]:
                raise ValueError(
                    "Lane diagnostic recomputed action differs from recorded policy request"
                )
            examples.append(row)
        return dict(
            summary=summary,
            examples=examples,
            extra_episodes=0,
            extra_teacher_queries=0,
            source_only=True,
            sampling="Per engine tick; utility state from most recent decision (at most one action interval earlier).",
        )
