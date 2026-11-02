"""Bounded source-episode diagnostics; no extra episodes or teacher queries."""

import copy
import math
from collections import Counter
from ppl_tsc.runtime.phase_summary import summarize, decompose


class _DirectionRecorder:
    def __init__(self, action_interval=20, limit=3):
        self.interval = int(action_interval)
        self.limit = int(limit)
        self.steps = {}
        self.runs = {}
        self.best = {}
        self.errors = []
        self.decisions = 0

    def observe(self, intersection, states, requested_action):
        try:
            self._observe(intersection, states, requested_action)
        except Exception as exc:
            if len(self.errors) < 3:
                self.errors.append(type(exc).__name__ + ": " + str(exc))

    def _observe(self, intersection, states, requested_action):
        # Read only the state already passed to the policy. Never query env again.
        step = self.steps.get(intersection, 0)
        self.steps[intersection] = step + 1
        self.decisions += 1
        current = int(next(iter(states.values()))["current_phase"])
        covered = set(states[current].get("controlled_directions", []))
        directions = {}
        for state in states.values():
            for direction, features in state.get("features_by_direction", {}).items():
                directions.setdefault(direction, features)
        active = set()
        for direction, features in directions.items():
            # Pressure is signed and is not a vehicle count or a queue length.
            key = next(
                (k for k in ("num_waiting_vehicle", "num_vehicle", "pressure") if k in features),
                None,
            )
            if key is None:
                continue
            value = float(features[key])
            if not math.isfinite(value):
                raise ValueError("Nonfinite diagnostic feature")
            ident = (intersection, direction, key)
            if value <= 0 or direction in covered:
                self.runs.pop(ident, None)
                continue
            active.add(ident)
            old = self.runs.get(ident)
            run = (
                {"count": old["count"] + 1, "first": old["first"]}
                if old
                else {"count": 1, "first": value}
            )
            self.runs[ident] = run
            if run["count"] < 3:
                continue
            growth = value - run["first"]
            relative_growth = growth / max(abs(run["first"]), 1e-12)
            rank = (run["count"], relative_growth)
            previous = self.best.get(ident)
            if previous is not None and rank <= previous["_rank"]:
                continue
            self.best[ident] = dict(
                _rank=rank,
                intersection=intersection,
                decision_index=step,
                source_time_seconds=step * self.interval,
                current_phase=current,
                requested_phase=int(requested_action),
                direction=direction,
                feature=key,
                first_value=run["first"],
                latest_value=value,
                growth=growth,
                consecutive_observations=run["count"],
                states=copy.deepcopy(states),
            )
            # Bounded retained state even with many intersections/directions.
            if len(self.best) > 12:
                worst = min(self.best, key=lambda k: self.best[k]["_rank"])
                del self.best[worst]
        for ident in list(self.runs):
            if ident[0] == intersection and ident not in active:
                del self.runs[ident]

    def finish(self, evaluator, utility, bias):
        examples = []
        # Prefer different directions, cap at 3 globally; no target information.
        for item in sorted(self.best.values(), key=lambda x: x["_rank"], reverse=True)[
            : self.limit
        ]:
            row = {k: v for k, v in item.items() if k not in {"_rank", "states"}}
            try:
                cached = evaluator._build_cached_bias_consistency(
                    utility, [(item["states"], item["requested_phase"])]
                )
                mode, _, values = cached.samples[0]
                inter_bias = (bias or {}).get(
                    item["intersection"], (bias or {}).get(str(item["intersection"]), {})
                ) or {}
                inter_bias = {int(k): float(v) for k, v in inter_bias.items()}
                if mode == "phase_argmax":
                    raw = dict(values)
                else:
                    current, following, keep, switch = values
                    raw = {current: keep, following: switch}
                adjusted = {p: q + inter_bias.get(p, 0.0) for p, q in raw.items()}
                selected = max(adjusted, key=adjusted.get)
                if selected != item["requested_phase"]:
                    raise ValueError("Recorded request disagrees with recomputed legal action")
                row.update(
                    raw_utilities=raw,
                    adjusted_utilities=adjusted,
                    legal_scored_phases=list(raw),
                    source_state=item["states"],
                )
                examples.append(row)
            except Exception as exc:
                if len(self.errors) < 3:
                    self.errors.append(type(exc).__name__ + ": " + str(exc))
        return dict(
            decisions=self.decisions,
            examples=examples,
            errors=self.errors,
            origin="existing calibrated student source episode",
            diagnostic_only=True,
            extra_episodes=0,
            extra_teacher_queries=0,
        )


class SourceRolloutRecorder(_DirectionRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_counts = {}
        self.phase_changes = {}
        self.phase_longest = {}
        self.phase_matches = {}
        self.phase_legal = {}
        self.phase_last = {}
        self.phase_run = {}

    def _observe(self, intersection, states, requested_action):
        super()._observe(intersection, states, requested_action)
        requested = int(requested_action)
        current = int(next(iter(states.values()))["current_phase"])
        assert requested in states
        if intersection not in self.phase_counts:
            self.phase_counts[intersection] = Counter()
            self.phase_changes[intersection] = 0
            self.phase_longest[intersection] = 0
            self.phase_matches[intersection] = 0
            self.phase_legal[intersection] = set()
            self.phase_run[intersection] = 0
        self.phase_counts[intersection][requested] += 1
        self.phase_legal[intersection].update(int(p) for p in states)
        self.phase_matches[intersection] += int(requested == current)
        previous = self.phase_last.get(intersection)
        self.phase_changes[intersection] += int(previous is not None and previous != requested)
        self.phase_run[intersection] = (
            self.phase_run[intersection] + 1 if previous == requested else 1
        )
        self.phase_longest[intersection] = max(
            self.phase_longest[intersection], self.phase_run[intersection]
        )
        self.phase_last[intersection] = requested

    def finish(self, evaluator, utility, bias):
        result = super().finish(evaluator, utility, bias)
        result["phase_request_summary"] = summarize(
            self.phase_counts,
            self.phase_changes,
            self.phase_longest,
            self.phase_matches,
            self.phase_legal,
        )
        assert result["phase_request_summary"]["decisions"] == result["decisions"]
        for example in result["examples"]:
            example["phase_bias_decomposition"] = decompose(example)
        return result
