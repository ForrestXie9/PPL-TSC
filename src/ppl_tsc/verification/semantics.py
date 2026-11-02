"""Shared executable and prompt contract for sampled weak monotonicity.

These are mathematical feature-coordinate checks, not simulated interventions
or all-state proofs. No fitness, calibration, bias or search settings live here.
"""

import copy
import math
from collections import OrderedDict, deque

SCALES = (0.5, 0.8, 1.0, 1.2, 1.5)
TOLERANCE = 1e-6
FIELDS = ("pressure", "num_vehicle", "num_waiting_vehicle", "waiting_time", "sum_waiting_time")
ALIASES = ("waiting_time", "sum_waiting_time")
STATS_KEYS = dict(
    pressure="pressure_raw",
    num_vehicle="vehicles_raw",
    num_waiting_vehicle="waiting_veh_raw",
    waiting_time="waiting_time_raw",
    sum_waiting_time="waiting_time_raw",
)


def contract_text(binary=False):
    scale_text = ", ".join(f"{a:g}" for a in SCALES)
    common = f"""EXECUTABLE VERIFIER CONTRACT:
- Sampled mathematical coordinate checks, not causal traffic simulations or an all-state proof. Utilities and binary preference differences must be finite; comparison tolerance is {TOLERANCE:g}.
- Positive source inputs use scales [{scale_text}]. Zero inputs use a nonnegative source-reference grid (unit fallback if the source maximum is zero). An empty test set is not a pass.
- Across source-derived equal-status traffic probes the program must show at least one demand-dependent phase score difference (binary: KEEP-minus-SWITCH preference difference). Equality on any particular input, unused individual features and saturation remain allowed. A holding-only or constant program is insufficient; no minimum response slope or switching action is required.
- Weak monotonicity allows equality, saturation and unused individual inputs. It does not require a minimum slope or an action switch. A fixed calibrated offset may reverse rankings; that alone is not a violation.
"""
    if binary:
        return (
            common
            + """- Increasing green_vehicle must not decrease Q_KEEP - Q_SWITCH; increasing red_vehicle must not decrease Q_SWITCH - Q_KEEP. Other inputs are fixed. KEEP=0 and SWITCH=1 are action IDs; SWITCH advances to the cyclic next physical phase, not every red direction.
- Physical-phase relabelling checks do not swap these binary action IDs or change the cyclic successor. Branching on KEEP=0 versus SWITCH=1 is valid and necessary.
"""
        )
    return (
        common
        + """- For each sampled controlled direction and available traffic field, compare Q in increasing NUMERIC input order. Signed pressure retains its sign; include negative, zero and positive probes. More-negative pressure is not an increase in demand.
- waiting_time and sum_waiting_time, when both present, are aliases and change together. Other fields, candidate phase ID and current phase are held fixed. Cross-feature traffic feasibility is not asserted by these coordinate probes.
- Sampling is spread across directions and features within the configured coordinate check cap; passing is limited to the checked inputs. Pressure also uses a sparse context with other controlled directions at zero pressure, including values close to zero. These are mathematical boundary probes, not observed joint traffic trajectories.
- For phase_argmax, physical phase IDs are labels, not traffic measurements. Consistently permuting the existing phase IDs and current_phase while keeping traffic and controlled directions fixed must leave each physical phase's raw Q unchanged. Comparing phase_id == current_phase for a holding reward is allowed. Do not derive a priority from an ID's magnitude, remainder, parity or hard-coded numeric identity. The verifier samples up to 40 additional label comparisons; this does not change candidate/retry/search budgets.
- Learned per-physical-phase calibration offsets remain external to this raw-utility check and follow their physical phase when labels change. Holding rewards and calibrated offsets are not removed or bounded by this rule. The check compares scores, not which ID wins an exact tie.
"""
    )


def _bundles(validator, limit):
    """Balanced source-only time/node coverage plus source-contrast probes."""
    loader = validator.data_loader
    if loader is None or limit <= 0:
        return []
    groups = []
    for inter in sorted(loader.intersection_ids):
        rows = [
            row[0]
            for row in loader.get_action_consistency_data(inter, max_timesteps=None)
            if row and row[0]
        ]
        if rows:
            groups.append((inter, rows))
    if not groups:
        return []
    if len(groups) > limit:
        n = limit
        groups = [groups[round(i * (len(groups) - 1) / max(n - 1, 1))] for i in range(n)]

    def contrast(states):
        scores = []
        for state in states.values():
            fbd = state.get("features_by_direction") or {}
            for key in FIELDS:
                values = [fbd.get(d, {}).get(key) for d in state.get("controlled_directions", [])]
                if len(values) > 1 and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in values
                ):
                    scale = max(1.0, sum(abs(v) for v in values) / len(values))
                    scores.append((max(values) - min(values)) / scale)
        return max(scores, default=0.0)

    total = sum(len(rows) for _, rows in groups)
    if total <= limit:
        return [(inter, state) for inter, rows in groups for state in rows]
    coverage = min(limit, max(len(groups), (limit + 1) // 2))
    per_node = {
        inter: coverage // len(groups) + int(rank < coverage % len(groups))
        for rank, (inter, _) in enumerate(groups)
    }
    selected = []
    seen = set()
    # Mid-bin source times avoid filling a small budget with early empty frames.
    time_queues = []
    for inter, rows in groups:
        n = min(per_node[inter], len(rows))
        indices = sorted({round((j + 0.5) * len(rows) / max(n, 1) - 0.5) for j in range(n)})
        time_queues.append(deque((inter, index, rows[index]) for index in indices))
    while any(time_queues) and len(selected) < coverage:
        for queue in time_queues:
            if queue and len(selected) < coverage:
                inter, index, state = queue.popleft()
                selected.append((inter, state))
                seen.add((inter, index))
    risk_queues = []
    for inter, rows in groups:
        ranked = sorted(range(len(rows)), key=lambda index: -contrast(rows[index]))
        risk_queues.append(
            deque((inter, index, rows[index]) for index in ranked if (inter, index) not in seen)
        )
    while any(risk_queues) and len(selected) < limit:
        for queue in risk_queues:
            if queue and len(selected) < limit:
                inter, index, state = queue.popleft()
                selected.append((inter, state))
                seen.add((inter, index))
    return selected


def _values(validator, key, base):
    if not math.isfinite(base):
        raise ValueError(f"non-finite source {key}={base}")
    if key != "pressure" and base < 0:
        raise ValueError(f"negative source count/time {key}={base}")
    stat = validator.feature_stats.get(STATS_KEYS.get(key, key), {})
    if not stat:
        stat = validator.feature_stats.get(key, {})
    if key == "pressure":
        low = float(stat.get("min", -2.0))
        values = [a * base for a in SCALES] + [
            min(low, -2.0),
            -1.0,
            -0.1,
            -0.001,
            0.0,
            0.001,
            0.1,
            1.0,
        ]
    elif base == 0:
        high = float(stat.get("max", 1.0))
        if high <= 0:
            high = 1.0
        values = [0.0, min(1.0, high * 0.25), min(2.0, high * 0.5), high * 0.75, high]
    else:
        values = [a * base for a in SCALES]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite probe range for {key}")
    return sorted(set(values))


def _failures(values, scores, label):
    if not all(math.isfinite(q) for q in scores):
        return [f"{label}: non-finite output; inputs={values}, outputs={scores}"]
    for x1, x2, q1, q2 in zip(values, values[1:], scores, scores[1:]):
        if q2 + TOLERANCE < q1:
            return [
                f"{label}: input {x1:.9g} -> {x2:.9g}, output {q1:.9g} -> {q2:.9g}; decreased in increasing numeric order"
            ]
    return []


def phase_monotonicity(
    self, utility_function, max_samples=100, max_feature_checks=256, verbose=False
):
    if self.action_mode == "binary_keep_switch":
        return binary_monotonicity(
            self, utility_function, max_feature_checks, max_samples=max_samples
        )
    # A queue per feature/direction/phase prevents repeated observations of
    # controlled_directions[0] from consuming the complete check allowance.
    by_feature = OrderedDict((key, OrderedDict()) for key in FIELDS)
    for inter, states in _bundles(self, max_samples):
        for phase, state in states.items():
            fbd = state.get("features_by_direction") or {}
            for direction in state.get("controlled_directions", []):
                feat = fbd.get(direction, {})
                for key in FIELDS:
                    if key not in feat or (
                        self.actual_feature_keys and key not in self.actual_feature_keys
                    ):
                        continue
                    if (
                        key == "sum_waiting_time"
                        and "waiting_time" in feat
                        and (
                            not self.actual_feature_keys
                            or "waiting_time" in self.actual_feature_keys
                        )
                    ):
                        continue
                    group = by_feature[key].setdefault((inter, phase, direction), deque())
                    group.append((phase, direction, key, state))

    # Within each coordinate, an early empty/equal state must not hide an
    # observed unequal-demand state later in the source sample.
    def coordinate_contrast(row):
        _, direction, key, state = row
        fbd = state.get("features_by_direction") or {}
        values = [fbd.get(d, {}).get(key) for d in state.get("controlled_directions", [])]
        base = fbd.get(direction, {}).get(key)
        if (
            not values
            or not isinstance(base, (int, float))
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        ):
            return 0.0
        return max(values) - base

    for groups in by_feature.values():
        for key, group in groups.items():
            groups[key] = deque(sorted(group, key=lambda row: -coordinate_contrast(row)))
    # Round-robin nodes within every feature; rank risks within each node.
    queues = []
    for groups in by_feature.values():
        per_node = OrderedDict()
        for (inter, phase, direction), group in groups.items():
            per_node.setdefault(inter, []).append(group)
        node_queues = [
            deque(sorted(values, key=lambda group: -coordinate_contrast(group[0])))
            for values in per_node.values()
        ]
        flattened = deque()
        while any(node_queues):
            for queue in node_queues:
                if queue:
                    flattened.append(queue.popleft())
        if flattened:
            queues.append(flattened)
    checks, errors, witnesses = 0, [], []
    while queues and checks < max_feature_checks:
        remaining = []
        for queue in queues:
            if checks >= max_feature_checks:
                break
            group = queue.popleft()
            phase, direction, key, state = group.popleft()
            values, scores = [], []
            label = f"phase={phase}, direction={direction}, feature={key}"
            try:
                values = _values(self, key, float(state["features_by_direction"][direction][key]))
                for value in values:
                    changed = copy.deepcopy(state)
                    feat = changed["features_by_direction"][direction]
                    for alias in ALIASES if key in ALIASES else (key,):
                        if alias in feat:
                            feat[alias] = value
                    scores.append(
                        float(
                            utility_function.compute(
                                changed,
                                phase_id=phase,
                                feature_stats=self.feature_stats,
                                raise_on_error=True,
                            )
                        )
                    )
                errors.extend(_failures(values, scores, label))
                if key == "pressure":
                    # Expose aggregate dispersion rewards near equal pressure.
                    # Still one coordinate within the same 256-coordinate cap.
                    sparse = copy.deepcopy(state)
                    for other in sparse.get("controlled_directions", []):
                        if "pressure" in sparse["features_by_direction"].get(other, {}):
                            sparse["features_by_direction"][other]["pressure"] = 0.0
                    sparse_scores = []
                    for value in values:
                        sparse["features_by_direction"][direction]["pressure"] = value
                        sparse_scores.append(
                            float(
                                utility_function.compute(
                                    sparse,
                                    phase_id=phase,
                                    feature_stats=self.feature_stats,
                                    raise_on_error=True,
                                )
                            )
                        )
                    errors.extend(
                        _failures(values, sparse_scores, label + ", sparse_equal_pressure")
                    )

            except Exception as exc:
                errors.append(
                    f"{label}: {type(exc).__name__}: {exc}; inputs={values}, outputs={scores}"
                )
            witnesses.append(
                dict(
                    phase=phase,
                    direction=direction,
                    feature=key,
                    inputs=values,
                    outputs=[q if math.isfinite(q) else repr(q) for q in scores],
                )
            )
            checks += 1
            if group:
                queue.append(group)
            if queue:
                remaining.append(queue)
        queues = remaining
    if checks == 0:
        errors.append("Monotonicity not evaluated: no usable source direction/feature probes")
    self._last_monotonicity_checks = witnesses
    return not errors, errors


def binary_monotonicity(self, utility_function, max_feature_checks=256, max_samples=100):
    checks, errors, witnesses = 0, [], []
    for inter, states in _bundles(self, max_samples):
        try:
            state, _, _ = utility_function._native_binary_context(states)
        except Exception as exc:
            errors.append(f"Binary context: {type(exc).__name__}: {exc}")
            continue
        for key in ("green_vehicle", "red_vehicle"):
            if checks >= max_feature_checks:
                break
            values, preferences = [], []
            label = f"intersection={inter}, feature={key}, preference={'KEEP' if key == 'green_vehicle' else 'SWITCH'}"
            try:
                values = _values(self, key, float(state["binary_features"][key]))
                for value in values:
                    changed = copy.deepcopy(states)
                    self._set_binary_feature(changed, key, value)
                    _, _, keep, switch = utility_function.compute_binary_action_values(
                        changed, feature_stats=self.feature_stats, raise_on_error=True
                    )
                    keep, switch = float(keep), float(switch)
                    if not (math.isfinite(keep) and math.isfinite(switch)):
                        raise ValueError(f"input={value}, Q_KEEP={keep}, Q_SWITCH={switch}")
                    preferences.append(keep - switch if key == "green_vehicle" else switch - keep)
                errors.extend(_failures(values, preferences, label))
            except Exception as exc:
                errors.append(
                    f"{label}: {type(exc).__name__}: {exc}; inputs={values}, preferences={preferences}"
                )
            witnesses.append(
                dict(
                    feature=key,
                    inputs=values,
                    preferences=[q if math.isfinite(q) else repr(q) for q in preferences],
                )
            )
            checks += 1
        if checks >= max_feature_checks:
            break
    if checks == 0:
        errors.append("Binary monotonicity not evaluated: no usable source probes")
    self._last_monotonicity_checks = witnesses
    return not errors, errors


def demand_discrimination(validator, utility_function, max_samples=25):
    """Existential response check; never demand every coordinate or frame differ."""
    checks = 0
    errors = []
    try:
        for _, states in _bundles(validator, max_samples):
            if validator.action_mode == "binary_keep_switch":
                state, _, _ = utility_function._native_binary_context(states)
                for key in ("green_vehicle", "red_vehicle"):
                    prefs = []
                    for value in _values(validator, key, 0.0):
                        changed = copy.deepcopy(states)
                        for item in changed.values():
                            item["binary_features"][key] = value
                        _, _, keep, switch = utility_function.compute_binary_action_values(
                            changed, feature_stats=validator.feature_stats, raise_on_error=True
                        )
                        prefs.append(float(keep - switch))
                    checks += 1
                    if not all(math.isfinite(q) for q in prefs):
                        raise ValueError("non-finite binary preference")
                    if max(prefs) - min(prefs) > TOLERANCE:
                        return dict(
                            name="Demand-dependent discrimination",
                            satisfied=True,
                            checked_coordinates=checks,
                            details=[],
                        )
                continue
            for phase, state in states.items():
                # Compare identical activity status, so holding alone cannot pass.
                for key in FIELDS:
                    if key == "sum_waiting_time":
                        continue
                    probes = []
                    for value in _values(validator, key, 0.0):
                        changed = copy.deepcopy(state)
                        for direction in changed.get("controlled_directions", []):
                            feat = changed["features_by_direction"].get(direction, {})
                            for alias in ALIASES if key in ALIASES else (key,):
                                if alias in feat:
                                    feat[alias] = value
                        probes.append(
                            float(
                                utility_function.compute(
                                    changed,
                                    phase_id=phase,
                                    feature_stats=validator.feature_stats,
                                    raise_on_error=True,
                                )
                            )
                        )
                    checks += 1
                    if not all(math.isfinite(q) for q in probes):
                        raise ValueError("non-finite phase score")
                    if max(probes) - min(probes) > TOLERANCE:
                        return dict(
                            name="Demand-dependent discrimination",
                            satisfied=True,
                            checked_coordinates=checks,
                            details=[],
                        )
        errors.append(
            "No demand-dependent score/preference difference in source-derived probes; constant or holding-only utilities do not discriminate traffic demand."
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return dict(
        name="Demand-dependent discrimination",
        satisfied=False,
        checked_coordinates=checks,
        details=errors,
    )
