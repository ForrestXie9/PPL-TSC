"""Source-wide active-phase counterfactual checks.

Finite counterfactual checks, not traffic simulation or a universal proof.
Every observed node/physical-phase gets every selected source/boundary context.
One real inactive phase is used per context, rotated across the available IDs.
"""

import copy
import hashlib
import json
import math
import time
from collections import Counter, deque

FIELDS = ("pressure", "num_vehicle", "num_waiting_vehicle", "waiting_time", "sum_waiting_time")
TOLERANCE = 1e-6


def _binding(validator):
    result = dict(mode=validator.action_mode, loader_identity=id(validator.data_loader))
    result["source_nodes"] = tuple(
        sorted(
            getattr(validator.data_loader, "intersection_ids", ()),
            key=lambda x: (type(x).__name__, str(x)),
        )
    )
    if hasattr(validator.data_loader, "timesteps"):
        result["source_rows"] = len(validator.data_loader.timesteps)
    if validator.action_mode in {"phase_argmax", "keep_switch", "auto"}:
        result["statistics_sha256"] = hashlib.sha256(
            json.dumps(validator.feature_stats, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    return result


def _demand(states):
    values = []
    for state in states.values():
        for features in state.get("features_by_direction", {}).values():
            for key in ("num_vehicle", "num_waiting_vehicle", "waiting_time"):
                value = features.get(key, 0.0)
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError("Non-finite or nonnumeric source traffic feature")
                values.append(max(0.0, value))
    return sum(values)


def _pressure(states):
    values = [
        f["pressure"]
        for s in states.values()
        for f in s.get("features_by_direction", {}).values()
        if "pressure" in f
    ]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Non-finite or nonnumeric source pressure")
    return min(values, default=0.0)


def _zero(state):
    result = copy.deepcopy(state)
    for features in result.get("features_by_direction", {}).values():
        for key in FIELDS:
            if key in features:
                features[key] = 0.0
    return result


def plan(validator):
    """Prepare once per source. Reject missing nodes and changing phase schemas."""
    binding = _binding(validator)
    if validator.action_mode == "binary_keep_switch":
        return dict(
            status="not_applicable",
            reason="Native KEEP/SWITCH uses its separate binary preference check",
            probes=[],
            nodes=[],
            binding=binding,
        )
    if validator.action_mode not in {"phase_argmax", "keep_switch", "auto"}:
        raise ValueError("Unsupported action interface")
    if validator.data_loader is None or not validator.feature_stats:
        raise ValueError("Source loader and source statistics are required")
    nodes = sorted(validator.data_loader.intersection_ids, key=lambda x: (type(x).__name__, str(x)))
    if not nodes:
        raise ValueError("No source intersection IDs")
    queues = []
    single_phase = []
    phase_count = 0
    for node in nodes:
        records = validator.data_loader.get_action_consistency_data(node, max_timesteps=None)
        if not records:
            raise ValueError(f"Missing source rows at node {node}")
        rows = []
        ids = None
        for item in records:
            if not item or not isinstance(item[0], dict) or not item[0]:
                raise ValueError(f"Empty source bundle at node {node}")
            states = item[0]
            current_ids = sorted(states)
            if not all(isinstance(pid, int) and not isinstance(pid, bool) for pid in current_ids):
                raise ValueError(f"Invalid physical phase IDs at node {node}")
            if ids is not None and ids != current_ids:
                raise ValueError(f"Changing source phase schema at node {node}")
            ids = current_ids
            for state in states.values():
                if not isinstance(state, dict) or state.get("current_phase") not in ids:
                    raise ValueError(f"Missing/invalid current_phase at node {node}")
                if not isinstance(state.get("features_by_direction"), dict):
                    raise ValueError(f"Missing directional features at node {node}")
            rows.append(states)
        if len(ids) == 1:
            single_phase.append(node)
            continue
        phase_count += len(ids)
        # Three source contexts plus a synthetic zero-traffic boundary. Labels are
        # provenance, not claims that a low observed frame is literally empty.
        demands = [_demand(row) for row in rows]
        pressures = [_pressure(row) for row in rows]
        high = max(range(len(rows)), key=lambda i: demands[i])
        low = min(range(len(rows)), key=lambda i: demands[i])
        negative = min(range(len(rows)), key=lambda i: pressures[i])
        contexts = [
            ("source_high", high),
            ("source_low", low),
            (
                "source_negative" if pressures[negative] < 0 else "source_midtime",
                negative if pressures[negative] < 0 else len(rows) // 2,
            ),
            ("synthetic_zero", high),
        ]
        queue = deque()
        # Context first gives each physical phase one source test before adding
        # further contexts. Outer round-robin below prevents node starvation.
        for context_rank, (label, index) in enumerate(contexts):
            for phase in ids:
                others = [pid for pid in ids if pid != phase]
                inactive = others[context_rank % len(others)]
                state = rows[index][phase]
                if label == "synthetic_zero":
                    state = _zero(state)
                queue.append(
                    dict(
                        node=node,
                        phase=phase,
                        inactive=inactive,
                        context=label,
                        source_index=index,
                        state=state,
                    )
                )
        queues.append(queue)
    probes = []
    while any(queues):
        for queue in queues:
            if queue:
                probes.append(queue.popleft())
    return dict(
        status="ready" if probes else "not_applicable",
        reason="" if probes else "Only single-physical-phase source nodes",
        nodes=nodes,
        single_phase_nodes=single_phase,
        physical_phase_count=phase_count,
        probes=probes,
        tolerance=TOLERANCE,
        binding=binding,
    )


def check(validator, utility, *, prepared=None, max_pairs=None, stop_on_failure=True):
    """Reuse the compiled utility; no recompilation, fitting or simulation.

    An explicit insufficient budget fails coverage instead of silently truncating.
    Missing execution/NaN never becomes a pass. NOT_APPLICABLE is a separate status.
    """
    started = time.perf_counter()
    result = dict(
        status="failed",
        applicable=True,
        satisfied=False,
        planned_pairs=0,
        executed_pairs=0,
        utility_calls=0,
        violations=[],
        comparisons=[],
        node_coverage={},
        phase_coverage={},
        inactive_coverage={},
        complete=False,
    )

    def finish():
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

    try:
        p = prepared if prepared is not None else plan(validator)
        if p.get("binding") != _binding(validator):
            raise ValueError(
                "Prepared plan belongs to different source loader, statistics or action interface"
            )
    except Exception as exc:
        result["violations"] = [
            dict(kind="coverage_error", error=type(exc).__name__ + ": " + str(exc))
        ]
        return finish()
    if p["status"] == "not_applicable":
        result.update(
            status="not_applicable",
            applicable=False,
            satisfied=None,
            reason=p["reason"],
            single_phase_nodes=p.get("single_phase_nodes", []),
        )
        return finish()
    probes = p["probes"]
    result["planned_pairs"] = len(probes)
    if not probes:
        result["violations"] = [dict(kind="coverage_error", error="No executable paired probes")]
        return finish()
    if max_pairs is not None and (
        not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1
    ):
        result["violations"] = [
            dict(kind="coverage_error", error="Pair budget must be a positive integer")
        ]
        return finish()
    if max_pairs is not None and max_pairs < len(probes):
        result["violations"] = [
            dict(
                kind="coverage_error",
                error=f"Budget {max_pairs} cannot cover planned {len(probes)} pairs",
            )
        ]
        return finish()
    nodes = Counter()
    phases = Counter()
    off = {}
    cache = {}
    # A score is reused only for the exact same planned source context and phase.
    # It is never reused across functions/check invocations or state values.
    for probe in probes:
        row = {k: probe[k] for k in ["node", "phase", "inactive", "context", "source_index"]}
        try:
            scores = []
            for current in [probe["phase"], probe["inactive"]]:
                kind = "zero" if probe["context"] == "synthetic_zero" else "source"
                key = (probe["node"], probe["phase"], kind, probe["source_index"], current)
                if key not in cache:
                    state = copy.deepcopy(probe["state"])
                    state["current_phase"] = current
                    result["utility_calls"] += 1
                    value = float(
                        utility.compute(
                            state,
                            phase_id=probe["phase"],
                            feature_stats=copy.deepcopy(validator.feature_stats),
                            raise_on_error=True,
                        )
                    )
                    if not math.isfinite(value):
                        raise ValueError("Non-finite utility")
                    cache[key] = value
                scores.append(cache[key])
            row.update(
                q_active=scores[0],
                q_inactive=scores[1],
                satisfied=scores[0] + TOLERANCE >= scores[1],
            )
            if not row["satisfied"]:
                result["violations"].append(dict(kind="active_status_lowered_utility", **row))
        except Exception as exc:
            row.update(satisfied=False, error=type(exc).__name__ + ": " + str(exc))
            result["violations"].append(dict(kind="execution_error", **row))
        result["comparisons"].append(row)
        result["executed_pairs"] += 1
        nodes[str(probe["node"])] += 1
        phase_key = f"{probe['node']}:{probe['phase']}"
        phases[phase_key] += 1
        off.setdefault(phase_key, set()).add(probe["inactive"])
        if result["violations"] and stop_on_failure:
            break
    result.update(
        node_coverage=dict(nodes),
        phase_coverage=dict(phases),
        inactive_coverage={k: sorted(v) for k, v in off.items()},
        single_phase_nodes=p.get("single_phase_nodes", []),
        complete=result["executed_pairs"] == len(probes),
    )
    result["satisfied"] = result["complete"] and not result["violations"]
    result["status"] = "passed" if result["satisfied"] else "failed"
    return finish()
