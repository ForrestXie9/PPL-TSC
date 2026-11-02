"""Compact source-only evolution feedback. No score, verifier or calibration edits."""

import math


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if number(value) else "unavailable"


def compact_state(states, relevant):
    """Losslessly merge shared directional values for the relevant phases only."""
    if not isinstance(states, dict):
        return states
    if "binary_features" in states:
        return states
    selected = {str(p): s for p, s in states.items() if str(p) in {str(q) for q in relevant}}
    directions = {}
    features = {}
    current = set()
    binary = []
    for p, s in selected.items():
        if not isinstance(s, dict):
            return selected
        current.add(s.get("current_phase"))
        directions[p] = s.get("controlled_directions", [])
        for direction, values in s.get("features_by_direction", {}).items():
            if direction in features and features[direction] != values:
                return selected
            features[direction] = values
        if s.get("binary_features") is not None and s["binary_features"] not in binary:
            binary.append(s["binary_features"])
    result = dict(
        current_phase=next(iter(current)) if len(current) == 1 else sorted(current, key=str),
        phase_directions=directions,
        direction_features=features,
    )
    if binary:
        result["binary_features"] = binary[0] if len(binary) == 1 else binary
    return result


def native_state(states):
    if "binary_features" in states:
        return {
            k: states[k] for k in ["current_phase", "next_phase", "binary_features"] if k in states
        }
    cells = list(states.values())
    if not cells:
        return {}
    first = cells[0]
    assert all(cell.get("binary_features") == first.get("binary_features") for cell in cells)
    return dict(
        current_phase=first.get("current_phase"),
        candidate_phases=list(states),
        binary_features=first.get("binary_features"),
    )


def one_example(metrics, diagnostic, action_mode=None):
    if diagnostic and diagnostic.get("reversal_examples"):
        e = diagnostic["reversal_examples"][0]
        relevant = [e["raw_choice"], e["calibrated_choice"], e["teacher_choice"]]
        keys = {str(k) for k in relevant}
        raw = {k: v for k, v in e["raw_scores"].items() if str(k) in keys}
        bias = {k: v for k, v in e["phase_bias"].items() if str(k) in keys}
        adjusted = {k: v + bias.get(k, bias.get(str(k), 0.0)) for k, v in raw.items()}
        state = (
            native_state(e.get("source_state", {}))
            if action_mode == "binary_keep_switch"
            else compact_state(e.get("source_state", {}), relevant)
        )
        return dict(
            origin="source teacher-rollout state;not a student rollout",
            state=state,
            raw_utilities=raw,
            phase_bias=bias,
            adjusted_utilities=adjusted,
            teacher_action=e["teacher_choice"],
            raw_action=e["raw_choice"],
            calibrated_action=e["calibrated_choice"],
            raw_pair_gap=e["raw_pair_gap"],
            matched_bias_advantage=e["matched_bias_advantage"],
        )
    lane = metrics.get("source_rollout_feedback", {}).get("lane_permission_diagnostic") or {}
    summary = lane.get("summary", {})
    cp = metrics.get("control_performance")
    if not (
        number(cp)
        and cp < 1.0
        and summary.get("max_continuous_waiting_without_green_seconds", 0) > 0
    ):
        return None
    for e in lane.get("examples", []):
        if not e.get("source_decision_state"):
            continue
        relevant = [e.get("raw_choice"), e.get("adjusted_choice"), e.get("requested_phase")]
        probes = e.get("joint_response", {}).get("probes", [])
        if probes:
            relevant.append(probes[0].get("target_phase"))
        result = dict(
            origin="source student-rollout snapshot;no teacher action label",
            state=(
                native_state(e["source_decision_state"])
                if action_mode == "binary_keep_switch"
                else compact_state(e["source_decision_state"], relevant)
            ),
            raw_utilities=e.get("raw_utilities"),
            adjusted_utilities=e.get("adjusted_utilities"),
            phase_bias={
                k: v
                for k, v in e.get("phase_bias", {}).items()
                if str(k) in {str(q) for q in relevant}
            },
            raw_action=e.get("raw_choice"),
            calibrated_action=e.get("adjusted_choice"),
            waiting_without_green_seconds=e.get("continuous_wait_without_green_seconds"),
        )
        if probes:
            probe = probes[0]
            points = probe.get("probes", [])
            result["algebraic_probe"] = dict(
                feature=probe.get("feature"),
                target_phase=probe.get("target_phase"),
                points=[points[0], points[-1]] if len(points) > 1 else points,
                meaning="other inputs fixed;not a simulated traffic outcome",
            )
        return result
    return None
