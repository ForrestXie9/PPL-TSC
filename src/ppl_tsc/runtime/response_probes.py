"""Bounded diagnostic of calibrated decisions, no fitness penalty or gate."""

import copy, math

MARKER = "SOURCE JOINT RESPONSE FEEDBACK v1"


def inspect_response(utility, states, stats, bias):
    def scores(bundle):
        from ppl_tsc.runtime.decision_values import decision_values

        raw = decision_values(utility, bundle, stats)
        fitted = {p: q + bias.get(p, bias.get(str(p), 0.0)) for p, q in raw.items()}
        assert all(math.isfinite(x) for x in fitted.values())
        return raw, fitted

    raw, fit = scores(states)
    winner = max(fit, key=fit.get)
    rows = []
    if utility.action_mode == "binary_keep_switch":
        state = next(iter(states.values()))
        old = float(state["binary_features"]["red_vehicle"])
        scale = stats.get("red_vehicle", {})
        upper = max(old, float(scale.get("max", old)))
        cur = int(state["current_phase"])
        target = (cur + 1) % len(states)
        if target == winner:
            return {"status": "review", "current_action_is_switch": True, "probes": []}

        def change(value):
            bundle = copy.deepcopy(states)
            for s in bundle.values():
                s["binary_features"]["red_vehicle"] = value
            return bundle

        work = [("red_vehicle", old, upper, target, change)]
    else:
        current = states[winner]
        served = set(current.get("controlled_directions", []))
        work = []
        for target in raw:
            state = states[target]
            if target == winner:
                continue
            exclusive = [d for d in state.get("controlled_directions", []) if d not in served]
            for direction in exclusive[:1]:
                features = state.get("features_by_direction", {}).get(direction, {})
                key = next(
                    (
                        k
                        for k in ("num_vehicle", "pressure", "num_waiting_vehicle")
                        if k in features
                    ),
                    None,
                )
                if key is None:
                    continue
                statkey = {
                    "num_vehicle": "vehicles_raw",
                    "pressure": "pressure_raw",
                    "num_waiting_vehicle": "waiting_veh_raw",
                }[key]
                old = float(features[key])
                upper = max(old, float(stats.get(statkey, {}).get("max", old)))
                if key == "num_waiting_vehicle":
                    upper = min(upper, float(features.get("num_vehicle", upper)))

                def change(value, d=direction, k=key):
                    bundle = copy.deepcopy(states)
                    for s in bundle.values():
                        f = s.get("features_by_direction", {}).get(d)
                        if f is not None and k in f:
                            f[k] = value
                    return bundle

                work.append((direction + ":" + key, old, upper, target, change))
            if len(work) >= 2:
                break
    for feature, old, upper, target, change in work:
        values = sorted(set([old, old + (upper - old) * 0.25, old + (upper - old) * 0.5, upper]))
        if len(values) < 2:
            continue
        probes = []
        for value in values:
            r, f = scores(change(value))
            choice = max(f, key=f.get)
            probes.append(
                {
                    "value": value,
                    "raw_target_minus_original_winner": r[target] - r[winner],
                    "adjusted_target_minus_original_winner": f[target] - f[winner],
                    "adjusted_choice": choice,
                }
            )
        rows.append(
            {
                "feature": feature,
                "target_phase": target,
                "original_winner": winner,
                "source_range_endpoint": upper,
                "probes": probes,
                "target_selected_within_probes": any(
                    p["adjusted_choice"] == target for p in probes
                ),
                "adjusted_margin_gain": probes[-1]["adjusted_target_minus_original_winner"]
                - probes[0]["adjusted_target_minus_original_winner"],
            }
        )
    return {
        "status": "diagnostic_only",
        "probes": rows,
        "no_hard_rejection": True,
        "limits": "Frozen source maximum, other inputs fixed; not proof of infinite-demand lock, physical joint feasibility, or fairness. No forced switching target, operator ban, or coefficient threshold.",
    }
