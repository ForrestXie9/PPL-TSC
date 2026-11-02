"""Manuscript templates and the single executable prompt renderer."""

import copy
import json


TEMPLATES = {
    "initialization": """Task: Discover a student utility function that approximates the teacher model's decision behavior. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. Prioritize diverse and expressive function structures during initialization.

Context: The available input features are {features}.

Constrains: The implementation must satisfy on {syntactic}, {mathematical}, and {semantic}.

Output: Provide the complete Python function Compute_utility(state, phase_id) that returns a float utility value, followed by a brief explanation (2–3 sentences) describing the design rationale.""",
    "evolution": """Task: Evolve an improved offspring utility function from the given parent. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. The offspring must be structurally distinct from the parent and combine raw features meaningfully to avoid trivial heuristics.

Context: The parent function is {parent}. Evaluation information is provided in {evaluation}. The available input features are {features}.

Requirements: Same as in the initialization prompt.

Output: Same as in the initialization prompt.""",
    "repair": """Task: Repair an invalid utility function that violates specified constraints. The function Compute_utility(state, phase_id) assigns a utility score to each candidate phase; the phase with the highest score is activated. The repaired function must preserve the original behavioral intent as much as possible while strictly satisfying all constraints and eliminating any invalid operations.

Context: The function to be repaired is {parent}. The identified constraint violations are detailed in {violations}. The available input features are {features}.

Requirements: Same as in the initialization prompt.

Output: Same as in the initialization prompt.""",
}


ALIASES = {
    "num_vehicle": "vehicles",
    "num_waiting_vehicle": "waiting_veh",
    "waiting_time": "waiting_time",
    "sum_waiting_time": "waiting_time",
    "pressure": "pressure",
}


def raw_keys(generator):
    keys = sorted(set(generator._get_state_keys_for_teacher()) - {"current_phase"})
    unknown = set(keys) - set(ALIASES)
    if unknown:
        raise ValueError(
            "Contract requires explicit semantics for raw fields: " + str(sorted(unknown))
        )
    return keys


def statistics(generator, binary):
    """Deduplicate only documented aliases; never merge unrelated equal distributions."""
    original = generator.feature_stats
    if original is None:
        return dict(values={}, aliases={}, available=[], supplied=False)
    wanted = (
        {"green_vehicle", "red_vehicle"}
        if binary
        else {"phase_active"}
        | {ALIASES[k] + suffix for k in raw_keys(generator) for suffix in ["_raw", "_norm"]}
    )
    values = {k: copy.deepcopy(v) for k, v in original.items() if k in wanted}
    aliases = {}
    for key in sorted(list(values)):
        if key.endswith("_norm"):
            canonical = key[:-5] + "_raw"
            if canonical in values and values[key] == values[canonical]:
                aliases[key] = canonical
                del values[key]
    return dict(
        values=values, aliases=aliases, available=sorted(set(values) | set(aliases)), supplied=True
    )


ACTIVE_PHASE_SCALE_REMINDER = "Keeping the current phase can reduce switching losses, but its influence should match traffic-score differences; after normalization, avoid masking higher traffic demand in competing phases."


def calibration_contract(generator, mode):
    if not generator.fit_phase_bias:
        return (
            "Calibration: disabled for this custom run. Raw f selects actions without external offsets. "
            "Parent ATT/ABF are measured without calibration. Verifier checks raw utilities; "
            "f stays fixed for same-network transfer."
        )
    bound = generator.calibration_bias_bound
    return (
        f"Calibration: modify only raw f. Each candidate separately fits external offsets per SOURCE intersection/physical phase in [-{bound:g},{bound:g}] to teacher action agreement, not ATT."
        + (
            " KEEP/SWITCH uses current/next physical-phase offsets."
            if mode in {"binary_keep_switch", "keep_switch"}
            else ""
        )
        + " Parent ATT/ABF are AFTER calibration; offsets may reverse rankings. Verifier checks raw utilities; f/offsets stay fixed for same-network transfer."
    )


def build_contract(generator):
    mode = generator._decision_mode()
    if mode not in {"phase_argmax", "keep_switch", "binary_keep_switch"}:
        raise ValueError(mode)
    binary = mode == "binary_keep_switch"
    s = statistics(generator, binary)
    lines = [
        "EXECUTABLE DATA CONTRACT:",
        "Function: compute_utility(state, phase_id=None, feature_stats=None).",
    ]
    if binary:
        lines += [
            "Interface: evaluate the same state for actions0=KEEP and1=SWITCH to the cyclic next phase; switch iff Q_SWITCH>Q_KEEP, otherwise KEEP. The template candidate phase means binary action here.",
            "state={current_phase:int,next_phase:int,binary_features:{green_vehicle:float,red_vehicle:float}}. Traffic values are nonnegative exact green/red lane totals, not direction means; phase_id is the action ID, not the physical phase number.",
        ]
        semantics = "Increasing green_vehicle must not decrease Q_KEEP-Q_SWITCH; increasing red_vehicle must not decrease Q_SWITCH-Q_KEEP, with other inputs fixed. Branch on KEEP/SWITCH, preserving these meanings under physical-phase relabelling; SWITCH does not serve every red direction."
    else:
        keys = raw_keys(generator)
        lines += [
            "Interface: evaluate each physical candidate phase_id:int, then argmax. state={current_phase:int,controlled_directions:list[str],features_by_direction:dict[str,dict[str,float]]}. Direction traffic fields: "
            + ",".join(keys)
            + "."
        ]
        if mode == "keep_switch":
            lines[-1] = lines[-1].replace(
                "evaluate each physical candidate phase_id:int, then argmax.",
                "evaluate physical phase_id:int only for the current phase and its cyclic successor; switch iff Q(next)>Q(current), otherwise keep. Do not select other phases.",
            )
        if "pressure" in keys:
            lines.append("pressure is signed movement/lanelink pressure, not a vehicle count.")
        if any(k != "pressure" for k in keys):
            lines.append(
                "Vehicle/waiting fields are nonnegative direction means of distinct compatible incoming lanes. Shared lanes overlap across directions; sums are not unique vehicle totals or observed turn counts."
            )
        semantics = "Increasing available controlled traffic must not decrease Q, other inputs fixed; signed pressure uses NUMERIC order (more-negative is not increased demand). Zeroing vehicle/waiting values and positive pressure, retaining nonpositive pressure, must not increase Q. Making a phase active must not lower Q; do not force inactive phases to zero. Reordering the same collection of controlled directions or consistently relabelling physical phase IDs must not change raw Q; names, positions and arbitrary IDs identify inputs, not traffic priorities."
        if "pressure" not in keys:
            semantics = semantics.replace(
                "; signed pressure uses NUMERIC order (more-negative is not increased demand)", ""
            )
        if {"waiting_time", "sum_waiting_time"} <= set(keys):
            semantics += " waiting_time/sum_waiting_time aliases vary together."
    lines += [
        "feature_stats: optional frozen SOURCE dict. min/max=empirical extrema; mean/std/count, p50/p75/p90/p95/p99 and zero_fraction describe the same observation pool.",
        (
            "Statistics describe exact green/red totals."
            if binary
            else "Statistics describe raw direction measurements, not phase aggregates. phase_active is binary support (0/1,mean0.5,std0.5), not observed frequencies."
        ),
    ]
    if s["aliases"]:
        lines.append(
            "Raw-measurement aliases (not normalized inputs): "
            + json.dumps(s["aliases"], sort_keys=True, separators=(",", ":"))
        )
    elif not binary and any(k.endswith(("_raw", "_norm")) for k in s["available"]):
        lines.append("Raw/norm names denote raw-measurement distributions, not normalized inputs.")
    lines.append(
        "SOURCE STAT VALUES: "
        + json.dumps(s["values"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    lines += [
        ACTIVE_PHASE_SCALE_REMINDER,
        "CONSTRAINTS:",
        "- Feature use: compute utility only from declared inputs and fixed source statistics; identical inputs must produce identical outputs. Read fields/statistics with safe .get defaults; invent no missing keys, derived phase_features or normalized inputs. Statistics are feature_stats[feature][statistic], e.g. green_vehicle/p95; raw field names such as num_vehicle are not statistic fields. Safe defaults must not hide misspelled or absent source statistic keys. Keep source statistics fixed on targets. Normalization inside f is optional; guard missing/constant ranges and out-of-source values (Min-Max may be <0 or >1)."
        + (" Do not normalize phase_active." if not binary else ""),
        "- Traffic semantics: "
        + semantics
        + " Traffic-dependent response is required; constants/holding-only rules are insufficient. Unused individual inputs, equality and saturation are allowed; no minimum slope, holding bonus or forced switching is required.",
        "- Numeric safety: import NumPy as np if used; initialize Q=0.0; return float(Q) if np.isfinite(Q) else 0.0. All intermediates/source bundles must be finite; the fallback does not excuse invalid arithmetic. Guard possible zero divisors, remainder, floor division, reciprocal and negative powers. Signed ratio: a/d if abs(d)>1e-8 else 0.0; magnitude scaling: a/max(abs(d),1e-8); adding1/epsilon to signed values is insufficient. sqrt argument>=0, log argument>0; negative-base fractional powers are invalid. Guard transformed inputs and keep code readable.",
        calibration_contract(generator, mode),
        "Checks/diagnostics sample source/boundary states, not causal traffic effects or all possible states.",
        "Output: one Python code block with ### Function, complete executable function, then ### Explanation and2–3 rationale sentences.",
    ]
    return "\n".join(lines)


def render_template(generator, stage, parent=None, explanation=None, feedback=None):
    # Fixed manuscript text is unchanged. New information lives in placeholders.
    parent_payload = "\n```python\n" + (parent or "") + "\n```\n"
    if stage != "evolution" and explanation:
        parent_payload += "Design rationale: " + explanation + "\n"
    fields = dict(
        features="\n" + build_contract(generator) + "\n",
        parent=parent_payload,
        evaluation="\n" + (feedback or "No performance metrics available") + "\n",
        violations="\n" + (feedback or "Correct the reported verifier failure.") + "\n",
        syntactic="syntactic correctness (the declared executable interface)",
        mathematical="mathematical validity (finite arithmetic with guarded operations)",
        semantic="semantic consistency (the declared decision and verifier properties)",
    )
    return TEMPLATES[stage].format_map(fields)


def namespace_note(generator):
    if generator._decision_mode() == "binary_keep_switch":
        return (
            'INPUT LOOKUPS: green=state.get("binary_features",{}).get("green_vehicle",0.0); '
            'green_scale=(feature_stats or {}).get("green_vehicle",{}).get("p95",1.0). '
            "The first reads current traffic; the second reads a frozen SOURCE statistic. "
            "red_vehicle follows the same two separate paths. No direction-feature fields exist in this interface."
        )
    keys = raw_keys(generator)
    key = keys[0]
    alias = ALIASES[key] + "_raw"
    pairs = {k: ALIASES[k] + "_raw" for k in keys}
    return (
        'INPUT LOOKUPS: direction_data=state.get("features_by_direction",{}).get(direction,{}); '
        f'value=direction_data.get("{key}",0.0); scale=(feature_stats or {{}}).get("{alias}",{{}}).get("p95",1.0). '
        "These are DIFFERENT dictionaries. Runtime direction_data keys are ONLY "
        + json.dumps(keys, separators=(",", ":"))
        + ". "
        "SOURCE statistic lookup mapping: "
        + json.dumps(pairs, sort_keys=True, separators=(",", ":"))
        + ". "
        "Statistic names such as vehicles_raw/pressure_raw/p95 and phase_active are not direction_data inputs. "
        'Compute active=float(phase_id==state.get("current_phase")); phase_active statistics only describe its binary support. '
        "Local variable names may be arbitrary; dictionary lookups must follow the declared paths."
    )


def render(generator, stage, parent=None, explanation=None, feedback=None):
    text = render_template(generator, stage, parent, explanation, feedback)
    marker = "CONSTRAINTS:"
    assert text.count(marker) == 1
    text = text.replace(marker, namespace_note(generator) + "\n" + marker, 1)
    old = "Guard transformed inputs and keep code readable."
    new = (
        "Guard transformed inputs and keep code readable. A guarded base does not make every transformed denominator nonzero: "
        "log(1), log1p(0) and sqrt(0) are zero. Compute scale=max(abs(float(value)),1e-8) when a positive magnitude scale is intended; "
        "for a signed ratio use an explicit abs(d)>1e-8 branch. Preserve monotonicity of the WHOLE utility when a denominator depends on runtime demand."
    )
    assert old in text
    text = text.replace(old, new, 1)
    if stage == "repair":
        extra = "REPAIR FOCUS: use the reported failing line or numeric counterexample to identify the affected lookup/arithmetic; recheck the complete function against all existing rules. Multiple changes are allowed; preserve the declared input/action interface and do not merely catch exceptions or return a constant."
        needle = "Output: one Python code block"
        assert needle in text
        text = text.replace(needle, extra + "\n" + needle, 1)
    return text
