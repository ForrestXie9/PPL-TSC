"""Sampled label consistency for physical-phase argmax utilities.

Relabel within the observed ID set; never permute KEEP/SWITCH action IDs.
This checks raw utility. Calibrated physical-phase offsets remain external
and must follow their physical phase when a deployment is relabelled.
"""
import copy
import math
from collections import OrderedDict, deque


def phase_label_consistency(validator, utility, max_samples=25, max_checks=40):
    # Import here: semantic_checks also supplies the shared prompt contract.
    from .semantic_checks import TOLERANCE, _bundles

    result = dict(name="Physical phase label consistency",
                  description="Raw utility is unchanged by consistent physical-phase relabelling",
                  applicable=validator.action_mode == "phase_argmax", satisfied=True,
                  checked_pairs=0, details=[], probes=[])
    if not result['applicable']:
        result['reason'] = "Only phase_argmax is relabelled; binary action identities and cyclic successor semantics are preserved"
        return result

    groups = OrderedDict()
    bundles = _bundles(validator, max_samples)
    for inter, states in bundles:
        ids = sorted(states)
        if len(ids) < 2:
            continue
        # Use only valid existing labels. Two distinct deterministic
        # permutations help expose arithmetic on IDs without growing search.
        targets = [ids[1:] + ids[:1], list(reversed(ids))]
        mappings = []
        for target in targets:
            mapping = dict(zip(ids, target))
            if mapping not in mappings:
                mappings.append(mapping)
        for phase, state in states.items():
            for index, mapping in enumerate(mappings):
                groups.setdefault((inter, phase, index), deque()).append((state, mapping))

    queues = deque(groups.items())
    while queues and result['checked_pairs'] < max_checks:
        (inter, phase, index), rows = queues.popleft()
        state, mapping = rows.popleft()
        current = state.get('current_phase')
        evidence = dict(intersection=inter, phase_before=phase, phase_after=mapping[phase],
                        current_before=current, current_after=mapping.get(current),
                        mapping=mapping)
        label = f"intersection={inter}, phase={phase}->{mapping[phase]}, current={current}->{mapping.get(current)}"
        try:
            if current not in mapping:
                raise ValueError("current_phase is missing from the observed physical phase IDs")
            changed = copy.deepcopy(state)
            changed['current_phase'] = mapping[current]
            before = float(utility.compute(copy.deepcopy(state), phase_id=phase,
                           feature_stats=validator.feature_stats, raise_on_error=True))
            after = float(utility.compute(changed, phase_id=mapping[phase],
                          feature_stats=validator.feature_stats, raise_on_error=True))
            evidence.update(utility_before=before if math.isfinite(before) else repr(before),
                            utility_after=after if math.isfinite(after) else repr(after))
            if not (math.isfinite(before) and math.isfinite(after)):
                raise ValueError("non-finite utility in phase-label comparison")
            if abs(after - before) > TOLERANCE:
                result['details'].append(
                    f"{label}: unchanged traffic but Q {before:.9g} -> {after:.9g}. "
                    "Use phase IDs to identify the current phase, not as numeric demand or priority. "
                    "Keep physical-phase calibration offsets external and mapped to their phase.")
        except Exception as exc:
            result['details'].append(f"{label}: {type(exc).__name__}: {exc}")
        result['probes'].append(evidence)
        result['checked_pairs'] += 1
        if rows:
            queues.append(((inter, phase, index), rows))

    if result['checked_pairs'] == 0:
        if bundles and all(len(states) == 1 for _, states in bundles):
            result.update(applicable=False, reason="Only one physical phase: no within-domain relabelling comparison")
        else:
            result['details'].append("Phase-label consistency not evaluated: no usable source phase comparisons")
    result['satisfied'] = not result['details']
    return result
