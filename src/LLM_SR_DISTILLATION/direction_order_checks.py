"""Finite direction-list permutations and concrete retry feedback."""
import copy
import math


def _spread(state, directions):
    features = state.get('features_by_direction', {})
    keys = {k for d in directions for k in features.get(d, {})}
    spread = 0.
    for key in keys:
        values = [features.get(d, {}).get(key) for d in directions]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            continue
        spread = max(spread, (max(values)-min(values))/max(1., max(abs(v) for v in values)))
    return spread


def direction_order_consistency(validator, utility, max_samples=25, max_checks=40):
    result = dict(name='Direction order consistency',
        description='Reordering the same controlled directions leaves raw Q unchanged',
        applicable=validator.action_mode != 'binary_keep_switch', satisfied=True,
        checked_pairs=0, details=[], probes=[], absolute_tolerance=1e-6, relative_tolerance=1e-10)
    if not result['applicable']:
        result['reason'] = 'Native KEEP/SWITCH uses binary inputs, not a controlled-direction list'
        return result
    loader = validator.data_loader
    groups, source_rows = {}, 0
    if loader is not None:
        for inter in sorted(loader.intersection_ids):
            rows = loader.get_action_consistency_data(inter, max_timesteps=None)
            if not rows:
                continue
            # Spread over source times and intersections. Prioritise unequal
            # direction measurements, since all-equal values hide position weights.
            n = min(max_samples, len(rows))
            indices = sorted({round(i*(len(rows)-1)/max(n-1, 1)) for i in range(n)})
            for index in indices:
                states = rows[index][0]
                if not states:
                    continue
                source_rows += 1
                for phase, state in states.items():
                    directions = list(state.get('controlled_directions', []))
                    if len(directions) < 2:
                        continue
                    alternatives = [list(reversed(directions)), directions[1:]+directions[:1]]
                    seen = set()
                    for alternative in alternatives:
                        if alternative == directions or tuple(alternative) in seen:
                            continue
                        seen.add(tuple(alternative))
                        key = (inter, phase, tuple(alternative))
                        priority = _spread(state, directions)
                        if key not in groups or priority > groups[key][0]:
                            groups[key] = (priority, inter, phase, state, alternative)
    for _, inter, phase, state, alternative in sorted(groups.values(), key=lambda row: -row[0])[:max_checks]:
        original = list(state['controlled_directions'])
        evidence = dict(intersection=inter, phase=phase, order_before=original, order_after=alternative)
        try:
            changed = copy.deepcopy(state)
            changed['controlled_directions'] = alternative
            before = float(utility.compute(copy.deepcopy(state), phase_id=phase,
                           feature_stats=validator.feature_stats, raise_on_error=True))
            after = float(utility.compute(changed, phase_id=phase,
                          feature_stats=validator.feature_stats, raise_on_error=True))
            evidence.update(Q_before=before if math.isfinite(before) else repr(before),
                            Q_after=after if math.isfinite(after) else repr(after))
            if not (math.isfinite(before) and math.isfinite(after)):
                raise ValueError('non-finite utility during direction reordering')
            if not math.isclose(before, after, abs_tol=1e-6, rel_tol=1e-10):
                result['details'].append(
                    f"intersection={inter}, phase={phase}: order {original} -> {alternative}; "
                    f"unchanged traffic, Q {before:.12g} -> {after:.12g}. "
                    "Fix priority derived from list position; use the same direction's actual traffic features "
                    "or explicit movement meaning consistently. Preserve current-phase rewards and external calibrated bias.")
        except Exception as exc:
            result['details'].append(f"intersection={inter}, phase={phase}, order {original} -> {alternative}: {type(exc).__name__}: {exc}")
        result['probes'].append(evidence)
        result['checked_pairs'] += 1
    if result['checked_pairs'] == 0:
        if source_rows and not groups:
            result.update(applicable=False, reason='No source phase has multiple controlled directions to permute')
        else:
            result['details'].append('Direction-order consistency not evaluated: no usable source comparisons')
    result['satisfied'] = not result['details']
    return result
