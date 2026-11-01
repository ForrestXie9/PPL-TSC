"""Bounded diagnostics from an already scheduled student source simulation.

No environment getters, teacher labels, policy changes, or additional episodes.
Current-phase coverage is a diagnostic proxy, not a starvation certificate.
"""
import copy
import math


class SourceRolloutRecorder:
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
                self.errors.append(type(exc).__name__ + ': ' + str(exc))

    def _observe(self, intersection, states, requested_action):
        # Read only the state already passed to the policy. Never query env again.
        step = self.steps.get(intersection, 0)
        self.steps[intersection] = step + 1
        self.decisions += 1
        current = int(next(iter(states.values()))['current_phase'])
        covered = set(states[current].get('controlled_directions', []))
        directions = {}
        for state in states.values():
            for direction, features in state.get('features_by_direction', {}).items():
                directions.setdefault(direction, features)
        active = set()
        for direction, features in directions.items():
            # Pressure is signed and is not a vehicle count or a queue length.
            key = next((k for k in ('num_waiting_vehicle', 'num_vehicle', 'pressure')
                        if k in features), None)
            if key is None:
                continue
            value = float(features[key])
            if not math.isfinite(value):
                raise ValueError('Nonfinite diagnostic feature')
            ident = (intersection, direction, key)
            if value <= 0 or direction in covered:
                self.runs.pop(ident, None)
                continue
            active.add(ident)
            old = self.runs.get(ident)
            run = {'count': old['count'] + 1, 'first': old['first']} if old else {'count': 1, 'first': value}
            self.runs[ident] = run
            if run['count'] < 3:
                continue
            growth = value - run['first']
            relative_growth = growth / max(abs(run['first']), 1e-12)
            rank = (run['count'], relative_growth)
            previous = self.best.get(ident)
            if previous is not None and rank <= previous['_rank']:
                continue
            self.best[ident] = dict(
                _rank=rank, intersection=intersection, decision_index=step,
                source_time_seconds=step * self.interval,
                current_phase=current, requested_phase=int(requested_action),
                direction=direction, feature=key, first_value=run['first'],
                latest_value=value, growth=growth,
                consecutive_observations=run['count'], states=copy.deepcopy(states),
            )
            # Bounded retained state even with many intersections/directions.
            if len(self.best) > 12:
                worst = min(self.best, key=lambda k: self.best[k]['_rank'])
                del self.best[worst]
        for ident in list(self.runs):
            if ident[0] == intersection and ident not in active:
                del self.runs[ident]

    def finish(self, evaluator, utility, bias):
        examples = []
        # Prefer different directions, cap at 3 globally; no target information.
        for item in sorted(self.best.values(), key=lambda x: x['_rank'], reverse=True)[:self.limit]:
            row = {k: v for k, v in item.items() if k not in {'_rank', 'states'}}
            try:
                cached = evaluator._build_cached_bias_consistency(utility, [(item['states'], item['requested_phase'])])
                mode, _, values = cached.samples[0]
                inter_bias = (bias or {}).get(item['intersection'], (bias or {}).get(str(item['intersection']), {})) or {}
                inter_bias = {int(k): float(v) for k, v in inter_bias.items()}
                if mode == 'phase_argmax':
                    raw = dict(values)
                else:
                    current, following, keep, switch = values
                    raw = {current: keep, following: switch}
                adjusted = {p: q + inter_bias.get(p, 0.) for p, q in raw.items()}
                selected = max(adjusted, key=adjusted.get)
                if selected != item['requested_phase']:
                    raise ValueError('Recorded request disagrees with recomputed legal action')
                row.update(raw_utilities=raw, adjusted_utilities=adjusted,
                           legal_scored_phases=list(raw), source_state=item['states'])
                examples.append(row)
            except Exception as exc:
                if len(self.errors) < 3:
                    self.errors.append(type(exc).__name__ + ': ' + str(exc))
        return dict(decisions=self.decisions, examples=examples, errors=self.errors,
                    origin='existing calibrated student source episode',
                    diagnostic_only=True, extra_episodes=0, extra_teacher_queries=0)


def render_rollout(result):
    import json
    if result is None:
        raise ValueError('Missing source student rollout diagnostic')
    if result.get('errors'):
        raise ValueError('Source student rollout diagnostic failed: ' + '; '.join(result['errors']))
    lines = [
        f"Observed {result['decisions']} decisions in the parent's existing calibrated student source episode; no additional episode or teacher query.",
        'The following bounded examples show positive observed demand/pressure while a direction was outside the recorded current phase for at least three successive observations. Requested phase is the next command, not proof of immediate service. Pressure is signed, is not a queue count, and positive pressure alone does not prove congestion. Growth is measured on that same direction and feature. These are diagnostic correlations, not causal or starvation certificates; phase overlap and transition delays matter.',
    ]
    if not result['examples']:
        lines.append('No example met this fixed diagnostic rule; this does not prove that traffic was good.')
    for example in result['examples']:
        lines.append('Student source example: ' + json.dumps(example, sort_keys=True, separators=(',', ':'), allow_nan=False))
    lines.append('Use only the existing feature interface and legal actions. Consider general traffic-responsive structural changes within the next ordinary mutation; do not encode example intersection IDs, phase IDs or times as constants. Keep the original ABF/control objective and calibration procedure.')
    return '\n'.join(lines)
