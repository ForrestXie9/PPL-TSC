"""Descriptive action diagnostics for one on-policy simulation rollout.

Raw (uncalibrated) actions are evaluated on states visited by the calibrated
policy. They are not a counterfactual uncalibrated rollout.
"""

from collections import defaultdict


class PolicyDecisionDiagnostics:
    def __init__(self):
        self._counts = defaultdict(lambda: {
            "decisions": 0,
            "switches": 0,
            "raw_switches": 0,
            "bias_changed_actions": 0,
        })

    def record(self, intersection, current_phase, selected_phase, raw_phase):
        counts = self._counts[int(intersection)]
        counts["decisions"] += 1
        counts["switches"] += int(selected_phase != current_phase)
        counts["raw_switches"] += int(raw_phase != current_phase)
        counts["bias_changed_actions"] += int(selected_phase != raw_phase)

    def summary(self):
        per_intersection = {}
        total = {key: 0 for key in (
            "decisions", "switches", "raw_switches", "bias_changed_actions"
        )}
        for intersection, counts in sorted(self._counts.items()):
            n = counts["decisions"]
            per_intersection[str(intersection)] = {
                **counts,
                "switch_rate": counts["switches"] / n,
                "raw_switch_rate": counts["raw_switches"] / n,
                "bias_changed_action_rate": counts["bias_changed_actions"] / n,
            }
            for key in total:
                total[key] += counts[key]
        n = total["decisions"]
        flags = []
        for intersection, item in per_intersection.items():
            if item['decisions'] >= 20:
                rate=item['switch_rate']
                if rate <= 0.01 or rate >= 0.99:
                    flags.append({'intersection':intersection,
                                  'kind':'near_always_hold' if rate <= 0.01 else 'near_always_switch',
                                  'decisions':item['decisions'],'switch_rate':rate})
        return {
            'diagnostic_flags': flags,
            'diagnostic_rule': {'minimum_decisions':20,'hold_rate_max':0.01,'switch_rate_min':0.99},
            'interpretation': 'Descriptive only: extreme action rates do not establish congestion, starvation, invalidity or causality; inspect demand and ATT. These flags do not affect fitness or verification.',
            **total,
            "switch_rate": total["switches"] / n if n else None,
            "raw_switch_rate": total["raw_switches"] / n if n else None,
            "bias_changed_action_rate": total["bias_changed_actions"] / n if n else None,
            "per_intersection": per_intersection,
            "raw_action_note": (
                "Raw actions are evaluated on states visited by the calibrated "
                "policy; this is not an uncalibrated rollout."
            ),
        }
