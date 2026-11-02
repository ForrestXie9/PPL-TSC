"""Source action summaries and same-state bias decomposition."""

import math


def summarize(counts, transitions, longest, current_matches, legal):
    rows = []
    for inter, hist in sorted(counts.items()):
        n = sum(hist.values())
        assert n > 0
        entropy = -sum((v / n) * math.log(v / n) for v in hist.values())
        phase_count = len(legal[inter])
        assert phase_count > 0
        normalized = entropy / math.log(phase_count) if phase_count > 1 else 0.0
        assert -1e-12 <= normalized <= 1 + 1e-12
        rows.append(
            dict(
                intersection=inter,
                decisions=n,
                requested_phase_counts=dict(hist),
                legal_phase_count=phase_count,
                distinct_requested_phases=len(hist),
                dominant_request_fraction=max(hist.values()) / n,
                normalized_request_entropy=normalized,
                longest_same_request_run=longest[inter],
                request_changes=transitions[inter],
                current_phase_matches=current_matches[inter],
            )
        )
    assert rows
    total = sum(r["decisions"] for r in rows)
    opportunities = sum(r["decisions"] - 1 for r in rows)
    return dict(
        intersections=len(rows),
        decisions=total,
        single_requested_phase_intersections=sum(r["distinct_requested_phases"] == 1 for r in rows),
        decision_weighted_dominant_request_fraction=sum(
            r["dominant_request_fraction"] * r["decisions"] for r in rows
        )
        / total,
        decision_weighted_normalized_request_entropy=sum(
            r["normalized_request_entropy"] * r["decisions"] for r in rows
        )
        / total,
        request_change_fraction=(
            sum(r["request_changes"] for r in rows) / opportunities if opportunities else 0.0
        ),
        current_phase_match_fraction=sum(r["current_phase_matches"] for r in rows) / total,
        per_intersection=rows,
    )


def decompose(example):
    raw = {int(k): float(v) for k, v in example["raw_utilities"].items()}
    adjusted = {int(k): float(v) for k, v in example["adjusted_utilities"].items()}
    assert raw.keys() == adjusted.keys() and all(
        math.isfinite(v) for v in [*raw.values(), *adjusted.values()]
    )
    a = max(raw, key=raw.get)
    b = max(adjusted, key=adjusted.get)
    assert b == int(example["requested_phase"])
    gap = raw[a] - raw[b]
    bias_advantage = (adjusted[b] - raw[b]) - (adjusted[a] - raw[a])
    adjusted_gap = adjusted[b] - adjusted[a]
    assert abs(adjusted_gap - (bias_advantage - gap)) < 1e-9
    return dict(
        raw_choice=a,
        requested_choice=b,
        bias_reversed_raw_choice=a != b,
        raw_pair_gap=gap,
        matched_bias_advantage=bias_advantage,
        adjusted_pair_advantage=adjusted_gap,
    )
