"""Deterministic diagnostics over existing source teacher states."""

import hashlib
import json
from collections import Counter
import numpy as np
from ppl_tsc.calibration.ties import actions


def probe_indices(length, uniform=False, limit=60):
    if length <= limit:
        return list(range(length))
    if uniform:
        return [i * (length - 1) // (limit - 1) for i in range(limit)]
    return list(range(limit))


def probe_data(loader, intersection, uniform=False):
    data = loader.get_action_consistency_data(intersection, include_timestep=True)
    return [data[i] for i in probe_indices(len(data), uniform)], len(data)


def profile_samples(blocks):
    raw, calibrated, labels, spans, bias_spans = [], [], [], [], []
    details, reversals = [], []
    for inter, samples, bias in blocks:
        before = actions(samples, {})
        after = actions(samples, bias)
        raw.extend((inter, a) for a in before)
        calibrated.extend((inter, a) for a in after)
        labels.extend((inter, s[1]) for s in samples)
        bias_spans.append(max(bias.values()) - min(bias.values()) if bias else 0.0)
        pair_gaps, pair_biases, flips = [], [], []
        for index, ((mode, teacher, values), a, b) in enumerate(zip(samples, before, after)):
            scores = (
                dict(values)
                if mode == "phase_argmax"
                else {values[0]: values[2], values[1]: values[3]}
            )
            spans.append(max(scores.values()) - min(scores.values()))
            if a != b:
                gap = scores[a] - scores[b]
                bias_advantage = bias.get(b, 0.0) - bias.get(a, 0.0)
                pair_gaps.append(gap)
                pair_biases.append(bias_advantage)
                row = dict(
                    intersection=inter,
                    probe_index=index,
                    raw_choice=a,
                    calibrated_choice=b,
                    teacher_choice=teacher,
                    raw_pair_gap=gap,
                    matched_bias_advantage=bias_advantage,
                    calibrated_pair_advantage=bias_advantage - gap,
                    raw_scores=scores,
                    phase_bias=bias,
                    before_correct=a == teacher,
                    after_correct=b == teacher,
                )
                flips.append(row)
        details.append(
            dict(
                intersection=inter,
                source_samples=len(samples),
                raw_actions=dict(Counter(before)),
                calibrated_actions=dict(Counter(after)),
                teacher_actions=dict(Counter(s[1] for s in samples)),
                flips=len(flips),
                helps_teacher=sum(not x["before_correct"] and x["after_correct"] for x in flips),
                hurts_teacher=sum(x["before_correct"] and not x["after_correct"] for x in flips),
                median_flipped_raw_pair_gap=float(np.median(pair_gaps)) if pair_gaps else None,
                median_matched_bias_advantage=(
                    float(np.median(pair_biases)) if pair_biases else None
                ),
            )
        )
        reversals.extend(flips)
    if not raw:
        raise ValueError("No source states for diagnostic")
    n = len(raw)
    # Harmful teacher reversals first; ties have deterministic source ordering.
    reversals.sort(
        key=lambda x: (
            x["before_correct"] and not x["after_correct"],
            x["calibrated_pair_advantage"],
        ),
        reverse=True,
    )
    return dict(
        fingerprint=hashlib.sha256(
            json.dumps(calibrated, separators=(",", ":")).encode()
        ).hexdigest(),
        samples=n,
        sampled_calibrated_actions=calibrated,
        raw_ABF=sum(a == b for a, b in zip(raw, labels)) / n,
        calibrated_ABF=sum(a == b for a, b in zip(calibrated, labels)) / n,
        changed_fraction=sum(a != b for a, b in zip(raw, calibrated)) / n,
        median_raw_action_span=float(np.median(spans)),
        max_bias_span=max(bias_spans),
        per_intersection=details,
        reversal_examples=reversals[:2],
    )


def profile(evolution, code, bias, uniform=False):
    from ppl_tsc.runtime.utility import FunctionUtility

    bias = bias or {}
    cache = getattr(evolution, "_feedback_profiles", None)
    if cache is None:
        cache = {}
        evolution._feedback_profiles = cache
    key = (code, json.dumps(bias, sort_keys=True), uniform)
    if key in cache:
        return cache[key]
    u = FunctionUtility(code, action_mode=evolution.action_mode)
    try:
        blocks, source_data, coverage = [], {}, []
        for inter in sorted(evolution.data_loader.intersection_ids):
            data, total = probe_data(evolution.data_loader, inter, uniform)
            cached = evolution.evaluator._build_cached_bias_consistency(u, data)
            b = {
                int(k): float(v)
                for k, v in (bias.get(inter, bias.get(str(inter), {})) or {}).items()
            }
            blocks.append((inter, cached.samples, b))
            source_data[inter] = data
            coverage.append(
                dict(
                    intersection=inter,
                    total_source_states=total,
                    sampled_states=len(data),
                    first_decision=data[0][3],
                    last_decision=data[-1][3],
                )
            )
        result = profile_samples(blocks)
        result["coverage"] = coverage
        result["sampling"] = (
            "uniform whole source episode" if uniform else "first up to 60 source timesteps"
        )
        for example in result["reversal_examples"]:
            sample = source_data[example["intersection"]][example["probe_index"]]
            relevant = {
                example["raw_choice"],
                example["calibrated_choice"],
                example["teacher_choice"],
            }
            example["source_decision_index"] = sample[3]
            example["source_state"] = {p: state for p, state in sample[0].items() if p in relevant}
    finally:
        u._compiled_function.close()
    cache[key] = result
    return result
