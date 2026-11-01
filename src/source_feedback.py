"""Controlled refinements of the already evaluated combined source feedback."""
import hashlib
import json
import os
from collections import Counter

import numpy as np
from calibration_ties import actions
from source_rollout import render_rollout

MODES = {'combined', 'uniform', 'matched', 'student'}


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
        bias_spans.append(max(bias.values()) - min(bias.values()) if bias else 0.)
        pair_gaps, pair_biases, flips = [], [], []
        for index, ((mode, teacher, values), a, b) in enumerate(zip(samples, before, after)):
            scores = dict(values) if mode == 'phase_argmax' else {values[0]: values[2], values[1]: values[3]}
            spans.append(max(scores.values()) - min(scores.values()))
            if a != b:
                gap = scores[a] - scores[b]
                bias_advantage = bias.get(b, 0.) - bias.get(a, 0.)
                pair_gaps.append(gap)
                pair_biases.append(bias_advantage)
                row = dict(intersection=inter, probe_index=index, raw_choice=a,
                           calibrated_choice=b, teacher_choice=teacher,
                           raw_pair_gap=gap, matched_bias_advantage=bias_advantage,
                           calibrated_pair_advantage=bias_advantage - gap,
                           raw_scores=scores, phase_bias=bias,
                           before_correct=a == teacher, after_correct=b == teacher)
                flips.append(row)
        details.append(dict(intersection=inter, source_samples=len(samples),
                            raw_actions=dict(Counter(before)), calibrated_actions=dict(Counter(after)),
                            teacher_actions=dict(Counter(s[1] for s in samples)), flips=len(flips),
                            helps_teacher=sum(not x['before_correct'] and x['after_correct'] for x in flips),
                            hurts_teacher=sum(x['before_correct'] and not x['after_correct'] for x in flips),
                            median_flipped_raw_pair_gap=float(np.median(pair_gaps)) if pair_gaps else None,
                            median_matched_bias_advantage=float(np.median(pair_biases)) if pair_biases else None))
        reversals.extend(flips)
    if not raw:
        raise ValueError('No source states for diagnostic')
    n = len(raw)
    # Harmful teacher reversals first; ties have deterministic source ordering.
    reversals.sort(key=lambda x: (x['before_correct'] and not x['after_correct'],
                                 x['calibrated_pair_advantage']), reverse=True)
    return dict(fingerprint=hashlib.sha256(json.dumps(calibrated, separators=(',', ':')).encode()).hexdigest(),
                samples=n, sampled_calibrated_actions=calibrated, raw_ABF=sum(a == b for a, b in zip(raw, labels)) / n,
                calibrated_ABF=sum(a == b for a, b in zip(calibrated, labels)) / n,
                changed_fraction=sum(a != b for a, b in zip(raw, calibrated)) / n,
                median_raw_action_span=float(np.median(spans)), max_bias_span=max(bias_spans),
                per_intersection=details, reversal_examples=reversals[:2])


def profile(evolution, code, bias, uniform=False):
    from LLM_SR_DISTILLATION.function_utility import FunctionUtility
    bias = bias or {}
    cache = getattr(evolution, '_feedback_profiles', None)
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
            b = {int(k): float(v) for k, v in (bias.get(inter, bias.get(str(inter), {})) or {}).items()}
            blocks.append((inter, cached.samples, b))
            source_data[inter] = data
            coverage.append(dict(intersection=inter, total_source_states=total,
                                 sampled_states=len(data), first_decision=data[0][3], last_decision=data[-1][3]))
        result = profile_samples(blocks)
        result['coverage'] = coverage
        result['sampling'] = 'uniform whole source episode' if uniform else 'first up to 60 source timesteps'
        for example in result['reversal_examples']:
            sample = source_data[example['intersection']][example['probe_index']]
            relevant = {example['raw_choice'], example['calibrated_choice'], example['teacher_choice']}
            example['source_decision_index'] = sample[3]
            example['source_state'] = {p: state for p, state in sample[0].items() if p in relevant}
    finally:
        u._compiled_function.close()
    cache[key] = result
    return result


def render(mode, parent_profile, duplicate_count, distinct_codes, rollout=None):
    if mode not in MODES:
        raise ValueError(mode)
    p = parent_profile
    lines = [
        f"On the same source-state probe, {duplicate_count} other distinct evaluated programs chose exactly the same actions as this calibrated parent, among {distinct_codes} distinct successful codes.",
        'This is sampled behavioral overlap, not proof that programs are equivalent. Keep the existing objective. In the next ordinary mutation, consider meaningful changes to decision boundaries or traffic responses rather than cosmetic code changes. No fixed number of edits is required; do not invent absent features.',
    ]
    sampling = 'uniformly spaced over the whole source episode, up to 60 timesteps per intersection' if mode == 'uniform' else 'first up to 60 timesteps per intersection'
    lines.append(f"Source-only diagnostic on {p['samples']} states ({sampling}): sampled teacher agreement before calibration={p['raw_ABF']:.4f}, after={p['calibrated_ABF']:.4f}; calibration changes {100*p['changed_fraction']:.1f}% of actions.")
    if mode == 'matched':
        lines.append('For the same intersection, source state and action pair: raw_choice utility minus calibrated_choice utility is the raw_pair_gap; bias[calibrated_choice] minus bias[raw_choice] is the matched_bias_advantage. An action reversal occurs when this bias advantage exceeds the raw gap (with the existing tie rule). Counts below distinguish teacher-helping and teacher-hurting reversals; neither alone proves closed-loop control quality.')
        for row in sorted(p['per_intersection'], key=lambda x: x['flips'], reverse=True)[:3]:
            lines.append('Matched intersection diagnostic: ' + json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False))
        for row in p['reversal_examples']:
            lines.append('Same-state action reversal example: ' + json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False))
        if not p['reversal_examples']:
            lines.append('No action reversal occurred on this probe; that does not establish calibration safety outside these states.')
    else:
        # Keep baseline wording identical to the previous combined condition.
        lines.append(f"Median uncalibrated within-state action score span={p['median_raw_action_span']:.6g}; largest per-intersection bias span={p['max_bias_span']:.6g}. Each bias is bounded by +/-1, so a pairwise bias difference can reach 2.")
    lines.append('These are source teacher states, not a causal diagnosis of closed-loop traffic. The parent source ATT above was evaluated with calibration. Preserve demand-responsive utility differences; offsets can dominate a nearly flat program. Improve the existing fidelity/control tradeoff without changing its scoring formula, hard-coding intersection identities, or accessing target statistics.')
    if mode == 'student':
        lines.append(render_rollout(rollout))
    return '\n'.join(lines)


def _feedback(evolution, parent, mode):
    uniform = mode == 'uniform'
    parent_profile = profile(evolution, parent.function_code, parent.per_intersection_bias, uniform)
    observed = {code: profile(evolution, code, record.get('per_intersection_bias'), uniform)
                for code, record in evolution._raw_evaluation_cache.items()}
    observed[parent.function_code] = parent_profile
    duplicate_count = sum(code != parent.function_code and p['fingerprint'] == parent_profile['fingerprint']
                          for code, p in observed.items())
    rollout = (parent.metrics or {}).get('source_rollout_feedback')
    text = render(mode, parent_profile, duplicate_count, len(observed), rollout)
    from search_feedback_addendum import make_addendum
    search_variant=os.environ.get('PPL_SEARCH_FEEDBACK','baseline')
    search_appendix=make_addendum(search_variant,evolution,parent,parent_profile,observed,rollout)
    if search_appendix:text+='\n'+search_appendix
    variant = os.environ.get('PPL_PHASE_BIAS_FEEDBACK', 'baseline')
    if variant not in {'baseline', 'phase_bias_summary'}:
        raise ValueError('Unknown phase/bias feedback variant: ' + variant)
    enhanced_feedback = None
    if variant == 'phase_bias_summary':
        from source_phase_summary import render as render_phase_bias
        enhanced_feedback = render_phase_bias(rollout)
        text += '\n' + enhanced_feedback
    path = os.environ.get('PPL_FEEDBACK_TRACE')
    if path:
        record = dict(mode=mode, generation=evolution.current_generation_index,
                      parent_code_sha256=hashlib.sha256(parent.function_code.encode()).hexdigest(),
                      parent=parent_profile, duplicate_other_codes=duplicate_count,
                      observed_codes=len(observed), unique_observed_behaviors=len({p['fingerprint'] for p in observed.values()}),
                      source_rollout=rollout, feedback=text, feedback_variant=variant, enhanced_feedback=enhanced_feedback,search_feedback_variant=search_variant,search_feedback_appendix=search_appendix)
        with open(path, 'a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
    return text


def feedback(evolution, parent, mode):
    if os.environ.get("PPL_METHOD_PROFILE") != "full_100_256_v1":
        return None
    try:
        return _feedback(evolution, parent, mode)
    except Exception as exc:
        path = os.environ.get('PPL_FEEDBACK_TRACE')
        if path:
            with open(path, 'a') as f:
                f.write(json.dumps(dict(mode=mode, status='diagnostic_failed', error=type(exc).__name__ + ': ' + str(exc))) + '\n')
        return 'Additional source behavior diagnostics were unavailable; use the existing parent metrics and constraints.'
