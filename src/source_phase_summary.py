"""Source episode request summaries and same-state bias decomposition."""
import json
import math

MARKER = 'SOURCE STUDENT PHASE/BIASED UTILITY FEEDBACK v1'


def summarize(counts, transitions, longest, current_matches, legal):
    rows=[]
    for inter,hist in sorted(counts.items()):
        n=sum(hist.values());assert n>0
        entropy=-sum((v/n)*math.log(v/n) for v in hist.values())
        phase_count=len(legal[inter]);assert phase_count>0
        normalized=entropy/math.log(phase_count) if phase_count>1 else 0.
        assert -1e-12<=normalized<=1+1e-12
        rows.append(dict(intersection=inter,decisions=n,requested_phase_counts=dict(hist),
                         legal_phase_count=phase_count,distinct_requested_phases=len(hist),
                         dominant_request_fraction=max(hist.values())/n,
                         normalized_request_entropy=normalized,
                         longest_same_request_run=longest[inter],
                         request_changes=transitions[inter],current_phase_matches=current_matches[inter]))
    assert rows
    total=sum(r['decisions'] for r in rows)
    opportunities=sum(r['decisions']-1 for r in rows)
    return dict(intersections=len(rows),decisions=total,
                single_requested_phase_intersections=sum(r['distinct_requested_phases']==1 for r in rows),
                decision_weighted_dominant_request_fraction=sum(r['dominant_request_fraction']*r['decisions'] for r in rows)/total,
                decision_weighted_normalized_request_entropy=sum(r['normalized_request_entropy']*r['decisions'] for r in rows)/total,
                request_change_fraction=sum(r['request_changes'] for r in rows)/opportunities if opportunities else 0.,
                current_phase_match_fraction=sum(r['current_phase_matches'] for r in rows)/total,
                per_intersection=rows)


def decompose(example):
    raw={int(k):float(v) for k,v in example['raw_utilities'].items()}
    adjusted={int(k):float(v) for k,v in example['adjusted_utilities'].items()}
    assert raw.keys()==adjusted.keys() and all(math.isfinite(v) for v in [*raw.values(),*adjusted.values()])
    a=max(raw,key=raw.get);b=max(adjusted,key=adjusted.get)
    assert b==int(example['requested_phase'])
    gap=raw[a]-raw[b]
    bias_advantage=(adjusted[b]-raw[b])-(adjusted[a]-raw[a])
    adjusted_gap=adjusted[b]-adjusted[a]
    assert abs(adjusted_gap-(bias_advantage-gap))<1e-9
    return dict(raw_choice=a,requested_choice=b,bias_reversed_raw_choice=a!=b,
                raw_pair_gap=gap,matched_bias_advantage=bias_advantage,
                adjusted_pair_advantage=adjusted_gap)


def render(result):
    from source_rollout_legacy import render_rollout
    if result is None or result.get('errors'):
        raise ValueError('Source phase/bias diagnostic unavailable or failed')
    summary=result['phase_request_summary']
    assert summary['decisions']==result['decisions']
    top=sorted(summary['per_intersection'],key=lambda r:(-r['dominant_request_fraction'],-r['longest_same_request_run'],r['intersection']))[:3]
    overview={k:v for k,v in summary.items() if k!='per_intersection'}
    lines=[MARKER,
           'These summaries cover the complete existing calibrated student SOURCE episode, not the sampled teacher states above. No extra environment read, source episode, teacher query, or target sample was added.',
           'Complete source requested-action summary: '+json.dumps(overview,sort_keys=True,separators=(',',':'),allow_nan=False),
           'High request concentration can be appropriate for asymmetric demand. It is a diagnostic, not a failure label. Requests are commands, not the physical executed phase or proof of service.',
           'Up to three source intersections with the most concentrated requests: '+json.dumps(top,sort_keys=True,separators=(',',':'),allow_nan=False),
           render_rollout(result),
           'Each source example below compares the SAME intersection, state, and raw/requested action pair. raw_pair_gap is Q(raw_choice)-Q(requested_choice); matched_bias_advantage is bias(requested_choice)-bias(raw_choice). A bias-induced reversal is an action disagreement between the same raw and adjusted utilities, not proof that the teacher action or traffic decision is wrong.']
    for example in result['examples']:
        row=dict(intersection=example['intersection'],decision_index=example['decision_index'],**example['phase_bias_decomposition'])
        lines.append('Same student-state bias decomposition: '+json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False))
    lines.append('In the next ordinary mutation, use this evidence to consider traffic-responsive structural changes and the scale of raw utility differences relative to the fixed +/-1 phase offsets. No particular scale, edit count, phase choice, penalty or fitness term is prescribed. Preserve the original two-term fitness, teacher-agreement calibration, documented feature types, finite arithmetic and legal action interface. Do not hard-code diagnostic intersection IDs, phase IDs or times. There are no teacher labels on these student snapshots; do not invent them.')
    return '\n'.join(lines)
