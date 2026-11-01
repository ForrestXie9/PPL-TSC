"""Independent source-only mutation feedback variants. No policy or fitness changes."""
import hashlib
import json
VARIANTS=('baseline','scale','examples','diversity')
MARKERS={'scale':'SOURCE MATCHED SCALE FEEDBACK v1','examples':'SOURCE COMPACT COUNTEREXAMPLES v1','diversity':'SOURCE BEHAVIORAL NEIGHBORS v1'}
def dump(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def state_view(states,relevant,mode):
    states={int(k):v for k,v in states.items()}
    if mode=='binary_keep_switch':
        s=next(iter(states.values()));cur=int(s['current_phase']);keys=sorted(states);nxt=keys[(keys.index(cur)+1)%len(keys)]
        return dict(current_phase=cur,next_phase=nxt,binary_features=s['binary_features'],binary_action_ids={'KEEP':0,'SWITCH':1})
    return {str(k):states[k] for k in sorted(set(map(int,relevant))) if k in states}
def make_addendum(variant,evolution,parent,profile,observed,rollout):
    if variant not in VARIANTS:raise ValueError('Unknown feedback variant')
    if variant=='baseline':return None
    lines=[MARKERS[variant], 'Only already available SOURCE evidence follows. Keep the existing candidate budget, teacher-agreement calibration, +/-1 biases and ABF+control fitness. Do not hard-code example identities or times; no target information is present.']
    if variant=='scale':
        lines.append('The existing span summary can hide state/intersection differences. Compare utility and bias differences for the SAME state and action pair below. A large bias range alone does not prove harmful calibration; teacher-helping reversals may be useful. Adjust structure or interpretable score scale only if supported; no prescribed scale, formula or mandatory edit count.')
        rows=sorted(profile['per_intersection'],key=lambda x:(-x['flips'],x['intersection']))[:2]
        lines.append('Matched source summaries: '+dump(rows))
        for item in profile['reversal_examples'][:2]:
            row={k:v for k,v in item.items() if k!='source_state'}
            row['source_state']=state_view(item['source_state'],[item['raw_choice'],item['calibrated_choice'],item['teacher_choice']],evolution.action_mode)
            lines.append('Matched source teacher-state example: '+dump(row))
        if not profile['reversal_examples']:lines.append('No sampled action reversal occurred; do not invent one or force a scale change.')
    elif variant=='examples':
        assert rollout is not None and not rollout.get('errors')
        lines.append('These are from the existing calibrated student source episode, not teacher trajectories. They show positive observed demand/pressure outside the recorded current phase across successive observations; requests do not prove immediate service. Positive signed pressure is not a queue length. These correlations do not certify starvation or identify a cause. No teacher labels exist for these snapshots.')
        for item in rollout['examples'][:2]:
            keys=['intersection','decision_index','direction','feature','first_value','latest_value','growth','consecutive_observations','current_phase','requested_phase','raw_utilities','adjusted_utilities','phase_bias_decomposition']
            row={k:item[k] for k in keys}
            states={int(k):v for k,v in item['source_state'].items()};cur=item['current_phase'];req=item['requested_phase']
            serving=next((k for k,s in sorted(states.items()) if item['direction'] in s.get('controlled_directions',[])),cur)
            row['source_state']=state_view(states,[cur,req,serving],evolution.action_mode)
            lines.append('Compact student source example: '+dump(row))
        if not rollout['examples']:lines.append('No snapshot met the fixed recorder rule; this is not a success certificate.')
        lines.append('Consider a general demand-responsive decision boundary that addresses the observed pattern while preserving teacher fidelity. Existing validation and total fitness still decide admission; no new safety penalty or rejection rule is added.')
    else:
        reference=profile['sampled_calibrated_actions'];neighbors=[]
        for code,p in observed.items():
            if code==parent.function_code:continue
            other=p['sampled_calibrated_actions'];assert len(other)==len(reference)
            assert [v[0] for v in other]==[v[0] for v in reference]
            disagree=sum(a!=b for a,b in zip(reference,other))
            record=evolution._raw_evaluation_cache.get(code)
            if not record:continue
            metrics=record['all_results']['average']
            neighbors.append(dict(code_sha256=hashlib.sha256(code.encode()).hexdigest(),disagreements=disagree,sampled_states=len(reference),disagreement_fraction=disagree/len(reference),joint_fitness=metrics['score'],source_ABF=metrics['behavioral_fidelity'],source_ATT=metrics['student_travel_time'],_profile=p))
        neighbors.sort(key=lambda r:(r['disagreement_fraction'],-r['joint_fitness'],r['code_sha256']))
        lines.append('Existing feedback counts exact sampled duplicates. Here are up to three nearest distinct evaluated programs, including near-overlap and their unchanged joint fitness: '+dump([{k:v for k,v in r.items() if k!='_profile'} for r in neighbors[:3]]))
        distinct=next((r for r in neighbors if r['disagreements']),None)
        if distinct:
            other=distinct['_profile']['sampled_calibrated_actions'];idx=next(i for i,(a,b) in enumerate(zip(reference,other)) if a!=b)
            inter,choice=reference[idx];other_choice=other[idx][1]
            offset=sum(1 for i,_ in reference[:idx] if i==inter)
            from source_feedback import probe_data
            samples,_=probe_data(evolution.data_loader,inter,False);state,teacher,*rest=samples[offset]
            lines.append('Source state distinguishing sampled behaviors: '+dump(dict(intersection=inter,source_decision_index=rest[-1],parent_requested_phase=choice,neighbor_requested_phase=other_choice,teacher_phase=teacher,source_state=state_view(state,[choice,other_choice,teacher],evolution.action_mode))))
        lines.append('Sampled overlap does not establish global equivalence or lack of usefulness. Try a traffic-meaningful structural change where supported, instead of cosmetic rewrites. Do not change actions merely to be different; diversity is feedback only, never a fitness term or an automatic rejection rule.')
    return '\n'.join(lines)
