"""Execute source inputs and check finite replay before caching raw utilities."""
import json
import math

def state_key(states):
    return json.dumps(states,sort_keys=False,separators=(',',':'),allow_nan=False)


def raw_values(utility, states, stats):
    mode=utility.action_mode
    if mode=='phase_argmax':
        values=utility.compute_all_phases(states,feature_extractor=None,feature_stats=stats,phase_bias=None,raise_on_error=True)
        if set(values)!=set(states):raise ValueError('Raw utilities do not cover every source phase')
        values=tuple((p,float(q)) for p,q in values.items())
        scalars=[q for p,q in values]
    else:
        if mode=='binary_keep_switch':
            current,following,keep,switch=utility.compute_binary_action_values(states,feature_stats=stats,phase_bias=None,raise_on_error=True)
        else:
            ids=utility._sorted_phase_ids(states);current=states[ids[0]].get('current_phase',ids[0])
            if current not in states:current=ids[0]
            following=ids[(ids.index(current)+1)%len(ids)]
            keep=utility.compute(states[current],phase_id=current,feature_stats=stats,raise_on_error=True)
            switch=utility.compute(states[following],phase_id=following,feature_stats=stats,raise_on_error=True)
        values=(current,following,float(keep),float(switch));scalars=list(values[2:])
    if not all(math.isfinite(value) for value in scalars):raise ValueError('Nonfinite raw source utility')
    return mode,values


def validate_all_source(validator,utility):
    result=dict(name='Complete original source execution',satisfied=False,source_bundles=0,raw_utility_values=0,
                repeatability_bundles=0,repeatability_checks=0,details=[])
    loader=validator.data_loader;stats=validator.feature_stats
    if loader is None:
        result['details'].append('All-source numerical validation requires source observations')
        return result
    if hasattr(utility,'_validated_source_utilities'):
        del utility._validated_source_utilities
    cache=dict(feature_stats=stats,feature_stats_key=state_key(stats),values={})
    replay=[]
    for inter in sorted(loader.intersection_ids):
        rows=loader.get_action_consistency_data(inter,max_timesteps=None)
        for index,row in enumerate(rows):
            states=row[0]
            try:
                if not states:raise ValueError('Empty source phase bundle')
                key=state_key(states)
                value=raw_values(utility,states,stats)
                if key in cache['values'] and cache['values'][key]!=value:
                    raise ValueError('Identical source inputs produced different raw utilities; remove random or persistent state')
                if key not in cache['values'] and len(replay)<8:replay.append((states,key))
                cache['values'][key]=value
                result['source_bundles']+=1
                result['raw_utility_values']+=len(value[1]) if value[0]=='phase_argmax' else 2
            except Exception as exc:
                result['details'].append(f'intersection={inter}, source_index={index}: {type(exc).__name__}: {exc}')
                return result
    if result['source_bundles']==0:
        result['details'].append('No usable original source observations')
        return result
    # Revisit the same inputs after intervening calls, in both orders. The
    # bound adds at most 16 bundles and does not change the search/CMA budget.
    result['repeatability_bundles']=len(replay)
    try:
        for states,key in [*reversed(replay),*replay]:
            if raw_values(utility,states,stats)!=cache['values'][key]:
                raise ValueError('Identical source inputs produced different raw utilities; remove random or persistent state')
            result['repeatability_checks']+=1
    except Exception as exc:
        result['details'].append(f'Source repeatability: {type(exc).__name__}: {exc}')
        return result
    utility._validated_source_utilities=cache
    result.update(satisfied=True,scope='Every original source bundle plus bounded repeatability replay; not a proof of determinism or all-input safety')
    return result
