"""Execute every original source input before calibration; cache only finite raw Q."""
import json
import math

def state_key(states):
    return json.dumps(states,sort_keys=False,separators=(',',':'),allow_nan=False)

def validate_all_source(validator,utility):
    result=dict(name='Complete original source execution',satisfied=False,source_bundles=0,raw_utility_values=0,details=[])
    loader=validator.data_loader;stats=validator.feature_stats
    if loader is None:
        result['details'].append('All-source numerical validation requires source observations')
        return result
    if hasattr(utility,'_validated_source_utilities'):
        del utility._validated_source_utilities
    cache=dict(feature_stats=stats,feature_stats_key=state_key(stats),values={})
    for inter in sorted(loader.intersection_ids):
        rows=loader.get_action_consistency_data(inter,max_timesteps=None)
        for index,row in enumerate(rows):
            states=row[0]
            try:
                if not states:raise ValueError('Empty source phase bundle')
                key=state_key(states)
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
                cache['values'][key]=(mode,values)
                result['source_bundles']+=1;result['raw_utility_values']+=len(scalars)
            except Exception as exc:
                result['details'].append(f'intersection={inter}, source_index={index}: {type(exc).__name__}: {exc}')
                return result
    if result['source_bundles']==0:
        result['details'].append('No usable original source observations')
        return result
    utility._validated_source_utilities=cache
    result.update(satisfied=True,scope='Every original teacher source bundle; finite counterfactual tests are separate')
    return result
