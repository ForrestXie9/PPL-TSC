"""Incoming lanes contribute once to each compatible movement direction."""
def lane_directions(roadlinks,direction_codes):
    result={}
    for link,direction in zip(roadlinks,direction_codes):
        if direction:
            for lane in link.n_startlane_id:
                result.setdefault(lane,set()).add(direction)
    return {lane:sorted(values) for lane,values in result.items()}


def restore_contract(config,run_config):
    semantics=run_config.get('direction_feature_semantics','legacy_first_match')
    if semantics not in {'legacy_first_match','compatible_shared_lane_mean_v1'}:
        raise ValueError('Unknown direction feature semantics: '+str(semantics))
    if run_config.get('method_profile')=='full_100_256_v1' and semantics!='compatible_shared_lane_mean_v1':
        raise ValueError('Current deployment requires its recorded compatible shared-lane contract')
    config['direction_feature_semantics']=semantics
    return config
