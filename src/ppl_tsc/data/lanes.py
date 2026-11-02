"""Incoming lanes contribute once to each compatible movement direction."""


def lane_directions(roadlinks, direction_codes):
    result = {}
    for link, direction in zip(roadlinks, direction_codes):
        if direction:
            for lane in link.n_startlane_id:
                result.setdefault(lane, set()).add(direction)
    return {lane: sorted(values) for lane, values in result.items()}


def restore_contract(config, run_config):
    semantics = run_config.get("direction_feature_semantics", "first_matching_roadlink")
    if semantics not in {"first_matching_roadlink", "compatible_shared_lane_mean"}:
        raise ValueError("Unknown direction feature semantics: " + str(semantics))
    if (
        run_config.get("method_profile") == "full"
        and semantics != "compatible_shared_lane_mean"
    ):
        raise ValueError("Current deployment requires its recorded compatible shared-lane contract")
    config["direction_feature_semantics"] = semantics
    return config
