"""Feature descriptions used to construct LLM prompts for different agent interfaces."""


FEATURE_DESCRIPTIONS = {

    'inter_2_current_phase': 'Current active phase index (one-hot encoded, 0 to num_phase-1)',
    'inter_current_phase': 'Current active phase index',


    'lane_2_num_vehicle': 'Number of vehicles in each lane',
    'lane_2_num_waiting_vehicle': 'Number of waiting vehicles in each lane',
    'lane_2_sum_waiting_time': 'Sum of waiting times for vehicles in each lane',
    'lane_2_delay': 'Average delay for vehicles in each lane (seconds)',
    'lane_2_pressure': 'Pressure on each lane (vehicles / capacity ratio)',


    'lanelink_2_num_vehicle': 'Number of vehicles on each lane link',
    'lanelink_2_pressure': 'Pressure on each lane link (vehicles / capacity ratio)',
    'lanelink_2_waiting_time': 'Average waiting time on each lane link (seconds)',


    'inlane_2_num_vehicle': 'Number of vehicles in incoming lanes',
    'inlane_2_num_waiting_vehicle': 'Number of waiting vehicles in incoming lanes',
    'inlane_2_sum_waiting_time': 'Sum of waiting times in incoming lanes',
    'inlane_2_delay': 'Average delay in incoming lanes (seconds)',
    'inlane_2_pressure': 'Pressure on incoming lanes',
    'inlane_2_vehicle_dist': 'Vehicle distribution in incoming lanes',


    'outlane_2_num_vehicle': 'Number of vehicles in outgoing lanes',
    'outlane_2_vehicle_dist': 'Vehicle distribution in outgoing lanes',


    'phase_2_num_vehicle': 'Number of vehicles in each phase',
    'phase_2_num_waiting_vehicle': 'Number of waiting vehicles in each phase',
    'phase_2_sum_waiting_time': 'Sum of waiting times for each phase',
    'phase_2_delay': 'Average delay for each phase (seconds)',
    'phase_2_pressure': 'Pressure for each phase',


    'inter_2_num_waiting_vehicle': 'Total number of waiting vehicles at intersection',
    'inter_2_pressure': 'Total pressure at intersection',
}


def get_feature_descriptions_for_agent(agent_name: str, observation_feature_list: list) -> dict:
    """Look up descriptions for the agent observation features."""
    descriptions = {}

    for feature_name in observation_feature_list:
        if feature_name in FEATURE_DESCRIPTIONS:
            descriptions[feature_name] = FEATURE_DESCRIPTIONS[feature_name]
        else:

            descriptions[feature_name] = f'Observation feature: {feature_name}'

    return descriptions


def get_feature_names_mapping(observation_feature_list: list) -> dict:
    """Map observation features to compact variable names x0, x1, and so on."""
    mapping = {}
    for idx, feature_name in enumerate(observation_feature_list):
        mapping[feature_name] = f'x{idx}'
    return mapping


def format_features_for_prompt(feature_names: list, feature_descriptions: dict) -> str:
    """Format feature names and descriptions as prompt text."""
    lines = []
    for idx, name in enumerate(feature_names):
        var_name = f'x{idx}'
        desc = feature_descriptions.get(name, f'Feature {idx}: {name}')
        lines.append(f"- {var_name} ({name}): {desc}")
    return "\n".join(lines)

