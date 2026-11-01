"""
特征描述字典，用于LLM提示词生成
根据不同的agent类型提供特征描述
"""

# 通用特征描述（基于config.json中的observation_feature_list）
FEATURE_DESCRIPTIONS = {
    # Phase相关特征
    'inter_2_current_phase': 'Current active phase index (one-hot encoded, 0 to num_phase-1)',
    'inter_current_phase': 'Current active phase index',
    
    # Lane相关特征
    'lane_2_num_vehicle': 'Number of vehicles in each lane',
    'lane_2_num_waiting_vehicle': 'Number of waiting vehicles in each lane',
    'lane_2_sum_waiting_time': 'Sum of waiting times for vehicles in each lane',
    'lane_2_delay': 'Average delay for vehicles in each lane (seconds)',
    'lane_2_pressure': 'Pressure on each lane (vehicles / capacity ratio)',
    
    # LaneLink相关特征
    'lanelink_2_num_vehicle': 'Number of vehicles on each lane link',
    'lanelink_2_pressure': 'Pressure on each lane link (vehicles / capacity ratio)',
    'lanelink_2_waiting_time': 'Average waiting time on each lane link (seconds)',
    
    # InLane相关特征
    'inlane_2_num_vehicle': 'Number of vehicles in incoming lanes',
    'inlane_2_num_waiting_vehicle': 'Number of waiting vehicles in incoming lanes',
    'inlane_2_sum_waiting_time': 'Sum of waiting times in incoming lanes',
    'inlane_2_delay': 'Average delay in incoming lanes (seconds)',
    'inlane_2_pressure': 'Pressure on incoming lanes',
    'inlane_2_vehicle_dist': 'Vehicle distribution in incoming lanes',
    
    # OutLane相关特征
    'outlane_2_num_vehicle': 'Number of vehicles in outgoing lanes',
    'outlane_2_vehicle_dist': 'Vehicle distribution in outgoing lanes',
    
    # Phase相关聚合特征
    'phase_2_num_vehicle': 'Number of vehicles in each phase',
    'phase_2_num_waiting_vehicle': 'Number of waiting vehicles in each phase',
    'phase_2_sum_waiting_time': 'Sum of waiting times for each phase',
    'phase_2_delay': 'Average delay for each phase (seconds)',
    'phase_2_pressure': 'Pressure for each phase',
    
    # Intersection相关特征
    'inter_2_num_waiting_vehicle': 'Total number of waiting vehicles at intersection',
    'inter_2_pressure': 'Total pressure at intersection',
}


def get_feature_descriptions_for_agent(agent_name: str, observation_feature_list: list) -> dict:
    """
    根据agent名称和观察特征列表获取特征描述
    
    Args:
        agent_name: Agent名称（如'MPLight', 'FRAP'等）
        observation_feature_list: 观察特征列表
    
    Returns:
        特征描述字典，key为特征名，value为描述
    """
    descriptions = {}
    
    for feature_name in observation_feature_list:
        if feature_name in FEATURE_DESCRIPTIONS:
            descriptions[feature_name] = FEATURE_DESCRIPTIONS[feature_name]
        else:
            # 如果特征名不在字典中，生成通用描述
            descriptions[feature_name] = f'Observation feature: {feature_name}'
    
    return descriptions


def get_feature_names_mapping(observation_feature_list: list) -> dict:
    """
    将特征名映射为简短的变量名（x0, x1, x2, ...）
    
    Args:
        observation_feature_list: 观察特征列表
    
    Returns:
        映射字典，key为特征名，value为变量名（如'x0', 'x1'）
    """
    mapping = {}
    for idx, feature_name in enumerate(observation_feature_list):
        mapping[feature_name] = f'x{idx}'
    return mapping


def format_features_for_prompt(feature_names: list, feature_descriptions: dict) -> str:
    """
    格式化特征列表为提示词文本
    
    Args:
        feature_names: 特征名列表
        feature_descriptions: 特征描述字典
    
    Returns:
        格式化的特征描述文本
    """
    lines = []
    for idx, name in enumerate(feature_names):
        var_name = f'x{idx}'
        desc = feature_descriptions.get(name, f'Feature {idx}: {name}')
        lines.append(f"- {var_name} ({name}): {desc}")
    return "\n".join(lines)

