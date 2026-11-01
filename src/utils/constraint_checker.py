"""
检测大模型生成的通用相位效用函数应该满足的约束
包括边界条件和变化趋势检查
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import re


def _load_timesteps_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载timesteps数据（内部辅助函数）
    
    Args:
        json_file_path: JSON文件路径
    
    Returns:
        timesteps列表
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从JSON数据中提取timesteps
    timesteps = data.get('data', {}).get('timesteps', [])
    
    if not timesteps:
        # 如果是JSONL格式，尝试读取
        timesteps = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'intersection_id' in record and 'phases' in record:
                        timesteps.append(record)
                except:
                    pass
    
    return timesteps


def check_phase_utility_constraints(
    timesteps: List[Dict[str, Any]],
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    【已废弃】检测收集的老师数据本身的约束（检查teacher Q/logits是否满足约束）
    
    注意：此函数用于检查收集的老师数据质量，不是检查大模型生成的函数表达式。
    如果要检查生成的函数表达式，请使用 check_generated_function_for_teacher_model()
    
    Args:
        timesteps: 时间步数据列表
        intersection_id: 如果指定，只检查该路口的数据
        verbose: 是否打印详细信息
    
    Returns:
        包含约束检查结果的字典
    """
    # 过滤指定路口的数据
    if intersection_id is not None:
        timesteps = [t for t in timesteps if t.get('intersection_id') == intersection_id]
    
    if len(timesteps) == 0:
        return {
            'boundary_violations': [],
            'trend_violations': [],
            'summary': {'total_timesteps': 0, 'error': '没有找到有效的时间步数据'}
        }
    
    boundary_violations = []
    trend_violations = []
    
    # ========== 1. 边界条件检查 ==========
    for timestep_idx, timestep in enumerate(timesteps):
        phases = timestep.get('phases', [])
        if not phases:
            continue
        
        # 提取所有相位的Q值和logit值
        q_values = []
        logits = []
        selected_phase_id = None
        selected_q = None
        
        for phase in phases:
            q_val = phase.get('Q_value')
            logit = phase.get('logit')
            
            if q_val is not None:
                q_values.append((phase['phase_id'], q_val))
                if phase.get('selected', False):
                    selected_phase_id = phase['phase_id']
                    selected_q = q_val
            
            if logit is not None:
                logits.append((phase['phase_id'], logit))
        
        # 检查1: Q值是否有效（不是NaN, Inf）
        for phase_id, q_val in q_values:
            if not np.isfinite(q_val):
                violation = {
                    'type': 'invalid_q_value',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'q_value': q_val,
                    'message': f'Q值无效: {q_val} (NaN或Inf)'
                }
                boundary_violations.append(violation)
        
        # 检查2: logit值是否有效
        for phase_id, logit in logits:
            if not np.isfinite(logit):
                violation = {
                    'type': 'invalid_logit',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'logit': logit,
                    'message': f'logit值无效: {logit} (NaN或Inf)'
                }
                boundary_violations.append(violation)
        
        # 检查3: 被选择的相位的Q值应该是最高的（或至少是较高的）
        if selected_phase_id is not None and len(q_values) > 1:
            max_q_phase_id, max_q = max(q_values, key=lambda x: x[1])
            if selected_phase_id != max_q_phase_id:
                # 允许一定的误差（比如由于探索策略）
                q_diff = max_q - selected_q
                if q_diff > 0.1:  # 如果差距较大，记录违反
                    violation = {
                        'type': 'selected_phase_not_max_q',
                        'intersection_id': timestep.get('intersection_id'),
                        'timestep': timestep.get('timestep'),
                        'selected_phase_id': selected_phase_id,
                        'selected_q': selected_q,
                        'max_q_phase_id': max_q_phase_id,
                        'max_q': max_q,
                        'q_diff': q_diff,
                        'message': f'被选择的相位{selected_phase_id}的Q值({selected_q:.4f})不是最高的，最高Q值相位是{max_q_phase_id}({max_q:.4f})'
                    }
                    boundary_violations.append(violation)
        
        # 检查4: Q值是否在合理范围内（可选，根据实际情况调整）
        for phase_id, q_val in q_values:
            if abs(q_val) > 1000:  # 如果Q值过大，可能有问题
                violation = {
                    'type': 'q_value_out_of_range',
                    'intersection_id': timestep.get('intersection_id'),
                    'timestep': timestep.get('timestep'),
                    'phase_id': phase_id,
                    'q_value': q_val,
                    'message': f'Q值超出合理范围: {q_val}'
                }
                boundary_violations.append(violation)
    
    # ========== 2. 变化趋势检查 ==========
    # 比较相邻时间步的交通状态变化和Q值变化
    for timestep_idx in range(len(timesteps) - 1):
        current_timestep = timesteps[timestep_idx]
        next_timestep = timesteps[timestep_idx + 1]
        
        # 确保是同一个路口
        if current_timestep.get('intersection_id') != next_timestep.get('intersection_id'):
            continue
        
        current_phases = {p['phase_id']: p for p in current_timestep.get('phases', [])}
        next_phases = {p['phase_id']: p for p in next_timestep.get('phases', [])}
        
        # 检查每个相位
        for phase_id in current_phases.keys():
            if phase_id not in next_phases:
                continue
            
            current_phase = current_phases[phase_id]
            next_phase = next_phases[phase_id]
            
            current_q = current_phase.get('Q_value')
            next_q = next_phase.get('Q_value')
            
            if current_q is None or next_q is None:
                continue
            
            # 获取相位的控制方向
            current_state = current_phase.get('state', {})
            next_state = next_phase.get('state', {})
            
            controlled_directions = current_state.get('controlled_directions', [])
            if not controlled_directions:
                continue
            
            # 获取特征变化
            current_features = current_state.get('features_by_direction', {})
            next_features = next_state.get('features_by_direction', {})
            
            # 检查每个控制方向的特征变化
            for direction in controlled_directions:
                if direction not in current_features or direction not in next_features:
                    continue
                
                current_dir_features = current_features[direction]
                next_dir_features = next_features[direction]
                
                # 检查车辆数变化
                if 'num_vehicle' in current_dir_features and 'num_vehicle' in next_dir_features:
                    current_num = current_dir_features['num_vehicle']
                    next_num = next_dir_features['num_vehicle']
                    num_change = next_num - current_num
                    q_change = next_q - current_q
                    
                    # 趋势：车辆数增加时，Q值应该增加（或至少不显著减少）
                    if num_change > 0.1 and q_change < -0.1:  # 车辆增加但Q值减少
                        violation = {
                            'type': 'trend_violation_vehicle_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'num_vehicle_change': num_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'方向{direction}的车辆数增加{num_change:.2f}，但相位{phase_id}的Q值减少{q_change:.4f}'
                        }
                        trend_violations.append(violation)
                
                # 检查压力变化
                if 'pressure' in current_dir_features and 'pressure' in next_dir_features:
                    current_pressure = current_dir_features['pressure']
                    next_pressure = next_dir_features['pressure']
                    pressure_change = next_pressure - current_pressure
                    q_change = next_q - current_q
                    
                    # 趋势：压力增加时，Q值应该增加
                    if pressure_change > 0.1 and q_change < -0.1:
                        violation = {
                            'type': 'trend_violation_pressure_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'pressure_change': pressure_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'方向{direction}的压力增加{pressure_change:.2f}，但相位{phase_id}的Q值减少{q_change:.4f}'
                        }
                        trend_violations.append(violation)
                
                # 检查等待时间变化
                if 'waiting_time' in current_dir_features and 'waiting_time' in next_dir_features:
                    current_wait = current_dir_features['waiting_time']
                    next_wait = next_dir_features['waiting_time']
                    wait_change = next_wait - current_wait
                    q_change = next_q - current_q
                    
                    # 趋势：等待时间增加时，Q值应该增加
                    if wait_change > 0.1 and q_change < -0.1:
                        violation = {
                            'type': 'trend_violation_waiting_time_increase',
                            'intersection_id': current_timestep.get('intersection_id'),
                            'timestep_from': current_timestep.get('timestep'),
                            'timestep_to': next_timestep.get('timestep'),
                            'phase_id': phase_id,
                            'direction': direction,
                            'waiting_time_change': wait_change,
                            'q_value_change': q_change,
                            'current_q': current_q,
                            'next_q': next_q,
                            'message': f'方向{direction}的等待时间增加{wait_change:.2f}，但相位{phase_id}的Q值减少{q_change:.4f}'
                        }
                        trend_violations.append(violation)
            
            # 检查不控制该方向的相位的Q值变化
            # 当某个方向的特征值增加时，不控制该方向的相位的Q值可能不变或减少
            all_directions = set()
            for phase in current_phases.values():
                all_directions.update(phase.get('state', {}).get('controlled_directions', []))
            
            for direction in all_directions:
                if direction in controlled_directions:
                    continue  # 跳过已控制的方向
                
                # 检查该方向在其他相位中的特征变化
                for other_phase_id, other_phase in current_phases.items():
                    if other_phase_id == phase_id:
                        continue
                    
                    other_controlled = other_phase.get('state', {}).get('controlled_directions', [])
                    if direction in other_controlled:
                        # 这个相位控制该方向，跳过
                        continue
                    
                    # 检查该方向在全局的特征变化（如果可用）
                    if direction in current_features and direction in next_features:
                        current_dir_features = current_features[direction]
                        next_dir_features = next_features[direction]
                        
                        # 如果该方向的车辆数/压力增加，不控制该方向的相位的Q值应该不变或减少
                        if 'num_vehicle' in current_dir_features and 'num_vehicle' in next_dir_features:
                            num_change = next_dir_features['num_vehicle'] - current_dir_features['num_vehicle']
                            q_change = next_q - current_q
                            
                            # 如果该方向的车辆数增加，但不控制该方向的相位的Q值也增加很多，可能有问题
                            if num_change > 0.1 and q_change > 0.2:  # 车辆增加，Q值也显著增加（可能不合理）
                                violation = {
                                    'type': 'trend_violation_uncontrolled_direction',
                                    'intersection_id': current_timestep.get('intersection_id'),
                                    'timestep_from': current_timestep.get('timestep'),
                                    'timestep_to': next_timestep.get('timestep'),
                                    'phase_id': phase_id,
                                    'direction': direction,
                                    'num_vehicle_change': num_change,
                                    'q_value_change': q_change,
                                    'message': f'方向{direction}的车辆数增加{num_change:.2f}，但相位{phase_id}不控制该方向，Q值却增加{q_change:.4f}（可能不合理）'
                                }
                                trend_violations.append(violation)
    
    # ========== 3. 生成总结 ==========
    summary = {
        'total_timesteps': len(timesteps),
        'total_boundary_violations': len(boundary_violations),
        'total_trend_violations': len(trend_violations),
        'boundary_violation_types': {},
        'trend_violation_types': {}
    }
    
    # 统计边界条件违反类型
    for violation in boundary_violations:
        v_type = violation['type']
        summary['boundary_violation_types'][v_type] = summary['boundary_violation_types'].get(v_type, 0) + 1
    
    # 统计趋势违反类型
    for violation in trend_violations:
        v_type = violation['type']
        summary['trend_violation_types'][v_type] = summary['trend_violation_types'].get(v_type, 0) + 1
    
    # 打印总结
    if verbose:
        print("=" * 80)
        print("约束检查结果")
        print("=" * 80)
        print(f"总时间步数: {summary['total_timesteps']}")
        print(f"边界条件违反数: {summary['total_boundary_violations']}")
        print(f"变化趋势违反数: {summary['total_trend_violations']}")
        
        if summary['boundary_violation_types']:
            print("\n边界条件违反类型:")
            for v_type, count in summary['boundary_violation_types'].items():
                print(f"  - {v_type}: {count}次")
        
        if summary['trend_violation_types']:
            print("\n变化趋势违反类型:")
            for v_type, count in summary['trend_violation_types'].items():
                print(f"  - {v_type}: {count}次")
        
        # 打印前几个违反示例
        if boundary_violations:
            print("\n边界条件违反示例（前5个）:")
            for i, violation in enumerate(boundary_violations[:5]):
                print(f"  {i+1}. {violation['message']}")
        
        if trend_violations:
            print("\n变化趋势违反示例（前5个）:")
            for i, violation in enumerate(trend_violations[:5]):
                print(f"  {i+1}. {violation['message']}")
        
        print("=" * 80)
    
    return {
        'boundary_violations': boundary_violations,
        'trend_violations': trend_violations,
        'summary': summary
    }


def check_generated_function_constraints(
    expression_template: str,
    timesteps: List[Dict[str, Any]],
    feature_names: List[str],
    feature_mapping: Dict[str, Callable],
    intersection_id: Optional[int] = None,
    num_phases: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    检查大模型生成的函数表达式是否满足约束
    
    Args:
        expression_template: 大模型生成的函数表达式模板（如 "Q = a1 * x1 + a2 * log(phase_active + 1) + ..."）
        timesteps: 时间步数据列表（用于验证函数行为）
        feature_names: 特征名称列表（如 ['phase_active', 'x1', 'x2', ...]）
        feature_mapping: 特征映射字典，将特征名映射到提取函数
            {feature_name: extract_function(state, phase) -> float}
        intersection_id: 如果指定，只检查该路口的数据
        num_phases: 相位数量（如果为None，从数据中推断）
        verbose: 是否打印详细信息
    
    Returns:
        包含约束检查结果的字典：
            - boundary_violations: 边界条件违反情况
            - trend_violations: 变化趋势违反情况
            - expression_valid: 表达式是否有效（可编译）
            - summary: 总结信息
    """
    from agent.imitation_based.llm_unified_compiler import UnifiedExpressionCompiler
    
    # 过滤指定路口的数据
    if intersection_id is not None:
        timesteps = [t for t in timesteps if t.get('intersection_id') == intersection_id]
    
    if len(timesteps) == 0:
        return {
            'boundary_violations': [],
            'trend_violations': [],
            'expression_valid': False,
            'summary': {'total_timesteps': 0, 'error': '没有找到有效的时间步数据'}
        }
    
    # 推断相位数量
    if num_phases is None:
        if timesteps and timesteps[0].get('phases'):
            num_phases = len(timesteps[0]['phases'])
        else:
            num_phases = 8  # 默认值
    
    boundary_violations = []
    trend_violations = []
    expression_valid = True
    
    # ========== 1. 检查表达式是否可编译 ==========
    compiler = UnifiedExpressionCompiler()
    
    # 尝试编译表达式（为每个相位）
    compiled_functions = {}
    for phase_idx in range(num_phases):
        try:
            func, coeff_names = compiler.compile_unified_expression(expression_template, phase_idx)
            compiled_functions[phase_idx] = (func, coeff_names)
        except Exception as e:
            expression_valid = False
            violation = {
                'type': 'expression_compile_error',
                'phase_idx': phase_idx,
                'error': str(e),
                'message': f'相位{phase_idx}的表达式编译失败: {str(e)}'
            }
            boundary_violations.append(violation)
            if verbose:
                print(f"警告: 相位{phase_idx}的表达式编译失败: {e}")
    
    if not expression_valid:
        return {
            'boundary_violations': boundary_violations,
            'trend_violations': [],
            'expression_valid': False,
            'summary': {'total_timesteps': len(timesteps), 'error': '表达式编译失败'}
        }
    
    # ========== 2. 边界条件检查 ==========
    # 使用随机系数测试表达式的基本行为
    for timestep_idx, timestep in enumerate(timesteps[:100]):  # 限制检查前100个时间步以提高效率
        phases = timestep.get('phases', [])
        if not phases:
            continue
        
        for phase in phases:
            phase_id = phase.get('phase_id')
            if phase_id not in compiled_functions:
                continue
            
            func, coeff_names = compiled_functions[phase_id]
            state = phase.get('state', {})
            
            # 提取特征值
            try:
                feature_values = []
                for feat_name in feature_names:
                    if feat_name in feature_mapping:
                        feat_value = feature_mapping[feat_name](state, phase)
                        feature_values.append(feat_value)
                    else:
                        # 如果特征映射不存在，尝试从state中提取
                        feature_values.append(0.0)  # 默认值
                
                X = np.array([feature_values])
                
                # 使用随机系数测试（实际使用时应该用拟合的系数）
                num_coeffs = len(coeff_names)
                test_coeffs = np.random.randn(num_coeffs) * 0.1
                
                # 计算Q值
                try:
                    q_value = func(X, *test_coeffs)[0]
                    
                    # 检查Q值是否有效
                    if not np.isfinite(q_value):
                        violation = {
                            'type': 'invalid_q_value',
                            'intersection_id': timestep.get('intersection_id'),
                            'timestep': timestep.get('timestep'),
                            'phase_id': phase_id,
                            'q_value': q_value,
                            'message': f'表达式计算的Q值无效: {q_value} (NaN或Inf)'
                        }
                        boundary_violations.append(violation)
                    
                    # 检查Q值是否在合理范围内
                    if abs(q_value) > 1000:
                        violation = {
                            'type': 'q_value_out_of_range',
                            'intersection_id': timestep.get('intersection_id'),
                            'timestep': timestep.get('timestep'),
                            'phase_id': phase_id,
                            'q_value': q_value,
                            'message': f'表达式计算的Q值超出合理范围: {q_value}'
                        }
                        boundary_violations.append(violation)
                
                except Exception as e:
                    violation = {
                        'type': 'expression_evaluation_error',
                        'intersection_id': timestep.get('intersection_id'),
                        'timestep': timestep.get('timestep'),
                        'phase_id': phase_id,
                        'error': str(e),
                        'message': f'表达式计算失败: {str(e)}'
                    }
                    boundary_violations.append(violation)
            
            except Exception as e:
                if verbose:
                    print(f"警告: 提取特征失败 (timestep={timestep_idx}, phase={phase_id}): {e}")
    
    # ========== 3. 变化趋势检查 ==========
    # 检查表达式对输入特征变化的响应是否符合预期
    for timestep_idx in range(min(len(timesteps) - 1, 100)):  # 限制检查前100个时间步
        current_timestep = timesteps[timestep_idx]
        next_timestep = timesteps[timestep_idx + 1]
        
        if current_timestep.get('intersection_id') != next_timestep.get('intersection_id'):
            continue
        
        current_phases = {p['phase_id']: p for p in current_timestep.get('phases', [])}
        next_phases = {p['phase_id']: p for p in next_timestep.get('phases', [])}
        
        for phase_id in current_phases.keys():
            if phase_id not in next_phases or phase_id not in compiled_functions:
                continue
            
            current_phase = current_phases[phase_id]
            next_phase = next_phases[phase_id]
            
            func, coeff_names = compiled_functions[phase_id]
            num_coeffs = len(coeff_names)
            test_coeffs = np.random.randn(num_coeffs) * 0.1
            
            # 提取当前和下一个状态的特征
            try:
                current_features = []
                next_features = []
                
                for feat_name in feature_names:
                    if feat_name in feature_mapping:
                        current_features.append(feature_mapping[feat_name](current_phase.get('state', {}), current_phase))
                        next_features.append(feature_mapping[feat_name](next_phase.get('state', {}), next_phase))
                    else:
                        current_features.append(0.0)
                        next_features.append(0.0)
                
                current_X = np.array([current_features])
                next_X = np.array([next_features])
                
                current_q = func(current_X, *test_coeffs)[0]
                next_q = func(next_X, *test_coeffs)[0]
                
                # 检查特征变化和Q值变化的关系
                # 根据teacher model关注的特征进行不同的检查
                current_state = current_phase.get('state', {})
                next_state = next_phase.get('state', {})
                
                controlled_directions = current_state.get('controlled_directions', [])
                current_features_by_dir = current_state.get('features_by_direction', {})
                next_features_by_dir = next_state.get('features_by_direction', {})
                
                for direction in controlled_directions:
                    if direction not in current_features_by_dir or direction not in next_features_by_dir:
                        continue
                    
                    current_dir_feat = current_features_by_dir[direction]
                    next_dir_feat = next_features_by_dir[direction]
                    
                    # 检查车辆数变化（适用于FRAP, MaxPressure, PiLight等关注车辆数的模型）
                    if 'num_vehicle' in current_dir_feat and 'num_vehicle' in next_dir_feat:
                        num_change = next_dir_feat['num_vehicle'] - current_dir_feat['num_vehicle']
                        q_change = next_q - current_q
                        
                        if num_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_vehicle_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'num_vehicle_change': num_change,
                                'q_value_change': q_change,
                                'message': f'方向{direction}的车辆数增加{num_change:.2f}，但表达式计算的Q值减少{q_change:.4f}（不符合预期趋势）'
                            }
                            trend_violations.append(violation)
                    
                    # 检查压力变化（适用于MPLight等关注压力的模型）
                    if 'pressure' in current_dir_feat and 'pressure' in next_dir_feat:
                        pressure_change = next_dir_feat['pressure'] - current_dir_feat['pressure']
                        q_change = next_q - current_q
                        
                        if pressure_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_pressure_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'pressure_change': pressure_change,
                                'q_value_change': q_change,
                                'message': f'方向{direction}的压力增加{pressure_change:.2f}，但表达式计算的Q值减少{q_change:.4f}（不符合预期趋势）'
                            }
                            trend_violations.append(violation)
                    
                    # 检查等待车辆数变化（适用于SOTL等关注等待车辆的模型）
                    if 'num_waiting_vehicle' in current_dir_feat and 'num_waiting_vehicle' in next_dir_feat:
                        waiting_change = next_dir_feat['num_waiting_vehicle'] - current_dir_feat['num_waiting_vehicle']
                        q_change = next_q - current_q
                        
                        if waiting_change > 0.1 and q_change < -0.1:
                            violation = {
                                'type': 'trend_violation_waiting_vehicle_increase',
                                'intersection_id': current_timestep.get('intersection_id'),
                                'timestep_from': current_timestep.get('timestep'),
                                'timestep_to': next_timestep.get('timestep'),
                                'phase_id': phase_id,
                                'direction': direction,
                                'waiting_vehicle_change': waiting_change,
                                'q_value_change': q_change,
                                'message': f'方向{direction}的等待车辆数增加{waiting_change:.2f}，但表达式计算的Q值减少{q_change:.4f}（不符合预期趋势）'
                            }
                            trend_violations.append(violation)
            
            except Exception as e:
                if verbose:
                    print(f"警告: 趋势检查失败 (timestep={timestep_idx}, phase={phase_id}): {e}")
    
    # ========== 4. 生成总结 ==========
    summary = {
        'total_timesteps': len(timesteps),
        'expression_valid': expression_valid,
        'total_boundary_violations': len(boundary_violations),
        'total_trend_violations': len(trend_violations),
        'boundary_violation_types': {},
        'trend_violation_types': {}
    }
    
    for violation in boundary_violations:
        v_type = violation['type']
        summary['boundary_violation_types'][v_type] = summary['boundary_violation_types'].get(v_type, 0) + 1
    
    for violation in trend_violations:
        v_type = violation['type']
        summary['trend_violation_types'][v_type] = summary['trend_violation_types'].get(v_type, 0) + 1
    
    if verbose:
        print("=" * 80)
        print("大模型生成函数的约束检查结果")
        print("=" * 80)
        print(f"表达式模板: {expression_template[:100]}...")
        print(f"表达式是否有效: {expression_valid}")
        print(f"总时间步数: {summary['total_timesteps']}")
        print(f"边界条件违反数: {summary['total_boundary_violations']}")
        print(f"变化趋势违反数: {summary['total_trend_violations']}")
        
        if summary['boundary_violation_types']:
            print("\n边界条件违反类型:")
            for v_type, count in summary['boundary_violation_types'].items():
                print(f"  - {v_type}: {count}次")
        
        if summary['trend_violation_types']:
            print("\n变化趋势违反类型:")
            for v_type, count in summary['trend_violation_types'].items():
                print(f"  - {v_type}: {count}次")
        
        if boundary_violations:
            print("\n边界条件违反示例（前5个）:")
            for i, violation in enumerate(boundary_violations[:5]):
                print(f"  {i+1}. {violation['message']}")
        
        if trend_violations:
            print("\n变化趋势违反示例（前5个）:")
            for i, violation in enumerate(trend_violations[:5]):
                print(f"  {i+1}. {violation['message']}")
        
        print("=" * 80)
    
    return {
        'boundary_violations': boundary_violations,
        'trend_violations': trend_violations,
        'expression_valid': expression_valid,
        'summary': summary
    }


def build_feature_mapping_for_teacher_model(
    teacher_model: str,
    observation_feature_list: List[str]
) -> Dict[str, Callable]:
    """
    根据teacher model类型构建特征映射函数
    
    Args:
        teacher_model: Teacher模型名称（'MPLight', 'FRAP', 'MaxPressure', 'SOTL', 'PiLight'等）
        observation_feature_list: 该模型使用的观察特征列表
    
    Returns:
        特征映射字典
    """
    feature_mapping = {}
    
    for feat_name in observation_feature_list:
        if feat_name == 'inter_2_current_phase':
            def extract_current_phase(state, phase, feat_name=feat_name):
                return float(state.get('current_phase', 0))
            feature_mapping[feat_name] = extract_current_phase
        
        elif feat_name == 'lanelink_2_pressure':
            # MPLight使用：lane-link压力
            def extract_lanelink_pressure(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_pressure = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        pressure = features_by_dir[direction].get('pressure', 0.0)
                        total_pressure += pressure
                        count += 1
                return total_pressure / max(count, 1)
            feature_mapping[feat_name] = extract_lanelink_pressure
        
        elif feat_name == 'lanelink_2_num_vehicle':
            # FRAP使用：lane-link车辆数
            def extract_lanelink_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_lanelink_num_vehicle
        
        elif feat_name == 'lane_2_num_vehicle':
            # MaxPressure使用：lane车辆数
            def extract_lane_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_lane_num_vehicle
        
        elif feat_name == 'inlane_2_num_waiting_vehicle':
            # SOTL使用：进入车道等待车辆数
            def extract_inlane_waiting_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_waiting = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        waiting = features_by_dir[direction].get('num_waiting_vehicle', 0.0)
                        total_waiting += waiting
                        count += 1
                return total_waiting / max(count, 1)
            feature_mapping[feat_name] = extract_inlane_waiting_vehicle
        
        elif feat_name == 'inlane_2_num_vehicle':
            # PiLight使用：进入车道车辆数
            def extract_inlane_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_inlane_num_vehicle
        
        elif feat_name == 'outlane_2_num_vehicle':
            # PiLight使用：离开车道车辆数
            def extract_outlane_num_vehicle(state, phase, feat_name=feat_name):
                # 注意：outlane特征可能不在features_by_direction中
                # 这里使用other_features或其他方式提取
                other_features = state.get('other_features', {})
                if feat_name in other_features:
                    if isinstance(other_features[feat_name], dict):
                        return other_features[feat_name].get('mean', 0.0)
                    return float(other_features[feat_name])
                return 0.0
            feature_mapping[feat_name] = extract_outlane_num_vehicle
        
        else:
            # 默认：尝试从other_features或其他地方提取
            def extract_default(state, phase, feat_name=feat_name):
                other_features = state.get('other_features', {})
                if feat_name in other_features:
                    if isinstance(other_features[feat_name], dict):
                        return other_features[feat_name].get('mean', 0.0)
                    return float(other_features[feat_name])
                return 0.0
            feature_mapping[feat_name] = extract_default
    
    return feature_mapping


def build_default_feature_mapping(feature_names: List[str]) -> Dict[str, Callable]:
    """
    构建默认的特征映射函数
    
    Args:
        feature_names: 特征名称列表（如 ['phase_active', 'x1', 'x2', ...]）
    
    Returns:
        特征映射字典
    """
    feature_mapping = {}
    
    for feat_name in feature_names:
        # 根据特征名创建提取函数
        if feat_name in ('phase_active', 'x0') or 'current_phase' in feat_name.lower():
            # 当前相位特征
            def extract_current_phase(state, phase, feat_name=feat_name):
                return float(state.get('current_phase', 0))
            feature_mapping[feat_name] = extract_current_phase
        
        elif 'pressure' in feat_name.lower() or 'x1' in feat_name:
            # 压力特征：从features_by_direction中提取
            def extract_pressure(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_pressure = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        pressure = features_by_dir[direction].get('pressure', 0.0)
                        total_pressure += pressure
                        count += 1
                return total_pressure / max(count, 1)
            feature_mapping[feat_name] = extract_pressure
        
        elif 'vehicle' in feat_name.lower() or 'num_vehicle' in feat_name.lower():
            # 车辆数特征
            def extract_num_vehicle(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_vehicles = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        num_veh = features_by_dir[direction].get('num_vehicle', 0.0)
                        total_vehicles += num_veh
                        count += 1
                return total_vehicles / max(count, 1)
            feature_mapping[feat_name] = extract_num_vehicle
        
        elif 'waiting' in feat_name.lower() or 'wait_time' in feat_name.lower():
            # 等待时间特征
            def extract_waiting_time(state, phase, feat_name=feat_name):
                features_by_dir = state.get('features_by_direction', {})
                controlled_dirs = state.get('controlled_directions', [])
                total_wait = 0.0
                count = 0
                for direction in controlled_dirs:
                    if direction in features_by_dir:
                        wait_time = features_by_dir[direction].get('waiting_time', 0.0)
                        total_wait += wait_time
                        count += 1
                return total_wait / max(count, 1)
            feature_mapping[feat_name] = extract_waiting_time
        
        else:
            # 默认：返回0
            def extract_default(state, phase, feat_name=feat_name):
                return 0.0
            feature_mapping[feat_name] = extract_default
    
    return feature_mapping


def check_generated_function_for_teacher_model(
    expression_template: str,
    json_file_path: str,
    teacher_model: str,
    observation_feature_list: List[str],
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    针对特定teacher model检查生成的函数表达式
    
    Args:
        expression_template: 大模型生成的函数表达式模板
        json_file_path: JSON文件路径
        teacher_model: Teacher模型名称（'MPLight', 'FRAP', 'MaxPressure', 'SOTL', 'PiLight'等）
        observation_feature_list: 该模型使用的观察特征列表
        intersection_id: 如果指定，只检查该路口的数据
        verbose: 是否打印详细信息
    
    Returns:
        约束检查结果
    """
    # 根据teacher model构建特征映射
    feature_mapping = build_feature_mapping_for_teacher_model(teacher_model, observation_feature_list)
    
    # 将特征名映射为变量名（phase_active, x1, x2, ...）
    from agent.imitation_based.feature_descriptions import get_feature_names_mapping
    feature_name_to_var = get_feature_names_mapping(observation_feature_list)
    feature_names = [feature_name_to_var[feat] for feat in observation_feature_list]
    
    # 创建变量名到特征提取函数的映射
    var_feature_mapping = {}
    for feat_name, var_name in feature_name_to_var.items():
        if feat_name in feature_mapping:
            var_feature_mapping[var_name] = feature_mapping[feat_name]
    
    return check_generated_function_from_json(
        expression_template, json_file_path, feature_names, var_feature_mapping, intersection_id, verbose
    )


def check_generated_function_from_json(
    expression_template: str,
    json_file_path: str,
    feature_names: List[str],
    feature_mapping: Optional[Dict[str, Callable]] = None,
    intersection_id: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    从JSON文件加载数据并检查生成的函数表达式
    
    Args:
        expression_template: 大模型生成的函数表达式模板（如 "Q = a1 * x1 + a2 * log(phase_active + 1)"）
        json_file_path: JSON文件路径
        feature_names: 特征名称列表（如 ['phase_active', 'x1', 'x2', ...]）
        feature_mapping: 特征映射字典，如果为None则使用默认映射
        intersection_id: 如果指定，只检查该路口的数据
        verbose: 是否打印详细信息
    
    Returns:
        约束检查结果
    """
    timesteps = _load_timesteps_from_json(json_file_path)
    
    # 如果没有提供特征映射，使用默认映射
    if feature_mapping is None:
        feature_mapping = build_default_feature_mapping(feature_names)
        if verbose:
            print(f"使用默认特征映射（共{len(feature_mapping)}个特征）")
    
    return check_generated_function_constraints(
        expression_template, timesteps, feature_names, feature_mapping, intersection_id, None, verbose
    )


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python constraint_checker.py <json_file_path> [intersection_id]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    intersection_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    results = check_constraints_from_json_file(json_file, intersection_id, verbose=True)
    
    # 保存结果
    output_file = json_file.replace('.json', '_constraint_check.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n检查结果已保存到: {output_file}")

