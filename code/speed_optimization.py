"""
速度策略优化 v2.0
"""

def get_optimal_speed_v2(distance_to_target: float) -> float:
    """根据距离计算最优速度"""
    if distance_to_target > 3.0:
        return 1.2
    elif distance_to_target > 1.5:
        return 1.0
    elif distance_to_target > 0.5:
        return 0.7
    else:
        return 0.3

def get_optimal_steering_v2(distance_to_target: float, angle_error: float) -> float:
    """根据距离和角度误差计算最优转向"""
    if distance_to_target > 2.0:
        return min(max(angle_error * 0.3, -15), 15)
    elif distance_to_target > 1.0:
        return min(max(angle_error * 0.2, -8), 8)
    else:
        return min(max(angle_error * 0.1, -3), 3)
