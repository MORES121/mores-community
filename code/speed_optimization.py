"""
速度策略优化 v2.0
根据距离目标动态调整速度
"""

def get_optimal_speed(distance_to_target: float) -> float:
    """根据距离计算最优速度"""
    if distance_to_target > 2.5:
        return 1.8
    elif distance_to_target > 1.5:
        return 1.5
    elif distance_to_target > 0.8:
        return 1.2
    elif distance_to_target > 0.3:
        return 0.8
    else:
        return 0.4

def get_optimal_steering(distance_to_target: float, angle_error: float) -> float:
    """根据距离和角度误差计算最优转向"""
    if distance_to_target > 2.0:
        return min(max(angle_error * 0.3, -15), 15)
    elif distance_to_target > 1.0:
        return min(max(angle_error * 0.2, -8), 8)
    else:
        return min(max(angle_error * 0.1, -3), 3)
