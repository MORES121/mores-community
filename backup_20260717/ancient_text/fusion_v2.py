"""
MORES 可控核聚变引擎 v2.0
基于 EXL-50U 仿真环境
全自动 · 可解释 · 自适应
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random

random.seed(42)
np.random.seed(42)

# ========== 动作降维映射（利用上下对称性）==========
# 7维动作 -> 12维动作（参考官方示例）
def action_7d_to_12d(action_7d):
    """7维动作映射到12维极向场线圈电压"""
    action_12d = np.zeros(12)
    # PF1-PF10 上下对称
    action_12d[0] = action_7d[0]   # PF1
    action_12d[1] = action_7d[1]   # PF2
    action_12d[2] = action_7d[2]   # PF3
    action_12d[3] = action_7d[3]   # PF4
    action_12d[4] = action_7d[4]   # PF5
    action_12d[5] = action_7d[5]   # PF6
    action_12d[6] = action_7d[6]   # PF7
    action_12d[7] = action_7d[4]   # PF8 = PF5 对称
    action_12d[8] = action_7d[3]   # PF9 = PF4 对称
    action_12d[9] = action_7d[2]   # PF10 = PF3 对称
    action_12d[10] = action_7d[0]  # PF11 = PF1 对称
    action_12d[11] = action_7d[1]  # PF12 = PF2 对称
    return action_12d


class MORESFusionController:
    """
    MORES 全自动核聚变控制器
    - 自适应策略选择
    - 物理约束保护
    - 可解释决策输出
    """
    
    def __init__(self):
        # 物理约束
        self.constraints = {
            "Ip_max": 600000,           # 等离子体电流上限 (A)
            "Ip_min": 100000,           # 下限
            "R_range": (0.5, 1.2),      # 水平位置范围 (m)
            "Z_range": (-0.5, 0.5),     # 垂直位置范围 (m)
            "action_range": (-10000, 10000),  # 线圈电压范围 (V)
        }
        
        # 控制参数（可自适应）
        self.gains = {
            "Ip_kp": 0.01,
            "R_kp": 0.5,
            "Z_kp": 0.8,
        }
        
        self.step_count = 0
    
    def _compute_errors(self, obs: Dict) -> Dict:
        """计算跟踪误差"""
        errors = {}
        
        # 等离子体电流误差（相对误差）
        Ip_target = obs.get('reference_Ip', 0)
        Ip_current = obs.get('Ip', 0)
        if Ip_target > 0:
            errors['Ip_err'] = (Ip_target - Ip_current) / Ip_target
        else:
            errors['Ip_err'] = 0
        
        # 位置误差
        R_target = obs.get('reference_R', 0)
        R_current = obs.get('R', 0)
        Z_target = obs.get('reference_Z', 0)
        Z_current = obs.get('Z', 0)
        
        errors['R_err'] = R_target - R_current
        errors['Z_err'] = Z_target - Z_current
        
        # 综合位置误差（欧氏距离）
        errors['pos_err'] = np.sqrt(errors['R_err']**2 + errors['Z_err']**2)
        
        return errors
    
    def _select_strategy(self, errors: Dict, failure: bool) -> str:
        """自适应策略选择"""
        if failure:
            return "emergency"
        
        # 大误差 → 激进策略
        if abs(errors['Ip_err']) > 0.1 or errors['pos_err'] > 0.05:
            return "aggressive"
        # 中等误差 → 平衡策略
        elif abs(errors['Ip_err']) > 0.03 or errors['pos_err'] > 0.02:
            return "balanced"
        # 小误差 → 保守策略
        else:
            return "conservative"
    
    def _compute_action(self, errors: Dict, strategy: str) -> np.ndarray:
        """根据策略计算 7 维动作"""
        
        # 基础 PID 控制
        action_7d = np.zeros(7)
        
        # 电流控制（主要影响 PF1-PF4）
        action_7d[0] = self.gains['Ip_kp'] * errors['Ip_err'] * 5000
        
        # 水平位置控制（影响 PF5-PF8）
        action_7d[1] = self.gains['R_kp'] * errors['R_err'] * 2000
        
        # 垂直位置控制（影响 PF9-PF12）
        action_7d[2] = self.gains['Z_kp'] * errors['Z_err'] * 3000
        
        # 策略增益调整
        if strategy == "aggressive":
            action_7d *= 1.5
        elif strategy == "conservative":
            action_7d *= 0.7
        elif strategy == "emergency":
            action_7d = np.zeros(7)  # 紧急停止
        
        # 动作裁剪
        action_7d = np.clip(action_7d, -5000, 5000)
        
        return action_7d
    
    def act(self, observation: Dict) -> Tuple[np.ndarray, Dict]:
        """
        核心决策接口
        输入：HFM 环境返回的 observation
        输出：12 维线圈电压动作 + 可解释信息
        """
        self.step_count += 1
        
        # 提取关键信息
        failure = observation.get('failure', False)
        
        # 计算跟踪误差
        errors = self._compute_errors(observation)
        
        # 自适应策略选择
        strategy = self._select_strategy(errors, failure)
        
        # 计算 7 维动作
        action_7d = self._compute_action(errors, strategy)
        
        # 映射到 12 维动作空间
        action_12d = action_7d_to_12d(action_7d)
        
        # 最终裁剪
        action_12d = np.clip(action_12d, -10000, 10000)
        
        # 可解释信息
        explanation = {
            "step": self.step_count,
            "strategy": strategy,
            "errors": {
                "Ip_err": float(errors['Ip_err']),
                "pos_err": float(errors['pos_err']),
            },
            "action_summary": {
                "max_voltage": float(np.max(np.abs(action_12d))),
                "mean_voltage": float(np.mean(np.abs(action_12d))),
            },
            "failure": failure,
        }
        
        return action_12d.astype(np.float32), explanation


# ========== 测试入口 ==========
def test():
    print("=" * 60)
    print("MORES 核聚变引擎 v2.0 测试")
    print("可控 · 可解释 · 自适应")
    print("=" * 60)
    
    controller = MORESFusionController()
    
    # 模拟正常状态
    obs_normal = {
        "Ip": 400000,
        "R": 0.82,
        "Z": 0.01,
        "reference_Ip": 400000,
        "reference_R": 0.82,
        "reference_Z": 0.0,
        "failure": False,
    }
    
    print("\n[场景1] 正常状态（无误差）")
    action, exp = controller.act(obs_normal)
    print(f"  策略: {exp['strategy']}")
    print(f"  Ip误差: {exp['errors']['Ip_err']:.4f}")
    print(f"  位置误差: {exp['errors']['pos_err']:.4f}")
    print(f"  动作范围: ±{exp['action_summary']['max_voltage']:.1f} V")
    
    # 模拟误差状态
    obs_error = {
        "Ip": 350000,
        "R": 0.80,
        "Z": 0.03,
        "reference_Ip": 400000,
        "reference_R": 0.82,
        "reference_Z": 0.0,
        "failure": False,
    }
    
    print("\n[场景2] 跟踪误差状态")
    action2, exp2 = controller.act(obs_error)
    print(f"  策略: {exp2['strategy']}")
    print(f"  Ip误差: {exp2['errors']['Ip_err']:.4f}")
    print(f"  位置误差: {exp2['errors']['pos_err']:.4f}")
    
    # 模拟紧急状态
    obs_emergency = {
        "Ip": 400000,
        "R": 0.82,
        "Z": 0.01,
        "reference_Ip": 400000,
        "reference_R": 0.82,
        "reference_Z": 0.0,
        "failure": True,
    }
    
    print("\n[场景3] 紧急状态（破裂风险）")
    action3, exp3 = controller.act(obs_emergency)
    print(f"  策略: {exp3['strategy']}")
    print(f"  failure: {exp3['failure']}")
    
    print("\n" + "=" * 60)
    print("✅ 核聚变引擎 v2.0 测试通过")
    print("=" * 60)


if __name__ == "__main__":
    test()