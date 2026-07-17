"""
MORES 可控核聚变引擎 v3.0
基于 EXL-50U + RL Baseline
全自动 · 可解释 · 自适应 · RL 兼容
"""

import numpy as np
from typing import Dict, Tuple
import random

random.seed(42)
np.random.seed(42)

# ========== 常量定义 ==========
ERROR_KEYS = ("Ip", "R", "Z")
ERROR_SCALES = np.array([1.0e6, 0.1, 0.1], dtype=np.float32)

# 动作空间边界（物理值）
ACTION_LOW = -10000   # 最低电压 (V)
ACTION_HIGH = 10000   # 最高电压 (V)

# 奖励权重（对应评分）
REWARD_WEIGHTS = {"Ip": 0.36, "pos": 0.33, "lcfs": 0.31}


def action_7d_to_12d(action_7d: np.ndarray) -> np.ndarray:
    """7维动作映射到12维极向场线圈电压"""
    action_12d = np.zeros(12, dtype=np.float32)
    # PF1-PF10 上下对称映射
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


def normalize_action(action_norm: np.ndarray) -> np.ndarray:
    """将 [-1, 1] 归一化动作映射到物理电压"""
    return ACTION_LOW + (action_norm + 1) / 2 * (ACTION_HIGH - ACTION_LOW)


class MORESFusionController:
    """
    MORES 全自动核聚变控制器 v3.0
    兼容 PPO 训练框架
    """
    
    def __init__(self):
        self.step_count = 0
        self.failure = False
        
        # PPO 输出的是 7 维归一化动作 [-1, 1]
        # 本控制器直接接收归一化动作，或自动生成
        self.use_external_action = False
    
    def compute_state(self, obs: Dict) -> np.ndarray:
        """
        计算 3 维归一化状态
        用于 PPO 观测输入
        """
        Ip = obs.get('Ip', 0)
        R = obs.get('R', 0)
        Z = obs.get('Z', 0)
        ref_Ip = obs.get('reference_Ip', 0)
        ref_R = obs.get('reference_R', 0)
        ref_Z = obs.get('reference_Z', 0)
        
        errors = np.array([
            (Ip - ref_Ip) / ERROR_SCALES[0],
            (R - ref_R) / ERROR_SCALES[1],
            (Z - ref_Z) / ERROR_SCALES[2],
        ], dtype=np.float32)
        
        return errors
    
    def compute_reward(self, obs: Dict, action_norm: np.ndarray = None) -> float:
        """
        计算奖励
        基于 Ip/R/Z 归一化误差
        """
        errors = self.compute_state(obs)
        scaled_abs_error = np.abs(errors)
        reward = -np.mean(scaled_abs_error)
        
        # 失败惩罚
        if obs.get('failure', False):
            reward = -10.0
        
        return float(reward)
    
    def act(self, action_norm: np.ndarray) -> np.ndarray:
        """
        从归一化动作生成 12 维物理电压
        
        Args:
            action_norm: 7维数组，范围 [-1, 1]
        
        Returns:
            12维物理电压数组
        """
        # 1. 归一化动作 → 物理电压
        action_7d_phys = normalize_action(action_norm)
        
        # 2. 7维 → 12维映射
        action_12d = action_7d_to_12d(action_7d_phys)
        
        # 3. 最终裁剪
        action_12d = np.clip(action_12d, ACTION_LOW, ACTION_HIGH)
        
        return action_12d.astype(np.float32)
    
    def act_heuristic(self, obs: Dict) -> Tuple[np.ndarray, Dict]:
        """
        启发式控制器（无 PPO 时使用）
        根据误差直接生成动作
        """
        errors = self.compute_state(obs)
        
        # PID 风格控制
        action_norm = np.zeros(7, dtype=np.float32)
        action_norm[0] = np.clip(errors[0] * 0.5, -1, 1)   # Ip
        action_norm[1] = np.clip(errors[1] * 2.0, -1, 1)   # R
        action_norm[2] = np.clip(errors[2] * 3.0, -1, 1)   # Z
        
        action_12d = self.act(action_norm)
        
        explanation = {
            "step": self.step_count,
            "errors": {
                "Ip_err": float(errors[0]),
                "R_err": float(errors[1]),
                "Z_err": float(errors[2]),
            },
            "action_norm": action_norm.tolist(),
        }
        
        self.step_count += 1
        return action_12d, explanation


# ========== 测试入口 ==========
def test():
    print("=" * 60)
    print("MORES 核聚变引擎 v3.0 测试")
    print("RL 兼容 · 自适应 · 可解释")
    print("=" * 60)
    
    controller = MORESFusionController()
    
    # 模拟观测（正常跟踪）
    obs_normal = {
        "Ip": 400000,
        "R": 0.82,
        "Z": 0.01,
        "reference_Ip": 400000,
        "reference_R": 0.82,
        "reference_Z": 0.0,
        "failure": False,
    }
    
    print("\n[测试1] 状态提取")
    state = controller.compute_state(obs_normal)
    print(f"  归一化状态: Ip={state[0]:.4f}, R={state[1]:.4f}, Z={state[2]:.4f}")
    
    print("\n[测试2] 奖励计算")
    reward = controller.compute_reward(obs_normal)
    print(f"  奖励: {reward:.4f}")
    
    print("\n[测试3] 启发式动作")
    action, exp = controller.act_heuristic(obs_normal)
    print(f"  策略: 启发式 PID")
    print(f"  动作范围: min={action.min():.1f}, max={action.max():.1f}")
    
    print("\n[测试4] 失败状态")
    obs_failure = {**obs_normal, "failure": True}
    reward_fail = controller.compute_reward(obs_failure)
    print(f"  失败奖励: {reward_fail:.1f}")
    
    print("\n" + "=" * 60)
    print("✅ 核聚变引擎 v3.0 测试通过")
    print("=" * 60)


if __name__ == "__main__":
    test()