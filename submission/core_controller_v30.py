"""
MORES Core Controller v30.0
增强版：LCFS + X 点 + 打击点控制
专利保护 - 核心算法
"""

import numpy as np
from typing import Dict
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MORESCoreController:
    def __init__(self, task='F1'):
        self.task = task
        self.action_low = -3000
        self.action_high = 3000
        
        # PID 增益（基础）
        self.kp_ip = 0.48
        self.kp_r = 1.3
        self.kp_z = 2.2
        
        self.ki_ip = 0.10
        self.ki_r = 0.18
        self.ki_z = 0.30
        
        # LCFS 控制增益（新增）
        self.kp_lcfs = 0.8
        
        # X 点控制增益
        self.kp_x = 1.5
        self.kp_zx = 1.5
        self.kp_x2 = 1.2
        
        # 打击点控制增益
        self.kp_strike = 0.5
        
        self.integral_ip = 0.0
        self.integral_r = 0.0
        self.integral_z = 0.0
        self.integral_limit = 0.4
        
        self.alpha = 0.8
        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.coil_violation = False
    
    def compute_errors(self, obs: Dict) -> np.ndarray:
        Ip = obs.get('Ip', 0)
        R = obs.get('R', 0)
        Z = obs.get('Z', 0)
        ref_Ip = obs.get('reference_Ip', 0)
        ref_R = obs.get('reference_R', 0)
        ref_Z = obs.get('reference_Z', 0)
        
        scale_Ip = 1e6
        scale_R = 0.1
        scale_Z = 0.1
        
        err_Ip = (Ip - ref_Ip) / scale_Ip if scale_Ip != 0 else 0
        err_R = (R - ref_R) / scale_R if scale_R != 0 else 0
        err_Z = (Z - ref_Z) / scale_Z if scale_Z != 0 else 0
        
        return np.array([err_Ip, err_R, err_Z], dtype=np.float32)
    
    def compute_lcfs_error(self, obs: Dict) -> float:
        """计算 LCFS 边界距离误差"""
        lcfs = obs.get('lcfs_points', np.zeros((32, 2)))
        ref_lcfs = obs.get('reference_lcfs_points', np.zeros((32, 2)))
        
        if lcfs.shape != ref_lcfs.shape:
            return 0.0
        
        # 计算平均欧氏距离
        diff = lcfs - ref_lcfs
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return np.mean(distances) / 10.0  # 归一化到 cm 量级
    
    def compute_xpoint_errors(self, obs: Dict) -> float:
        """计算主 X 点位置误差"""
        R_X = obs.get('R_X', 0)
        Z_X = obs.get('Z_X', 0)
        ref_R_X = obs.get('reference_R_X', 0)
        ref_Z_X = obs.get('reference_Z_X', 0)
        return np.sqrt((R_X - ref_R_X)**2 + (Z_X - ref_Z_X)**2)
    
    def compute_strike_errors(self, obs: Dict) -> float:
        """计算打击点位置误差"""
        strikes = obs.get('strike_points', np.zeros((8, 2)))
        ref_strikes = obs.get('reference_strike_points', np.zeros((8, 2)))
        
        if strikes.shape != ref_strikes.shape:
            return 0.0
        
        diff = strikes - ref_strikes
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return np.mean(distances) / 10.0
    
    def get_configuration(self, obs: Dict) -> str:
        lX = obs.get('lX', 0)
        if lX == 0:
            return "limiter"
        else:
            return "divertor"
    
    def check_current_limits(self, obs: Dict) -> bool:
        I_CS = obs.get('I_CS', 0)
        I_PF = obs.get('I_PF', np.zeros(10))
        I_VS = obs.get('I_VS', 0)
        
        if I_CS > 45000:
            return True
        if np.any(I_PF > 14000):
            return True
        if I_VS > 4000:
            return True
        return False
    
    def check_ip_deviation(self, obs: Dict) -> bool:
        Ip = obs.get('Ip', 0)
        ref_Ip = obs.get('reference_Ip', 0)
        return abs(Ip - ref_Ip) > 50000
    
    def action_7d_to_12d(self, action_7d: np.ndarray) -> np.ndarray:
        action_phys_7d = self.action_low + (action_7d + 1) / 2 * (self.action_high - self.action_low)
        
        action_12d = np.zeros(12, dtype=np.float32)
        action_12d[0] = action_phys_7d[0]
        action_12d[1] = action_phys_7d[1]
        action_12d[2] = action_phys_7d[2]
        action_12d[3] = action_phys_7d[3]
        action_12d[4] = action_phys_7d[4]
        action_12d[5] = action_phys_7d[5]
        action_12d[6] = action_phys_7d[6]
        action_12d[7] = action_phys_7d[5]
        action_12d[8] = action_phys_7d[4]
        action_12d[9] = action_phys_7d[3]
        action_12d[10] = action_phys_7d[2]
        action_12d[11] = action_phys_7d[1]
        
        return action_12d
    
    def act_f1(self, observation: Dict) -> np.ndarray:
        """F1 任务：偏滤器 → 限制器（含 LCFS 控制）"""
        self.step_count += 1
        
        if self.check_current_limits(observation):
            self.coil_violation = True
        
        if self.coil_violation:
            return np.zeros(12, dtype=np.float32)
        
        if self.check_ip_deviation(observation):
            return np.ones(12, dtype=np.float32) * 100
        
        errors = self.compute_errors(observation)
        lcfs_err = self.compute_lcfs_error(observation)
        
        self.integral_ip += errors[0]
        self.integral_r += errors[1]
        self.integral_z += errors[2]
        self.integral_ip = np.clip(self.integral_ip, -self.integral_limit, self.integral_limit)
        self.integral_r = np.clip(self.integral_r, -self.integral_limit, self.integral_limit)
        self.integral_z = np.clip(self.integral_z, -self.integral_limit, self.integral_limit)
        
        action_7d = np.zeros(7, dtype=np.float32)
        action_7d[0] = self.kp_ip * errors[0] + self.ki_ip * self.integral_ip
        action_7d[0] = np.clip(action_7d[0], -0.7, 0.7)
        action_7d[1] = self.kp_r * errors[1] + self.ki_r * self.integral_r + self.kp_lcfs * np.clip(lcfs_err, -0.3, 0.3)
        action_7d[1] = np.clip(action_7d[1], -0.7, 0.7)
        action_7d[2] = self.kp_z * errors[2] + self.ki_z * self.integral_z
        action_7d[2] = np.clip(action_7d[2], -0.7, 0.7)
        
        action_12d = self.action_7d_to_12d(action_7d)
        action_12d = self.alpha * action_12d + (1 - self.alpha) * self.last_action
        self.last_action = action_12d.copy()
        action_12d = np.clip(action_12d, self.action_low, self.action_high)
        
        return action_12d.astype(np.float32)
    
    def act_f2(self, observation: Dict) -> np.ndarray:
        """F2 任务：→ XPT（含 LCFS + X 点 + 打击点控制）"""
        self.step_count += 1
        
        if self.check_current_limits(observation):
            self.coil_violation = True
        
        if self.coil_violation:
            return np.zeros(12, dtype=np.float32)
        
        if self.check_ip_deviation(observation):
            return np.ones(12, dtype=np.float32) * 100
        
        config = self.get_configuration(observation)
        errors = self.compute_errors(observation)
        lcfs_err = self.compute_lcfs_error(observation)
        x_err = self.compute_xpoint_errors(observation)
        strike_err = self.compute_strike_errors(observation)
        
        self.integral_ip += errors[0]
        self.integral_r += errors[1]
        self.integral_z += errors[2]
        self.integral_ip = np.clip(self.integral_ip, -self.integral_limit, self.integral_limit)
        self.integral_r = np.clip(self.integral_r, -self.integral_limit, self.integral_limit)
        self.integral_z = np.clip(self.integral_z, -self.integral_limit, self.integral_limit)
        
        action_7d = np.zeros(7, dtype=np.float32)
        action_7d[0] = self.kp_ip * errors[0] + self.ki_ip * self.integral_ip
        action_7d[0] = np.clip(action_7d[0], -0.7, 0.7)
        
        # 位置控制（融合 LCFS + X 点）
        pos_err = self.kp_r * errors[1] + self.ki_r * self.integral_r
        pos_err += self.kp_lcfs * np.clip(lcfs_err, -0.3, 0.3)
        
        vert_err = self.kp_z * errors[2] + self.ki_z * self.integral_z
        
        if config == "divertor":
            pos_err += self.kp_x * np.clip(x_err, -0.3, 0.3)
            vert_err += self.kp_zx * np.clip(x_err, -0.3, 0.3)
            pos_err += self.kp_strike * np.clip(strike_err, -0.2, 0.2)
        
        action_7d[1] = np.clip(pos_err, -0.7, 0.7)
        action_7d[2] = np.clip(vert_err, -0.7, 0.7)
        
        action_12d = self.action_7d_to_12d(action_7d)
        action_12d = self.alpha * action_12d + (1 - self.alpha) * self.last_action
        self.last_action = action_12d.copy()
        action_12d = np.clip(action_12d, self.action_low, self.action_high)
        
        return action_12d.astype(np.float32)
    
    def reset(self):
        self.integral_ip = 0.0
        self.integral_r = 0.0
        self.integral_z = 0.0
        self.step_count = 0
        self.coil_violation = False
        self.last_action = np.zeros(12, dtype=np.float32)