import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v32.0 - F1 (PID Control)")
        # PID 增益
        self.kp_ip = 0.48
        self.kp_r = 1.3
        self.kp_z = 2.2
        self.ki_ip = 0.10
        self.ki_r = 0.18
        self.ki_z = 0.30
        
        # 积分累积
        self.integral_ip = 0.0
        self.integral_r = 0.0
        self.integral_z = 0.0
        self.integral_limit = 0.4
        
        # 动作限幅
        self.action_low = -3000
        self.action_high = 3000
        
        # 滤波
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
    
    def reset(self):
        self.integral_ip = 0.0
        self.integral_r = 0.0
        self.integral_z = 0.0
        self.step_count = 0
        self.coil_violation = False
        self.last_action = np.zeros(12, dtype=np.float32)
    
    def act(self, observation: Dict) -> np.ndarray:
        self.step_count += 1
        
        # 线圈电流约束
        if self.check_current_limits(observation):
            self.coil_violation = True
        
        if self.coil_violation:
            return np.zeros(12, dtype=np.float32)
        
        # 电流偏差熔断
        if self.check_ip_deviation(observation):
            return np.ones(12, dtype=np.float32) * 100
        
        # PID 控制
        errors = self.compute_errors(observation)
        
        self.integral_ip += errors[0]
        self.integral_r += errors[1]
        self.integral_z += errors[2]
        self.integral_ip = np.clip(self.integral_ip, -self.integral_limit, self.integral_limit)
        self.integral_r = np.clip(self.integral_r, -self.integral_limit, self.integral_limit)
        self.integral_z = np.clip(self.integral_z, -self.integral_limit, self.integral_limit)
        
        action_7d = np.zeros(7, dtype=np.float32)
        action_7d[0] = self.kp_ip * errors[0] + self.ki_ip * self.integral_ip
        action_7d[0] = np.clip(action_7d[0], -0.7, 0.7)
        action_7d[1] = self.kp_r * errors[1] + self.ki_r * self.integral_r
        action_7d[1] = np.clip(action_7d[1], -0.7, 0.7)
        action_7d[2] = self.kp_z * errors[2] + self.ki_z * self.integral_z
        action_7d[2] = np.clip(action_7d[2], -0.7, 0.7)
        
        action_12d = self.action_7d_to_12d(action_7d)
        action_12d = self.alpha * action_12d + (1 - self.alpha) * self.last_action
        self.last_action = action_12d.copy()
        action_12d = np.clip(action_12d, self.action_low, self.action_high)
        
        return action_12d.astype(np.float32)


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v32.0 - F1 (PID Control)")
        self.controller = Policy()
        self.controller.reset = lambda: None
    
    def reset(self):
        self.controller.reset()
    
    def act(self, observation: Dict) -> np.ndarray:
        return self.controller.act(observation)