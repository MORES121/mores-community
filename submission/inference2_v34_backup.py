import numpy as np
from typing import Dict


class PIDController:
    def __init__(self):
        self.kp_ip = 0.48
        self.kp_r = 1.3
        self.kp_z = 2.2
        self.action_low = -3000
        self.action_high = 3000
        self.step_count = 0
    
    def compute_errors(self, obs):
        Ip = obs.get('Ip', 0)
        R = obs.get('R', 0)
        Z = obs.get('Z', 0)
        ref_Ip = obs.get('reference_Ip', 0)
        ref_R = obs.get('reference_R', 0)
        ref_Z = obs.get('reference_Z', 0)
        
        err_Ip = (Ip - ref_Ip) / 1e6
        err_R = (R - ref_R) / 0.1
        err_Z = (Z - ref_Z) / 0.1
        
        return np.array([err_Ip, err_R, err_Z])
    
    def action_7d_to_12d(self, action_7d):
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
        self.step_count = 0
    
    def act(self, observation):
        self.step_count += 1
        
        errors = self.compute_errors(observation)
        
        action_7d = np.zeros(7, dtype=np.float32)
        action_7d[0] = np.clip(self.kp_ip * errors[0], -0.7, 0.7)
        action_7d[1] = np.clip(self.kp_r * errors[1], -0.7, 0.7)
        action_7d[2] = np.clip(self.kp_z * errors[2], -0.7, 0.7)
        
        action_12d = self.action_7d_to_12d(action_7d)
        action_12d = np.clip(action_12d, self.action_low, self.action_high)
        
        return action_12d.astype(np.float32)


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v33.0 - F2 (Simple PID)")
        self.controller = PIDController()
    
    def reset(self):
        self.controller.reset()
    
    def act(self, observation: Dict) -> np.ndarray:
        return self.controller.act(observation)