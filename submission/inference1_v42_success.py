import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v42 - Inline Quantization (Action Range ±1500V)")
        
        # PID 增益（进一步提高）
        self.kp_ip = 0.52
        self.kp_r = 1.45
        self.kp_z = 2.50
        
        # 动作限幅（进一步降低到 ±1500V）
        self.action_low = -1500
        self.action_high = 1500
        
        # 误差量化边界（7 级）
        errors_range = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]
        
        def action_7d_to_12d(a7):
            a_phys = self.action_low + (a7 + 1) / 2 * (self.action_high - self.action_low)
            a12 = np.zeros(12, dtype=np.float32)
            a12[0] = a_phys[0]
            a12[1] = a_phys[1]
            a12[2] = a_phys[2]
            a12[3] = a_phys[3]
            a12[4] = a_phys[4]
            a12[5] = a_phys[5]
            a12[6] = a_phys[6]
            a12[7] = a_phys[5]
            a12[8] = a_phys[4]
            a12[9] = a_phys[3]
            a12[10] = a_phys[2]
            a12[11] = a_phys[1]
            return a12
        
        # 预计算动作表
        self.action_table = {}
        for e_ip in errors_range:
            for e_r in errors_range:
                for e_z in errors_range:
                    a7 = np.array([
                        np.clip(self.kp_ip * e_ip, -0.7, 0.7),
                        np.clip(self.kp_r * e_r, -0.7, 0.7),
                        np.clip(self.kp_z * e_z, -0.7, 0.7),
                        0, 0, 0, 0
                    ])
                    key = (round(e_ip, 2), round(e_r, 2), round(e_z, 2))
                    self.action_table[key] = action_7d_to_12d(a7)
        
        print(f"[MORES] v42 Precomputed {len(self.action_table)} actions")
        print(f"[MORES] Action range: [{self.action_low}, {self.action_high}]")
        print(f"[MORES] Gains: kp_ip={self.kp_ip}, kp_r={self.kp_r}, kp_z={self.kp_z}")
        self.default_action = np.zeros(12, dtype=np.float32)
    
    def reset(self):
        pass
    
    def act(self, observation: Dict) -> np.ndarray:
        # 直接计算误差
        Ip = observation.get('Ip', 0)
        R = observation.get('R', 0)
        Z = observation.get('Z', 0)
        ref_Ip = observation.get('reference_Ip', 0)
        ref_R = observation.get('reference_R', 0)
        ref_Z = observation.get('reference_Z', 0)
        
        err_Ip = (Ip - ref_Ip) / 1000000.0
        err_R = (R - ref_R) / 0.1
        err_Z = (Z - ref_Z) / 0.1
        
        # 内联量化（无函数调用）
        if err_Ip < -0.4:
            q_ip = -0.5
        elif err_Ip < -0.2:
            q_ip = -0.3
        elif err_Ip < -0.05:
            q_ip = -0.1
        elif err_Ip < 0.05:
            q_ip = 0.0
        elif err_Ip < 0.2:
            q_ip = 0.1
        elif err_Ip < 0.4:
            q_ip = 0.3
        else:
            q_ip = 0.5
        
        if err_R < -0.4:
            q_r = -0.5
        elif err_R < -0.2:
            q_r = -0.3
        elif err_R < -0.05:
            q_r = -0.1
        elif err_R < 0.05:
            q_r = 0.0
        elif err_R < 0.2:
            q_r = 0.1
        elif err_R < 0.4:
            q_r = 0.3
        else:
            q_r = 0.5
        
        if err_Z < -0.4:
            q_z = -0.5
        elif err_Z < -0.2:
            q_z = -0.3
        elif err_Z < -0.05:
            q_z = -0.1
        elif err_Z < 0.05:
            q_z = 0.0
        elif err_Z < 0.2:
            q_z = 0.1
        elif err_Z < 0.4:
            q_z = 0.3
        else:
            q_z = 0.5
        
        key = (q_ip, q_r, q_z)
        
        return self.action_table.get(key, self.default_action)