import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v36 - Precomputed Table for F2")
        
        self.action_table = {}
        errors_range = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]
        
        action_low = -3000
        action_high = 3000
        
        def action_7d_to_12d(a7):
            a_phys = action_low + (a7 + 1) / 2 * (action_high - action_low)
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
        
        for e_ip in errors_range:
            for e_r in errors_range:
                for e_z in errors_range:
                    a7 = np.array([
                        np.clip(0.48 * e_ip, -0.7, 0.7),
                        np.clip(1.3 * e_r, -0.7, 0.7),
                        np.clip(2.2 * e_z, -0.7, 0.7),
                        0, 0, 0, 0
                    ])
                    key = (round(e_ip, 2), round(e_r, 2), round(e_z, 2))
                    self.action_table[key] = action_7d_to_12d(a7)
        
        print(f"[MORES] Precomputed {len(self.action_table)} actions")
        self.default_action = np.zeros(12, dtype=np.float32)
    
    def _quantize_error(self, err):
        if err < -0.4:
            return -0.5
        elif err < -0.2:
            return -0.3
        elif err < -0.05:
            return -0.1
        elif err < 0.05:
            return 0
        elif err < 0.2:
            return 0.1
        elif err < 0.4:
            return 0.3
        else:
            return 0.5
    
    def reset(self):
        pass
    
    def act(self, observation: Dict) -> np.ndarray:
        Ip = observation.get('Ip', 0)
        R = observation.get('R', 0)
        Z = observation.get('Z', 0)
        ref_Ip = observation.get('reference_Ip', 0)
        ref_R = observation.get('reference_R', 0)
        ref_Z = observation.get('reference_Z', 0)
        
        err_Ip = (Ip - ref_Ip) / 1e6
        err_R = (R - ref_R) / 0.1
        err_Z = (Z - ref_Z) / 0.1
        
        key = (
            self._quantize_error(err_Ip),
            self._quantize_error(err_R),
            self._quantize_error(err_Z)
        )
        
        return self.action_table.get(key, self.default_action)