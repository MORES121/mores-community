import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v56 - PID + LCFS for F2")
        
        # ========== 阶段1：极端保守（步数 1-80）==========
        self.kp_ip_stage1 = 0.90
        self.kp_r_stage1 = 2.30
        self.kp_z_stage1 = 4.00
        self.action_low_stage1 = -100
        self.action_high_stage1 = 100
        
        # ========== 阶段2：保守（步数 81-200）==========
        self.kp_ip_stage2 = 0.58
        self.kp_r_stage2 = 1.60
        self.kp_z_stage2 = 2.80
        self.action_low_stage2 = -800
        self.action_high_stage2 = 800
        
        # ========== 阶段3：平衡（步数 201+）==========
        self.kp_ip_stage3 = 0.52
        self.kp_r_stage3 = 1.40
        self.kp_z_stage3 = 2.40
        self.action_low_stage3 = -1200
        self.action_high_stage3 = 1200
        
        # LCFS 控制增益（仅 F2 启用）
        self.kp_lcfs = 0.3
        
        errors_range = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]
        
        self.table_stage1 = self._build_table(
            self.kp_ip_stage1, self.kp_r_stage1, self.kp_z_stage1,
            self.action_low_stage1, self.action_high_stage1, errors_range
        )
        self.table_stage2 = self._build_table(
            self.kp_ip_stage2, self.kp_r_stage2, self.kp_z_stage2,
            self.action_low_stage2, self.action_high_stage2, errors_range
        )
        self.table_stage3 = self._build_table(
            self.kp_ip_stage3, self.kp_r_stage3, self.kp_z_stage3,
            self.action_low_stage3, self.action_high_stage3, errors_range
        )
        
        self.default_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        
        print("[MORES] v56 ready for F2")
        print(f"  Stage1 (1-80): ±{self.action_high_stage1}V, gains={self.kp_ip_stage1}/{self.kp_r_stage1}/{self.kp_z_stage1}")
        print(f"  Stage2 (81-200): ±{self.action_high_stage2}V, gains={self.kp_ip_stage2}/{self.kp_r_stage2}/{self.kp_z_stage2}")
        print(f"  Stage3 (201+): ±{self.action_high_stage3}V, gains={self.kp_ip_stage3}/{self.kp_r_stage3}/{self.kp_z_stage3}")
        print(f"  LCFS Gain: {self.kp_lcfs}")
    
    def _build_table(self, kp_ip, kp_r, kp_z, action_low, action_high, errors_range):
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
        
        table = {}
        for e_ip in errors_range:
            for e_r in errors_range:
                for e_z in errors_range:
                    a7 = np.array([
                        np.clip(kp_ip * e_ip, -0.7, 0.7),
                        np.clip(kp_r * e_r, -0.7, 0.7),
                        np.clip(kp_z * e_z, -0.7, 0.7),
                        0, 0, 0, 0
                    ])
                    key = (round(e_ip, 2), round(e_r, 2), round(e_z, 2))
                    table[key] = action_7d_to_12d(a7)
        return table
    
    def _compute_lcfs_error(self, obs: Dict) -> float:
        lcfs_actual = obs.get('lcfs_points', np.zeros((32, 2)))
        lcfs_target = obs.get('reference_lcfs_points', np.zeros((32, 2)))
        
        if lcfs_actual.shape != lcfs_target.shape or lcfs_actual.size == 0:
            return 0.0
        
        actual_center = np.mean(lcfs_actual, axis=0)
        target_center = np.mean(lcfs_target, axis=0)
        shift = target_center - actual_center
        aligned_lcfs = lcfs_actual + shift
        distances = np.sqrt(np.sum((aligned_lcfs - lcfs_target)**2, axis=1))
        lcfs_err = np.mean(distances) / 10.0
        
        return np.clip(lcfs_err, -0.3, 0.3)
    
    def reset(self):
        self.step_count = 0
    
    def act(self, observation: Dict) -> np.ndarray:
        self.step_count += 1
        
        Ip = observation.get('Ip', 0)
        R = observation.get('R', 0)
        Z = observation.get('Z', 0)
        ref_Ip = observation.get('reference_Ip', 0)
        ref_R = observation.get('reference_R', 0)
        ref_Z = observation.get('reference_Z', 0)
        
        err_Ip = (Ip - ref_Ip) / 1000000.0
        err_R = (R - ref_R) / 0.1
        err_Z = (Z - ref_Z) / 0.1
        
        # LCFS 反馈（仅 F2）
        lcfs_err = self._compute_lcfs_error(observation)
        err_R += self.kp_lcfs * lcfs_err
        err_R = np.clip(err_R, -0.7, 0.7)
        
        # 内联量化
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
        
        if self.step_count <= 80:
            action = self.table_stage1.get(key, self.default_action)
        elif self.step_count <= 200:
            action = self.table_stage2.get(key, self.default_action)
        else:
            action = self.table_stage3.get(key, self.default_action)
        
        if self.step_count in [1, 81, 201]:
            stage = "Stage1" if self.step_count <= 80 else ("Stage2" if self.step_count <= 200 else "Stage3")
            print(f"[F2] Step {self.step_count}: {stage}, LCFS_err={lcfs_err:.4f}")
        
        return action