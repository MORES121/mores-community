import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v31.0 - F2 (Simple Test)")
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        print("[MORES] F2 reset called")
    
    def act(self, observation: Dict) -> np.ndarray:
        self.step_count += 1
        
        # 打印前3步的观测keys，便于调试
        if self.step_count <= 3:
            print(f"[MORES] F2 step {self.step_count}: observation keys = {list(observation.keys())[:5]}")
        
        # 返回零动作（12维）
        return np.zeros(12, dtype=np.float32)