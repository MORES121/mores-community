import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v31.0 - F1 (Simple Test)")
    
    def reset(self):
        pass
    
    def act(self, observation: Dict) -> np.ndarray:
        # 返回零动作，测试服务是否正常
        return np.zeros(12, dtype=np.float32)