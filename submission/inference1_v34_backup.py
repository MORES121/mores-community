import numpy as np
from typing import Dict


class Policy:
    def __init__(self, model_dir: str = None):
        print("[MORES] v34 - Minimal Test")
    
    def reset(self):
        pass
    
    def act(self, observation: Dict) -> np.ndarray:
        return np.zeros(12, dtype=np.float32)