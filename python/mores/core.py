"""核心决策引擎（调用 Rust 底层库）"""

import ctypes
import sys
from pathlib import Path
from typing import List, Optional
from .types import DecisionRequest, DecisionResult, Rule

# 尝试加载 Rust 编译的共享库
_lib = None

def _load_library():
    """加载 Rust 核心库"""
    global _lib
    if _lib is not None:
        return _lib
    
    # 查找库文件路径
    lib_paths = [
        Path(__file__).parent / "libmores.so",      # Linux
        Path(__file__).parent / "libmores.dylib",   # macOS
        Path(__file__).parent / "mores.dll",        # Windows
        Path(__file__).parent.parent / "target/release/libmores.so",
        Path(__file__).parent.parent / "target/release/mores.dll",
    ]
    
    for path in lib_paths:
        if path.exists():
            try:
                _lib = ctypes.CDLL(str(path))
                return _lib
            except Exception:
                continue
    
    # 如果找不到库，返回 None（纯 Python 回退模式）
    return None

class DecisionEngine:
    """决策引擎"""
    
    def __init__(self):
        self._rules: List[Rule] = []
        self._threshold: float = 0.5
        self._lib = _load_library()
    
    def set_threshold(self, threshold: float) -> None:
        """设置置信度阈值"""
        self._threshold = max(0.0, min(1.0, threshold))
    
    def add_rule(self, rule: Rule) -> None:
        """添加决策规则"""
        self._rules.append(rule)
    
    def decide(self, request: DecisionRequest) -> DecisionResult:
        """执行决策"""
        # 简单的规则匹配
        triggered = []
        total_weight = 0.0
        
        for rule in self._rules:
            if rule.condition in request.input:
                triggered.append(rule.name)
                total_weight += rule.weight
        
        if self._rules:
            confidence = total_weight / sum(r.weight for r in self._rules)
            confidence = min(1.0, confidence)
        else:
            confidence = 0.5
        
        decision = "通过" if confidence >= self._threshold else "需人工审核"
        
        reasoning = f"触发规则: {triggered}, 置信度: {confidence:.2f}" if triggered else "未触发规则"
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            rules_triggered=triggered,
        )