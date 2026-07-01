"""
MORES 古文字识别引擎 v33 - 优化版
参数: 检测阈值 0.2
"""

from mores_ocr_engine import MORESOCREngine, ImageResult, CharResult
from pathlib import Path
import json
import time

class MORESEngineOptimized(MORESOCREngine):
    """优化版引擎 - 使用最佳阈值 0.2"""
    
    def __init__(self, lang='ch'):
        super().__init__(
            text_det_thresh=0.2,      # 最佳阈值
            text_det_box_thresh=0.3,   # 框阈值相应降低
            lang=lang
        )
        print("[MORES] 优化版引擎 v33 已启动")
        print(f"   配置: text_det_thresh=0.2, text_det_box_thresh=0.3")


def quick_test_optimized():
    """快速测试优化版"""
    print("=" * 60)
    print("MORES 优化版引擎 v33 测试")
    print("=" * 60)
    
    engine = MORESEngineOptimized()
    
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))[:20]
    
    total = 0
    for img_path in test_files:
        chars = engine.detect(str(img_path))
        total += len(chars)
    
    avg = total / len(test_files)
    print(f"\n20张图平均字符数: {avg:.2f}")
    print(f"   (对比阈值0.3时的 1.26)")
    
    if avg > 1.3:
        print("\n✅ 优化成功！检测率提升")
    else:
        print("\n⚠️ 需要进一步调优")

if __name__ == "__main__":
    quick_test_optimized()