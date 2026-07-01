"""
MORES 古文字识别引擎 v34
全面优化版 - 准备复赛
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re

class CharKnowledge:
    """古文字知识库（核心资产，不提交）"""
    
    def __init__(self):
        # 常见古文字映射（可扩展）
        self.ancient_to_modern = {
            "𠀀": "一", "𠀁": "上", "𠀂": "下",
        }
        
        # 偏旁部首权重
        self.radical_weight = {
            "口": 1.2, "木": 1.1, "水": 1.1, "金": 1.2, "火": 1.0
        }
    
    def correct(self, text: str, confidence: float) -> Tuple[str, float]:
        """知识纠正"""
        if text in self.ancient_to_modern:
            text = self.ancient_to_modern[text]
            confidence = min(confidence * 1.1, 1.0)
        
        if len(text) == 1 and not self._is_valid_char(text):
            confidence *= 0.5
        
        return text, confidence
    
    def _is_valid_char(self, char: str) -> bool:
        """判断是否为有效汉字"""
        return '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf'


class MORESEngineV34:
    """MORES 引擎 v34 - 全面优化版"""
    
    def __init__(self, 
                 det_thresh=0.15,
                 box_thresh=0.25,
                 use_angle_cls=True,
                 use_knowledge=True):
        
        print("[MORES] 初始化 v34 引擎...")
        print(f"   检测阈值: {det_thresh}")
        print(f"   方向分类: {use_angle_cls}")
        print(f"   知识库: {use_knowledge}")
        
        self.ocr = PaddleOCR(
            text_det_thresh=det_thresh,
            text_det_box_thresh=box_thresh,
            use_angle_cls=use_angle_cls,
            lang='ch',
            show_log=False
        )
        
        self.use_knowledge = use_knowledge
        if use_knowledge:
            self.knowledge = CharKnowledge()
    
    def detect(self, image_path: str) -> List[Dict]:
        """检测图片中的文字"""
        result = self.ocr.ocr(image_path, cls=False)
        
        chars = []
        if result and result[0]:
            for line in result[0]:
                points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                if self.use_knowledge:
                    text, confidence = self.knowledge.correct(text, confidence)
                
                x1 = int(min(p[0] for p in points))
                y1 = int(min(p[1] for p in points))
                x2 = int(max(p[0] for p in points))
                y2 = int(max(p[1] for p in points))
                
                chars.append({
                    "text": text,
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence
                })
        
        return chars
    
    def detect_batch(self, image_dir: str, limit: int = None) -> List:
        """批量检测"""
        img_dir = Path(image_dir)
        img_files = list(img_dir.glob("*.png"))
        
        if limit:
            img_files = img_files[:limit]
        
        results = []
        total = 0
        for i, img_path in enumerate(img_files):
            chars = self.detect(str(img_path))
            total += len(chars)
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1} 张, 总字符: {total}")
            results.append({"file": img_path.name, "count": len(chars)})
        
        avg = total / len(img_files)
        print(f"\n{'='*50}")
        print(f"批量检测完成")
        print(f"总图片数: {len(img_files)}")
        print(f"总字符数: {total}")
        print(f"平均每图: {avg:.2f}")
        print(f"{'='*50}")
        
        return results


def test_v34():
    print("=" * 60)
    print("MORES v34 引擎测试（全量数据）")
    print("=" * 60)
    
    engine = MORESEngineV34(det_thresh=0.15, box_thresh=0.25)
    
    test_dir = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    print(f"\n开始检测 {test_dir}")
    print("这可能需要几分钟...\n")
    
    results = engine.detect_batch(test_dir)
    
    return results


if __name__ == "__main__":
    test_v34()