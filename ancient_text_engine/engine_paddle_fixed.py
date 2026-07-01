"""
MORES 引擎 v31 - 基于 PaddleOCR
修复参数名
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass

import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

@dataclass
class CharBox:
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    confidence: float

class MORESPaddleEngine:
    def __init__(self, lang='ch', use_textline_orientation=False, 
                 text_det_thresh=0.3, text_det_box_thresh=0.5):
        
        print("[MORES] 初始化 PaddleOCR 引擎...")
        print(f"   语言: {lang}")
        print(f"   检测阈值: {text_det_thresh}")
        
        from paddleocr import PaddleOCR
        
        # 使用新的参数名
        self.ocr = PaddleOCR(
            use_textline_orientation=use_textline_orientation,
            lang=lang,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            log_level='ERROR',  # 只显示错误
        )
        print("[MORES] 引擎就绪")
    
    def detect(self, image_path: str) -> List[CharBox]:
        if not Path(image_path).exists():
            return []
        
        result = self.ocr.ocr(image_path, cls=False)
        
        boxes = []
        if result and result[0]:
            for line in result[0]:
                points = line[0]
                text_info = line[1]
                
                x1 = int(points[0][0])
                y1 = int(points[0][1])
                x2 = int(points[2][0])
                y2 = int(points[2][1])
                
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                boxes.append(CharBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    text=text_info[0],
                    confidence=float(text_info[1])
                ))
        
        return boxes
    
    def visualize(self, image_path: str, boxes: List[CharBox], save_path: str = None):
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        for box in boxes:
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            label = f"{box.text} ({box.confidence:.2f})"
            cv2.putText(img, label, (box.x1, box.y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if save_path:
            cv2.imwrite(save_path, img)
        return img


def quick_test():
    print("=" * 60)
    print("MORES v31 引擎测试（修复版）")
    print("=" * 60)
    
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))
    
    if not test_files:
        print("未找到测试图片")
        return
    
    test_file = test_files[0]
    print(f"\n测试图片: {test_file.name}")
    
    engine = MORESPaddleEngine(lang='ch', text_det_thresh=0.3)
    
    print("\n检测中...")
    boxes = engine.detect(str(test_file))
    
    print(f"\n检测到 {len(boxes)} 个文字框")
    for i, box in enumerate(boxes[:5]):
        print(f"  [{i+1}] '{box.text}' | conf={box.confidence:.3f}")
    
    # 可视化
    output_path = r"C:\mores_engine_v30\test_output.jpg"
    engine.visualize(str(test_file), boxes, output_path)
    print(f"\n可视化保存至: {output_path}")
    
    if len(boxes) > 0:
        print("\n✅ PaddleOCR 工作正常！")
    else:
        print("\n⚠️ 未检测到文字，尝试降低阈值")

if __name__ == "__main__":
    quick_test()