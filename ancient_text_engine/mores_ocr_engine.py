"""
MORES 古文字识别引擎 v32
基于 PaddleOCR 生产版本
可控 · 可解释 · 可决策
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
import time

# 设置环境
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['GLOG_minloglevel'] = '3'

@dataclass
class CharResult:
    """单个字符/文字识别结果"""
    text: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]
    
    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[2]
    @property
    def y2(self): return self.box[3]

@dataclass
class ImageResult:
    """单张图片完整结果"""
    path: str
    chars: List[CharResult]
    elapsed_time: float = 0.0
    total_chars: int = 0
    
    def __post_init__(self):
        self.total_chars = len(self.chars)
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "total_chars": self.total_chars,
            "elapsed_time": self.elapsed_time,
            "chars": [
                {"text": c.text, "confidence": c.confidence, "box": c.box}
                for c in self.chars
            ]
        }


class MORESOCREngine:
    """
    墨睿思古文字识别引擎
    可控参数：
    - text_det_thresh: 检测阈值 (0.1-0.9)
    - text_det_box_thresh: 检测框阈值 (0.1-0.9)
    - lang: 语言 ('ch', 'en', 'korean', 'japan')
    """
    
    def __init__(self, 
                 text_det_thresh: float = 0.3,
                 text_det_box_thresh: float = 0.5,
                 lang: str = 'ch',
                 use_angle_cls: bool = False):
        
        from paddleocr import PaddleOCR
        
        print("[MORES] 初始化 OCR 引擎...")
        print(f"   参数: det_thresh={text_det_thresh}, box_thresh={text_det_box_thresh}, lang={lang}")
        
        self.ocr = PaddleOCR(
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            use_angle_cls=use_angle_cls,
            lang=lang,
            show_log=False
        )
        
        # 可调参数
        self.config = {
            "text_det_thresh": text_det_thresh,
            "text_det_box_thresh": text_det_box_thresh,
            "lang": lang
        }
        
        print("[MORES] 引擎就绪")
    
    def detect(self, image_path: str) -> List[CharResult]:
        """检测图片中的文字"""
        if not Path(image_path).exists():
            print(f"[ERROR] 图片不存在: {image_path}")
            return []
        
        start_time = time.time()
        
        # 执行 OCR
        result = self.ocr.ocr(image_path, cls=False)
        
        elapsed = time.time() - start_time
        
        chars = []
        if result and result[0]:
            for line in result[0]:
                points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                # 提取边界框
                x1 = int(min(p[0] for p in points))
                y1 = int(min(p[1] for p in points))
                x2 = int(max(p[0] for p in points))
                y2 = int(max(p[1] for p in points))
                
                chars.append(CharResult(
                    text=text,
                    confidence=confidence,
                    box=[x1, y1, x2, y2]
                ))
        
        return chars
    
    def detect_batch(self, image_dir: str, limit: int = None) -> List[ImageResult]:
        """批量检测图片"""
        img_dir = Path(image_dir)
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        
        if limit:
            img_files = img_files[:limit]
        
        results = []
        for img_path in img_files:
            chars = self.detect(str(img_path))
            results.append(ImageResult(
                path=str(img_path),
                chars=chars
            ))
            print(f"  {img_path.name}: {len(chars)} 个字符")
        
        return results
    
    def visualize(self, image_path: str, chars: List[CharResult], 
                  save_path: str = None) -> np.ndarray:
        """可视化检测结果"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        for char in chars:
            x1, y1, x2, y2 = char.box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{char.text} ({char.confidence:.2f})"
            cv2.putText(img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"可视化保存: {save_path}")
        
        return img
    
    def update_config(self, **kwargs):
        """动态更新配置"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"[MORES] 配置更新: {key} = {value}")
        
        # 重新初始化 OCR（参数生效需要重建）
        self.__init__(**self.config)
    
    def get_config(self) -> dict:
        """获取当前配置"""
        return self.config.copy()


# 快速测试
def test():
    print("=" * 60)
    print("MORES 古文字识别引擎 v32 - 测试")
    print("=" * 60)
    
    # 测试图片
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))[:3]
    
    if not test_files:
        print("未找到测试图片")
        return
    
    # 创建引擎
    engine = MORESOCREngine(text_det_thresh=0.3, lang='ch')
    
    print("\n" + "=" * 60)
    print("批量检测")
    print("=" * 60)
    
    for test_file in test_files:
        print(f"\n图片: {test_file.name}")
        chars = engine.detect(str(test_file))
        
        print(f"  检测到 {len(chars)} 个文字块:")
        for char in chars:
            print(f"    '{char.text}' | 置信度: {char.confidence:.3f} | 位置: {char.box}")
        
        # 可视化
        output_path = f"C:\\mores_engine_v30\\result_{test_file.stem}.jpg"
        engine.visualize(str(test_file), chars, output_path)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test()