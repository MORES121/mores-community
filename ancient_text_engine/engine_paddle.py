"""
MORES 引擎 v31 - 基于 PaddleOCR
可控 · 可解释 · 可决策
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# 抑制 PaddleOCR 的冗余日志
import os
os.environ['GLOG_minloglevel'] = '3'

@dataclass
class CharBox:
    """单个字符/文字的检测结果"""
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    confidence: float
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class ImageResult:
    """单张图片的整体识别结果"""
    path: str
    boxes: List[CharBox]
    total_chars: int = 0
    
    def __post_init__(self):
        self.total_chars = sum(len(box.text) for box in self.boxes)


class MORESPaddleEngine:
    """
    基于 PaddleOCR 的古文字识别引擎
    可控参数：
    - lang: 语言（ch / en / korean / japan）
    - use_angle_cls: 是否使用方向分类
    - det_db_thresh: 检测阈值
    - det_db_box_thresh: 检测框阈值
    """
    
    def __init__(self, lang='ch', use_angle_cls=False, 
                 det_db_thresh=0.3, det_db_box_thresh=0.5,
                 show_log=False):
        
        print("[MORES] 初始化 PaddleOCR 引擎...")
        print(f"   语言: {lang}")
        print(f"   检测阈值: {det_db_thresh}")
        
        from paddleocr import PaddleOCR
        
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            show_log=show_log,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
        )
        print("[MORES] 引擎就绪")
    
    def detect(self, image_path: str) -> List[CharBox]:
        """
        检测图片中的文字
        返回: List[CharBox]
        """
        if not Path(image_path).exists():
            print(f"[ERROR] 图片不存在: {image_path}")
            return []
        
        # 执行 OCR
        result = self.ocr.ocr(image_path, cls=False)
        
        boxes = []
        if result and result[0]:
            for line in result[0]:
                # PaddleOCR 返回格式: [[[x1,y1], [x2,y1], [x2,y2], [x1,y2]], (text, confidence)]
                points = line[0]
                text_info = line[1]
                
                # 提取坐标
                x1 = int(points[0][0])
                y1 = int(points[0][1])
                x2 = int(points[2][0])
                y2 = int(points[2][1])
                
                # 确保 x1<x2, y1<y2
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
    
    def detect_batch(self, image_dir: str, limit: int = None) -> List[ImageResult]:
        """
        批量检测图片
        """
        img_dir = Path(image_dir)
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        
        if limit:
            img_files = img_files[:limit]
        
        results = []
        for img_path in img_files:
            boxes = self.detect(str(img_path))
            results.append(ImageResult(
                path=str(img_path),
                boxes=boxes
            ))
            print(f"  {img_path.name}: {len(boxes)} 个文字框")
        
        return results
    
    def visualize(self, image_path: str, boxes: List[CharBox], 
                  save_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] 无法读取图片: {image_path}")
            return None
        
        for box in boxes:
            # 画框
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            # 写文字和置信度
            label = f"{box.text} ({box.confidence:.2f})"
            cv2.putText(img, label, (box.x1, box.y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"可视化保存至: {save_path}")
        
        return img


# ========== 快速测试 ==========

def quick_test():
    """快速测试 PaddleOCR 是否可用"""
    print("=" * 60)
    print("MORES v31 引擎测试")
    print("=" * 60)
    
    # 测试图片路径
    test_path = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    test_file = Path(test_path) / "ZHJWD000009-000001-JICHENG001103.png"
    
    if not test_file.exists():
        print(f"测试图片不存在: {test_file}")
        print("尝试查找第一张可用图片...")
        test_files = list(Path(test_path).glob("*.png"))
        if not test_files:
            print("未找到任何 PNG 图片")
            return
        test_file = test_files[0]
        print(f"使用: {test_file.name}")
    
    # 创建引擎
    engine = MORESPaddleEngine(lang='ch', det_db_thresh=0.2)
    
    # 检测
    print(f"\n检测图片: {test_file.name}")
    boxes = engine.detect(str(test_file))
    
    print(f"\n检测结果:")
    print(f"  文字框数量: {len(boxes)}")
    
    for i, box in enumerate(boxes[:10]):
        print(f"  [{i+1}] {box.text} | 置信度: {box.confidence:.3f} | 位置: ({box.x1},{box.y1})-({box.x2},{box.y2})")
    
    # 可视化
    output_path = r"C:\mores_engine_v30\test_output.jpg"
    engine.visualize(str(test_file), boxes, output_path)
    
    print(f"\n可视化结果已保存: {output_path}")
    print("\n✅ PaddleOCR 引擎测试完成！")


if __name__ == "__main__":
    quick_test()