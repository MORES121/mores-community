"""
集成 v27 检测模型到 MORES 引擎
"""

import sys
import cv2
import numpy as np
import paddle
from pathlib import Path

# 添加训练脚本路径
sys.path.insert(0, r'C:\mores_fusion')
from train_det_full import TextDetector, safe_imread

# 导入引擎
sys.path.insert(0, r'C:\mores_engine_v30')
from mores_engine_v30 import MORESCharacterEngine, DetectionBox

class V27DetectorAdapter:
    """
    v27 模型适配器
    将原始模型输出转换为引擎可用的 DetectionBox 格式
    """
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 v27 模型"""
        print(f"[V27] 加载模型: {self.checkpoint_path}")
        self.model = TextDetector()
        state_dict = paddle.load(self.checkpoint_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()
        print("[V27] 模型加载完成")
    
    def _extract_boxes_from_heatmap(self, heatmap: np.ndarray, 
                                     original_h: int, original_w: int,
                                     input_size=(640, 640)) -> list:
        """
        从热力图提取检测框
        这是关键：需要找到正确的提取方法
        """
        boxes = []
        
        # 方法1：尝试不同的阈值
        for thresh in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            binary = (heatmap > thresh).astype(np.uint8)
            if binary.sum() > 0:
                print(f"[V27] 阈值 {thresh} 下激活像素: {binary.sum()}")
                
                # 形态学操作
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # 提取轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 10:
                        continue
                    
                    x, y, bw, bh = cv2.boundingRect(contour)
                    
                    # 缩放到原图尺寸
                    scale_x = original_w / input_size[0]
                    scale_y = original_h / input_size[1]
                    
                    x1 = int(x * scale_x)
                    y1 = int(y * scale_y)
                    x2 = int((x + bw) * scale_x)
                    y2 = int((y + bh) * scale_y)
                    
                    # 计算该区域的置信度
                    region = heatmap[y:y+bh, x:x+bw]
                    confidence = float(region.mean()) if region.size > 0 else 0.5
                    
                    boxes.append(DetectionBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=confidence
                    ))
                
                if boxes:
                    print(f"[V27] 阈值 {thresh} 下提取到 {len(boxes)} 个框")
                    return boxes
        
        print(f"[V27] 未提取到任何框")
        return boxes
    
    def detect(self, image: np.ndarray) -> list:
        """
        检测接口，供引擎调用
        """
        if self.model is None:
            return []
        
        h, w = image.shape[:2]
        
        # 预处理
        img_resized = cv2.resize(image, (640, 640))
        img_tensor = img_resized / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
        
        # 推理
        with paddle.no_grad():
            heatmap = self.model(img_tensor).squeeze().numpy()
        
        # 调试：输出热力图统计
        print(f"[V27] 热力图: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
        
        # 提取框
        boxes = self._extract_boxes_from_heatmap(heatmap, h, w)
        
        return boxes


def main():
    print("=" * 60)
    print("测试检测器集成")
    print("=" * 60)
    
    # 创建引擎
    engine = MORESCharacterEngine()
    
    # 创建检测器适配器（使用 v27 模型）
    checkpoint = r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"
    
    # 检查文件是否存在
    if not Path(checkpoint).exists():
        print(f"模型不存在: {checkpoint}")
        print("尝试其他路径...")
        checkpoint = r"C:\mores_fusion\checkpoints\det_model_epoch10.pdparams"
    
    if Path(checkpoint).exists():
        detector = V27DetectorAdapter(checkpoint)
        
        # 将检测器注册到引擎
        engine.set_detector(detector)
        engine._detector = detector  # 直接设置
        
        # 注意：引擎的 detect 方法需要适配
        # 临时方案：直接调用检测器
        print("\n" + "=" * 60)
        print("测试单张图片")
        print("=" * 60)
        
        # 加载测试图片
        img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
        img_files = list(img_dir.glob("*.png"))[:3]
        
        for img_path in img_files:
            print(f"\n图片: {img_path.name}")
            img = cv2.imread(str(img_path))
            if img is None:
                print("  读取失败")
                continue
            
            boxes = detector.detect(img)
            print(f"  检测到 {len(boxes)} 个框")
            
            for i, box in enumerate(boxes[:3]):
                print(f"    框{i+1}: ({box.x1},{box.y1})-({box.x2},{box.y2}), conf={box.confidence:.3f}")
    else:
        print("未找到 v27 模型文件")
        print("请检查 C:\\mores_fusion\\checkpoints\\ 目录")

if __name__ == "__main__":
    main()