"""
MORES v36 - 基于 v27 模型的后处理优化
策略：高阈值 + 多级过滤，大幅降低 FP
"""

import cv2
import numpy as np
import paddle
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from train_det_full import TextDetector, safe_imread

# ========== 后处理配置（核心优化） ==========
CONFIG = {
    "score_threshold": 0.65,      # 提高阈值（原 v27 可能是 0.3-0.5）
    "nms_threshold": 0.6,         # NMS IoU 阈值
    "min_area": 200,              # 最小面积（过滤噪声）
    "max_aspect_ratio": 3.5,      # 最大宽高比
    "border_margin": 30,          # 边缘过滤（靠近边缘的框往往是噪声）
    "input_size": (640, 640),
}

class MORESInferenceV36:
    def __init__(self, model_path, config=None):
        self.config = CONFIG if config is None else config
        self.model = self._load_model(model_path)
        print(f"[MORES v36] 后处理配置: {self.config}")
    
    def _load_model(self, model_path):
        print(f"[MORES] 加载模型: {model_path}")
        model = TextDetector()
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        model.eval()
        return model
    
    def _predict_heatmap(self, img_path):
        img = safe_imread(str(img_path))
        if img is None:
            return None, None, None
        
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, self.config["input_size"])
        img_tensor = img_resized / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            heatmap = self.model(img_tensor).squeeze().numpy()
        
        return heatmap, h, w
    
    def _extract_boxes(self, heatmap, orig_h, orig_w):
        """后处理优化：多级过滤"""
        cfg = self.config
        input_h, input_w = cfg["input_size"]
        
        # 1. 二值化
        binary = (heatmap > cfg["score_threshold"]).astype(np.uint8)
        
        # 2. 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 3. 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg["min_area"]:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # 4. 宽高比过滤
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > cfg["max_aspect_ratio"]:
                continue
            
            # 5. 边缘过滤
            if (x < cfg["border_margin"] or y < cfg["border_margin"] or
                x + bw > input_w - cfg["border_margin"] or
                y + bh > input_h - cfg["border_margin"]):
                continue
            
            # 6. 计算置信度
            region = heatmap[y:y+bh, x:x+bw]
            score = float(region.mean()) if region.size > 0 else cfg["score_threshold"]
            
            # 缩放到原图
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + bw) * scale_x)
            y2 = int((y + bh) * scale_y)
            
            x1 = max(0, min(x1, orig_w))
            x2 = max(0, min(x2, orig_w))
            y1 = max(0, min(y1, orig_h))
            y2 = max(0, min(y2, orig_h))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
        
        # 7. NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, cfg["score_threshold"], cfg["nms_threshold"])
            if len(indices) > 0:
                boxes = [boxes[i] for i in indices.flatten()]
        
        return boxes
    
    def predict(self, img_path):
        heatmap, h, w = self._predict_heatmap(img_path)
        if heatmap is None:
            return []
        return self._extract_boxes(heatmap, h, w)


def test():
    print("=" * 60)
    print("MORES v36 后处理优化测试")
    print("=" * 60)
    
    model_path = r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"
    if not Path(model_path).exists():
        print("模型不存在")
        return
    
    infer = MORESInferenceV36(model_path)
    
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))[:20]
    
    total = 0
    for img_path in test_files:
        boxes = infer.predict(str(img_path))
        total += len(boxes)
        print(f"  {img_path.name}: {len(boxes)} 个框")
    
    print(f"\n20张图平均框数: {total/20:.1f}")
    print("\n目标：平均每图 0.5-2 个框")

if __name__ == "__main__":
    test()