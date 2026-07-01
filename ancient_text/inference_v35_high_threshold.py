"""
MORES 古文字识别推理 v35
策略：高阈值 + 规则过滤，大幅降低 FP
适用：冲刺复赛前100名
"""

import os
import cv2
import numpy as np
import paddle
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 导入训练脚本中的模型定义
from train_det_full import TextDetector, safe_imread

# ========== 配置参数（可调） ==========
CONFIG = {
    "score_threshold": 0.75,      # 大幅提高，过滤低置信度框
    "nms_threshold": 0.6,         # NMS 阈值
    "min_area": 150,              # 最小框面积（像素，原图尺寸下）
    "max_aspect_ratio": 4.0,      # 最大宽高比
    "input_size": (640, 640),     # 输入尺寸
}

class MORESInferenceV35:
    def __init__(self, model_path, config=None):
        self.config = CONFIG if config is None else config
        self.model = self._load_model(model_path)
        print(f"[MORES] 推理配置: {self.config}")
    
    def _load_model(self, model_path):
        """加载模型"""
        print(f"[MORES] 加载模型: {model_path}")
        model = TextDetector()
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        model.eval()
        print("[MORES] 模型加载完成")
        return model
    
    def _predict_heatmap(self, img_path):
        """预测热力图"""
        img = safe_imread(str(img_path))
        if img is None:
            return None, None, None
        
        h, w = img.shape[:2]
        
        # 预处理
        img_resized = cv2.resize(img, self.config["input_size"])
        img_tensor = img_resized / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            heatmap = self.model(img_tensor).squeeze().numpy()
        
        return heatmap, h, w
    
    def _extract_boxes(self, heatmap, orig_h, orig_w):
        """从热力图提取检测框（高阈值 + 规则过滤）"""
        cfg = self.config
        input_h, input_w = cfg["input_size"]
        
        # 二值化（高阈值）
        binary = (heatmap > cfg["score_threshold"]).astype(np.uint8)
        
        # 形态学操作（连接邻近区域）
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg["min_area"]:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # 规则1：宽高比过滤
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect_ratio > cfg["max_aspect_ratio"]:
                continue
            
            # 规则2：位置过滤（不处理太靠近边缘的框，这些往往是噪声）
            if x < 10 or y < 10 or (x + bw) > (input_w - 10) or (y + bh) > (input_h - 10):
                continue
            
            # 计算该区域的置信度（热力图均值）
            region = heatmap[y:y+bh, x:x+bw]
            score = float(region.mean()) if region.size > 0 else cfg["score_threshold"]
            
            # 缩放到原图尺寸
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + bw) * scale_x)
            y2 = int((y + bh) * scale_y)
            
            # 确保坐标有效
            x1 = max(0, min(x1, orig_w))
            x2 = max(0, min(x2, orig_w))
            y1 = max(0, min(y1, orig_h))
            y2 = max(0, min(y2, orig_h))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
        
        # NMS 去重
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, cfg["score_threshold"], cfg["nms_threshold"])
            if len(indices) > 0:
                boxes = [boxes[i] for i in indices.flatten()]
        
        return boxes
    
    def predict(self, img_path):
        """单张图片预测"""
        heatmap, h, w = self._predict_heatmap(img_path)
        if heatmap is None:
            return []
        
        boxes = self._extract_boxes(heatmap, h, w)
        return boxes
    
    def predict_batch(self, img_dir, output_file="submission_v35.json"):
        """批量预测并生成提交文件"""
        img_dir = Path(img_dir)
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        
        print(f"\n[MORES] 开始批量预测，共 {len(img_files)} 张图片")
        
        results = {}
        total_boxes = 0
        
        for i, img_path in enumerate(img_files):
            boxes = self.predict(str(img_path))
            total_boxes += len(boxes)
            
            # 保存结果（格式：每张图的框列表）
            results[img_path.name] = [{"box": box} for box in boxes]
            
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1}/{len(img_files)} 张，累计框数: {total_boxes}")
        
        # 保存提交文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[MORES] 预测完成！")
        print(f"  总图片数: {len(img_files)}")
        print(f"  总预测框数: {total_boxes}")
        print(f"  平均每图: {total_boxes/len(img_files):.2f}")
        print(f"  提交文件: {output_file}")
        
        return results


# ========== 快速测试 ==========
def test():
    print("=" * 60)
    print("MORES v35 高阈值推理测试")
    print("=" * 60)
    
    # 模型路径
    model_path = r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"
    
    if not Path(model_path).exists():
        print(f"模型不存在: {model_path}")
        return
    
    infer = MORESInferenceV35(model_path)
    
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))[:20]
    
    print(f"\n测试 {len(test_files)} 张图片...")
    for img_path in test_files:
        boxes = infer.predict(str(img_path))
        print(f"  {img_path.name}: {len(boxes)} 个框")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test()