"""
测试不同阈值的检测效果
"""

import cv2
import numpy as np
import paddle
from pathlib import Path
from train_det_full import TextDetector, safe_imread

# 加载模型
model_path = r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"
print(f"加载模型: {model_path}")
model = TextDetector()
state_dict = paddle.load(model_path)
model.set_state_dict(state_dict)
model.eval()

# 测试图片
test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
img_files = list(test_dir.glob("*.png"))[:5]

# 测试不同阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

print("\n" + "=" * 60)
print("阈值测试（5张图平均框数）")
print("=" * 60)

for thresh in thresholds:
    total_boxes = 0
    
    for img_path in img_files:
        img = safe_imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = img_resized / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            heatmap = model(img_tensor).squeeze().numpy()
        
        # 简单提取：直接阈值
        binary = (heatmap > thresh).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 简单过滤
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                total_boxes += 1
    
    avg = total_boxes / len(img_files)
    print(f"  阈值 {thresh}: 平均 {avg:.1f} 个框")

print("\n" + "=" * 60)
print("建议：选择平均框数在 1-3 之间的阈值")