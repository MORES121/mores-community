import sys
sys.path.insert(0, r'C:\mores_fusion')
from train_det_full import TextDetector, safe_imread
import paddle
import cv2
import numpy as np
from pathlib import Path

# 测试 epoch10
checkpoint = r"C:\mores_fusion\checkpoints\det_model_epoch10.pdparams"
if not Path(checkpoint).exists():
    print(f"文件不存在: {checkpoint}")
    exit()

print(f"加载: {checkpoint}")
model = TextDetector()
state_dict = paddle.load(checkpoint)
model.set_state_dict(state_dict)
model.eval()

# 测试图片
img_path = r"C:\Users\klidw\Downloads\train\train\out_of_domain\ZHJWD000009-000001-JICHENG001103.png"
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败")
    exit()

h, w = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
img_tensor = img_resized / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)

with paddle.no_grad():
    heatmap = model(img_tensor).squeeze().numpy()

print(f"热力图: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")

# 尝试提取框
for thresh in [0.001, 0.01, 0.05, 0.1, 0.2]:
    binary = (heatmap > thresh).astype(np.uint8)
    if binary.sum() > 0:
        print(f"阈值 {thresh}: 激活 {binary.sum()} 像素")
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  轮廓数: {len(contours)}")