"""
检查输出值的分布
"""
import paddle
import cv2
import numpy as np
from pathlib import Path
from train_det_full import TextDetector, safe_imread

checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")

# 加载模型
model = TextDetector()
state_dict = paddle.load(checkpoint_path)
model.set_state_dict(state_dict)
model.eval()

# 取一张图片
img_files = list(img_dir.glob("*.png"))[:1]
img_path = img_files[0]
print(f"图片: {img_path.name}")

img = safe_imread(str(img_path))
img_resized = cv2.resize(img, (640, 640))
img_tensor = img_resized / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)

with paddle.no_grad():
    output = model(img_tensor).squeeze().numpy()

print(f"\n输出统计:")
print(f"  shape: {output.shape}")
print(f"  min: {output.min():.6f}")
print(f"  max: {output.max():.6f}")
print(f"  mean: {output.mean():.6f}")
print(f"  std: {output.std():.6f}")

# 检查是否所有值相等
unique_vals = np.unique(output)
print(f"  唯一值数量: {len(unique_vals)}")

if len(unique_vals) <= 2:
    print(f"  ⚠️ 输出几乎全是常数！唯一值: {unique_vals[:10]}")

# 输出直方图分段
print(f"\n值域分布:")
bins = [-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000]
for i in range(len(bins)-1):
    count = ((output >= bins[i]) & (output < bins[i+1])).sum()
    if count > 0:
        print(f"  [{bins[i]}, {bins[i+1]}): {count} 像素 ({count/output.size*100:.2f}%)")