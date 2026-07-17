"""
直接调试模型推理 - 不经过后处理
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
print(f"测试图片: {img_files[0].name}")

img = safe_imread(str(img_files[0]))
if img is None:
    print("图片读取失败")
    exit()

h, w = img.shape[:2]
print(f"原图尺寸: {w} x {h}")

# 预处理
img_resized = cv2.resize(img, (640, 640))
img_tensor = img_resized / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)

# 推理
with paddle.no_grad():
    output = model(img_tensor)

print(f"\n模型输出 shape: {output.shape}")
print(f"输出值统计:")
print(f"  min: {output.min().numpy():.6f}")
print(f"  max: {output.max().numpy():.6f}")
print(f"  mean: {output.mean().numpy():.6f}")

# 输出前10个值（看看是不是真的全0）
output_np = output.numpy().flatten()
print(f"\n前20个输出值:")
for i in range(min(20, len(output_np))):
    print(f"  [{i}] = {output_np[i]:.6f}")

# 尝试不同的阈值
print(f"\n不同阈值下的激活像素数:")
for thresh in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
    active = (output_np > thresh).sum()
    print(f"  > {thresh}: {active} 像素 ({active/output_np.size*100:.4f}%)")