"""
测试所有 checkpoint 的输出
"""

import cv2
import numpy as np
import paddle
from pathlib import Path
from train_det_full import TextDetector, safe_imread

# 模型列表
models = [
    ("epoch10", r"C:\mores_fusion\checkpoints\det_model_epoch10.pdparams"),
    ("epoch20", r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"),
    ("epoch30", r"C:\mores_fusion\checkpoints\det_model_epoch30.pdparams"),
    ("epoch40", r"C:\mores_fusion\checkpoints\det_model_epoch40.pdparams"),
    ("epoch50", r"C:\mores_fusion\checkpoints\det_model_epoch50.pdparams"),
    ("final", r"C:\mores_fusion\checkpoints\det_model_final.pdparams"),
]

# 测试图片
test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
img_files = list(test_dir.glob("*.png"))[:5]

print("=" * 60)
print("测试所有模型（5张图平均热力图最大值）")
print("=" * 60)

for name, path in models:
    print(f"\n加载 {name}...")
    model = TextDetector()
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict)
    model.eval()
    
    max_vals = []
    for img_path in img_files:
        img = safe_imread(str(img_path))
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = img_resized / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            heatmap = model(img_tensor).squeeze().numpy()
        
        max_vals.append(heatmap.max())
    
    avg_max = np.mean(max_vals)
    print(f"  {name}: 热力图最大值范围 {min(max_vals):.6f} ~ {max(max_vals):.6f}, 平均 {avg_max:.6f}")
    
    if avg_max > 0.01:
        print(f"    ✅ 这个模型可用！阈值建议设为 {avg_max * 0.3:.3f}")

print("\n" + "=" * 60)
print("只有热力图最大值 > 0.01 的模型才能产生检测框")