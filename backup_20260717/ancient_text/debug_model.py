import paddle
import cv2
import numpy as np
from pathlib import Path
from train_det_full import TextDetector, safe_imread

checkpoint_path = "/checkpoints/det_model_final.pdparams"
img_path = Path("/data")

# 加载模型
model = TextDetector()
state_dict = paddle.load(checkpoint_path)
model.set_state_dict(state_dict)
model.eval()

# 取一张图片
img_files = list(img_path.glob("*.png"))[:1]
img = safe_imread(str(img_files[0]))
h, w = img.shape[:2]

# 预处理
img_resized = cv2.resize(img, (640, 640))
img_tensor = img_resized / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)

# 推理
with paddle.no_grad():
    output = model(img_tensor)

print(f"模型输出 shape: {output.shape}")
print(f"模型输出 dtype: {output.dtype}")
print(f"输出值范围: min={output.min().numpy():.4f}, max={output.max().numpy():.4f}, mean={output.mean().numpy():.4f}")
print(f"输出值 > 0.5 的比例: {(output > 0.5).numpy().sum() / output.numpy().size * 100:.2f}%")