"""
测试去掉 sigmoid 后的输出
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from train_det_full import safe_imread

class DBHeadNoSigmoid(nn.Layer):
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2D(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2D(32, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # 去掉 sigmoid
        return x

class TextDetectorNoSigmoid(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2D(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2D(128, 64, 3, padding=1)
        self.pool = nn.MaxPool2D(2, 2)
        self.head = DBHeadNoSigmoid(64)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        x = self.head(x)
        return x

# 加载模型
checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")

print("=" * 50)
print("测试去掉 sigmoid")
print("=" * 50)

model = TextDetectorNoSigmoid()
state_dict = paddle.load(checkpoint_path)

# 需要修改 state_dict 的 key（head. 前缀）
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('head.'):
        new_state_dict[k] = v
    else:
        new_state_dict[k] = v

model.set_state_dict(new_state_dict)
model.eval()

# 取一张图片
img_files = list(img_dir.glob("*.png"))[:1]
img = safe_imread(str(img_files[0]))
if img is None:
    print("图片读取失败")
    exit()

# 预处理
img_resized = cv2.resize(img, (640, 640))
img_tensor = img_resized / 255.0
img_tensor = np.transpose(img_tensor, (2, 0, 1))
img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)

with paddle.no_grad():
    output = model(img_tensor)

print(f"\n去掉 sigmoid 后的输出:")
print(f"  shape: {output.shape}")
print(f"  min: {output.min().numpy():.6f}")
print(f"  max: {output.max().numpy():.6f}")
print(f"  mean: {output.mean():.6f}")

output_np = output.numpy().flatten()
print(f"\n不同阈值下的激活像素数:")
for thresh in [-10, -5, 0, 5, 10]:
    active = (output_np > thresh).sum()
    print(f"  > {thresh}: {active} 像素 ({active/output_np.size*100:.4f}%)")