"""
逐层调试 - 找出哪一层输出变成 0
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from train_det_full import TextDetector, safe_imread
import cv2
from pathlib import Path

class DebugModel(TextDetector):
    """带调试输出的模型"""
    def forward(self, x):
        print(f"输入 shape: {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}]")
        
        x = F.relu(self.conv1(x))
        print(f"conv1 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}], 均值 {x.mean():.4f}")
        
        x = self.pool(x)
        print(f"pool1 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}]")
        
        x = F.relu(self.conv2(x))
        print(f"conv2 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}], 均值 {x.mean():.4f}")
        
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        print(f"conv3 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}], 均值 {x.mean():.4f}")
        
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        print(f"conv4 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}], 均值 {x.mean():.4f}")
        
        x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        print(f"上采样后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}]")
        
        x = self.head(x)
        print(f"head 后: shape {x.shape}, 范围 [{x.min():.4f}, {x.max():.4f}]")
        
        return x

# 加载模型
checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")

print("=" * 50)
print("逐层调试")
print("=" * 50)

model = DebugModel()
state_dict = paddle.load(checkpoint_path)
model.set_state_dict(state_dict)
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

print("\n开始前向传播:\n")
with paddle.no_grad():
    output = model(img_tensor)

print("\n" + "=" * 50)
print("如果某一层输出全 0，那就是问题所在")