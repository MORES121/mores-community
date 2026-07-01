"""
检查训练是否正常 - 看 checkpoint 是否有有效参数
"""
import paddle
import numpy as np

checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_epoch50.pdparams"

print("=" * 50)
print("检查模型权重")
print("=" * 50)

# 加载权重
state_dict = paddle.load(checkpoint_path)

print(f"\n模型层数: {len(state_dict.keys())}")
print(f"\n前5个层的权重统计:")

for i, (name, param) in enumerate(state_dict.items()):
    if i >= 5:
        break
    param_np = param.numpy()
    print(f"\n{name}:")
    print(f"  shape: {param_np.shape}")
    print(f"  min: {param_np.min():.6f}")
    print(f"  max: {param_np.max():.6f}")
    print(f"  mean: {param_np.mean():.6f}")
    print(f"  std: {param_np.std():.6f}")

# 检查最后一层卷积的权重
print("\n" + "=" * 50)
print("检查最后一层（DBHead.conv3.weight）:")

last_conv = None
for name, param in state_dict.items():
    if 'conv3.weight' in name:
        last_conv = param.numpy()
        print(f"找到: {name}")
        print(f"  min: {last_conv.min():.6f}")
        print(f"  max: {last_conv.max():.6f}")
        print(f"  mean: {last_conv.mean():.6f}")
        break

if last_conv is None:
    print("未找到 conv3.weight")

print("\n" + "=" * 50)
print("如果所有权重都是 0 或接近 0，说明训练失败")
print("如果权重正常（有正有负），说明推理代码有问题")