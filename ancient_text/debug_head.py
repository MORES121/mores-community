"""
检查 head 层的权重
"""
import paddle
import numpy as np

checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
state_dict = paddle.load(checkpoint_path)

print("=" * 50)
print("检查 head 层权重")
print("=" * 50)

# 找出所有 head 相关的层
head_layers = ['head.conv1.weight', 'head.conv1.bias', 
               'head.conv2.weight', 'head.conv2.bias',
               'head.conv3.weight', 'head.conv3.bias']

for name in head_layers:
    if name in state_dict:
        param = state_dict[name].numpy()
        print(f"\n{name}:")
        print(f"  shape: {param.shape}")
        print(f"  min: {param.min():.6f}")
        print(f"  max: {param.max():.6f}")
        print(f"  mean: {param.mean():.6f}")
        print(f"  std: {param.std():.6f}")
    else:
        print(f"\n{name}: 未找到")

# 特别检查 conv3（最后一层）的 bias
if 'head.conv3.bias' in state_dict:
    bias = state_dict['head.conv3.bias'].numpy()
    print(f"\nconv3 bias 值: {bias}")