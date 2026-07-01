import sys
sys.path.insert(0, 'C:\\mores_fusion')
from inference_v27 import predict_boxes_v27
from train_det_full import TextDetector
import paddle
from pathlib import Path

# 加载模型
print("加载模型中...")
model = TextDetector()
state_dict = paddle.load('C:\\mores_fusion\\checkpoints\\det_model_epoch20.pdparams')
model.set_state_dict(state_dict)
model.eval()
print("模型加载完成")

# 图片路径
img_path = r'C:\test_saisdata\13\eval\images\0032.png'

# 测试不同阈值
for thresh in [0.3, 0.5, 0.7]:
    boxes = predict_boxes_v27(model, img_path, prob_thresh=thresh, nms_thresh=0.6, min_area=100)
    print(f'阈值 {thresh}: 检测到 {len(boxes)} 个框')
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f'  框{i+1}: ({x1},{y1})-({x2},{y2})')