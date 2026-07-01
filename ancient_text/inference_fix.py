"""
修复版推理 - 对负数输出做归一化
"""
import paddle
import cv2
import numpy as np
from pathlib import Path
from train_det_full import TextDetector, safe_imread

def predict_boxes(model, img_path, input_size=(640, 640)):
    """修复版预测"""
    img = safe_imread(str(img_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # 预处理
    img_resized = cv2.resize(img, input_size)
    img_tensor = img_resized / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
    
    with paddle.no_grad():
        output = model(img_tensor).squeeze().numpy()
    
    # 关键修复：因为输出全是负数，用 min-max 归一化到 [0, 1]
    output_min = output.min()
    output_max = output.max()
    if output_max > output_min:
        output = (output - output_min) / (output_max - output_min)
    else:
        # 如果全相同，返回空
        return []
    
    # 使用阈值提取框
    threshold = 0.3  # 可调整
    binary = (output > threshold).astype(np.uint8)
    
    # 形态学操作（可选）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # 缩放到原图尺寸
        scale_x = w / input_size[0]
        scale_y = h / input_size[1]
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + bw) * scale_x
        y2 = (y + bh) * scale_y
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    
    return boxes

def main():
    checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
    img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    
    # 加载模型
    print("加载模型...")
    model = TextDetector()
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)
    model.eval()
    
    # 测试第一张图
    img_files = list(img_dir.glob("*.png"))[:1]
    img_path = img_files[0]
    
    print(f"测试图片: {img_path.name}")
    boxes = predict_boxes(model, img_path)
    
    print(f"检测到 {len(boxes)} 个框")
    for i, box in enumerate(boxes[:5]):
        print(f"  框 {i+1}: {box}")
    
    if len(boxes) > 0:
        print("\n✅ 修复成功！模型有输出")
    else:
        print("\n⚠️ 仍未检测到框，需要调整阈值")

if __name__ == '__main__':
    main()