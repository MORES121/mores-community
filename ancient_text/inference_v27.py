"""
v27 模型推理 - 优化后处理
"""
import paddle
import cv2
import numpy as np
from pathlib import Path
from train_det_full import TextDetector, safe_imread

def predict_boxes_v27(model, img_path, input_size=(640, 640), 
                      prob_thresh=0.3, nms_thresh=0.5, min_area=20):
    """
    v27 模型推理，可调节参数
    """
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
        heatmap = model(img_tensor).squeeze().numpy()
    
    # 阈值过滤
    binary = (heatmap > prob_thresh).astype(np.uint8)
    
    # 形态学操作（连接邻近区域）
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    scores = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 获取外接矩形
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # 计算该区域的置信度（热力图均值）
        region = heatmap[y:y+bh, x:x+bw]
        score = region.mean() if region.size > 0 else prob_thresh
        
        # 缩放到原图尺寸
        scale_x = w / input_size[0]
        scale_y = h / input_size[1]
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + bw) * scale_x)
        y2 = int((y + bh) * scale_y)
        
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
    
    # NMS 去重
    indices = cv2.dnn.NMSBoxes(boxes, scores, prob_thresh, nms_thresh)
    if len(indices) > 0:
        boxes = [boxes[i] for i in indices.flatten()]
    else:
        boxes = []
    
    return boxes

def main():
    # 模型路径（使用 v27 的模型）
    checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_epoch20.pdparams"
    
    # 如果 epoch20 不存在，尝试其他
    if not Path(checkpoint_path).exists():
        checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_epoch10.pdparams"
    
    print(f"使用模型: {checkpoint_path}")
    
    # 加载模型
    model = TextDetector()
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)
    model.eval()
    
    # 测试单张图
    img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    img_files = list(img_dir.glob("*.png"))[:5]
    
    print(f"\n测试 {len(img_files)} 张图片")
    print("参数: prob_thresh=0.3, nms_thresh=0.5, min_area=20\n")
    
    total_boxes = 0
    for img_path in img_files:
        boxes = predict_boxes_v27(model, img_path)
        total_boxes += len(boxes)
        print(f"  {img_path.name}: {len(boxes)} 个框")
    
    avg_boxes = total_boxes / len(img_files)
    print(f"\n平均每图框数: {avg_boxes:.1f}")
    
    if avg_boxes > 0.5:
        print("\n✅ 调参成功！可以进行完整评估")
        print("\n下一步: 运行 evaluate_v27.py 计算 TP/FP")
    else:
        print("\n⚠️ 检测框太少，请降低 prob_thresh 或 min_area")

if __name__ == '__main__':
    main()