"""
v29 模型评估脚本 - PaddlePaddle 版
评估 TP/FP/FN，诊断预测问题
"""

import os
import cv2
import numpy as np
import paddle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入训练脚本中的类和函数
import sys
sys.path.insert(0, r'C:\mores_fusion')
from train_det_full import TextDetector, Config, safe_imread

def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = TextDetector()
    state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(state_dict)
    model.eval()
    return model

def predict_single_image(model, img_path, input_size=(640, 640), threshold=0.5):
    """单张图片预测，返回检测框"""
    img = safe_imread(str(img_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, input_size)
    img_tensor = img_resized / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = paddle.to_tensor(img_tensor, dtype='float32').unsqueeze(0)
    
    with paddle.no_grad():
        heatmap = model(img_tensor).squeeze().numpy()
    
    # 从热力图提取检测框
    heatmap = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # 缩放到原图尺寸
        scale_x = w / input_size[0]
        scale_y = h / input_size[1]
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + bw) * scale_x
        y2 = (y + bh) * scale_y
        boxes.append([x1, y1, x2, y2])
    
    return boxes

def load_ground_truth(label_path, img_w, img_h):
    """加载真实标注框"""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw/2) * img_w
                    y1 = (cy - bh/2) * img_h
                    x2 = (cx + bw/2) * img_w
                    y2 = (cy + bh/2) * img_h
                    boxes.append([x1, y1, x2, y2])
    return boxes

def compute_iou(box1, box2):
    """计算 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

def evaluate_model(model, img_dir, label_dir, img_list, iou_thresh=0.5):
    """评估模型"""
    tp, fp, fn = 0, 0, 0
    all_pred_counts = []
    all_gt_counts = []
    
    for img_path in tqdm(img_list, desc="评估中"):
        # 获取原图尺寸
        img = safe_imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 预测
        pred_boxes = predict_single_image(model, img_path)
        
        # 真实标注
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth(label_path, w, h)
        
        all_pred_counts.append(len(pred_boxes))
        all_gt_counts.append(len(gt_boxes))
        
        # 匹配预测和真实框
        matched_gt = [False] * len(gt_boxes)
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt[best_idx] = True
            else:
                fp += 1
        
        fn += matched_gt.count(False)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'pred_counts': all_pred_counts,
        'gt_counts': all_gt_counts,
    }

def main():
    print("=" * 50)
    print("v29 模型评估")
    print("=" * 50)
    
    # 配置路径
    checkpoint_path = r"C:\mores_fusion\checkpoints\det_model_final.pdparams"
    img_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    label_dir = Path(r"C:\mores_fusion\labels")
    
    # 获取所有图片（取前100张评估，节省时间）
    img_files = list(img_dir.glob("*.png"))[:100]
    print(f"评估图片数: {len(img_files)}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(checkpoint_path)
    
    # 评估
    print("开始评估...")
    results = evaluate_model(model, img_dir, label_dir, img_files)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"True Positive (TP):  {results['tp']}")
    print(f"False Positive (FP): {results['fp']}")
    print(f"False Negative (FN): {results['fn']}")
    
    total_gt = results['tp'] + results['fn']
    total_pred = results['tp'] + results['fp']
    
    precision = results['tp'] / total_pred if total_pred > 0 else 0
    recall = results['tp'] / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数:            {f1:.4f}")
    
    avg_pred = np.mean(results['pred_counts'])
    avg_gt = np.mean(results['gt_counts'])
    print(f"\n平均每图预测框数: {avg_pred:.2f}")
    print(f"平均每图真实框数: {avg_gt:.2f}")
    
    print("=" * 50)
    
    if results['tp'] > 3:
        print(f"\n✅ TP 从 3 提升到 {results['tp']}！")
    else:
        print(f"\n⚠️ TP 未明显提升，需要进一步调参")

if __name__ == '__main__':
    main()