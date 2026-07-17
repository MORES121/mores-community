# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import json
import random
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_det_full import TextDetector, safe_imread
import paddle
import cv2

random.seed(42)
np.random.seed(42)

def adaptive_threshold(avg_boxes, target_boxes=1.5):
    if avg_boxes > target_boxes * 1.5:
        return 0.75, 150
    elif avg_boxes > target_boxes:
        return 0.70, 120
    elif avg_boxes < target_boxes * 0.5:
        return 0.55, 80
    elif avg_boxes < target_boxes:
        return 0.60, 100
    else:
        return 0.65, 100

def predict_boxes(model, img_path, input_size=(640, 640), prob_thresh=0.65, nms_thresh=0.6, min_area=100):
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
    binary = (heatmap > prob_thresh).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        region = heatmap[y:y+bh, x:x+bw]
        score = region.mean() if region.size > 0 else prob_thresh
        scale_x = w / input_size[0]
        scale_y = h / input_size[1]
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + bw) * scale_x)
        y2 = int((y + bh) * scale_y)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(boxes, scores, prob_thresh, nms_thresh)
    if len(indices) > 0:
        boxes = [boxes[i] for i in indices.flatten()]
    return boxes

def main():
    print("=" * 60)
    print("MORES Self-developed Engine - Adaptive Threshold v212")
    print("=" * 60)

    model = TextDetector()
    state_dict = paddle.load('/app/model.pdparams')
    model.set_state_dict(state_dict)
    model.eval()

    input_dir = Path("/saisdata/13/eval/images")
    output_file = Path("/saisresult/prediction.json")

    print("Sampling...")
    sample_files = list(input_dir.glob("*.png"))[:50]
    sample_boxes = []
    for img_path in sample_files:
        boxes = predict_boxes(model, str(img_path), prob_thresh=0.65, min_area=100)
        sample_boxes.append(len(boxes))
    avg_boxes = sum(sample_boxes) / len(sample_boxes)
    print(f"Average boxes: {avg_boxes:.2f}")

    prob_thresh, min_area = adaptive_threshold(avg_boxes)
    print(f"Adaptive params: prob_thresh={prob_thresh}, min_area={min_area}")

    print("Full inference...")
    results = {}
    total_boxes = 0
    for i, img_path in enumerate(sorted(input_dir.glob("*.png"))):
        image_id = img_path.stem
        boxes = predict_boxes(model, str(img_path), prob_thresh=prob_thresh, min_area=min_area)
        detections = [{"bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)], "text": ""} for (x1, y1, x2, y2) in boxes]
        results[image_id] = detections
        total_boxes += len(boxes)
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1} images, total boxes: {total_boxes}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Total images: {len(results)}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Average: {total_boxes/len(results):.2f}")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    main()