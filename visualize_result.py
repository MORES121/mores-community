import sys
sys.path.insert(0, 'ancient_text')
import cv2
import paddleocr
from train_det_full import TextDetector
from evaluate_v29 import predict_single_image

# 加载检测模型
model = TextDetector()
result = predict_single_image(model, 'jiaguwentuopian.jpg', threshold=0.3)
print('检测到', len(result), '个文字框')

# 读取原图
img = cv2.imread('jiaguwentuopian.jpg')

# 加载 OCR 识别器（use_angle_cls 在初始化时指定）
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')

# 对每个检测框进行识别
for i, box in enumerate(result):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    # 保存裁剪区域
    cv2.imwrite(f'crop_{i}.jpg', crop)
    # OCR 识别（去掉 cls 参数）
    ocr_result = ocr.ocr(crop)
    if ocr_result and ocr_result[0]:
        # 处理可能的返回值格式
        if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
            line = ocr_result[0][0]
            if isinstance(line, list) and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                print(f'框{i+1} 识别结果: {text}')
            else:
                print(f'框{i+1} 识别结果格式: {line}')
        else:
            print(f'框{i+1} 未识别到文字')