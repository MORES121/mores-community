import sys
sys.path.insert(0, 'ancient_text')
import cv2
import easyocr
from train_det_full import TextDetector
from evaluate_v29 import predict_single_image

# 加载检测模型
model = TextDetector()
result = predict_single_image(model, 'jiaguwentuopian.jpg', threshold=0.3)
print('检测到', len(result), '个文字框')

# 读取原图
img = cv2.imread('jiaguwentuopian.jpg')

# 加载 EasyOCR 识别器
reader = easyocr.Reader(['ch_sim', 'en'])

# 对每个检测框进行识别
for i, box in enumerate(result):
    x1, y1, x2, y2 = map(int, box)
    print(f'框{i+1} 坐标: ({x1},{y1}) -> ({x2},{y2})')
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f'crop_{i}.jpg', crop)
    
    ocr_result = reader.readtext(crop)
    if ocr_result:
        text = ocr_result[0][1]
        print(f'框{i+1} 识别结果: {text}')
    else:
        print(f'框{i+1} 未识别到文字')