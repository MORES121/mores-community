from paddleocr import PaddleOCR
import json

# 初始化 OCR（与比赛基线一致）
ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

# 测试图片路径
img_path = r'C:\test_saisdata\13\eval\images\0032.png'

print(f"正在识别图片: {img_path}")
result = ocr.ocr(img_path, cls=True)

# 输出识别结果
print("\n===== 识别结果 =====")
if result and result[0]:
    for i, line in enumerate(result[0]):
        text = line[1][0]
        confidence = line[1][1]
        print(f"{i+1}. 文字: {text} (置信度: {confidence:.4f})")
else:
    print("未识别到任何文字")