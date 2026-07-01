from paddleocr import PaddleOCR
import os

# 绕过网络检查
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

print("初始化 OCR...")
ocr = PaddleOCR()  # 使用全部默认参数

img_path = r'C:\test_saisdata\13\eval\images\0032.png'

print(f"正在识别图片: {img_path}")
result = ocr.predict(img_path)  # 使用新版 API

print("\n===== 识别结果 =====")
if result and len(result) > 0:
    # 新版返回格式
    first_result = result[0]
    if 'rec_texts' in first_result:
        texts = first_result['rec_texts']
        scores = first_result['rec_scores']
        for i, (text, score) in enumerate(zip(texts, scores)):
            print(f"{i+1}. 文字: {text} (置信度: {score:.4f})")
    else:
        print("返回格式:", first_result.keys() if hasattr(first_result, 'keys') else type(first_result))
else:
    print("未识别到文字")

print("\n✅ 测试完成")