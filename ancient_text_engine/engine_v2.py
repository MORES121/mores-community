"""
MORES 引擎 - PaddleOCR 新版 API
"""

import cv2
from pathlib import Path
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

def test_paddleocr():
    print("=" * 60)
    print("MORES 引擎 - PaddleOCR 测试")
    print("=" * 60)
    
    from paddleocr import PaddleOCR
    
    print("\n初始化 OCR...")
    ocr = PaddleOCR()
    print("初始化成功")
    
    # 测试图片
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))
    
    if not test_files:
        print("未找到测试图片")
        return
    
    test_file = test_files[0]
    print(f"\n测试图片: {test_file.name}")
    
    # 使用新版 API
    print("识别中...")
    result = ocr.predict(str(test_file))
    
    if result and len(result) > 0:
        # 新版返回格式需要解析
        print(f"\n检测结果:")
        
        # 尝试不同的返回格式
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            
            # 检查是否有检测到的文本块
            if 'rec_texts' in first_result:
                texts = first_result['rec_texts']
                scores = first_result['rec_scores']
                boxes = first_result['boxes']
                
                print(f"检测到 {len(texts)} 个文字块:")
                for i, (text, score) in enumerate(zip(texts[:5], scores[:5])):
                    print(f"  [{i+1}] {text} (置信度: {score:.3f})")
            else:
                print("返回格式:", type(first_result))
                print(first_result)
        else:
            print("result:", result)
    else:
        print("未检测到文字")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_paddleocr()