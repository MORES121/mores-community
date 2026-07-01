"""
MORES 引擎 - PaddleOCR 最简版本
"""

import cv2
from pathlib import Path

# 设置环境变量绕过网络检查
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

def test_paddleocr():
    print("=" * 60)
    print("测试 PaddleOCR 基础功能")
    print("=" * 60)
    
    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR 导入成功")
    except ImportError as e:
        print(f"导入失败: {e}")
        return
    
    # 最简单的初始化
    print("\n初始化 OCR...")
    try:
        ocr = PaddleOCR()
        print("初始化成功")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("\n尝试安装旧版本...")
        print("pip install paddleocr==2.7.3")
        return
    
    # 找测试图片
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    test_files = list(test_dir.glob("*.png"))
    
    if not test_files:
        print("未找到测试图片")
        return
    
    test_file = test_files[0]
    print(f"\n测试图片: {test_file.name}")
    
    # OCR
    print("识别中...")
    result = ocr.ocr(str(test_file), cls=False)
    
    if result and result[0]:
        print(f"\n检测到 {len(result[0])} 个文字块:")
        for i, line in enumerate(result[0][:5]):
            text = line[1][0]
            conf = line[1][1]
            print(f"  [{i+1}] {text} (置信度: {conf:.3f})")
    else:
        print("未检测到文字")
    
    print("\n测试完成")

if __name__ == "__main__":
    test_paddleocr()