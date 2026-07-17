from paddleocr import PaddleOCR
import json
from pathlib import Path
import os

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

def main():
    print("=" * 60)
    print("MORES PaddleOCR 提交脚本（简化版）")
    print("=" * 60)
    
    # 最简初始化，不加任何参数
    print("\n初始化 OCR...")
    ocr = PaddleOCR()
    
    # 测试图片路径
    test_dir = Path(r"C:\test_saisdata\13\eval\images")
    img_files = list(test_dir.glob("*.png"))
    
    print(f"找到 {len(img_files)} 张图片")
    
    results = {}
    total_boxes = 0
    
    for i, img_path in enumerate(img_files):
        result = ocr.ocr(str(img_path))
        
        boxes = []
        if result and result[0]:
            for line in result[0]:
                points = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                x1 = int(min(p[0] for p in points))
                y1 = int(min(p[1] for p in points))
                x2 = int(max(p[0] for p in points))
                y2 = int(max(p[1] for p in points))
                
                boxes.append({
                    "char": text,
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence
                })
        
        results[img_path.name] = boxes
        total_boxes += len(boxes)
        
        print(f"  {img_path.name}: {len(boxes)} 个字符")
    
    # 保存结果
    output_file = "test_prediction.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 提交文件已生成: {output_file}")
    print(f"  总图片数: {len(img_files)}")
    print(f"  总预测字符数: {total_boxes}")
    if len(img_files) > 0:
        print(f"  平均每图: {total_boxes/len(img_files):.2f}")

if __name__ == "__main__":
    main()