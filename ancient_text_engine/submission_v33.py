"""
MORES v33 提交脚本 - 第四届世界科学智能大赛
古文字识别赛道 - 复赛提交
"""

import json
from pathlib import Path

# 导入优化版引擎
from mores_ocr_engine_optimized import MORESEngineOptimized

class Submission:
    """比赛提交类"""
    
    def __init__(self, model_dir: str = None):
        self.engine = MORESEngineOptimized(lang='ch')
        self.model_dir = model_dir or Path(__file__).parent / "model"
        
    def predict(self, image_path: str):
        """预测单张图片"""
        chars = self.engine.detect(str(image_path))
        
        results = []
        for char in chars:
            results.append({
                "char": char.text,
                "box": char.box,
                "confidence": float(char.confidence)
            })
        
        return results
    
    def predict_batch(self, image_dir: str):
        """批量预测"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        all_results = {}
        for img_path in image_files:
            results = self.predict(str(img_path))
            all_results[img_path.name] = results
            print(f"  {img_path.name}: {len(results)} 个字符")
        
        return all_results
    
    def save_submission(self, image_dir: str, output_file: str = "submission.json"):
        """保存提交文件"""
        print(f"\n处理目录: {image_dir}")
        results = self.predict_batch(image_dir)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n提交文件已保存: {output_file}")
        print(f"总图片数: {len(results)}")
        total_chars = sum(len(v) for v in results.values())
        print(f"总预测字符数: {total_chars}")
        
        return output_file


def main():
    print("=" * 60)
    print("MORES v33 古文字识别提交脚本")
    print("=" * 60)
    
    # 创建提交器
    sub = Submission()
    
    # 测试目录
    test_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    
    # 测试单张
    test_file = list(test_dir.glob("*.png"))[0]
    print(f"\n测试单张图片: {test_file.name}")
    result = sub.predict(str(test_file))
    print(f"预测结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 生成提交文件
    sub.save_submission(str(test_dir), "test_submission.json")
    
    print("\n✅ 提交脚本就绪！")


if __name__ == "__main__":
    main()