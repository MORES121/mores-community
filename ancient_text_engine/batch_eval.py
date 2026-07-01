"""
批量评估 MORES 引擎在训练集上的表现
"""

from mores_ocr_engine import MORESOCREngine
from pathlib import Path
import json
import time

def batch_evaluate(engine, image_dir, limit=100, save_results=True):
    """批量评估引擎性能"""
    
    img_dir = Path(image_dir)
    img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    
    if limit:
        img_files = img_files[:min(limit, len(img_files))]
    
    print(f"\n{'='*60}")
    print(f"批量评估")
    print(f"图片目录: {image_dir}")
    print(f"图片数量: {len(img_files)}")
    print(f"{'='*60}\n")
    
    results = []
    total_chars = 0
    total_time = 0
    char_count_dist = []
    
    for i, img_path in enumerate(img_files):
        start = time.time()
        chars = engine.detect(str(img_path))
        elapsed = time.time() - start
        
        num_chars = len(chars)
        total_chars += num_chars
        total_time += elapsed
        char_count_dist.append(num_chars)
        
        print(f"[{i+1:3d}/{len(img_files)}] {img_path.name[:30]:30s} → {num_chars} 个字符 ({elapsed:.2f}s)")
        
        # 保存前5个图片的详细结果
        if i < 5 and save_results and chars:
            results.append({
                "file": img_path.name,
                "char_count": num_chars,
                "chars": [{"text": c.text, "confidence": c.confidence, "box": c.box} for c in chars]
            })
    
    # 统计结果
    avg_chars = total_chars / len(img_files)
    avg_time = total_time / len(img_files)
    
    print(f"\n{'='*60}")
    print(f"评估总结")
    print(f"{'='*60}")
    print(f"总图片数:       {len(img_files)}")
    print(f"总检测字符数:   {total_chars}")
    print(f"平均每图字符数: {avg_chars:.2f}")
    print(f"最快单图:       {min(char_count_dist)} 个字符")
    print(f"最慢单图:       {max(char_count_dist)} 个字符")
    print(f"平均耗时:       {avg_time:.2f} 秒/图")
    
    # 字符数分布
    print(f"\n字符数分布:")
    for count in sorted(set(char_count_dist))[:10]:
        num = char_count_dist.count(count)
        print(f"  {count} 个字符: {num} 张图片 ({num/len(img_files)*100:.1f}%)")
    
    # 保存结果
    if save_results:
        report = {
            "config": engine.get_config(),
            "total_images": len(img_files),
            "total_chars": total_chars,
            "avg_chars_per_image": avg_chars,
            "avg_time_per_image": avg_time,
            "char_count_distribution": char_count_dist,
            "sample_results": results
        }
        
        with open("batch_eval_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: batch_eval_report.json")
    
    return {
        "total_images": len(img_files),
        "total_chars": total_chars,
        "avg_chars": avg_chars,
        "avg_time": avg_time
    }


def test_threshold_comparison():
    """测试不同阈值的影响"""
    
    print("\n" + "="*60)
    print("阈值对比测试")
    print("="*60)
    
    test_dir = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    
    for thresh in thresholds:
        print(f"\n--- 阈值: {thresh} ---")
        engine = MORESOCREngine(text_det_thresh=thresh, text_det_box_thresh=thresh, lang='ch')
        
        # 只测试前20张图，节省时间
        img_dir = Path(test_dir)
        img_files = list(img_dir.glob("*.png"))[:20]
        
        total = 0
        for img_path in img_files:
            chars = engine.detect(str(img_path))
            total += len(chars)
        
        avg = total / len(img_files)
        print(f"  20张图平均字符数: {avg:.2f}")
    
    print("\n阈值越低，检测越多，但误检可能增加")
    print("建议根据复赛数据调优")


if __name__ == "__main__":
    print("="*60)
    print("MORES 引擎批量评估")
    print("="*60)
    
    # 创建引擎（使用默认配置）
    engine = MORESOCREngine(text_det_thresh=0.3, text_det_box_thresh=0.5, lang='ch')
    
    # 评估
    test_dir = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    stats = batch_evaluate(engine, test_dir, limit=100)
    
    # 可选：对比不同阈值
    print("\n" + "="*60)
    run_comparison = input("是否运行阈值对比测试？(y/n): ").strip().lower()
    if run_comparison == 'y':
        test_threshold_comparison()
    
    print("\n✅ 批量评估完成！")