"""
粤语 TTS 音频合成器
使用 edge-tts 将文本合成粤语音频
"""

import asyncio
import json
import os
import edge_tts

# 粤语发音人（香港粤语）
CANTONESE_VOICE = "zh-HK-HiuGaaiNeural"

async def synthesize_single(text: str, output_path: str):
    """合成单个音频"""
    communicate = edge_tts.Communicate(text, CANTONESE_VOICE)
    await communicate.save(output_path)
    return output_path

async def synthesize_batch(texts: list, output_dir: str = "data/audio"):
    """批量合成音频"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for idx, item in enumerate(texts[:50]):  # 先合成前50条
        text = item.get("text", "")
        if not text:
            continue
        
        audio_file = f"{output_dir}/audio_{idx+1:03d}.mp3"
        
        try:
            await synthesize_single(text, audio_file)
            results.append({
                "text": text,
                "audio": audio_file,
                "status": "success"
            })
            print(f"  ✅ [{idx+1}] {text[:20]}... -> {audio_file}")
        except Exception as e:
            print(f"  ❌ [{idx+1}] 合成失败: {e}")
            results.append({
                "text": text,
                "audio": None,
                "status": "failed",
                "error": str(e)
            })
    
    return results

async def main():
    print("=" * 50)
    print("🎤 粤语 TTS 音频合成器")
    print("=" * 50)
    
    # 加载语料
    with open("data/corpus_all.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n📊 总语料: {len(data)} 条")
    print(f"🎤 合成前 50 条...")
    print(f"🗣️ 发音人: {CANTONESE_VOICE}")
    
    results = await synthesize_batch(data[:50])
    
    # 统计
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    
    print(f"\n📊 合成完成:")
    print(f"  成功: {success} 条")
    print(f"  失败: {failed} 条")
    
    # 保存结果
    with open("data/tts_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存到: data/tts_results.json")

if __name__ == "__main__":
    asyncio.run(main())
