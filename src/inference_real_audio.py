"""
使用真实音频文件进行 Whisper 推理
"""

import os
import json
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

def main():
    print("🎤 使用真实音频进行 Whisper 推理...")
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model.eval()
    
    # 音频目录
    audio_dir = "data/test_audio/test_audio"
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')][:30]
    
    print(f"📊 找到 {len(audio_files)} 个音频文件，取前30个进行推理")
    
    # 加载模板获取正确的 audio_path
    with open('official_files/template.jsonl', 'r', encoding='utf-8') as f:
        template_lines = f.readlines()
    
    results = []
    for idx, audio_file in enumerate(tqdm(audio_files[:30])):
        audio_path = os.path.join(audio_dir, audio_file)
        template_data = json.loads(template_lines[idx].strip())
        
        # 加载音频
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"⚠️ 加载失败: {audio_file}, 跳过")
            continue
        
        # 推理
        inputs = processor(audio_array, sampling_rate=sr, return_tensors='pt')
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs.input_features,
                language='zh',
                task='transcribe',
                max_length=30
            )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        results.append({
            'audio_path': template_data.get('audio_path', audio_file),
            'pred_text': transcription if transcription else ''
        })
        print(f"  {idx+1}: {audio_file} -> {transcription}")
    
    with open('output_real.jsonl', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 推理完成: output_real.jsonl ({len(results)} 条)")

if __name__ == '__main__':
    main()
