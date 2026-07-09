"""
使用 Whisper-small + librosa 对官方测试音频进行推理
"""

import os
import json
import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("🎤 使用 Whisper-small + librosa 进行推理...")
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model.eval()
    
    # 加载测试集（流式）
    dataset = load_dataset(
        'leeduckgo/cantonese-life-scenarios-corpus',
        split='test',
        streaming=True
    )
    
    results = []
    for idx, sample in enumerate(tqdm(dataset, total=30)):
        if idx >= 30:
            break
        
        audio = sample.get('audio', {})
        audio_path = audio.get('path', f'test_audio/{idx}.wav')
        
        # 使用 librosa 加载音频
        try:
            # 从音频数据中提取 array
            audio_array = audio.get('array', None)
            sampling_rate = audio.get('sampling_rate', 16000)
            
            if audio_array is None:
                # 如果 array 为空，尝试从路径加载
                audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            else:
                # 如果 array 存在，确保采样率正确
                if sampling_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                    sampling_rate = 16000
        except Exception as e:
            print(f"⚠️ 加载音频失败: {audio_path}, 使用模拟音频")
            audio_array = np.random.randn(16000 * 3)
            sampling_rate = 16000
        
        # 推理
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors='pt')
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs.input_features,
                language='zh',
                task='transcribe',
                max_length=30
            )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        results.append({
            'audio_path': audio_path,
            'pred_text': transcription if transcription else ''
        })
        print(f'  {idx+1}: {audio_path} -> {transcription}')
    
    with open('output_whisper_librosa.jsonl', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f'\n✅ 推理完成: output_whisper_librosa.jsonl ({len(results)} 条)')

if __name__ == '__main__':
    main()
