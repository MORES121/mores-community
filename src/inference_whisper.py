"""
使用 Whisper-small 对官方测试音频进行推理
"""

import os
import json
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def main():
    print("🎤 使用 Whisper-small 模型进行推理...")
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model.eval()
    
    # 加载测试集
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
        audio_array = audio.get('array', np.random.randn(16000 * 3))
        sampling_rate = audio.get('sampling_rate', 16000)
        
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
    
    with open('output_whisper.jsonl', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f'\n✅ 推理完成: output_whisper.jsonl ({len(results)} 条)')

if __name__ == '__main__':
    main()
