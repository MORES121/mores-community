#!/usr/bin/env python3
"""
粤语ASR推理脚本 - 点心杯离线评测入口
"""

import os
import json
import argparse
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='粤语ASR离线推理')
    parser.add_argument('--audio_dir', type=str, required=True, help='音频目录')
    parser.add_argument('--output_jsonl', type=str, required=True, help='输出jsonl路径')
    parser.add_argument('--test_list', type=str, required=True, help='测试集CSV列表')
    return parser.parse_args()


def load_model(model_dir='.'):
    """加载模型"""
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)
    model.eval()
    return model, processor


def transcribe_audio(model, processor, audio_path, sr=16000):
    """转写单个音频"""
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        inputs = processor(audio, sampling_rate=sr, return_tensors='pt')
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs.input_features,
                language='zh',
                task='transcribe',
                max_length=30
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    args = parse_args()
    print(f"🎤 开始推理...")
    print(f"   音频目录: {args.audio_dir}")
    print(f"   输出文件: {args.output_jsonl}")
    print(f"   测试列表: {args.test_list}")
    
    # 加载模型（从当前目录）
    model, processor = load_model('.')
    
    # 读取测试列表
    with open(args.test_list, 'r') as f:
        audio_files = [line.strip() for line in f.readlines() if line.strip()]
    
    results = []
    for audio_file in audio_files[:30]:
        audio_path = os.path.join(args.audio_dir, audio_file)
        text = transcribe_audio(model, processor, audio_path)
        results.append({
            'audio_path': audio_file,
            'pred_text': text
        })
        print(f"  {audio_file} -> {text}")
    
    # 输出结果
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 推理完成，结果已保存到 {args.output_jsonl}")


if __name__ == '__main__':
    main()
