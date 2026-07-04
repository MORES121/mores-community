"""
粤语ASR数据预处理（支持padding）
"""

import json
import os
import numpy as np
import torch
from transformers import WhisperProcessor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CantoneseASRDataset(Dataset):
    def __init__(self, data_path, processor, sample_rate=16000):
        self.processor = processor
        self.sample_rate = sample_rate
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        
        # 模拟音频（3秒）
        audio = np.random.randn(int(self.sample_rate * 3)) * 0.1
        
        # 处理音频
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # 处理文本
        labels = self.processor(
            text=text,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            'input_features': inputs.input_features.squeeze(0),
            'labels': labels,
            'text': text
        }


def collate_fn(batch):
    """自定义collate函数，处理不同长度的labels"""
    input_features = torch.stack([item['input_features'] for item in batch])
    
    # labels需要padding到相同长度
    labels = [item['labels'] for item in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    texts = [item['text'] for item in batch]
    
    return {
        'input_features': input_features,
        'labels': labels_padded,
        'texts': texts,
        'attention_mask': (labels_padded != 0).long()  # 用于后续训练
    }


def prepare_dataset(data_path, output_path=None):
    """准备数据集：统计信息"""
    print(f"📊 正在处理数据集: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   样本数量: {len(data)}")
    
    # 统计信息
    text_lengths = [len(item['text']) for item in data]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)
    min_length = min(text_lengths)
    
    print(f"\n📊 文本统计:")
    print(f"   平均长度: {avg_length:.1f} 字")
    print(f"   最大长度: {max_length} 字")
    print(f"   最小长度: {min_length} 字")
    
    # 高频字符统计
    all_text = ' '.join([item['text'] for item in data])
    char_counts = {}
    for char in all_text:
        if char.strip():
            char_counts[char] = char_counts.get(char, 0) + 1
    
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n📝 高频字符:")
    for char, count in sorted_chars:
        print(f"   '{char}': {count} 次")
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_samples': len(data),
                'avg_length': avg_length,
                'max_length': max_length,
                'min_length': min_length,
                'char_counts': dict(sorted_chars),
                'samples': data[:5]
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 预处理结果已保存到: {output_path}")
    
    return data


if __name__ == "__main__":
    print("📥 加载 Whisper 处理器...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    print("✅ 处理器加载成功")
    
    # 确保数据存在
    os.makedirs('data', exist_ok=True)
    mock_path = "data/mock_dataset.json"
    
    if not os.path.exists(mock_path):
        mock_data = [
            {"text": "今日天氣好好"},
            {"text": "我想去飲茶"},
            {"text": "唔該晒"},
            {"text": "點樣去呢度"},
            {"text": "好耐冇見"},
            {"text": "請問呢個點賣"},
            {"text": "我要兩個叉燒包"},
            {"text": "唔該俾杯水我"},
            {"text": "呢度有冇停車場"},
            {"text": "幾多錢"}
        ]
        with open(mock_path, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 模拟数据集已创建: {mock_path}")
    
    # 执行预处理
    prepare_dataset(mock_path, "data/processed_data.json")
    
    # 测试数据加载器
    print("\n🧪 测试数据加载器...")
    dataset = CantoneseASRDataset(mock_path, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"  批次 {batch_idx+1}:")
        print(f"    input_features shape: {batch['input_features'].shape}")
        print(f"    labels shape: {batch['labels'].shape}")
        print(f"    attention_mask shape: {batch['attention_mask'].shape}")
        print(f"    texts: {batch['texts']}")
        if batch_idx >= 1:
            break
    
    print("\n✅ 预处理测试完成！")
