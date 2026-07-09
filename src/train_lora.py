"""
粤语ASR · Whisper-small 全参数微调（流程验证版）
不使用 LoRA，避免兼容性问题
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np


class SimpleDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        audio = np.random.randn(16000 * 3) * 0.1
        
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        labels = self.processor(
            text=text,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            'input_features': inputs.input_features.squeeze(0),
            'labels': labels
        }


def collate_fn(batch):
    """批次整理函数"""
    input_features = torch.stack([item['input_features'] for item in batch])
    
    labels = [item['labels'] for item in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_features': input_features,
        'labels': labels_padded
    }


def train():
    print("🚀 开始全参数微调（流程验证）")
    
    # 准备数据
    data_path = "data/mock_dataset.json"
    if not os.path.exists(data_path):
        data = [
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
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    print(f"📊 数据样本数: {len(data)}")
    
    # 加载模型
    print("📥 加载模型...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 总参数量: {total_params:,}")
    
    # 准备数据集
    dataset = SimpleDataset(data, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 训练
    print("\n🏋️ 开始训练...")
    model.train()
    
    for epoch in range(2):  # 只跑2个epoch用于验证
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(
                input_features=batch['input_features'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} 完成, 平均 loss: {avg_loss:.4f}")
    
    # 保存模型
    os.makedirs("./checkpoints/full_model", exist_ok=True)
    model.save_pretrained("./checkpoints/full_model")
    processor.save_pretrained("./checkpoints/full_model")
    print("✅ 模型已保存到 ./checkpoints/full_model")
    
    return model, processor


if __name__ == "__main__":
    train()
    print("\n✅ 训练完成！")
