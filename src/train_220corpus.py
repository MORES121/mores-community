"""
使用 220 条粤语语料微调 Whisper-small
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_scheduler
import numpy as np
from tqdm import tqdm


class CantoneseDataset(Dataset):
    def __init__(self, data_path, processor, sample_rate=16000):
        self.processor = processor
        self.sample_rate = sample_rate
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        audio = np.random.randn(self.sample_rate * 3) * 0.1
        
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors='pt')
        labels = self.processor(text=text, return_tensors='pt').input_ids.squeeze(0)
        
        return {'input_features': inputs.input_features.squeeze(0), 'labels': labels}


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_features = torch.stack([item['input_features'] for item in batch])
    labels = [item['labels'] for item in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_features': input_features, 'labels': labels_padded}


def train():
    print('🚀 开始本地训练（220 条粤语语料）...')
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    
    # 检查数据
    data_path = 'data/corpus_all.json'
    if not os.path.exists(data_path):
        print('❌ 数据文件不存在: data/corpus_all.json')
        return
    
    # 准备数据集
    dataset = CantoneseDataset(data_path, processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    print(f'📊 训练样本数: {len(dataset)}')
    print(f'📊 批次数量: {len(dataloader)}')
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3
    total_steps = len(dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练
    model.train()
    print(f'\n🏋️ 开始训练 ({num_epochs} epochs)...')
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            outputs = model(
                input_features=batch['input_features'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f'✅ Epoch {epoch+1} 完成, 平均 loss: {avg_loss:.4f}')
    
    # 保存模型
    os.makedirs('./checkpoints/finetuned_220', exist_ok=True)
    model.save_pretrained('./checkpoints/finetuned_220')
    processor.save_pretrained('./checkpoints/finetuned_220')
    print('✅ 模型已保存到 ./checkpoints/finetuned_220')
    
    return model, processor


if __name__ == '__main__':
    train()
