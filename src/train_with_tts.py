"""
使用 TTS 合成音频训练 Whisper 模型
"""

import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_scheduler
import numpy as np
from tqdm import tqdm


class TTSDataset(Dataset):
    def __init__(self, data_path, audio_dir, processor, sample_rate=16000):
        self.processor = processor
        self.sample_rate = sample_rate
        
        # 加载 TTS 结果
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.audio_dir = audio_dir
        # 只保留成功合成的样本
        self.data = [item for item in self.data if item.get('status') == 'success']
        
        print(f"📊 加载 {len(self.data)} 条音视频对")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        audio_path = item.get('audio', '')
        
        # 加载音频
        try:
            waveform, sr = torchaudio.load(audio_path)
            # 重采样到 16000 Hz
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            audio = waveform.squeeze().numpy()
        except Exception as e:
            print(f"⚠️ 加载音频失败: {audio_path}, 使用模拟音频")
            audio = np.random.randn(self.sample_rate * 3) * 0.1
        
        # 处理音频
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors='pt'
        )
        
        # 处理文本
        labels = self.processor(
            text=text,
            return_tensors='pt'
        ).input_ids.squeeze(0)
        
        return {
            'input_features': inputs.input_features.squeeze(0),
            'labels': labels
        }


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_features = torch.stack([item['input_features'] for item in batch])
    labels = [item['labels'] for item in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_features': input_features, 'labels': labels_padded}


def train():
    print("=" * 50)
    print("🚀 使用 TTS 音频训练 Whisper 模型")
    print("=" * 50)
    
    # 1. 加载模型
    print("\n📥 加载模型...")
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    
    # 2. 准备数据集
    print("\n📊 准备数据集...")
    dataset = TTSDataset(
        data_path='data/tts_results.json',
        audio_dir='data/audio',
        processor=processor
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"📊 批次数量: {len(dataloader)}")
    
    # 3. 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3
    total_steps = len(dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 4. 训练
    print(f"\n🏋️ 开始训练 ({num_epochs} epochs)...")
    model.train()
    
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
    
    # 5. 保存模型
    os.makedirs('./checkpoints/tts_finetuned', exist_ok=True)
    model.save_pretrained('./checkpoints/tts_finetuned')
    processor.save_pretrained('./checkpoints/tts_finetuned')
    print("\n✅ 模型已保存到 ./checkpoints/tts_finetuned")
    
    return model, processor


if __name__ == '__main__':
    train()
