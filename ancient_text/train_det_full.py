# -*- coding: utf-8 -*-
"""
墨睿思 MORES 古文字检测微调 - 完整版 (libpng 修复版)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ========== 修复 libpng 警告 ==========
from PIL import Image

def safe_imread(path):
    """安全读取图片，解决 libpng 灰度图问题"""
    # 先用 PIL 读取，解决色彩空间问题
    try:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        # 转为 BGR 格式（因为后续用 cv2 的处理逻辑）
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        # 降级方案：用 cv2 直接读
        img = cv2.imread(str(path))
        return img
# ====================================

class Config:
    train_img_dir = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    train_label_dir = r"C:\mores_fusion\labels"
    batch_size = 8
    learning_rate = 0.001
    epochs = 50
    num_workers = 0
    save_dir = r"C:\mores_fusion\checkpoints"
    input_size = (640, 640)

class TextDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(640,640)):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.img_files = list(self.img_dir.glob("*.png"))[:500]
        self.img_w = 1655
        self.img_h = 2674
        print(f"加载 {len(self.img_files)} 张图片")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # 使用安全读取函数
        img = safe_imread(str(img_path))
        if img is None:
            return self.__getitem__((idx + 1) % len(self.img_files))
        
        h, w = img.shape[:2]
        
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        boxes.append([x1, y1, x2, y2])
        
        img, boxes = self.augment(img, boxes)
        img = cv2.resize(img, self.input_size)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        heatmap = self.generate_heatmap(boxes)
        
        return paddle.to_tensor(img, dtype='float32'), paddle.to_tensor(heatmap, dtype='float32')
    
    def augment(self, img, boxes):
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                new_x1 = img.shape[1] - x2
                new_x2 = img.shape[1] - x1
                new_boxes.append([new_x1, y1, new_x2, y2])
            boxes = new_boxes
        return img, boxes
    
    def generate_heatmap(self, boxes):
        output_size = 80
        heatmap = np.zeros((output_size, output_size), dtype=np.float32)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / self.img_w * output_size
            cy = (y1 + y2) / 2 / self.img_h * output_size
            cx_int = int(cx)
            cy_int = int(cy)
            
            if 0 <= cx_int < output_size and 0 <= cy_int < output_size:
                sigma = 2
                for i in range(max(0, cx_int-5), min(output_size, cx_int+5)):
                    for j in range(max(0, cy_int-5), min(output_size, cy_int+5)):
                        d2 = (i-cx_int)**2 + (j-cy_int)**2
                        heatmap[j, i] = max(heatmap[j, i], np.exp(-d2 / (2*sigma**2)))
        
        heatmap = cv2.resize(heatmap, (640, 640))
        heatmap = heatmap.reshape(1, 640, 640)
        return heatmap

class DBHead(nn.Layer):
    def __init__(self, in_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2D(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2D(32, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x

class TextDetector(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2D(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2D(128, 64, 3, padding=1)
        self.pool = nn.MaxPool2D(2, 2)
        self.head = DBHead(64)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        x = self.head(x)
        return x

def train():
    print("=" * 50)
    print("墨睿思 MORES 古文字检测微调 - libpng修复版 v29")
    print("=" * 50)
    
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("\n加载数据集...")
    dataset = TextDataset(cfg.train_img_dir, cfg.train_label_dir, cfg.input_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    print("创建模型...")
    model = TextDetector()
    
    optimizer = paddle.optimizer.Adam(learning_rate=cfg.learning_rate, parameters=model.parameters())
    loss_fn = nn.MSELoss()
    
    print(f"\n开始训练，共 {cfg.epochs} 轮")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.learning_rate}")
    print(f"数据批次: {len(dataloader)}")
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for imgs, heatmaps in pbar:
            preds = model(imgs)
            loss = loss_fn(preds, heatmaps)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.numpy().item()
            pbar.set_postfix({'loss': f"{loss.numpy().item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, 平均损失: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(cfg.save_dir, f"det_model_epoch{epoch+1}.pdparams")
            paddle.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}")
    
    final_path = os.path.join(cfg.save_dir, "det_model_final.pdparams")
    paddle.save(model.state_dict(), final_path)
    print(f"\n训练完成！最终模型: {final_path}")

if __name__ == '__main__':
    train()