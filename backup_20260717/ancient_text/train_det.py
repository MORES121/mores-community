# -*- coding: utf-8 -*-
"""
墨睿思 MORES 古文字检测微调
基于 PaddleOCR DB 检测模型 + 比赛数据微调
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 配置参数
class Config:
    # 数据路径
    train_img_dir = r"C:\Users\klidw\Downloads\train\train\out_of_domain"
    train_label_dir = r"C:\mores_fusion\labels"
    
    # 训练参数
    batch_size = 4
    learning_rate = 0.001
    epochs = 10  # 先跑10轮测试
    num_workers = 0  # Windows下设为0避免多进程问题
    save_dir = r"C:\mores_fusion\checkpoints"

def main():
    print("=== 墨睿思 MORES 古文字检测微调 ===")
    print(f"训练图片目录: {Config.train_img_dir}")
    print(f"训练标签目录: {Config.train_label_dir}")
    
    # 检查数据
    img_files = list(Path(Config.train_img_dir).glob("*.png"))
    label_files = list(Path(Config.train_label_dir).glob("*.txt"))
    
    print(f"图片数量: {len(img_files)}")
    print(f"标签数量: {len(label_files)}")
    
    # 创建保存目录
    os.makedirs(Config.save_dir, exist_ok=True)
    
    # 初始化 PaddleOCR（加载预训练模型）
    print("加载预训练模型...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='ch')
    print("预训练模型加载完成")
    
    print("\n训练准备就绪！")
    print("注意：完整训练需要更多代码适配 PaddleOCR 的检测模型")
    print("当前为数据验证脚本，确认数据加载正常")

if __name__ == '__main__':
    main()