# 点心杯 · 粤语ASR 初赛方案

## 项目简介
基于 Whisper-small + LoRA 微调的粤语语音识别方案。

## 技术路线
- 基础模型: openai/whisper-small
- 微调方法: LoRA
- 数据集: 粤语万句多用途生活场景有声语料集

## 文件结构
dimsum_asr/
├── src/
│ ├── preprocess.py
│ ├── train_lora.py
│ └── generate_submission.py
├── output.jsonl
└── README.md

## 运行方式
```bash
pip install -r requirements.txt
python src/train_lora.py
python src/generate_submission.py
