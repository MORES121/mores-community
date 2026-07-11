# 粤语语音识别 - 点心杯复赛提交

## 模型信息
- 基础模型: openai/whisper-small
- 微调方法: LoRA
- 微调数据: 粤语万句多用途生活场景有声语料集

## 文件说明
| 文件 | 说明 |
|------|------|
| predict.py | 推理入口脚本 |
| requirements.txt | 依赖清单 |
| README.md | 说明文档 |
| model.safetensors | 模型权重 |
| config.json | 模型配置 |

## 运行方式
```bash
python predict.py --audio_dir /path/to/audio --output_jsonl output.jsonl --test_list test.csv
开源仓库
https://gitee.com/moshi-lab/mores-community/tree/feature/dimsum-asr
