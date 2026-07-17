# MORES 古文字识别 Skill（AI4SS 赛道）

## 项目简介
本 Skill 面向「科艺融合 × AI4SS」赛道，提供基于 MORES 检测引擎 + 千问多模态 API 的古文字识别能力。

## 核心功能
- 自动检测图片中的文字区域
- 调用千问 API 进行古文字（甲骨文/金文等）识别与现代汉语转译
- 输出结构化 JSON 结果（含原文、现代汉语、历史解读）

## 文件说明
- `ancient_text_skill_qwen.py`：主 Skill 文件，集成检测 + 千问 API
- `qwen_test.py`：千问 API 独立测试脚本
- `visualize_result.py`：检测框可视化脚本

## 依赖环境
- Python 3.12+
- PaddlePaddle 2.6.2+
- dashscope SDK

## 快速开始
```bash
python skill/ancient_text_skill_qwen.py
```

## 成果归档
本次提交为 2026-07-17 最终版本，代码已备份至 `backup_20260717/`。
