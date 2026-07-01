# RNA 蛋白结构预测 — 墨睿思 MORES 最优得分方案

## 比赛成绩
- 最优提交分数：0.188
- 系统能力：全自动 / 可解释 / 低算力

## 核心文件
- `mores_light.pt`：轻量模型（2.5万参数）
- `generate_report.py`：全自动靶点评估报告
- `mores_local_infer.py`：本地离线推理
- `mores_batch_rank.py`：批量靶点排序

## 运行环境
- Python 3.8+
- 依赖：`torch`, `numpy`, `json`

## 快速运行
```bash
python mores_local_infer.py
MIT License
