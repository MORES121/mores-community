---
name: MORES 古文字智能识别
description: 面向数字人文的甲骨文/金文检测与识别工具，集成MORES检测引擎与千问多模态大模型，实现从图像到现代汉语的端到端解读。由墨睿思（东莞市轩钰希智能科技有限公司）研发。
license: Apache-2.0
languages:
  - Python
tags:
  - 数字人文
  - 古文字识别
  - 甲骨文
  - AI4SS
  - 多模态
  - 墨睿思
---

# MORES 古文字智能识别

> **研发机构**：墨睿思 · 东莞市轩钰希智能科技有限公司  
> **官网**：[https://kairosmores.cn](https://kairosmores.cn)

## 📖 简介

本 Skill 由 **墨睿思（东莞市轩钰希智能科技有限公司）** 研发，面向「科艺融合 × AI4SS」赛道，为历史、考古、文献学等研究者提供一套开箱即用的古文字识别工具。

它融合了 **MORES 自主检测引擎** 与 **千问多模态大模型**，能够从古籍、拓片、文物照片中自动检测文字区域，并完成从古文字（甲骨文/金文/篆书）到现代汉语的识别与解读。

**MORES 核心资产**：
- 6 项发明专利（推理混淆/知识蒸馏防御/因果推理/反事实推理等）
- 2 项软件著作权
- 8 个赛道全部跑通，一套引擎打穿

## 🎯 核心功能

| 功能模块 | 能力描述 |
|---|---|
| **智能检测** | 基于 MORES 引擎定位图像中的文字区域，输出精确坐标框 |
| **多模态识别** | 调用千问大模型对检测区域进行古文字识别，支持甲骨文、金文等 |
| **现代转译** | 自动输出对应的现代汉字及历史背景解读 |
| **结构化输出** | 返回 JSON 格式结果，包含原文、现代汉语、解读三个维度 |

## 🛠️ 技术架构

```text
用户上传图片
    ↓
MORES 检测引擎（PaddlePaddle）
    ↓ （裁剪文字区域）
千问多模态 API（Qwen-VL-Plus）
    ↓ （识别+解读）
输出结构化 JSON

📦 依赖环境

· Python 3.12+
· PaddlePaddle 2.6.2+
· dashscope SDK
· OpenCV
· 网络环境（用于调用千问 API）

🚀 快速开始

1. 安装依赖

```bash
pip install paddlepaddle dashscope opencv-python
```

2. 配置 API Key

```python
# 在 ancient_text_skill_qwen.py 中设置
DASHSCOPE_API_KEY = "your-api-key"
```

3. 运行识别

```bash
python skill/ancient_text_skill_qwen.py
```

📝 输入输出示例

输入

一张甲骨文拓片图片（jiaguwentuopian.jpg）

输出（JSON 结构化结果）

```json
{
  "boxes": [[0.0, 0.0, 268.0, 602.0]],
  "results": [{
    "box": [0.0, 0.0, 268.0, 602.0],
    "text": "【识别结果】\n王氏田\n于王田\n\n【现代汉字】\n王家的田地\n在王的田地里\n\n【解读】\n这是一段关于土地归属的记载..."
  }],
  "total_boxes": 1
}
```

📂 项目结构

```
mores-community/
├── ancient_text/               # 检测引擎核心代码
├── skill/                      # Skill 主文件
│   └── ancient_text_skill_qwen.py
├── docs/README.md              # 项目说明
└── backup_20260717/            # 完整备份
```

📊 性能表现

测试场景 检测耗时 识别耗时 总耗时
甲骨文拓片 (908x1235) 0.8s 2.1s 2.9s

🌟 应用场景

· 数字人文：古籍数字化、碑帖整理
· 考古研究：出土文献快速识读
· 文博保护：文物档案智能化处理
· 教育教学：古文字教学辅助工具

🔗 相关资源

· 开源仓库：Gitee | GitHub
· 官网：https://kairosmores.cn
· 研发机构：墨睿思 · 东莞市轩钰希智能科技有限公司

📄 许可证

Apache-2.0

---

MORES 不是理论，是实战。8 个赛道全部跑通，一套引擎打穿