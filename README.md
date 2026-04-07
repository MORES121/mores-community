# MORES 社区版
# MORES 社区版

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Gitee](https://gitee.com/moshi-lab/mores-community/badge/star.svg)](https://gitee.com/moshi-lab/mores-community)

> 可控可解释的决策引擎 · 古文字识别 · 昇腾适配 · 国产可信AI底层框架

MORES 社区版是墨睿思核心引擎的**开源通用版本**，提供决策框架、古文字识别接口和昇腾适配示例。

---

## ✨ 特性


- 🧠 **可解释决策引擎** — 决策过程可追溯、可理解
- 🔤 **古文字识别** — 甲骨文、金文等古文字检测与识别
- 🚀 **昇腾适配示例** — 华为昇腾 NPU 适配参考
- 🇨🇳 **国产 AI 生态** — 优先支持信创环境

---

## 📦 快速开始

### 添加依赖

```toml
[dependencies]
mores-community = "0.1.0"

基础使用示例

use mores_community::{DecisionEngine, DecisionRequest};

fn main() {
    let mut engine = DecisionEngine::new();
    engine.set_threshold(0.75);

    let request = DecisionRequest {
        input: "测试输入".to_string(),
        context: None,
    };

    match engine.decide(&request) {
        Ok(result) => println!("决策结果: {}", result),
        Err(e) => eprintln!("错误: {}", e),
    }
}

更多示例example/basic_usage.rs


---

## ✅ 提交信息填
