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

## 🚀 快速体验

无需安装，一条命令启动 API 服务：

```bash
docker run -p 8000:8000 mores-core:api
启动后，打开下方网页即可体验决策引擎：

👉 MORES 决策引擎演示（需先启动服务）

或使用 curl 测试：

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"input":"这是一个测试"}'
```

---

📊 项目状态

https://gitee.com/moshi-lab/mores-community/badge/star.svg
https://img.shields.io/badge/License-Apache%202.0-blue.svg
https://img.shields.io/badge/Rust-1.70+-orange.svg
https://img.shields.io/badge/Python-3.7+-blue.svg

---

🤝 贡献

欢迎提交 Issue 和 Pull Request！

请阅读 CONTRIBUTING.md 了解详情。

---

📄 许可证

Apache 2.0 © MORES Team

```
