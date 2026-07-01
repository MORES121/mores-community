# MORES 快速入门

5 分钟上手 MORES 决策引擎。

---

## 环境要求

- Rust 1.70 或更高版本
- Cargo 包管理器

---

## 第一步：创建新项目

```bash
cargo new my_mores_app
cd my_mores_app

##第二步：添加依赖

编辑 Cargo.toml，添加：

```toml
[dependencies]
mores-community = "0.1.0"
```

##第三步：编写代码

编辑 src/main.rs：

```rust
use mores_community::{DecisionEngine, DecisionRequest};

fn main() {
    // 创建决策引擎
    let mut engine = DecisionEngine::new();
    
    // 设置置信度阈值（0-1）
    engine.set_threshold(0.75);
    
    // 创建决策请求
    let request = DecisionRequest {
        input: "这是一个测试输入".to_string(),
        context: None,
    };
    
    // 执行决策
    match engine.decide(&request) {
        Ok(result) => {
            println!("决策结果: {}", result.decision);
            println!("置信度: {:.2}%", result.confidence * 100.0);
            println!("推理依据: {}", result.reasoning);
        }
        Err(e) => {
            eprintln!("决策失败: {}", e);
        }
    }
}
```

---
##第四步：运行

```bash
cargo run
```

---
##下一步

· 查看 API 文档
· 浏览 更多示例
· 了解 贡献指南

---

遇到问题？

· 提交 Issue
· 查看 FAQ（编写中）

```

---

## ✅ 完成后