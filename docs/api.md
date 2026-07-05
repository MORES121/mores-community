# MORES API 参考

## DecisionEngine

核心决策引擎结构体。

### `new()`

创建新的决策引擎实例。

```rust
let mut engine = DecisionEngine::new();
```

set_threshold(threshold: f64)

设置置信度阈值，范围 0.0 ~ 1.0。

```rust
engine.set_threshold(0.75);
```

load_model(path: &str) -> Result<(), String>

加载模型文件。

```rust
engine.load_model("./models/default.model")?;
```

decide(request: &DecisionRequest) -> Result<DecisionResult, String>

执行决策。

```rust
let request = DecisionRequest {
    input: "测试输入".to_string(),
    context: None,
};
let result = engine.decide(&request)?;
```

---

DecisionRequest

决策请求结构体。

字段 类型 说明
input String 输入内容
context Option<serde_json::Value> 可选上下文

---

DecisionResult

决策结果结构体。

字段 类型 说明
decision String 决策结果
confidence f64 置信度 (0~1)
reasoning String 推理依据

---

昇腾适配模块（可选）

启用 ascend feature：

```toml
[dependencies]
mores-community = { version = "0.1.0", features = ["ascend"] }
```

```rust
use mores_community::ascend::AscendManager;

let mut manager = AscendManager::new(0);
manager.init()?;
```

---

错误处理

所有可能失败的接口均返回 Result<T, String>，建议使用 ? 操作符或 match 处理。

```

---

## ✅ 完成后