//! MORES 社区版 - 基础使用示例
//!
//! 本示例演示如何使用 MORES 决策引擎的基本功能

use mores_community::{DecisionEngine, DecisionRequest};

fn main() {
    // 创建决策引擎实例
    let mut engine = DecisionEngine::new();
    
    // 设置置信度阈值
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
            println!("推理过程: {}", result.reasoning);
        }
        Err(e) => {
            eprintln!("决策失败: {}", e);
        }
    }
}