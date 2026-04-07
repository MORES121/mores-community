"""Python SDK 使用示例"""

from mores import DecisionEngine, DecisionRequest, Rule

def main():
    # 创建决策引擎
    engine = DecisionEngine()
    
    # 设置阈值
    engine.set_threshold(0.7)
    
    # 添加规则
    rule1 = Rule(
        name="古文字检测",
        condition="甲骨文",
        action="识别",
        weight=0.9,
    )
    rule2 = Rule(
        name="昇腾优化",
        condition="昇腾",
        action="加速",
        weight=0.8,
    )
    
    engine.add_rule(rule1)
    engine.add_rule(rule2)
    
    # 执行决策
    request = DecisionRequest(input="这是一个甲骨文识别请求")
    result = engine.decide(request)
    
    print(f"决策结果: {result.decision}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"推理: {result.reasoning}")
    print(f"触发规则: {result.rules_triggered}")

if __name__ == "__main__":
    main()