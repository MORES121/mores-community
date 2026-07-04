#!/usr/bin/env python3
"""
code/main.py
MORES 叉车冠军方案 · 主入口
树根杯 · 具身智能平衡重叉车挑战赛
"""

import argparse
import sys
import os
import time
import json
import yaml
import numpy as np
from typing import Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.decision_engine import MORESDecisionEngine
from code.forklift_env import ForkliftEnv


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MORES 叉车冠军方案 - 树根杯挑战赛"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["train", "eval", "demo", "check"],
        default="demo",
        help="运行模式: train(训练), eval(评估), demo(演示), check(检查)"
    )

    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["isaac_sim", "mujoco", "simulated"],
        default="simulated",
        help="仿真后端 (默认: simulated)"
    )

    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=200,
        help="运行步数 (默认: 100)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="启用渲染"
    )

    parser.add_argument(
        "--export-log",
        type=str,
        default=None,
        help="导出决策日志到指定文件"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✅ 配置文件加载成功: {config_path}")
    return config


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🏗️  MORES 叉车冠军方案 · 树根杯挑战赛              ║
    ║                                                           ║
    ║     🧠 因果推理  |  🔮 反事实推演  |  👁️ 多模态感知   ║
    ║                                                           ║
    ║     大公子莫孜轩 · 末将墨睿思                            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status(status: str, step: int, action: tuple, explanation: str):
    """打印状态信息"""
    steering, speed, fork = action
    print(f"\r  步骤 {step:4d} | 转向 {steering:+6.1f}° | "
          f"速度 {speed:5.2f}m/s | 叉齿 {fork:5.2f}m | "
          f"{status:10s} | {explanation[:30]:30s}", end="", flush=True)


def run_demo(config: Dict, args) -> Dict:
    """演示模式"""
    print("\n🎬 演示模式启动")

    # 创建决策引擎
    engine = MORESDecisionEngine(config)

    # 创建环境
    env_config = config.get('simulation', {})
    env_config['max_steps'] = args.steps
    env = ForkliftEnv(env_config, backend=args.backend)

    # 重置
    obs = env.reset()
    total_reward = 0.0
    all_decisions = []

    print(f"\n  目标位置: {env.target_position}")
    print(f"  目标方向: {env.target_orientation:.1f}°")
    print("\n  执行任务...")

    # 模拟RGB图像和激光雷达
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lidar_points = np.random.randn(100, 3) * 0.5 + np.array([2.0, 0.0, 0.5])

    start_time = time.time()

    for step in range(args.steps):
        # 感知
        perception_result = engine.perceive(rgb_image, lidar_points)

        # 因果推理
        causal_result = engine.reason_causal(perception_result)

        # 反事实推演
        alternatives = engine.plan_alternatives(engine.current_state)

        # 决策
        decision = engine.decide(perception_result, alternatives)

        # 执行动作
        action = decision.action
        # 限幅
        steering, speed, fork_height = action
        steering = np.clip(steering, -30.0, 30.0)
        speed = np.clip(speed, -2.2, 2.2)
        fork_height = np.clip(fork_height, 0.0, 2.0)
        action = (steering, speed, fork_height)
        obs, reward, done, info = env.step(action)

        total_reward += reward

        # 记录
        all_decisions.append({
            'step': step + 1,
            'action': action,
            'reward': reward,
            'info': info,
            'explanation': decision.explanation,
            'causal_path': decision.causal_path,
        })

        # 更新引擎状态
        engine.current_state = env.get_state()

        # 打印状态
        status_text = "✅ 成功" if done and info.get('stability', 0) > 0.8 else "🔄 运行中"
        print_status(status_text, step + 1, action, decision.explanation[:30])

        if done:
            break

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 60)
    print("📊 任务完成报告")
    print("=" * 60)
    print(f"  总步数:    {step + 1}")
    print(f"  总奖励:    {total_reward:.2f}")
    print(f"  耗时:      {elapsed:.2f}s")
    print(f"  状态:      {'✅ 成功' if done and info.get('stability', 0) > 0.8 else '❌ 未完成'}")
    print(f"  稳定性:    {info.get('stability', 0):.2f}")
    print(f"  距离目标:  {info.get('distance_to_target', 0):.3f}m")

    # 显示最优决策
    if engine.decision_history:
        latest = engine.decision_history[-1]
        print(f"\n  最优决策:")
        print(f"    动作: {latest.action}")
        print(f"    解释: {latest.explanation}")
        if latest.alternatives:
            print(f"    候选方案数: {len(latest.alternatives)}")

    # 导出日志
    if args.export_log:
        engine.export_decision_log(args.export_log)

    print("=" * 60)

    return {
        'steps': step + 1,
        'total_reward': total_reward,
        'elapsed': elapsed,
        'success': done and info.get('stability', 0) > 0.8,
        'decisions': all_decisions,
    }


def run_check(config: Dict, args) -> Dict:
    """检查模式：验证所有模块是否正常工作"""
    print("\n🔍 系统检查模式")

    results = {}
    checks = [
        ("causal_graph", "因果图"),
        ("counterfactual", "反事实引擎"),
        ("perception", "感知模块"),
        ("controller", "控制器"),
        ("decision_engine", "决策引擎"),
        ("forklift_env", "环境封装"),
    ]

    try:
        engine = MORESDecisionEngine(config)
        results['engine'] = "✅ 决策引擎初始化成功"
    except Exception as e:
        results['engine'] = f"❌ 决策引擎初始化失败: {e}"

    try:
        env = ForkliftEnv(config.get('simulation', {}), backend=args.backend)
        results['env'] = "✅ 环境初始化成功"
    except Exception as e:
        results['env'] = f"❌ 环境初始化失败: {e}"

    try:
        obs = env.reset()
        results['reset'] = f"✅ 环境重置成功, obs维度: {len(obs)}"
    except Exception as e:
        results['reset'] = f"❌ 环境重置失败: {e}"

    try:
        action = (0.0, 0.0, 0.0)
        obs, reward, done, info = env.step(action)
        results['step'] = f"✅ 环境步进成功, reward={reward:.2f}"
    except Exception as e:
        results['step'] = f"❌ 环境步进失败: {e}"

    try:
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        perception = engine.perceive(rgb_image, None)
        results['perception'] = f"✅ 感知成功, 目标位置: {perception.target_position}"
    except Exception as e:
        results['perception'] = f"❌ 感知失败: {e}"

    try:
        alt = engine.plan_alternatives(engine.current_state)
        results['alternatives'] = f"✅ 反事实推演成功, 候选数: {len(alt)}"
    except Exception as e:
        results['alternatives'] = f"❌ 反事实推演失败: {e}"

    print("\n  检查结果:")
    for key, value in results.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 60)
    all_ok = all("✅" in v for v in results.values())
    print(f"  {'✅ 所有模块检查通过!' if all_ok else '❌ 部分模块检查失败'}")
    print("=" * 60)

    return {'passed': all_ok, 'results': results}


def main():
    """主函数"""
    args = parse_args()
    print_banner()

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 根据模式运行
    if args.mode == "check":
        result = run_check(config, args)
    elif args.mode == "demo":
        result = run_demo(config, args)
    elif args.mode == "eval":
        # 评估模式 - 多次运行取平均
        print("\n📊 评估模式 (10次运行)")
        results = []
        for i in range(10):
            print(f"\n  运行 {i+1}/10")
            result = run_demo(config, args)
            results.append(result)
        # 汇总
        success_count = sum(1 for r in results if r.get('success', False))
        avg_reward = np.mean([r.get('total_reward', 0) for r in results])
        print("\n" + "=" * 60)
        print("📊 评估汇总")
        print("=" * 60)
        print(f"  成功率: {success_count}/10 ({success_count*10:.0f}%)")
        print(f"  平均奖励: {avg_reward:.2f}")
        print("=" * 60)
        result = {'success_rate': success_count / 10, 'avg_reward': avg_reward}
    else:
        print(f"⚠️ 未知模式: {args.mode}")
        result = {}

    return result


if __name__ == "__main__":
    main()
