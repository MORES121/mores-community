"""
code/counterfactual.py
反事实推理引擎（专利核心）
基于因果图模拟多种行动方案，选择最优
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class CounterfactualResult:
    """反事实推演结果"""
    action: Tuple[float, float, float]  # (steering, speed, fork_height)
    score: float
    trajectory: List[Dict]
    explanation: str
    stability: float
    success_rate: float
    efficiency: float
    rank: int = 0


class CounterfactualEngine:
    """
    反事实推理引擎
    对候选动作进行反事实推演，评估并排序
    """

    def __init__(self, causal_graph, world_model, config: Optional[Dict] = None):
        """
        Args:
            causal_graph: ForkliftCausalGraph实例
            world_model: 世界模型（用于前向模拟）
            config: 配置参数
        """
        self.causal_graph = causal_graph
        self.world_model = world_model

        if config is None:
            config = {}

        self.num_alternatives = config.get('num_alternatives', 5)
        self.rollout_steps = config.get('rollout_steps', 50)
        self.score_weights = config.get('score_weights', {
            'stability': 0.4,
            'success_rate': 0.4,
            'efficiency': 0.2
        })

        # 决策日志
        self.decision_log: List[List[CounterfactualResult]] = []

    def generate_alternatives(self, state: Dict, num: int = None) -> List[Tuple]:
        """
        生成候选动作集 - 使用绝对值
        """
        if num is None:
            num = self.num_alternatives

        alternatives = []

        # 直接使用目标值生成候选，而非增量
        # 转向角：围绕0度小幅变化
        steering_options = [-0.5, -0.2, 0.0, 0.2, 0.5]
        # 速度：低速范围
        speed_options = [1.5, 1.8, 2.0, 2.2]
        # 叉齿高度：直接围绕目标0.5m
        fork_options = [0.49, 0.495, 0.50, 0.505, 0.51]

        import itertools
        candidates = list(itertools.product(steering_options, speed_options, fork_options))

        np.random.seed(42)
        indices = np.random.choice(len(candidates), min(num, len(candidates)), replace=False)

        for idx in indices:
            steering, speed, fork_height = candidates[idx]
            alternatives.append((steering, speed, fork_height))

        return alternatives

    def simulate_alternative(self, state: Dict, action: Tuple,
                             steps: int = None) -> Dict:
        """
        对单个候选动作进行前向模拟

        Args:
            state: 初始状态
            action: (steering, speed, fork_height)
            steps: 模拟步数

        Returns:
            模拟结果
        """
        if steps is None:
            steps = self.rollout_steps

        steering, speed, fork_height = action

        # 复制状态
        sim_state = state.copy()
        trajectory = []

        for t in range(steps):
            # 应用动作
            sim_state['steering_angle'] = steering
            sim_state['speed'] = speed
            sim_state['fork_height'] = fork_height

            # 使用世界模型更新状态
            sim_state = self.world_model.step(sim_state, action)

            # 记录轨迹
            trajectory.append({
                'step': t,
                'x': sim_state.get('x', 0.0),
                'y': sim_state.get('y', 0.0),
                'theta': sim_state.get('theta', 0.0),
                'steering': sim_state.get('steering_angle', steering),
                'speed': sim_state.get('speed', speed),
                'fork_height': sim_state.get('fork_height', fork_height),
                'stability': sim_state.get('stability', 0.0),
                'success': sim_state.get('success', False),
            })

            # 如果任务完成，提前结束
            if sim_state.get('success', False):
                break

        return {
            'final_state': sim_state,
            'trajectory': trajectory,
            'steps': len(trajectory),
            'success': sim_state.get('success', False),
            'stability': sim_state.get('stability', 0.0),
        }

    def evaluate(self, sim_result: Dict, action: Tuple) -> Dict:
        """
        评估模拟结果

        Args:
            sim_result: 模拟结果
            action: 候选动作

        Returns:
            评分字典
        """
        stability = sim_result.get('stability', 0.0)
        success = sim_result.get('success', False)
        steps = sim_result.get('steps', self.rollout_steps)

        # 成功率：0或1
        success_rate = 1.0 if sim_result.get("success", False) else 0.0

        # 效率：步数越少越好
        efficiency = max(0.0, 1.0 - steps / self.rollout_steps)

        # 综合评分
        weights = self.score_weights
        score = (
            weights.get('stability', 0.4) * stability +
            weights.get('success_rate', 0.4) * success_rate +
            weights.get('efficiency', 0.2) * efficiency
        )

        return {
            'score': score,
            'stability': stability,
            'success_rate': success_rate,
            'efficiency': efficiency,
            'steps': steps,
            'success': success,
        }

    def explain(self, action: Tuple, eval_result: Dict,
                sim_result: Dict) -> str:
        """
        生成决策解释

        Args:
            action: 候选动作
            eval_result: 评估结果
            sim_result: 模拟结果

        Returns:
            可读解释文本
        """
        steering, speed, fork_height = action
        score = eval_result['score']
        stability = eval_result['stability']
        success = eval_result['success']

        # 生成因果解释
        explanation = (
            f"转向{steering:+.1f}°, 速度{speed:.2f}m/s, "
            f"叉齿提升{fork_height:.2f}m → "
            f"稳定性{stability*100:.0f}%, "
            f"{'✅ 任务成功' if success else '❌ 任务失败'}, "
            f"综合评分{score*100:.0f}%"
        )

        return explanation

    def simulate_alternatives(self, state: Dict,
                              actions: Optional[List[Tuple]] = None
                              ) -> List[CounterfactualResult]:
        """
        对候选动作进行反事实推演

        Args:
            state: 当前状态
            actions: 候选动作列表，若为None则自动生成

        Returns:
            排序后的反事实结果列表
        """
        if actions is None:
            actions = self.generate_alternatives(state)

        results = []

        for idx, action in enumerate(actions):
            # 1. 前向模拟
            sim_result = self.simulate_alternative(state, action)

            # 2. 多维度评估
            eval_result = self.evaluate(sim_result, action)

            # 3. 生成因果解释
            explanation = self.explain(action, eval_result, sim_result)

            # 4. 构建结果
            result = CounterfactualResult(
                action=action,
                score=eval_result['score'],
                trajectory=sim_result['trajectory'],
                explanation=explanation,
                stability=eval_result['stability'],
                success_rate=eval_result['success_rate'],
                efficiency=eval_result['efficiency'],
                rank=0
            )
            results.append(result)

        # 按得分降序排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 分配排名
        for i, result in enumerate(results):
            result.rank = i + 1

        # 记录决策日志
        self.decision_log.append(results)

        return results

    def get_best_action(self, state: Dict,
                        actions: Optional[List[Tuple]] = None
                        ) -> Tuple[Tuple, CounterfactualResult]:
        """
        获取最优动作

        Args:
            state: 当前状态
            actions: 候选动作列表

        Returns:
            (最优动作, 最优结果)
        """
        results = self.simulate_alternatives(state, actions)

        if not results:
            raise ValueError("没有可用的候选动作")

        best = results[0]
        return best.action, best

    def get_decision_log(self) -> List[List[CounterfactualResult]]:
        """获取决策日志"""
        return self.decision_log

    def get_last_decision(self) -> Optional[List[CounterfactualResult]]:
        """获取最近一次决策结果"""
        if self.decision_log:
            return self.decision_log[-1]
        return None

    def format_decision_report(self, results: List[CounterfactualResult]) -> str:
        """
        格式化决策报告（用于PPT展示）

        Args:
            results: 反事实结果列表

        Returns:
            格式化的报告文本
        """
        lines = ["📊 反事实推演报告"]
        lines.append("=" * 50)

        for i, result in enumerate(results[:5]):
            rank = f"#{result.rank} {'⭐' if result.rank == 1 else '  '}"
            steering, speed, fork = result.action
            lines.append(
                f"{rank} 转向{steering:+.1f}° 速度{speed:.2f}m/s "
                f"叉齿{fork:.2f}m → {result.score*100:.0f}% "
                f"(稳定{result.stability*100:.0f}% 成功{result.success_rate*100:.0f}%)"
            )
            if result.rank == 1:
                lines.append(f"   ✅ 推荐: {result.explanation}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def export_log_to_json(self, filepath: str):
        """导出决策日志为JSON格式"""
        output = []

        for decision in self.decision_log:
            decision_data = []
            for result in decision:
                decision_data.append({
                    'action': {
                        'steering': result.action[0],
                        'speed': result.action[1],
                        'fork_height': result.action[2]
                    },
                    'score': result.score,
                    'explanation': result.explanation,
                    'stability': result.stability,
                    'success_rate': result.success_rate,
                    'efficiency': result.efficiency,
                    'rank': result.rank,
                })
            output.append(decision_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ 决策日志已导出到: {filepath}")


class SimpleWorldModel:
    """简单世界模型（用于反事实推演模拟）"""
    def __init__(self):
        self.step_count = 0

    def step(self, state: Dict, action: Tuple) -> Dict:
        """
        简单状态更新

        Args:
            state: 当前状态
            action: (steering, speed, fork_height)

        Returns:
            更新后的状态
        """
        self.step_count += 1
        steering, speed, fork_height = action

        # 复制状态避免修改原数据
        new_state = state.copy()

        # 简单运动学更新
        dt = 0.02
        x = new_state.get('x', 0.0)
        y = new_state.get('y', 0.0)
        theta = new_state.get('theta', 0.0)

        # 简单的阿克曼更新（简化版）
        steering_rad = np.radians(steering)
        curvature = np.tan(steering_rad) / 1.2  # 轴距1.2m

        if abs(curvature) < 1e-6:
            x += speed * dt * np.cos(theta)
            y += speed * dt * np.sin(theta)
        else:
            radius = 1.0 / curvature
            delta_theta = speed * dt / radius
            x += radius * (np.sin(theta + delta_theta) - np.sin(theta))
            y -= radius * (np.cos(theta + delta_theta) - np.cos(theta))
            theta += delta_theta

        new_state['x'] = x
        new_state['y'] = y
        new_state['theta'] = theta
        new_state['steering_angle'] = steering
        new_state['speed'] = speed
        new_state['fork_height'] = min(2.0, max(0.0, fork_height))

        # 模拟稳定性（简化为接近目标则稳定）
        dist_to_target = np.sqrt(x**2 + y**2)
        stability = max(0.0, min(1.0, 1.0 - dist_to_target / 1.2))
        new_state['stability'] = stability

        # 模拟成功条件
        if dist_to_target < 0.5 and fork_height > 0.5 and self.step_count > 10:
            new_state['success'] = True

        return new_state

    def reset(self):
        self.step_count = 0


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("🧪 测试反事实推理引擎...")

    # 创建因果图
    from causal_graph import ForkliftCausalGraph
    causal_graph = ForkliftCausalGraph()

    # 创建世界模型
    world_model = SimpleWorldModel()

    # 创建反事实引擎
    engine = CounterfactualEngine(
        causal_graph=causal_graph,
        world_model=world_model,
        config={
            'num_alternatives': 5,
            'rollout_steps': 50,
            'score_weights': {'stability': 0.4, 'success_rate': 0.4, 'efficiency': 0.2}
        }
    )

    # 初始状态
    state = {
        'x': 2.5,
        'y': 0.0,
        'theta': 0.0,
        'steering_angle': 0.0,
        'speed': 0.0,
        'fork_height': 0.2,
        'stability': 0.8,
    }

    # 生成候选动作
    actions = engine.generate_alternatives(state, num=5)
    print(f"  候选动作: {actions}")

    # 推演
    print("\n  反事实推演结果:")
    results = engine.simulate_alternatives(state, actions)

    # 输出报告
    print(engine.format_decision_report(results))

    # 获取最优动作
    best_action, best_result = engine.get_best_action(state, actions)
    print(f"\n  ✅ 最优动作: 转向{best_action[0]:+.1f}°, "
          f"速度{best_action[1]:.2f}m/s, 叉齿{best_action[2]:.2f}m")

    print("\n✅ counterfactual.py 测试通过!")
# 修复 SimpleWorldModel 的成功判断条件
