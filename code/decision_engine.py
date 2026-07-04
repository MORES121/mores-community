"""
code/decision_engine.py
MORES 决策集成引擎
融合因果推理 + 反事实推演 + 感知 + 控制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json

from causal_graph import ForkliftCausalGraph
from counterfactual import CounterfactualEngine, CounterfactualResult, SimpleWorldModel
from perception import PerceptionModule, PerceptionResult
from controller import ForkliftController


@dataclass
class DecisionResult:
    """决策结果"""
    action: Tuple[float, float, float]
    explanation: str
    confidence: float
    timestamp: float
    causal_path: List[str]
    alternatives: List[CounterfactualResult]
    state: Dict


class MORESDecisionEngine:
    """MORES 决策集成引擎"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        self.config = config

        causal_config = config.get('causal_graph', {})
        self.causal_graph = ForkliftCausalGraph(
            nodes=causal_config.get('nodes'),
            edges=causal_config.get('edges')
        )

        self.world_model = SimpleWorldModel()

        cf_config = config.get('counterfactual', {})
        self.counterfactual = CounterfactualEngine(
            causal_graph=self.causal_graph,
            world_model=self.world_model,
            config=cf_config
        )

        perception_config = config.get('perception', {})
        self.perception = PerceptionModule(perception_config)

        controller_config = config.get('forklift', {})
        controller_config['pid'] = config.get('controller', {}).get('pid', {})
        controller_config['max_steering_rate'] = config.get('controller', {}).get('max_steering_rate', 15.0)
        self.controller = ForkliftController(controller_config)

        self.decision_history: List[DecisionResult] = []

        self.current_state = {
            'x': 0.0, 'y': 0.0, 'theta': 0.0,
            'steering_angle': 0.0, 'speed': 0.0,
            'fork_height': 0.0, 'stability': 1.0,
            'task_success': False,
        }

        self.step_count = 0
        self.max_steps = config.get('simulation', {}).get('max_steps', 5000)

    def perceive(self, rgb_image=None, lidar_points=None, depth_image=None):
        return self.perception.process(rgb_image, lidar_points, depth_image)

    def reason_causal(self, perception_result):
        node_values = {
            'perception_error': 1.0 - perception_result.confidence,
            'path_quality': 0.8,
            'fork_alignment': 0.7,
        }
        self.causal_graph.update_from_config(node_values)
        shortest = self.causal_graph.get_shortest_causal_path('perception_error', 'task_success')
        return {
            'shortest_path': shortest,
            'explanation': self.causal_graph.explain_path(shortest) if shortest else "无路径",
            'node_values': node_values,
        }

    def plan_alternatives(self, state):
        # 动态速度：根据距离调整
        dist = state.get('distance_to_target', 2.0)
        if dist > 1.5:
            state['preferred_speed'] = 0.9
        elif dist > 0.5:
            state['preferred_speed'] = 0.7
        else:
            state['preferred_speed'] = 0.3
        # 确保状态中有目标叉齿高度
        if "target_fork_height" not in state:
            state["target_fork_height"] = 0.5
        # 根据距离动态调整速度
        dist = state.get("distance_to_target", 2.0)
        if dist > 1.5:
            state["preferred_speed"] = 1.8
        elif dist > 0.5:
            state["preferred_speed"] = 1.2
        else:
            state["preferred_speed"] = 0.3
        actions = self.counterfactual.generate_alternatives(state)
        # 限幅所有候选动作
        actions = [(np.clip(s, -30.0, 30.0), np.clip(v, -2.0, 2.0), np.clip(f, 0.0, 2.0)) for s, v, f in actions]
        return self.counterfactual.simulate_alternatives(state, actions)

    def decide(self, perception_result, alternatives):
        # 获取目标叉齿高度
        target_fork = self.current_state.get('target_fork_height', 0.5)
        if not alternatives:
            action = (0.0, 0.0, 0.0)
            explanation = "无候选动作，执行默认"
            confidence = 0.0
        else:
            best = alternatives[0]
            # 调整叉齿高度到目标值
            s, v, f = best.action
            best.action = (s, v, target_fork)
            action = best.action
            explanation = best.explanation
            confidence = best.score

        causal_path = self.causal_graph.get_shortest_causal_path('perception_error', 'task_success')

        result = DecisionResult(
            action=action,
            explanation=explanation,
            confidence=confidence,
            timestamp=time.time(),
            causal_path=causal_path or [],
            alternatives=alternatives,
            state=self.current_state.copy()
        )

        self.decision_history.append(result)
        return result

    def control(self, action):
        steering, speed, fork_height = action
        # 限幅
        steering = np.clip(steering, -30.0, 30.0)
        speed = np.clip(speed, -2.2, 2.2)
        fork_height = np.clip(fork_height, 0.0, 2.0)
        current_speed = self.current_state.get('speed', 0.0)
        current_steering = self.current_state.get('steering_angle', 0.0)
        current_fork = self.current_state.get('fork_height', 0.0)
        dt = self.config.get('simulation', {}).get('timestep', 0.02)

        speed_cmd, steering_cmd = self.controller.update(
            target_speed=speed,
            target_steering=steering,
            current_speed=current_speed,
            current_steering=current_steering,
            dt=dt
        )

        fork_speed = self.controller.control_fork(
            target_height=fork_height,
            current_height=current_fork,
            dt=dt
        )

        return {'speed_cmd': speed_cmd, 'steering_cmd': steering_cmd, 'fork_speed': fork_speed, 'dt': dt}

    def step(self, rgb_image=None, lidar_points=None, depth_image=None, guidance=None):
        self.step_count += 1

        if self.step_count > self.max_steps:
            return {'status': 'timeout', 'state': self.current_state, 'step': self.step_count}

        perception_result = self.perceive(rgb_image, lidar_points, depth_image)
        causal_result = self.reason_causal(perception_result)
        alternatives = self.plan_alternatives(self.current_state)
        decision = self.decide(perception_result, alternatives)
        # 限幅动作
        steering, speed, fork_height = decision.action
        steering = np.clip(steering, -30.0, 30.0)
        speed = np.clip(speed, -2.2, 2.2)
        fork_height = np.clip(fork_height, 0.0, 2.0)
        clipped_action = (steering, speed, fork_height)
        control_result = self.control(clipped_action)

        self.current_state = self.world_model.step(self.current_state, clipped_action)
        self.causal_graph.set_node_value('task_success', self.current_state.get('success', 0.0))

        status = 'running'
        if self.current_state.get('success', False):
            status = 'success'
        elif self.current_state.get('collision', False):
            status = 'collision'
        elif self.current_state.get('tipped', False):
            status = 'tipped'

        return {
            'status': status,
            'step': self.step_count,
            'perception': perception_result,
            'causal': causal_result,
            'decision': decision,
            'control': control_result,
            'state': self.current_state.copy(),
        }

    def get_decision_report(self):
        if not self.decision_history:
            return "暂无决策记录"

        latest = self.decision_history[-1]
        lines = [
            "📊 MORES 决策报告",
            "=" * 60,
            f"步骤: {self.step_count}",
            f"状态: {self.current_state}",
            "",
            "因果路径:",
            f"  {' → '.join(latest.causal_path) if latest.causal_path else '无路径'}",
            "",
            "反事实推演:",
        ]

        for alt in latest.alternatives[:3]:
            steering, speed, fork = alt.action
            rank = "⭐" if alt.rank == 1 else "  "
            lines.append(
                f"  {rank} #{alt.rank}: 转向{steering:+.1f}° "
                f"速度{speed:.2f}m/s 叉齿{fork:.2f}m "
                f"→ {alt.score*100:.0f}%"
            )

        lines.append("")
        lines.append(f"最优动作: {latest.action}")
        lines.append(f"解释: {latest.explanation}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_summary(self):
        return {
            'total_steps': self.step_count,
            'total_decisions': len(self.decision_history),
            'current_state': self.current_state,
            'latest_decision': self.decision_history[-1] if self.decision_history else None,
            'causal_graph': {
                'nodes': len(self.causal_graph.get_nodes()),
                'edges': len(self.causal_graph.get_edges()),
            },
            'counterfactual_logs': len(self.counterfactual.decision_log),
        }

    def export_decision_log(self, filepath):
        output = []
        for decision in self.decision_history:
            output.append({
                'step': len(output) + 1,
                'action': {
                    'steering': decision.action[0],
                    'speed': decision.action[1],
                    'fork_height': decision.action[2]
                },
                'explanation': decision.explanation,
                'confidence': decision.confidence,
                'timestamp': decision.timestamp,
                'causal_path': decision.causal_path,
                'state': decision.state,
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ 决策日志已导出: {filepath}")

    def reset(self):
        self.causal_graph.reset_interventions()
        self.counterfactual.decision_log = []
        self.perception.reset()
        self.controller.reset()
        self.decision_history = []
        self.current_state = {
            'x': 0.0, 'y': 0.0, 'theta': 0.0,
            'steering_angle': 0.0, 'speed': 0.0,
            'fork_height': 0.0, 'stability': 1.0,
            'task_success': False,
        }
        self.step_count = 0
        self.world_model.reset()


if __name__ == "__main__":
    print("🧪 测试 MORES 决策集成引擎...")

    config = {
        'simulation': {'timestep': 0.02, 'max_steps': 1000},
        'forklift': {'wheelbase': 1.2, 'max_steering_angle': 30.0, 'max_speed': 1.0},
        'controller': {'pid': {'kp': 1.2, 'ki': 0.05, 'kd': 0.01}, 'max_steering_rate': 15.0},
        'counterfactual': {'num_alternatives': 5, 'rollout_steps': 50},
        'perception': {'camera': {'enabled': True}, 'lidar': {'enabled': True}},
    }

    engine = MORESDecisionEngine(config)
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lidar_points = np.random.randn(100, 3) * 0.5 + np.array([2.0, 0.0, 0.5])

    print("  执行决策步骤...")
    for i in range(5):
        result = engine.step(rgb_image=rgb_image, lidar_points=lidar_points)
        print(f"    步骤 {i+1}: {result['status']}, 动作: {result['decision'].action}")

    print("\n" + engine.get_decision_report())

    summary = engine.get_summary()
    print(f"\n  系统摘要:")
    print(f"    总步数: {summary['total_steps']}")
    print(f"    总决策数: {summary['total_decisions']}")
    print(f"    因果图节点: {summary['causal_graph']['nodes']}")
    print(f"    因果图边: {summary['causal_graph']['edges']}")

    print("\n✅ decision_engine.py 测试通过!")
# 在 control 方法中增加限幅
# 修改 ForkliftController 的 control_fork 方法
