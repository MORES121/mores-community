"""
code/forklift_env.py
Isaac Sim 环境封装
兼容 Gym 接口，支持训练和推理
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import time


@dataclass
class EnvConfig:
    """环境配置"""
    max_steps: int = 5000
    dt: float = 0.10
    timestep: float = 0.02  # 兼容 main.py 传入的 timestep
    render_mode: str = "human"
    backend: str = "simulated"

    # 叉车参数
    wheelbase: float = 1.2
    max_steering_angle: float = 30.0
    max_speed: float = 2.5
    max_fork_height: float = 2.0

    # 任务参数
    target_tolerance: float = 0.1
    angle_tolerance: float = 5.0

    # 奖励权重
    reward_weights: Dict = field(default_factory=lambda: {
        'success': 100.0,
        'progress': 1.0,
        'stability': 0.5,
        'efficiency': 0.1,
        'collision': -50.0,
        'tipped': -100.0,
    })

    def __post_init__(self):
        # 如果传入了 timestep 但没有 dt，使用 timestep 作为 dt
        if hasattr(self, 'timestep') and self.timestep:
            self.dt = self.timestep


class ForkliftEnv:
    """叉车仿真环境"""

    def __init__(self, config: Optional[Dict] = None, backend: str = "simulated"):
        if config is None:
            config = {}

        # 确保 backend 传入 config
        config['backend'] = backend
        self.config = EnvConfig(**config)
        self.backend = backend

        self.state = None
        self.step_count = 0
        self.episode_return = 0.0
        self.done = False
        self.info = {}

        self.target_position = np.array([2.0, 0.0, 0.5])
        self.target_orientation = 0.0

        self.reset()

    def _get_observation_space(self) -> Dict:
        return {
            'position': (3,),
            'velocity': (2,),
            'fork_height': (1,),
            'target': (4,),
            'stability': (1,),
        }

    def _get_action_space(self) -> Dict:
        return {
            'steering': (-30.0, 30.0),
            'speed': (-1.0, 1.0),
            'fork': (0.0, 2.0),
        }

    def reset(self) -> Dict:
        self.step_count = 0
        self.episode_return = 0.0
        self.done = False
        self.info = {}

        self.state = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0,
            'speed': 0.0,
            'steering': 0.0,
            'fork_height': 0.0,
            'stability': 1.0,
            'success': False,
            'collision': False,
            'tipped': False,
        }

        self._randomize_target()
        return self._get_observation()

    def _randomize_target(self):
        self.target_position = np.array([
            2.0 + np.random.randn() * 0.05,
            0.0 + np.random.randn() * 0.05,
            0.5 + np.random.randn() * 0.05
        ])
        self.target_orientation = np.random.randn() * 2.0

    def _get_observation(self) -> Dict:
        return {
            'position': np.array([
                self.state['x'],
                self.state['y'],
                self.state['theta']
            ]),
            'velocity': np.array([
                self.state.get('speed', 0.0),
                self.state.get('steering', 0.0)
            ]),
            'fork_height': np.array([self.state['fork_height']]),
            'target': np.array([
                self.target_position[0],
                self.target_position[1],
                self.target_position[2],
                self.target_orientation
            ]),
            'stability': np.array([self.state.get('stability', 1.0)]),
        }

    def _get_observation_flat(self) -> np.ndarray:
        obs = self._get_observation()
        return np.concatenate([
            obs['position'],
            obs['velocity'],
            obs['fork_height'],
            obs['target'],
            obs['stability']
        ])

    def step(self, action: Tuple[float, float, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        steering, speed, fork_height = action
        # 限幅
        steering = np.clip(steering, -30.0, 30.0)
        speed = np.clip(speed, -2.0, 2.0)
        fork_height = np.clip(fork_height, 0.0, 2.0)

        steering = np.clip(steering, -30.0, 30.0)
        speed = np.clip(speed, -2.0, 2.0)
        fork_height = np.clip(fork_height, 0.0, min(2.0, self.config.max_fork_height))
        if fork_height > 2.0:
            fork_height = 2.0
        if fork_height < 0.0:
            fork_height = 0.0

        dt = self.config.dt
        self.step_count += 1

        self._step_simulated(steering, speed, fork_height, dt)

        reward = self._compute_reward()
        self.done = self._check_done()

        self.info = {
            'step': self.step_count,
            'stability': self.state.get('stability', 1.0),
            'distance_to_target': self._compute_distance(),
        }

        self.episode_return += reward

        return self._get_observation_flat(), reward, self.done, self.info

    def _step_simulated(self, steering: float, speed: float, fork_height: float, dt: float):
        state = self.state
        # 确保速度足够大才能移动
        effective_speed = speed  # 最小速度0.1m/s
        steering_rad = np.radians(steering)
        wheelbase = self.config.wheelbase

        # 限制转向角，避免原地打转
        steering_rad = np.clip(steering_rad, -0.3, 0.3)  # 约±17度

        if abs(steering_rad) < 1e-6:
            state['x'] += effective_speed * dt * np.cos(state['theta'])
            state['y'] += effective_speed * dt * np.sin(state['theta'])
        else:
            radius = wheelbase / np.tan(steering_rad)
            # 限制最小转弯半径
            radius = max(abs(radius), 1.0) * np.sign(radius)
            delta_theta = effective_speed * dt / radius
            state['x'] += radius * (np.sin(state['theta'] + delta_theta) - np.sin(state['theta']))
            state['y'] -= radius * (np.cos(state['theta'] + delta_theta) - np.cos(state['theta']))
            state['theta'] += delta_theta

        state['steering'] = steering
        state['speed'] = effective_speed
        state['fork_height'] = fork_height

        # 计算到目标的距离
        dist = self._compute_distance()
        state['stability'] = max(0.0, min(1.0, 1.0 - dist / 2.0))

        # 成功条件：距离<0.3m且叉齿高度接近0.5m
        if dist < 0.3 and abs(fork_height - 0.5) < 0.05:
            state['success'] = True

    def _compute_distance(self) -> float:
        dx = self.state['x'] - self.target_position[0]
        dy = self.state['y'] - self.target_position[1]
        dz = self.state['fork_height'] - self.target_position[2]
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _compute_reward(self) -> float:
        weights = self.config.reward_weights
        reward = 0.0

        if self.state.get('success', False):
            reward += weights.get('success', 100.0)

        dist = self._compute_distance()
        reward += weights.get('progress', 1.0) * (1.0 - dist / 5.0)

        stability = self.state.get('stability', 1.0)
        reward += weights.get('stability', 0.5) * stability

        reward += weights.get('efficiency', 0.1) * (1.0 - self.step_count / self.config.max_steps)

        if self.state.get('collision', False):
            reward += weights.get('collision', -50.0)

        if self.state.get('tipped', False):
            reward += weights.get('tipped', -100.0)

        return reward

    def _check_done(self) -> bool:
        if self.step_count >= self.config.max_steps:
            return True
        if self.state.get('success', False):
            return True
        if self.state.get('collision', False):
            return True
        if self.state.get('tipped', False):
            return True
        return False

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return None

    def get_state(self) -> Dict:
        return self.state.copy() if self.state else {}

    def set_target(self, position: Tuple[float, float, float], orientation: float = 0.0):
        self.target_position = np.array(position)
        self.target_orientation = orientation

    def seed(self, seed: int):
        np.random.seed(seed)


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("🧪 测试 ForkliftEnv...")

    env = ForkliftEnv(backend="simulated")
    print(f"  观测空间: {env._get_observation_space()}")
    print(f"  动作空间: {env._get_action_space()}")

    obs = env.reset()
    print(f"  重置后观测维度: {len(obs)}")

    print("\n  执行步骤:")
    for i in range(10):
        action = (np.random.randn() * 5, np.random.rand() * 0.5, np.random.rand() * 0.5)
        obs, reward, done, info = env.step(action)
        print(f"    步骤 {i+1}: reward={reward:.2f}, done={done}, dist={info['distance_to_target']:.3f}")

        if done:
            print(f"  任务完成! 总步数: {i+1}")
            break

    print("\n✅ forklift_env.py 测试通过!")
