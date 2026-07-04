"""
code/controller.py
阿克曼底盘运动学模型 + PID控制器
用于叉车精确速度/转向控制
"""

import numpy as np
from typing import Tuple, Dict, Optional


class AckermannKinematics:
    """
    阿克曼底盘运动学模型
    适用于前轮转向、后轮驱动的叉车/车辆
    """

    def __init__(self, wheelbase: float = 1.2, max_steering_angle: float = 30.0):
        """
        Args:
            wheelbase: 轴距 (m)
            max_steering_angle: 最大转向角 (度)
        """
        self.wheelbase = wheelbase
        self.max_steering_angle = np.radians(max_steering_angle)

    def compute_curvature(self, steering_angle: float) -> float:
        """根据转向角计算曲率"""
        steering_rad = np.radians(steering_angle)
        steering_rad = np.clip(steering_rad, -self.max_steering_angle, self.max_steering_angle)
        return np.tan(steering_rad) / self.wheelbase

    def forward(self, speed: float, steering_angle: float, dt: float,
                x: float, y: float, theta: float) -> Tuple[float, float, float]:
        """
        前向运动学更新

        Args:
            speed: 线速度 (m/s)
            steering_angle: 转向角 (度)
            dt: 时间步长 (s)
            x, y, theta: 当前位姿 (m, m, rad)

        Returns:
            更新后的位姿 (x, y, theta)
        """
        curvature = self.compute_curvature(steering_angle)

        if abs(curvature) < 1e-6:
            # 近似直线运动
            x_new = x + speed * dt * np.cos(theta)
            y_new = y + speed * dt * np.sin(theta)
            theta_new = theta
        else:
            # 圆弧运动
            radius = 1.0 / curvature
            delta_theta = speed * dt / radius
            x_new = x + radius * (np.sin(theta + delta_theta) - np.sin(theta))
            y_new = y - radius * (np.cos(theta + delta_theta) - np.cos(theta))
            theta_new = theta + delta_theta

        return x_new, y_new, theta_new


class PIDController:
    """
    标准PID控制器
    用于速度/转向角闭环控制
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 output_limits: Tuple[float, float] = (-float('inf'), float('inf'))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def reset(self):
        """重置积分项和误差状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        计算PID输出

        Args:
            setpoint: 目标值
            measurement: 当前测量值
            dt: 时间步长 (s)

        Returns:
            控制输出
        """
        error = setpoint - measurement

        # 比例项
        p_term = self.kp * error

        # 积分项（带防积分饱和）
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 微分项（防止噪声放大，对测量值微分）
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt
        else:
            d_term = 0.0

        output = p_term + i_term + d_term

        # 输出限幅
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # 防积分饱和（钳位积分项）
        if output != self.prev_output:
            self.integral -= (error * dt) if self._saturated(output) else 0.0

        self.prev_error = error
        self.prev_output = output

        return output

    def _saturated(self, output: float) -> bool:
        """检查输出是否饱和"""
        return output <= self.output_limits[0] or output >= self.output_limits[1]


class ForkliftController:
    """
    叉车一体化控制器
    整合阿克曼运动学 + PID + 叉齿控制
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 配置字典，包含:
                - wheelbase: 轴距
                - max_steering_angle: 最大转向角
                - max_speed: 最大线速度
                - pid.kp, pid.ki, pid.kd
                - max_steering_rate: 最大转向角速度
        """
        if config is None:
            config = {}

        self.wheelbase = config.get('wheelbase', 1.2)
        self.max_steering_angle = config.get('max_steering_angle', 30.0)
        self.max_speed = config.get('max_speed', 1.0)
        self.max_steering_rate = np.radians(config.get('max_steering_rate', 15.0))

        # 运动学模型
        self.kinematics = AckermannKinematics(
            wheelbase=self.wheelbase,
            max_steering_angle=self.max_steering_angle
        )

        # 速度PID
        pid_config = config.get('pid', {})
        self.speed_pid = PIDController(
            kp=pid_config.get('kp', 1.2),
            ki=pid_config.get('ki', 0.05),
            kd=pid_config.get('kd', 0.01),
            output_limits=(-self.max_speed, self.max_speed)
        )

        # 转向PID
        self.steering_pid = PIDController(
            kp=pid_config.get('kp', 1.2),
            ki=pid_config.get('ki', 0.05),
            kd=pid_config.get('kd', 0.01),
            output_limits=(-self.max_steering_angle, self.max_steering_angle)
        )

        # 当前状态
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_fork_height = 0.0

    def update(self, target_speed: float, target_steering: float,
               current_speed: float, current_steering: float,
               dt: float) -> Tuple[float, float]:
        """
        更新速度/转向控制

        Args:
            target_speed: 目标线速度 (m/s)
            target_steering: 目标转向角 (度)
            current_speed: 当前线速度 (m/s)
            current_steering: 当前转向角 (度)
            dt: 时间步长 (s)

        Returns:
            (速度指令, 转向指令)
        """
        # 限幅
        target_speed = np.clip(target_speed, -self.max_speed, self.max_speed)
        target_steering = np.clip(target_steering, -self.max_steering_angle, self.max_steering_angle)

        # PID计算
        speed_cmd = self.speed_pid.compute(target_speed, current_speed, dt)
        steering_cmd = self.steering_pid.compute(target_steering, current_steering, dt)

        # 转向速率限制
        steering_rate = (steering_cmd - self.current_steering) / dt
        steering_rate = np.clip(steering_rate, -self.max_steering_rate, self.max_steering_rate)
        steering_cmd = self.current_steering + steering_rate * dt

        self.current_speed = speed_cmd
        self.current_steering = steering_cmd

        return speed_cmd, steering_cmd

    def control_fork(self, target_height: float, current_height: float,
                     dt: float, max_speed: float = 0.2) -> float:
        """
        叉齿高度控制（简单P控制器）

        Args:
            target_height: 目标高度 (m)
            current_height: 当前高度 (m)
            dt: 时间步长 (s)
            max_speed: 最大升降速度 (m/s)

        Returns:
            叉齿速度指令 (m/s)
        """
        error = target_height - current_height

        # 简单P控制，带死区
        if abs(error) < 0.01:
            return 0.0

        speed = np.clip(error / dt, -max_speed, max_speed)
        return speed

    def get_state(self) -> Dict:
        """获取当前控制器状态"""
        return {
            'speed': self.current_speed,
            'steering': self.current_steering,
            'fork_height': self.current_fork_height
        }

    def reset(self):
        """重置控制器状态"""
        self.speed_pid.reset()
        self.steering_pid.reset()
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_fork_height = 0.0


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("🧪 测试阿克曼运动学模型...")

    kin = AckermannKinematics(wheelbase=1.2, max_steering_angle=30.0)

    # 测试转向曲率
    curvature = kin.compute_curvature(10.0)
    print(f"  转向角10° → 曲率: {curvature:.4f} 1/m")

    # 测试前向运动学
    x, y, theta = 0.0, 0.0, 0.0
    x_new, y_new, theta_new = kin.forward(
        speed=0.5, steering_angle=10.0, dt=0.1,
        x=x, y=y, theta=theta
    )
    print(f"  前进0.1s → x={x_new:.3f}, y={y_new:.3f}, theta={np.degrees(theta_new):.1f}°")

    print("\n🧪 测试PID控制器...")
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    output = pid.compute(setpoint=1.0, measurement=0.0, dt=0.1)
    print(f"  第一次输出: {output:.3f}")

    print("\n✅ controller.py 测试通过!")
