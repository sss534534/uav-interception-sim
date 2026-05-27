"""
无人机拦截仿真系统 - 无人机动力学模型
UAV Interception Simulation - Drone Dynamics Model

3D空间中的无人机运动学模型，包含动力学约束
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DroneState:
    """无人机状态"""
    position: np.ndarray      # 3D位置 [x, y, z] (m)
    velocity: np.ndarray      # 3D速度 [vx, vy, vz] (m/s)
    acceleration: np.ndarray  # 3D加速度 [ax, ay, az] (m/s²)
    time: float = 0.0         # 当前时间 (s)

    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    def speed_horizontal(self) -> float:
        return np.linalg.norm(self.velocity[:2])

    def altitude(self) -> float:
        return self.position[2]

    def copy(self):
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            time=self.time
        )


@dataclass
class DroneParams:
    """无人机物理参数"""
    max_speed: float = 30.0          # 最大速度 (m/s)
    min_speed: float = 5.0           # 最小速度 (m/s)
    max_acceleration: float = 10.0   # 最大加速度 (m/s²)
    max_turn_rate: float = 1.5       # 最大转弯角速度 (rad/s)
    max_climb_rate: float = 5.0      # 最大爬升率 (m/s)
    max_descent_rate: float = 3.0    # 最大下降率 (m/s)
    drag_coefficient: float = 0.1    # 简化阻力系数

    @staticmethod
    def interceptor():
        """拦截机参数 - 速度优势"""
        return DroneParams(
            max_speed=35.0,
            min_speed=5.0,
            max_acceleration=12.0,
            max_turn_rate=2.0,
            max_climb_rate=6.0,
            max_descent_rate=4.0,
            drag_coefficient=0.08
        )

    @staticmethod
    def target():
        """目标机参数"""
        return DroneParams(
            max_speed=25.0,
            min_speed=8.0,
            max_acceleration=8.0,
            max_turn_rate=1.2,
            max_climb_rate=4.0,
            max_descent_rate=3.0,
            drag_coefficient=0.1
        )


class DroneDynamics:
    """
    无人机3D动力学模型
    包含速度、加速度、转弯半径等物理约束
    """

    def __init__(self, params: DroneParams):
        self.params = params

    def update(self, state: DroneState, accel_cmd: np.ndarray, dt: float) -> DroneState:
        """
        根据控制指令更新无人机状态

        参数:
            state: 当前状态
            accel_cmd: 期望加速度指令 [ax, ay, az] (m/s²)
            dt: 时间步长 (s)

        返回:
            新的DroneState
        """
        # 1. 限制加速度大小
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.params.max_acceleration:
            accel_cmd = accel_cmd * (self.params.max_acceleration / accel_mag)

        # 2. 计算新速度
        new_velocity = state.velocity + accel_cmd * dt

        # 3. 限制速度大小
        speed = np.linalg.norm(new_velocity)
        if speed > self.params.max_speed:
            new_velocity = new_velocity * (self.params.max_speed / speed)
        elif speed < self.params.min_speed and speed > 1e-6:
            new_velocity = new_velocity * (self.params.min_speed / speed)

        # 4. 限制爬升/下降率
        if new_velocity[2] > self.params.max_climb_rate:
            new_velocity[2] = self.params.max_climb_rate
        elif new_velocity[2] < -self.params.max_descent_rate:
            new_velocity[2] = -self.params.max_descent_rate

        # 5. 限制转弯率（水平面）
        if state.speed_horizontal() > 1e-6:
            current_heading = np.arctan2(state.velocity[1], state.velocity[0])
            new_heading = np.arctan2(new_velocity[1], new_velocity[0])
            heading_change = self._normalize_angle(new_heading - current_heading)
            max_heading_change = self.params.max_turn_rate * dt

            if abs(heading_change) > max_heading_change:
                clamped_heading = current_heading + np.sign(heading_change) * max_heading_change
                h_speed = np.linalg.norm(new_velocity[:2])
                new_velocity[0] = h_speed * np.cos(clamped_heading)
                new_velocity[1] = h_speed * np.sin(clamped_heading)

        # 6. 计算新位置
        new_position = state.position + 0.5 * (state.velocity + new_velocity) * dt

        # 7. 地面约束（不允许低于地面）
        if new_position[2] < 0:
            new_position[2] = 0
            new_velocity[2] = max(0, new_velocity[2])

        # 8. 实际加速度（考虑约束后的）
        actual_accel = (new_velocity - state.velocity) / dt if dt > 0 else np.zeros(3)

        return DroneState(
            position=new_position,
            velocity=new_velocity,
            acceleration=actual_accel,
            time=state.time + dt
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class TargetStrategy:
    """
    目标无人机运动策略
    支持多种运动模式
    """

    STRAIGHT = "straight"
    SINUSOIDAL = "sinusoidal"
    RANDOM = "random"

    def __init__(self, mode: str = STRAIGHT, **kwargs):
        self.mode = mode
        self.kwargs = kwargs
        self._random_timer = 0
        self._current_random_accel = np.zeros(3)

    def get_acceleration(self, state: DroneState, params: DroneParams) -> np.ndarray:
        """
        根据运动模式生成目标加速度

        参数:
            state: 当前状态
            params: 无人机参数

        返回:
            加速度指令
        """
        if self.mode == self.STRAIGHT:
            return self._straight_accel(state, params)
        elif self.mode == self.SINUSOIDAL:
            return self._sinusoidal_accel(state, params)
        elif self.mode == self.RANDOM:
            return self._random_acceleration(state, params)
        else:
            return np.zeros(3)

    def _straight_accel(self, state: DroneState, params: DroneParams) -> np.ndarray:
        """匀速直线飞行 - 仅维持当前速度"""
        return np.zeros(3)

    def _sinusoidal_accel(self, state: DroneState, params: DroneParams) -> np.ndarray:
        """正弦机动飞行"""
        amplitude = self.kwargs.get('amplitude', 5.0)   # 加速度幅值 (m/s²)
        frequency = self.kwargs.get('frequency', 0.3)    # 机动频率 (Hz)

        # 水平面正弦机动
        speed_h = state.speed_horizontal()
        if speed_h < 1e-6:
            return np.zeros(3)

        # 垂直于速度方向的加速度
        heading = np.arctan2(state.velocity[1], state.velocity[0])
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)

        accel_mag = amplitude * np.sin(2 * np.pi * frequency * state.time)
        accel = np.array([
            perp_x * accel_mag,
            perp_y * accel_mag,
            0.5 * amplitude * np.sin(2 * np.pi * frequency * 0.5 * state.time)
        ])

        return accel

    def _random_acceleration(self, state: DroneState, params: DroneParams) -> np.ndarray:
        """随机机动飞行"""
        dt = self.kwargs.get('dt', 0.1)
        change_interval = self.kwargs.get('change_interval', 2.0)  # 每2秒改变一次方向
        max_accel = self.kwargs.get('max_accel', 6.0)

        self._random_timer += dt
        if self._random_timer >= change_interval:
            self._random_timer = 0
            # 生成随机加速度方向
            random_dir = np.random.randn(3)
            random_dir /= np.linalg.norm(random_dir)
            random_mag = np.random.uniform(max_accel * 0.3, max_accel)
            self._current_random_accel = random_dir * random_mag

        return self._current_random_accel

    def reset(self):
        """重置随机状态"""
        self._random_timer = 0
        self._current_random_accel = np.zeros(3)
