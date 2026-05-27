"""
无人机拦截仿真系统 - 仿真引擎（升级版）
UAV Interception Simulation - Simulation Engine (Upgraded)

升级内容:
- 同步更新：先计算所有指令，再同步更新双方状态（消除一拍延迟）
- 能量/载荷因子记录
- 电量耗尽判定
- 兼容新动力学模型（气动/能量/执行器延迟）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from drone_dynamics import DroneState, DroneParams, DroneDynamics, TargetStrategy
from guidance_algorithms import (
    PurePursuitGuidance,
    ProportionalNavigationGuidance,
    AugmentedPNGuidance,
    create_guidance,
    InterceptionResult
)
from evasion_algorithms import EvasionStrategy, ThreatAssessment


@dataclass
class SimulationConfig:
    """仿真配置"""
    dt: float = 0.05                # 仿真步长 (s)
    max_time: float = 60.0          # 最大仿真时间 (s)
    intercept_radius: float = 5.0   # 拦截判定半径 (m)
    max_miss_distance: float = 50.0 # 超过此距离判定为丢失 (m)

    # 拦截机初始状态
    interceptor_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 100.0]))
    interceptor_vel: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # 目标机初始状态
    target_pos: np.ndarray = field(default_factory=lambda: np.array([800.0, 600.0, 120.0]))
    target_vel: np.ndarray = field(default_factory=lambda: np.array([-15.0, -10.0, 0.0]))

    # 目标运动模式
    target_mode: str = TargetStrategy.STRAIGHT

    # 目标运动参数
    target_amplitude: float = 5.0
    target_frequency: float = 0.3
    target_change_interval: float = 2.0
    target_max_accel: float = 6.0


@dataclass
class SimulationRecord:
    """仿真记录数据"""
    times: List[float] = field(default_factory=list)
    interceptor_positions: List[np.ndarray] = field(default_factory=list)
    interceptor_velocities: List[np.ndarray] = field(default_factory=list)
    interceptor_accelerations: List[np.ndarray] = field(default_factory=list)
    target_positions: List[np.ndarray] = field(default_factory=list)
    target_velocities: List[np.ndarray] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    closing_speeds: List[float] = field(default_factory=list)
    # 升级新增
    interceptor_load_factors: List[float] = field(default_factory=list)
    interceptor_battery: List[float] = field(default_factory=list)
    target_battery: List[float] = field(default_factory=list)

    def add_record(self, t: float, i_state: DroneState, t_state: DroneState):
        self.times.append(t)
        self.interceptor_positions.append(i_state.position.copy())
        self.interceptor_velocities.append(i_state.velocity.copy())
        self.interceptor_accelerations.append(i_state.acceleration.copy())
        self.target_positions.append(t_state.position.copy())
        self.target_velocities.append(t_state.velocity.copy())

        dist = np.linalg.norm(t_state.position - i_state.position)
        self.distances.append(dist)

        rel_vel = t_state.velocity - i_state.velocity
        if dist > 1e-6:
            los = (t_state.position - i_state.position) / dist
            v_closing = -np.dot(rel_vel, los)
        else:
            v_closing = 0.0
        self.closing_speeds.append(v_closing)

        # 升级：记录载荷因子和电量
        self.interceptor_load_factors.append(i_state.load_factor)
        self.interceptor_battery.append(i_state.battery_remaining)
        self.target_battery.append(t_state.battery_remaining)

    def to_arrays(self):
        """转换为numpy数组便于分析"""
        result = {
            'times': np.array(self.times),
            'interceptor_pos': np.array(self.interceptor_positions),
            'interceptor_vel': np.array(self.interceptor_velocities),
            'interceptor_acc': np.array(self.interceptor_accelerations),
            'target_pos': np.array(self.target_positions),
            'target_vel': np.array(self.target_velocities),
            'distances': np.array(self.distances),
            'closing_speeds': np.array(self.closing_speeds),
        }
        if self.interceptor_load_factors:
            result['interceptor_load_factors'] = np.array(self.interceptor_load_factors)
            result['interceptor_battery'] = np.array(self.interceptor_battery)
            result['target_battery'] = np.array(self.target_battery)
        return result


class SimulationEngine:
    """
    仿真引擎
    管理仿真循环、状态更新和拦截判定
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.record = SimulationRecord()

        # 初始化无人机参数
        self.interceptor_params = DroneParams.interceptor()
        self.target_params = DroneParams.target()

        # 初始化动力学模型
        self.interceptor_dynamics = DroneDynamics(self.interceptor_params)
        self.target_dynamics = DroneDynamics(self.target_params)

        # 初始化目标策略
        self.target_strategy = TargetStrategy(
            mode=config.target_mode,
            amplitude=config.target_amplitude,
            frequency=config.target_frequency,
            change_interval=config.target_change_interval,
            max_accel=config.target_max_accel,
            dt=config.dt
        )

        # 初始化状态
        self.interceptor_state = DroneState(
            position=config.interceptor_pos.copy(),
            velocity=config.interceptor_vel.copy(),
            acceleration=np.zeros(3),
            time=0.0
        )
        self.target_state = DroneState(
            position=config.target_pos.copy(),
            velocity=config.target_vel.copy(),
            acceleration=np.zeros(3),
            time=0.0
        )

        # 结果
        self.result = InterceptionResult()
        self.finished = False

    def set_guidance(self, guidance):
        """设置制导算法"""
        self.guidance = guidance
        if hasattr(guidance, 'reset'):
            guidance.reset()

    def set_evasion(self, evasion: EvasionStrategy):
        """设置逃脱策略（替代目标原生的TargetStrategy）"""
        self.evasion = evasion
        if hasattr(evasion, 'reset'):
            evasion.reset()

    def step(self) -> bool:
        """
        执行一步仿真（升级版：同步更新，消除一拍延迟）

        返回:
            True 如果仿真继续, False 如果仿真结束
        """
        if self.finished:
            return False

        dt = self.config.dt
        i_state = self.interceptor_state
        t_state = self.target_state

        # ── 同步计算所有指令（消除一拍延迟） ──
        # 1a. 目标机计算加速度指令
        if hasattr(self, 'evasion') and self.evasion is not None:
            threat = self.evasion.assess_threat(t_state, i_state)
            target_accel = self.evasion.compute_acceleration(t_state, i_state, threat, dt)
        else:
            target_accel = self.target_strategy.get_acceleration(t_state, self.target_params)

        # 1b. 拦截机计算制导指令（使用同一时刻的状态）
        interceptor_accel = self.guidance.compute_acceleration(i_state, t_state, dt)

        # ── 同步更新双方状态 ──
        self.target_state = self.target_dynamics.update(t_state, target_accel, dt)
        self.interceptor_state = self.interceptor_dynamics.update(i_state, interceptor_accel, dt)

        # ── 记录数据 ──
        self.record.add_record(
            self.interceptor_state.time,
            self.interceptor_state,
            self.target_state
        )

        # ── 拦截判定 ──
        distance = np.linalg.norm(self.target_state.position - self.interceptor_state.position)

        if distance <= self.config.intercept_radius:
            self.result.intercepted = True
            self.result.miss_distance = distance
            self.result.intercept_time = self.interceptor_state.time
            self.result.total_time = self.interceptor_state.time
            self.finished = True
            return False

        # ── 电量耗尽判定 ──
        if (self.interceptor_state.battery_remaining <= 0 or
                self.target_state.battery_remaining <= 0):
            self.result.intercepted = False
            self.result.miss_distance = distance
            self.result.total_time = self.interceptor_state.time
            self.finished = True
            return False

        # ── 超时/丢失判定 ──
        if self.interceptor_state.time >= self.config.max_time:
            self.result.intercepted = False
            self.result.miss_distance = distance
            self.result.total_time = self.config.max_time
            self.finished = True
            return False

        # ── 目标飞出范围判定 ──
        if np.linalg.norm(self.target_state.position) > 5000:
            self.result.intercepted = False
            self.result.miss_distance = distance
            self.result.total_time = self.interceptor_state.time
            self.finished = True
            return False

        return True

    def run(self) -> Tuple[InterceptionResult, SimulationRecord]:
        """
        运行完整仿真

        返回:
            (拦截结果, 仿真记录)
        """
        while self.step():
            pass

        # 计算能量消耗 (加速度积分)
        acc_array = np.array(self.record.interceptor_accelerations)
        times = np.array(self.record.times)
        if len(times) > 1:
            dt_array = np.diff(times)
            acc_mag = np.linalg.norm(acc_array[:-1], axis=1)
            self.result.energy_consumed = float(np.sum(acc_mag * dt_array))

        return self.result, self.record

    def get_current_states(self) -> Tuple[DroneState, DroneState]:
        """获取当前双方状态"""
        return self.interceptor_state, self.target_state


class ScenarioFactory:
    """场景工厂 - 预定义典型拦截场景"""

    @staticmethod
    def head_on() -> SimulationConfig:
        """迎头拦截"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([20.0, 5.0, 0.0]),
            target_pos=np.array([800.0, 0.0, 100.0]),
            target_vel=np.array([-15.0, 0.0, 0.0]),
            target_mode=TargetStrategy.STRAIGHT,
        )

    @staticmethod
    def crossing() -> SimulationConfig:
        """交叉拦截"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([600.0, 600.0, 120.0]),
            target_vel=np.array([-15.0, -10.0, 0.0]),
            target_mode=TargetStrategy.STRAIGHT,
        )

    @staticmethod
    def sinusoidal_evasion() -> SimulationConfig:
        """正弦机动规避"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([800.0, 400.0, 120.0]),
            target_vel=np.array([-15.0, -5.0, 0.0]),
            target_mode=TargetStrategy.SINUSOIDAL,
            target_amplitude=6.0,
            target_frequency=0.25,
        )

    @staticmethod
    def random_evasion() -> SimulationConfig:
        """随机机动规避"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([800.0, 400.0, 120.0]),
            target_vel=np.array([-15.0, -5.0, 0.0]),
            target_mode=TargetStrategy.RANDOM,
            target_change_interval=1.5,
            target_max_accel=7.0,
        )

    @staticmethod
    def tail_chase() -> SimulationConfig:
        """尾追拦截"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([25.0, 0.0, 0.0]),
            target_pos=np.array([500.0, 0.0, 100.0]),
            target_vel=np.array([20.0, 0.0, 0.0]),
            target_mode=TargetStrategy.STRAIGHT,
        )

    @staticmethod
    def high_speed_dive() -> SimulationConfig:
        """高速俯冲拦截"""
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 300.0]),
            interceptor_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([600.0, 300.0, 50.0]),
            target_vel=np.array([-10.0, -10.0, 0.0]),
            target_mode=TargetStrategy.SINUSOIDAL,
            target_amplitude=4.0,
            target_frequency=0.2,
        )
