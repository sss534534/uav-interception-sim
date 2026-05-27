"""
无人机拦截仿真系统 - 无人机动力学模型
UAV Interception Simulation - Drone Dynamics Model

3D空间中的无人机运动学模型，包含动力学约束、气动模型、能量模型和执行器延迟模型
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
    battery_remaining: float = 1.0  # 剩余电量比例 (0-1)
    load_factor: float = 1.0  # 当前载荷因子

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
            time=self.time,
            battery_remaining=self.battery_remaining,
            load_factor=self.load_factor
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
    mass: float = 5.0                # 质量 (kg)
    battery_capacity_wh: float = 500.0  # 电池容量 (Wh)
    wing_area: float = 0.5            # 机翼面积 (m²)
    C_L_base: float = 0.3             # 基础升力系数
    C_D_base: float = 0.05            # 基础阻力系数
    C_D_induced: float = 0.04         # 诱导阻力系数
    n_max: float = 8.0                # 最大正载荷因子
    n_min: float = -4.0               # 最大负载荷因子
    actuator_response_time: float = 0.1  # 执行器响应时间 (s)
    use_aero_model: bool = False      # 是否启用气动模型
    use_energy_model: bool = False    # 是否启用能量模型

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
            drag_coefficient=0.08,
            mass=4.5,
            battery_capacity_wh=600.0,
            wing_area=0.45,
            C_L_base=0.35,
            C_D_base=0.04,
            C_D_induced=0.035,
            n_max=9.0,
            n_min=-4.0,
            actuator_response_time=0.08,
            use_aero_model=False,
            use_energy_model=False
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
            drag_coefficient=0.1,
            mass=5.0,
            battery_capacity_wh=500.0,
            wing_area=0.5,
            C_L_base=0.3,
            C_D_base=0.05,
            C_D_induced=0.04,
            n_max=6.0,
            n_min=-3.0,
            actuator_response_time=0.1,
            use_aero_model=False,
            use_energy_model=False
        )


class AerodynamicsModel:
    """气动模型：升力、阻力、大气密度"""

    def __init__(self, params: DroneParams):
        self.params = params

    def air_density(self, altitude: float) -> float:
        """
        国际标准大气密度模型 (ISA)

        参数:
            altitude: 海拔高度 (m)

        返回:
            空气密度 (kg/m³)
        """
        # 海平面标准参数
        rho_0 = 1.225       # 海平面空气密度 (kg/m³)
        T_0 = 288.15        # 海平面标准温度 (K)
        L = 0.0065          # 温度递减率 (K/m)
        g = 9.80665         # 重力加速度 (m/s²)
        R = 287.058         # 空气气体常数 (J/(kg·K))

        if altitude < 11000:
            # 对流层
            T = T_0 - L * altitude
            rho = rho_0 * (T / T_0) ** (g / (R * L) - 1)
        else:
            # 平流层（简化处理）
            T_11 = T_0 - L * 11000
            rho_11 = rho_0 * (T_11 / T_0) ** (g / (R * L) - 1)
            rho = rho_11 * np.exp(-g * (altitude - 11000) / (R * T_11))

        return rho

    def compute_lift(self, velocity: float, altitude: float, bank_angle: float = 0) -> float:
        """
        计算升力 (N)

        参数:
            velocity: 飞行速度 (m/s)
            altitude: 飞行高度 (m)
            bank_angle: 倾斜角 (rad)

        返回:
            升力 (N)
        """
        rho = self.air_density(altitude)
        v = max(velocity, 1e-6)
        C_L = self.params.C_L_base / max(np.cos(bank_angle), 0.1)
        lift = 0.5 * rho * v ** 2 * self.params.wing_area * C_L
        return lift

    def compute_drag(self, velocity: float, altitude: float, C_L: float = None) -> float:
        """
        计算阻力 (N)

        参数:
            velocity: 飞行速度 (m/s)
            altitude: 飞行高度 (m)
            C_L: 升力系数，若为None则使用基础值

        返回:
            阻力 (N)
        """
        rho = self.air_density(altitude)
        v = max(velocity, 1e-6)
        if C_L is None:
            C_L = self.params.C_L_base
        # 总阻力 = 寄生阻力 + 诱导阻力
        C_D = self.params.C_D_base + self.params.C_D_induced * C_L ** 2
        drag = 0.5 * rho * v ** 2 * self.params.wing_area * C_D
        return drag

    def compute_aero_acceleration(self, velocity: np.ndarray, altitude: float,
                                   accel_cmd: np.ndarray) -> np.ndarray:
        """
        计算考虑气动效应后的实际加速度

        推力 = 阻力 + 加速度所需力
        如果推力超过可用推力(由max_acceleration和mass决定)，则限幅

        参数:
            velocity: 速度矢量 (m/s)
            altitude: 飞行高度 (m)
            accel_cmd: 期望加速度指令 (m/s²)

        返回:
            修正后的加速度 (m/s²)
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return accel_cmd

        # 计算当前阻力
        drag_force = self.compute_drag(speed, altitude)
        drag_accel_mag = drag_force / self.params.mass

        # 阻力方向与速度方向相反
        drag_accel = drag_accel_mag * (velocity / speed)

        # 可用推力加速度
        max_thrust_accel = self.params.max_acceleration

        # 所需推力加速度 = 阻力加速度 + 指令加速度
        required_accel = accel_cmd + drag_accel
        required_mag = np.linalg.norm(required_accel)

        if required_mag > max_thrust_accel:
            # 推力不足，按比例缩减指令加速度
            accel_cmd = accel_cmd * (max_thrust_accel / required_mag)
            drag_accel = drag_accel * (max_thrust_accel / required_mag)

        return accel_cmd - drag_accel

    def compute_drag_acceleration(self, velocity: np.ndarray, altitude: float) -> np.ndarray:
        """
        计算阻力加速度矢量

        参数:
            velocity: 速度矢量 (m/s)
            altitude: 飞行高度 (m)

        返回:
            阻力加速度矢量 (m/s²)
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)

        drag_force = self.compute_drag(speed, altitude)
        drag_accel_mag = drag_force / self.params.mass
        return drag_accel_mag * (velocity / speed)


class EnergyModel:
    """能量消耗模型"""

    def __init__(self, params: DroneParams):
        self.params = params
        self._battery_remaining = 1.0  # 剩余电量比例
        self._energy_remaining_wh = params.battery_capacity_wh  # 剩余能量 (Wh)

    def compute_power(self, accel: np.ndarray, velocity: np.ndarray) -> float:
        """
        计算瞬时功率 (W)

        P = |F · v|，其中 F = m * a

        参数:
            accel: 加速度矢量 (m/s²)
            velocity: 速度矢量 (m/s)

        返回:
            瞬时功率 (W)
        """
        force = self.params.mass * accel
        power = abs(np.dot(force, velocity))
        # 加上悬停功率（简化模型）
        hover_power = self.params.mass * 9.81 * 0.1  # 悬停效率系数
        return power + hover_power

    def update(self, accel: np.ndarray, velocity: np.ndarray, dt: float) -> float:
        """
        更新能量状态

        参数:
            accel: 加速度矢量 (m/s²)
            velocity: 速度矢量 (m/s)
            dt: 时间步长 (s)

        返回:
            剩余电量比例 (0-1)
        """
        power = self.compute_power(accel, velocity)
        energy_consumed = power * dt / 3600.0  # 转换为 Wh
        self._energy_remaining_wh -= energy_consumed

        if self._energy_remaining_wh <= 0:
            self._energy_remaining_wh = 0
            self._battery_remaining = 0.0
        else:
            self._battery_remaining = self._energy_remaining_wh / self.params.battery_capacity_wh

        return self._battery_remaining

    def reset(self):
        """重置能量状态"""
        self._battery_remaining = 1.0
        self._energy_remaining_wh = self.params.battery_capacity_wh


class ActuatorModel:
    """执行器响应延迟模型（一阶惯性）"""

    def __init__(self, response_time: float):
        """
        参数:
            response_time: 执行器响应时间常数 (s)
        """
        self.response_time = max(response_time, 1e-6)
        self._current_force = np.zeros(3)  # 当前实际输出力/加速度
        self._initialized = False

    def update(self, force_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        一阶低通滤波更新

        参数:
            force_cmd: 指令力/加速度矢量
            dt: 时间步长 (s)

        返回:
            滤波后的实际力/加速度矢量
        """
        if not self._initialized:
            self._current_force = force_cmd.copy()
            self._initialized = True
            return self._current_force

        # 一阶低通滤波: alpha = dt / (tau + dt)
        alpha = dt / (self.response_time + dt)
        self._current_force = self._current_force + alpha * (force_cmd - self._current_force)
        return self._current_force.copy()

    def reset(self):
        """重置执行器状态"""
        self._current_force = np.zeros(3)
        self._initialized = False


class DroneDynamics:
    """
    无人机3D动力学模型
    包含速度、加速度、转弯半径等物理约束
    可选集成气动模型、能量模型和执行器延迟模型
    """

    def __init__(self, params: DroneParams):
        self.params = params

        # 初始化可选模型
        self._aero: Optional[AerodynamicsModel] = None
        self._energy: Optional[EnergyModel] = None
        self._actuator: Optional[ActuatorModel] = None

        if params.use_aero_model:
            self._aero = AerodynamicsModel(params)
            self._actuator = ActuatorModel(params.actuator_response_time)

        if params.use_energy_model:
            self._energy = EnergyModel(params)

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
        accel_cmd = accel_cmd.copy().astype(float)

        # 0. 执行器延迟（可选）
        if self.params.use_aero_model and self._actuator is not None:
            accel_cmd = self._actuator.update(accel_cmd, dt)

        # 1. 限制加速度大小
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.params.max_acceleration:
            accel_cmd = accel_cmd * (self.params.max_acceleration / accel_mag)

        # 2. 载荷因子限制
        accel_lateral = accel_cmd.copy()
        accel_lateral[2] = 0  # 仅考虑水平分量
        n = np.linalg.norm(accel_lateral) / 9.81
        if n > self.params.n_max:
            accel_lateral = accel_lateral * (self.params.n_max / n)
            accel_cmd[0] = accel_lateral[0]
            accel_cmd[1] = accel_lateral[1]
        elif n < self.params.n_min:
            accel_lateral = accel_lateral * (self.params.n_min / n)
            accel_cmd[0] = accel_lateral[0]
            accel_cmd[1] = accel_lateral[1]

        # 3. 气动阻力（可选）
        if self.params.use_aero_model and self._aero is not None:
            drag_accel = self._aero.compute_drag_acceleration(state.velocity, state.altitude())
            accel_cmd = accel_cmd - drag_accel  # 阻力减速

        # 4. 能量消耗（可选）
        battery = state.battery_remaining
        if self.params.use_energy_model and self._energy is not None:
            battery = self._energy.update(accel_cmd, state.velocity, dt)
            if battery <= 0:
                # 电量耗尽，加速度归零
                accel_cmd = np.zeros(3)

        # 5. RK4积分
        new_position, new_velocity = self._rk4_step(state, accel_cmd, dt)

        # 6. 限制速度大小
        speed = np.linalg.norm(new_velocity)
        if speed > self.params.max_speed:
            new_velocity = new_velocity * (self.params.max_speed / speed)
        elif speed < self.params.min_speed and speed > 1e-6:
            new_velocity = new_velocity * (self.params.min_speed / speed)

        # 7. 限制爬升/下降率
        if new_velocity[2] > self.params.max_climb_rate:
            new_velocity[2] = self.params.max_climb_rate
        elif new_velocity[2] < -self.params.max_descent_rate:
            new_velocity[2] = -self.params.max_descent_rate

        # 8. 限制转弯率（水平面）- 速度相关转弯率
        if state.speed_horizontal() > 1e-6:
            current_heading = np.arctan2(state.velocity[1], state.velocity[0])
            new_heading = np.arctan2(new_velocity[1], new_velocity[0])
            heading_change = self._normalize_angle(new_heading - current_heading)

            speed = np.linalg.norm(new_velocity)
            if speed > 1e-6:
                # 物理正确：转弯率与速度成反比
                a_lateral_max = self.params.max_acceleration * 0.8
                omega_max = min(a_lateral_max / speed, self.params.max_turn_rate)
                max_heading_change = omega_max * dt
            else:
                max_heading_change = self.params.max_turn_rate * dt

            if abs(heading_change) > max_heading_change:
                clamped_heading = current_heading + np.sign(heading_change) * max_heading_change
                h_speed = np.linalg.norm(new_velocity[:2])
                new_velocity[0] = h_speed * np.cos(clamped_heading)
                new_velocity[1] = h_speed * np.sin(clamped_heading)

        # 9. 地面约束（不允许低于地面）
        if new_position[2] < 0:
            new_position[2] = 0
            new_velocity[2] = max(0, new_velocity[2])

        # 10. 实际加速度（考虑约束后的）
        actual_accel = (new_velocity - state.velocity) / dt if dt > 0 else np.zeros(3)

        # 11. 计算载荷因子
        accel_lateral_actual = actual_accel.copy()
        accel_lateral_actual[2] = 0
        load_factor = np.linalg.norm(accel_lateral_actual) / 9.81

        return DroneState(
            position=new_position,
            velocity=new_velocity,
            acceleration=actual_accel,
            time=state.time + dt,
            battery_remaining=battery,
            load_factor=load_factor
        )

    def _rk4_step(self, state: DroneState, accel_cmd: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4积分步进

        参数:
            state: 当前状态
            accel_cmd: 加速度指令（一步内视为常数）
            dt: 时间步长 (s)

        返回:
            (new_position, new_velocity)
        """
        # k1
        a1 = accel_cmd
        v1 = state.velocity
        # k2
        v2 = state.velocity + 0.5 * dt * a1
        a2 = accel_cmd  # 加速度指令在一步内不变
        # k3
        v3 = state.velocity + 0.5 * dt * a2
        a3 = accel_cmd
        # k4
        v4 = state.velocity + dt * a3
        a4 = accel_cmd

        new_velocity = state.velocity + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        new_position = state.position + dt / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        return new_position, new_velocity

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
