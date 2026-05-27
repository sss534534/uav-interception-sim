"""
无人机逃脱仿真系统 - 逃脱策略算法（优化版）
UAV Evasion Simulation - Evasion Strategy Algorithms (Optimized)

优化要点:
1. 正交机动: 突发-滑行模式 + 随机方向切换，破坏APN预测
2. 径向机动: 更激进的速度脉冲
3. 欺骗性: 延迟激活距离缩短，逃脱方向更随机
4. 混合策略: 追击方类型检测 + 自适应切换
5. 新增反预测机动: 检测追击方加速度方向并反向操作
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from drone_dynamics import DroneState, DroneParams


class ThreatLevel(Enum):
    """威胁等级"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ThreatAssessment:
    """威胁评估结果"""
    level: ThreatLevel
    distance: float
    time_to_intercept: float
    closing_speed: float
    los_unit: np.ndarray
    interceptor_speed: float

    def __post_init__(self):
        if not isinstance(self.los_unit, np.ndarray):
            self.los_unit = np.array(self.los_unit)


class EvasionStrategy(ABC):
    """逃脱策略抽象基类"""

    def __init__(self, params: DroneParams, name: str = "BaseEvasion"):
        self.params = params
        self.name = name
        self._time_in_current_mode = 0.0

    @abstractmethod
    def assess_threat(self, own_state: DroneState,
                      interceptor_state: DroneState) -> ThreatAssessment:
        pass

    @abstractmethod
    def compute_acceleration(self, own_state: DroneState,
                             interceptor_state: DroneState,
                             threat: ThreatAssessment,
                             dt: float) -> np.ndarray:
        pass

    def reset(self):
        self._time_in_current_mode = 0.0


class CollisionConeAnalyzer:
    """碰撞锥分析器"""

    def __init__(self, safety_margin: float = 1.5):
        self.safety_margin = safety_margin

    def compute_safe_directions(self, own_state: DroneState,
                                interceptor_state: DroneState,
                                interceptor_params: DroneParams) -> List[np.ndarray]:
        relative_pos = own_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)

        if distance < 1e-6:
            return []

        los_to_target = relative_pos / distance

        if interceptor_state.speed() > 1e-6:
            interceptor_dir = interceptor_state.velocity / interceptor_state.speed()
        else:
            interceptor_dir = los_to_target

        max_turn_rate = interceptor_params.max_turn_rate
        interceptor_speed = interceptor_state.speed()

        cone_half_angle = np.arctan2(
            interceptor_speed * np.sin(max_turn_rate * 0.5) * self.safety_margin,
            distance
        )
        cone_half_angle = min(cone_half_angle, np.pi / 3)

        cone_axis = interceptor_dir

        safe_directions = []
        n_samples = 16

        for i in range(n_samples):
            theta = 2 * np.pi * i / n_samples
            for j in range(4):
                phi = np.pi * (j + 1) / 5

                candidate = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])

                angle_to_cone_axis = np.arccos(
                    np.clip(np.dot(candidate, cone_axis), -1, 1)
                )

                if angle_to_cone_axis > cone_half_angle:
                    safety_score = angle_to_cone_axis - cone_half_angle

                    if own_state.speed() > 1e-6:
                        own_dir = own_state.velocity / own_state.speed()
                        continuity = np.dot(candidate, own_dir)
                        safety_score += continuity * 0.3

                    safe_directions.append((safety_score, candidate))

        safe_directions.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in safe_directions]


# ─────────────────────────────────────────────────────────────
# 共用威胁评估
# ─────────────────────────────────────────────────────────────

def _assess_threat(own_state: DroneState,
                   interceptor_state: DroneState) -> ThreatAssessment:
    """通用威胁评估"""
    relative_pos = own_state.position - interceptor_state.position
    distance = np.linalg.norm(relative_pos)

    if distance < 1e-6:
        los_unit = np.array([1.0, 0.0, 0.0])
    else:
        los_unit = relative_pos / distance

    relative_vel = interceptor_state.velocity - own_state.velocity
    closing_speed = -np.dot(relative_vel, los_unit)

    if closing_speed > 1.0:
        time_to_intercept = distance / closing_speed
    else:
        time_to_intercept = float('inf')

    if distance < 50:
        level = ThreatLevel.CRITICAL
    elif distance < 100 or time_to_intercept < 5:
        level = ThreatLevel.HIGH
    elif time_to_intercept < 15:
        level = ThreatLevel.MEDIUM
    else:
        level = ThreatLevel.LOW

    return ThreatAssessment(
        level=level, distance=distance,
        time_to_intercept=time_to_intercept,
        closing_speed=closing_speed, los_unit=los_unit,
        interceptor_speed=interceptor_state.speed()
    )


def _orthogonal_basis(los_unit: np.ndarray):
    """计算垂直于视线的两个正交基向量"""
    if abs(los_unit[2]) < 0.9:
        perp1 = np.cross(los_unit, np.array([0.0, 0.0, 1.0]))
    else:
        perp1 = np.cross(los_unit, np.array([0.0, 1.0, 0.0]))
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(los_unit, perp1)
    perp2 /= np.linalg.norm(perp2)
    return perp1, perp2


# ─────────────────────────────────────────────────────────────
# 1. 正交机动（优化版：突发-滑行 + 随机方向切换）
# ─────────────────────────────────────────────────────────────

class OrthogonalEvasion(EvasionStrategy):
    """
    正交机动逃脱策略（优化版）

    改进:
    - 突发-滑行模式: 高G机动后滑行，节省能量同时保持不可预测
    - 随机方向切换: 周期性反转正交方向，破坏APN的加速度估计
    - 自适应切换间隔: 距离越近切换越频繁
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Orthogonal Evasion")
        self.cone_analyzer = CollisionConeAnalyzer()
        self._orth_dir = None          # 当前正交方向
        self._burst_timer = 0.0        # 突发计时器
        self._switch_timer = 0.0       # 方向切换计时器
        self._switch_interval = 3.0    # 方向切换间隔

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt
        self._burst_timer += dt
        self._switch_timer += dt

        # ── 随机方向切换（破坏APN预测） ──
        # 距离越近切换越频繁
        if threat.distance < 100:
            self._switch_interval = 1.5
        elif threat.distance < 200:
            self._switch_interval = 2.5
        else:
            self._switch_interval = 4.0

        if self._switch_timer >= self._switch_interval or self._orth_dir is None:
            self._switch_timer = 0.0
            perp1, perp2 = _orthogonal_basis(threat.los_unit)
            # 随机选择正交方向（加入随机角度而非仅左右）
            angle = np.random.uniform(0, 2 * np.pi)
            self._orth_dir = np.cos(angle) * perp1 + np.sin(angle) * perp2

        # ── 突发-滑行模式 ──
        burst_duration = 1.5   # 突发持续
        coast_duration = 0.5   # 滑行持续（缩短，减少PP接近机会）
        cycle = burst_duration + coast_duration
        phase = self._burst_timer % cycle

        if phase < burst_duration:
            # 突发阶段: 最大加速度正交 + 径向远离
            if threat.level == ThreatLevel.CRITICAL:
                accel = (self._orth_dir * 0.6 + threat.los_unit * 0.4) * self.params.max_acceleration
            elif threat.level == ThreatLevel.HIGH:
                accel = (self._orth_dir * 0.7 + threat.los_unit * 0.3) * self.params.max_acceleration
            else:
                accel = (self._orth_dir * 0.8 + threat.los_unit * 0.2) * self.params.max_acceleration
        else:
            # 滑行阶段: 保持中等正交加速度 + 径向远离（不完全放弃机动）
            accel = (self._orth_dir * 0.4 + threat.los_unit * 0.6) * self.params.max_acceleration

        return accel

    def reset(self):
        super().reset()
        self._orth_dir = None
        self._burst_timer = 0.0
        self._switch_timer = 0.0


# ─────────────────────────────────────────────────────────────
# 2. 径向机动（优化版：更激进的速度脉冲）
# ─────────────────────────────────────────────────────────────

class RadialEvasion(EvasionStrategy):
    """
    径向机动逃脱策略（优化版）

    改进:
    - 更短更频繁的速度脉冲
    - 脉冲后加入随机正交分量
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Radial Evasion")
        self._phase = "burst"
        self._phase_timer = 0.0

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt
        self._phase_timer += dt

        burst_dur = 1.0
        coast_dur = 1.0

        if self._phase == "burst":
            if self._phase_timer > burst_dur:
                self._phase = "coast"
                self._phase_timer = 0.0
            # 突发加速远离 + 随机正交扰动
            perp1, _ = _orthogonal_basis(threat.los_unit)
            random_perp = perp1 * np.random.choice([-1, 1])
            accel = (threat.los_unit * 0.8 + random_perp * 0.2) * self.params.max_acceleration
        else:
            if self._phase_timer > coast_dur:
                self._phase = "burst"
                self._phase_timer = 0.0
            # 滑行: 微弱随机扰动
            noise = np.random.randn(3) * 0.2
            accel = noise * self.params.max_acceleration * 0.15

        return accel

    def reset(self):
        super().reset()
        self._phase = "burst"
        self._phase_timer = 0.0


# ─────────────────────────────────────────────────────────────
# 3. 欺骗性逃脱（优化版：更晚激活 + 更随机方向）
# ─────────────────────────────────────────────────────────────

class DeceptiveEvasion(EvasionStrategy):
    """
    欺骗性逃脱策略（优化版）

    改进:
    - 更晚激活（距离 < 60 才触发）
    - 逃脱方向完全随机
    - 逃脱后立即切换到突发-滑行模式
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Deceptive Evasion")
        self._deception_active = False
        self._evasion_dir = None
        self._post_evasion_timer = 0.0

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt

        if not self._deception_active:
            if threat.distance < 60 and threat.time_to_intercept < 6:
                self._deception_active = True
                # 随机选择逃脱方向
                perp1, perp2 = _orthogonal_basis(threat.los_unit)
                angle = np.random.uniform(0, 2 * np.pi)
                self._evasion_dir = np.cos(angle) * perp1 + np.sin(angle) * perp2
                self._post_evasion_timer = 0.0
            else:
                # 假装匀速
                noise = np.random.randn(3) * 0.05
                return noise * self.params.max_acceleration * 0.05

        # 逃脱阶段
        self._post_evasion_timer += dt

        # 前2秒: 最大加速度逃脱
        if self._post_evasion_timer < 2.0:
            accel = (self._evasion_dir * 0.7 + threat.los_unit * 0.3) * self.params.max_acceleration
        # 2-3秒: 滑行
        elif self._post_evasion_timer < 3.0:
            accel = threat.los_unit * self.params.max_acceleration * 0.2
        # 3秒后: 切换方向再次突发（二次逃脱）
        elif self._post_evasion_timer < 3.5:
            if abs(self._post_evasion_timer - 3.0) < dt * 1.1:
                # 重新随机方向
                perp1, perp2 = _orthogonal_basis(threat.los_unit)
                angle = np.random.uniform(0, 2 * np.pi)
                self._evasion_dir = np.cos(angle) * perp1 + np.sin(angle) * perp2
            accel = (self._evasion_dir * 0.6 + threat.los_unit * 0.4) * self.params.max_acceleration
        else:
            # 持续随机突发-滑行
            cycle = 2.0
            phase = self._post_evasion_timer % cycle
            if phase < 1.2:
                accel = (self._evasion_dir * 0.5 + threat.los_unit * 0.5) * self.params.max_acceleration
            else:
                accel = threat.los_unit * self.params.max_acceleration * 0.2

        return accel

    def reset(self):
        super().reset()
        self._deception_active = False
        self._evasion_dir = None
        self._post_evasion_timer = 0.0


# ─────────────────────────────────────────────────────────────
# 4. 反预测机动（新增）
# ─────────────────────────────────────────────────────────────

class AntiPredictiveEvasion(EvasionStrategy):
    """
    反预测逃脱策略

    原理: 检测追击方的加速度方向，选择相反方向机动
    对APN特别有效: APN的预测修正基于目标加速度估计，
    如果目标加速度方向与追击方期望的相反，预测修正会放大误差
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Anti-Predictive Evasion")
        self._prev_interceptor_accel = None
        self._interceptor_accel_estimate = np.zeros(3)
        self._alpha = 0.4
        self._orth_dir = None
        self._switch_timer = 0.0

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt
        self._switch_timer += dt

        # 估计追击方加速度方向
        if dt > 1e-6:
            measured = interceptor_state.acceleration
            self._interceptor_accel_estimate = (
                self._alpha * measured +
                (1 - self._alpha) * self._interceptor_accel_estimate
            )

        interceptor_accel_mag = np.linalg.norm(self._interceptor_accel_estimate)

        # 周期性切换正交方向（防止被预测）
        if self._switch_timer > 2.0 or self._orth_dir is None:
            self._switch_timer = 0.0
            perp1, perp2 = _orthogonal_basis(threat.los_unit)
            angle = np.random.uniform(0, 2 * np.pi)
            self._orth_dir = np.cos(angle) * perp1 + np.sin(angle) * perp2

        if interceptor_accel_mag > 1.0:
            # 追击方在机动: 选择反方向
            interceptor_accel_dir = self._interceptor_accel_estimate / interceptor_accel_mag

            # 计算追击方加速度在正交平面上的分量
            ortho_component = (interceptor_accel_dir -
                               np.dot(interceptor_accel_dir, threat.los_unit) * threat.los_unit)
            ortho_mag = np.linalg.norm(ortho_component)

            if ortho_mag > 0.1:
                # 反方向机动
                anti_dir = -ortho_component / ortho_mag
                # 混合反方向 + 随机正交方向
                accel = (anti_dir * 0.5 + self._orth_dir * 0.3 +
                         threat.los_unit * 0.2) * self.params.max_acceleration
            else:
                # 追击方主要径向机动，用正交方向逃脱
                accel = (self._orth_dir * 0.6 +
                         threat.los_unit * 0.4) * self.params.max_acceleration
        else:
            # 追击方未机动: 正交 + 径向远离
            accel = (self._orth_dir * 0.5 +
                     threat.los_unit * 0.5) * self.params.max_acceleration

        # 突发-滑行调制
        cycle = 2.0
        phase = self._time_in_current_mode % cycle
        if phase > 1.5:
            # 滑行阶段: 降低但不为零，保持正交分量
            accel = (self._orth_dir * 0.3 + threat.los_unit * 0.7) * self.params.max_acceleration

        return accel

    def reset(self):
        super().reset()
        self._prev_interceptor_accel = None
        self._interceptor_accel_estimate = np.zeros(3)
        self._orth_dir = None
        self._switch_timer = 0.0


# ─────────────────────────────────────────────────────────────
# 5. 混合策略（优化版：追击方类型检测 + 自适应切换）
# ─────────────────────────────────────────────────────────────

class HybridEvasion(EvasionStrategy):
    """
    混合逃脱策略（优化版）

    改进:
    - 检测追击方行为类型（PP/PN/APN）
    - 根据追击方类型选择最佳对抗策略
    - 距离自适应切换
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Hybrid Evasion")
        self.orthogonal = OrthogonalEvasion(params)
        self.anti_predictive = AntiPredictiveEvasion(params)
        self.deceptive = DeceptiveEvasion(params)

        self._current = self.orthogonal
        self._current_name = "orthogonal"

        # 追击方行为追踪
        self._interceptor_accel_history: List[float] = []
        self._interceptor_heading_history: List[float] = []
        self._detected_type = "unknown"  # pp, pn, apn

    def _detect_interceptor_type(self, interceptor_state: DroneState, dt: float):
        """检测追击方类型"""
        if dt < 1e-6:
            return

        # 记录追击方加速度大小
        self._interceptor_accel_history.append(np.linalg.norm(interceptor_state.acceleration))
        if len(self._interceptor_accel_history) > 60:
            self._interceptor_accel_history.pop(0)

        # 记录追击方航向变化
        if interceptor_state.speed() > 1e-6:
            heading = np.arctan2(interceptor_state.velocity[1], interceptor_state.velocity[0])
            self._interceptor_heading_history.append(heading)
            if len(self._interceptor_heading_history) > 60:
                self._interceptor_heading_history.pop(0)

        # 需要足够数据
        if len(self._interceptor_accel_history) < 20:
            return

        accel_mean = np.mean(self._interceptor_accel_history[-20:])
        accel_std = np.std(self._interceptor_accel_history[-20:])

        # PP: 持续高加速度，方向始终指向目标
        # PN: 中等加速度，航向变化与视线角速度相关
        # APN: 加速度有脉冲特征（突发-滑行），加速度估计导致修正项

        if accel_mean > 8.0 and accel_std < 3.0:
            self._detected_type = "pp"
        elif accel_std > 3.0:
            self._detected_type = "apn"  # APN有更不规则的加速度
        else:
            self._detected_type = "pn"

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt

        # 检测追击方类型
        self._detect_interceptor_type(interceptor_state, dt)

        # 根据追击方类型和距离选择策略
        if self._detected_type == "pp":
            # PP: 正交机动最有效
            desired = "orthogonal"
        elif self._detected_type == "apn":
            # APN: 反预测机动
            desired = "anti_predictive"
        else:
            # PN或未知: 正交机动
            desired = "orthogonal"

        # 距离很近时切换到欺骗性（已来不及用其他策略）
        if threat.distance < 60 and threat.time_to_intercept < 5:
            desired = "deceptive"

        # 切换策略
        if desired != self._current_name:
            if desired == "orthogonal":
                self.orthogonal.reset()
                self._current = self.orthogonal
            elif desired == "anti_predictive":
                self.anti_predictive.reset()
                self._current = self.anti_predictive
            elif desired == "deceptive":
                self.deceptive.reset()
                self._current = self.deceptive
            self._current_name = desired

        return self._current.compute_acceleration(
            own_state, interceptor_state, threat, dt
        )

    def reset(self):
        super().reset()
        self.orthogonal.reset()
        self.anti_predictive.reset()
        self.deceptive.reset()
        self._current = self.orthogonal
        self._current_name = "orthogonal"
        self._interceptor_accel_history.clear()
        self._interceptor_heading_history.clear()
        self._detected_type = "unknown"


# ─────────────────────────────────────────────────────────────
# 6. 针对 ZEM 的逃脱策略（新增）
# ─────────────────────────────────────────────────────────────

class HighGJinkEvasion(EvasionStrategy):
    """
    高G抖动逃脱策略

    原理: 持续高加速度 + 随机方向切换，破坏 ZEM 的零加速度假设
    ZEM 假设目标在 t_go 内保持当前加速度，如果加速度持续变化，
    ZEM 的预测会系统性偏差

    策略:
    - 始终以最大加速度机动
    - 每 0.5-1.5 秒随机切换方向
    - 方向选择偏向正交（最大化视线角速度）
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "High-G Jink Evasion")
        self._jink_dir = None
        self._switch_timer = 0.0
        self._next_switch_time = 1.0

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt
        self._switch_timer += dt

        # 随机切换时间（0.5-1.5秒）
        if self._switch_timer >= self._next_switch_time or self._jink_dir is None:
            self._switch_timer = 0.0
            self._next_switch_time = np.random.uniform(0.5, 1.5)

            # 选择新方向：主要正交 + 随机扰动
            perp1, perp2 = _orthogonal_basis(threat.los_unit)
            # 随机角度，但偏向正交方向
            angle = np.random.uniform(0, 2 * np.pi)
            # 加入径向分量，确保始终远离追击方
            radial_weight = 0.3
            ortho_weight = 0.7

            self._jink_dir = (ortho_weight * (np.cos(angle) * perp1 + np.sin(angle) * perp2) +
                              radial_weight * threat.los_unit)
            self._jink_dir /= np.linalg.norm(self._jink_dir)

        # 始终以最大加速度
        return self._jink_dir * self.params.max_acceleration

    def reset(self):
        super().reset()
        self._jink_dir = None
        self._switch_timer = 0.0
        self._next_switch_time = 1.0


class CloseRangeBreakEvasion(EvasionStrategy):
    """
    近距离突破逃脱策略

    原理: ZEM 在近距离（<150m）会混入 PN 项增强
    在临界点（约 150m）突然最大加速度垂直于视线方向，
    制造极大的视线角速度，使 PN 项发散

    策略:
    - 远距离: 匀速直线（节省能量，让 ZEM 低估威胁）
    - 中距离 (150-200m): 开始轻微机动
    - 近距离 (<150m): 突然最大G垂直机动
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "Close-Range Break Evasion")
        self._phase = "far"
        self._break_dir = None

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        self._time_in_current_mode += dt

        # 阶段判定
        if threat.distance > 250:
            self._phase = "far"
        elif threat.distance > 150:
            self._phase = "medium"
        else:
            if self._phase != "close":
                # 刚进入近距离，选择突破方向
                self._phase = "close"
                perp1, perp2 = _orthogonal_basis(threat.los_unit)
                # 选择与我方速度方向一致的正交方向（保持速度连续性）
                if own_state.speed() > 1e-6:
                    own_dir = own_state.velocity / own_state.speed()
                    proj1 = np.dot(own_dir, perp1)
                    proj2 = np.dot(own_dir, perp2)
                    if abs(proj1) > abs(proj2):
                        self._break_dir = perp1 if proj1 > 0 else -perp1
                    else:
                        self._break_dir = perp2 if proj2 > 0 else -perp2
                else:
                    self._break_dir = perp1

        # 根据阶段执行机动
        if self._phase == "far":
            # 远距离: 匀速直线 + 微小噪声
            noise = np.random.randn(3) * 0.05
            return noise * self.params.max_acceleration * 0.05

        elif self._phase == "medium":
            # 中距离: 轻微正交机动（预热）
            perp1, _ = _orthogonal_basis(threat.los_unit)
            return (perp1 * 0.3 + threat.los_unit * 0.3) * self.params.max_acceleration

        else:  # close
            # 近距离: 最大G突破
            if self._break_dir is not None:
                return (self._break_dir * 0.7 + threat.los_unit * 0.3) * self.params.max_acceleration
            else:
                perp1, _ = _orthogonal_basis(threat.los_unit)
                return (perp1 * 0.7 + threat.los_unit * 0.3) * self.params.max_acceleration

    def reset(self):
        super().reset()
        self._phase = "far"
        self._break_dir = None


class ZEMCounterEvasion(EvasionStrategy):
    """
    ZEM 对抗逃脱策略（终极混合）

    针对 ZEM 的多重弱点设计:
    1. 持续高G机动破坏零加速度假设
    2. 随机方向切换使加速度估计失效
    3. 近距离突然变向制造 PN 项发散
    4. 能量管理避免过早耗尽

    策略切换:
    - 远距离 (>300m): 高G抖动
    - 中距离 (100-300m): 随机加速度切换
    - 近距离 (<100m): 突然变向突破
    """

    def __init__(self, params: DroneParams):
        super().__init__(params, "ZEM Counter Evasion")

        # 子策略
        self.high_g = HighGJinkEvasion(params)
        self.close_break = CloseRangeBreakEvasion(params)

        self._current = self.high_g
        self._current_name = "high_g"

    def assess_threat(self, own_state, interceptor_state):
        return _assess_threat(own_state, interceptor_state)

    def compute_acceleration(self, own_state, interceptor_state, threat, dt):
        # 根据距离选择策略
        if threat.distance > 200:
            desired = "high_g"
        elif threat.distance > 100:
            # 中距离：随机切换高G和突破准备
            if self._time_in_current_mode > 3.0:
                desired = "close_break" if self._current_name == "high_g" else "high_g"
            else:
                desired = self._current_name
        else:
            desired = "close_break"

        # 切换策略
        if desired != self._current_name:
            if desired == "high_g":
                self.high_g.reset()
                self._current = self.high_g
            elif desired == "close_break":
                self.close_break.reset()
                self._current = self.close_break
            self._current_name = desired
            self._time_in_current_mode = 0.0

        return self._current.compute_acceleration(
            own_state, interceptor_state, threat, dt
        )

    def reset(self):
        super().reset()
        self.high_g.reset()
        self.close_break.reset()
        self._current = self.high_g
        self._current_name = "high_g"


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

def create_evasion(strategy_name: str, params: DroneParams) -> EvasionStrategy:
    """创建逃脱策略实例"""
    strategies = {
        "orthogonal": OrthogonalEvasion,
        "radial": RadialEvasion,
        "deceptive": DeceptiveEvasion,
        "anti_predictive": AntiPredictiveEvasion,
        "hybrid": HybridEvasion,
        "high_g_jink": HighGJinkEvasion,
        "close_break": CloseRangeBreakEvasion,
        "zem_counter": ZEMCounterEvasion,
    }

    if strategy_name not in strategies:
        raise ValueError(f"未知逃脱策略: {strategy_name}")

    return strategies[strategy_name](params)
