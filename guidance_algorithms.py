"""
无人机拦截仿真系统 - 拦截控制算法（升级版）
UAV Interception Simulation - Interception Control Algorithms (Upgraded)

原有算法:
1. 纯追击制导 (Pure Pursuit)
2. 比例导航制导 (PN)
3. 增强型比例导航制导 (APN)

新增升级算法:
4. 升级版APN (APN-v2): 快速加速度估计 + 自适应增益 + 突发-滑行检测
5. 零脱靶量制导 (ZEM): 基于最优控制理论
6. 自适应制导 (Adaptive): 检测逃脱类型并切换策略
"""

import numpy as np
from typing import Optional, Tuple, List

from drone_dynamics import DroneState, DroneParams


class InterceptionResult:
    """拦截结果记录"""
    def __init__(self):
        self.intercepted = False
        self.miss_distance = float('inf')
        self.intercept_time = None
        self.total_time = 0
        self.energy_consumed = 0.0


# ─────────────────────────────────────────────────────────────
# 共用工具函数
# ─────────────────────────────────────────────────────────────

def _bootstrap(own_state: DroneState, target_state: DroneState,
               params: DroneParams, bootstrap_steps: float,
               bootstrap_duration: float) -> Tuple[np.ndarray, bool]:
    """
    初始引导阶段：拦截机速度不足时，使用纯追击加速
    返回: (accel_cmd, is_bootstrap)
    """
    if own_state.speed() < params.max_speed * 0.5 and \
            bootstrap_steps < bootstrap_duration:
        relative_pos = target_state.position - own_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3), True
        desired_dir = relative_pos / distance
        desired_vel = desired_dir * params.max_speed
        velocity_error = desired_vel - own_state.velocity
        return 3.0 * velocity_error, True
    return None, False


def _compute_los_rate(los_unit: np.ndarray, prev_los_unit: Optional[np.ndarray],
                      prev_time: Optional[float], current_time: float) -> np.ndarray:
    """计算视线角速度"""
    if prev_los_unit is not None and prev_time is not None:
        actual_dt = current_time - prev_time
        if actual_dt > 1e-6:
            return (los_unit - prev_los_unit) / actual_dt
    return np.zeros(3)


def _ensure_los_initialized(prev_los_unit, prev_time, los_unit, current_time):
    """确保视线信息已初始化（处理尾追等场景）"""
    if prev_los_unit is None:
        return los_unit.copy(), current_time
    return prev_los_unit, prev_time


# ─────────────────────────────────────────────────────────────
# 1. 纯追击制导 (Pure Pursuit) — 保持不变
# ─────────────────────────────────────────────────────────────

class PurePursuitGuidance:
    """纯追击制导算法"""

    def __init__(self, interceptor_params: DroneParams):
        self.params = interceptor_params
        self.name = "Pure Pursuit"
        self.name_en = "Pure Pursuit"

    def compute_acceleration(self, interceptor_state, target_state, dt):
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3)
        desired_dir = relative_pos / distance
        desired_vel = desired_dir * self.params.max_speed
        velocity_error = desired_vel - interceptor_state.velocity
        return 3.0 * velocity_error

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# 2. 比例导航制导 (PN) — 保持不变
# ─────────────────────────────────────────────────────────────

class ProportionalNavigationGuidance:
    """比例导航制导算法 (PN)"""

    def __init__(self, interceptor_params: DroneParams, N: float = 4.0):
        self.params = interceptor_params
        self.N = N
        self.name = f"PN (N={N})"
        self.name_en = f"PN (N={N})"
        self._prev_los_unit = None
        self._prev_time = None
        self._bootstrap_steps = 0.0
        self._bootstrap_duration = 1.0

    def compute_acceleration(self, interceptor_state, target_state, dt):
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3)

        los_unit = relative_pos / distance
        self._prev_los_unit, self._prev_time = _ensure_los_initialized(
            self._prev_los_unit, self._prev_time, los_unit, interceptor_state.time
        )

        relative_vel = target_state.velocity - interceptor_state.velocity
        v_closing = -np.dot(relative_vel, los_unit)

        self._bootstrap_steps += dt
        accel_cmd, is_boot = _bootstrap(
            interceptor_state, target_state, self.params,
            self._bootstrap_steps, self._bootstrap_duration
        )
        if is_boot:
            self._prev_los_unit = los_unit.copy()
            self._prev_time = interceptor_state.time
            return accel_cmd

        los_rate = _compute_los_rate(
            los_unit, self._prev_los_unit, self._prev_time, interceptor_state.time
        )

        accel_cmd = self.N * v_closing * los_rate

        if v_closing < 0:
            accel_cmd += los_unit * abs(v_closing) * 0.5

        self._prev_los_unit = los_unit.copy()
        self._prev_time = interceptor_state.time
        return accel_cmd

    def reset(self):
        self._prev_los_unit = None
        self._prev_time = None
        self._bootstrap_steps = 0.0


# ─────────────────────────────────────────────────────────────
# 3. 增强型比例导航制导 (APN) — 保持不变
# ─────────────────────────────────────────────────────────────

class AugmentedPNGuidance:
    """增强型比例导航制导 (APN)"""

    def __init__(self, interceptor_params: DroneParams, N: float = 4.0):
        self.params = interceptor_params
        self.N = N
        self.name = f"APN (N={N})"
        self.name_en = f"APN (N={N})"
        self._prev_los_unit = None
        self._prev_time = None
        self._target_accel_estimate = np.zeros(3)
        self._alpha = 0.3
        self._bootstrap_steps = 0.0
        self._bootstrap_duration = 1.0

    def compute_acceleration(self, interceptor_state, target_state, dt):
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3)

        los_unit = relative_pos / distance
        self._prev_los_unit, self._prev_time = _ensure_los_initialized(
            self._prev_los_unit, self._prev_time, los_unit, interceptor_state.time
        )

        relative_vel = target_state.velocity - interceptor_state.velocity
        v_closing = -np.dot(relative_vel, los_unit)

        if dt > 1e-6:
            self._target_accel_estimate = (
                self._alpha * target_state.acceleration +
                (1 - self._alpha) * self._target_accel_estimate
            )

        self._bootstrap_steps += dt
        accel_cmd, is_boot = _bootstrap(
            interceptor_state, target_state, self.params,
            self._bootstrap_steps, self._bootstrap_duration
        )
        if is_boot:
            self._prev_los_unit = los_unit.copy()
            self._prev_time = interceptor_state.time
            return accel_cmd

        a_target_perp = (self._target_accel_estimate -
                         np.dot(self._target_accel_estimate, los_unit) * los_unit)

        los_rate = _compute_los_rate(
            los_unit, self._prev_los_unit, self._prev_time, interceptor_state.time
        )

        accel_cmd = self.N * v_closing * los_rate + (self.N / 2.0) * a_target_perp

        pip_correction = self._predictive_correction(
            interceptor_state, target_state, los_unit, distance, v_closing
        )
        accel_cmd += pip_correction

        if v_closing < 0:
            accel_cmd += los_unit * abs(v_closing) * 0.5

        self._prev_los_unit = los_unit.copy()
        self._prev_time = interceptor_state.time
        return accel_cmd

    def _predictive_correction(self, interceptor_state, target_state,
                                los_unit, distance, v_closing):
        if v_closing < 2.0 or distance > 800:
            return np.zeros(3)
        t_go = min(distance / max(v_closing, 1.0), 5.0)
        predicted_pos = (target_state.position +
                         target_state.velocity * t_go +
                         0.5 * self._target_accel_estimate * t_go ** 2)
        desired_rel = predicted_pos - interceptor_state.position
        desired_dist = np.linalg.norm(desired_rel)
        if desired_dist < 1e-6:
            return np.zeros(3)
        correction = desired_rel / desired_dist - los_unit
        if np.linalg.norm(correction) < 1e-6:
            return np.zeros(3)
        if distance < 100:
            gain = 0.3
        elif distance < 300:
            gain = 0.8
        else:
            gain = 0.4
        return correction * gain

    def reset(self):
        self._prev_los_unit = None
        self._prev_time = None
        self._target_accel_estimate = np.zeros(3)
        self._bootstrap_steps = 0.0


# ─────────────────────────────────────────────────────────────
# 4. 升级版 APN (APN-v2)
# ─────────────────────────────────────────────────────────────

class APNv2Guidance:
    """
    升级版增强比例导航制导 (APN-v2)

    针对逃脱策略的改进:
    1. 快速加速度估计 (α=0.6): 更快跟踪目标加速度变化
    2. 突发-滑行检测: 检测目标加速度突然消失（滑行阶段），
       在滑行阶段直接飞向预测拦截点而非依赖过时的加速度估计
    3. 自适应导航增益: 远距离用高N快速修正，近距离用低N避免过冲
    4. 改进的预测修正: 使用加权历史加速度而非单次估计
    """

    def __init__(self, interceptor_params: DroneParams, N: float = 4.0):
        self.params = interceptor_params
        self.N_base = N
        self.name = f"APN-v2 (N={N})"
        self.name_en = f"APN-v2 (N={N})"

        self._prev_los_unit = None
        self._prev_time = None
        self._bootstrap_steps = 0.0
        self._bootstrap_duration = 0.8

        # 快速加速度估计
        self._target_accel_estimate = np.zeros(3)
        self._alpha = 0.6  # 更快的跟踪速度

        # 加速度历史（用于突发-滑行检测和加权预测）
        self._accel_history: List[np.ndarray] = []
        self._accel_history_max = 20

        # 突发-滑行检测
        self._prev_target_accel_mag = 0.0
        self._is_coasting = False
        self._coast_counter = 0

    def compute_acceleration(self, interceptor_state, target_state, dt):
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3)

        los_unit = relative_pos / distance
        self._prev_los_unit, self._prev_time = _ensure_los_initialized(
            self._prev_los_unit, self._prev_time, los_unit, interceptor_state.time
        )

        relative_vel = target_state.velocity - interceptor_state.velocity
        v_closing = -np.dot(relative_vel, los_unit)

        # ── 快速加速度估计 ──
        if dt > 1e-6:
            self._target_accel_estimate = (
                self._alpha * target_state.acceleration +
                (1 - self._alpha) * self._target_accel_estimate
            )
            # 记录历史
            self._accel_history.append(target_state.acceleration.copy())
            if len(self._accel_history) > self._accel_history_max:
                self._accel_history.pop(0)

        # ── 突发-滑行检测 ──
        current_accel_mag = np.linalg.norm(target_state.acceleration)
        if self._prev_target_accel_mag > self.params.max_acceleration * 0.5 and \
                current_accel_mag < self.params.max_acceleration * 0.15:
            self._coast_counter += 1
        else:
            self._coast_counter = max(0, self._coast_counter - 2)

        self._is_coasting = self._coast_counter > 3  # 连续3拍低加速度判定为滑行
        self._prev_target_accel_mag = current_accel_mag

        # ── Bootstrap ──
        self._bootstrap_steps += dt
        accel_cmd, is_boot = _bootstrap(
            interceptor_state, target_state, self.params,
            self._bootstrap_steps, self._bootstrap_duration
        )
        if is_boot:
            self._prev_los_unit = los_unit.copy()
            self._prev_time = interceptor_state.time
            return accel_cmd

        # ── 自适应导航增益 ──
        if distance > 300:
            N = self.N_base * 1.3  # 远距离高增益
        elif distance < 80:
            N = self.N_base * 0.7  # 近距离低增益避免过冲
        else:
            N = self.N_base

        # ── 视线角速度 ──
        los_rate = _compute_los_rate(
            los_unit, self._prev_los_unit, self._prev_time, interceptor_state.time
        )

        # ── 目标加速度补偿 ──
        if self._is_coasting:
            # 滑行阶段: 不使用加速度补偿（估计已过时）
            # 转而使用历史平均加速度的衰减版本
            if len(self._accel_history) > 5:
                # 使用最近5步的平均值，但乘以衰减因子
                recent_avg = np.mean(self._accel_history[-5:], axis=0)
                a_target_perp = (recent_avg -
                                 np.dot(recent_avg, los_unit) * los_unit)
                a_target_perp *= 0.3  # 衰减
            else:
                a_target_perp = np.zeros(3)
        else:
            a_target_perp = (self._target_accel_estimate -
                             np.dot(self._target_accel_estimate, los_unit) * los_unit)

        # ── APN 基础指令 ──
        accel_cmd = N * v_closing * los_rate + (N / 2.0) * a_target_perp

        # ── 改进的预测修正 ──
        pip = self._improved_predictive_correction(
            interceptor_state, target_state, los_unit, distance, v_closing
        )
        accel_cmd += pip

        # ── 远离修正 ──
        if v_closing < 0:
            accel_cmd += los_unit * abs(v_closing) * 0.5

        self._prev_los_unit = los_unit.copy()
        self._prev_time = interceptor_state.time
        return accel_cmd

    def _improved_predictive_correction(self, interceptor_state, target_state,
                                         los_unit, distance, v_closing):
        """改进的预测拦截点修正"""
        if v_closing < 2.0 or distance > 1000:
            return np.zeros(3)

        t_go = min(distance / max(v_closing, 1.0), 8.0)

        # 使用历史加速度的加权平均进行预测
        if len(self._accel_history) > 3:
            # 近期权重更大
            weights = np.exp(np.linspace(-1, 0, len(self._accel_history)))
            weights /= weights.sum()
            weighted_accel = np.zeros(3)
            for w, a in zip(weights, self._accel_history):
                weighted_accel += w * a
        else:
            weighted_accel = self._target_accel_estimate

        predicted_pos = (target_state.position +
                         target_state.velocity * t_go +
                         0.5 * weighted_accel * t_go ** 2)

        desired_rel = predicted_pos - interceptor_state.position
        desired_dist = np.linalg.norm(desired_rel)
        if desired_dist < 1e-6:
            return np.zeros(3)

        desired_dir = desired_rel / desired_dist
        correction = desired_dir - los_unit
        correction_mag = np.linalg.norm(correction)
        if correction_mag < 1e-6:
            return np.zeros(3)

        # 自适应增益
        if distance < 80:
            gain = 0.4
        elif distance < 200:
            gain = 1.0
        elif distance < 500:
            gain = 0.6
        else:
            gain = 0.3

        return correction * gain

    def reset(self):
        self._prev_los_unit = None
        self._prev_time = None
        self._target_accel_estimate = np.zeros(3)
        self._bootstrap_steps = 0.0
        self._accel_history.clear()
        self._prev_target_accel_mag = 0.0
        self._is_coasting = False
        self._coast_counter = 0


# ─────────────────────────────────────────────────────────────
# 5. 零脱靶量制导 (ZEM)
# ─────────────────────────────────────────────────────────────

class ZEMGuidance:
    """
    零脱靶量制导 (Zero Effort Miss - ZEM)

    原理: 基于最优控制理论，计算使脱靶量为零所需的加速度
    a_cmd = N * ZEM / t_go^2

    其中 ZEM = r + v_rel * t_go + 0.5 * a_rel * t_go^2
    (零加速度下预测的最终脱靶距离)

    优势:
    - 理论最优（最小能量拦截）
    - 自然处理机动目标
    - 对突发-滑行模式有天然抗性（基于位置/速度而非加速度估计）
    """

    def __init__(self, interceptor_params: DroneParams, N: float = 3.0):
        self.params = interceptor_params
        self.N = N
        self.name = f"ZEM (N={N})"
        self.name_en = f"ZEM (N={N})"
        self._bootstrap_steps = 0.0
        self._bootstrap_duration = 0.8

        # 目标加速度估计（仅用于ZEM预测，不直接用于制导指令）
        self._target_accel_estimate = np.zeros(3)
        self._alpha = 0.5

    def compute_acceleration(self, interceptor_state, target_state, dt):
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        if distance < 1e-6:
            return np.zeros(3)

        los_unit = relative_pos / distance
        relative_vel = target_state.velocity - interceptor_state.velocity
        v_closing = -np.dot(relative_vel, los_unit)

        # 加速度估计
        if dt > 1e-6:
            self._target_accel_estimate = (
                self._alpha * target_state.acceleration +
                (1 - self._alpha) * self._target_accel_estimate
            )

        # Bootstrap
        self._bootstrap_steps += dt
        accel_cmd, is_boot = _bootstrap(
            interceptor_state, target_state, self.params,
            self._bootstrap_steps, self._bootstrap_duration
        )
        if is_boot:
            return accel_cmd

        # 估计剩余飞行时间
        if v_closing > 2.0:
            t_go = distance / v_closing
        else:
            t_go = distance / max(interceptor_state.speed(), 1.0)

        t_go = np.clip(t_go, 0.5, 20.0)

        # 计算零脱靶量 (ZEM)
        # 如果双方都不加速，最终会偏多远？
        relative_accel = self._target_accel_estimate  # 拦截机加速度为0（假设）
        zem = (relative_pos +
               relative_vel * t_go +
               0.5 * relative_accel * t_go ** 2)

        # ZEM制导指令: a = N * ZEM / t_go^2
        # 这直接指向"需要修正的位置偏差"
        accel_cmd = self.N * zem / (t_go ** 2)

        # 限制指令大小
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.params.max_acceleration:
            accel_cmd = accel_cmd * (self.params.max_acceleration / accel_mag)

        # 近距离增强：加入PN项防止视线角速度发散
        if distance < 150:
            los_rate = np.zeros(3)
            # 使用速度叉积计算视线角速度（比差分更稳定）
            if distance > 1e-6:
                omega = np.cross(relative_pos, relative_vel) / (distance ** 2)
                # 将角速度转换为线加速度方向
                los_rate_correction = np.cross(omega, los_unit) * v_closing
                pn_gain = 2.0 * (1.0 - distance / 150.0)  # 距离越近PN项越大
                accel_cmd += pn_gain * los_rate_correction

        # 远离修正
        if v_closing < 0:
            accel_cmd += los_unit * abs(v_closing) * 0.3

        return accel_cmd

    def reset(self):
        self._bootstrap_steps = 0.0
        self._target_accel_estimate = np.zeros(3)


# ─────────────────────────────────────────────────────────────
# 6. 混合制导 (Hybrid) — 替代原 Adaptive
# ─────────────────────────────────────────────────────────────

class HybridGuidance:
    """
    混合制导 (Hybrid Guidance)

    核心创新：同时计算 ZEM 和 APN-v2 指令，按权重融合
    a_cmd = w * a_zem + (1-w) * a_apn_v2

    权重策略：
    - 远距离 (>300m): APN-v2 权重高 (w=0.3)，快速修正
    - 中距离 (100-300m): 均衡 (w=0.5)
    - 近距离 (<100m): ZEM 权重高 (w=0.7)，精确拦截

    逃脱类型检测：
    - 反预测机动 → 提高 ZEM 权重（ZEM 免疫反预测）
    - 正交机动 → 保持均衡
    - 突发-滑行 → 略提高 ZEM 权重
    """

    def __init__(self, interceptor_params: DroneParams):
        self.params = interceptor_params
        self.name = "Hybrid Guidance"
        self.name_en = "Hybrid"

        # 子策略
        self.zem = ZEMGuidance(interceptor_params, N=3.0)
        self.apn_v2 = APNv2Guidance(interceptor_params, N=4.0)

        # 目标行为追踪
        self._target_accel_mag_history: List[float] = []
        self._target_heading_changes: List[float] = []
        self._prev_target_heading = None
        self._detected_evasion = "unknown"

    def _detect_evasion_type(self, target_state: DroneState, dt: float):
        """检测目标逃脱类型"""
        if dt < 1e-6:
            return

        accel_mag = np.linalg.norm(target_state.acceleration)
        self._target_accel_mag_history.append(accel_mag)
        if len(self._target_accel_mag_history) > 40:
            self._target_accel_mag_history.pop(0)

        if target_state.speed() > 1e-6:
            heading = np.arctan2(target_state.velocity[1], target_state.velocity[0])
            if self._prev_target_heading is not None:
                dh = abs(heading - self._prev_target_heading)
                if dh > np.pi:
                    dh = 2 * np.pi - dh
                self._target_heading_changes.append(dh / dt)
                if len(self._target_heading_changes) > 40:
                    self._target_heading_changes.pop(0)
            self._prev_target_heading = heading

        if len(self._target_accel_mag_history) < 15:
            return

        recent = self._target_accel_mag_history[-15:]
        accel_mean = np.mean(recent)
        accel_std = np.std(recent)

        if accel_std > 3.0 and accel_mean > 2.0:
            self._detected_evasion = "burst_coast"
        elif accel_mean > 4.0 and len(self._target_heading_changes) > 10:
            heading_rate_mean = np.mean(self._target_heading_changes[-10:])
            if heading_rate_mean > 0.3:
                self._detected_evasion = "anti_predictive"
            else:
                self._detected_evasion = "orthogonal"
        elif accel_mean > 2.0 and len(self._target_heading_changes) > 10:
            self._detected_evasion = "orthogonal"
        elif accel_mean < 1.5:
            self._detected_evasion = "low maneuvers"
        else:
            self._detected_evasion = "unknown"

    def _compute_weight(self, distance: float) -> float:
        """计算 ZEM 权重 (0-1)"""
        # 核心洞察：ZEM 对大多数逃脱策略都有效，应该作为主力
        # APN-v2 仅在远距离作为快速修正辅助

        # 基础权重：始终以 ZEM 为主
        if distance > 400:
            w_base = 0.5  # 远距离均衡
        elif distance > 200:
            w_base = 0.65
        elif distance > 100:
            w_base = 0.75
        elif distance > 50:
            w_base = 0.85
        else:
            w_base = 0.95  # 近距离几乎纯 ZEM

        # 根据逃脱类型微调
        if self._detected_evasion == "anti_predictive":
            # 反预测：ZEM 完全免疫，大幅提高权重
            w_adj = w_base + 0.1
        elif self._detected_evasion == "burst_coast":
            # 突发-滑行：ZEM 有抗性，提高权重
            w_adj = w_base + 0.05
        elif self._detected_evasion == "orthogonal":
            # 正交机动：ZEM 也很有效，提高权重
            w_adj = w_base + 0.05
        else:
            w_adj = w_base

        return np.clip(w_adj, 0.3, 0.98)

    def compute_acceleration(self, interceptor_state, target_state, dt):
        self._detect_evasion_type(target_state, dt)

        # 计算相对位置和距离
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)

        # 同时计算两个指令
        a_zem = self.zem.compute_acceleration(interceptor_state, target_state, dt)
        a_apn = self.apn_v2.compute_acceleration(interceptor_state, target_state, dt)

        # 计算权重
        w = self._compute_weight(distance)

        # 混合指令
        accel_cmd = w * a_zem + (1 - w) * a_apn

        # 限制加速度大小
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.params.max_acceleration:
            accel_cmd = accel_cmd * (self.params.max_acceleration / accel_mag)

        return accel_cmd

    def reset(self):
        self.zem.reset()
        self.apn_v2.reset()
        self._target_accel_mag_history.clear()
        self._target_heading_changes.clear()
        self._prev_target_heading = None
        self._detected_evasion = "unknown"


# 保留 AdaptiveGuidance 作为别名
AdaptiveGuidance = HybridGuidance


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

def create_guidance(algorithm_name: str, params: DroneParams, **kwargs) -> object:
    """创建制导算法实例"""
    N = kwargs.get('N', 4.0)

    strategies = {
        'pure_pursuit': lambda: PurePursuitGuidance(params),
        'pn': lambda: ProportionalNavigationGuidance(params, N=N),
        'apn': lambda: AugmentedPNGuidance(params, N=N),
        'apn_v2': lambda: APNv2Guidance(params, N=N),
        'zem': lambda: ZEMGuidance(params, N=N),
        'adaptive': lambda: AdaptiveGuidance(params),
    }

    if algorithm_name not in strategies:
        raise ValueError(f"未知算法: {algorithm_name}")

    return strategies[algorithm_name]()
