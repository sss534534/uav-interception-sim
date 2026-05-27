"""
无人机拦截仿真系统 - Pygame 实时动画
UAV Interception Simulation - Real-time Animation

使用 Pygame 实现3D拦截过程的实时可视化
支持视角旋转、缩放、暂停等交互功能
"""

import numpy as np
import sys
import math

try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("Pygame 未安装，请运行: pip install pygame")
    sys.exit(1)

from drone_dynamics import DroneState, DroneParams, DroneDynamics, TargetStrategy
from guidance_algorithms import create_guidance
from simulation_engine import SimulationEngine, SimulationConfig, ScenarioFactory


class Camera:
    """3D相机 - 支持旋转和缩放"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.distance = 1200.0
        self.azimuth = math.radians(45)    # 水平旋转角
        self.elevation = math.radians(30)   # 俯仰角
        self.target = np.array([400.0, 200.0, 50.0])  # 注视点
        self.fov = 60.0

    def project(self, point_3d):
        """将3D点投影到2D屏幕坐标"""
        # 平移到相机坐标系
        p = point_3d - self.target

        # 旋转
        cos_a = math.cos(self.azimuth)
        sin_a = math.sin(self.azimuth)
        cos_e = math.cos(self.elevation)
        sin_e = math.sin(self.elevation)

        # 绕Z轴旋转 (方位角)
        x1 = p[0] * cos_a + p[1] * sin_a
        y1 = -p[0] * sin_a + p[1] * cos_a
        z1 = p[2]

        # 绕X轴旋转 (俯仰角)
        x2 = x1
        y2 = y1 * cos_e - z1 * sin_e
        z2 = y1 * sin_e + z1 * cos_e

        # 透视投影
        z_offset = self.distance
        z_proj = z2 + z_offset

        if z_proj < 1.0:
            z_proj = 1.0

        scale = (self.width / 2) / math.tan(math.radians(self.fov / 2))
        sx = int(self.width / 2 + x2 * scale / z_proj)
        sy = int(self.height / 2 - y2 * scale / z_proj)

        return sx, sy, z_proj

    def rotate(self, d_azimuth, d_elevation):
        """旋转相机"""
        self.azimuth += d_azimuth
        self.elevation += d_elevation
        self.elevation = max(math.radians(-89), min(math.radians(89), self.elevation))

    def zoom(self, factor):
        """缩放"""
        self.distance *= factor
        self.distance = max(100, min(5000, self.distance))

    def pan(self, dx, dy):
        """平移注视点"""
        cos_a = math.cos(self.azimuth)
        sin_a = math.sin(self.azimuth)
        self.target[0] += (-sin_a * dx + cos_a * dy) * self.distance * 0.001
        self.target[1] += (cos_a * dx + sin_a * dy) * self.distance * 0.001


class RealtimeAnimation:
    """实时动画系统"""

    # 颜色定义
    BG_COLOR = (15, 15, 30)
    GRID_COLOR = (40, 40, 60)
    INTERCEPTOR_COLOR = (30, 144, 255)     # 蓝色
    TARGET_COLOR = (255, 60, 60)           # 红色
    TRAIL_INTERCEPTOR = (30, 100, 200, 128)
    TRAIL_TARGET = (200, 50, 50, 128)
    TEXT_COLOR = (220, 220, 220)
    SUCCESS_COLOR = (50, 255, 50)
    FAIL_COLOR = (255, 50, 50)
    PANEL_COLOR = (25, 25, 50, 200)

    def __init__(self, config: SimulationConfig, algorithm_name: str = 'apn',
                 speed_multiplier: float = 1.0):
        self.config = config
        self.algorithm_name = algorithm_name
        self.speed_multiplier = speed_multiplier

        # Pygame 初始化
        pygame.init()
        self.screen_w = 1280
        self.screen_h = 720
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h),
                                              pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption(f'UAV Interception - {algorithm_name.upper()}')

        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('consolas', 20)
        self.font_medium = pygame.font.SysFont('consolas', 16)
        self.font_small = pygame.font.SysFont('consolas', 13)

        # 相机
        self.camera = Camera(self.screen_w, self.screen_h)

        # 仿真引擎
        self.engine = SimulationEngine(config)
        guidance = create_guidance(algorithm_name, DroneParams.interceptor())
        self.engine.set_guidance(guidance)
        self.guidance_name = guidance.name

        # 轨迹历史
        self.interceptor_trail = []
        self.target_trail = []
        self.max_trail = 2000

        # 状态
        self.paused = False
        self.running = True
        self.finished = False
        self.sim_speed = speed_multiplier
        self.show_info = True
        self.view_3d = True  # True=3D视角, False=俯视图

        # 鼠标状态
        self.mouse_dragging = False
        self.last_mouse_pos = None

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self._reset()
                elif event.key == K_UP:
                    self.sim_speed = min(10.0, self.sim_speed * 1.5)
                elif event.key == K_DOWN:
                    self.sim_speed = max(0.1, self.sim_speed / 1.5)
                elif event.key == K_i:
                    self.show_info = not self.show_info
                elif event.key == K_v:
                    self.view_3d = not self.view_3d
                elif event.key == K_1:
                    self._switch_algorithm('pure_pursuit')
                elif event.key == K_2:
                    self._switch_algorithm('pn')
                elif event.key == K_3:
                    self._switch_algorithm('apn')

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # 滚轮上
                    self.camera.zoom(0.9)
                elif event.button == 5:  # 滚轮下
                    self.camera.zoom(1.1)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False

            elif event.type == MOUSEMOTION:
                if self.mouse_dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    if self.view_3d:
                        self.camera.rotate(dx * 0.005, dy * 0.005)
                    else:
                        self.camera.pan(-dx, -dy)
                    self.last_mouse_pos = event.pos

            elif event.type == VIDEORESIZE:
                self.screen_w, self.screen_h = event.w, event.h
                self.screen = pygame.display.set_mode((self.screen_w, self.screen_h),
                                                      pygame.RESIZABLE | pygame.DOUBLEBUF)
                self.camera.width = self.screen_w
                self.camera.height = self.screen_h

    def _reset(self):
        """重置仿真"""
        self.engine = SimulationEngine(self.config)
        guidance = create_guidance(self.algorithm_name, DroneParams.interceptor())
        self.engine.set_guidance(guidance)
        self.interceptor_trail.clear()
        self.target_trail.clear()
        self.finished = False
        self.paused = False

    def _switch_algorithm(self, algo_name):
        """切换算法"""
        self.algorithm_name = algo_name
        self._reset()

    def update(self):
        """更新仿真"""
        if self.paused or self.finished:
            return

        # 根据速度倍率执行多步
        steps = max(1, int(self.sim_speed))
        for _ in range(steps):
            if not self.engine.step():
                self.finished = True
                break

        # 记录轨迹
        i_state, t_state = self.engine.get_current_states()
        self.interceptor_trail.append(i_state.position.copy())
        self.target_trail.append(t_state.position.copy())

        if len(self.interceptor_trail) > self.max_trail:
            self.interceptor_trail.pop(0)
        if len(self.target_trail) > self.max_trail:
            self.target_trail.pop(0)

    def draw(self):
        """绘制画面"""
        self.screen.fill(self.BG_COLOR)

        if self.view_3d:
            self._draw_3d_view()
        else:
            self._draw_top_view()

        if self.show_info:
            self._draw_info_panel()

        if self.finished:
            self._draw_result()

        pygame.display.flip()

    def _draw_3d_view(self):
        """绘制3D视角"""
        # 绘制地面网格
        self._draw_ground_grid()

        # 绘制轨迹
        self._draw_trail_3d(self.interceptor_trail, self.INTERCEPTOR_COLOR)
        self._draw_trail_3d(self.target_trail, self.TARGET_COLOR)

        # 绘制无人机
        i_state, t_state = self.engine.get_current_states()

        # 拦截机
        sx, sy, sz = self.camera.project(i_state.position)
        size = max(4, int(800 / sz))
        pygame.draw.circle(self.screen, self.INTERCEPTOR_COLOR, (sx, sy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), size, 1)

        # 速度方向指示
        if i_state.speed() > 0.5:
            vel_end = i_state.position + i_state.velocity * 2
            ex, ey, ez = self.camera.project(vel_end)
            pygame.draw.line(self.screen, (100, 200, 255), (sx, sy), (ex, ey), 2)

        # 目标机
        sx, sy, sz = self.camera.project(t_state.position)
        size = max(4, int(800 / sz))
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (sx, sy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), size, 1)

        # 目标速度方向
        if t_state.speed() > 0.5:
            vel_end = t_state.position + t_state.velocity * 2
            ex, ey, ez = self.camera.project(vel_end)
            pygame.draw.line(self.screen, (255, 150, 150), (sx, sy), (ex, ey), 2)

        # 拦截点标记
        if self.engine.result.intercepted and len(self.interceptor_trail) > 0:
            idx = int(np.argmin(np.array(self.engine.record.distances)))
            if idx < len(self.interceptor_trail):
                px, py, pz = self.camera.project(self.interceptor_trail[idx])
                pygame.draw.circle(self.screen, self.SUCCESS_COLOR, (px, py), 12, 2)
                pygame.draw.line(self.screen, self.SUCCESS_COLOR, (px - 8, py), (px + 8, py), 2)
                pygame.draw.line(self.screen, self.SUCCESS_COLOR, (px, py - 8), (px, py + 8), 2)

    def _draw_top_view(self):
        """绘制俯视图"""
        # 坐标变换参数
        scale = 0.8
        offset_x = self.screen_w // 2
        offset_y = self.screen_h // 2

        def to_screen(pos):
            sx = int(offset_x + (pos[0] - self.camera.target[0]) * scale)
            sy = int(offset_y - (pos[1] - self.camera.target[1]) * scale)
            return sx, sy

        # 网格
        grid_spacing = 100
        for x in range(-1000, 2000, grid_spacing):
            sx = int(offset_x + (x - self.camera.target[0]) * scale)
            pygame.draw.line(self.screen, self.GRID_COLOR, (sx, 0), (sx, self.screen_h), 1)
        for y in range(-1000, 2000, grid_spacing):
            sy = int(offset_y - (y - self.camera.target[1]) * scale)
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, sy), (self.screen_w, sy), 1)

        # 轨迹
        if len(self.interceptor_trail) > 1:
            points = [to_screen(p) for p in self.interceptor_trail]
            pygame.draw.lines(self.screen, self.INTERCEPTOR_COLOR, False, points, 2)
        if len(self.target_trail) > 1:
            points = [to_screen(p) for p in self.target_trail]
            pygame.draw.lines(self.screen, self.TARGET_COLOR, False, points, 2)

        # 无人机
        i_state, t_state = self.engine.get_current_states()
        sx, sy = to_screen(i_state.position)
        pygame.draw.circle(self.screen, self.INTERCEPTOR_COLOR, (sx, sy), 6)
        sx, sy = to_screen(t_state.position)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (sx, sy), 6)

        # 距离线
        si = to_screen(i_state.position)
        st = to_screen(t_state.position)
        pygame.draw.line(self.screen, (100, 100, 100), si, st, 1)

    def _draw_ground_grid(self):
        """绘制地面参考网格"""
        grid_y = 0  # 地面高度
        grid_range = 1000
        step = 100

        for x in range(-grid_range, grid_range * 2, step):
            p1 = self.camera.project(np.array([float(x), -float(grid_range), float(grid_y)]))
            p2 = self.camera.project(np.array([float(x), float(grid_range * 2), float(grid_y)]))
            pygame.draw.line(self.screen, self.GRID_COLOR, (p1[0], p1[1]), (p2[0], p2[1]), 1)

        for y in range(-grid_range, grid_range * 2, step):
            p1 = self.camera.project(np.array([-float(grid_range), float(y), float(grid_y)]))
            p2 = self.camera.project(np.array([float(grid_range * 2), float(y), float(grid_y)]))
            pygame.draw.line(self.screen, self.GRID_COLOR, (p1[0], p1[1]), (p2[0], p2[1]), 1)

    def _draw_trail_3d(self, trail, color):
        """绘制3D轨迹"""
        if len(trail) < 2:
            return

        # 每隔几个点绘制以提高性能
        step = max(1, len(trail) // 500)
        points = []
        for i in range(0, len(trail), step):
            sx, sy, sz = self.camera.project(trail[i])
            points.append((sx, sy))

        if len(points) > 1:
            # 渐变效果
            for i in range(1, len(points)):
                alpha = int(80 + 175 * (i / len(points)))
                c = tuple(min(255, int(ch * alpha / 255)) for ch in color[:3])
                pygame.draw.line(self.screen, c, points[i - 1], points[i], 2)

    def _draw_info_panel(self):
        """绘制信息面板"""
        panel_w = 300
        panel_h = 280
        panel_x = 10
        panel_y = 10

        # 半透明背景
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((20, 20, 45, 200))
        self.screen.blit(panel_surface, (panel_x, panel_y))

        i_state, t_state = self.engine.get_current_states()
        distance = np.linalg.norm(t_state.position - i_state.position)
        rel_vel = t_state.velocity - i_state.velocity
        if distance > 1e-6:
            los = (t_state.position - i_state.position) / distance
            v_closing = -np.dot(rel_vel, los)
        else:
            v_closing = 0

        lines = [
            (f"Algorithm: {self.guidance_name}", self.TEXT_COLOR),
            (f"Time: {i_state.time:.2f}s", self.TEXT_COLOR),
            (f"Speed: {self.sim_speed:.1f}x", self.TEXT_COLOR),
            ("", self.TEXT_COLOR),
            (f"--- Interceptor ---", self.INTERCEPTOR_COLOR),
            (f"Pos: ({i_state.position[0]:.0f}, {i_state.position[1]:.0f}, {i_state.position[2]:.0f})",
             self.INTERCEPTOR_COLOR),
            (f"Speed: {i_state.speed():.1f} m/s", self.INTERCEPTOR_COLOR),
            (f"Accel: {np.linalg.norm(i_state.acceleration):.1f} m/s²", self.INTERCEPTOR_COLOR),
            ("", self.TEXT_COLOR),
            (f"--- Target ---", self.TARGET_COLOR),
            (f"Pos: ({t_state.position[0]:.0f}, {t_state.position[1]:.0f}, {t_state.position[2]:.0f})",
             self.TARGET_COLOR),
            (f"Speed: {t_state.speed():.1f} m/s", self.TARGET_COLOR),
            ("", self.TEXT_COLOR),
            (f"Distance: {distance:.1f} m", (255, 200, 50)),
            (f"Closing: {v_closing:.1f} m/s", (255, 200, 50)),
        ]

        for i, (text, color) in enumerate(lines):
            surface = self.font_small.render(text, True, color)
            self.screen.blit(surface, (panel_x + 10, panel_y + 8 + i * 18))

        # 控制提示
        help_y = self.screen_h - 100
        help_surface = pygame.Surface((350, 90), pygame.SRCALPHA)
        help_surface.fill((20, 20, 45, 180))
        self.screen.blit(help_surface, (10, help_y))

        help_lines = [
            "SPACE: Pause | R: Reset | ESC: Quit",
            "UP/DOWN: Speed | V: View | I: Info",
            "1: PurePursuit | 2: PN | 3: APN",
            "Mouse: Rotate | Scroll: Zoom",
        ]
        for i, line in enumerate(help_lines):
            surface = self.font_small.render(line, True, (150, 150, 170))
            self.screen.blit(surface, (20, help_y + 8 + i * 20))

    def _draw_result(self):
        """绘制结果"""
        result = self.engine.result
        if result.intercepted:
            text = f"INTERCEPTED! Time: {result.total_time:.2f}s  Miss: {result.miss_distance:.2f}m"
            color = self.SUCCESS_COLOR
        else:
            text = f"MISSED! Time: {result.total_time:.2f}s  Miss: {result.miss_distance:.2f}m"
            color = self.FAIL_COLOR

        surface = self.font_large.render(text, True, color)
        rect = surface.get_rect(center=(self.screen_w // 2, 40))

        bg = pygame.Surface((rect.width + 20, rect.height + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.screen.blit(bg, (rect.x - 10, rect.y - 5))
        self.screen.blit(surface, rect)

        hint = self.font_medium.render("Press R to restart | ESC to quit", True, (180, 180, 180))
        hint_rect = hint.get_rect(center=(self.screen_w // 2, 70))
        self.screen.blit(hint, hint_rect)

    def run(self):
        """运行动画主循环"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


def run_animation(scenario_func=None, algorithm='apn', speed=1.0):
    """
    便捷函数: 启动实时动画

    参数:
        scenario_func: 场景工厂方法 (None则使用默认交叉拦截)
        algorithm: 制导算法 ('pure_pursuit', 'pn', 'apn')
        speed: 仿真速度倍率
    """
    if scenario_func is None:
        scenario_func = ScenarioFactory.crossing

    config = scenario_func()
    animation = RealtimeAnimation(config, algorithm_name=algorithm, speed_multiplier=speed)
    animation.run()


if __name__ == '__main__':
    # 默认运行交叉拦截场景
    run_animation(algorithm='apn', speed=2.0)
