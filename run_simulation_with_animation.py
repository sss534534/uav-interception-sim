"""
拦截仿真 + 动画生成
Interception Simulation with Animation Generation

运行完整仿真并生成轨迹动画GIF
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

from drone_dynamics import DroneState, DroneParams, DroneDynamics
from guidance_algorithms import create_guidance
from evasion_algorithms import create_evasion


def run_interception_simulation(interceptor_algo='zem', evasion_strategy='high_g_jink',
                                  max_time=30.0, dt=0.05):
    """
    运行拦截仿真
    
    返回:
        dict: 包含轨迹数据和结果
    """
    print(f"\n{'='*60}")
    print(f"  Interception Simulation")
    print(f"  Interceptor: {interceptor_algo.upper()}")
    print(f"  Target: {evasion_strategy}")
    print(f"{'='*60}")
    
    # 创建参数
    interceptor_params = DroneParams.interceptor()
    target_params = DroneParams.target()
    
    # 创建动力学模型
    interceptor_dyn = DroneDynamics(interceptor_params)
    target_dyn = DroneDynamics(target_params)
    
    # 创建算法
    guidance = create_guidance(interceptor_algo, interceptor_params)
    evasion = create_evasion(evasion_strategy, target_params)
    
    # 初始状态
    interceptor_state = DroneState(
        position=np.array([0.0, 0.0, 50.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        acceleration=np.zeros(3),
        time=0.0
    )
    
    target_state = DroneState(
        position=np.array([200.0, 150.0, 60.0]),
        velocity=np.array([-10.0, -8.0, 0.0]),
        acceleration=np.zeros(3),
        time=0.0
    )
    
    # 记录数据
    records = {
        'times': [],
        'interceptor_pos': [],
        'interceptor_vel': [],
        'target_pos': [],
        'target_vel': [],
        'distances': [],
        'interceptor_acc': [],
        'target_acc': [],
    }
    
    # 仿真参数
    intercept_radius = 5.0
    intercepted = False
    final_distance = 0.0
    
    # 仿真循环
    t = 0.0
    step = 0
    while t < max_time:
        # 计算距离
        relative_pos = target_state.position - interceptor_state.position
        distance = np.linalg.norm(relative_pos)
        
        # 记录数据
        records['times'].append(t)
        records['interceptor_pos'].append(interceptor_state.position.copy())
        records['interceptor_vel'].append(interceptor_state.velocity.copy())
        records['target_pos'].append(target_state.position.copy())
        records['target_vel'].append(target_state.velocity.copy())
        records['distances'].append(distance)
        records['interceptor_acc'].append(interceptor_state.acceleration.copy())
        records['target_acc'].append(target_state.acceleration.copy())
        
        # 检查拦截
        if distance <= intercept_radius:
            intercepted = True
            final_distance = distance
            print(f"\n[INTERCEPTED] Time: {t:.2f}s, Distance: {distance:.2f}m")
            break
        
        # 每2秒打印状态
        if step % int(2.0/dt) == 0:
            print(f"  T={t:5.1f}s | Distance: {distance:7.2f}m | "
                  f"Interceptor: ({interceptor_state.position[0]:6.1f}, "
                  f"{interceptor_state.position[1]:6.1f}) | "
                  f"Target: ({target_state.position[0]:6.1f}, "
                  f"{target_state.position[1]:6.1f})")
        
        # 计算制导指令
        threat = evasion.assess_threat(target_state, interceptor_state)
        
        interceptor_accel = guidance.compute_acceleration(
            interceptor_state, target_state, dt
        )
        
        target_accel = evasion.compute_acceleration(
            target_state, interceptor_state, threat, dt
        )
        
        # 更新状态
        interceptor_state = interceptor_dyn.update(interceptor_state, interceptor_accel, dt)
        target_state = target_dyn.update(target_state, target_accel, dt)
        
        t += dt
        step += 1
        final_distance = distance
    
    # 结果
    result = {
        'intercepted': intercepted,
        'final_distance': final_distance,
        'final_time': t,
        'interceptor_algo': guidance.name,
        'evasion_strategy': evasion.name,
    }
    
    # 转换为numpy数组
    for key in records:
        records[key] = np.array(records[key])
    
    return records, result


def create_animation(records, result, save_path):
    """创建3D轨迹动画"""
    print(f"\n  Creating animation...")
    
    # 提取数据
    times = records['times']
    i_pos = records['interceptor_pos']
    t_pos = records['target_pos']
    distances = records['distances']
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    
    # XY平面图
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 距离-时间图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('Distance vs Time')
    ax3.grid(True, alpha=0.3)
    
    # 速度-时间图
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed vs Time')
    ax4.grid(True, alpha=0.3)
    
    # 计算速度
    i_speed = np.linalg.norm(records['interceptor_vel'], axis=1)
    t_speed = np.linalg.norm(records['target_vel'], axis=1)
    
    # 动画元素
    trail_length = 50  # 轨迹长度
    
    # 初始化
    def init():
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        return []
    
    # 更新函数
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # 当前帧索引
        idx = frame * 2  # 每2帧取一帧
        if idx >= len(times):
            idx = len(times) - 1
        
        # 轨迹起点
        start = max(0, idx - trail_length)
        
        # 3D轨迹
        ax1.plot(i_pos[start:idx+1, 0], i_pos[start:idx+1, 1], i_pos[start:idx+1, 2],
                 'b-', linewidth=1.5, alpha=0.8, label='Interceptor')
        ax1.plot(t_pos[start:idx+1, 0], t_pos[start:idx+1, 1], t_pos[start:idx+1, 2],
                 'r-', linewidth=1.5, alpha=0.8, label='Target')
        
        # 当前位置
        ax1.scatter(*i_pos[idx], color='blue', s=100, marker='o')
        ax1.scatter(*t_pos[idx], color='red', s=100, marker='o')
        
        # 起点标记
        ax1.scatter(*i_pos[0], color='blue', s=60, marker='^', alpha=0.5)
        ax1.scatter(*t_pos[0], color='red', s=60, marker='^', alpha=0.5)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory (T={times[idx]:.1f}s)')
        ax1.legend(fontsize=8)
        
        # XY平面
        ax2.plot(i_pos[start:idx+1, 0], i_pos[start:idx+1, 1], 'b-', linewidth=1.5, alpha=0.8)
        ax2.plot(t_pos[start:idx+1, 0], t_pos[start:idx+1, 1], 'r-', linewidth=1.5, alpha=0.8)
        ax2.scatter(i_pos[idx, 0], i_pos[idx, 1], color='blue', s=80, marker='o', label='Interceptor')
        ax2.scatter(t_pos[idx, 0], t_pos[idx, 1], color='red', s=80, marker='o', label='Target')
        ax2.plot([i_pos[idx, 0], t_pos[idx, 0]], [i_pos[idx, 1], t_pos[idx, 1]], 
                 'g--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'XY Plane (Distance: {distances[idx]:.1f}m)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # 距离-时间
        ax3.plot(times[:idx+1], distances[:idx+1], 'g-', linewidth=1.5)
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Intercept Radius')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title('Distance vs Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, times[-1])
        ax3.set_ylim(0, max(distances) * 1.1)
        
        # 速度-时间
        ax4.plot(times[:idx+1], i_speed[:idx+1], 'b-', linewidth=1.5, label='Interceptor')
        ax4.plot(times[:idx+1], t_speed[:idx+1], 'r-', linewidth=1.5, label='Target')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed vs Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        ax4.set_xlim(0, times[-1])
        ax4.set_ylim(0, max(max(i_speed), max(t_speed)) * 1.1)
        
        # 结果标注
        if result['intercepted'] and idx == len(times) - 1:
            fig.suptitle(f"INTERCEPTED! Time: {result['final_time']:.2f}s, Distance: {result['final_distance']:.2f}m",
                        fontsize=14, fontweight='bold', color='green')
        elif idx == len(times) - 1:
            fig.suptitle(f"ESCAPED! Final Distance: {result['final_distance']:.2f}m",
                        fontsize=14, fontweight='bold', color='red')
        else:
            fig.suptitle(f"Interceptor: {result['interceptor_algo']} | Target: {result['evasion_strategy']}",
                        fontsize=12)
        
        plt.tight_layout()
        return []
    
    # 创建动画
    n_frames = len(times) // 2
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=50, blit=False)
    
    # 保存GIF
    anim.save(save_path, writer='pillow', fps=20)
    plt.close()
    
    print(f"  Animation saved: {save_path}")
    return save_path


def main():
    """主函数"""
    output_dir = '/workspace/uav_interception_sim/simulation_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行仿真
    records, result = run_interception_simulation(
        interceptor_algo='zem',
        evasion_strategy='high_g_jink',
        max_time=30.0
    )
    
    # 创建动画
    gif_path = os.path.join(output_dir, 'interception_animation.gif')
    create_animation(records, result, gif_path)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"  SIMULATION RESULT")
    print(f"{'='*60}")
    print(f"  Interceptor: {result['interceptor_algo']}")
    print(f"  Target: {result['evasion_strategy']}")
    print(f"  Result: {'INTERCEPTED' if result['intercepted'] else 'ESCAPED'}")
    print(f"  Final Time: {result['final_time']:.2f}s")
    print(f"  Final Distance: {result['final_distance']:.2f}m")
    print(f"{'='*60}")
    
    return gif_path


if __name__ == '__main__':
    main()
