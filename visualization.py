"""
无人机拦截仿真系统 - Matplotlib 静态可视化
UAV Interception Simulation - Static Visualization

生成仿真结果的分析图表:
1. 3D轨迹图
2. 距离-时间曲线
3. 接近速度-时间曲线
4. 加速度-时间曲线
5. 多算法对比图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os

# Use English labels to avoid CJK font issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_single_simulation(record, result, guidance_name, save_path=None):
    """
    绘制单次仿真的完整分析图

    参数:
        record: SimulationRecord
        result: InterceptionResult
        guidance_name: 制导算法名称
        save_path: 保存路径 (None则不保存)
    """
    data = record.to_arrays()
    times = data['times']
    i_pos = data['interceptor_pos']
    t_pos = data['target_pos']
    distances = data['distances']
    closing_speeds = data['closing_speeds']
    i_acc = data['interceptor_acc']

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'UAV Interception Simulation - {guidance_name}\n'
                 f'{"INTERCEPTED" if result.intercepted else "MISSED"} | '
                 f'Time: {result.total_time:.2f}s | '
                 f'Miss Distance: {result.miss_distance:.2f}m | '
                 f'Energy: {result.energy_consumed:.1f}',
                 fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. 3D轨迹图
    ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax1.plot(i_pos[:, 0], i_pos[:, 1], i_pos[:, 2], 'b-', linewidth=1.5,
             label='Interceptor', alpha=0.8)
    ax1.plot(t_pos[:, 0], t_pos[:, 1], t_pos[:, 2], 'r-', linewidth=1.5,
             label='Target', alpha=0.8)
    ax1.scatter(*i_pos[0], color='blue', s=100, marker='o', label='Interceptor Start')
    ax1.scatter(*t_pos[0], color='red', s=100, marker='o', label='Target Start')
    ax1.scatter(*i_pos[-1], color='blue', s=80, marker='^', label='Interceptor End')
    ax1.scatter(*t_pos[-1], color='red', s=80, marker='^', label='Target End')

    # 标记拦截点
    if result.intercepted:
        idx = np.argmin(distances)
        ax1.scatter(*i_pos[idx], color='green', s=200, marker='*', label='Intercept Point', zorder=5)

    # 每隔一段画方向箭头
    step = max(1, len(times) // 10)
    for i in range(0, len(times) - step, step):
        ax1.quiver(i_pos[i, 0], i_pos[i, 1], i_pos[i, 2],
                   i_pos[i + step, 0] - i_pos[i, 0],
                   i_pos[i + step, 1] - i_pos[i, 1],
                   i_pos[i + step, 2] - i_pos[i, 2],
                   color='blue', alpha=0.3, arrow_length_ratio=0.3)
        ax1.quiver(t_pos[i, 0], t_pos[i, 1], t_pos[i, 2],
                   t_pos[i + step, 0] - t_pos[i, 0],
                   t_pos[i + step, 1] - t_pos[i, 1],
                   t_pos[i + step, 2] - t_pos[i, 2],
                   color='red', alpha=0.3, arrow_length_ratio=0.3)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend(fontsize=7, loc='upper left')

    # 2. XY平面投影
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(i_pos[:, 0], i_pos[:, 1], 'b-', linewidth=1.2, label='Interceptor')
    ax2.plot(t_pos[:, 0], t_pos[:, 1], 'r-', linewidth=1.2, label='Target')
    ax2.scatter(i_pos[0, 0], i_pos[0, 1], color='blue', s=60, marker='o')
    ax2.scatter(t_pos[0, 0], t_pos[0, 1], color='red', s=60, marker='o')
    if result.intercepted:
        idx = np.argmin(distances)
        ax2.scatter(i_pos[idx, 0], i_pos[idx, 1], color='green', s=150, marker='*')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Projection')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 3. XZ平面投影
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(i_pos[:, 0], i_pos[:, 2], 'b-', linewidth=1.2, label='Interceptor')
    ax3.plot(t_pos[:, 0], t_pos[:, 2], 'r-', linewidth=1.2, label='Target')
    ax3.scatter(i_pos[0, 0], i_pos[0, 2], color='blue', s=60, marker='o')
    ax3.scatter(t_pos[0, 0], t_pos[0, 2], color='red', s=60, marker='o')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane Projection')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. 距离-时间曲线
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(times, distances, 'g-', linewidth=1.5)
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Intercept Radius')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Distance vs Time')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # 5. 接近速度-时间曲线
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(times, closing_speeds, 'm-', linewidth=1.5)
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Closing Speed (m/s)')
    ax5.set_title('Closing Speed vs Time')
    ax5.grid(True, alpha=0.3)

    # 6. 加速度-时间曲线
    ax6 = fig.add_subplot(gs[2, 2])
    acc_mag = np.linalg.norm(i_acc, axis=1)
    ax6.plot(times, acc_mag, 'c-', linewidth=1.0, label='|a|')
    ax6.plot(times, i_acc[:, 0], 'r-', linewidth=0.8, alpha=0.6, label='ax')
    ax6.plot(times, i_acc[:, 1], 'g-', linewidth=0.8, alpha=0.6, label='ay')
    ax6.plot(times, i_acc[:, 2], 'b-', linewidth=0.8, alpha=0.6, label='az')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.set_title('Interceptor Acceleration')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 图表已保存: {save_path}")
    else:
        plt.show()


def plot_comparison(results_dict, save_path=None):
    """
    绘制多算法对比图

    参数:
        results_dict: {
            algorithm_name: {
                'record': SimulationRecord,
                'result': InterceptionResult,
            }
        }
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('UAV Interception Algorithm Comparison', fontsize=14, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # 1. 3D轨迹对比
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    for idx, (name, data) in enumerate(results_dict.items()):
        rec = data['record'].to_arrays()
        i_pos = rec['interceptor_pos']
        t_pos = rec['target_pos']
        c = colors[idx % len(colors)]
        ax1.plot(i_pos[:, 0], i_pos[:, 1], i_pos[:, 2], '-',
                 color=c, linewidth=1.5, label=name, alpha=0.8)

    # 目标轨迹（所有场景相同）
    first_key = list(results_dict.keys())[0]
    t_pos = results_dict[first_key]['record'].to_arrays()['target_pos']
    ax1.plot(t_pos[:, 0], t_pos[:, 1], t_pos[:, 2], 'k--',
             linewidth=1.5, label='Target', alpha=0.6)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend(fontsize=7)

    # 2. 距离对比
    ax2 = fig.add_subplot(gs[0, 2])
    for idx, (name, data) in enumerate(results_dict.items()):
        rec = data['record'].to_arrays()
        c = colors[idx % len(colors)]
        ax2.plot(rec['times'], rec['distances'], '-', color=c,
                 linewidth=1.5, label=name)
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Intercept Radius')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance vs Time')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. 接近速度对比
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, (name, data) in enumerate(results_dict.items()):
        rec = data['record'].to_arrays()
        c = colors[idx % len(colors)]
        ax3.plot(rec['times'], rec['closing_speeds'], '-', color=c,
                 linewidth=1.5, label=name)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Closing Speed (m/s)')
    ax3.set_title('Closing Speed vs Time')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. 加速度对比
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, (name, data) in enumerate(results_dict.items()):
        rec = data['record'].to_arrays()
        acc_mag = np.linalg.norm(rec['interceptor_acc'], axis=1)
        c = colors[idx % len(colors)]
        ax4.plot(rec['times'], acc_mag, '-', color=c,
                 linewidth=1.5, label=name)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('Interceptor Acceleration Magnitude')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # 5. 性能指标柱状图
    ax5 = fig.add_subplot(gs[1, 2])
    names = list(results_dict.keys())
    x = np.arange(len(names))
    width = 0.25

    miss_distances = [results_dict[n]['result'].miss_distance for n in names]
    intercept_times = [results_dict[n]['result'].total_time for n in names]
    energies = [results_dict[n]['result'].energy_consumed for n in names]

    bars1 = ax5.bar(x - width, miss_distances, width, label='Miss Dist (m)',
                    color='#1f77b4', alpha=0.8)
    bars2 = ax5.bar(x, intercept_times, width, label='Time (s)',
                    color='#ff7f0e', alpha=0.8)
    bars3 = ax5.bar(x + width, [e / 100 for e in energies], width,
                    label='Energy (x100)',
                    color='#2ca02c', alpha=0.8)

    ax5.set_xticks(x)
    ax5.set_xticklabels([n.split('(')[0].strip() for n in names], fontsize=8)
    ax5.set_title('Performance Metrics')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3, axis='y')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 对比图表已保存: {save_path}")
    else:
        plt.show()


def plot_scenario_comparison(all_results, save_path=None):
    """
    绘制多场景对比图（热力图风格）

    参数:
        all_results: {
            scenario_name: {
                algorithm_name: {'record': ..., 'result': ...}
            }
        }
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Multi-Scenario Algorithm Performance', fontsize=14, fontweight='bold')

    scenarios = list(all_results.keys())
    algorithms = list(list(all_results.values())[0].keys())

    # 提取数据矩阵
    miss_matrix = np.zeros((len(scenarios), len(algorithms)))
    time_matrix = np.zeros((len(scenarios), len(algorithms)))
    energy_matrix = np.zeros((len(scenarios), len(algorithms)))

    for i, scenario in enumerate(scenarios):
        for j, algo in enumerate(algorithms):
            r = all_results[scenario][algo]['result']
            miss_matrix[i, j] = r.miss_distance
            time_matrix[i, j] = r.total_time
            energy_matrix[i, j] = r.energy_consumed

    # 1. 脱靶距离
    im1 = axes[0].imshow(miss_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(algorithms)))
    axes[0].set_xticklabels([a.split('(')[0].strip() for a in algorithms], fontsize=8, rotation=15)
    axes[0].set_yticks(range(len(scenarios)))
    axes[0].set_yticklabels(scenarios, fontsize=8)
    axes[0].set_title('Miss Distance (m)')
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            axes[0].text(j, i, f'{miss_matrix[i, j]:.1f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 2. 拦截时间
    im2 = axes[1].imshow(time_matrix, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(algorithms)))
    axes[1].set_xticklabels([a.split('(')[0].strip() for a in algorithms], fontsize=8, rotation=15)
    axes[1].set_yticks(range(len(scenarios)))
    axes[1].set_yticklabels(scenarios, fontsize=8)
    axes[1].set_title('Intercept Time (s)')
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            axes[1].text(j, i, f'{time_matrix[i, j]:.1f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 3. 能量消耗
    im3 = axes[2].imshow(energy_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_xticks(range(len(algorithms)))
    axes[2].set_xticklabels([a.split('(')[0].strip() for a in algorithms], fontsize=8, rotation=15)
    axes[2].set_yticks(range(len(scenarios)))
    axes[2].set_yticklabels(scenarios, fontsize=8)
    axes[2].set_title('Energy Consumption')
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            axes[2].text(j, i, f'{energy_matrix[i, j]:.0f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 场景对比图已保存: {save_path}")
    else:
        plt.show()
