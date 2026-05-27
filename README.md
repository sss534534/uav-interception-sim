# UAV Interception Simulation System

无人机拦截仿真系统 - 包含制导算法、逃脱策略、PX4 SITL 仿真适配

## 项目结构

```
uav_interception_sim/
├── drone_dynamics.py           # 无人机动力学模型
├── guidance_algorithms.py      # 拦截制导算法
├── evasion_algorithms.py       # 逃脱策略算法
├── simulation_engine.py        # 仿真引擎
├── visualization.py            # 可视化工具
├── realtime_animation.py       # Pygame 实时动画
├── px4_mavsdk_adapter.py       # PX4 MAVSDK 适配层
├── run_px4_interception.py     # PX4 拦截仿真运行器
├── run_simulation_with_animation.py  # 带动画的仿真
├── main.py                     # 主程序入口
├── evasion_test.py             # 逃脱策略测试
├── setup_px4_sitl.sh           # PX4 SITL 环境安装脚本
└── README.md                   # 本文件
```

## 制导算法

| 算法 | 描述 |
|------|------|
| Pure Pursuit | 纯追击制导 |
| PN | 比例导航制导 |
| APN | 增强型比例导航制导 |
| APN-v2 | 升级版 APN（快速估计 + 突发滑行检测）|
| ZEM | 零脱靶量制导（最优）|
| Hybrid | 混合制导（自适应切换）|

## 逃脱策略

| 策略 | 描述 |
|------|------|
| Orthogonal | 正交机动 |
| Radial | 径向机动 |
| Deceptive | 欺骗性逃脱 |
| Anti-Predictive | 反预测机动 |
| High-G Jink | 高G随机机动 |
| ZEM Counter | 针对 ZEM 的反制策略 |

## 快速开始

### 1. 基础仿真（无需 PX4）

```bash
python3 run_simulation_with_animation.py
```

### 2. 对抗测试

```bash
python3 evasion_test.py
```

### 3. PX4 SITL 仿真

```bash
# 安装环境
bash setup_px4_sitl.sh

# 启动仿真
$HOME/px4_interception_scripts/start_simulation.sh

# 运行测试
python3 run_px4_interception.py
```

## 依赖

```bash
pip install numpy matplotlib
```

可选（PX4 仿真）：
```bash
pip install mavsdk asyncio
```

## 许可证

MIT License
