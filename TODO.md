# UAV 拦截仿真系统 - TODO

## P0 - 必须修复（影响正确性）

- [ ] **尾追场景 PN/APN 完全失败** — `guidance_algorithms.py:101-106,194-199`
  - 根因：bootstrap 条件仅检查速度大小，尾追时拦截机已有速度直接跳过 bootstrap，但 `_prev_los_unit` 为 None 导致 los_rate=0，指令为零
  - 方案：bootstrap 条件改为检查 `_prev_los_unit is None` 或视线角速度是否有效

- [ ] **报告结论与实验数据矛盾** — `main.py:231-247`
  - 硬编码结论声称 APN 最优，实际 Pure Pursuit 成功率 100%，APN 仅 67%
  - 方案：结论改为基于实际数据动态生成

- [ ] **实时动画速度倍率非整数部分失效** — `realtime_animation.py:229`
  - `int(self.sim_speed)` 丢弃小数部分，1.5x 和 1.0x 效果相同
  - 方案：使用累积时间方式实现亚整数倍速

- [ ] **仿真引擎状态更新顺序引入一拍延迟** — `simulation_engine.py:159-164`
  - 目标先更新，拦截机用旧目标状态计算指令
  - 方案：先计算双方指令，再同步更新状态

## P1 - 强烈建议（影响工程质量）

- [ ] **补充单元测试** — 新建 `tests/`
  - 动力学模型：加速度/速度/转弯率限幅、地面约束、零速边界
  - 制导算法：各算法方向正确性、bootstrap 行为、尾追回归测试
  - 仿真引擎：拦截/超时/飞出判定、能量计算
  - 目标策略：三种模式输出验证
  - **逃脱策略**：碰撞锥计算、威胁评估、各策略行为验证

- [ ] **定义制导算法抽象基类** — `guidance_algorithms.py`
  - 三种算法无公共接口，`create_guidance` 返回 `object` 类型
  - 方案：创建 `GuidanceAlgorithm` ABC，统一 `compute_acceleration`/`reset`/`name`

- [ ] **修复 Pygame 模块级导入导致 import 即退出** — `realtime_animation.py:13-18`
  - 方案：改为延迟导入或在 `run_animation` 内部处理 ImportError

- [ ] **修复可视化 save_path 目录处理** — `visualization.py:151,268,344`
  - `os.path.dirname('file.png')` 返回空字符串，`os.makedirs('')` 可能异常

- [ ] **移除硬编码输出路径** — `main.py:255`
  - 当前写死 `/workspace/uav_interception_sim/results`

- [ ] **添加 requirements.txt**

## P2 - 建议改进

- [ ] 添加 `README.md`
- [ ] 添加 `logging` 替代 `print`
- [ ] 添加 CLI 参数解析（argparse）
- [ ] 修复 Pygame 2.x 滚轮事件兼容性
- [ ] 使用 `np.trapz` 替代手动能量积分
- [ ] `max_miss_distance` 配置项未使用，实现或移除
- [ ] 热力图时间轴颜色语义修正（当前时间越长越绿，语义有误导）

## P3 - 锦上添花

- [ ] 添加 `.gitignore`
- [ ] 场景工厂参数化
- [ ] `SimulationRecord` 预分配数组
- [ ] 视线角速度改用球面几何计算

## 已完成 ✓

- [x] **逃脱策略算法框架** — `evasion_algorithms.py`
  - 抽象基类 `EvasionStrategy`
  - 碰撞锥分析器 `CollisionConeAnalyzer`
  - 四种策略：正交机动、径向机动、欺骗性、混合
  - 工厂函数 `create_evasion`

- [x] **逃脱策略对抗测试框架** — `evasion_test.py`
  - 3×4 测试矩阵（3种追击算法 × 4种逃脱策略）
  - 自动图表生成和统计分析
  - 当前结果：正交机动对抗 PN 成功率 66.7%，对抗 Pure Pursuit 成功率 66.7%

- [x] **逃脱策略优化** — `evasion_algorithms.py` 重写
  - 正交机动: 突发-滑行模式 + 随机方向切换 + 自适应间隔
  - 径向机动: 更短脉冲 + 随机正交扰动
  - 欺骗性: 更晚激活 + 二次逃脱 + 持续突发-滑行
  - **新增反预测机动**: 检测追击方加速度方向并反向操作
  - 混合策略: 追击方类型检测(PP/PN/APN) + 自适应策略切换
  - **最终结果: 反预测机动 100% 逃脱率(3/3)，正交/径向/混合 66.7%**
