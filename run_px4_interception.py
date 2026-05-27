"""
PX4 SITL 拦截仿真运行器
PX4 SITL Interception Simulation Runner

在 PX4 SITL 仿真环境中运行拦截/逃脱算法
支持仿真模式（无 MAVSDK）和真实 SITL 模式
"""

import asyncio
import numpy as np
import time
from typing import Optional

from drone_dynamics import DroneParams, DroneState
from guidance_algorithms import create_guidance, InterceptionResult
from evasion_algorithms import create_evasion
from px4_mavsdk_adapter import PX4MAVSDKAdapter, PX4DroneConfig, PX4MultiDroneManager


class PX4InterceptionSimulation:
    """
    PX4 拦截仿真
    
    将我们的算法与 PX4 SITL 连接，实现真实飞控硬件在环仿真
    """
    
    def __init__(self, 
                 interceptor_algo: str = "zem",
                 evasion_strategy: str = "high_g_jink",
                 update_rate: float = 20.0):  # Hz
        self.interceptor_algo = interceptor_algo
        self.evasion_strategy = evasion_strategy
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        # 创建无人机配置
        self.interceptor_config = PX4DroneConfig(
            name="interceptor",
            mavsdk_server_port=50051,
            mavlink_port=14540,
            initial_position=np.array([0.0, 0.0, 10.0])  # ENU: 原点，高度10m
        )
        
        self.target_config = PX4DroneConfig(
            name="target",
            mavsdk_server_port=50052,
            mavlink_port=14541,
            initial_position=np.array([100.0, 50.0, 10.0])  # ENU: 100m东，50m北，高度10m
        )
        
        # 创建适配器
        interceptor_params = DroneParams.interceptor()
        target_params = DroneParams.target()
        
        self.interceptor = PX4MAVSDKAdapter(self.interceptor_config, interceptor_params)
        self.target = PX4MAVSDKAdapter(self.target_config, target_params)
        
        # 创建算法实例
        self.guidance = create_guidance(interceptor_algo, interceptor_params)
        self.evasion = create_evasion(evasion_strategy, target_params)
        
        # 仿真状态
        self.running = False
        self.intercepted = False
        self.start_time = None
        self.max_time = 60.0  # 最大仿真时间
        self.intercept_radius = 5.0  # 拦截判定半径
        
        # 记录数据
        self.records = []
        
    async def setup(self):
        """初始化设置"""
        print("=" * 60)
        print("PX4 SITL Interception Simulation")
        print("=" * 60)
        print(f"Interceptor: {self.guidance.name}")
        print(f"Target: {self.evasion.name}")
        print("=" * 60)
        
        # 连接无人机
        print("\n[1/4] Connecting to drones...")
        await self.interceptor.connect()
        await self.target.connect()
        
        # 解锁起飞
        print("\n[2/4] Arming and taking off...")
        await self.interceptor.arm_and_takeoff(altitude=10.0)
        await self.target.arm_and_takeoff(altitude=10.0)
        await asyncio.sleep(2)  # 等待稳定
        
        # 启动 Offboard 模式
        print("\n[3/4] Starting offboard mode...")
        await self.interceptor.start_offboard()
        await self.target.start_offboard()
        await asyncio.sleep(1)
        
        print("\n[4/4] Simulation ready!")
        
    async def run_simulation(self):
        """运行仿真主循环"""
        print("\n>>> Starting interception simulation...")
        print("(Press Ctrl+C to stop)\n")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 获取当前状态
                interceptor_state = self.interceptor.get_current_state()
                target_state = self.target.get_current_state()
                
                if interceptor_state is None or target_state is None:
                    await asyncio.sleep(self.dt)
                    continue
                
                # 计算相对距离
                relative_pos = target_state.position - interceptor_state.position
                distance = np.linalg.norm(relative_pos)
                
                # 检查拦截
                if distance <= self.intercept_radius:
                    self.intercepted = True
                    elapsed = time.time() - self.start_time
                    print(f"\n[INTERCEPTED] Time: {elapsed:.2f}s, Distance: {distance:.2f}m")
                    break
                
                # 检查超时
                elapsed = time.time() - self.start_time
                if elapsed >= self.max_time:
                    print(f"\n[TIMEOUT] Max time reached. Final distance: {distance:.2f}m")
                    break
                
                # 计算制导指令
                threat = self.evasion.assess_threat(target_state, interceptor_state)
                
                # 拦截机使用制导算法
                interceptor_accel = self.guidance.compute_acceleration(
                    interceptor_state, target_state, self.dt
                )
                
                # 目标机使用逃脱策略
                target_accel = self.evasion.compute_acceleration(
                    target_state, interceptor_state, threat, self.dt
                )
                
                # 发送指令
                await self.interceptor.send_acceleration(interceptor_accel)
                await self.target.send_acceleration(target_accel)
                
                # 记录数据
                self.records.append({
                    'time': elapsed,
                    'distance': distance,
                    'interceptor_pos': interceptor_state.position.copy(),
                    'target_pos': target_state.position.copy(),
                })
                
                # 打印状态（每2秒）
                if int(elapsed * 10) % 20 == 0:
                    print(f"T={elapsed:5.1f}s | Distance: {distance:7.2f}m | "
                          f"Interceptor: ({interceptor_state.position[0]:6.1f}, "
                          f"{interceptor_state.position[1]:6.1f}) | "
                          f"Target: ({target_state.position[0]:6.1f}, "
                          f"{target_state.position[1]:6.1f})")
                
                # 维持更新率
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.dt - loop_time)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Simulation interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            
    async def cleanup(self):
        """清理和降落"""
        print("\n>>> Cleaning up...")
        
        await self.interceptor.stop_offboard()
        await self.target.stop_offboard()
        
        await self.interceptor.land()
        await self.target.land()
        
        print("Simulation complete!")
        
    def print_summary(self):
        """打印仿真总结"""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Interceptor Algorithm: {self.guidance.name}")
        print(f"Evasion Strategy: {self.evasion.name}")
        print(f"Result: {'INTERCEPTED' if self.intercepted else 'ESCAPED'}")
        
        if self.records:
            final_distance = self.records[-1]['distance']
            min_distance = min(r['distance'] for r in self.records)
            print(f"Final Distance: {final_distance:.2f}m")
            print(f"Min Distance: {min_distance:.2f}m")
            
        print("=" * 60)


async def main():
    """主函数"""
    # 创建仿真实例
    sim = PX4InterceptionSimulation(
        interceptor_algo="zem",        # ZEM 制导
        evasion_strategy="high_g_jink",  # High-G Jink 逃脱
        update_rate=20.0
    )
    
    try:
        # 设置
        await sim.setup()
        
        # 运行仿真
        await sim.run_simulation()
        
    finally:
        # 清理
        await sim.cleanup()
        sim.print_summary()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
