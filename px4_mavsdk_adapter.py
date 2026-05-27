"""
PX4 MAVSDK 适配层
PX4 MAVSDK Adapter Layer

将我们的制导/逃脱算法与 PX4 SITL 仿真连接
提供加速度指令接口和遥测数据接收
"""

import asyncio
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

# MAVSDK 导入（可选，如果未安装则使用模拟模式）
try:
    from mavsdk import System
    from mavsdk.offboard import AccelerationNed, OffboardError
    from mavsdk.telemetry import PositionNed, VelocityNed
    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False
    print("[WARNING] MAVSDK not installed. Running in simulation mode.")

from drone_dynamics import DroneState, DroneParams


@dataclass
class PX4DroneConfig:
    """PX4 无人机配置"""
    name: str
    mavsdk_server_port: int = 50051
    mavlink_port: int = 14540
    initial_position: np.ndarray = None
    
    def __post_init__(self):
        if self.initial_position is None:
            self.initial_position = np.array([0.0, 0.0, -10.0])  # NED坐标，-10m = 地上10m


class PX4MAVSDKAdapter:
    """
    PX4 MAVSDK 适配器
    
    功能:
    1. 连接 PX4 SITL 实例
    2. 发送加速度指令 (Offboard 模式)
    3. 接收位置/速度/加速度遥测
    4. 转换为 DroneState 供算法使用
    """
    
    def __init__(self, config: PX4DroneConfig, params: DroneParams):
        self.config = config
        self.params = params
        self.name = config.name
        
        # MAVSDK 系统
        self.drone: Optional[System] = None
        self._connected = False
        
        # 当前状态
        self._current_state: Optional[DroneState] = None
        self._state_callback: Optional[Callable[[DroneState], None]] = None
        
        # 仿真模式（无 MAVSDK 时使用）
        self._simulation_mode = not MAVSDK_AVAILABLE
        self._sim_position = config.initial_position.copy()
        self._sim_velocity = np.zeros(3)
        self._sim_acceleration = np.zeros(3)
        self._sim_time = 0.0
        
    async def connect(self):
        """连接到 PX4 SITL"""
        if self._simulation_mode:
            print(f"[{self.name}] Running in simulation mode (no MAVSDK)")
            self._connected = True
            return True
        
        try:
            self.drone = System(mavsdk_server_address="localhost", 
                               port=self.config.mavsdk_server_port)
            await self.drone.connect(system_address=f"udp://:{self.config.mavlink_port}")
            
            print(f"[{self.name}] Waiting for connection...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    print(f"[{self.name}] Connected to PX4!")
                    self._connected = True
                    break
            
            # 启动遥测监听
            asyncio.create_task(self._telemetry_loop())
            return True
            
        except Exception as e:
            print(f"[{self.name}] Connection failed: {e}")
            print(f"[{self.name}] Falling back to simulation mode")
            self._simulation_mode = True
            self._connected = True
            return True
    
    async def _telemetry_loop(self):
        """遥测数据监听循环"""
        if self._simulation_mode:
            return
        
        try:
            # 合并位置、速度和姿态数据
            async for position in self.drone.telemetry.position_velocity_ned():
                # 提取 NED 坐标
                north = position.position.north_m
                east = position.position.east_m
                down = position.position.down_m
                
                # NED to ENU (我们的算法使用 ENU: East-North-Up)
                # NED: North-East-Down
                # ENU: East-North-Up
                position_enu = np.array([east, north, -down])
                
                velocity_north = position.velocity.north_m_s
                velocity_east = position.velocity.east_m_s
                velocity_down = position.velocity.down_m_s
                velocity_enu = np.array([velocity_east, velocity_north, -velocity_down])
                
                # 创建 DroneState
                self._current_state = DroneState(
                    position=position_enu,
                    velocity=velocity_enu,
                    acceleration=np.zeros(3),  # 遥测不直接提供加速度
                    time=asyncio.get_event_loop().time()
                )
                
                if self._state_callback:
                    self._state_callback(self._current_state)
                    
        except Exception as e:
            print(f"[{self.name}] Telemetry error: {e}")
    
    async def arm_and_takeoff(self, altitude: float = 10.0):
        """解锁并起飞"""
        if self._simulation_mode:
            print(f"[{self.name}] Simulated takeoff to {altitude}m")
            self._sim_position[2] = altitude
            return True
        
        try:
            print(f"[{self.name}] Arming...")
            await self.drone.action.arm()
            
            print(f"[{self.name}] Taking off...")
            await self.drone.action.takeoff()
            await asyncio.sleep(5)  # 等待起飞完成
            
            return True
        except Exception as e:
            print(f"[{self.name}] Takeoff failed: {e}")
            return False
    
    async def start_offboard(self):
        """启动 Offboard 模式"""
        if self._simulation_mode:
            print(f"[{self.name}] Simulated offboard mode started")
            return True
        
        try:
            # 设置初始 setpoint
            await self.drone.offboard.set_acceleration_ned(
                AccelerationNed(0.0, 0.0, 0.0)
            )
            
            await self.drone.offboard.start()
            print(f"[{self.name}] Offboard mode started")
            return True
            
        except OffboardError as error:
            print(f"[{self.name}] Offboard start failed: {error}")
            return False
    
    async def send_acceleration(self, acceleration_enu: np.ndarray):
        """
        发送加速度指令
        
        参数:
            acceleration_enu: [ax, ay, az] in m/s², ENU坐标系
        """
        # 限制加速度
        accel_mag = np.linalg.norm(acceleration_enu)
        if accel_mag > self.params.max_acceleration:
            acceleration_enu = acceleration_enu * (self.params.max_acceleration / accel_mag)
        
        if self._simulation_mode:
            # 仿真模式：积分更新状态
            dt = 0.05  # 假设 20Hz 更新
            self._sim_acceleration = acceleration_enu
            self._sim_velocity += acceleration_enu * dt
            
            # 限制速度
            speed = np.linalg.norm(self._sim_velocity)
            if speed > self.params.max_speed:
                self._sim_velocity = self._sim_velocity * (self.params.max_speed / speed)
            
            self._sim_position += self._sim_velocity * dt
            self._sim_time += dt
            
            # 更新状态
            self._current_state = DroneState(
                position=self._sim_position.copy(),
                velocity=self._sim_velocity.copy(),
                acceleration=self._sim_acceleration.copy(),
                time=self._sim_time
            )
            return
        
        # ENU to NED
        # ENU: East-North-Up
        # NED: North-East-Down
        ax_ned = acceleration_enu[1]  # North = East_enu
        ay_ned = acceleration_enu[0]  # East = North_enu
        az_ned = -acceleration_enu[2]  # Down = -Up_enu
        
        try:
            await self.drone.offboard.set_acceleration_ned(
                AccelerationNed(ax_ned, ay_ned, az_ned)
            )
        except Exception as e:
            print(f"[{self.name}] Failed to send acceleration: {e}")
    
    def get_current_state(self) -> Optional[DroneState]:
        """获取当前状态"""
        return self._current_state
    
    def set_state_callback(self, callback: Callable[[DroneState], None]):
        """设置状态更新回调"""
        self._state_callback = callback
    
    async def stop_offboard(self):
        """停止 Offboard 模式"""
        if self._simulation_mode:
            return
        
        try:
            await self.drone.offboard.stop()
            print(f"[{self.name}] Offboard mode stopped")
        except Exception as e:
            print(f"[{self.name}] Error stopping offboard: {e}")
    
    async def land(self):
        """降落"""
        if self._simulation_mode:
            print(f"[{self.name}] Simulated landing")
            return
        
        try:
            await self.drone.action.land()
            print(f"[{self.name}] Landing...")
        except Exception as e:
            print(f"[{self.name}] Landing failed: {e}")
    
    async def disconnect(self):
        """断开连接"""
        if self._simulation_mode:
            self._connected = False
            return
        
        try:
            await self.stop_offboard()
            await self.land()
            self._connected = False
            print(f"[{self.name}] Disconnected")
        except Exception as e:
            print(f"[{self.name}] Disconnect error: {e}")


class PX4MultiDroneManager:
    """多机 PX4 管理器"""
    
    def __init__(self):
        self.drones: dict[str, PX4MAVSDKAdapter] = {}
        
    def add_drone(self, adapter: PX4MAVSDKAdapter):
        """添加无人机"""
        self.drones[adapter.name] = adapter
        
    async def connect_all(self):
        """连接所有无人机"""
        tasks = [drone.connect() for drone in self.drones.values()]
        await asyncio.gather(*tasks)
        
    async def arm_and_takeoff_all(self, altitude: float = 10.0):
        """所有无人机解锁起飞"""
        tasks = [drone.arm_and_takeoff(altitude) for drone in self.drones.values()]
        await asyncio.gather(*tasks)
        
    async def start_offboard_all(self):
        """所有无人机启动 Offboard"""
        tasks = [drone.start_offboard() for drone in self.drones.values()]
        await asyncio.gather(*tasks)
        
    async def stop_all(self):
        """停止所有无人机"""
        tasks = [drone.stop_offboard() for drone in self.drones.values()]
        await asyncio.gather(*tasks)
        
    async def land_all(self):
        """所有无人机降落"""
        tasks = [drone.land() for drone in self.drones.values()]
        await asyncio.gather(*tasks)
        
    def get_states(self) -> dict[str, Optional[DroneState]]:
        """获取所有无人机状态"""
        return {name: drone.get_current_state() for name, drone in self.drones.items()}
