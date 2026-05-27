"""
逃脱策略对抗测试
Evasion Strategy vs Interception Algorithms

测试各种逃脱策略对抗不同追击算法的效果
"""

import numpy as np
import os
import sys

from drone_dynamics import DroneParams
from guidance_algorithms import create_guidance
from evasion_algorithms import create_evasion
from simulation_engine import SimulationEngine, SimulationConfig
from visualization import plot_single_simulation, plot_comparison


def create_evasion_test_scenario(scenario_type: str = "crossing") -> SimulationConfig:
    """
    创建逃脱测试场景
    
    参数:
        scenario_type: "head_on", "crossing", "pursuit"
    """
    if scenario_type == "head_on":
        # 迎头场景 - 逃脱方需要快速变向
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([25.0, 0.0, 0.0]),
            target_pos=np.array([600.0, 0.0, 100.0]),
            target_vel=np.array([-20.0, 0.0, 0.0]),
            target_mode="straight",
            max_time=60.0,
        )
    elif scenario_type == "pursuit":
        # 尾追场景 - 逃脱方有速度优势但需持续机动
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([28.0, 0.0, 0.0]),
            target_pos=np.array([400.0, 0.0, 100.0]),
            target_vel=np.array([22.0, 0.0, 0.0]),
            target_mode="straight",
            max_time=90.0,
        )
    else:  # crossing
        # 交叉场景 - 给逃脱方更大的初始距离和速度优势
        return SimulationConfig(
            interceptor_pos=np.array([0.0, 0.0, 100.0]),
            interceptor_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([800.0, 600.0, 120.0]),
            target_vel=np.array([-20.0, -15.0, 0.0]),
            target_mode="straight",
            max_time=60.0,
        )


def run_evasion_test(interceptor_algo: str, evasion_strategy: str,
                     scenario_type: str = "crossing", N: float = 4.0) -> dict:
    """
    运行单次逃脱对抗测试
    
    参数:
        interceptor_algo: 追击算法 ("pure_pursuit", "pn", "apn")
        evasion_strategy: 逃脱策略 ("orthogonal", "radial", "deceptive", "hybrid")
        scenario_type: 场景类型
        N: 导航比
    
    返回:
        {"result": ..., "record": ..., "interceptor_name": ..., "evasion_name": ...}
    """
    config = create_evasion_test_scenario(scenario_type)
    engine = SimulationEngine(config)
    
    # 设置追击算法
    guidance = create_guidance(interceptor_algo, DroneParams.interceptor(), N=N)
    engine.set_guidance(guidance)
    
    # 设置逃脱策略
    evasion = create_evasion(evasion_strategy, DroneParams.target())
    engine.set_evasion(evasion)
    
    # 运行仿真
    result, record = engine.run()
    
    return {
        "result": result,
        "record": record,
        "interceptor_name": guidance.name,
        "evasion_name": evasion.name,
    }


def run_full_evasion_comparison(output_dir: str):
    """
    运行完整的逃脱策略对比实验
    
    测试矩阵:
    - 追击算法: Pure Pursuit, PN, APN
    - 逃脱策略: Orthogonal, Radial, Deceptive, Hybrid
    - 场景: Crossing
    """
    interceptor_algos = [
        ("pure_pursuit", 4.0),
        ("pn", 4.0),
        ("apn", 4.0),
        ("apn_v2", 4.0),
        ("zem", 3.0),
        ("adaptive", 4.0),
    ]
    
    evasion_strategies = ["orthogonal", "radial", "deceptive", "anti_predictive", "hybrid",
                          "high_g_jink", "close_break", "zem_counter"]
    scenario_type = "crossing"
    
    print("\n" + "=" * 70)
    print("  逃脱策略对抗测试")
    print("=" * 70)
    
    results = {}
    
    for interceptor_algo, N in interceptor_algos:
        print(f"\n{'─' * 60}")
        print(f"  追击算法: {interceptor_algo.upper()}")
        print(f"{'─' * 60}")
        
        algo_results = {}
        
        for evasion_strategy in evasion_strategies:
            data = run_evasion_test(interceptor_algo, evasion_strategy, scenario_type, N)
            result = data["result"]
            
            status = "被拦截" if result.intercepted else "逃脱成功"
            print(f"  [{status}] {data['evasion_name']:25s} | "
                  f"时间: {result.total_time:6.2f}s | "
                  f"距离: {result.miss_distance:8.2f}m")
            
            algo_results[data['evasion_name']] = data
            
            # 保存单次仿真图
            safe_name = f"{interceptor_algo}_{evasion_strategy}"
            plot_single_simulation(
                data["record"], result,
                f"{data['interceptor_name']} vs {data['evasion_name']}",
                save_path=os.path.join(output_dir, f"{safe_name}_detail.png")
            )
        
        # 保存该追击算法下的逃脱策略对比
        plot_comparison(
            {k: {"result": v["result"], "record": v["record"]} 
             for k, v in algo_results.items()},
            save_path=os.path.join(output_dir, f"{interceptor_algo}_comparison.png")
        )
        
        results[interceptor_algo] = algo_results
    
    return results


def print_summary(results: dict):
    """打印逃脱测试总结"""
    print("\n\n" + "=" * 70)
    print("  逃脱策略效果总结")
    print("=" * 70)
    
    # 统计各逃脱策略的成功率
    evasion_stats = {}
    
    for interceptor_algo, algo_results in results.items():
        for evasion_name, data in algo_results.items():
            if evasion_name not in evasion_stats:
                evasion_stats[evasion_name] = {"total": 0, "escaped": 0}
            
            evasion_stats[evasion_name]["total"] += 1
            if not data["result"].intercepted:
                evasion_stats[evasion_name]["escaped"] += 1
    
    print("\n  逃脱策略成功率:")
    print(f"  {'策略':<25s} {'逃脱次数':>10s} {'总次数':>8s} {'成功率':>10s}")
    print("  " + "-" * 60)
    
    for evasion_name, stats in sorted(evasion_stats.items()):
        rate = stats["escaped"] / stats["total"] * 100
        print(f"  {evasion_name:<25s} {stats['escaped']:>10d} {stats['total']:>8d} {rate:>9.1f}%")
    
    # 统计各追击算法下的最佳逃脱策略
    print("\n  各追击算法下的最佳逃脱策略:")
    print("  " + "-" * 60)
    
    for interceptor_algo, algo_results in results.items():
        best_evasion = None
        best_distance = 0
        
        for evasion_name, data in algo_results.items():
            if not data["result"].intercepted:
                # 逃脱成功，按最终距离排序
                if data["result"].miss_distance > best_distance:
                    best_distance = data["result"].miss_distance
                    best_evasion = evasion_name
        
        if best_evasion:
            print(f"  {interceptor_algo:<20s}: {best_evasion} (最终距离: {best_distance:.1f}m)")
        else:
            print(f"  {interceptor_algo:<20s}: 无逃脱成功案例")


def main():
    """主函数"""
    output_dir = os.path.join(os.path.dirname(__file__), "evasion_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行完整对比实验
    results = run_full_evasion_comparison(output_dir)
    
    # 打印总结
    print_summary(results)
    
    print(f"\n\n[OK] 逃脱测试结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
