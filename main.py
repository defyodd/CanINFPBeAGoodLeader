#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多主体建模实验主程序
用于对比INFP和ESTP两种领导者类型在群体协作中的表现差异
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt

# 确保中文字体正确显示（与demo.py保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from simulation import Simulation, SimulationConfig
from analysis import ResultAnalyzer

def print_banner():
    """打印程序标题"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    多主体建模实验系统                           ║
    ║                Agent-Based Modeling for Leadership            ║
    ║                                                               ║
    ║    对比分析INFP与ESTP领导者在群体协作中的表现差异               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_experiment_config():
    """创建实验配置"""
    print("\n=== 实验配置 ===")
    
    # 默认配置
    default_config = {
        'team_size': 8,  # 增加团队规模以增加复杂性
        'max_rounds': 30,  # 增加轮次
        'task_complexity': 3.0,  # 显著提高复杂度
        'external_pressure': 0.3,
        'random_seed': 42
    }
    
    print("使用默认配置还是自定义配置？")
    print("1. 使用默认配置（推荐）")
    print("2. 自定义配置")
    
    while True:
        choice = input("请选择 (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("无效选择，请输入 1 或 2")
    
    if choice == '1':
        config = SimulationConfig(**default_config)
        print(f"使用默认配置: 团队规模={config.team_size}, 最大轮次={config.max_rounds}")
    else:
        print("\n请输入自定义配置（直接回车使用默认值）:")
        
        team_size = input(f"团队规模 (默认: {default_config['team_size']}): ").strip()
        team_size = int(team_size) if team_size else default_config['team_size']
        
        max_rounds = input(f"最大轮次 (默认: {default_config['max_rounds']}): ").strip()
        max_rounds = int(max_rounds) if max_rounds else default_config['max_rounds']
        
        task_complexity = input(f"任务复杂度 (推荐: 2.0-5.0, 默认: {default_config['task_complexity']}): ").strip()
        task_complexity = float(task_complexity) if task_complexity else default_config['task_complexity']
        
        random_seed = input(f"随机种子 (默认: {default_config['random_seed']}): ").strip()
        random_seed = int(random_seed) if random_seed else default_config['random_seed']
        
        config = SimulationConfig(
            team_size=team_size,
            max_rounds=max_rounds,
            task_complexity=task_complexity,
            external_pressure=default_config['external_pressure'],
            random_seed=random_seed
        )
        
        print(f"自定义配置: 团队规模={config.team_size}, 最大轮次={config.max_rounds}")
    
    return config

def run_simulation(config: SimulationConfig):
    """运行仿真实验"""
    print("\n=== 开始仿真实验 ===")
    
    # 创建仿真实例
    sim = Simulation(config)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行实验
        results = sim.run_experiment()
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n实验完成！用时: {duration:.2f}秒")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"simulation_results_{timestamp}.json"
        sim.save_results(results_filename)
        
        return results, results_filename
        
    except Exception as e:
        print(f"实验运行过程中出现错误: {e}")
        raise

def analyze_results(results_filename: str):
    """分析实验结果"""
    print("\n=== 开始结果分析 ===")
    
    try:
        # 加载结果
        with open(results_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 创建分析器
        analyzer = ResultAnalyzer(results)
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"analysis_output_{timestamp}"
        analyzer.create_comprehensive_report(output_dir)
        
        print(f"分析完成！结果保存在目录: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        raise

def print_quick_summary(results: dict):
    """打印快速摘要"""
    print("\n=== 实验结果快速摘要 ===")
    
    try:
        comparison = results.get('comparison_data', {})
        overall = comparison.get('overall_performance', {})
        
        print(f"🏆 总体表现获胜者: {overall.get('winner', 'N/A')} 类型领导者")
        print(f"📊 综合得分: INFP={overall.get('infp_score', 0):.4f}, ESTP={overall.get('estp_score', 0):.4f}")
        print(f"🔍 得分差异: {overall.get('score_difference', 0):.4f}")
        
        print("\n📈 关键指标对比:")
        key_metrics = ['task_efficiency', 'avg_satisfaction', 'leader_influence', 'adoption_rate']
        metric_names = ['任务效率', '团队满意度', '领导影响力', '创意采纳率']
        
        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison:
                data = comparison[metric]
                infp_val = data.get('infp', 0)
                estp_val = data.get('estp', 0)
                diff = data.get('difference', 0)
                
                winner = "INFP" if infp_val > estp_val else "ESTP"
                print(f"  {name}: {winner} 领先 (INFP={infp_val:.3f}, ESTP={estp_val:.3f}, 差异={diff:.3f})")
        
        print("\n💡 主要发现:")
        infp_metrics = results.get('infp_results', {}).get('final_metrics', {})
        estp_metrics = results.get('estp_results', {}).get('final_metrics', {})
        
        if infp_metrics.get('avg_satisfaction', 0) > estp_metrics.get('avg_satisfaction', 0):
            print("  • INFP领导者在维护团队满意度方面表现更优")
        
        if estp_metrics.get('task_efficiency', 0) > infp_metrics.get('task_efficiency', 0):
            print("  • ESTP领导者在任务执行效率方面表现更优")
        
        if infp_metrics.get('adoption_rate', 0) > estp_metrics.get('adoption_rate', 0):
            print("  • INFP领导者在促进创新方面表现更优")
        
        if estp_metrics.get('conflict_count', 0) < infp_metrics.get('conflict_count', 0):
            print("  • ESTP领导者在冲突管理方面表现更优")
        
    except Exception as e:
        print(f"生成摘要时出现错误: {e}")
        print("请查看详细的分析报告")

def main():
    """主函数"""
    print_banner()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多主体建模实验系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--analyze-only', type=str, help='仅分析指定的结果文件')
    parser.add_argument('--quick-run', action='store_true', help='快速运行（使用默认配置）')
    
    args = parser.parse_args()
    
    try:
        # 仅分析模式
        if args.analyze_only:
            print("运行模式: 仅分析已有结果")
            output_dir = analyze_results(args.analyze_only)
            print(f"\n✅ 分析完成！请查看 {output_dir} 目录中的结果")
            return
        
        # 快速运行模式
        if args.quick_run:
            print("运行模式: 快速运行（使用默认配置）")
            config = SimulationConfig(
                team_size=8,
                max_rounds=25,
                task_complexity=3.0,
                random_seed=42
            )
        else:
            # 交互式配置
            config = create_experiment_config()
        
        # 运行仿真
        results, results_filename = run_simulation(config)
        
        # 显示快速摘要
        print_quick_summary(results)
        
        # 询问是否进行详细分析
        print("\n是否进行详细分析并生成可视化报告？")
        print("1. 是（推荐，生成完整的图表和报告）")
        print("2. 否（仅保存原始数据）")
        
        while True:
            choice = input("请选择 (1/2): ").strip()
            if choice in ['1', '2']:
                break
            print("无效选择，请输入 1 或 2")
        
        if choice == '1':
            output_dir = analyze_results(results_filename)
            print(f"\n✅ 实验和分析全部完成！")
            print(f"📁 原始数据: {results_filename}")
            print(f"📊 分析报告: {output_dir}")
        else:
            print(f"\n✅ 仿真实验完成！")
            print(f"📁 结果已保存: {results_filename}")
            print("💡 如需分析，请使用: python main.py --analyze-only " + results_filename)
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查配置和输入数据")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
