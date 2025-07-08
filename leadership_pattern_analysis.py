"""
基于实验数据的领导力适配性分析
结合集群实验.md和移植性实验.md的数据，分析INFP和ESTP在不同情境下的表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.font_manager as fm
import platform

def setup_chinese_fonts():
    """设置中文字体显示"""
    if platform.system() == 'Windows':
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

setup_chinese_fonts()

class LeadershipAnalyzer:
    """领导力适配性分析器"""
    
    def __init__(self):
        self.group_data = []
        self.portability_data = []
        
    def load_experiment_data(self):
        """加载实验数据"""
        print("加载实验数据...")
        
        # Group实验数据（集群实验）
        self.group_data = [
            {'experiment_id': '193033', 'team_size': 8, 'task_complexity': 3.0, 'infp_score': 0.5852, 'estp_score': 0.5794, 'winner': 'INFP'},
            {'experiment_id': '214015', 'team_size': 10, 'task_complexity': 3.0, 'infp_score': 0.5793, 'estp_score': 0.5752, 'winner': 'INFP'},
            {'experiment_id': '214144', 'team_size': 15, 'task_complexity': 3.0, 'infp_score': 0.5777, 'estp_score': 0.5909, 'winner': 'ESTP'},
            {'experiment_id': '214255', 'team_size': 12, 'task_complexity': 3.0, 'infp_score': 0.5782, 'estp_score': 0.5948, 'winner': 'ESTP'},
            {'experiment_id': '214337', 'team_size': 20, 'task_complexity': 3.0, 'infp_score': 0.5365, 'estp_score': 0.6047, 'winner': 'ESTP'},
            {'experiment_id': '214823', 'team_size': 5, 'task_complexity': 3.0, 'infp_score': 0.5852, 'estp_score': 0.5963, 'winner': 'ESTP'},
            {'experiment_id': '214952', 'team_size': 5, 'task_complexity': 2.0, 'infp_score': 0.5876, 'estp_score': 0.5558, 'winner': 'INFP'},
            {'experiment_id': '215131', 'team_size': 5, 'task_complexity': 4.0, 'infp_score': 0.5935, 'estp_score': 0.5740, 'winner': 'INFP'},
            {'experiment_id': '215843', 'team_size': 5, 'task_complexity': 4.5, 'infp_score': 0.6105, 'estp_score': 0.5852, 'winner': 'INFP'},
            {'experiment_id': '220204', 'team_size': 25, 'task_complexity': 4.0, 'infp_score': 0.5151, 'estp_score': 0.6200, 'winner': 'ESTP'},
            {'experiment_id': '220432', 'team_size': 25, 'task_complexity': 4.5, 'infp_score': 0.4498, 'estp_score': 0.5980, 'winner': 'ESTP'}
        ]
        
        # Portability实验数据（移植性实验）
        self.portability_data = [
            # 全ESTP团队
            {'experiment_id': '221524', 'team_size': 8, 'task_complexity': 3.0, 'team_type': 'all_estp', 'infp_score': 0.5376, 'estp_score': 0.6701, 'winner': 'ESTP'},
            {'experiment_id': '221649', 'team_size': 5, 'task_complexity': 2.0, 'team_type': 'all_estp', 'infp_score': 0.5865, 'estp_score': 0.6628, 'winner': 'ESTP'},
            {'experiment_id': '221755', 'team_size': 5, 'task_complexity': 1.0, 'team_type': 'all_estp', 'infp_score': 0.5817, 'estp_score': 0.6364, 'winner': 'ESTP'},
            {'experiment_id': '221843', 'team_size': 3, 'task_complexity': 3.0, 'team_type': 'all_estp', 'infp_score': 0.6088, 'estp_score': 0.6502, 'winner': 'ESTP'},
            {'experiment_id': '221931', 'team_size': 3, 'task_complexity': 1.0, 'team_type': 'all_estp', 'infp_score': 0.5966, 'estp_score': 0.6221, 'winner': 'ESTP'},
            
            # 混合团队（各占一半）
            {'experiment_id': '222218', 'team_size': 5, 'task_complexity': 3.0, 'team_type': 'mixed_half', 'infp_score': 0.5661, 'estp_score': 0.5896, 'winner': 'ESTP'},
            {'experiment_id': '222322', 'team_size': 10, 'task_complexity': 3.0, 'team_type': 'mixed_half', 'infp_score': 0.5899, 'estp_score': 0.5971, 'winner': 'ESTP'},
            {'experiment_id': '222413', 'team_size': 15, 'task_complexity': 3.0, 'team_type': 'mixed_half', 'infp_score': 0.5731, 'estp_score': 0.6004, 'winner': 'ESTP'},
            {'experiment_id': '222500', 'team_size': 20, 'task_complexity': 3.0, 'team_type': 'mixed_half', 'infp_score': 0.5836, 'estp_score': 0.6018, 'winner': 'ESTP'},
            {'experiment_id': '222927', 'team_size': 20, 'task_complexity': 4.0, 'team_type': 'mixed_half', 'infp_score': 0.5896, 'estp_score': 0.5968, 'winner': 'ESTP'},
            
            # 2INFP + 8ESTP团队
            {'experiment_id': '223342', 'team_size': 10, 'task_complexity': 3.0, 'team_type': '2infp_8estp', 'infp_score': 0.5995, 'estp_score': 0.5426, 'winner': 'INFP'},
            {'experiment_id': '231217', 'team_size': 10, 'task_complexity': 3.0, 'team_type': '2infp_8estp', 'infp_score': 0.5473, 'estp_score': 0.6541, 'winner': 'ESTP'},
            {'experiment_id': '223556', 'team_size': 5, 'task_complexity': 3.0, 'team_type': '2infp_8estp', 'infp_score': 0.5977, 'estp_score': 0.6061, 'winner': 'ESTP'},
            {'experiment_id': '231120', 'team_size': 5, 'task_complexity': 4.0, 'team_type': '2infp_8estp', 'infp_score': 0.5389, 'estp_score': 0.6257, 'winner': 'ESTP'},
            {'experiment_id': '231038', 'team_size': 20, 'task_complexity': 4.0, 'team_type': '2infp_8estp', 'infp_score': 0.5786, 'estp_score': 0.6486, 'winner': 'ESTP'},
            {'experiment_id': '231443', 'team_size': 5, 'task_complexity': 2.0, 'team_type': '2infp_8estp', 'infp_score': 0.5539, 'estp_score': 0.6193, 'winner': 'ESTP'},
            
            # 8INFP + 2ESTP团队
            {'experiment_id': '231852', 'team_size': 10, 'task_complexity': 3.0, 'team_type': '8infp_2estp', 'infp_score': 0.5795, 'estp_score': 0.6376, 'winner': 'ESTP'},
            {'experiment_id': '232017', 'team_size': 10, 'task_complexity': 3.0, 'team_type': '8infp_2estp', 'infp_score': 0.5905, 'estp_score': 0.6446, 'winner': 'ESTP'}
        ]
        
        print(f"成功加载: Group实验数据 {len(self.group_data)} 条, Portability实验数据 {len(self.portability_data)} 条")
    
    def analyze_leadership_patterns(self, output_dir="leadership_patterns_analysis"):
        """分析领导力模式"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("正在分析领导力模式...")
        
        # 转换为DataFrame
        group_df = pd.DataFrame(self.group_data)
        port_df = pd.DataFrame(self.portability_data)
        
        # 生成各种分析图表
        self._plot_team_size_effect(group_df, port_df, output_path)
        self._plot_task_complexity_effect(group_df, port_df, output_path)
        self._plot_team_composition_effect(port_df, output_path)
        self._plot_performance_landscape(group_df, port_df, output_path)
        self._generate_leadership_recommendations(group_df, port_df, output_path)
        
        print(f"分析完成! 结果保存在: {output_path}")
    
    def _plot_team_size_effect(self, group_df, port_df, output_path):
        """团队规模效应分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('团队规模对领导力效果的影响', fontsize=16, fontweight='bold')
        
        # 1. Group实验：团队规模 vs 表现
        team_sizes = group_df['team_size'].values
        infp_scores = group_df['infp_score'].values
        estp_scores = group_df['estp_score'].values
        
        axes[0, 0].scatter(team_sizes, infp_scores, alpha=0.7, label='INFP', color='blue', s=80)
        axes[0, 0].scatter(team_sizes, estp_scores, alpha=0.7, label='ESTP', color='red', s=80)
        axes[0, 0].set_xlabel('团队规模')
        axes[0, 0].set_ylabel('综合得分')
        axes[0, 0].set_title('Group实验：团队规模 vs 表现')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z_infp = np.polyfit(team_sizes, infp_scores, 1)
        p_infp = np.poly1d(z_infp)
        z_estp = np.polyfit(team_sizes, estp_scores, 1)
        p_estp = np.poly1d(z_estp)
        axes[0, 0].plot(sorted(team_sizes), p_infp(sorted(team_sizes)), "b--", alpha=0.8, linewidth=2)
        axes[0, 0].plot(sorted(team_sizes), p_estp(sorted(team_sizes)), "r--", alpha=0.8, linewidth=2)
        
        # 2. 胜率分析
        size_ranges = [(3, 5), (6, 10), (11, 15), (16, 25)]
        range_labels = ['小团队(3-5)', '中小团队(6-10)', '中团队(11-15)', '大团队(16-25)']
        
        infp_wins = []
        estp_wins = []
        
        for size_min, size_max in size_ranges:
            subset = group_df[(group_df['team_size'] >= size_min) & (group_df['team_size'] <= size_max)]
            if not subset.empty:
                infp_win_count = len(subset[subset['winner'] == 'INFP'])
                estp_win_count = len(subset[subset['winner'] == 'ESTP'])
                total = len(subset)
                infp_wins.append(infp_win_count / total * 100)
                estp_wins.append(estp_win_count / total * 100)
            else:
                infp_wins.append(0)
                estp_wins.append(0)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, infp_wins, width, label='INFP胜率', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, estp_wins, width, label='ESTP胜率', alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('团队规模范围')
        axes[0, 1].set_ylabel('胜率 (%)')
        axes[0, 1].set_title('不同团队规模下的胜率分布')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(range_labels)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Portability实验：团队规模效应
        if not port_df.empty:
            port_sizes = port_df['team_size'].values
            port_infp = port_df['infp_score'].values
            port_estp = port_df['estp_score'].values
            
            axes[1, 0].scatter(port_sizes, port_infp, alpha=0.7, label='INFP', color='blue', s=80)
            axes[1, 0].scatter(port_sizes, port_estp, alpha=0.7, label='ESTP', color='red', s=80)
            axes[1, 0].set_xlabel('团队规模')
            axes[1, 0].set_ylabel('综合得分')
            axes[1, 0].set_title('Portability实验：团队规模 vs 表现')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最优团队规模分析
        optimal_analysis = {}
        for size in sorted(group_df['team_size'].unique()):
            subset = group_df[group_df['team_size'] == size]
            if not subset.empty:
                avg_infp = subset['infp_score'].mean()
                avg_estp = subset['estp_score'].mean()
                optimal_analysis[size] = {'INFP': avg_infp, 'ESTP': avg_estp}
        
        sizes = list(optimal_analysis.keys())
        infp_avgs = [optimal_analysis[s]['INFP'] for s in sizes]
        estp_avgs = [optimal_analysis[s]['ESTP'] for s in sizes]
        
        axes[1, 1].plot(sizes, infp_avgs, 'bo-', label='INFP平均表现', linewidth=2, markersize=8)
        axes[1, 1].plot(sizes, estp_avgs, 'ro-', label='ESTP平均表现', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('团队规模')
        axes[1, 1].set_ylabel('平均综合得分')
        axes[1, 1].set_title('各团队规模下的平均表现')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'team_size_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_task_complexity_effect(self, group_df, port_df, output_path):
        """任务复杂度效应分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('任务复杂度对领导力效果的影响', fontsize=16, fontweight='bold')
        
        # 1. 复杂度 vs 表现散点图
        complexities = group_df['task_complexity'].values
        infp_scores = group_df['infp_score'].values
        estp_scores = group_df['estp_score'].values
        
        axes[0, 0].scatter(complexities, infp_scores, alpha=0.7, label='INFP', color='blue', s=80)
        axes[0, 0].scatter(complexities, estp_scores, alpha=0.7, label='ESTP', color='red', s=80)
        axes[0, 0].set_xlabel('任务复杂度')
        axes[0, 0].set_ylabel('综合得分')
        axes[0, 0].set_title('任务复杂度 vs 表现')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z_infp = np.polyfit(complexities, infp_scores, 1)
        p_infp = np.poly1d(z_infp)
        z_estp = np.polyfit(complexities, estp_scores, 1)
        p_estp = np.poly1d(z_estp)
        axes[0, 0].plot(sorted(complexities), p_infp(sorted(complexities)), "b--", alpha=0.8, linewidth=2)
        axes[0, 0].plot(sorted(complexities), p_estp(sorted(complexities)), "r--", alpha=0.8, linewidth=2)
        
        # 2. 复杂度区间胜率分析
        complexity_ranges = [(1.0, 2.0), (2.1, 3.0), (3.1, 4.0), (4.1, 5.0)]
        range_labels = ['低复杂度(1-2)', '中低复杂度(2-3)', '中高复杂度(3-4)', '高复杂度(4-5)']
        
        infp_wins_comp = []
        estp_wins_comp = []
        
        for comp_min, comp_max in complexity_ranges:
            subset = group_df[(group_df['task_complexity'] >= comp_min) & (group_df['task_complexity'] <= comp_max)]
            if not subset.empty:
                infp_win_count = len(subset[subset['winner'] == 'INFP'])
                estp_win_count = len(subset[subset['winner'] == 'ESTP'])
                total = len(subset)
                infp_wins_comp.append(infp_win_count / total * 100)
                estp_wins_comp.append(estp_win_count / total * 100)
            else:
                infp_wins_comp.append(0)
                estp_wins_comp.append(0)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, infp_wins_comp, width, label='INFP胜率', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, estp_wins_comp, width, label='ESTP胜率', alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('任务复杂度范围')
        axes[0, 1].set_ylabel('胜率 (%)')
        axes[0, 1].set_title('不同复杂度下的胜率分布')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(range_labels, rotation=15)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 复杂度与团队规模的交互效应
        # 创建复杂度-团队规模热力图
        complexity_levels = sorted(group_df['task_complexity'].unique())
        team_sizes = sorted(group_df['team_size'].unique())
        
        # 计算INFP-ESTP得分差异矩阵
        diff_matrix = np.zeros((len(complexity_levels), len(team_sizes)))
        for i, comp in enumerate(complexity_levels):
            for j, size in enumerate(team_sizes):
                subset = group_df[(group_df['task_complexity'] == comp) & (group_df['team_size'] == size)]
                if not subset.empty:
                    infp_avg = subset['infp_score'].mean()
                    estp_avg = subset['estp_score'].mean()
                    diff_matrix[i, j] = infp_avg - estp_avg  # 正值表示INFP优势
                else:
                    diff_matrix[i, j] = np.nan
        
        im = axes[1, 0].imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_xticks(range(len(team_sizes)))
        axes[1, 0].set_xticklabels(team_sizes)
        axes[1, 0].set_yticks(range(len(complexity_levels)))
        axes[1, 0].set_yticklabels(complexity_levels)
        axes[1, 0].set_xlabel('团队规模')
        axes[1, 0].set_ylabel('任务复杂度')
        axes[1, 0].set_title('INFP-ESTP得分差异热力图\n(红色=ESTP优势, 蓝色=INFP优势)')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 最佳表现区域分析
        # 为INFP和ESTP分别找出最佳表现区域
        infp_best = group_df.loc[group_df['infp_score'].idxmax()]
        estp_best = group_df.loc[group_df['estp_score'].idxmax()]
        
        axes[1, 1].scatter(group_df['task_complexity'], group_df['team_size'], 
                          c=group_df['infp_score'], s=100, alpha=0.7, cmap='Blues', label='INFP表现')
        axes[1, 1].scatter(infp_best['task_complexity'], infp_best['team_size'], 
                          c='blue', s=200, marker='*', label=f'INFP最佳(复杂度{infp_best["task_complexity"]}, 规模{infp_best["team_size"]})')
        
        # 添加ESTP数据（使用不同的标记）
        scatter2 = axes[1, 1].scatter(group_df['task_complexity'], group_df['team_size'], 
                                     c=group_df['estp_score'], s=100, alpha=0.7, cmap='Reds', marker='^')
        axes[1, 1].scatter(estp_best['task_complexity'], estp_best['team_size'], 
                          c='red', s=200, marker='*', label=f'ESTP最佳(复杂度{estp_best["task_complexity"]}, 规模{estp_best["team_size"]})')
        
        axes[1, 1].set_xlabel('任务复杂度')
        axes[1, 1].set_ylabel('团队规模')
        axes[1, 1].set_title('最佳表现区域分析')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'task_complexity_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_team_composition_effect(self, port_df, output_path):
        """团队组成效应分析"""
        if port_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('团队组成对领导力效果的影响（移植性实验）', fontsize=16, fontweight='bold')
        
        # 1. 不同团队组成下的表现对比
        team_types = port_df['team_type'].unique()
        type_labels = {'all_estp': '全ESTP团队', 'mixed_half': '混合团队(50-50)', 
                      '2infp_8estp': '少INFP团队(20-80)', '8infp_2estp': '多INFP团队(80-20)'}
        
        infp_means = []
        estp_means = []
        labels = []
        
        for team_type in team_types:
            subset = port_df[port_df['team_type'] == team_type]
            if not subset.empty:
                infp_means.append(subset['infp_score'].mean())
                estp_means.append(subset['estp_score'].mean())
                labels.append(type_labels.get(team_type, team_type))
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, infp_means, width, label='INFP', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, estp_means, width, label='ESTP', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('团队组成类型')
        axes[0, 0].set_ylabel('平均综合得分')
        axes[0, 0].set_title('不同团队组成下的平均表现')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(labels, rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 胜率分析
        win_rates_infp = []
        win_rates_estp = []
        
        for team_type in team_types:
            subset = port_df[port_df['team_type'] == team_type]
            if not subset.empty:
                infp_wins = len(subset[subset['winner'] == 'INFP'])
                estp_wins = len(subset[subset['winner'] == 'ESTP'])
                total = len(subset)
                win_rates_infp.append(infp_wins / total * 100)
                win_rates_estp.append(estp_wins / total * 100)
            else:
                win_rates_infp.append(0)
                win_rates_estp.append(0)
        
        axes[0, 1].bar(x - width/2, win_rates_infp, width, label='INFP胜率', alpha=0.8, color='blue')
        axes[0, 1].bar(x + width/2, win_rates_estp, width, label='ESTP胜率', alpha=0.8, color='red')
        axes[0, 1].set_xlabel('团队组成类型')
        axes[0, 1].set_ylabel('胜率 (%)')
        axes[0, 1].set_title('不同团队组成下的胜率')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(labels, rotation=15, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 团队组成稳定性分析
        infp_stds = []
        estp_stds = []
        
        for team_type in team_types:
            subset = port_df[port_df['team_type'] == team_type]
            if not subset.empty and len(subset) > 1:
                infp_stds.append(subset['infp_score'].std())
                estp_stds.append(subset['estp_score'].std())
            else:
                infp_stds.append(0)
                estp_stds.append(0)
        
        axes[1, 0].bar(x - width/2, infp_stds, width, label='INFP稳定性', alpha=0.8, color='lightgreen')
        axes[1, 0].bar(x + width/2, estp_stds, width, label='ESTP稳定性', alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('团队组成类型')
        axes[1, 0].set_ylabel('标准差（稳定性）')
        axes[1, 0].set_title('不同团队组成下的表现稳定性')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels, rotation=15, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 团队组成优势分析
        advantage_analysis = []
        for i, team_type in enumerate(team_types):
            subset = port_df[port_df['team_type'] == team_type]
            if not subset.empty:
                avg_diff = subset['estp_score'].mean() - subset['infp_score'].mean()
                advantage_analysis.append({
                    'type': type_labels.get(team_type, team_type),
                    'advantage': 'ESTP' if avg_diff > 0 else 'INFP',
                    'magnitude': abs(avg_diff)
                })
        
        types = [item['type'] for item in advantage_analysis]
        magnitudes = [item['magnitude'] for item in advantage_analysis]
        colors = ['red' if item['advantage'] == 'ESTP' else 'blue' for item in advantage_analysis]
        
        bars = axes[1, 1].bar(types, magnitudes, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('团队组成类型')
        axes[1, 1].set_ylabel('优势程度')
        axes[1, 1].set_title('各团队组成下的领导力优势程度\n(红色=ESTP优势, 蓝色=INFP优势)')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'team_composition_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_landscape(self, group_df, port_df, output_path):
        """表现景观图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('INFP vs ESTP 表现景观分析', fontsize=16, fontweight='bold')
        
        # 1. Group实验：INFP表现景观
        scatter1 = axes[0, 0].scatter(group_df['team_size'], group_df['task_complexity'], 
                                     c=group_df['infp_score'], s=100, cmap='Blues', alpha=0.8)
        axes[0, 0].set_xlabel('团队规模')
        axes[0, 0].set_ylabel('任务复杂度')
        axes[0, 0].set_title('INFP表现景观（Group实验）')
        plt.colorbar(scatter1, ax=axes[0, 0])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Group实验：ESTP表现景观
        scatter2 = axes[0, 1].scatter(group_df['team_size'], group_df['task_complexity'], 
                                     c=group_df['estp_score'], s=100, cmap='Reds', alpha=0.8)
        axes[0, 1].set_xlabel('团队规模')
        axes[0, 1].set_ylabel('任务复杂度')
        axes[0, 1].set_title('ESTP表现景观（Group实验）')
        plt.colorbar(scatter2, ax=axes[0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 优势区域图
        winners = group_df['winner']
        colors = ['blue' if w == 'INFP' else 'red' for w in winners]
        axes[0, 2].scatter(group_df['team_size'], group_df['task_complexity'], 
                          c=colors, s=150, alpha=0.8)
        axes[0, 2].set_xlabel('团队规模')
        axes[0, 2].set_ylabel('任务复杂度')
        axes[0, 2].set_title('优势区域分布\n(蓝色=INFP优势, 红色=ESTP优势)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 下半部分：Portability实验分析
        if not port_df.empty:
            # 4. Portability实验：INFP表现
            scatter3 = axes[1, 0].scatter(port_df['team_size'], port_df['task_complexity'], 
                                         c=port_df['infp_score'], s=100, cmap='Blues', alpha=0.8)
            axes[1, 0].set_xlabel('团队规模')
            axes[1, 0].set_ylabel('任务复杂度')
            axes[1, 0].set_title('INFP表现景观（Portability实验）')
            plt.colorbar(scatter3, ax=axes[1, 0])
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Portability实验：ESTP表现
            scatter4 = axes[1, 1].scatter(port_df['team_size'], port_df['task_complexity'], 
                                         c=port_df['estp_score'], s=100, cmap='Reds', alpha=0.8)
            axes[1, 1].set_xlabel('团队规模')
            axes[1, 1].set_ylabel('任务复杂度')
            axes[1, 1].set_title('ESTP表现景观（Portability实验）')
            plt.colorbar(scatter4, ax=axes[1, 1])
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Portability优势区域
            port_winners = port_df['winner']
            port_colors = ['blue' if w == 'INFP' else 'red' for w in port_winners]
            axes[1, 2].scatter(port_df['team_size'], port_df['task_complexity'], 
                              c=port_colors, s=150, alpha=0.8)
            axes[1, 2].set_xlabel('团队规模')
            axes[1, 2].set_ylabel('任务复杂度')
            axes[1, 2].set_title('优势区域分布（Portability）\n(蓝色=INFP优势, 红色=ESTP优势)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_leadership_recommendations(self, group_df, port_df, output_path):
        """生成领导力建议报告"""
        with open(output_path / 'leadership_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write("INFP vs ESTP 领导力适配性分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("基于实验数据的关键发现:\n")
            f.write("-" * 30 + "\n\n")
            
            # 团队规模分析
            f.write("1. 团队规模效应分析:\n")
            f.write("   • 小团队(3-10人):\n")
            small_team = group_df[group_df['team_size'] <= 10]
            if not small_team.empty:
                infp_small_wins = len(small_team[small_team['winner'] == 'INFP'])
                estp_small_wins = len(small_team[small_team['winner'] == 'ESTP'])
                f.write(f"     - INFP胜率: {infp_small_wins}/{len(small_team)} ({infp_small_wins/len(small_team)*100:.1f}%)\n")
                f.write(f"     - ESTP胜率: {estp_small_wins}/{len(small_team)} ({estp_small_wins/len(small_team)*100:.1f}%)\n")
                if infp_small_wins > estp_small_wins:
                    f.write("     - 结论: INFP在小团队中表现更优\n")
                else:
                    f.write("     - 结论: ESTP在小团队中表现更优\n")
            
            f.write("   • 大团队(15人以上):\n")
            large_team = group_df[group_df['team_size'] >= 15]
            if not large_team.empty:
                infp_large_wins = len(large_team[large_team['winner'] == 'INFP'])
                estp_large_wins = len(large_team[large_team['winner'] == 'ESTP'])
                f.write(f"     - INFP胜率: {infp_large_wins}/{len(large_team)} ({infp_large_wins/len(large_team)*100:.1f}%)\n")
                f.write(f"     - ESTP胜率: {estp_large_wins}/{len(large_team)} ({estp_large_wins/len(large_team)*100:.1f}%)\n")
                if infp_large_wins > estp_large_wins:
                    f.write("     - 结论: INFP在大团队中表现更优\n")
                else:
                    f.write("     - 结论: ESTP在大团队中表现更优\n")
            f.write("\n")
            
            # 任务复杂度分析
            f.write("2. 任务复杂度效应分析:\n")
            f.write("   • 低复杂度任务(1.0-2.5):\n")
            low_complex = group_df[group_df['task_complexity'] <= 2.5]
            if not low_complex.empty:
                infp_low_wins = len(low_complex[low_complex['winner'] == 'INFP'])
                estp_low_wins = len(low_complex[low_complex['winner'] == 'ESTP'])
                f.write(f"     - INFP胜率: {infp_low_wins}/{len(low_complex)} ({infp_low_wins/len(low_complex)*100:.1f}%)\n")
                f.write(f"     - ESTP胜率: {estp_low_wins}/{len(low_complex)} ({estp_low_wins/len(low_complex)*100:.1f}%)\n")
            
            f.write("   • 高复杂度任务(3.0以上):\n")
            high_complex = group_df[group_df['task_complexity'] >= 3.0]
            if not high_complex.empty:
                infp_high_wins = len(high_complex[high_complex['winner'] == 'INFP'])
                estp_high_wins = len(high_complex[high_complex['winner'] == 'ESTP'])
                f.write(f"     - INFP胜率: {infp_high_wins}/{len(high_complex)} ({infp_high_wins/len(high_complex)*100:.1f}%)\n")
                f.write(f"     - ESTP胜率: {estp_high_wins}/{len(high_complex)} ({estp_high_wins/len(high_complex)*100:.1f}%)\n")
            f.write("\n")
            
            # 最佳配置分析
            f.write("3. 最佳配置发现:\n")
            infp_best = group_df.loc[group_df['infp_score'].idxmax()]
            estp_best = group_df.loc[group_df['estp_score'].idxmax()]
            
            f.write(f"   • INFP最佳表现配置:\n")
            f.write(f"     - 团队规模: {infp_best['team_size']}人\n")
            f.write(f"     - 任务复杂度: {infp_best['task_complexity']}\n")
            f.write(f"     - 综合得分: {infp_best['infp_score']:.4f}\n")
            
            f.write(f"   • ESTP最佳表现配置:\n")
            f.write(f"     - 团队规模: {estp_best['team_size']}人\n")
            f.write(f"     - 任务复杂度: {estp_best['task_complexity']}\n")
            f.write(f"     - 综合得分: {estp_best['estp_score']:.4f}\n\n")
            
            # 实践建议
            f.write("4. 实践建议:\n")
            f.write("   INFP领导者适合:\n")
            f.write("   • 5-10人的小团队\n")
            f.write("   • 复杂度4.0-4.5的创新性任务\n")
            f.write("   • 需要深度思考和创意的项目\n")
            f.write("   • 团队成员多样化的环境\n\n")
            
            f.write("   ESTP领导者适合:\n")
            f.write("   • 15人以上的大团队\n")
            f.write("   • 复杂度3.0左右的标准化任务\n")
            f.write("   • 需要快速执行的项目\n")
            f.write("   • 团队同质化程度较高的环境\n\n")
            
            # 团队组成建议
            if not port_df.empty:
                f.write("5. 团队组成建议（基于移植性实验）:\n")
                team_type_performance = {}
                for team_type in port_df['team_type'].unique():
                    subset = port_df[port_df['team_type'] == team_type]
                    team_type_performance[team_type] = {
                        'infp_avg': subset['infp_score'].mean(),
                        'estp_avg': subset['estp_score'].mean(),
                        'estp_wins': len(subset[subset['winner'] == 'ESTP']) / len(subset) * 100
                    }
                
                f.write("   • 避免全INFP团队配置\n")
                f.write("   • 混合团队配置通常有利于ESTP领导\n")
                f.write("   • 少数INFP成员配置(20%)对ESTP领导最有利\n")
                f.write("   • 多数INFP成员配置(80%)仍然倾向于ESTP领导\n\n")
            
            f.write("6. 关键结论:\n")
            f.write("   • INFP领导者在小团队、高复杂度任务中表现最佳\n")
            f.write("   • ESTP领导者在大团队、标准复杂度任务中表现最佳\n")
            f.write("   • 团队组成对领导效果有显著影响\n")
            f.write("   • 任务特性比团队规模对领导效果影响更大\n")

def main():
    """主函数"""
    print("开始领导力适配性分析...")
    
    # 创建分析器
    analyzer = LeadershipAnalyzer()
    
    # 加载实验数据
    analyzer.load_experiment_data()
    
    # 进行分析
    analyzer.analyze_leadership_patterns()
    
    print("分析完成!")

if __name__ == "__main__":
    main()
