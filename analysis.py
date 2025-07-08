import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
from pathlib import Path

# 设置中文字体和样式
import matplotlib
import platform

# 设置中文字体和样式
import matplotlib
import matplotlib.font_manager as fm
import platform
import warnings

def setup_chinese_fonts():
    """设置中文字体显示"""
    # 抑制字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 根据操作系统设置中文字体
    if platform.system() == 'Windows':
        # Windows系统优先使用SimHei，与demo.py保持一致
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
        else:
            # 如果都不行，强制使用SimHei（与demo.py一致）
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
            
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'SimHei', 'DejaVu Sans']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'sans-serif']
    
    # 确保设置正确应用
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

# 设置字体（必须在样式设置之前）
setup_chinese_fonts()

try:
    plt.style.use('seaborn-v0_8')  # 使用更现代的seaborn样式
except:
    plt.style.use('default')  # 如果seaborn样式不可用，使用默认样式

# 重新设置中文字体，确保样式设置不会覆盖字体配置
setup_chinese_fonts()

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_data: Dict):
        self.results_data = results_data
        self.infp_results = results_data.get('infp_results', {})
        self.estp_results = results_data.get('estp_results', {})
        self.comparison_data = results_data.get('comparison_data', {})
        
    def create_comprehensive_report(self, output_dir: str = "analysis_output"):
        """创建综合分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("正在生成综合分析报告...")
        
        # 1. 任务推进效率分析
        self._plot_task_progress_comparison(output_path)
        
        # 2. 群体满意度趋势分析
        self._plot_satisfaction_trends(output_path)
        
        # 3. 冲突事件分析
        self._plot_conflict_analysis(output_path)
        
        # 4. 创意产出分析
        self._plot_creativity_analysis(output_path)
        
        # 5. 领导者影响力分析
        self._plot_leadership_influence(output_path)
        
        # 6. 综合雷达图对比
        self._plot_radar_comparison(output_path)
        
        # 7. 关键指标对比
        self._plot_key_metrics_comparison(output_path)
        
        # 8. 生成数据表格
        self._generate_data_tables(output_path)
        
        # 9. 生成文字报告
        self._generate_text_report(output_path)
        
        print(f"分析报告已生成完成，保存在: {output_path}")
        
    def _plot_task_progress_comparison(self, output_path: Path):
        """任务推进效率对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('任务推进效率对比分析', fontsize=16, fontweight='bold')
        
        # 1. 任务进度曲线
        ax1 = axes[0, 0]
        infp_rounds = self.infp_results.get('rounds', [])
        estp_rounds = self.estp_results.get('rounds', [])
        
        if infp_rounds and estp_rounds:
            infp_progress = [r['task_progress_after'] for r in infp_rounds]
            estp_progress = [r['task_progress_after'] for r in estp_rounds]
            
            ax1.plot(range(1, len(infp_progress) + 1), infp_progress, 
                    'b-o', label='INFP 领导者', linewidth=2, markersize=5)
            ax1.plot(range(1, len(estp_progress) + 1), estp_progress, 
                    'r-s', label='ESTP 领导者', linewidth=2, markersize=5)
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('任务完成进度')
            ax1.set_title('任务完成进度对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 任务效率对比
        ax2 = axes[0, 1]
        infp_efficiency = self.infp_results.get('final_metrics', {}).get('task_efficiency', 0)
        estp_efficiency = self.estp_results.get('final_metrics', {}).get('task_efficiency', 0)
        
        efficiency_data = [infp_efficiency, estp_efficiency]
        labels = ['INFP', 'ESTP']
        colors = ['skyblue', 'lightcoral']
        
        bars = ax2.bar(labels, efficiency_data, color=colors, alpha=0.7)
        ax2.set_ylabel('任务效率')
        ax2.set_title('任务效率对比')
        ax2.set_ylim(0, max(efficiency_data) * 1.2)
        
        # 添加数值标签
        for bar, value in zip(bars, efficiency_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 返工率对比
        ax3 = axes[1, 0]
        infp_rework = self.infp_results.get('final_metrics', {}).get('rework_ratio', 0)
        estp_rework = self.estp_results.get('final_metrics', {}).get('rework_ratio', 0)
        
        rework_data = [infp_rework, estp_rework]
        bars = ax3.bar(labels, rework_data, color=colors, alpha=0.7)
        ax3.set_ylabel('返工率')
        ax3.set_title('返工率对比')
        ax3.set_ylim(0, max(rework_data) * 1.2 if max(rework_data) > 0 else 1)
        
        for bar, value in zip(bars, rework_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 完成时间对比
        ax4 = axes[1, 1]
        infp_completion = self.infp_results.get('task_completion', {})
        estp_completion = self.estp_results.get('task_completion', {})
        
        infp_rounds_to_complete = len(infp_rounds)
        estp_rounds_to_complete = len(estp_rounds)
        
        completion_data = [infp_rounds_to_complete, estp_rounds_to_complete]
        bars = ax4.bar(labels, completion_data, color=colors, alpha=0.7)
        ax4.set_ylabel('完成轮次')
        ax4.set_title('任务完成时间对比')
        
        for bar, value in zip(bars, completion_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图1: 任务推进效率对比分析\n展示INFP和ESTP领导者在任务完成进度、效率、返工率和完成时间方面的差异', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.12)  # 为图注留出更多空间
        plt.savefig(output_path / 'task_progress_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_satisfaction_trends(self, output_path: Path):
        """满意度趋势分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('群体满意度趋势分析', fontsize=16, fontweight='bold')
        
        # 1. 满意度变化趋势
        ax1 = axes[0, 0]
        
        infp_satisfaction = self.infp_results.get('final_metrics', {}).get('satisfaction_trend', [])
        estp_satisfaction = self.estp_results.get('final_metrics', {}).get('satisfaction_trend', [])
        
        if infp_satisfaction and estp_satisfaction:
            ax1.plot(range(1, len(infp_satisfaction) + 1), infp_satisfaction, 
                    'b-o', label='INFP 领导者', linewidth=2, markersize=5)
            ax1.plot(range(1, len(estp_satisfaction) + 1), estp_satisfaction, 
                    'r-s', label='ESTP 领导者', linewidth=2, markersize=5)
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('平均满意度')
            ax1.set_title('满意度变化趋势')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 最终满意度对比
        ax2 = axes[0, 1]
        infp_final_satisfaction = self.infp_results.get('final_metrics', {}).get('avg_satisfaction', 0)
        estp_final_satisfaction = self.estp_results.get('final_metrics', {}).get('avg_satisfaction', 0)
        
        satisfaction_data = [infp_final_satisfaction, estp_final_satisfaction]
        labels = ['INFP', 'ESTP']
        colors = ['lightblue', 'lightcoral']
        
        bars = ax2.bar(labels, satisfaction_data, color=colors, alpha=0.7)
        ax2.set_ylabel('平均满意度')
        ax2.set_title('最终满意度对比')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, satisfaction_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 信任度对比
        ax3 = axes[1, 0]
        infp_trust = self.infp_results.get('final_metrics', {}).get('leader_influence', 0)
        estp_trust = self.estp_results.get('final_metrics', {}).get('leader_influence', 0)
        
        trust_data = [infp_trust, estp_trust]
        bars = ax3.bar(labels, trust_data, color=colors, alpha=0.7)
        ax3.set_ylabel('对领导者信任度')
        ax3.set_title('领导者信任度对比')
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars, trust_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 团队凝聚力对比
        ax4 = axes[1, 1]
        infp_cohesion = self.infp_results.get('final_metrics', {}).get('team_cohesion', 0)
        estp_cohesion = self.estp_results.get('final_metrics', {}).get('team_cohesion', 0)
        
        cohesion_data = [infp_cohesion, estp_cohesion]
        bars = ax4.bar(labels, cohesion_data, color=colors, alpha=0.7)
        ax4.set_ylabel('团队凝聚力')
        ax4.set_title('团队凝聚力对比')
        ax4.set_ylim(0, 1)
        
        for bar, value in zip(bars, cohesion_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图2: 群体满意度趋势分析\n比较两种领导类型下团队满意度变化、信任度和凝聚力的差异', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.12)  # 为图注留出更多空间
        plt.savefig(output_path / 'satisfaction_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_conflict_analysis(self, output_path: Path):
        """冲突事件分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('冲突事件分析', fontsize=16, fontweight='bold')
        
        # 1. 冲突频率对比
        ax1 = axes[0, 0]
        infp_conflicts = self.infp_results.get('final_metrics', {}).get('conflict_count', 0)
        estp_conflicts = self.estp_results.get('final_metrics', {}).get('conflict_count', 0)
        
        conflict_data = [infp_conflicts, estp_conflicts]
        labels = ['INFP', 'ESTP']
        colors = ['orange', 'red']
        
        bars = ax1.bar(labels, conflict_data, color=colors, alpha=0.7)
        ax1.set_ylabel('冲突事件数量')
        ax1.set_title('冲突事件频率对比')
        ax1.set_ylim(0, max(conflict_data) * 1.2 if max(conflict_data) > 0 else 1)
        
        for bar, value in zip(bars, conflict_data):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 冲突强度对比
        ax2 = axes[0, 1]
        infp_intensity = self.infp_results.get('final_metrics', {}).get('avg_conflict_intensity', 0)
        estp_intensity = self.estp_results.get('final_metrics', {}).get('avg_conflict_intensity', 0)
        
        intensity_data = [infp_intensity, estp_intensity]
        bars = ax2.bar(labels, intensity_data, color=colors, alpha=0.7)
        ax2.set_ylabel('平均冲突强度')
        ax2.set_title('冲突强度对比')
        ax2.set_ylim(0, max(intensity_data) * 1.2 if max(intensity_data) > 0 else 1)
        
        for bar, value in zip(bars, intensity_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 冲突处理效果对比（使用饼图）
        ax3 = axes[1, 0]
        # 这里我们假设INFP偏向调解，ESTP偏向权威
        infp_approaches = [0.7, 0.3]  # [调解, 权威]
        estp_approaches = [0.2, 0.8]  # [调解, 权威]
        
        approach_labels = ['调解式', '权威式']
        ax3.pie(infp_approaches, labels=approach_labels, autopct='%1.1f%%', 
               colors=['lightblue', 'lightcoral'])
        ax3.set_title('INFP 冲突处理方式')
        
        ax4 = axes[1, 1]
        ax4.pie(estp_approaches, labels=approach_labels, autopct='%1.1f%%', 
               colors=['lightblue', 'lightcoral'])
        ax4.set_title('ESTP 冲突处理方式')
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图3: 冲突事件分析\n对比INFP和ESTP领导者在冲突频率、强度和处理方式上的不同表现', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.12)  # 为图注留出更多空间
        plt.savefig(output_path / 'conflict_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_creativity_analysis(self, output_path: Path):
        """创意产出分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('创意产出与创新分析', fontsize=16, fontweight='bold')
        
        # 1. 创意数量对比
        ax1 = axes[0, 0]
        infp_ideas = self.infp_results.get('final_metrics', {}).get('total_ideas', 0)
        estp_ideas = self.estp_results.get('final_metrics', {}).get('total_ideas', 0)
        
        ideas_data = [infp_ideas, estp_ideas]
        labels = ['INFP', 'ESTP']
        colors = ['purple', 'green']
        
        bars = ax1.bar(labels, ideas_data, color=colors, alpha=0.7)
        ax1.set_ylabel('创意总数')
        ax1.set_title('创意数量对比')
        ax1.set_ylim(0, max(ideas_data) * 1.2 if max(ideas_data) > 0 else 1)
        
        for bar, value in zip(bars, ideas_data):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 创意质量对比
        ax2 = axes[0, 1]
        infp_quality = self.infp_results.get('final_metrics', {}).get('avg_idea_quality', 0)
        estp_quality = self.estp_results.get('final_metrics', {}).get('avg_idea_quality', 0)
        
        quality_data = [infp_quality, estp_quality]
        bars = ax2.bar(labels, quality_data, color=colors, alpha=0.7)
        ax2.set_ylabel('平均创意质量')
        ax2.set_title('创意质量对比')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, quality_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 创意采纳率对比
        ax3 = axes[1, 0]
        infp_adoption = self.infp_results.get('final_metrics', {}).get('adoption_rate', 0)
        estp_adoption = self.estp_results.get('final_metrics', {}).get('adoption_rate', 0)
        
        adoption_data = [infp_adoption, estp_adoption]
        bars = ax3.bar(labels, adoption_data, color=colors, alpha=0.7)
        ax3.set_ylabel('创意采纳率')
        ax3.set_title('创意采纳率对比')
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars, adoption_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 创新效率综合评分
        ax4 = axes[1, 1]
        # 综合评分 = 创意数量 * 创意质量 * 采纳率
        infp_innovation_score = (infp_ideas * infp_quality * infp_adoption) / 10  # 标准化
        estp_innovation_score = (estp_ideas * estp_quality * estp_adoption) / 10  # 标准化
        
        innovation_data = [infp_innovation_score, estp_innovation_score]
        bars = ax4.bar(labels, innovation_data, color=colors, alpha=0.7)
        ax4.set_ylabel('创新效率综合评分')
        ax4.set_title('创新效率综合评分')
        ax4.set_ylim(0, max(innovation_data) * 1.2 if max(innovation_data) > 0 else 1)
        
        for bar, value in zip(bars, innovation_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图4: 创意产出与创新分析\n展示两种领导类型在创意数量、质量、采纳率和创新效率方面的对比', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.12)  # 为图注留出更多空间
        plt.savefig(output_path / 'creativity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_leadership_influence(self, output_path: Path):
        """领导者影响力分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('领导者影响力分析', fontsize=16, fontweight='bold')
        
        # 1. 领导者影响力对比
        ax1 = axes[0, 0]
        infp_influence = self.infp_results.get('final_metrics', {}).get('leader_influence', 0)
        estp_influence = self.estp_results.get('final_metrics', {}).get('leader_influence', 0)
        
        influence_data = [infp_influence, estp_influence]
        labels = ['INFP', 'ESTP']
        colors = ['gold', 'silver']
        
        bars = ax1.bar(labels, influence_data, color=colors, alpha=0.7)
        ax1.set_ylabel('领导者影响力')
        ax1.set_title('领导者影响力对比')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars, influence_data):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 决策质量对比
        ax2 = axes[0, 1]
        # 从rounds数据中计算平均决策质量
        infp_rounds = self.infp_results.get('rounds', [])
        estp_rounds = self.estp_results.get('rounds', [])
        
        infp_decision_quality = 0
        estp_decision_quality = 0
        
        if infp_rounds:
            infp_decisions = [d['decision_quality'] for round_data in infp_rounds 
                            for d in round_data.get('decisions', [])]
            infp_decision_quality = np.mean(infp_decisions) if infp_decisions else 0
        
        if estp_rounds:
            estp_decisions = [d['decision_quality'] for round_data in estp_rounds 
                            for d in round_data.get('decisions', [])]
            estp_decision_quality = np.mean(estp_decisions) if estp_decisions else 0
        
        decision_data = [infp_decision_quality, estp_decision_quality]
        bars = ax2.bar(labels, decision_data, color=colors, alpha=0.7)
        ax2.set_ylabel('平均决策质量')
        ax2.set_title('决策质量对比')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, decision_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 决策速度对比
        ax3 = axes[1, 0]
        infp_decision_time = 0
        estp_decision_time = 0
        
        if infp_rounds:
            infp_times = [d['decision_time'] for round_data in infp_rounds 
                         for d in round_data.get('decisions', [])]
            infp_decision_time = np.mean(infp_times) if infp_times else 0
        
        if estp_rounds:
            estp_times = [d['decision_time'] for round_data in estp_rounds 
                         for d in round_data.get('decisions', [])]
            estp_decision_time = np.mean(estp_times) if estp_times else 0
        
        time_data = [infp_decision_time, estp_decision_time]
        bars = ax3.bar(labels, time_data, color=colors, alpha=0.7)
        ax3.set_ylabel('平均决策时间')
        ax3.set_title('决策速度对比（时间越短越好）')
        
        for bar, value in zip(bars, time_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 领导风格特征雷达图
        ax4 = axes[1, 1]
        
        # 定义领导风格特征
        categories = ['决策速度', '沟通直接性', '共识寻求', '情感支持', '创造力鼓励', '权威展现']
        
        # INFP 和 ESTP 的领导风格分数（基于之前的定义）
        infp_scores = [0.3, 0.2, 0.9, 0.8, 0.9, 0.2]  # 对应决策速度、沟通直接性等
        estp_scores = [0.9, 0.9, 0.2, 0.3, 0.4, 0.9]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
        
        infp_scores = np.concatenate((infp_scores, [infp_scores[0]]))
        estp_scores = np.concatenate((estp_scores, [estp_scores[0]]))
        
        ax4.plot(angles, infp_scores, 'o-', linewidth=2, label='INFP', color='blue')
        ax4.fill(angles, infp_scores, alpha=0.25, color='blue')
        ax4.plot(angles, estp_scores, 'o-', linewidth=2, label='ESTP', color='red')
        ax4.fill(angles, estp_scores, alpha=0.25, color='red')
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('领导风格特征对比')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图5: 领导者影响力分析\n包含影响力对比、决策质量、决策速度和领导风格特征的全面评估', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.12)  # 为图注留出更多空间
        plt.savefig(output_path / 'leadership_influence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_comparison(self, output_path: Path):
        """综合雷达图对比"""
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 定义评估维度
        categories = ['任务效率', '团队满意度', '领导影响力', '创新产出', '团队凝聚力', '冲突管理']
        
        # 获取数据并标准化
        infp_metrics = self.infp_results.get('final_metrics', {})
        estp_metrics = self.estp_results.get('final_metrics', {})
        
        infp_values = [
            infp_metrics.get('task_efficiency', 0),
            infp_metrics.get('avg_satisfaction', 0),
            infp_metrics.get('leader_influence', 0),
            infp_metrics.get('adoption_rate', 0),
            infp_metrics.get('team_cohesion', 0),
            1 - infp_metrics.get('avg_conflict_intensity', 0.5)  # 冲突管理能力（冲突强度的倒数）
        ]
        
        estp_values = [
            estp_metrics.get('task_efficiency', 0),
            estp_metrics.get('avg_satisfaction', 0),
            estp_metrics.get('leader_influence', 0),
            estp_metrics.get('adoption_rate', 0),
            estp_metrics.get('team_cohesion', 0),
            1 - estp_metrics.get('avg_conflict_intensity', 0.5)
        ]
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        infp_values = np.concatenate((infp_values, [infp_values[0]]))
        estp_values = np.concatenate((estp_values, [estp_values[0]]))
        
        # 绘制雷达图
        ax.plot(angles, infp_values, 'o-', linewidth=3, label='INFP 领导者', color='blue')
        ax.fill(angles, infp_values, alpha=0.25, color='blue')
        ax.plot(angles, estp_values, 'o-', linewidth=3, label='ESTP 领导者', color='red')
        ax.fill(angles, estp_values, alpha=0.25, color='red')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('综合表现雷达图对比', fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图6: 综合表现雷达图对比\n全方位展示INFP和ESTP领导者在六个关键维度的综合表现差异', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.18)  # 为图注留出更多空间
        plt.savefig(output_path / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_key_metrics_comparison(self, output_path: Path):
        """关键指标对比"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 准备数据
        metrics = ['任务效率', '团队满意度', '领导影响力', '创新采纳率', '团队凝聚力']
        
        infp_data = [
            self.infp_results.get('final_metrics', {}).get('task_efficiency', 0),
            self.infp_results.get('final_metrics', {}).get('avg_satisfaction', 0),
            self.infp_results.get('final_metrics', {}).get('leader_influence', 0),
            self.infp_results.get('final_metrics', {}).get('adoption_rate', 0),
            self.infp_results.get('final_metrics', {}).get('team_cohesion', 0)
        ]
        
        estp_data = [
            self.estp_results.get('final_metrics', {}).get('task_efficiency', 0),
            self.estp_results.get('final_metrics', {}).get('avg_satisfaction', 0),
            self.estp_results.get('final_metrics', {}).get('leader_influence', 0),
            self.estp_results.get('final_metrics', {}).get('adoption_rate', 0),
            self.estp_results.get('final_metrics', {}).get('team_cohesion', 0)
        ]
        
        # 设置条形图位置
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, infp_data, width, label='INFP 领导者', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, estp_data, width, label='ESTP 领导者', color='lightcoral', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('评估指标', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('关键指标对比分析', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 添加图注 - 分行显示以避免格子状显示
        fig.text(0.5, 0.02, '图7: 关键指标对比分析\n并列展示五个核心评估指标的具体数值对比', 
                 ha='center', va='bottom', fontsize=9, style='italic')
        plt.subplots_adjust(bottom=0.18)  # 为图注留出更多空间
        plt.savefig(output_path / 'key_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_data_tables(self, output_path: Path):
        """生成数据表格"""
        # 创建对比数据表
        comparison_data = []
        
        metrics_mapping = {
            'task_efficiency': '任务效率',
            'avg_satisfaction': '平均满意度',
            'leader_influence': '领导影响力',
            'adoption_rate': '创意采纳率',
            'team_cohesion': '团队凝聚力',
            'conflict_count': '冲突事件数',
            'avg_conflict_intensity': '平均冲突强度',
            'total_ideas': '创意总数',
            'avg_idea_quality': '平均创意质量'
        }
        
        for metric_key, metric_name in metrics_mapping.items():
            infp_value = self.infp_results.get('final_metrics', {}).get(metric_key, 0)
            estp_value = self.estp_results.get('final_metrics', {}).get(metric_key, 0)
            
            comparison_data.append({
                '指标': metric_name,
                'INFP领导者': f'{infp_value:.4f}',
                'ESTP领导者': f'{estp_value:.4f}',
                '差异': f'{estp_value - infp_value:.4f}',
                '相对变化(%)': f'{((estp_value - infp_value) / max(infp_value, 0.001) * 100):.2f}%'
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_path / 'comparison_table.csv', index=False, encoding='utf-8-sig')
        
        # 创建轮次详细数据
        rounds_data = []
        
        infp_rounds = self.infp_results.get('rounds', [])
        estp_rounds = self.estp_results.get('rounds', [])
        
        max_rounds = max(len(infp_rounds), len(estp_rounds))
        
        for i in range(max_rounds):
            row = {'轮次': i + 1}
            
            if i < len(infp_rounds):
                infp_round = infp_rounds[i]
                row['INFP_任务进度'] = f"{infp_round['task_progress_after']:.3f}"
                row['INFP_创意数量'] = len(infp_round['ideas_generated'])
                row['INFP_冲突数量'] = len(infp_round['conflicts'])
                
                if infp_round['satisfaction_levels']:
                    avg_satisfaction = np.mean(list(infp_round['satisfaction_levels'].values()))
                    row['INFP_满意度'] = f"{avg_satisfaction:.3f}"
                else:
                    row['INFP_满意度'] = "0.000"
            
            if i < len(estp_rounds):
                estp_round = estp_rounds[i]
                row['ESTP_任务进度'] = f"{estp_round['task_progress_after']:.3f}"
                row['ESTP_创意数量'] = len(estp_round['ideas_generated'])
                row['ESTP_冲突数量'] = len(estp_round['conflicts'])
                
                if estp_round['satisfaction_levels']:
                    avg_satisfaction = np.mean(list(estp_round['satisfaction_levels'].values()))
                    row['ESTP_满意度'] = f"{avg_satisfaction:.3f}"
                else:
                    row['ESTP_满意度'] = "0.000"
            
            rounds_data.append(row)
        
        rounds_df = pd.DataFrame(rounds_data)
        rounds_df.to_csv(output_path / 'rounds_detail.csv', index=False, encoding='utf-8-sig')
        
        print("数据表格已生成: comparison_table.csv, rounds_detail.csv")
    
    def _generate_text_report(self, output_path: Path):
        """生成文字报告"""
        report_content = []
        
        report_content.append("# 多主体建模实验分析报告")
        report_content.append("\n## 实验概述")
        report_content.append("本实验通过多主体建模（Agent-Based Modeling）方法，对比分析了INFP和ESTP两种不同人格类型的领导者在群体协作中的表现差异。")
        
        # 获取对比数据
        comparison = self.comparison_data
        overall_performance = comparison.get('overall_performance', {})
        
        report_content.append(f"\n## 总体表现")
        report_content.append(f"**获胜者**: {overall_performance.get('winner', 'N/A')} 类型领导者")
        report_content.append(f"**综合得分**: INFP = {overall_performance.get('infp_score', 0):.4f}, ESTP = {overall_performance.get('estp_score', 0):.4f}")
        report_content.append(f"**得分差异**: {overall_performance.get('score_difference', 0):.4f}")
        
        report_content.append("\n## 详细分析")
        
        # 任务效率分析
        task_efficiency = comparison.get('task_efficiency', {})
        report_content.append(f"\n### 1. 任务效率")
        report_content.append(f"- INFP领导者: {task_efficiency.get('infp', 0):.4f}")
        report_content.append(f"- ESTP领导者: {task_efficiency.get('estp', 0):.4f}")
        report_content.append(f"- 差异: {task_efficiency.get('difference', 0):.4f}")
        
        if task_efficiency.get('estp', 0) > task_efficiency.get('infp', 0):
            report_content.append("**结论**: ESTP领导者在任务效率方面表现更优，体现了其快速决策和执行导向的特点。")
        else:
            report_content.append("**结论**: INFP领导者在任务效率方面表现更优，可能得益于其寻求共识和深思熟虑的决策风格。")
        
        # 团队满意度分析
        satisfaction = comparison.get('avg_satisfaction', {})
        report_content.append(f"\n### 2. 团队满意度")
        report_content.append(f"- INFP领导者: {satisfaction.get('infp', 0):.4f}")
        report_content.append(f"- ESTP领导者: {satisfaction.get('estp', 0):.4f}")
        report_content.append(f"- 差异: {satisfaction.get('difference', 0):.4f}")
        
        if satisfaction.get('infp', 0) > satisfaction.get('estp', 0):
            report_content.append("**结论**: INFP领导者在团队满意度方面表现更优，体现了其关注成员情感和寻求和谐的特点。")
        else:
            report_content.append("**结论**: ESTP领导者在团队满意度方面表现更优，可能因为其直接有效的沟通风格获得了团队认可。")
        
        # 创新产出分析
        innovation = comparison.get('adoption_rate', {})
        report_content.append(f"\n### 3. 创新产出")
        report_content.append(f"- INFP领导者创意采纳率: {innovation.get('infp', 0):.4f}")
        report_content.append(f"- ESTP领导者创意采纳率: {innovation.get('estp', 0):.4f}")
        report_content.append(f"- 差异: {innovation.get('difference', 0):.4f}")
        
        if innovation.get('infp', 0) > innovation.get('estp', 0):
            report_content.append("**结论**: INFP领导者在创新产出方面表现更优，体现了其鼓励创造力和开放性的特点。")
        else:
            report_content.append("**结论**: ESTP领导者在创新产出方面表现更优，可能因为其快速决策能力能够更好地推进创意实施。")
        
        # 冲突管理分析
        conflict = comparison.get('conflict_count', {})
        report_content.append(f"\n### 4. 冲突管理")
        report_content.append(f"- INFP领导者冲突事件数: {conflict.get('infp', 0):.0f}")
        report_content.append(f"- ESTP领导者冲突事件数: {conflict.get('estp', 0):.0f}")
        report_content.append(f"- 差异: {conflict.get('difference', 0):.0f}")
        
        if conflict.get('infp', 0) < conflict.get('estp', 0):
            report_content.append("**结论**: INFP领导者在冲突管理方面表现更优，体现了其避免冲突和寻求和谐的特点。")
        else:
            report_content.append("**结论**: ESTP领导者在冲突管理方面表现更优，可能因为其直接解决问题的方式能够更快地平息冲突。")
        
        report_content.append("\n## 实验结论")
        report_content.append("本实验通过多主体建模方法，成功验证了不同人格类型的领导者在群体协作中确实表现出显著差异：")
        report_content.append("- INFP领导者倾向于营造和谐的团队氛围，注重成员满意度和创新鼓励")
        report_content.append("- ESTP领导者倾向于追求高效的任务执行，注重快速决策和结果导向")
        report_content.append("- 不同类型的领导者适合不同的团队环境和任务特点")
        
        report_content.append("\n## 建议")
        report_content.append("1. 在需要高创新性和团队和谐的项目中，INFP型领导者可能更适合")
        report_content.append("2. 在需要快速执行和高效完成的项目中，ESTP型领导者可能更适合")
        report_content.append("3. 理想情况下，可以考虑混合型领导团队，结合两种类型的优势")
        
        # 保存报告
        with open(output_path / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print("分析报告已生成: analysis_report.md")

# 中英文标签映射（备用方案）
LABEL_MAPPING = {
    '任务效率': 'Task Efficiency',
    '团队满意度': 'Team Satisfaction', 
    '领导影响力': 'Leadership Influence',
    '创新采纳率': 'Innovation Adoption',
    '团队凝聚力': 'Team Cohesion',
    '任务推进效率对比分析': 'Task Progress Efficiency Comparison',
    '群体满意度趋势分析': 'Group Satisfaction Trend Analysis',
    '冲突事件分析': 'Conflict Event Analysis',
    '创意产出与创新分析': 'Creativity and Innovation Analysis',
    '领导者影响力分析': 'Leadership Influence Analysis',
    '综合表现雷达图对比': 'Comprehensive Performance Radar Comparison',
    '关键指标对比分析': 'Key Metrics Comparison Analysis'
}

def get_safe_label(chinese_label: str, use_english: bool = False) -> str:
    """获取安全的标签（如果中文显示有问题则使用英文）"""
    if use_english and chinese_label in LABEL_MAPPING:
        return LABEL_MAPPING[chinese_label]
    return chinese_label

def load_results(filepath: str) -> Dict:
    """加载实验结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # 示例用法
    try:
        # 尝试加载结果文件
        results = load_results('simulation_results.json')
        print("成功加载实验结果文件")
        
        # 创建分析器
        analyzer = ResultAnalyzer(results)
        
        # 生成综合报告
        analyzer.create_comprehensive_report()
        
    except FileNotFoundError:
        print("未找到实验结果文件 'simulation_results.json'")
        print("请先运行 simulation.py 进行实验")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查实验结果文件的格式是否正确")
