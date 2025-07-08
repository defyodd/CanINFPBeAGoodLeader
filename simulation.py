import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from agents import Agent, LeaderAgent, PersonalityType, create_team

@dataclass
class SimulationConfig:
    """仿真配置"""
    team_size: int = 6
    max_rounds: int = 30  # 增加轮次以体现差异
    task_complexity: float = 2.5  # 提高基础复杂度（不再限制在0-1）
    external_pressure: float = 0.3
    random_seed: Optional[int] = None
    difficulty_scaling: bool = True  # 是否启用难度递增
    min_quality_threshold: float = 0.6  # 最低质量要求

class Task:
    """任务类"""
    def __init__(self, task_id: int, complexity: float, required_rounds: int, config: SimulationConfig = None):
        self.task_id = task_id
        self.base_complexity = complexity
        self.complexity = complexity
        self.required_rounds = required_rounds
        self.progress = 0.0
        self.quality = 0.0
        self.revisions_needed = 0
        self.ideas_incorporated = []
        self.contributors = set()
        self.current_round = 0
        self.config = config or SimulationConfig()
        
    def update_progress(self, work_quality: float, efficiency: float, round_num: int):
        """更新任务进度"""
        self.current_round = round_num
        
        # 动态调整复杂度（随轮次增加而增加）
        if self.config.difficulty_scaling:
            difficulty_multiplier = 1 + (round_num / self.required_rounds) * 0.5
            self.complexity = self.base_complexity * difficulty_multiplier
        
        # 改进的进度计算公式
        # 基础进度增量：考虑团队效率和工作质量
        base_increment = efficiency * work_quality * 0.3
        
        # 复杂度惩罚：越复杂的任务推进越慢
        complexity_factor = 1.0 / (1.0 + self.complexity)
        
        # 质量要求：如果质量不达标，进度受限
        quality_factor = 1.0 if work_quality >= self.config.min_quality_threshold else 0.5
        
        # 最终进度增量
        progress_increment = base_increment * complexity_factor * quality_factor
        
        self.progress += progress_increment
        self.progress = min(1.0, self.progress)
        
        # 更新质量（使用更保守的更新策略）
        self.quality = 0.8 * self.quality + 0.2 * work_quality
        
        # 更严格的返工检查
        rework_probability = 0.4 if work_quality < self.config.min_quality_threshold else 0.1
        if work_quality < 0.6 and random.random() < rework_probability:
            self.revisions_needed += 1
            # 返工导致更大的进度损失
            progress_loss = 0.15 + (1.0 - work_quality) * 0.1
            self.progress = max(0.0, self.progress - progress_loss)

class GroupDynamics:
    """群体动力学模型"""
    def __init__(self, leader: LeaderAgent, members: List[Agent]):
        self.leader = leader
        self.members = members
        self.all_agents = [leader] + members
        
        # 群体状态
        self.cohesion = 0.5  # 凝聚力
        self.communication_efficiency = 0.5  # 沟通效率
        self.trust_network = self._initialize_trust_network()
        self.conflict_level = 0.0  # 冲突水平
        
        # 统计数据
        self.round_stats = []
        self.conflict_events = []
        self.idea_history = []
        self.communication_history = []
        
    def _initialize_trust_network(self) -> Dict:
        """初始化信任网络"""
        trust_network = {}
        for agent in self.all_agents:
            trust_network[agent.agent_id] = {}
            for other_agent in self.all_agents:
                if agent.agent_id != other_agent.agent_id:
                    # 初始信任值基于人格相似性
                    trust_network[agent.agent_id][other_agent.agent_id] = random.uniform(0.3, 0.7)
        return trust_network
    
    def simulate_round(self, round_num: int, task: Task) -> Dict:
        """模拟一轮交互"""
        round_results = {
            'round': round_num,
            'task_progress_before': task.progress,
            'conflicts': [],
            'ideas_generated': [],
            'communications': [],
            'decisions': [],
            'satisfaction_levels': {},
            'trust_levels': {},
            'stress_levels': {},
            'random_events': []
        }
        
        # 0. 处理随机事件
        random_events = self._handle_random_events(round_num, task)
        round_results['random_events'] = random_events
        
        # 1. 成员生成创意
        task_context = {'current_round': round_num, 'task_progress': task.progress}
        for member in self.members:
            idea = member.generate_idea(task_context)
            if idea:
                round_results['ideas_generated'].append(idea)
                self.idea_history.append(idea)
        
        # 2. 领导者沟通
        communications = self.leader.communicate_with_team(self.members, 'task_update')
        round_results['communications'] = communications
        self.communication_history.extend(communications)
        
        # 3. 模拟冲突
        conflict_probability = 0.1 + self.conflict_level * 0.2
        if random.random() < conflict_probability:
            conflict = self._generate_conflict(round_num)
            round_results['conflicts'].append(conflict)
            self.conflict_events.append(conflict)
        
        # 4. 领导者决策
        member_inputs = [
            {
                'agent_id': member.agent_id,
                'agreement': member.trust_in_leader,
                'idea_quality': max([idea['quality'] for idea in round_results['ideas_generated'] 
                                   if idea['agent_id'] == member.agent_id], default=0)
            }
            for member in self.members
        ]
        
        decision = self.leader.make_decision(task_context, member_inputs)
        round_results['decisions'].append(decision)
        
        # 5. 更新任务进度
        work_quality = decision['decision_quality']
        efficiency = self._calculate_team_efficiency()
        task.update_progress(work_quality, efficiency, round_num)
        
        # 6. 记录任务进度（在更新智能体状态之前）
        round_results['task_progress_after'] = task.progress
        
        # 7. 更新智能体状态
        self._update_agent_states(decision, round_results)
        round_results['satisfaction_levels'] = {agent.agent_id: agent.satisfaction for agent in self.all_agents}
        round_results['trust_levels'] = {member.agent_id: member.trust_in_leader for member in self.members}
        round_results['stress_levels'] = {agent.agent_id: agent.stress_level for agent in self.all_agents}
        
        # 8. 处理随机事件
        random_events = self._handle_random_events(round_num, task)
        round_results['random_events'] = random_events
        
        self.round_stats.append(round_results)
        return round_results
    
    def _generate_conflict(self, round_num: int) -> Dict:
        """生成冲突事件"""
        # 随机选择冲突参与者
        conflict_parties = random.sample([agent.agent_id for agent in self.members], 
                                       min(2, len(self.members)))
        conflict_intensity = random.uniform(0.2, 0.8)
        
        # 让参与者响应冲突
        responses = []
        for party_id in conflict_parties:
            agent = next(agent for agent in self.all_agents if agent.agent_id == party_id)
            response = agent.respond_to_conflict(conflict_intensity)
            responses.append(response)
        
        # 领导者处理冲突
        resolution = self.leader.handle_conflict(conflict_parties, conflict_intensity)
        
        # 更新冲突水平
        if resolution['effectiveness'] > 0.6:
            self.conflict_level *= 0.8
        else:
            self.conflict_level = min(1.0, self.conflict_level + 0.1)
        
        return {
            'round': round_num,
            'parties': conflict_parties,
            'intensity': conflict_intensity,
            'responses': responses,
            'resolution': resolution,
            'new_conflict_level': self.conflict_level
        }
    
    def _calculate_team_efficiency(self) -> float:
        """计算团队效率"""
        avg_satisfaction = sum(agent.satisfaction for agent in self.all_agents) / len(self.all_agents)
        avg_trust = sum(member.trust_in_leader for member in self.members) / len(self.members)
        avg_stress = sum(agent.stress_level for agent in self.all_agents) / len(self.all_agents)
        
        efficiency = (avg_satisfaction * 0.4 + 
                     avg_trust * 0.3 + 
                     (1 - avg_stress) * 0.3)
        
        return max(0.1, min(1.0, efficiency))
    
    def _update_agent_states(self, decision: Dict, round_results: Dict):
        """更新智能体状态"""
        leadership_style = self.leader.leadership_style
        task_progress = round_results['task_progress_after']
        
        for member in self.members:
            # 更新满意度
            member.update_satisfaction(leadership_style, task_progress)
            
            # 更新动机（基于满意度和信任）
            member.motivation = 0.5 * member.satisfaction + 0.5 * member.trust_in_leader
            
            # 处理创意采纳
            member_ideas = [idea for idea in round_results['ideas_generated'] 
                          if idea['agent_id'] == member.agent_id]
            if member_ideas and random.random() < decision['decision_quality']:
                member.ideas_adopted += 1
                member.creativity_score += 0.1
        
        # 更新群体凝聚力
        avg_satisfaction = sum(agent.satisfaction for agent in self.all_agents) / len(self.all_agents)
        self.cohesion = 0.7 * self.cohesion + 0.3 * avg_satisfaction

    def _handle_random_events(self, round_num: int, task: Task) -> Dict:
        """处理随机事件"""
        events = []
        
        # 1. 外部压力变化（20%概率）
        if random.random() < 0.2:
            pressure_change = random.uniform(-0.2, 0.3)
            events.append({
                'type': 'pressure_change',
                'description': '外部压力变化',
                'impact': pressure_change,
                'round': round_num
            })
            
            # 影响所有成员的压力水平
            for agent in self.all_agents:
                agent.stress_level = max(0, min(1, agent.stress_level + pressure_change))
        
        # 2. 技术难题（15%概率，复杂度高时概率更大）
        tech_problem_prob = 0.15 + (task.complexity - 2.0) * 0.05
        if random.random() < tech_problem_prob:
            complexity_increase = random.uniform(0.1, 0.4)
            task.complexity += complexity_increase
            events.append({
                'type': 'technical_challenge',
                'description': '技术难题出现',
                'impact': complexity_increase,
                'round': round_num
            })
        
        # 3. 团队成员疲劳（轮次越多概率越大）
        fatigue_prob = (round_num / 30) * 0.25
        if random.random() < fatigue_prob:
            affected_member = random.choice(self.members)
            motivation_loss = random.uniform(0.1, 0.3)
            affected_member.motivation = max(0, affected_member.motivation - motivation_loss)
            affected_member.stress_level = min(1, affected_member.stress_level + motivation_loss * 0.5)
            
            events.append({
                'type': 'member_fatigue',
                'description': f'成员{affected_member.name}出现疲劳',
                'impact': motivation_loss,
                'affected_member': affected_member.agent_id,
                'round': round_num
            })
        
        # 4. 积极事件：突破性进展（10%概率）
        if random.random() < 0.1:
            progress_boost = random.uniform(0.05, 0.15)
            task.progress = min(1.0, task.progress + progress_boost)
            
            # 提升团队士气
            for agent in self.all_agents:
                agent.satisfaction = min(1, agent.satisfaction + 0.1)
                agent.motivation = min(1, agent.motivation + 0.1)
            
            events.append({
                'type': 'breakthrough',
                'description': '获得突破性进展',
                'impact': progress_boost,
                'round': round_num
            })
        
        return events

class Simulation:
    """仿真主类"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        self.results = {
            'infp_results': [],
            'estp_results': [],
            'comparison_data': None
        }
    
    def run_experiment(self) -> Dict:
        """运行完整实验"""
        print("开始运行多主体建模实验...")
        
        # 运行INFP领导者实验
        print("\n=== 运行INFP领导者实验 ===")
        infp_results = self._run_single_experiment(PersonalityType.INFP)
        self.results['infp_results'] = infp_results
        
        # 运行ESTP领导者实验
        print("\n=== 运行ESTP领导者实验 ===")
        estp_results = self._run_single_experiment(PersonalityType.ESTP)
        self.results['estp_results'] = estp_results
        
        # 对比分析
        print("\n=== 进行对比分析 ===")
        comparison_data = self._compare_results(infp_results, estp_results)
        self.results['comparison_data'] = comparison_data
        
        return self.results
    
    def _run_single_experiment(self, leader_type: PersonalityType) -> Dict:
        """运行单个实验"""
        # 创建团队
        leader, members = create_team(self.config.team_size, leader_type)
        
        # 创建任务
        task = Task(
            task_id=1,
            complexity=self.config.task_complexity,
            required_rounds=self.config.max_rounds,
            config=self.config
        )
        
        # 创建群体动力学模型
        group_dynamics = GroupDynamics(leader, members)
        
        # 运行仿真
        experiment_results = {
            'leader_type': leader_type.value,
            'team_size': self.config.team_size,
            'rounds': [],
            'final_metrics': {},
            'task_completion': {}
        }
        
        for round_num in range(1, self.config.max_rounds + 1):
            round_result = group_dynamics.simulate_round(round_num, task)
            experiment_results['rounds'].append(round_result)
            
            # 打印进度
            if round_num % 5 == 0:
                avg_satisfaction = np.mean(list(round_result['satisfaction_levels'].values())) if round_result['satisfaction_levels'] else 0
                print(f"  轮次 {round_num}/{self.config.max_rounds}: "
                      f"任务进度 {task.progress:.2%}, "
                      f"团队满意度 {avg_satisfaction:.2%}")
            
            # 检查任务完成
            if task.progress >= 1.0:
                print(f"  任务在第 {round_num} 轮完成!")
                break
        
        # 计算最终指标
        final_metrics = self._calculate_final_metrics(experiment_results, task, group_dynamics)
        experiment_results['final_metrics'] = final_metrics
        experiment_results['task_completion'] = {
            'progress': task.progress,
            'quality': task.quality,
            'revisions_needed': task.revisions_needed,
            'completion_round': round_num if task.progress >= 1.0 else None
        }
        
        return experiment_results
    
    def _calculate_final_metrics(self, experiment_results: Dict, task: Task, 
                               group_dynamics: GroupDynamics) -> Dict:
        """计算最终指标"""
        rounds = experiment_results['rounds']
        
        # 任务推进效率
        task_efficiency = task.progress / len(rounds)
        rework_ratio = task.revisions_needed / max(1, len(rounds))
        
        # 群体满意度
        satisfaction_scores = []
        for round_data in rounds:
            if round_data['satisfaction_levels']:
                avg_satisfaction = np.mean(list(round_data['satisfaction_levels'].values()))
                satisfaction_scores.append(avg_satisfaction)
        
        # 冲突事件统计
        conflict_count = len(group_dynamics.conflict_events)
        avg_conflict_intensity = np.mean([c['intensity'] for c in group_dynamics.conflict_events]) if group_dynamics.conflict_events else 0
        
        # 创意统计
        total_ideas = len(group_dynamics.idea_history)
        avg_idea_quality = np.mean([idea['quality'] for idea in group_dynamics.idea_history]) if group_dynamics.idea_history else 0
        adopted_ideas = sum(member.ideas_adopted for member in group_dynamics.members)
        adoption_rate = adopted_ideas / max(1, total_ideas)
        
        # 领导者影响力
        final_trust_scores = []
        if rounds:
            last_round = rounds[-1]
            if last_round['trust_levels']:
                final_trust_scores = list(last_round['trust_levels'].values())
        
        leader_influence = np.mean(final_trust_scores) if final_trust_scores else 0
        
        return {
            'task_efficiency': task_efficiency,
            'rework_ratio': rework_ratio,
            'avg_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0,
            'satisfaction_trend': satisfaction_scores,
            'conflict_count': conflict_count,
            'avg_conflict_intensity': avg_conflict_intensity,
            'total_ideas': total_ideas,
            'avg_idea_quality': avg_idea_quality,
            'adoption_rate': adoption_rate,
            'leader_influence': leader_influence,
            'team_cohesion': group_dynamics.cohesion
        }
    
    def _compare_results(self, infp_results: Dict, estp_results: Dict) -> Dict:
        """对比分析结果"""
        infp_metrics = infp_results['final_metrics']
        estp_metrics = estp_results['final_metrics']
        
        comparison = {}
        
        # 对比各项指标
        metrics_to_compare = [
            'task_efficiency', 'rework_ratio', 'avg_satisfaction',
            'conflict_count', 'avg_conflict_intensity', 'total_ideas',
            'avg_idea_quality', 'adoption_rate', 'leader_influence', 'team_cohesion'
        ]
        
        for metric in metrics_to_compare:
            infp_value = infp_metrics.get(metric, 0)
            estp_value = estp_metrics.get(metric, 0)
            
            comparison[metric] = {
                'infp': infp_value,
                'estp': estp_value,
                'difference': estp_value - infp_value,
                'relative_change': (estp_value - infp_value) / max(infp_value, 0.001) * 100
            }
        
        # 综合评估
        infp_score = (infp_metrics['task_efficiency'] * 0.3 + 
                     infp_metrics['avg_satisfaction'] * 0.25 + 
                     infp_metrics['leader_influence'] * 0.2 + 
                     infp_metrics['team_cohesion'] * 0.15 + 
                     infp_metrics['adoption_rate'] * 0.1)
        
        estp_score = (estp_metrics['task_efficiency'] * 0.3 + 
                     estp_metrics['avg_satisfaction'] * 0.25 + 
                     estp_metrics['leader_influence'] * 0.2 + 
                     estp_metrics['team_cohesion'] * 0.15 + 
                     estp_metrics['adoption_rate'] * 0.1)
        
        comparison['overall_performance'] = {
            'infp_score': infp_score,
            'estp_score': estp_score,
            'winner': 'INFP' if infp_score > estp_score else 'ESTP',
            'score_difference': abs(infp_score - estp_score)
        }
        
        return comparison
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {filepath}")

if __name__ == "__main__":
    # 测试仿真
    config = SimulationConfig(
        team_size=3,
        max_rounds=15,
        task_complexity=0.9,
        random_seed=42
    )
    
    sim = Simulation(config)
    results = sim.run_experiment()
    
    # 打印简要结果
    print("\n=== 实验结果摘要 ===")
    comparison = results['comparison_data']
    
    print(f"总体表现: {comparison['overall_performance']['winner']} 领导者更优")
    print(f"分数差异: {comparison['overall_performance']['score_difference']:.3f}")
    
    print("\n关键指标对比:")
    for metric in ['task_efficiency', 'avg_satisfaction', 'leader_influence', 'adoption_rate']:
        comp_data = comparison[metric]
        print(f"  {metric}: INFP={comp_data['infp']:.3f}, ESTP={comp_data['estp']:.3f}, 差异={comp_data['difference']:.3f}")
    
    # 保存结果
    sim.save_results('simulation_results.json')
