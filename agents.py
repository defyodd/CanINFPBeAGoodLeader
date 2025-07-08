import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PersonalityType(Enum):
    """人格类型枚举"""
    INFP = "INFP"
    ESTP = "ESTP"
    OTHER = "OTHER"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVISION = "needs_revision"

@dataclass
class PersonalityTraits:
    """人格特质数据结构"""
    extraversion: float  # 外向性 (0-1)
    sensing: float       # 感觉 (0-1)
    thinking: float      # 思考 (0-1)
    judging: float       # 判断 (0-1)
    openness: float      # 开放性 (0-1)
    
    def __post_init__(self):
        # 确保所有值在0-1范围内
        for field in ['extraversion', 'sensing', 'thinking', 'judging', 'openness']:
            setattr(self, field, max(0, min(1, getattr(self, field))))

class Agent:
    """智能体基类"""
    def __init__(self, agent_id: int, personality_type: PersonalityType, 
                 traits: PersonalityTraits, name: str = None):
        self.agent_id = agent_id
        self.personality_type = personality_type
        self.traits = traits
        self.name = name or f"Agent_{agent_id}"
        
        # 状态变量
        self.satisfaction = 0.5  # 满意度 (0-1)
        self.trust_in_leader = 0.5  # 对领导者的信任 (0-1)
        self.stress_level = 0.0  # 压力水平 (0-1)
        self.motivation = 0.7  # 动机水平 (0-1)
        self.creativity_score = 0.0  # 创造力得分
        
        # 社交网络
        self.connections = {}  # {agent_id: trust_level}
        self.communication_history = []
        
        # 任务相关
        self.task_contributions = []
        self.conflict_count = 0
        self.ideas_generated = 0
        self.ideas_adopted = 0
        
    def update_satisfaction(self, leadership_style: Dict, task_progress: float):
        """更新满意度"""
        # 基于人格特质和领导风格的匹配度
        style_match = self._calculate_style_match(leadership_style)
        progress_satisfaction = task_progress * 0.3
        
        # 更新满意度
        self.satisfaction = 0.4 * self.satisfaction + 0.6 * (style_match + progress_satisfaction)
        self.satisfaction = max(0, min(1, self.satisfaction))
    
    def _calculate_style_match(self, leadership_style: Dict) -> float:
        """计算与领导风格的匹配度"""
        # 这里可以根据具体的人格特质来计算匹配度
        base_match = 0.5
        
        # 如果领导者决策风格与成员偏好匹配
        if leadership_style.get('decision_speed', 0.5) > 0.7 and self.traits.sensing > 0.6:
            base_match += 0.2
        elif leadership_style.get('decision_speed', 0.5) < 0.3 and self.traits.sensing < 0.4:
            base_match += 0.2
            
        # 如果沟通方式匹配
        if leadership_style.get('communication_directness', 0.5) > 0.7 and self.traits.thinking > 0.6:
            base_match += 0.1
        elif leadership_style.get('communication_directness', 0.5) < 0.3 and self.traits.thinking < 0.4:
            base_match += 0.1
            
        return min(1, base_match)
    
    def generate_idea(self, task_context: Dict) -> Optional[Dict]:
        """生成创意"""
        # 基于人格特质和当前状态生成创意
        idea_probability = (self.traits.openness * 0.4 + 
                          self.motivation * 0.3 + 
                          (1 - self.stress_level) * 0.3)
        
        if random.random() < idea_probability:
            self.ideas_generated += 1
            idea_quality = random.uniform(0.3, 1.0) * (self.traits.openness + self.motivation) / 2
            
            return {
                'agent_id': self.agent_id,
                'quality': idea_quality,
                'type': 'innovation' if self.traits.openness > 0.7 else 'improvement',
                'timestamp': task_context.get('current_round', 0)
            }
        return None
    
    def respond_to_conflict(self, conflict_intensity: float) -> Dict:
        """响应冲突"""
        self.conflict_count += 1
        
        # 基于人格特质决定冲突响应
        if self.traits.thinking > 0.6:  # 思考型
            response_type = 'analytical'
            stress_increase = conflict_intensity * 0.3
        else:  # 情感型
            response_type = 'emotional'
            stress_increase = conflict_intensity * 0.5
            
        self.stress_level = min(1, self.stress_level + stress_increase)
        
        return {
            'agent_id': self.agent_id,
            'response_type': response_type,
            'stress_increase': stress_increase,
            'cooperation_willingness': max(0, 1 - self.stress_level)
        }

class LeaderAgent(Agent):
    """领导者智能体"""
    def __init__(self, agent_id: int, personality_type: PersonalityType, 
                 traits: PersonalityTraits, name: str = None):
        super().__init__(agent_id, personality_type, traits, name)
        self.leadership_style = self._define_leadership_style()
        self.influence_network = {}  # {agent_id: influence_level}
        self.decisions_made = []
        
    def _define_leadership_style(self) -> Dict:
        """定义领导风格"""
        if self.personality_type == PersonalityType.INFP:
            return {
                'decision_speed': 0.3,  # 决策速度慢
                'communication_directness': 0.2,  # 间接沟通
                'consensus_seeking': 0.9,  # 寻求共识
                'emotional_support': 0.8,  # 情感支持
                'creativity_encouragement': 0.9,  # 鼓励创造力
                'conflict_avoidance': 0.8,  # 避免冲突
                'authority_display': 0.2   # 低权威展现
            }
        elif self.personality_type == PersonalityType.ESTP:
            return {
                'decision_speed': 0.9,  # 决策速度快
                'communication_directness': 0.9,  # 直接沟通
                'consensus_seeking': 0.2,  # 不太寻求共识
                'emotional_support': 0.3,  # 低情感支持
                'creativity_encouragement': 0.4,  # 中等创造力鼓励
                'conflict_avoidance': 0.1,  # 不避免冲突
                'authority_display': 0.9   # 高权威展现
            }
        else:
            # 默认中等风格
            return {
                'decision_speed': 0.5,
                'communication_directness': 0.5,
                'consensus_seeking': 0.5,
                'emotional_support': 0.5,
                'creativity_encouragement': 0.5,
                'conflict_avoidance': 0.5,
                'authority_display': 0.5
            }
    
    def make_decision(self, task_context: Dict, member_inputs: List[Dict]) -> Dict:
        """做出决策"""
        current_round = task_context.get('current_round', 1)
        task_progress = task_context.get('task_progress', 0.0)
        
        decision_quality = 0.5
        decision_time = 1.0
        
        # 预先计算共同需要的因子
        consensus_factor = sum(input.get('agreement', 0.5) for input in member_inputs) / len(member_inputs) if member_inputs else 0.5
        avg_idea_quality = sum(input.get('idea_quality', 0.5) for input in member_inputs) / len(member_inputs) if member_inputs else 0.5
        
        if self.personality_type == PersonalityType.INFP:
            # INFP: 基于价值判断，慢决策，重共识
            decision_time = 2.0 + len(member_inputs) * 0.3
            
            # 团队和谐因子（基于压力水平）
            team_harmony = 1.0 - (self.stress_level * 0.5)
            
            # INFP决策质量：在团队和谐、有共识、有好创意时表现最佳
            decision_quality = (0.2 +  # 基础质量
                              0.4 * consensus_factor +  # 重视共识
                              0.3 * avg_idea_quality +  # 重视创意
                              0.1 * team_harmony)  # 受团队状态影响
            
            # 后期决策质量会因过度考虑而下降
            if current_round > 15:
                decision_quality *= (1.0 - (current_round - 15) * 0.02)
                
        elif self.personality_type == PersonalityType.ESTP:
            # ESTP: 快速决策，重效率，适应性强
            decision_time = 0.5 + random.uniform(0, 0.3)  # 快速但有些随机性
            
            # 效率因子（ESTP关注任务推进）
            efficiency_factor = min(1.0, 0.8 + task_progress * 0.2)
            
            # 压力应对能力（ESTP在压力下表现更好）
            stress_bonus = min(0.2, self.stress_level * 0.3)
            
            # 紧迫感（后期表现更好）
            urgency_bonus = min(0.1, (current_round / 30) * 0.1)
            
            # ESTP决策质量：在压力下、任务紧迫时表现最佳
            decision_quality = (0.4 +  # 较高基础质量
                              0.4 * efficiency_factor +  # 重视效率
                              stress_bonus +  # 压力下表现更好
                              urgency_bonus)  # 紧迫情况下更好
            
            # 但缺乏耐心，在需要深度思考时质量下降
            if consensus_factor < 0.3:  # 团队不团结时
                decision_quality *= 0.8
        
        # 确保决策质量在合理范围内
        decision_quality = max(0.1, min(1.0, decision_quality))
        
        decision = {
            'leader_id': self.agent_id,
            'decision_quality': decision_quality,
            'decision_time': decision_time,
            'consensus_level': sum(input.get('agreement', 0.5) for input in member_inputs) / len(member_inputs) if member_inputs else 0.5,
            'timestamp': current_round,
            'context_factors': {
                'round': current_round,
                'progress': task_progress,
                'team_inputs': len(member_inputs)
            }
        }
        
        self.decisions_made.append(decision)
        return decision
    
    def handle_conflict(self, conflict_parties: List[int], conflict_intensity: float) -> Dict:
        """处理冲突"""
        if self.personality_type == PersonalityType.INFP:
            # INFP: 避免冲突，寻求妥协
            resolution_time = 3.0 + conflict_intensity * 2.0
            resolution_effectiveness = 0.4 + 0.6 * (1 - conflict_intensity)
            approach = 'mediation'
            
        elif self.personality_type == PersonalityType.ESTP:
            # ESTP: 快速压制，直接解决
            resolution_time = 0.5
            resolution_effectiveness = 0.7 + 0.3 * (1 - conflict_intensity)
            approach = 'authoritative'
        else:
            resolution_time = 2.0
            resolution_effectiveness = 0.5
            approach = 'collaborative'
        
        return {
            'leader_id': self.agent_id,
            'resolution_time': resolution_time,
            'effectiveness': resolution_effectiveness,
            'approach': approach,
            'conflict_parties': conflict_parties,
            'conflict_intensity': conflict_intensity
        }
    
    def communicate_with_team(self, team_members: List[Agent], message_type: str) -> List[Dict]:
        """与团队沟通"""
        communications = []
        
        for member in team_members:
            if self.personality_type == PersonalityType.INFP:
                # INFP: 鼓励式、倾听式沟通
                communication_effectiveness = 0.6 + 0.4 * (1 - member.stress_level)
                message_tone = 'encouraging'
                
            elif self.personality_type == PersonalityType.ESTP:
                # ESTP: 直接、指令式沟通
                communication_effectiveness = 0.5 + 0.5 * member.traits.thinking
                message_tone = 'directive'
            else:
                communication_effectiveness = 0.5
                message_tone = 'neutral'
            
            communication = {
                'from': self.agent_id,
                'to': member.agent_id,
                'effectiveness': communication_effectiveness,
                'tone': message_tone,
                'type': message_type
            }
            
            communications.append(communication)
            
            # 更新成员对领导者的信任
            trust_change = (communication_effectiveness - 0.5) * 0.1
            member.trust_in_leader = max(0, min(1, member.trust_in_leader + trust_change))
        
        return communications

def create_personality_traits(personality_type: PersonalityType) -> PersonalityTraits:
    """创建人格特质"""
    if personality_type == PersonalityType.INFP:
        return PersonalityTraits(
            extraversion=0.2,  # 内向
            sensing=0.3,       # 直觉
            thinking=0.2,      # 情感
            judging=0.3,       # 知觉
            openness=0.9       # 高开放性
        )
    elif personality_type == PersonalityType.ESTP:
        return PersonalityTraits(
            extraversion=0.9,  # 外向
            sensing=0.8,       # 感觉
            thinking=0.8,      # 思考
            judging=0.2,       # 知觉
            openness=0.5       # 中等开放性
        )
    else:
        # 其他类型的随机特质
        return PersonalityTraits(
            extraversion=random.uniform(0.3, 0.7),
            sensing=random.uniform(0.3, 0.7),
            thinking=random.uniform(0.3, 0.7),
            judging=random.uniform(0.3, 0.7),
            openness=random.uniform(0.4, 0.8)
        )

def create_team(team_size: int, leader_type: PersonalityType) -> Tuple[LeaderAgent, List[Agent]]:
    """创建团队"""
    # 创建领导者
    leader_traits = create_personality_traits(leader_type)
    leader = LeaderAgent(0, leader_type, leader_traits, f"Leader_{leader_type.value}")
    
    # 创建成员
    members = []
    INFP_count = (int)(team_size*0.8)
    ESTP_count = (int)(team_size*0.2)
    for i in range (1,INFP_count+1):
        # 队员都是INFP类型
        member_traits = create_personality_traits(PersonalityType.INFP)
        member = Agent(i, PersonalityType.INFP, member_traits, f"Member_{i}")
        members.append(member)
    for i in range (ESTP_count+1, team_size+1):
        # 队员都是ESTP类型
        member_traits = create_personality_traits(PersonalityType.ESTP)
        member = Agent(i, PersonalityType.ESTP, member_traits, f"Member_{i}")
        members.append(member)
    # for i in range(1, team_size):
        # 队员都是INFP类型
        # if leader_type == PersonalityType.INFP:
        #     member_traits = create_personality_traits(PersonalityType.INFP)
        #     member = Agent(i, PersonalityType.INFP, member_traits, f"Member_{i}")
        #     members.append(member)
        # # 队员都是ESTP类型
        # elif leader_type == PersonalityType.ESTP:
        #     member_traits = create_personality_traits(PersonalityType.ESTP)
        #     member = Agent(i, PersonalityType.ESTP, member_traits, f"Member_{i}")
        #     members.append(member)
        
        # # 队员当中INFP和ESTP各占一半
        # if leader_type == PersonalityType.INFP or leader_type == PersonalityType.ESTP:
        #     if i % 2 == 0:
        #         member_traits = create_personality_traits(PersonalityType.INFP)
        #         member = Agent(i, PersonalityType.INFP, member_traits, f"Member_{i}")
        #     else:
        #         member_traits = create_personality_traits(PersonalityType.ESTP)
        #         member = Agent(i, PersonalityType.ESTP, member_traits, f"Member_{i}")
        #     members.append(member)
        

        # 队员都是其他类型
        #member_traits = create_personality_traits(PersonalityType.OTHER)
        # member = Agent(i, PersonalityType.OTHER, member_traits, f"Member_{i}")
        # members.append(member)
    
    return leader, members

if __name__ == "__main__":
    # 测试代码
    print("Agent类定义完成！")
    
    # 创建测试团队
    leader, members = create_team(6, PersonalityType.INFP)
    print(f"创建了一个包含 {len(members)} 个成员的团队，领导者类型: {leader.personality_type.value}")
    
    # 测试一些基本功能
    task_context = {'current_round': 1}
    idea = members[0].generate_idea(task_context)
    if idea:
        print(f"成员 {members[0].name} 生成了一个创意: {idea}")
    
    decision = leader.make_decision(task_context, [])
    print(f"领导者做出决策: 质量={decision['decision_quality']:.2f}, 时间={decision['decision_time']:.2f}")
