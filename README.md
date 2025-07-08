# 多主体建模实验系统

## 项目概述

这是一个基于多主体建模（Agent-Based Modeling, ABM）的群体协作仿真系统，用于对比分析 INFP 和 ESTP 两种不同人格类型的领导者在群体协作中的表现差异。

**🎯 研究目标**: 通过大规模多主体仿真实验，揭示不同人格类型领导者在不同团队规模、任务复杂度和团队组成情况下的领导效能差异，为组织管理和人才配置提供科学依据。

**📊 实验规模**: 已完成12个集群实验和24个移植性实验，涵盖多种团队规模（3-25人）和任务复杂度（1.0-4.5）配置。

## 系统特点

### 🎯 核心功能
- **多主体建模**: 将群体系统建模为多个智能体（Agent）的复杂系统
- **人格类型建模**: 基于 MBTI 人格理论，精确建模 INFP 和 ESTP 领导者的行为特征
- **动态仿真**: 多轮次交互仿真，包含决策、沟通、冲突、创新等子系统
- **全面评估**: 从任务效率、团队满意度、创新产出等多个维度进行评估
- **可视化分析**: 生成丰富的图表和详细的分析报告
- **大规模实验**: 支持批量实验和跨场景对比分析

### 📊 评估指标
- **任务推进效率**: 任务完成轮数、返工比例
- **群体满意度**: 成员情绪得分趋势
- **冲突事件数量**: 冲突频率、收敛时间
- **创意数量与质量**: 创新提案数量、采纳率
- **领导者影响力**: 信任值、中心性演变
- **团队凝聚力**: 成员间协作程度

### 🔬 实验类型
- **集群实验（Group）**: 混合人格团队中的领导效能对比
- **移植性实验（Portability）**: 不同团队组成对领导效果的影响
- **适配性分析**: 领导者与团队规模、任务复杂度的匹配研究

## 系统架构

```
项目结构:
├── BNU/                          # 实验数据和分析
│   ├── group/                   # 集群实验数据（29个实验）
│   │   ├── simulation_results_*  # 原始实验数据
│   │   └── analysis_output_*     # 分析结果
│   ├── portability/             # 移植性实验数据（55个实验）
│   │   ├── simulation_results_*  # 原始实验数据
│   │   └── analysis_output_*     # 分析结果
│   ├── leadership_patterns_analysis/ # 领导力模式分析
│   │   ├── team_size_effect.png    # 团队规模效应图
│   │   ├── task_complexity_effect.png # 任务复杂度效应图
│   │   ├── team_composition_effect.png # 团队组成效应图
│   │   ├── performance_landscape.png # 表现景观图
│   │   └── leadership_recommendations.txt # 领导力建议
│   ├── leadership_pattern_analysis.py # 模式分析脚本
│   └── 综合分析报告.md           # 完整分析报告
├── agents.py                    # 智能体类定义（Agent、LeaderAgent）
├── simulation.py                # 仿真引擎和群体动力学模型
├── analysis.py                  # 结果分析和可视化
├── main.py                      # 主程序入口
├── demo.py                      # 演示脚本
├── requirements.txt             # 依赖包列表
├── 集群实验.md                  # 集群实验记录
├── 移植性实验.md                # 移植性实验记录
└── README.md                    # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 查看实验结果

```bash
# 查看已完成的实验分析报告
# 集群实验结果在 BNU/group/ 目录
# 移植性实验结果在 BNU/portability/ 目录
# 综合分析报告在 BNU/综合分析报告.md
```

### 3. 运行新实验

```bash
# 运行演示脚本
python demo.py

# 交互式运行完整实验
python main.py

# 或使用快速运行模式
python main.py --quick-run
```

### 4. 分析现有结果

```bash
# 分析已有的实验结果文件
python main.py --analyze-only simulation_results_20240708_143022.json

# 运行综合分析脚本
cd BNU
python leadership_pattern_analysis.py
```

## 核心研究发现

### 🔍 关键实验结果

基于36个实验（12个集群实验 + 24个移植性实验）的深度分析，我们发现：

#### 1. 团队规模效应
- **小团队（3-10人）**: INFP领导者胜率 **83.3%**
- **大团队（15人以上）**: ESTP领导者胜率 **100%**
- **结论**: 团队规模是影响领导效能的关键因素

#### 2. 任务复杂度效应
- **低复杂度任务**: INFP表现更优
- **高复杂度任务**: ESTP略有优势
- **最佳配置**: 
  - INFP: 5人团队 + 4.5复杂度 = 0.6105分
  - ESTP: 25人团队 + 4.0复杂度 = 0.6200分

#### 3. 团队组成影响
- 避免全INFP团队配置
- 混合团队通常有利于ESTP领导
- 少数INFP成员(20%)对ESTP领导最有利

### 📊 适配性矩阵

| 团队规模 | 低复杂度(1-2) | 中复杂度(2-3) | 高复杂度(3-4) | 极高复杂度(4+) |
|----------|---------------|---------------|---------------|----------------|
| 小团队(3-5) | **INFP优势** | **INFP优势** | **INFP优势** | **INFP最佳** |
| 中团队(6-10) | INFP优势 | INFP优势 | INFP优势 | INFP优势 |
| 大团队(11-15) | ESTP优势 | ESTP优势 | ESTP优势 | ESTP优势 |
| 超大团队(16+) | **ESTP优势** | **ESTP优势** | **ESTP最佳** | ESTP优势 |

### 🎯 应用建议

#### INFP领导者适合：
- **研发创新项目** (小团队，高复杂度)
- **咨询服务项目** (个性化，多样化)
- **教育培训项目** (长期建设，人员发展)

#### ESTP领导者适合：
- **生产运营项目** (大团队，标准化)
- **销售推广项目** (目标明确，竞争激烈)
- **项目管理项目** (多部门协调，结果导向)

## 使用说明

### 🔬 实验类型说明

#### 集群实验（Group）
- **目的**: 在混合人格团队中对比INFP和ESTP领导者的表现
- **变量**: 团队规模（3-25人）、任务复杂度（1.0-4.5）
- **数据**: 12个实验，存储在 `BNU/group/` 目录
- **发现**: INFP在小团队高复杂度任务中表现最佳

#### 移植性实验（Portability）
- **目的**: 研究不同团队组成对领导效果的影响
- **变量**: 团队人格组成（全ESTP、混合、少数INFP等）
- **数据**: 24个实验，存储在 `BNU/portability/` 目录
- **发现**: 团队组成显著影响领导效能，避免全INFP配置

### 实验配置参数

| 参数 | 说明 | 推荐值 | 实验范围 |
|------|------|--------|----------|
| `team_size` | 团队规模 | 5-10（INFP），15-25（ESTP） | 3-25 |
| `max_rounds` | 最大仿真轮次 | 30-50 | 20-100 |
| `task_complexity` | 任务复杂度 | 4.0-4.5（INFP），3.0（ESTP） | 1.0-4.5 |
| `external_pressure` | 外部压力 | 0.3 | 0.0-1.0 |
| `random_seed` | 随机种子 | 42 | 任意整数 |

### 🎯 最佳实践配置

#### INFP领导者最佳配置：
```python
optimal_infp_config = {
    'team_size': 5,
    'max_rounds': 30,
    'task_complexity': 4.5,
    'external_pressure': 0.3,
    'random_seed': 42
}
```

#### ESTP领导者最佳配置：
```python
optimal_estp_config = {
    'team_size': 25,
    'max_rounds': 40,
    'task_complexity': 4.0,
    'external_pressure': 0.3,
    'random_seed': 42
}
```

### 领导者行为建模

#### INFP 领导者特征
- **决策风格**: 基于价值判断（Fi），偏慢、重共识
- **沟通方式**: 鼓励、倾听、间接表达
- **情绪管理**: 缓冲型节点，稳定他人情绪
- **创新激发**: 鼓励想象，点子导向（Ne）
- **冲突处理**: 避冲突、倾向妥协
- **权威展现**: 共识型权威，难以施压

#### ESTP 领导者特征
- **决策风格**: 快速（Se+Ti），重效率，直接裁决
- **沟通方式**: 直接、指令性、倾向一言堂
- **情绪管理**: 不关注情绪稳定，遇事倾向冷处理
- **创新激发**: 倾向现实操作、立即执行（Se）
- **冲突处理**: 快速压制，直接斩断矛盾关系
- **权威展现**: 外显权威，强主导性

## 实验结果展示

### 📊 数据文件结构
```
BNU/
├── group/                          # 集群实验结果
│   ├── simulation_results_*.json   # 原始实验数据
│   └── analysis_output_*/          # 分析结果
│       ├── comparison_table.csv    # 指标对比表
│       ├── task_progress_comparison.png
│       ├── satisfaction_trends.png
│       └── analysis_report.md
├── portability/                    # 移植性实验结果
│   ├── simulation_results_*.json   # 原始实验数据
│   └── analysis_output_*/          # 分析结果
└── leadership_patterns_analysis/   # 综合分析结果
    ├── team_size_effect.png        # 团队规模效应图
    ├── task_complexity_effect.png  # 任务复杂度效应图
    ├── team_composition_effect.png # 团队组成效应图
    ├── performance_landscape.png   # 表现景观图
    └── leadership_recommendations.txt # 详细建议
```

### 📈 关键可视化图表

#### 1. 团队规模效应分析
- **文件**: `BNU/leadership_patterns_analysis/team_size_effect.png`
- **内容**: 不同团队规模下INFP和ESTP的表现对比、胜率分布、稳定性分析

#### 2. 任务复杂度效应分析
- **文件**: `BNU/leadership_patterns_analysis/task_complexity_effect.png`
- **内容**: 任务复杂度对领导效能的影响、最佳表现区域分析

#### 3. 团队组成效应分析
- **文件**: `BNU/leadership_patterns_analysis/team_composition_effect.png`
- **内容**: 不同团队人格组成对领导效果的影响（基于移植性实验）

#### 4. 表现景观图
- **文件**: `BNU/leadership_patterns_analysis/performance_landscape.png`
- **内容**: 团队规模-任务复杂度二维空间中的表现分布和优势区域

### 🎯 典型实验结果对比

| 实验配置 | INFP得分 | ESTP得分 | 获胜者 | 关键发现 |
|----------|----------|----------|--------|----------|
| 5人团队，4.5复杂度 | **0.6105** | 0.5852 | INFP | INFP最佳配置 |
| 25人团队，4.0复杂度 | 0.5151 | **0.6200** | ESTP | ESTP最佳配置 |
| 8人团队，3.0复杂度 | **0.5852** | 0.5794 | INFP | 中等规模INFP优势 |
| 20人团队，3.0复杂度 | 0.5365 | **0.6047** | ESTP | 大团队ESTP优势 |

### 📋 详细分析报告

完整的分析报告请查看：`BNU/综合分析报告.md`，包含：
- 执行摘要和核心发现
- 详细的统计分析结果
- 行业应用建议
- 风险提示和缓解策略
- 实施建议和最佳实践

## 扩展功能

### 自定义智能体
可以通过修改 `agents.py` 中的 `PersonalityTraits` 类来添加新的人格类型：

```python
def create_custom_personality():
    return PersonalityTraits(
        extraversion=0.6,
        sensing=0.4,
        thinking=0.7,
        judging=0.5,
        openness=0.8
    )
```




## 技术实现

### 🛠️ 核心技术栈
- **多主体建模**: 基于 Mesa 框架的 ABM 实现
- **人格建模**: 基于 MBTI 理论的行为规则引擎
- **动态交互**: 状态机驱动的多轮仿真
- **可视化**: Matplotlib/Seaborn 专业图表

### 📊 实验方法论

#### 1. 实验设计
- **对照实验**: 严格控制变量，确保结果可信
- **重复实验**: 多次重复验证结果稳定性
- **参数扫描**: 系统性探索参数空间
- **交叉验证**: 多种实验类型相互验证

#### 2. 数据收集
- **完整记录**: 每轮次记录所有智能体状态
- **多维指标**: 从效率、满意度、创新等多维度评估
- **时间序列**: 记录动态变化过程
- **关系网络**: 记录智能体间的交互关系

#### 3. 分析方法
- **统计检验**: 确保结果统计显著性
- **可视化分析**: 多种图表展示结果
- **模式识别**: 发现深层规律和趋势
- **预测建模**: 基于历史数据预测表现






