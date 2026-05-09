# Smartphone Pricing Optimization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img src="https://img.shields.io/badge/Modeling-Mathematical%20Modeling-green" />
  <img src="https://img.shields.io/badge/Tools-Pandas%20%7C%20Scikit--learn%20%7C%20Statsmodels-orange" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

## 项目简介

本项目围绕 **智能手机产品设计优化与定价策略问题** 展开，基于用户评价数据与市场模拟销售数据，完成了用户评分影响因素分析、销量预测、定价策略比较以及产品设计方案优化。

项目主要使用 Python 进行数据处理、统计建模、机器学习建模与可视化分析，最终输出完整的建模结果、图表和论文。

---

## 项目目标

本项目主要解决以下三个问题：

### Q1 用户评分影响因素分析

基于用户对手机不同维度的评价数据，分析：

- 外观
- 屏幕
- 摄像
- 续航
- 性能
- 发热控制

等因素对总体评分的影响，并通过多元线性回归、岭回归、随机森林等方法比较不同模型的表现。

### Q2 定价策略与销量预测

基于产品评分、销售价格、竞品价格、市场规模等变量，建立销量预测模型，并进一步分析不同定价策略下的销量和利润变化。

### Q3 产品设计优化

综合用户评分模型与销量利润模型，对不同产品参数组合进行模拟，寻找利润较优的产品设计方案，为产品设计与市场定价提供决策参考。

---

## 项目结构

```text
smartphone-pricing-optimization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                  # 原始数据
│   └── processed/            # 清洗后数据
│
├── src/
│   ├── Data Profiling and Feature Analysis.py
│   ├── Q1 Rating Model.py
│   ├── Q2 Pricing Model.py
│   └── Q3 Design Optimization.py
│
├── results/
│   ├── analysis_outputs/     # 数据概况与特征分析结果
│   ├── 问题1输出结果/          # 问题一：评分影响因素分析结果
│   ├── 问题2输出结果/          # 问题二：定价与销量预测结果
│   └── 问题3输出结果/          # 问题三：产品设计优化结果
│
└── paper/
    ├── B069.pdf              # 最终论文 PDF
    ├── B069.tex              # 论文 LaTeX 源文件
    └── 承诺书B069.pdf         # 比赛承诺书
```

## 数据说明

### 用户评价数据

用户评价数据主要包含用户对智能手机不同维度的评分信息，包括总体评分以及外观、屏幕、摄像、续航、性能、发热控制等指标。

### 市场销售数据

市场销售数据主要包含产品价格、竞品价格、市场规模、销量等变量，用于建立销量预测模型和利润分析模型。

---

## 方法概述

本项目主要使用以下方法完成建模与分析：

| 模块 | 方法 |
|---|---|
| 数据预处理 | 缺失值检查、异常值检查、数据清洗 |
| 探索性分析 | 描述性统计、相关性分析、可视化分析 |
| 用户评分建模 | 多元线性回归、岭回归、随机森林 |
| 销量预测 | 回归建模、模型评估、特征影响分析 |
| 定价分析 | 利润函数构建、价格敏感性分析 |
| 设计优化 | 参数组合模拟、利润最优方案筛选 |

---

## 技术栈

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- OpenPyXL

---

## 环境配置

建议使用 Python 3.9 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容如下：

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
openpyxl
```

---

## 运行方式

在项目根目录下运行以下命令：

```bash
python "src/Data Profiling and Feature Analysis.py"
python "src/Q1 Rating Model.py"
python "src/Q2 Pricing Model.py"
python "src/Q3 Design Optimization.py"
```

由于部分 Python 文件名中包含空格，因此运行时需要使用英文双引号包裹文件路径。

运行完成后，结果将保存到 `results/` 目录中。

---

## 输出结果

项目运行后，主要结果保存在 `results/` 目录下：

```text
results/analysis_outputs/      数据概况、相关性分析与特征分析结果
results/问题1输出结果/           用户评分影响因素分析结果
results/问题2输出结果/           定价策略与销量预测结果
results/问题3输出结果/           产品设计优化结果
```

其中包括：

- 数据描述性统计结果
- 相关性热力图
- 回归模型评价指标
- 特征重要性分析图
- 销量预测结果
- 利润对比结果
- 产品设计优化方案

## 结果展示

部分结果文件位于：

```text
results/eda/
results/q1_rating_analysis/
results/q2_pricing_strategy/
results/q3_design_optimization/
```

## 论文与附件

最终论文和相关附件保存在 `paper/` 目录下：

```text
paper/B069.pdf          最终论文 PDF
paper/B069.tex          论文 LaTeX 源文件
paper/承诺书B069.pdf     比赛承诺书
```

其中，`B069.pdf` 为本项目最终提交论文。

## 项目亮点

- 完成从数据清洗、建模分析到结果解释的完整流程
- 同时使用统计模型与机器学习模型进行对比分析
- 将用户评分、销量预测与利润优化结合起来
- 输出了完整的图表、表格和论文，具备较好的可复现性
- 项目结构清晰，便于后续维护、展示与扩展

---

## 项目背景

本项目来源于数学建模实践，围绕智能手机产品设计与定价决策展开。通过对用户评价数据和市场销售数据的分析，尝试从数据角度理解用户偏好、市场响应与企业利润之间的关系。

---

## 个人工作

本人主要参与了以下工作：

- 数据读取、清洗与预处理
- 用户评分影响因素分析
- 多元线性回归、岭回归、随机森林等模型实现
- 模型评价指标计算与结果对比
- 特征重要性分析与可视化
- 部分结果解释与论文撰写支持

---

## 后续改进方向

后续可以从以下方面进一步优化：

- 引入更多真实市场数据，提高模型泛化能力
- 使用更复杂的机器学习模型进行销量预测
- 加入价格弹性分析与用户分群分析
- 将优化模型扩展为多目标优化问题
- 构建交互式可视化页面展示分析结果

---

## Author

**周祥宇**

- GitHub: [Jay1022zxy](https://github.com/Jay1022zxy)
- Research Interest: Computer Vision, OCR, Machine Learning

---
