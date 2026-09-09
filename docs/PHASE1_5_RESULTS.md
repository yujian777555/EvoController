# Phase 1.5 实验记录：Evolution Decision Understanding

日期：2026-09-09。计划文档：[PHASE1_5_PLAN.md](PHASE1_5_PLAN.md)。核心问题：**controller 学到的是状态依赖的进化决策，还是仅仅发现了一个全局更优的 mutation probability？**

结果数据：`results/phase1_5/results.json`、`results/analysis/action_dynamics.json`（未入 git）。

## 实验组成

1. **Action 动态分析**（Phase 1 数据，`experiments/analyze_actions.py`）：4 臂 × 5 问题 × 5 seeds 的 pm 轨迹统计、Spearman 状态相关性、Kruskal-Wallis 跨问题检验。
2. **扩展数据集**：新增 50 条全 action 空间随机轨迹（seeds 20–29，operator ∈ {polynomial, gaussian}，pm ∈ [0.25, 8]×1/n，exploration ∈ eta_m[2,50]/sigma[0.02,0.3]）。训练集合计 125 条轨迹（25 fixed + 50 random-pm + 50 random-full）。
3. **Controller v2 评估**：`mlp2`（problem-aware 特征，69 维输入）、`mlp2_nopf`（无问题特征 ablation，60 维），三头输出（operator 分类 + log pm + log exploration）。评估协议与 Phase 1 完全相同（seeds 100–104，100 代，pop 100），fixed/constant/mlp_w10 三臂复用 Phase 1 结果。

## 主要结果（final HV，mean±std，5 held-out seeds）

| problem | fixed | constant | mlp_w10 | mlp2 | mlp2_nopf |
|---|---|---|---|---|---|
| zdt1 | 0.848±.004 | 0.848±.003 | 0.851±.003 | 0.858±.002 | **0.862±.003** |
| zdt2 | 0.330±.160 | 0.473±.051 | 0.468±.059 | 0.509±.011 | **0.512±.003** |
| zdt3 | 1.266±.039 | 1.299±.005 | 1.303±.005 | 1.312±.007 | **1.313±.007** |
| zdt4 | 0.265±.190 | 0.156±.192 | 0.042±.094 | 0.410±.099 | **0.518±.245** |
| zdt6 | 0.300±.027 | 0.275±.036 | 0.279±.037 | **0.367±.078** | 0.349±.050 |

**Wilcoxon vs fixed（final HV）**：mlp2_nopf 在**全部 5 个问题** p=0.031（n=5 的最小可达 p 值，即 5/5 seeds 全胜）；mlp2 在 4/5 问题 p=0.031（zdt4 p=0.156）。

**Wilcoxon vs constant（Criterion 3 关键检验）**：mlp2_nopf 在 zdt1/2/3/4 上 p=0.031，zdt6 p=0.063；mlp2 在 zdt1/2/6 上 p=0.031，zdt3 p=0.063，zdt4 p=0.094。

## Success Criteria 逐条判定

1. **Controller actions 随 evolution state 变化 — 成立。** fixed/constant 的 run 内 pm std 严格为 0；mlp_w10 为 0.050（mlp_w1 为 0.031，历史窗口使动态幅度 ×1.6）。mlp2 还展现 operator 切换行为（每代切换率 8%–20%，zdt3 最高）。
2. **Controller 行为因问题而异 — 成立（附混淆说明）。** mlp_w10 跨问题 KW 检验 p=1.96e-291。mlp2_nopf 无显式问题特征时 mean_pm 仍从 zdt1 的 0.054 到 zdt4 的 0.115——说明 state 特征（HV/IGD 量纲）已隐式编码问题身份。**混淆**：controller 学的是绝对 pm，而问题间 base pm=1/n 不同（n=30 vs 10），跨问题差异部分是尺度效应；问题特征的加入（mlp2）未带来额外收益（见失败分析）。
3. **优于 constant 基线 — 成立。** mlp2_nopf 在 4/5 问题显著优于 constant（p=0.031），zdt6 p=0.063 边缘。Phase 1 的"收益仅来自更好常数 pm"假设被否定：**扩展 action 空间 + 状态依赖决策提供了常数无法表达的收益**（constant 只能调 pm，mlp2 还能选 operator 和 exploration）。
4. **是否支持进入序列模型（Phase 2）— 支持。** 三条独立证据：① w10 vs w1 的动态幅度与性能差异（zdt3 显著）；② mlp_w10 学到退火式策略（zdt1 上 pm 随收敛单调下降，ρ(hv)=−0.688）——这类时间结构策略正是序列模型的建模对象；③ 当前 MLP 训练损失仍高（mlp2 final 1.28），单步映射可能已饱和。

## 失败分析与诚实记录

- **Problem-aware 特征未带来收益，zdt4 上反而更差**（mlp2 0.410 vs mlp2_nopf 0.518）。假设：state 特征已隐式携带问题身份信息，9 维显式特征只增加了过拟合通道（输入 60→69 维，训练轨迹仍仅 125 条）。教训：**特征工程不能替代数据量**。
- **zdt4 修复成功但不稳定**：mlp2_nopf 在 zdt4 上 4/5 seeds 大幅超过 fixed（HV 0.58–0.66 vs 0.27），但 seed 104 仍失败（0.083）——多峰 g 函数的收敛仍依赖运气。std 0.245 为全表最大。
- **mlp2_nopf 在 zdt1/zdt3 上优于 mlp_w10 的主因可能不是"更聪明"**，而是扩展 action 空间的训练数据（zdt6 在全动作随机轨迹中 HV 达 0.49 vs fixed 0.30——该问题的搜索空间被 operator 多样性显著改善）。即：**部分收益来自数据分布而非模型结构**。这一混淆在 Phase 2 对比序列模型时需控制（同数据训练）。
- **训练损失远高于 Phase 1**（1.28 vs 0.07）属预期：全 action 空间的监督信号噪声更大（随机 operator 选择的即时奖励方差大）。

## 对 Phase 2 的输入

1. 序列模型必须与 MLP 在**同一训练分布**上对比（125 条轨迹合并集已成标准训练集）。
2. 重点检验：序列模型能否学到 MLP 无法表达的**长程退火/阶段切换策略**（mlp_w10 的 pm-收敛负相关是已知存在的时间结构）。
3. zdt4 的不稳定性建议引入**多臂评估的 mean±std 报告之外的风险指标**（如失败率 P(HV<阈值)）。
