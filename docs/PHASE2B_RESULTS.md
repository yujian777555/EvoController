# Phase 2B 实验记录：Search Policy Optimization

日期：2026-09-11（进行中）。计划：[PHASE2_EXPERIMENT_B_PLAN.md](PHASE2_EXPERIMENT_B_PLAN.md)、
协议修正：[PHASE2B_STABILIZATION_PLAN.md](PHASE2B_STABILIZATION_PLAN.md)、
评审核验：[PHASE2B_REVIEW_VERIFICATION.md](PHASE2B_REVIEW_VERIFICATION.md)。

## 核心问题

> 学习到的 outcome predictor 能否**选出比固定 schedule 更好的进化动作**？

即：预测质量能否转化为决策质量。

## 协议

- **Planner**：每代从全 action 空间采样 16 个候选（operator / multiplier / exploration），
  用 Phase 2A 的 OutcomePredictor 预测各候选的 (t+1, t+5, t+10, t+20) HV，
  按 horizon 权重 (0.1, 0.2, 0.3, 0.4) 加权取 argmax
- **B1 主评估**：5 ZDT 问题 × 20 held-out seeds (1000–1019) × 100 代 × pop 100，
  与 Phase 1.75 的 9 个 arms 完全配对
- **统计**：paired Wilcoxon（单侧 greater）+ **Holm 校正**（按 (metric, 对比臂) 家族跨 5 问题）
  + 配对 bootstrap 95% CI（10000 重采样）+ 失败率
- **B2 反事实**：从相同种群快照分支，比较 planner 动作与随机候选

## B1 结果（final HV，mean±std，20 seeds；fail = 失败率）

| problem | planning | fixed_nsga2 | static_full_global | open_loop_global | generation_only | mlp2_normalized |
|---|---|---|---|---|---|---|
| zdt1 | 0.8608±.0020 (0.00) | 0.8460±.0057 (0.15) | **0.8674±.0008** (0.00) | 0.8641 (0.00) | 0.8677 (0.00) | 0.8615 (0.00) |
| zdt2 | 0.5225±.0039 (0.00) | 0.3348±.1364 (0.25) | **0.5340±.0008** (0.00) | 0.5205 (0.00) | 0.5338 (0.00) | 0.5293 (0.00) |
| zdt3 | 1.3196±.0019 (0.00) | 1.2936±.0067 (0.00) | 1.3235±.0012 (0.00) | 1.3196 (0.00) | **1.3245** (0.00) | 1.3173 (0.00) |
| zdt4 | 0.0865±.1693 (**0.85**) | 0.2450±.2055 (0.50) | 0.0000±.0000 (1.00) | 0.3699 (0.20) | 0.4076 (0.05) | **0.4316** (0.15) |
| zdt6 | 0.4783±.0188 (0.00) | 0.2682±.0311 (0.15) | **0.5028±.0003** (0.00) | 0.4974 (0.00) | 0.5020 (0.00) | 0.5002 (0.00) |

**Holm 校正后的显著性**（final HV）：

| 对比 | 结果 |
|---|---|
| planning vs fixed_nsga2 | **显著胜** zdt1/zdt2/zdt6（Holm p = 4.8e-06）、zdt3（Holm p 显著）；**zdt4 显著负**（p=0.994 方向相反） |
| planning vs static_full_global | 全部问题**显著负**（p≈1）；仅 zdt4 原始 p=0.021 → **Holm 后 0.103，不显著** |
| planning vs open_loop_global | 全部**显著负**（p≈1） |
| planning vs generation_only_mlp | 全部**显著负**（p≈1） |
| planning vs mlp2_closed_loop_normalized | 仅 zdt3 **显著胜**（Holm p = 0.027）；zdt1/zdt2 无差异；zdt4/zdt6 负 |

## 关键观察

1. **planner 优于固定策略，但不及静态最优与开环 schedule**：planner 在 4/5 问题显著优于
   fixed NSGA-II，却在**所有 5 个问题**上不及 `static_full_global` 与 `generation_only_mlp`。
2. **zdt4 是 planner 的主要失败点**：失败率 0.85（vs generation_only 的 0.05、static_full_per_problem 的 0.30）。
   计划书预期"闭环价值在困难问题上显现"——但当前 planner 在 zdt4 上反而崩溃。
   可能原因：predictor 在 zdt4 的失败分布上训练不足（Phase 1.75 语料中 zdt4 有 13% HV=0 的 run），
   且 planner 的 argmax 可能过度乐观。
3. **static_full_global 在 zdt4 上 HV=0（失败率 1.00）**：全局静态 pm 在多峰问题上失效，
   这正是 planner 本应发挥价值的场景，但 planner 未能利用。

## B2 反事实：一步 vs 长 horizon

- **一步评估（secondary diagnostic）**：mean percentile rank = 0.5492（Phase 1.75 协议，
  250 状态），95% CI 包含 0.50。
- **长 horizon 评估（primary，对齐 planner 目标）**：⏳ 运行中
  （`results/phase2b/counterfactual/counterfactual_horizon*.json`，horizons 5/10/20，
  30 状态/问题 × 11 候选 × 3 重复）。待完成后填入。

> **评审指出的关键点**：一步 rank ≈ 0.50 **不能证伪**长 horizon planning，
> 因为 planner 优化的不是一步 reward。必须以长 horizon 结果作为主判据。

## 已知限制（诚实记录）

1. **候选集种子共享**：`PlanningController` 用 `PCG64([candidate_seed, len(history)])` 播种，
   同一代在不同 run 间共享同一候选流（绝对 pm 仍随 1/n_vars 缩放）。
   这会降低跨 run 的候选多样性、可能造成决策相关。评审建议后续 robustness 实验
   把 (problem, run seed, generation) 混入种子。
2. **缺少 action-conditional 诊断**：Phase 2A 的高 R² 可能主要来自 state/history，
   而非学到了 action 的影响。任务卡 #8（action ablation / within-state ranking /
   top-1 regret / action-effect SNR）尚未交付。
3. **失败率不作为主指标**：zdt4 的 0.85 失败率表明均值被少数成功 run 主导。

## Go/No-Go 状态

成功标准（PHASE2B_STABILIZATION_PLAN.md）：

| 标准 | 状态 |
|---|---|
| 1. planner 经校正后优于 fixed/static 基线 | ❌ 仅优于 fixed，**不及 static** |
| 2. 反事实显示选中动作有正 advantage | ⏳ 长 horizon 结果待出 |
| 3. 预测排序与真实未来结果相关 | ⏳ 待出（horizon 评估的 Spearman/Kendall 字段） |
| 4. 收益不能由静态 schedule 解释 | ❌ 目前 generation_only_mlp 全面优于 planner，**静态/schedule 解释力更强** |

**当前倾向：不通过 Phase 2B 的 gate，不进入 Mamba/SSM。** 最终判定待长 horizon 反事实与
action-conditional 诊断结果。
