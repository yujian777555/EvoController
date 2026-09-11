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

## 关键发现 A：planner 的排序器对 action 不敏感（D1–D5 诊断）

任务卡 #8 的 action-conditional 诊断给出了 **Phase 2B 失败的根因**。

**D1 — action 特征消融**（held-out 4050 样本）：

| 变体 | R² | MSE |
|---|---|---|
| baseline | 0.99229 | 0.0014784 |
| **shuffled**（跨样本打乱 action） | 0.99228 | 0.0014803 |
| **mean**（替换为训练均值） | 0.99275 | 0.0013904 |
| zero | 0.97080 | 0.0055984 |

把整块 4 维 action 特征**打乱后性能几乎不变**（ΔR² ≈ 1e-5），换成均值甚至更好。
→ **模型实质是 `state → future HV` 回归器，action 通道基本未被使用。**

**D2 严格版 — 固定 state 下的排序质量**（20 快照 × 8 候选 × 3 分支种子 × 5 代，唯一真正受控的实验）：

| 指标 | 值 | 解读 |
|---|---|---|
| Spearman（预测 vs 真实收益） | **−0.216** | 负相关 |
| Kendall | −0.139 | 负相关 |
| 显著比例 | 0.071（1/14） | 近乎无 |
| **argmax 命中 oracle 最优** | **14.3%** | 8 候选随机基线 = 12.5% |
| 对齐 horizon（h=5 列）后的 Spearman | −0.182 | 不能用"目标 horizon 不匹配"解释 |

> D2 近似版（340 组）看似 Spearman 0.641，但**把 action 行打乱后仍为 0.639** — 该相关完全由状态驱动，
> 是 checksum 而非因果证据。严格版才是有效证据。

**D3/D4/D5 — 机制**：

- top-1 regret 均值 **0.0422 HV**（中位 0.0136，n=14）
- **action-effect SNR = 1.35**（同状态候选间方差仅为重复噪声的 1.35 倍）→ 5 代内 action 的真实影响本身很弱
- predicted-vs-actual Pearson 0.324，**斜率 0.114** → 预测幅度被压缩到真值尺度的零头

**结论**：planner 输给 static / generation-only 的根因在**排序器**——用对 action 不敏感的 outcome model
做 argmax，等价于**在噪声上取最大**。候选集不是瓶颈。

## 关键观察

1. **planner 优于固定策略，但不及静态最优与开环 schedule**：planner 在 4/5 问题显著优于
   fixed NSGA-II，却在**所有 5 个问题**上不及 `static_full_global` 与 `generation_only_mlp`。
2. **zdt4 是 planner 的主要失败点**：失败率 0.85（vs generation_only 的 0.05、static_full_per_problem 的 0.30）。
   计划书预期"闭环价值在困难问题上显现"——但当前 planner 在 zdt4 上反而崩溃。
   D4 的 SNR=1.35 提供了机制解释：困难问题上 planner 的排序更不可靠。
3. **static_full_global 在 zdt4 上 HV=0（失败率 1.00）**：全局静态 pm 在多峰问题上失效，
   这正是 planner 本应发挥价值的场景，但 planner 未能利用。

## B2 反事实：一步 vs 长 horizon（已定稿）

- **一步评估（secondary diagnostic）**：mean percentile rank = 0.5492（Phase 1.75 协议，
  250 状态），95% CI 包含 0.50。
- **长 horizon 评估（primary，对齐 planner 目标）**——5 问题 × 30 状态 = 150 状态/horizon，
  11 候选 × 3 重复，分支 20 代：

| horizon | mean percentile rank | 95% CI | oracle hit rate | mean regret |
|---|---|---|---|---|
| h=5 | **0.4672** | [0.4319, 0.5028] | — | — |
| h=10 | **0.4717** | [0.4362, 0.5079] | — | — |
| h=20 | **0.4668** | [0.4251, 0.5083] | — | — |

分问题明细（mean Spearman 预测排序 vs 真实收益，n=30 状态/问题）：

| problem | h=5 | h=10 | h=20 |
|---|---|---|---|
| zdt1 | −0.217 | −0.153 | −0.174 |
| zdt2 | +0.090 | +0.101 | +0.100 |
| zdt3 | +0.029 | +0.044 | −0.163 |
| zdt4 | 负（rank 0.33→0.25→0.12） | | |
| zdt6 | 负（rank 0.33→0.27→0.30） | | |

**结论**：三个 horizon 的 percentile rank 点估计**全部低于 0.5**，zdt1 明确低于随机
（Spearman −0.15 ~ −0.22）。**即使采用与 planner 目标完全对齐的长 horizon 口径，
其动作选择仍不优于随机** —— 外部评审"一步结果不能证否长 horizon planning"的保留意见
由此解决：**长 horizon 规划同样不成立**。

> 这一负结果的价值：它把 Phase 2B 的失败从"协议错配的假象"确证为**真实的排序能力缺失**，
> 并直接指向根因（D1–D5 的 action 通道未被学习），从而确立了 Phase 2.75 的改进方向。

## 已知限制（诚实记录）

1. **候选集种子共享**：`PlanningController` 用 `PCG64([candidate_seed, len(history)])` 播种，
   同一代在不同 run 间共享同一候选流（绝对 pm 仍随 1/n_vars 缩放）。
   这会降低跨 run 的候选多样性、可能造成决策相关。评审建议后续 robustness 实验
   把 (problem, run seed, generation) 混入种子。
2. **D1/D2-近似使用 `results/trajectory_full`，其 action 缺 `mutation_multiplier`**：
   `build_outcome_samples` 回落到 `n_vars=30`，对 ZDT4（n_vars=10）放大 3 倍。
   诊断脚本默认按 run config 还原 `pm * n_vars`；对照实验（`--no-multiplier-normalization`）
   给出 baseline R²=0.9912，**结论不变**。
3. **严格版 D2 有 6/20 状态被剔除**（各候选真实增益完全相同，多为 gen=2 早世代）；
   原始与计分状态数均写入 artifact。
4. **失败率不作为主指标**：zdt4 的 0.85 失败率表明均值被少数成功 run 主导。

## Go/No-Go 状态

成功标准（PHASE2B_STABILIZATION_PLAN.md）：

| 标准 | 状态 |
|---|---|
| 1. planner 经校正后优于 fixed/static 基线 | ❌ 仅优于 fixed，**不及 static** |
| 2. 反事实显示选中动作有正 advantage | ❌ 长 horizon rank 0.47（三个 horizon 点估计均 < 0.5，CI 含 0.5） |
| 3. 预测排序与真实未来结果相关 | ❌ 长 horizon mean Spearman = −0.16 ~ +0.10（zdt1 为负），oracle hit 0–33% |
| 4. 收益不能由静态 schedule 解释 | ❌ generation_only_mlp 全面优于 planner |

**最终判定：Phase 2B 不通过 gate，不进入 Mamba/SSM。** 核心阻塞点是**排序器而非模型容量**：
outcome predictor 的高 R² 由 state 驱动（D1），action 通道未被学习（D2/D5），5 代尺度上 action
真实影响的信噪比仅 1.35（D4）；长 horizon 评估进一步确证排序质量在随机水平。详见
[PHASE2_75_RESULTS.md](PHASE2_75_RESULTS.md) 的后续改进与结果。

**下一步建议（不实施，仅记录）**：
1. 改学习目标为 **action advantage**（相对 state-only 基线的增量），而非绝对 future HV
2. 训练时加入 **action-shuffled 负样本**做对比学习，强制模型使用 action 通道
3. 先提升 D4 的 **SNR > 3**（更长 horizon、更强算子范围、降低重复噪声）再谈学习排序
4. 若以上均无效，则 Phase 1.75 的结论（闭环价值有限）得到独立二次确认，
   研究方向应转向 generation-only schedule 的最优化或问题自适应策略选择
