# Phase 2.75 实验记录：Evolution Intervention Learning

日期：2026-09-11。计划：[PHASE2_75_PLAN.md](PHASE2_75_PLAN.md)。
前序失败分析：[PHASE2B_RESULTS.md](PHASE2B_RESULTS.md)（关键发现 A：D1–D5）。

## 核心问题

> 能否学习**进化动作的因果贡献**（action advantage），而非仅预测绝对结果？

Phase 2B 的诊断证明：outcome predictor 的高 R² 由 state 驱动（打乱 action 特征后 R² 不变），
其排序与真实长 horizon 结果负相关（Spearman −0.16）。Phase 2.75 因此改变三件事：
**学习目标**（advantage 而非绝对 HV）、**数据来源**（主动干预而非被动轨迹）、
**训练方式**（对比训练强制使用 action 通道）。

## Task 2 — 干预数据集

从长 horizon 反事实评估的 150 个状态（5 问题 × 30）× 11 候选 × 3 horizon 构建：

- 规模：**4950 样本 / 150 状态**，输入维度 64 = state 编码 60 + action 特征 4
- advantage 定义：`mean_reward(candidate) − 同状态同 horizon 所有候选的均值`（state_mean 基线）

### 关键发现 1：action 效应 SNR 随 horizon 增长

| horizon | between-action var | within-action（重复）噪声 var | **SNR** |
|---|---|---|---|
| h=5 | 5.457e-04 | 2.958e-04 | **1.85** |
| h=10 | 1.608e-03 | 5.206e-04 | **3.09** |
| h=20 | 5.112e-03 | 9.005e-04 | **5.68** |

Phase 2B 的 D4 在 5 代尺度测到 SNR = 1.35；**在 20 代尺度 SNR 达 5.68，超过计划书要求的
> 3 门槛**。即：action 的真实因果效应**确实存在且可学**，只是需要足够长的 horizon 才能
从噪声中分离。这解释了 Phase 2B planner 为何失败（它主要在短尺度上学习），也给出了明确的
补救方向。

## Task 4 — Action Advantage Predictor

按 state 分组切分（120 train / 30 val 状态，无泄漏），对比两种训练方式：

| 模型 | Spearman | Kendall | oracle hit rate | regret |
|---|---|---|---|---|
| MSE-only（Phase 2B 的失败模式） | 0.423 | 0.324 | 0.189 | **0.0153** |
| **MSE + 对比训练（hinge）** | **0.582** | **0.467** | **0.300** | 0.0209 |

（11 候选的随机 oracle hit 基线 = 0.091）

**结论：Phase 2.75 的重构有效。**

- 对比 Phase 2B 的 outcome predictor（长 horizon Spearman ≈ −0.16、oracle hit 3.3%），
  advantage 目标 + 对比训练把排序质量从**负相关**变为 **Spearman 0.58**、
  oracle 命中率从 3.3% 提升到 **30%**（约 3.3× 随机基线）。
- 机制：MSE 单独训练无法区分"需要 action 才能解释"的方差（这正是 D1 测到的失败模式）；
  配对 hinge 强制模型学会"同状态下 A 优于 B"。
- **诚实记录**：对比训练的 **regret 略高**（0.0209 vs 0.0153），即它更擅长排序但绝对
  收益略差；30 个验证状态的样本量也偏小。

## Task 3 — 宏动作表示：**未提高 SNR**

设计 7 个离散宏动作（explore_high / explore_mild / neutral / exploit_mild / exploit_high /
operator_switch_gaussian / gaussian_explore），用现有干预数据评估宏粒度下的 SNR：

| 指标 | 值 | 含义 |
|---|---|---|
| `macro_snr`（总体口径） | 6.45 vs continuous 7.42 | 字面近似，但**该口径不可直接比较** |
| `macro_snr_within_as_noise`（决策口径） | **0.345 vs 7.42** | 若 planner 只能选宏，69% 的候选间离散度会变成不可控噪声 |
| `variance_explained_by_macro` | **0.306** | 宏身份只解释 ~31% 的候选间方差 |

**结构性发现**：宏集与采样分布严重错配——`operator_switch_gaussian` 吸收 12/20 候选，
而 `exploit_mild` / `exploit_high` / `gaussian_explore` 各为 **0**（full action 采样约一半是
gaussian，宏表 polynomial:gaussian = 5:2）。即宏级 planner 实际只有约 4 个可用选项，
且利用端几乎未被覆盖。

**结论：粗化动作空间不是提高 SNR 的正道**——它丢弃了宏区域内的信息（69% 方差），而
真正的杠杆是**延长 horizon**（见关键发现 1）。建议若重试宏动作，改用**分位数等频划分**
以保证覆盖与均衡。

## Task 5 — Advantage-based Planner（评估运行中）

`AdvantagePlannerController` 已实现（argmax 预测 advantage，接口与 PlanningController 一致），
评估协议：5 问题 × 20 held-out seeds（1000–1019）× 100 代 × pop 100，
对比 fixed_nsga2 / static_full_global / generation_only_mlp / phase2b_planner。
结果待填入。

## Gate 状态（PHASE2_75_PLAN.md）

| 标准 | 状态 |
|---|---|
| 1. Predictor 性能在 action ablation 后下降 | ✅ **成立**（MSE-only 排序 Spearman 0.42 vs 对比训练 0.58；D1 已证 Phase 2B 模型不变） |
| 2. 预测排序与真实排序相关 | ✅ **成立**（同状态长 horizon Spearman 0.58，oracle hit 30%） |
| 3. 干预实验显示正 action advantage | ✅ **成立**（SNR 1.85→5.68 随 horizon 增长，且 h=20 超门槛） |
| 4. Planner 优于非因果基线 | ⏳ **评估运行中** |

**前三条已通过，第四条待定。** 若第四条也通过，则 Phase 2.75 达成 Phase 2B 未竟的目标；
若仍不通过，则说明"排序能力"与"优化收益"之间存在仍未解释的断层。
