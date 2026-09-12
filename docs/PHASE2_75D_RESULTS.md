# Phase 2.75D 结果：最终干预评估（1200 状态）

日期：2026-09-12。计划：[PHASE2_75D_FINAL_RUN_PLAN.md](PHASE2_75D_FINAL_RUN_PLAN.md)。
前序：[PHASE2_75_RESULTS.md](PHASE2_75_RESULTS.md)（含 2.75/2.75C 与干跑验证）。

## 结论（先行）

**Phase 3（memory / SSM）gate 未通过。**

在 4 个问题 × 300 状态 = **1200 状态**、960/240 训练/留出划分、
compact 64 维表示、三种 advantage 目标下，**advantage predictor 的留出排序质量
不优于随机**；更根本的是，**它在训练集上也没有学会**——机制分析显示模型的
**同状态内预测几乎恒定（变化量为真实值的 0.1%）**，即**完全忽略了 action 通道**。

这不是过拟合、不是数据量不足、也不是目标定义问题，而是**优化/表述层面的失败**：
Phase 2B 的 D1 失败模式（action 通道未被使用）在 advantage 表述下**原样重现**。

## 实验设置（按计划 Task 1 冻结）

| 项 | 值 |
|---|---|
| 表示 | compact 64 维（state 60 + action 4），**不含问题上下文** |
| 问题 | zdt1 / zdt2 / zdt4 / zdt6 |
| 状态 | **300/问题，共 1200**（来自 2.75d 快照池，各池 ≥340） |
| 每状态候选 | 12 = 1 controller + **1 default（NSGA-II 默认动作）** + 10 采样 |
| 重复 | 3 个独立分支 RNG seed |
| Horizons | 5 / 10 / 20 |
| 样本 | 39,600（丢弃 controller 行后 43,200 → 39,600） |
| 划分 | **state 级**（960 训练 / 240 留出），三目标共享同一划分 |
| 模型 | MLP 64→128→128→3，Adam lr 1e-3，200 epochs，对比训练（paired hinge） |

**Target B 首次真正生效**：builder 报告 `rules={'exact': 300}`（每个状态都使用真正的
NSGA-II 默认动作作为基线），而非此前 legacy 语料上退化为 `controller` 的情形。

## Task 3：优势目标对比（240 留出状态）

| target | h=5 ρ | h=10 ρ | h=20 ρ | overall ρ | oracle hit | regret |
|---|---|---|---|---|---|---|
| A `state_mean` | +0.001 | +0.034 | −0.021 | +0.005 | 0.125 / 0.075 / 0.071 | 0.056 / 0.056 / 0.069 |
| B `default_action` | +0.004 | +0.012 | +0.019 | **+0.012** | 0.096 / 0.104 / 0.092 | 0.052 / 0.057 / 0.063 |
| C `future_improvement` | −0.021 | −0.011 | −0.041 | −0.025 | 0.087 / 0.092 / 0.100 | 0.058 / 0.061 / 0.064 |

（11 个备选候选的随机排序基线：oracle hit = 1/11 = **0.091**，top-3 = 3/11 = **0.273**）

**三种目标在统计上均与随机不可区分**，且彼此不可区分。数据量比 2.75C 增加 8×
（留出集 30 → 240 状态）后，结论没有改变——**排除了"数据量不足"这一解释**。

## 关键诊断：训练集上同样失败（排除过拟合）

| target | **TRAIN ρ** | VAL ρ | TRAIN top-3 | VAL top-3 |
|---|---|---|---|---|
| A `state_mean` | +0.013 | +0.004 | 0.284 | 0.268 |
| B `default_action` | +0.012 | +0.012 | 0.267 | 0.290 |
| C `future_improvement` | −0.004 | −0.025 | 0.278 | 0.244 |

训练集 ρ ≈ 0（而非"高训练低留出"的过拟合形态），top-3 ≈ 0.27 = 随机。

**机制验证**（state_mean @ h=20）：

| 量 | 值 |
|---|---|
| 真实 advantage 状态内 std | 0.03108 |
| 模型预测状态内 std | **0.00003** |
| 比值 | **0.001** |
| 预测总 std | 0.00150（真实 0.05093，小 34 倍） |

**模型的同状态内预测几乎恒定** → 它学到的是"每个状态一个常数"，action 特征
（4 维）对输出没有影响。这与 Phase 2B 的 D1 诊断（打乱 action 特征后 R² 不变）
**完全一致**：advantage 目标 + 对比训练**没有恢复 action 通道的使用**。

**可能的根因（供后续修订参考，本次不实施）**：
1. 目标已按状态居中（`state_mean` 下每状态均值为 0）→ **预测常数 0 即为 MSE 的局部最优**，
   模型缺乏拟合"小幅 action 偏差"（std 0.031 vs 0 基线）的梯度信号；
2. 4 维 action 特征相对 60 维 state 块太弱，优化路径上被忽略；
3. 训练目标以 MSE 为主（hinge 权重 1.0），排序信号未被直接优化。

## Task 4：信号与决策质量（1200 状态，planner 动作）

| horizon | SNR | planner 平均 rank | oracle hit | regret | oracle gap |
|---|---|---|---|---|---|
| h=5 | 1.55 | 0.462 | 0.418 | 0.0253 | 0.0243 |
| h=10 | 2.76 | 0.452 | 0.307 | 0.0505 | 0.0488 |
| h=20 | **5.09** | **0.442** | 0.176 | 0.1051 | 0.0986 |

**信号随 horizon 增强 3.3 倍（1.55→5.09），但 planner 排名反而下降（0.462→0.442）、
regret 增大 4 倍（0.025→0.105）** —— 信号存在且可测量，但**未被任何所学策略捕获**。

分问题（h=20 SNR / oracle hit）：zdt1 4.82/0.040、zdt2 6.11/0.080、zdt4 3.97/0.467、zdt6 5.45/0.117。

> **口径警告（来自干跑验证）**：zdt4 的 oracle hit 显著高于其它问题，但其同格点
> action variance 最低——高 hit 部分是**并列（tie）造成的假象**。本表已将
> oracle hit 与 action variance 并读，不再单独解读 hit。

## Task 5：困难景观分析

已运行 `diagnose_hard_landscape.py`（zdt4 / zdt6，30 状态 × 12 候选 × 3 重复 × 20 代，
含逐代 diversity / population-std / 停滞与逃逸代数）：

| 量 | zdt4 | zdt6 |
|---|---|---|
| final-HV SNR（状态内候选间/重复噪声） | 4.32 | **11.59** |
| planner 按 final HV 的平均 percentile | **0.755** | **0.612** |
| planner 中位排名 / 平均排名（12 候选中） | 2.0 / 3.7 | 4.0 / 5.27 |
| planner 命中最优比例 | 0.467 | 0.133 |
| 平均停滞代数（alternative / controller / default） | 15.9 / 16.8 / 14.9 | 10.5 / 11.0 / 11.3 |
| 逃逸成功率（50% 参考 HV，alternative） | 0.153 | 0.373 |

产物：`results/phase2_75d/hard_landscape_zdt4.json`、`hard_landscape_zdt6.json`；
`--report` 模式按计划的四个候选失败原因逐条给出数据支持的判定（不足时输出
`INDISTINGUISHABLE` 而非猜测）。

### ⚠️ 必须标注的不一致（未解决）

**两套分析对 planner 排序能力的结论相反**：

| 分析 | 状态数 | planner 平均 percentile |
|---|---|---|
| 干预评估（本文件 Task 3/4） | 300/问题（1200 总） | **0.442**（h=20，低于随机） |
| 困难景观诊断 | 30/问题 | **0.755**（zdt4）/ **0.612**（zdt6）（高于随机） |

两者都使用同一 planner 工件与 12 候选布局，差异来自：**分支 RNG 方案不同**
（干预评估用 crc32 派生的 `PCG64([branch_seed, rep])`；诊断脚本用
`PCG64([snapshot_hash_seed, candidate_index, rep])`）、**状态子集不同**（两次各取
前 30/300 个快照）、以及**聚合口径**（干预评估对 3 个重复取均值的 mean_reward，
诊断按 final HV 的重复均值）。

**该不一致尚未解决，因此"planner 排序能力"的结论在本次评估中不能视为确定**——
本文件把干预评估（更大的样本、与训练一致的协议）作为主口径，但诊断脚本的相反
信号说明：用不同 RNG 方案复现同一动作的真实收益会产生系统性差异。**后续必须
用统一的分支 RNG 方案重跑两套分析以消除这一歧义**（这是当前最优先的技术债）。

## 最终 Gate 判定（PHASE2_75D_FINAL_RUN_PLAN.md）

| Gate | 要求 | 结果 |
|---|---|---|
| **1** | advantage 预测优于随机排序基线 | ❌ **失败**：留出 ρ ≈ 0（±0.04），oracle hit 0.07–0.13 ≈ 1/11，top-3 ≈ 3/11；**训练集上同样 ≈ 0** |
| **2** | advantage planner 优于 NSGA-II / generation-only / outcome-predictor planner | ❌ **无法满足**：planner 的决策依赖 advantage 排序，而排序能力已被证明不存在（Phase 2.75 的 planner 收益来自更早的、包含训练状态的乐观评估口径） |
| **3** | 困难景观无严重退化 | ⚠️ 部分：zdt4 的失败率与 regret 均最高，且其高 oracle-hit 属并列假象 |

**结论：不进入 Phase 3（memory / SSM）。** 计划书要求的前置条件
（"advantage 可预测且可迁移"）未被满足，而**当前瓶颈是可学性/优化表述，不是序列建模能力**——
增加模型复杂度（Mamba/SSM）不会改变 action 通道未被使用这一事实。

## 对下一步的建议（不实施）

1. **优先修优化目标而非模型**：
   - 目标改为**排序损失为主**（pairwise/listwise ranking），MSE 仅作辅助；
   - 或对 advantage **按状态归一化**（除以状态内 std），消除"预测常数 0"的局部最优；
   - 或直接学习**相对优势的符号/序**而非数值。
2. **验证 action 通道是否被使用**作为每次训练的门禁指标：若同状态预测 std / 真实 std
   的比值 < 0.1，则该模型没有学到任何 action 效应，不应进入评估环节。
3. **重新审视可学性上限**：在 64 维表示下，action 对 20 代后结果的因果影响是否
   原则上可由这批特征推断（可用一个"Oracle 上界"实验界定：用真实 outcome 训练的
   分类器能达到多高的 within-state ρ）。
4. **只有在 action 效应被证明可学之后**，才讨论是否需要更长程的序列模型。

## 产物清单

| 路径 | 内容 |
|---|---|
| `results/phase2_75d/final/counterfactual_horizon_{zdt1,zdt2,zdt4,zdt6}.json` | 1200 状态的干预数据（各 300） |
| `results/phase2_75d/dataset_final/` | compact 数据集，4 种 advantage 定义 |
| `results/phase2_75d/target_comparison_final.json` | A/B/C 对比（240 留出状态） |
| `results/phase2_75d/intervention_signal_final.json` | 信号诊断（action variance / SNR / oracle 排名分布） |
| `results/phase2_75d/hard_landscape_{zdt4,zdt6}.json` | 困难景观动力学诊断 |
| `results/phase2_75d/models_final/` | 三个训练好的 advantage 模型 |
| `experiments/analyze_intervention_signal.py` | 信号分析工具（含 top-k 口径） |
