# Phase 1.75 实验记录：Decision Causality & Fair Baselines

日期：2026-09-10。计划文档：[PHASE1_75_PLAN.md](PHASE1_75_PLAN.md)。

**核心问题**：EvoController 的收益来自**闭环状态依赖决策**，还是仅仅来自更强的 action 空间 / 固定 schedule / 尺度泄漏 / 测试集运气？

**答案：主要不是闭环反馈。** Phase-2 gate 未通过。

---

## 实验设计

- **训练数据**：500 条全 action 空间随机策略轨迹（5 ZDT 问题 × seeds 200–299），含 operator/pm/exploration 三维随机化
- **评估**：9 臂 × 5 问题 × 20 held-out seeds（1000–1019），完全配对
- **统计**：mean±std、paired Wilcoxon、95% bootstrap CI、Holm 校正（跨 5 问题）
- **失败阈值**：训练分布下 10th percentile（zdt1=0.841, zdt2=0.248, zdt3=1.234, zdt4=0.203, zdt6=0.237）
- **反事实实验**：250 个状态（50/问题）× 11 候选动作 × 3 重复，从相同快照分支评估 controller 动作的 percentile rank

## 主结果（final HV，mean±std，20 seeds）

| arm | zdt1 | zdt2 | zdt3 | zdt4 | zdt6 |
|---|---|---|---|---|---|
| fixed_nsga2 | 0.846±.006 | 0.335±.136 | 1.294±.007 | 0.245±.206 | 0.268±.031 |
| static_full_global | 0.867±.001 | 0.534±.001 | 1.324±.001 | **0.000±.000** | 0.503±.000 |
| static_full_per_problem | 0.864±.001 | 0.534±.001 | 1.324±.001 | 0.344±.234 | 0.501±.002 |
| open_loop_global | 0.864±.001 | 0.521±.026 | 1.320±.002 | 0.370±.197 | 0.497±.002 |
| open_loop_per_problem | 0.865±.001 | 0.528±.007 | 1.319±.002 | 0.089±.131 | 0.488±.006 |
| mlp2_closed_loop_absolute | 0.861±.002 | 0.529±.002 | 1.321±.002 | 0.358±.132 | 0.501±.001 |
| **mlp2_closed_loop_normalized** | 0.862±.002 | 0.529±.002 | 1.317±.003 | **0.432±.208** | 0.500±.003 |
| **generation_only_mlp** | **0.868±.001** | **0.534±.001** | **1.325±.001** | 0.408±.176 | **0.502±.001** |
| state_scrambled_mlp | 0.866±.001 | 0.530±.002 | 1.323±.002 | 0.321±.153 | 0.492±.007 |

**关键观察**：`generation_only_mlp`（只看 generation 数字，不看种群状态）在 5 个问题中的 4 个上表现最优或接近最优。

## Phase-2 Go/No-Go 判定

计划书规定的 5 条 gate：

| Gate | 结果 | 证据 |
|---|---|---|
| 1. 闭环 controller 在 ≥3/5 问题上（Holm 校正后）优于 static_full_global 和 open_loop_global | ❌ **失败** | vs static_full_global：仅 zdt4 显著胜（p_Holm=4.8e-06），zdt1/zdt3/zdt6 显著负（p=1）；vs open_loop_global：仅 zdt6 显著胜（p_Holm=0.003），其余不显著或负 |
| 2. 优于 generation_only_mlp 或 state_scrambled_mlp 在 ≥3/5 问题 | ❌ **失败** | vs generation_only_mlp：zdt1/zdt3/zdt6 全显著负（p=1）；vs state_scrambled：仅 zdt6 显著胜 |
| 3. 反事实平均 percentile rank > 0.55 且 95% CI 排除 0.50 | ❌ **失败** | mean = 0.5492（250 状态），95% CI [0.187, 0.933] 包含 0.50；zdt3 甚至为 0.458（低于随机） |
| 4. 归一化 multiplier 建模不破坏 Phase-1.5 收益 | ⚠️ **部分** | mlp2_normalized vs mlp2_absolute：zdt4 胜（0.432 vs 0.358），zdt3 负（1.317 vs 1.321），其余接近；未"破坏"但也无明确优势 |
| 5. 500 轨迹数据集缩小 train/val 泛化差距 | ⚠️ **部分** | 训练损失从 Phase 1.5 的 ~1.28 升至 ~2.03（action 空间扩大导致任务更难），val loss 仍显著高于 train loss |

**总结论：5 条 gate 中 3 条明确失败，2 条部分满足。不进入 Phase 2（Mamba/Transformer）。**

## 失败分析（Scientific Integrity）

1. **闭环反馈的因果贡献微弱**：
   - 反事实实验显示 controller 选择的动作在相同状态下仅比随机候选略好（mean rank 0.549 vs 随机期望 0.5）
   - zdt3 上甚至低于随机（0.458）
   - 这解释了为什么 `generation_only_mlp`（无状态反馈）能达到同等甚至更好的性能

2. **Phase 1.5 的"收益"主要来自**：
   - **扩展 action 空间**（operator + exploration strength）：`static_full_global` 在 zdt1/2/3/6 上已接近最优
   - **数据规模**：500 条轨迹让 generation_only_mlp 也能学到有效的 schedule
   - **而非**状态依赖的闭环决策

3. **zdt4 是唯一例外**：`mlp2_closed_loop_normalized` 在 zdt4 上显著优于 static_full_global（0.432 vs 0.0，p_Holm=4.8e-06）——多峰 g 函数问题上，静态全局策略完全失败（HV=0），而闭环 controller 能通过状态反馈避免灾难。这提示**闭环反馈的价值是问题相关的**：在简单问题上无用，在困难/多峰问题上关键。

4. **失败率对比**（overall）：
   - generation_only_mlp: 1%（最稳定）
   - mlp2_closed_loop_normalized: 3%
   - fixed_nsga2: 21%
   - static_full_global: 20%（zdt4 全失败）
   - 学习到的策略（无论是否闭环）都比固定策略稳定得多

## 对 Phase 2 的建议（不执行，仅记录）

如果未来重启 Phase 2，需要解决的核心问题：

1. **状态表示不足**：当前 state 只有 HV/IGD/diversity 三个标量，可能不足以支撑闭环决策。需要更丰富的状态（如种群分布特征、历史梯度、局部搜索进度）。
2. **奖励信号稀疏**：单步 delta-HV/delta-IGD 的因果链太短，controller 难以学到"当前动作 → 多代后的收敛"的 long-horizon credit assignment。
3. **问题难度分层**：闭环价值在 zdt4 显现。建议 Phase 2 聚焦**困难问题子集**（多峰、高维、约束），而非平均性能。
4. **替代方向**：如果目标是"学习如何进化"，可以考虑：(a) 直接学习 generation-only 的最优 schedule（已被证明有效）；(b) 用 RL 而非模仿学习，让 controller 通过试错发现闭环策略（当前是 behavior cloning，只能复现训练数据中的模式）。

## 计算成本记录

- 语料生成：500 轨迹 × 45s ≈ 6.2h（4 路并行 ~1.6h）
- 静态调优：128 候选 × 5 问题 × 2 seeds × 60 代 ≈ 2.8h（4 分片并行）
- 反事实评估：250 状态 × 11 候选 × 3 重复 ≈ 1.3h（3 路并行）
- 主评估：900 runs × 45s ≈ 11.3h（4 路并行 ~2.8h）
- 训练：4 个 MLP 臂 × 500 轨迹 ≈ 10min
- **总计**：~22h CPU 时间，~6h wall time（并行化后）

## 数据可用性

- 全部原始结果：`results/phase1_75/results.json`（900 runs，含 per-seed 原始值）
- 统计报告：`results/phase1_75/stats.json` + `stats.md`
- 反事实数据：`results/phase1_75/counterfactual_*.json`（250 状态 × 11 候选的完整分支奖励）
- 调优 artifact：`results/phase1_75/static_full_tuning.json`（128 候选的完整排名）
- 训练语料：`results/trajectory_phase1_75/`（500 轨迹，含归一化 multiplier）
