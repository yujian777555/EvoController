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

## Task 5 — Advantage-based Planner（结果）

评估协议：5 问题 × 20 held-out seeds（1000–1019）× 100 代 × pop 100（100 runs）。

### final HV（mean±std，20 seeds）

| problem | **advantage_planner** | Phase 2B planner | static_full_global | generation_only |
|---|---|---|---|---|
| zdt1 | **0.8683±.0004** | 0.8608 | 0.8674 | 0.8677 |
| zdt2 | **0.5343±.0005** | 0.5225 | 0.5340 | 0.5338 |
| zdt3 | **1.3247±.0008** | 1.3196 | 1.3235 | 1.3245 |
| zdt4 | 0.3125±.1558 | 0.0865 | **0.0000** | **0.4076** |
| zdt6 | 0.5021±.0006 | 0.4783 | 0.5028 | 0.5020 |

### Holm 校正后的配对检验（final HV，advantage_planner greater）

| 对比 | 显著胜 | 不显著 | 显著负 |
|---|---|---|---|
| vs fixed_nsga2 | zdt1/zdt2/zdt3/zdt6（p_holm ≈ 0） | zdt4 (p=0.077，d=+0.068) | — |
| vs static_full_global | zdt1 (0.0022)、zdt3 (0.0020)、zdt4 (≈0) | zdt2 (0.177)、zdt6 (1.0) | — |
| vs generation_only_mlp | zdt1 (0.011)、zdt2 (0.048) | zdt3 (0.61)、zdt6 (0.61) | zdt4（d=−0.095） |

**核心变化**：Phase 2B 的 planner 输给所有非因果基线；advantage planner **在 zdt1/2/3 上取得全场最优**，
且在 zdt4 上从 Phase 2B 的 0.0865 提升到 0.3125（static 基线的 ∞ 倍，因为后者在该问题 HV=0）。

**诚实的不足**：

1. **zdt4 仍不及 generation_only_mlp**（0.3125 vs 0.4076，d=−0.095），即困难问题上的优势未建立。
2. **zdt6 与两个基线基本打平**（0.5021 vs 0.5028/0.5020），无显著提升。
3. **改善幅度很小**（zdt1 +0.0009、zdt2 +0.0003、zdt3 +0.0013）——虽统计显著，但实际优化收益微弱；
   这说明在 ZDT 这类相对简单的问题上，动作选择的提升空间本身有限。
4. 验证状态仅 30 个（排序指标），样本量偏小。

## 补充分析：分问题 SNR 与排序质量（Phase 2.75C Task 3 的诊断部分）

### 1. action 效应 SNR 分问题拆解（推翻前一节的假设）

| problem | SNR h=5 | SNR h=10 | SNR h=20 | mean_hv_before |
|---|---|---|---|---|
| zdt1 | 1.20 | 2.25 | 5.13 | 0.540 |
| zdt2 | 1.96 | 3.27 | 5.75 | 0.078 |
| zdt3 | **0.62** | **0.80** | **2.78** | 0.941 |
| **zdt4** | **3.15** | **5.36** | **6.21** | 0.020 |
| zdt6 | 1.34 | 2.04 | 5.88 | 0.044 |

**更正**：本文档早前推测"zdt4 的 advantage 信号可能被噪声吞没"——**该假设被数据否定**。
zdt4 的 action 效应 SNR 是五个问题中**最高**的（h=20 时 6.21），zdt3 反而最低（2.78）。

### 2. 分问题排序质量（held-out 状态，仅备选动作，对比训练模型）

| problem | SNR(h20) | Spearman h=5 | h=10 | h=20 |
|---|---|---|---|---|
| zdt1 | 5.13 | 0.830 | 0.539 | 0.859 |
| zdt2 | 5.75 | 0.685 | 0.818 | 0.304 |
| zdt3 | 2.78 | 0.560 | 0.485 | 0.610 |
| **zdt4** | **6.21** | **0.446** | **0.497** | **0.398** |
| zdt6 | 5.88 | 0.599 | 0.510 | 0.736 |

对照 MSE-only 模型：zdt1 h=5 为 −0.168、zdt6 h=5 为 0.190 —— 对比训练在多数格点上明显更好。

### 3. 结论：zdt4 的瓶颈是"排序质量 → 优化收益"的断层

zdt4 上两个可能解释**均被排除**：

- ❌ 不是信号弱——SNR 最高（6.21）
- ❌ 不是模型不排序——Spearman 全为正（0.40–0.50）

剩下的解释是**目标错配**：advantage 定义在 20 代窗口内以"同状态候选均值"为基线，衡量的是
*相对同代备选的优越性*；而 zdt4 的最终 HV 取决于**能否跳出多峰局部最优**——这是一个
跨越远超 20 代的收敛性质。在 zdt4 快照时刻 mean_hv_before 仅 0.020（接近零 HV），
"哪个动作在 20 代内更好"与"哪条轨迹最终收敛"可能系统性脱钩。

**对 Phase 2.75C 的直接输入**：

1. Task 2（改用 default NSGA-II action 作基线）方向正确——它把 advantage 从"相对同代均值"
   改为"相对标准策略"，更贴近"决策改进" 的语义。
2. Task 3 除增加样本外，应**延长干预 horizon**（≥50–100 代）或直接以**最终 HV** 为干预目标，
   因为 20 代窗口无法表征 zdt4 的收敛行为。
3. 需要区分两种失败：*排序失败*（模型能力）与本例的*目标错配*（学习目标设计），
   后者不能靠增加模型容量解决。

## Phase 2.75C 结果（问题感知 advantage learning）

计划：[PHASE2_75C_PLAN.md](PHASE2_75C_PLAN.md)。四个任务的执行结果：

### Task 1（加入问题上下文）—— **严重损害性能** ❌

v2 数据集把输入从 64 维扩展到 76 维（+9 维问题特征 +3 维运行时上下文）。
在**同一协议、同一状态集**下对比 v1（64 维）与 v2（76 维）模型：

| problem | v1 contrastive（ρ h=5/10/20） | v2 all_problems（ρ h=5/10/20） |
|---|---|---|
| zdt1 | +0.730 / +0.682 / +0.799 | (交叉训练) +0.210 / +0.316 / +0.198 |
| zdt2 | +0.651 / +0.572 / +0.689 | — |
| zdt3 | +0.575 / +0.619 / +0.718 | — |
| **zdt4** | **+0.673 / +0.739 / +0.650** | **+0.064 / +0.248 / +0.388** |
| **zdt6** | **+0.759 / +0.701 / +0.824** | **+0.209 / +0.190 / +0.108** |

**结论：加入问题上下文使困难问题的排序质量下降 2–7 倍。** 最可能的机制是问题特征提供了
"辨识问题"的捷径（state 特征已隐式编码问题身份——Phase 1.5 已发现过同一现象），
模型把容量花在识别问题而非学习动作效果上。**Phase 1.5 的教训在此复现：特征工程不能替代
正确的学习目标。**

### Task 2（改进 advantage 定义）—— **未达成** ⚠️

- `default_action`（以 NSGA-II 默认动作为基线）：**150/150 个状态全部回退到 `controller` 基线**
  ——因为默认动作（polynomial / mult 1.0 / eta_m 20）从未出现在干预候选集中
  （候选 = 1 个 controller 动作 + 10 个随机采样动作）。要真正实现该定义，必须在**采集阶段**
  把默认动作加入候选集，属于协议改动。
- `final_hv`（以最终代 HV 为目标的建议方案）：当前实现是 h=20 的代理并复制到三列，
  构造上冗余；真正的实现需要采集更长的 horizon（Task 3）。
- 可用的量化证据：`default_action` 基线的目标离散度比 `state_mean` 大 40–45%
  （h=20：0.0964 vs 0.0682），说明该方向在**量纲上**确实更有信息量。

### Task 4（跨问题泛化）—— **弱** ❌

在 zdt1/2/3 上训练、在 zdt4/zdt6 的**全部 30 个状态**上评估：
ρ 仅 −0.08 ~ +0.25（各 horizon），oracle hit ≤ 0.10。对照组（在全部问题训练）在 zdt4 上
ρ 达 +0.39。**结论：advantage 学习目前不具备跨问题泛化能力。**

### Task 3（困难问题干预）—— 成本评估与更优替代方案

长 horizon 采集（horizons 20/50/100）实测成本：**单问题 ≈ 13 h**（99,000 代 × 0.47 s/代），
zdt4+zdt6 双分片并行 ≈ 13 h wall，全 5 问题 ≈ 65 h。

**更优替代方案（已发现）**：快照库已有 **200 状态/问题**，而现有干预数据只用了 **30 个**。
因此**无需新采集即可获得 6.7 倍状态量**——只需用现有快照以 `--max-states 100` 重跑
`evaluate-horizon`（同样的 5/10/20 horizon），成本 ≈ 8.7 h/问题，且直接缓解 Task 4 的
样本不足（当前测试问题只有 16 个评测状态）。

### Phase 2.75C 阶段性结论

| Task | 结果 |
|---|---|
| 1 问题上下文 | ❌ 严重损害（ρ 下降 2–7 倍） |
| 2 advantage 定义改进 | ⚠️ 未达成（基线回退；需协议改动） |
| 3 困难问题干预 | ⏸ 成本 13–26 h；已有更廉价的 6.7× 扩样方案 |
| 4 跨问题泛化 | ❌ 弱（ρ ≤ 0.25） |

**Phase 3（memory/SSM）gate 未通过**——计划书要求的三条（可靠排序、困难景观超过 schedule、
跨问题泛化）中，第 1 条在**本问题内**成立（v1 模型 ρ 0.57–0.82）但第 3 条明确失败，
且新增特征反而有害。

**建议的下一步**（按性价比排序）：

1. **放弃问题上下文特征，回到 v1 的 64 维输入**（本阶段最明确的可执行结论）
2. **用现有 200 快照扩样到 100 状态/问题**（≈8.7 h/问题），先把样本量问题解决
3. **重定义 advantage 目标**：把 NSGA-II 默认动作加入候选集（协议改动），
   或以**最终 HV** 为目标并采集足够长的 horizon
4. **暂不进入 Phase 3**：跨问题泛化失败说明当前表述（per-problem 状态 + 动作特征）
   还没有学到"通用进化原理"；这是表述问题而非容量问题



## Phase 2.75D 结果（stabilization：harvest 排查 + compact 表示 + 目标对比）

计划：[PHASE2_75D_PLAN.md](PHASE2_75D_PLAN.md)。

### Task 1：harvest 性能瓶颈排查（澄清了一个误判）

**结论：harvest 不慢，也不需要缓存。**

代码审查（`experiments/counterfactual_actions.py:225` `harvest_snapshots`）确认：
**不做重复 NSGA-II 重算**——每个 (problem, seed) 只跑**一次** 100 代 NSGA-II，在单次循环中
于指定代数顺手 pickle 快照。算法复杂度 O(generations)，与快照数无关。

`cProfile` 剖析（1 seed × 40 状态 = 58.4s）：

| 项 | 耗时 | 占比 |
|---|---|---|
| `harvest_snapshots` 总计 | 58.4s | 100% |
| └ `NSGAII.step()` × 100 | 57.8s | 99% |
| └└ `_fast_nondominated_sort` | 56.7s | 97% |
| └└└ `_dominates`（**988 万次调用**） | 50.7s | 87% |
| harvest 自身的 snapshot/pickle | **0.6s** | **1%** |

干净吞吐实测（无竞争进程）：**996 ms/快照** ≈ 1 个 run（40s）/ 40 快照。

| 规模 | 实测耗时 |
|---|---|
| 100 状态/问题 | 1.7 min |
| 800 状态/问题（20 runs） | 13 min |
| 全 5 问题 800 状态/问题 | ~1.1 h（串行）|

**误判溯源**：我在 2.75D 前段报"慢 10 倍 / 13 小时"，那是 harvest 与两个重型
`evaluate-horizon` 采集进程**争抢 CPU** 的结果，不是 harvest 本身的问题。

**是否需要缓存**：不需要。若要提速，唯一有效杠杆是 `_dominates` 的向量化
（988 万次 numpy 逐次调用 → 可批量化），但那是**算法级**优化、影响所有阶段
（所有评估都受同一瓶颈支配），不属于 harvest 专属改造。

### Task 2：`--feature-set {compact,context}` 与目标对比

**实现**（`experiments/build_intervention_dataset_v2.py`）：
- 新增 `--feature-set {compact,context}`，**默认 compact**（= Phase-2.75D Task 2 要求的
  回到 v1 紧凑表示；context 保留 76 维旧布局以复现 2.75C 结果）
- compact = `[state 60][action 4]` = 64 维；同时修复了 npz 写出时 `state_block` /
  `action_features` 使用固定 context 偏移的 bug（compact 下偏移会错位）
- 新增测试 `test_compact_feature_set_is_the_default_and_matches_v1_layout`；
  受影响的两个既有测试改为显式 `--feature-set context`

**目标对比（仅 compact 特征）**：A=`state_mean`、B=`default_action`、C=`future_improvement`，
同一数据集（4500 样本 / 150 状态）、同架构、同 seed、**共享 state 级切分**
（120 train / 30 val）。判别性检查（同一模型分别在训练/留出状态上）：

| target | train ρ | val ρ | train hit | val hit |
|---|---|---|---|---|
| A `state_mean` | +0.247 | **+0.035** | 0.214 | 0.111 |
| B `default_action` | +0.211 | **+0.013** | 0.217 | 0.100 |
| C `future_improvement` | +0.235 | **+0.006** | 0.183 | 0.122 |

**结论（诚实）**：

1. **三种目标在现有语料上不可区分** —— 留出 Spearman 差异（0.006–0.035）远小于噪声。
2. **全部严重过拟合** —— 训练 ρ ≈ 0.21–0.25，留出 ρ ≈ 0.01–0.04（衰减 ~7–40 倍）。
   训练侧本身也仅 ~0.25，远低于 Phase 2.75 报告的口径（其 0.58 来自另一套划分与
   30 个留出状态，且 2.75 的"全部状态"数字 0.57–0.82 **包含训练状态**，不可作为泛化证据）。
3. **瓶颈是数据量而非目标定义** —— 150 个状态、30 个留出状态（每问题 ~6 个）的统计功效
   不足以区分目标。这直接支持 Task 3 的扩样。
4. 附带约束：B（`default_action`）在旧语料上仍退化为 `controller` 基线（旧语料无
   `default` 候选）；协议修复已在 Task 3 的新采集生效。

### Task 3：pilot 干预采集（进行中）

- **快照 harvest**（zdt1/zdt2/zdt4，100 状态/问题）：已完成，306 个新快照 / **6m23s**
  （~1.25 s/快照，与 Task 1 测算一致）。现状：zdt1 854、zdt2 356、zdt4 102 可用状态。
- **干预评估**（`--max-states 100`、12 候选、3 重复、horizons 5/10/20，3 问题并行）：
  已启动。成本 ≈ 100 × 12 × 3 × 20 = 72,000 代/问题 ≈ **10 h/问题**，3 路并行 ≈ 10 h wall。
  完成后将用 compact 表示重建数据集，并在**扩大后的留出集**上重跑目标对比。

### Task 4/5 流水线预验证（干跑，pilot 部分数据 162 状态）

在 pilot 完成前，用 checkpoint 中的 162 个已完成状态（zdt1 56 / zdt2 53 / zdt4 53）
**干跑完整链路**（输出到 `results/phase2_75d/dryrun/`），确认三个工具能正确衔接：

| 步骤 | 结果 |
|---|---|
| checkpoint → 中间产物 | ✅ 3 个问题各拼成与最终产物一致的 schema |
| `build_intervention_dataset_v2 --feature-set compact` | ✅ 5832 样本 / 162 状态，**4 种基线** |
| `analyze_intervention_signal` | ✅ per-problem + pooled 信号统计 |
| `compare_advantage_targets` | ✅ 3 目标 × 3 horizon + overall |

**两个重要发现：**

1. **Target B 首次真正成立**：新采集的候选集含 `default` 候选，builder 报告的
   `rules={'exact': N}`（legacy 语料是 `{'controller': N}` 的退化情形）。
   且 `default_action` 基线的均值从 ~0 变为 **+0.011**——说明采样候选平均略优于
   NSGA-II 默认动作，该基线现在是有信息的参照点。

2. **zdt4 的 oracle_hit 是并列假象**：干跑数据显示 zdt4 在 h=5 的 oracle_hit 高达
   **0.887**，但同格点的 action_variance 仅 0.0001、SNR 0.95——即候选结果几乎全相同，
   "命中最优"由**大量 tie** 造成，不是规划能力（12 候选随机基线为 0.083）。
   **后续报告必须把 oracle_hit 与 action_variance 并读**，否则会严重高估。

干跑的目标对比（仅 162 状态、130/32 训练/验证划分，**不足以判定目标优劣**）：
overall Spearman 依次为 `state_mean` 0.107、`default_action` 0.059、
`future_improvement` 0.035。**待 pilot 完成后用完整的 300 状态重跑再下结论。**

> 注意：干跑时 `compare_advantage_targets` 的默认 `--model-dir results/phase2_75d/models`
> 覆盖了 Task 2 的模型工件（可复现，无实质损失）；正式运行将显式指定独立的
> `--model-dir` 以避免混淆。

## Gate 状态（PHASE2_75_PLAN.md）

| 标准 | 状态 |
|---|---|
| 1. Predictor 性能在 action ablation 后下降 | ✅ |
| 2. 预测排序与真实排序相关 | ✅（长 horizon Spearman 0.58，oracle hit 30%） |
| 3. 干预实验显示正 action advantage | ✅（SNR 1.85→5.68 随 horizon 增长） |
| 4. Planner 优于非因果基线 | ⚠️ **部分成立**：vs static 3/5 显著胜、vs generation_only 2/5 显著胜，但 zdt4 负 |

**判定：Phase 2.75 达成决定性改善（从"全面落后"到"多个问题显著领先"），但第 4 条未达"全部问题"的严格口径。**
是否进入 Mamba/SSM 需权衡：排序能力问题已解决（gate 1–3 全过），剩余瓶颈是**收益幅度微弱**与
**困难问题（zdt4）未突破**——这两个问题都不是靠增加模型容量能解决的，而是需要更强的动作空间设计
或更好的问题特征。

**建议**：不急于上 Mamba/SSM；先在 zdt4 类困难问题上验证 advantage 信号是否被噪声吞没
（该问题 SNR 分问题拆解），再决定是否值得投入序列建模。
