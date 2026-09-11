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
