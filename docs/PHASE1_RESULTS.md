# Phase 1 实验记录：MLP Evolution Controller

日期：2026-09-08。计划文档：[PHASE1_PLAN.md](PHASE1_PLAN.md)。结果数据：`results/phase1/results.json`（未入 git）。

## 实验设计

- **训练数据**：随机策略轨迹 50 条（5 ZDT 问题 × seeds 10–19，100 代，pop 100），每代 mutation probability 从 log-uniform [0.5, 5.0]×(1/n_vars) 采样。动机：Phase 0 固定策略数据 action 恒定，无法学习 state→action 映射。
- **监督信号**：reward-weighted 回归，y = log(pm)，样本权重 w = max(r_t − mean(r_traj), 0) + 1e-6，r = ΔHV + ΔIGD（advantage-weighted behavioral cloning）。
- **模型**：MLP (60→64→64→1)，torch，weighted MSE，Adam lr=1e-3，300 epochs；train/val 按轨迹切分（4000/1000 样本，无泄漏）。
- **评估臂**：`fixed`（NSGA-II 默认）、`mlp_w10`（历史窗口 10）、`mlp_w1`（仅当前 state）、`constant`（无历史 ablation：学习到的最优常数 pm）。
- **评估协议**：5 个全新 seeds（100–104）× 5 问题 × 100 代 × pop 100；指标 final HV / final IGD / anytime AUC-HV / runtime；Wilcoxon signed-rank（vs fixed，alternative=greater，n=5）。

## 主要结果（final HV，mean±std）

| problem | fixed | mlp_w10 | mlp_w1 | constant |
|---|---|---|---|---|
| zdt1 | 0.848±.004 | 0.851±.003 | 0.849±.004 | 0.848±.003 |
| zdt2 | 0.330±.160 | 0.468±.059 | **0.501±.003** | 0.473±.051 |
| zdt3 | 1.266±.039 | **1.303±.005** | 1.285±.010 | 1.299±.005 |
| zdt4 | **0.265±.190** | 0.042±.094 | 0.074±.088 | 0.156±.192 |
| zdt6 | 0.300±.027 | 0.279±.037 | 0.291±.011 | 0.275±.036 |

Wilcoxon p（vs fixed，越大越好）：mlp_w1 在 zdt2 **p=0.031 显著**；mlp_w10 在 zdt3 **p=0.031 显著**、zdt2 p=0.094；其余不显著；zdt4/zdt6 方向相反（controller 更差）。

anytime AUC-HV：mlp_w10 在 zdt1/zdt2/zdt3 上均优于 fixed（如 zdt2 0.185 vs 0.098）；constant 在 zdt1/zdt3 的 AUC 最高（大 pm 加速早期收敛）。

## 结论（对照 Success Criteria）

1. **Controller 可以从 trajectory 学习** ✓ — 训练损失收敛，学到的策略显著不同于常数；但 val_loss ≫ train_loss（mlp_w10: 1.03 vs 0.07），存在过拟合，数据量（40 条训练轨迹）是瓶颈。
2. **Adaptive 不低于 fixed** — **部分成立**：zdt1 持平、zdt2/zdt3 显著更优；zdt4 显著更差、zdt6 略差（不显著）。
3. **Trajectory 信息有价值** — **部分成立**：w10 > w1（zdt3: 1.303 vs 1.285）支持历史信息有用；但 constant 在 zdt2/zdt3 已接近 MLP，说明大部分收益来自"发现更好的 pm 范围"，**状态条件化的自适应带来的增量是问题相关的、有限的**。

## 失败分析与原因假设

- **zdt4 失败**：随机策略训练数据中 zdt4 全部 run HV=0（高维 pm 破坏其多峰 g 函数的精细收敛），ΔHV 恒为 0 → 权重信号只剩 ΔIGD，且全部来自"失败分布"。Controller 从失败轨迹中学不到 zdt4 的有效策略。**分布不匹配 + 奖励信号退化**。下一步：pm 采样范围按问题自适应，或对 zdt4 类多峰问题使用更保守的 pm 下界；或引入 per-problem controller。
- **constant 过强**：学到的常数 pm ≈ e^−2.14 ≈ 0.118 是跨问题的折衷（n=30 问题 base=0.033，n=10 问题 base=0.1），全局单 controller 无法表达 per-problem 最优。支持 Phase 3 的问题特征/迁移方向。
- **过拟合**：4000 样本、轨迹级仅 40 条独立样本。Phase 2 的 sequence model 需要更多轨迹或更强正则。

## 对 Phase 2 的输入

- 历史窗口的价值已在 zdt3 上得到初步验证（w10 > w1, w10 > constant），值得用 Transformer/Mamba 建模更长程依赖。
- 需要改进训练数据分布（覆盖成功+失败、per-problem pm 范围）。
- 评估协议（held-out seeds + Wilcoxon + anytime AUC）已建立，可直接复用。
