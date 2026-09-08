# Phase 2 Plan: Trajectory-aware Evolution Controller

> 本计划由 Executor 依据 AGENTS.md 路线图（Phase 2: Transformer / Mamba / SSM）与 docs/PHASE1_RESULTS.md 的结论起草，Planner 可覆盖修订。

## Objective

验证 Phase 1 留下的核心问题：

> **长程 evolution history 是否包含超出短窗口（≤10 代）的增量信息？**

Phase 1 结果：窗口 10 的 MLP 在 zdt3 上优于窗口 1（1.303 vs 1.285），提示历史有价值；但 constant 基线接近 MLP，说明逐代自适应的增量有限。Phase 2 用序列模型建模完整轨迹历史（max_len=32，覆盖前 1/3 代程），检验更长的依赖能否带来增量收益。

## 范围

实现：

- `controller/sequence_encoder.py` — 变长历史 → 定长 padded 序列 (max_len, 6) + mask
- `controller/sequence_controller.py` — `TransformerController`（torch nn.TransformerEncoder）+ `GRUController`（序列基线）
- `experiments/run_phase2.py` — 训练 + 评估 + 与 Phase 1 结果合并统计

不实现：Mamba（mamba-ssm 需 torch≥2.x，本机 torch 1.10.1，不引入新依赖；记为已知限制）、RL、operator selection、LLM。

## 相对 Phase 1 的三处改进（均有 Phase 1 证据支持）

1. **训练数据分布**：合并 fixed（25 条，zdt4 上成功）+ random（50 条）共 75 条轨迹。Phase 1 中 zdt4 训练数据全部为失败轨迹（HV=0），是本阶段必须修复的分布缺陷。
2. **变长历史**：序列模型 + padding/mask 替代固定窗口截断，消除"早期代数不足窗口"的信息浪费。
3. **评估复用**：fixed / mlp_w10 / constant 三臂直接采用 `results/phase1/` 中同协议（seeds 100–104, 100 代, pop 100）的已有结果，仅新跑 transformer / gru 两臂。

## 实验设计

- 训练：advantage-weighted 回归（同 Phase 1，w = max(r − mean(r_traj), 0) + 1e-6，y = log pm），train/val 按轨迹切分，torch CPU，固定 seed。
- 臂：`transformer_s32`、`gru_s32`（新）；`fixed`、`mlp_w10`、`constant`（复用 Phase 1）。
- 协议：5 问题 × seeds 100–104 × 100 代 × pop 100；pm 界限同 Phase 1（[0.5, 5]×1/n）。
- 指标：final HV / final IGD / anytime AUC-HV / runtime；Wilcoxon signed-rank（vs fixed 及 vs mlp_w10，alternative=greater）。

## Ablation

- 历史长度：max_len ∈ {8, 32}（短 vs 长，检验长程依赖价值）
- 模型类型：Transformer vs GRU（架构敏感性）

## Success Criteria

1. 序列控制器 ≥ fixed baseline（5 问题上不显著更差）
2. 序列控制器 ≥ mlp_w10（至少 1 个问题显著更优 → 支持"长程历史有增量信息"）
3. 若 2 不成立，按 Scientific Integrity 分析：是历史无增量、数据不足、还是训练目标问题

## Reproducibility

同 Phase 1：全部 config / seed / 训练参数 / per-seed 原始值写入 `results/phase2/results.json`；评估轨迹存 `results/phase2/trajectories/`。
