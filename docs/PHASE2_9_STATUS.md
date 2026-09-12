# Phase 2.9 状态交接

更新时间：2026-09-12 深夜（上下文压缩前写入）。

## 当前正在运行

```bash
python -u experiments/action_representation_redesign.py
# 输出: results/phase2_9/action_representation.json
# 日志: results/phase2_9_run.log
# 规模: 960 训练 / 240 验证状态，3 模型（OLS/RF/HGB）× 3 horizon × 5 变体，200 bootstrap
```

预期耗时：冒烟（60/30 状态、1 模型、1 horizon）为 65s → 全量约 20–60 分钟。
完成后需：读取 artifact → 按 gate 判定 → 写 `docs/PHASE2_9_RESULTS.md` → commit。

## 已完成并推送（`e82fed5`）

| 项 | 内容 |
|---|---|
| Phase 2.9 Task 1 实现 | `experiments/action_representation_redesign.py`（1148 行）+ 19 测试 |
| 动作签名 | 桶 = operator(2) × multiplier 四分位 × exploration 四分位 = **32 桶**，**按算子分别拟合分位边界**（poly/gauss 的 exploration 边界差 3 个数量级）；6 维特征全部 **leave-one-state-out** |
| 5 个评估变体 | `action_only` / `action_only+signature` / `concat` / `concat+signature` / `signature_only` |
| 分箱粒度依据 | 实测桶内行数：4×4 最小 50、中位 303；8×8 出现空桶 → 选 4×4 |
| 诊断脚本 RNG 统一 | `diagnose_hard_landscape.py` 改用与 `evaluate-horizon` 相同的 `_branch_seed` 推导 + 对应测试 |

## 关键科学结论（截至本状态）

1. **Phase 2.75D**：1200 状态下 advantage 模型**忽略 action 通道**（同状态内预测 std / 真实 std = **0.001**），Phase 3 gate 失败。
2. **Phase 2.8 Task 1（Oracle）**：树模型的 action sensitivity 是 MLP 的 **200 倍**（0.21 vs 0.001），但排序仍与随机无异。
3. **Phase 2.8 Task 2/2.1（可辨识性审计）**：
   - SNR 不弱：h=20 池化 **4.82**，四问题 **3.79–6.21**，89% 状态 > 1
   - `state_only` **在构造上无法做 within-state 排序**（预测状态内恒定）
   - `action_only` ≈ `concat`（ρ 0.059 vs 0.060）→ 状态块零增益
   - pairwise AUC **0.498–0.525**、win-count Kendall 0.14–0.17
   - 决策门：**Case B**（信号存在，瓶颈在表示），但已论证 Case B 的三条路径中只有"表示重设计"能移动上界
4. **新洞察（外部模型发现，值得验证）**：同状态内配对时**状态特征逐位相减恒为 0**，故 `concat` 与 `action_only` 的 pairwise AUC **逐位相同**（0.5026817020034349）。这说明**成对偏好任务在构造上无法利用状态上下文**——若要在 pairwise 框架下检验状态条件化，必须改为跨状态回归或加入状态交互项。
5. **未解决的矛盾**：统一分支 RNG 后，诊断脚本的 planner percentile **0.721**（30 状态）与干预评估 **0.467**（300 状态）仍不一致。已排除：RNG 方案、候选构成（两者都用 `_build_candidate_actions`，12 候选）。**剩余怀疑点：状态子集选择与聚合口径**（诊断按 final HV 的重复均值排名；审计按 `mean_reward` 排名——理论上同状态内等价，但未验证）。

## 下一步（压缩后）

1. 读 `results/phase2_9/action_representation.json`，按 gate 判定：
   - AUC 显著 > 0.55 且 Kendall > 0.25 ⇒ 表示是瓶颈，继续 Task 2/3
   - AUC 仍 0.50–0.53 ⇒ failure case，论文转向"evolutionary action credit assignment 的根本困难"
2. 写 `docs/PHASE2_9_RESULTS.md` + commit
3. 可选：解决上面第 5 条的矛盾（逐状态对比两套分析的候选最终 HV，确认是否逐位一致）
4. 可选：若判为 failure case，考虑"极端动作对比"干预探针（同状态只比较极小 vs 极大探索），确认在最大对比下 pairwise AUC 能否显著 > 0.5

## 环境要点（避免重复踩坑）

- 用 `py -3.9`（anaconda 3.9.13）为默认 `python`；sklearn 1.0.2 与 scipy 1.13 不兼容
  （`Ridge` 会因 `sym_pos` 报错），脚本中已改用 `LinearRegression`
- `evaluate-horizon` 支持**断点续跑**（`.checkpoint_horizon_*.json`，已在 `.gitignore`），
  长时间采集随时可中断重跑
- `_dominates` 已向量化（28× 加速，逐位等价性有测试），长实验成本大幅下降
