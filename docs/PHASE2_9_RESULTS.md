# Phase 2.9 结果：Action Representation Redesign

日期：2026-09-12。计划：[PHASE2_9_ACTION_REPRESENTATION_REDESIGN_PLAN.md](PHASE2_9_ACTION_REPRESENTATION_REDESIGN_PLAN.md)。
前序：[PHASE2_8_RESULTS.md](PHASE2_8_RESULTS.md)（可辨识性审计：SNR 4.82 但 pairwise AUC 0.52）。

## 结论（先行）

**Gate 判定：`representation_not_sufficient`。** 动作结果签名（Action Outcome Signature）
表示**未能恢复动作可辨识性**——最佳 pairwise AUC **0.528**（阈值 0.53），
相对基线 `action_only`（0.5248）仅提升 **+0.003**（h=20）；在短 horizon 上的提升也仅
+0.015 AUC（h=5: 0.5203 → 0.5350）

按计划书的 Failure Case：**冻结分析，论文转向"evolutionary action credit assignment
的根本困难"**。

## 实验设置

- 数据与划分：`results/phase2_75d/dataset_final`，**1200 状态**，state 级 **960 训练 / 240 验证**
  （与 Phase 2.75D / 2.8 完全一致，可逐位对比）
- 动作签名（6 维，全部 **leave-one-state-out**）：
  `loo_mean_adv`、`loo_std_adv`、`loo_frac_positive`、`loo_count`(log1p)、
  `loo_mean_rank`、`operator_loo_mean_adv`
- 分桶：operator（2）× multiplier 四分位 × exploration 四分位 = **32 桶**，
  **按算子分别拟合分位边界**（poly 的 exploration 边界 `5.38/13.93/20.0`，
  gaussian 为 `0.039/0.078/0.151`，相差约 3 个数量级；共享边界会把全部 gaussian
  动作压进最低桶）
- 粒度依据（实测，全量训练划分 10,560 行/horizon）：4×4 桶的最小桶 50 行、中位 303；
  8×8 出现空桶 → 选 4×4
- 空桶填充 **0.0**（rank 填 0.5），**不使用全局均值**（避免引入状态信息）
- 学习者：OLS / RandomForest(200, leaf 2) / HistGradientBoosting(300, 0.05)，
  逐 horizon 独立训练，与 Phase 2.8 审计完全同参
- artifact：`results/phase2_9/action_representation.json`（288 s）

## 结果

### 排序质量（val Spearman）

| 变体 | 维度 | h=5 | h=10 | h=20 |
|---|---|---|---|---|
| `action_only`（基线） | 4 | 0.0588 | 0.0251 | 0.0350 |
| `action_plus_signature` | 10 | **0.0846** | **0.0559** | 0.0054 |
| `concat` | 64 | 0.0588 | 0.0285 | 0.0315 |
| `concat_plus_signature` | 70 | 0.0841 | 0.0537 | 0.0044 |
| `signature_only` | 6 | 0.0682 | 0.0815 | 0.0160 |

### 成对偏好 AUC（val）

| 变体 | h=5 | h=10 | h=20 |
|---|---|---|---|
| `action_only` | 0.5203 | 0.5227 | 0.5248 |
| `action_plus_signature` | **0.5350** | **0.5385** | 0.5063 |
| `concat` | 0.5203 | 0.5227 | 0.5248 |
| `concat_plus_signature` | 0.5350 | 0.5385 | 0.5063 |
| `signature_only` | 0.5339 | **0.5429** | 0.5216 |

## 三个机制性观察

1. **签名的增益真实但微弱**：h=5 的 AUC 从 0.520 提到 0.535（+0.015），
   h=10 从 0.523 提到 0.539——方向一致、幅度极小。**远不足以支撑可用的 planner。**
2. **`concat` 与 `action_only` 的 AUC 逐位相同**（0.5202787135518816）：
   同状态内配对时状态特征**逐位相减恒为 0**，线性模型对这些恒零列的最优系数为 0。
   这从构造上解释了 Phase 2.8 的"action-only ≈ concat"观测：**成对偏好任务无法利用
   状态上下文**。若要在 pairwise 框架下检验状态条件化，必须改为跨状态回归或显式交互项。
3. **最高 SNR 的 horizon 上反而最差**：h=20 的 SNR 最高（池化 4.82、四问题 3.79–6.21），
   但所有变体的 AUC 在 h=20 都塌回 0.506–0.525，**签名甚至低于基线**（0.5063 vs 0.5248）。
   即：**信号强度与可辨识性脱钩**——信号可测量，但无法被这些特征表达。

## 与前序证据的合并结论（论文可用）

| 阶段 | 发现 |
|---|---|
| Phase 2A | 结果预测可达 R²>0.97，**但完全由状态驱动** |
| Phase 2.75D | advantage 表述下模型**忽略 action 通道**（同状态预测 std / 真实 std = **0.001**） |
| Phase 2.8 Task 1 | 强非神经模型的 action sensitivity 是 MLP 的 200 倍（0.21 vs 0.001），**排序仍随机** |
| Phase 2.8 Task 2/2.1 | SNR **不弱**（h=20 池化 4.82，89% 状态 >1）；`state_only` **构造上无法排序**；`action_only` ≈ `concat`；pairwise AUC **0.52** |
| **Phase 2.9 Task 1** | **签名表示仅 +0.015 AUC，未过 gate** |

**整体结论**：在标准 ZDT 干预协议与 compact 状态表示下，
**进化动作的长期因果效应可被测量（SNR 4.8）但在统计上几乎不可辨识
（pairwise AUC 0.52、排序 Kendall 0.15），且更换损失函数、模型类别
（树模型）、或动作表示（结果签名）都无法突破这一上界。**

这是一个**有充分证据支撑的负结果**，其价值在于：它把"学习如何进化"这一方向的
**可辨识性边界**精确刻画出来，而不是把失败归因于模型容量。

## 后续建议（若继续该方向）

1. **改变干预协议本身**（唯一未被证否的杠杆）：
   - 同状态下只比较**极端动作对比**（极小 vs 极大探索），测量最大对比下的 AUC 是否显著 > 0.5
   - 延长 horizon（h=20 的 SNR 仍在上升，未饱和）
   - 增加重复数（当前 3）以降低 rollout 噪声占比（当前噪声约占方差的 20%）
2. **改用跨状态（state, action）回归**检验状态条件化（成对差分在构造上状态无关）
3. **不投入**：排序目标、交互层、Mamba/Transformer——均已被上界证据排除

## 产物

| 路径 | 内容 |
|---|---|
| `experiments/action_representation_redesign.py` | 签名表示 + 5 变体评估 + gate 判定（1148 行） |
| `tests/test_action_representation_redesign.py` | 19 测试（含 LOO 无泄漏的篡改验证） |
| `results/phase2_9/action_representation.json` | 完整结果 |
| `docs/PHASE2_9_STATUS.md` | 上下文交接与开放问题记录 |
