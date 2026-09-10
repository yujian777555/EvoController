# Phase 2 Redesign — Experiment A: Evolution Outcome Predictor

日期：2026-09-10。计划文档：[PHASE2_REDESIGN_PLAN.md](PHASE2_REDESIGN_PLAN.md)。

## 目标

验证新的学习公式是否可行：

```
(state history, candidate action) → future HV trajectory
```

替代 Phase 1.5/1.75 的"模仿历史动作"（已证明无法学到因果决策）。

## 方法

- **数据**：500 条全 action 空间随机轨迹（`results/trajectory_phase1_75`，5 ZDT 问题 × seeds 200–299），40500 个 (history, action, future HV) 样本
- **模型**：MLP (64 → 128 → 128 → 4)，torch，MSE + Adam，200 epochs
- **输入**：StateEncoder(window=10) 编码的历史状态（60 维）+ 4 维 action 特征（multiplier, exploration, operator one-hot）
- **输出**：未来 4 个 horizon 的绝对 HV（t+1, t+5, t+10, t+20）
- **基线**：persistence（hv[t+h] = hv[t]）、linear trend（hv[t] + h × 近期平均 delta_hv）
- **统计**：per-sample 平方误差的配对 Wilcoxon（alternative=less，即模型误差更小）

## 结果

### 训练集内（同分布，评估拟合质量）

| horizon | model MSE | persistence MSE | linear MSE | model R² | p vs persistence | p vs linear |
|---|---|---|---|---|---|---|
| t+1 | 9.8e-05 | 2.7e-04 | 1.4e-04 | **0.9995** | 1.2e-112 | 1.0（未显著优于） |
| t+5 | 3.3e-04 | 4.9e-03 | 2.3e-03 | **0.9982** | ~0 | ~0 |
| t+10 | 6.6e-04 | 1.8e-02 | 1.1e-02 | **0.9965** | ~0 | ~0 |
| t+20 | 1.4e-03 | 6.0e-02 | 5.2e-02 | **0.9928** | ~0 | ~0 |

### Held-out 泛化（seeds 20–29，`results/trajectory_full`，与训练集不同种子）

| horizon | model MSE | persistence MSE | linear MSE | model R² | p vs persistence | p vs linear |
|---|---|---|---|---|---|---|
| t+1 | 1.5e-04 | 2.7e-04 | 1.4e-04 | **0.9992** | 0.0012 | 1.0（未显著优于） |
| t+5 | 6.9e-04 | 4.8e-03 | 2.0e-03 | **0.9963** | 8e-180 | 2e-67 |
| t+10 | 1.8e-03 | 1.8e-02 | 9.6e-03 | **0.9905** | 1e-267 | 2e-198 |
| t+20 | 4.1e-03 | 5.9e-02 | 4.9e-02 | **0.9790** | ~0 | ~0 |

### Success criteria（计划书规定）

- ✅ Model MSE < persistence MSE，所有 horizon（两个评估集均成立）
- ✅ Model R² > 0.5 at h∈{1, 5}（实际 > 0.99）
- ✅ Long-horizon（h=20）仍显著优于基线

## 结论

**State + action 包含高度可预测的未来优化结果信息。** 即使只有 3 个标量状态特征（HV/IGD/diversity）+ 4 维 action，一个简单的 MLP 就能以 R² > 0.97 预测 20 代后的 HV，且泛化到未见种子。

**注意事项（诚实记录）**：

1. **h=1 时 linear trend 基线与模型相当**（p=1.0，未显著优于）——单步预测中，简单的趋势外推已足够。模型的价值在**长 horizon**（h≥5）才显著体现。
2. **评估的问题**：当前仅预测 HV，未预测 IGD 或失败风险。zdt4 等困难问题的失败预测可能需要额外特征。
3. **这是预测而非决策**：证明了"可以预测后果"，但 Phase 2 Experiment B 需要验证"能用预测做规划（选动作）"。
4. **轻微过拟合**：val_loss / train_loss ≈ 3.7，但泛化到异种子仍很强，说明过拟合在可接受范围。

## 对 Phase 2 的意义

**Experiment A 通过。** 新的学习公式（预测后果而非模仿动作）在信息层面是可行的。这为 Experiment B（Search Policy Optimization：用 predictor 做候选动作规划）奠定了基础。

**不进入 Mamba/Transformer**：当前 MLP + 窗口历史已足够。序列模型的价值需要在更复杂的任务（如更长历史依赖、跨问题迁移）中重新评估。
