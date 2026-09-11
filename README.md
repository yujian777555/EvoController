# EvoController

**Learning How to Evolve** — 让进化算法自身学习搜索策略。

传统进化算法只能进化 solution；本项目研究如何让 evolution strategy 本身被学习，最终构建一个通用的 learning-based Evolution Controller。

详见 [AGENTS.md](AGENTS.md)（研究路线、开发规则、实验规范）。

## Roadmap 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | Evolution Dataset Generation (NSGA-II + benchmarks + trajectory recorder) | ✅ 已完成 (2026-09-08) |
| Phase 1 | Learned Evolution Controller (MLP baseline) | ✅ 已完成 (2026-09-08) |
| Phase 1.5 | Evolution Decision Understanding（action 动态分析 + 扩展 action 空间 + problem 特征） | ✅ 已完成 (2026-09-09) |
| Phase 1.75 | Decision Causality & Fair Baselines（matched full-action + open-loop + counterfactual + 20 seeds） | ⚠️ 已完成 (2026-09-10)，**gate 未通过** |
| Phase 2 Redesign | Outcome Predictor（(state, action) → future HV 预测，替代模仿学习） | ✅ Experiment A 完成 (2026-09-10)，R² > 0.97 |
| Phase 2B | Search Policy Optimization（PlanningController：候选动作采样 + outcome 预测选优） | 🚧 协议已按外部评审修正，结果待重评 (2026-09-11) |
| Phase 2 | Trajectory-aware Controller (Transformer / Mamba / SSM) | ⏸ **暂停**（等待 Experiment B 验证） |
| Phase 3 | Advanced Controller (memory, credit assignment, transfer) | 未开始 |
| Phase 4 | Application (NeuroEvoScientist) | 未开始 |

## 项目结构

```
EvoController/
├── benchmarks/          # Problem 抽象基类 + ZDT1/2/3/4/6 基准问题 + describe() 问题特征
├── algorithms/          # NSGA-II（忠实复现 Deb et al. 2002）+ per-step action 注入（pm/operator/exploration）
├── metrics/             # hypervolume / igd / diversity_spread 质量指标
├── trajectory/          # EvolutionRecorder：记录 (state, action, reward)
├── controller/          # StateEncoder / ProblemAwareEncoder / MLPController / MultiHeadController / ConstantController
│                        # OutcomePredictor（(state, action) → future HV）/ PlanningController（Phase 2B 候选规划）
├── experiments/         # generate_dataset.py（fixed/random × pm/full action 空间）
│                        # run_phase1.py（MLP 基线）、run_phase1_5.py（controller v2）、analyze_actions.py（action 动态分析）
│                        # train/evaluate_outcome_predictor.py（Phase 2A）、counterfactual_actions.py（反事实快照评估，支持 planning controller）
│                        # run_phase2b.py（Phase 2B 闭环配对评估 + Holm 校正/CI/失败率 + 与 Phase-1.75 全部 arms 的统计对比）
│                        # analyze_phase1_75.py（论文级统计工具：Holm、paired bootstrap CI、Wilcoxon）
├── docs/                # Phase 计划与实验记录
├── tests/               # pytest 测试套件（396 项）
└── results/             # 实验输出（git 忽略）
    ├── trajectory/          # Phase 0 固定策略轨迹
    ├── trajectory_random/   # Phase 1 随机策略轨迹（仅 pm）
    ├── trajectory_full/     # Phase 1.5 全 action 空间随机轨迹
    ├── trajectory_phase1_75/ # Phase 1.75 500 条训练语料
    ├── phase1/              # Phase 1 评估结果
    ├── phase1_5/            # Phase 1.5 评估结果
    ├── phase1_75/           # Phase 1.75 评估结果（含 9 arms、快照、controllers）
    ├── phase2_outcome/      # Phase 2A 训练好的 OutcomePredictor + 评估
    ├── phase2b/             # Phase 2B 结果（runs/、results.json、comparison.json、counterfactual/）
    └── analysis/            # action 动态分析与图表
```

> **结果隔离约定**：Phase-2B 的反事实输出默认写入
> `results/phase2b/counterfactual/`，不会覆盖 `results/phase1_75/` 的既有工件。

## 外部评审与协议修正

Phase 2B 的初次结果触发了外部评审，发现评估协议与 planner 目标错配（planner 优化长 horizon
预测 HV，而反事实评估器只测一步 reward）。修正计划与核验记录见：

- [docs/PHASE2B_STABILIZATION_PLAN.md](docs/PHASE2B_STABILIZATION_PLAN.md) —— 协议对齐任务清单
- [docs/PHASE2B_REVIEW_VERIFICATION.md](docs/PHASE2B_REVIEW_VERIFICATION.md) —— 评审条目逐条核验

**B1 主评估结果为初步结论**（已加 Holm 校正、配对 bootstrap CI 与失败率）：planner 显著优于
fixed NSGA-II（zdt1/2/3/6，Holm p = 4.8e-06），但显著落后于 static full-action 与
generation-only schedule（p = 1）。一步反事实 rank ≈ 0.50，按评审意见**不足以证伪**
长 horizon planning —— 长 horizon 反事实评估正在按修正协议实现。

## Quickstart

运行全部测试：

```bash
python -m pytest
```

生成 Phase 0 轨迹数据集（默认：5 个 ZDT 问题 × 5 个 seed × 100 代 × 种群 100）：

```bash
python experiments/generate_dataset.py
```

可选参数（`--problems`、`--seeds`、`--generations`、`--pop-size`、`--out-dir`、`--n-reference-points`、`--ref-point`）详见 `python experiments/generate_dataset.py --help`。

## 轨迹 JSON Schema

每条 run 输出 `{problem}_nsga2_seed{seed}.json`，结构如下（节选）：

```json
{
  "config": {
    "problem": "zdt1", "n_vars": 30, "algorithm": "nsga2",
    "pop_size": 100, "generations": 100,
    "operators": {"crossover_operator": "sbx", "crossover_prob": 0.9,
                  "mutation_operator": "polynomial", "mutation_probability": 0.0333,
                  "eta_c": 20.0, "eta_m": 20.0},
    "ref_point": [1.1, 1.1], "n_reference_points": 200,
    "timestamp_utc": "..."
  },
  "seed": 0,
  "runtime_sec": 12.3,
  "schema_version": 1,
  "transitions": [
    {
      "generation": 0,
      "state": {"generation": 0, "hv": ..., "igd": ..., "diversity": ...},
      "action": {"mutation_operator": "polynomial", "mutation_probability": 0.0333,
                 "exploration_strength": 20.0},
      "reward": {"delta_hv": 0.0, "delta_igd": 0.0}
    }
  ],
  "final": {"hv": ..., "igd": ..., "diversity": ...}
}
```

reward 约定：第 0 代为 0；之后 `delta_hv = hv_t - hv_{t-1}`，`delta_igd = igd_{t-1} - igd_t`（改进为正）。每次批量生成同时写 `index.json` 汇总所有 run 的 final HV / IGD / runtime。

## 与后续 Phase 的关系

Phase 0 产出的是**固定策略（fixed-policy）基线轨迹**：action 在整条轨迹上保持不变（NSGA-II 默认算子配置）。

Phase 1 在此之上加入**随机策略轨迹**（`--policy random`，每代 pm ~ log-uniform [0.5, 5]×1/n）作为训练数据，训练 MLP Evolution Controller（历史 state 窗口 → mutation probability），并与 fixed baseline 对比（含无历史 / 仅当前 state / 历史窗口三组 ablation，5 held-out seeds，Wilcoxon 检验）：

```bash
python experiments/generate_dataset.py --policy random --seeds 10 11 12 13 14 15 16 17 18 19 --out-dir results/trajectory_random
python experiments/run_phase1.py --train-dir results/trajectory_random --out-dir results/phase1
```

**Phase 1 结论摘要**：controller 在 zdt2（p=0.031, w1）和 zdt3（p=0.031, w10）上显著优于 fixed baseline；zdt1 持平；zdt4 因训练分布退化（随机策略下全部 run 失败）而更差。历史窗口的价值得到初步但问题相关的验证。完整结果与失败分析见 [docs/PHASE1_RESULTS.md](docs/PHASE1_RESULTS.md)。

Phase 1.5 扩展 action 空间（operator 选择 + exploration strength）并加入 problem-aware 特征，训练数据扩充到 125 条轨迹（含全 action 空间随机策略）：

```bash
python experiments/generate_dataset.py --policy random --action-space full --seeds 20 21 22 23 24 25 26 27 28 29 --out-dir results/trajectory_full
python experiments/run_phase1_5.py        # 训练 mlp2 / mlp2_nopf 并评估，复用 Phase 1 三臂结果
python experiments/analyze_actions.py     # action 动态分析（Phase 1 轨迹）
```

**Phase 1.5 结论摘要**：核心问题“controller 学到的是状态依赖决策还是全局最优常数”——已有正向证据：mlp2_nopf 在全部 5 个问题上优于 fixed，在 4/5 问题上优于旧 constant-pm 基线；action 动态分析也显示状态相关变化。完整分析见 [docs/PHASE1_5_RESULTS.md](docs/PHASE1_5_RESULTS.md)。

**Planner Gate — Phase 1.75**：在进入 Mamba/Transformer 前，必须进一步排除三个混淆：旧 constant baseline 与 controller 的 action 空间不匹配、5 个 held-out seeds 统计功效不足、动态行为可能被 generation-only 退火 schedule 解释。执行 [docs/PHASE1_75_PLAN.md](docs/PHASE1_75_PLAN.md)，通过 matched full-action baseline、open-loop baseline、500 条训练轨迹、20 held-out seeds 和 counterfactual branch evaluation 验证 closed-loop state feedback 的因果价值。Phase 1.75 未通过 gate 前不得进入 Phase 2。

**Phase 2 重设计（Experiment A）**：模仿学习被证明无因果决策信号后，学习目标转向 outcome——`OutcomePredictor` 从 (encoded state history, candidate action) 回归未来 4 个 horizon 的绝对 HV，held-out R² > 0.97（显著优于 persistence / linear-trend 基线），证明 learned outcome model 是 planning controller 的可行前提。训练与评估细节见 [docs/PHASE2_OUTCOME_PREDICTOR.md](docs/PHASE2_OUTCOME_PREDICTOR.md)。

**Phase 2B（Experiment B）**：`PlanningController`（[controller/planning_controller.py](controller/planning_controller.py)）把 outcome predictor 变成决策规则——每代从 Phase-1.5 全 action 空间采样 `n_candidates` 个候选动作（复用 `sample_full_action`，与训练分布一致），逐候选预测未来 HV 向量，按 horizon 加权和（默认偏向长 horizon）取 argmax 执行。预测输入布局与训练样本 `build_outcome_samples` 逐位一致。反事实评估（B2）通过 `--controller-type planning` 接入既有 snapshot 框架：

```bash
python experiments/counterfactual_actions.py evaluate --problem zdt1 \
  --controller <planner.json> --controller-type planning \
  --predictor results/phase2_outcome/predictor.pt \
  --encoder results/phase2_outcome/encoder.json
```

计划见 [docs/PHASE2_EXPERIMENT_B_PLAN.md](docs/PHASE2_EXPERIMENT_B_PLAN.md)。B1 闭环配对评估（与 Phase-1.75 相同的 held-out 问题 × seed 网格、相同的部署协议，配对 Wilcoxon 对比全部 Phase-1.75 arms）：

```bash
python experiments/run_phase2b.py --stage all   # eval -> aggregate，输出 results/phase2b/{results,comparison}.json
```
