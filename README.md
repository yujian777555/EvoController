# EvoController

**Learning How to Evolve** — 让进化算法自身学习搜索策略。

传统进化算法只能进化 solution；本项目研究如何让 evolution strategy 本身被学习，最终构建一个通用的 learning-based Evolution Controller。

详见 [AGENTS.md](AGENTS.md)（研究路线、开发规则、实验规范）。

## Roadmap 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | Evolution Dataset Generation (NSGA-II + benchmarks + trajectory recorder) | ✅ 已完成 (2026-09-08) |
| Phase 1 | Learned Evolution Controller (MLP baseline) | ✅ 已完成 (2026-09-08) |
| Phase 2 | Trajectory-aware Controller (Transformer / Mamba / SSM) | 未开始 |
| Phase 3 | Advanced Controller (memory, credit assignment, transfer) | 未开始 |
| Phase 4 | Application (NeuroEvoScientist) | 未开始 |

## 项目结构

```
EvoController/
├── benchmarks/          # Problem 抽象基类 + ZDT1/2/3/4/6 基准问题
├── algorithms/          # NSGA-II（忠实复现 Deb et al. 2002）+ OperatorConfig + per-step action 注入
├── metrics/             # hypervolume / igd / diversity_spread 质量指标
├── trajectory/          # EvolutionRecorder：记录 (state, action, reward)
├── controller/          # Phase 1：StateEncoder / 数据集构建 / MLPController / ConstantController
├── experiments/         # generate_dataset.py（fixed/random 策略）、run_phase1.py（训练+评估+ablation）
├── docs/                # Phase 计划与实验记录
├── tests/               # pytest 测试套件（159 项）
└── results/             # 实验输出（git 忽略）
    ├── trajectory/          # Phase 0 固定策略轨迹
    ├── trajectory_random/   # Phase 1 随机策略训练轨迹
    └── phase1/              # Phase 1 评估结果与轨迹
```

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
      "action": {"mutation_operator": "polynomial", "mutation_probability": 0.0333},
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
