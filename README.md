# EvoController

**Learning How to Evolve** — 让进化算法自身学习搜索策略。

传统进化算法只能进化 solution；本项目研究如何让 evolution strategy 本身被学习，最终构建一个通用的 learning-based Evolution Controller。

详见 [AGENTS.md](AGENTS.md)（研究路线、开发规则、实验规范）。

## Roadmap 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | Evolution Dataset Generation (NSGA-II + benchmarks + trajectory recorder) | 未开始 |
| Phase 1 | Learned Evolution Controller (MLP baseline) | 未开始 |
| Phase 2 | Trajectory-aware Controller (Transformer / Mamba / SSM) | 未开始 |
| Phase 3 | Advanced Controller (memory, credit assignment, transfer) | 未开始 |
| Phase 4 | Application (NeuroEvoScientist) | 未开始 |
