# Phase 1 Plan: Learned Evolution Controller (MLP Baseline)

## Objective

验证核心假设：

> Evolution trajectory 中包含可学习的信息，历史搜索状态可以用于预测和改进未来 evolution strategy。

Phase 1 不实现 LLM Controller、Mamba Controller 或 Agent Search。

当前目标是建立第一个 Learned Evolution Controller baseline。

---

## Research Question

Given historical evolution trajectory:

```
(state_1, action_1, reward_1)
...
(state_t, action_t, reward_t)
```

是否可以学习一个 controller：

```
Evolution history
        |
        v
MLP Controller
        |
        v
Adaptive evolution strategy
```

---

## Implementation Tasks

### 1. Controller Module

新增：

```
controller/
├── dataset.py
├── state_encoder.py
└── mlp_controller.py
```

要求：

- 从 Phase 0 trajectory JSON 加载数据
- 支持 sequence window 输入
- 输出 evolution strategy prediction

---

### 2. Input State

初始版本使用：

- HV
- IGD
- diversity
- delta HV
- delta IGD
- generation

形成 evolution state representation。

---

### 3. Controller Output

预测：

- mutation probability
- exploration strength

后续 Phase 再扩展 operator selection。

---

## Experiments

比较：

### Baseline A

Fixed NSGA-II:

- fixed mutation probability
- fixed operators

### Baseline B

MLP Evolution Controller

---

## Evaluation Metrics

必须包含：

- Final HV
- Final IGD
- Anytime performance
- Evaluation efficiency

不要只比较最终指标。

---

## Required Ablation

至少包含：

1. Without trajectory history
2. Current state only
3. Historical trajectory input

验证历史 evolution information 是否有效。

---

## Reproducibility Requirements

实验必须保存：

- random seed
- config
- training parameters
- evaluation results

默认至少 5 个 random seeds。

---

## Success Criteria

Phase 1 成功标准：

1. Controller 可以从 trajectory 学习 evolution pattern
2. Adaptive strategy 不低于 fixed NSGA-II
3. 证明 trajectory information 对搜索策略学习有价值

---

## Next Phase

Phase 2:

将 MLP Controller 替换为 trajectory-aware Transformer / Mamba / SSM Controller，学习长期 evolution dynamics。
