# Phase 2.8 Task 2.1: Action Signal Identifiability Analysis

## Objective

Determine whether evolutionary action effects contain a stable and learnable signal.

The current failure of action credit learning may come from:

1. weak intervention signal
2. rollout noise domination
3. insufficient action representation

This task does not introduce new controllers. It only diagnoses whether the action signal exists.

---

## Experiment 1: Action Effect Variance

For a fixed evolutionary state `s`, sample multiple actions:

```
a1, a2, ..., ak
```

Run intervention rollouts and measure:

- action-induced reward variance
- advantage variance

Target quantity:

```
Var(R(s,a))
```

Question:

> Are different actions producing distinguishable outcomes under the same state?

---

## Experiment 2: Rollout Noise Estimation

For identical:

```
(state, action)
```

run multiple random seeds.

Estimate:

```
Var(R(s,a,seed))
```

This measures stochastic optimization noise.

---

## Experiment 3: Signal-to-Noise Ratio

Compute:

```
SNR = action_effect_variance / rollout_noise_variance
```

Evaluate across:

- problems
- horizons
- seeds

Recommended horizons:

- h=5
- h=10
- h=20

---

## Experiment 4: Action-only Baseline

Train a lightweight model using only action features.

Input:

```
action features
```

Target:

```
advantage
```

Purpose:

Determine whether action representation itself contains predictive information.

---

## Outputs

Generate:

```
results/phase2_8/action_identifiability.json
```

Include:

- per problem statistics
- per horizon statistics
- action variance
- noise variance
- SNR
- action-only baseline metrics

---

## Decision Gate

### Case A

If SNR is low and oracle ranking remains random:

Conclusion:

Evolutionary action credit assignment is fundamentally limited by weak intervention signals.

Proceed toward a mechanism/analysis paper.

---

### Case B

If SNR is high:

Conclusion:

The signal exists, but current representation/objective is insufficient.

Proceed to:

- action representation redesign
- ranking objectives
- interaction models

---

## Constraints

Before completing this task:

Do not add:

- Mamba
- Transformer
- larger controllers
- new optimization algorithms

The purpose is diagnosis, not model scaling.
