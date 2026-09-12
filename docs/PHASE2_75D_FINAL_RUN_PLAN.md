# Phase 2.75D Final Run Plan

## Goal

Complete the final scientific validation of EvoController:

> Determine whether intervention-based advantage learning can recover the long-horizon value of evolutionary actions.

At this stage, freeze the experimental protocol. Do not expand model complexity before validating the core hypothesis.

---

## Current Research Line

```
Phase 2B
Outcome Prediction
        |
        | failure: state dominates action
        v
Phase 2.75
Advantage Learning
        |
        | discovery: delayed action effects exist
        v
Phase 2.75C
Problem Context Analysis
        |
        | failure: shortcut / degradation
        v
Phase 2.75D
Final Evaluation
```

---

# Task 1: Freeze Configuration

## Feature representation

Use only:

```
compact 64-dimensional representation
```

Do not use problem context features in the final comparison because previous experiments indicate shortcut risks.

## Benchmarks

Final evaluation:

- ZDT1
- ZDT2
- ZDT4
- ZDT6

## Snapshot sampling

Normalize evaluation:

```
300 states per problem
```

Avoid imbalance from different snapshot pool sizes.

---

# Task 2: Intervention Dataset

Generate controlled samples:

```
(state, action, outcome)
```

For every state:

- same initial snapshot
- multiple candidate actions
- independent rollouts
- fixed random seed protocol

Store:

- state id
- problem
- action
- horizon
- reward
- advantage

---

# Task 3: Advantage Target Comparison

Compare three targets under identical settings.

## Target A

Relative advantage:

```
A(a)=R(a)-mean(R)
```

## Target B

Default-policy advantage:

```
A(a)=R(a)-R(default_action)
```

## Target C

Long-horizon improvement:

```
A(a)=R_future(a)-R_current
```

---

# Task 4: Evaluation Metrics

Do not rely only on regression error.

## Ranking

Report:

- Spearman correlation
- Kendall tau

## Decision quality

Report:

- regret
- oracle gap
- top-k hit rate

## Mechanism analysis

Report:

- action variance
- signal-to-noise ratio
- horizon dependency

Horizons:

- h=5
- h=10
- h=20

---

# Task 5: Hard Landscape Analysis

Focus on ZDT4.

Analyze:

- action separability
- population diversity change
- exploration behavior
- escape behavior

The goal is to distinguish controller failure from inherently weak action effects.

---

# Final Gate

Proceed to Phase 3 only if:

## Gate 1

Advantage prediction beats random ranking baseline.

## Gate 2

Advantage planner outperforms:

- NSGA-II
- generation-only controller
- outcome predictor planner

## Gate 3

No severe degradation on difficult landscapes.

---

# Next Execution Order

```
1. Complete 300-state intervention evaluation
        |
2. Train A/B/C advantage models
        |
3. Produce final comparison tables
        |
4. Analyze difficult landscapes
        |
5. Write Phase 2.75D results
        |
6. Decide Phase 3 direction
```

Do not introduce new architectures before this evaluation is complete.
