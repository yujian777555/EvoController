# Phase 2.75C: Problem-Aware Advantage Learning

## Motivation

Phase 2.75 demonstrated that intervention-based action advantage learning can recover evolutionary action signals. However, current results show weak generalization on difficult landscapes such as ZDT4 and ZDT6.

The next goal is to learn not only action advantages, but when and why an action is advantageous under different optimization landscapes.

## Research Question

Can an evolution controller learn problem-conditioned long-horizon action advantages instead of relying on problem-specific shortcuts?

## Phase 2.75C Goals

### Task 1: Add Problem Context

Extend the advantage predictor input:

```
state history + action + problem context
```

Potential problem features:

- dimensionality
- objective number
- fitness variance
- population diversity
- convergence speed
- landscape difficulty indicators

### Task 2: Improve Advantage Definition

Compare:

Current:

```
candidate outcome - candidate mean outcome
```

with:

```
candidate outcome - default NSGA-II action outcome
```

The second formulation better represents decision improvement over a baseline strategy.

### Task 3: Hard Landscape Intervention

Focus on difficult benchmarks:

- ZDT4
- ZDT6

Increase intervention samples and analyze why previous planners fail.

### Task 4: Cross-Problem Generalization

Evaluate whether the controller learns general evolution principles.

Training:

- ZDT1
- ZDT2
- ZDT3

Testing:

- ZDT4
- ZDT6

## Gate Before Phase 3

Phase 3 (memory models / SSM) should only start after:

1. Advantage predictor shows reliable action ranking.
2. Difficult landscapes outperform simple scheduling baselines.
3. Cross-problem generalization is demonstrated.

## Current Principle

Do not increase model complexity before solving action-effect generalization.