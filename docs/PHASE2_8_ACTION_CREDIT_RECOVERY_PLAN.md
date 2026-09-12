# Phase 2.8 Action Credit Recovery Plan

Date: 2026-09-12

## Motivation

Phase 2.75D established that naive advantage prediction does not recover action-dependent signals. The current bottleneck is not sequence modeling capacity, but whether evolutionary action effects are represented and optimized in a learnable way.

The Phase 2.8 goal is to determine whether action credit assignment is fundamentally difficult or whether the previous formulation prevented learning.

## Current Findings

- Outcome prediction is dominated by state information.
- Advantage regression collapses toward state-conditioned constants.
- The model prediction variance within the same state is far smaller than the true action effect variance.
- Increasing model complexity (Mamba/SSM) is not justified before recovering action sensitivity.

## Research Questions

### RQ1: Is action effect learnable?

Run an oracle learnability experiment:

- use stronger non-neural baselines (tree models / ranking models)
- measure within-state ranking correlation
- estimate an upper bound of action predictability

If oracle models fail, the issue is representation/action design.

If oracle models succeed, the issue is objective optimization.

## RQ2: Can ranking objectives recover action usage?

Replace regression-first learning with ranking-first objectives:

- pairwise ranking loss
- listwise ranking loss
- classification of better/worse actions

Metrics:

- Spearman correlation
- Kendall tau
- top-k hit rate
- regret

## RQ3: Does action-specific architecture help?

Compare:

Baseline:

```
concat(state, action) -> MLP
```

Against:

```
state encoder
      |
      +---- ranking head
      |
action encoder
```

Monitor:

- action sensitivity
- within-state prediction variance
- ranking quality

## Phase 2.8 Gates

Do not enter Phase 3 unless:

1. Within-state action sensitivity is demonstrated.
2. Ranking quality exceeds random baseline.
3. Planner performance improves over fixed baselines.

## Planned Experiments

Task 1: Oracle learnability benchmark

Task 2: Ranking-loss advantage learner

Task 3: Action encoder ablation

Task 4: Unified RNG evaluation protocol

Task 5: Final decision on Phase 3 feasibility

## Principle

Do not increase model complexity before proving the action signal is learnable.
