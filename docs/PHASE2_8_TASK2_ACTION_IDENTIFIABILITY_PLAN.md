# Phase 2.8 Task 2: Action Identifiability Audit Plan

## Objective

Determine whether current action credit assignment failure is caused by insufficient learning methods or by weak/non-identifiable action signals.

## Background

Phase 2B showed outcome prediction can mainly exploit state information rather than action effects.

Phase 2.75D attempted advantage learning:

A(s,a)=R(s,a)-R(s,a_default)

but ranking quality remained near random.

Phase 2.8 Task 1 Oracle learnability further tested stronger learners (OLS, Random Forest, Histogram Gradient Boosting). If stronger models cannot recover action ranking, the bottleneck is likely representation or intervention signal quality.

## Task 1: Action Effect Variance

For fixed states, sample multiple actions and estimate:

- action-induced variance
- rollout noise variance
- action signal-to-noise ratio

SNR = action_effect / noise

Interpretation:

- Low SNR: action effects are intrinsically difficult to learn.
- High SNR: representation or objective likely limits learning.

## Task 2: Action-only Learnability

Train models using only action features to predict advantage.

Purpose:

Determine whether action encoding alone contains predictive information.

## Task 3: Pairwise Action Ranking

Replace regression with pairwise comparison:

Given (s, a_i, a_j), predict whether a_i outperforms a_j.

Metrics:

- accuracy
- AUC
- Kendall tau

## Task 4: Representation Audit

Compare:

1. Concatenated state-action representation
2. Separate state encoder + action encoder + interaction layer

Measure whether action sensitivity improves.

## Decision Gate

If oracle ranking remains random and SNR is low:

Conclusion: evolutionary action credit assignment is limited by weak intervention signals.

If oracle ranking succeeds:

Investigate improved representations and objectives before introducing larger models.

## Restrictions

Before completing this audit, do not introduce Mamba, Transformer, or larger controllers. The current question is signal identifiability, not model capacity.
