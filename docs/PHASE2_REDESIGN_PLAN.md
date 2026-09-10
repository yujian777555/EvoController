# Phase 2 Redesign Plan: Learning Long-Horizon Evolution Dynamics

## Status

Phase 1.75 gate failed. Do not directly implement Mamba/Transformer yet.

## Motivation

Phase 1.75 showed that current closed-loop behavior cloning does not reliably learn causal state-dependent decisions. Generation-only schedules and larger action spaces explain most of the observed gains.

The next phase must change the learning formulation, not simply increase model capacity.

## Research Question

Can an agent learn long-horizon evolution dynamics and make decisions that improve optimization beyond static schedules?

## Key Changes

### 1. Move from imitation learning to decision learning

Current:

```
trajectory -> imitate recorded action
```

New:

```
state + action candidates -> predict long-term consequence
```

The controller should evaluate actions, not only copy historical choices.

### 2. Richer evolution state

Current state:

```
HV
IGD
diversity
```

Add:

- population distribution statistics
- convergence speed
- stagnation indicators
- operator history
- multi-generation trend features

### 3. Long-horizon reward

Replace only single-step delta reward with:

```
short-term improvement
+
future hypervolume gain
+
failure avoidance
```

### 4. Sequence model justification

Only introduce Mamba/SSM when the task requires:

```
long evolution history
        |
        v
future optimization outcome prediction
```

## Candidate Experiments

### Experiment A: Outcome Predictor

Train a model:

```
(state history, candidate action)
        |
        v
future HV trajectory
```

Compare predicted and actual outcomes.

### Experiment B: Search Policy Optimization

Use the predictor as a planning module:

```
candidate actions
        |
        v
predicted future reward
        |
        v
select action
```

### Experiment C: Hard Problem Focus

Prioritize:

- multimodal problems
- high-dimensional problems
- constrained optimization

because Phase 1.75 showed closed-loop value appears mainly on difficult landscapes.

## Phase 2 Entry Criteria

Before Mamba/SSM:

1. Learned decision model beats generation-only schedule.
2. Counterfactual action ranking is significantly above random.
3. Long-horizon prediction provides measurable value.

## Restrictions

Do not implement Mamba immediately.
First prove that the new learning formulation requires sequence modeling.
