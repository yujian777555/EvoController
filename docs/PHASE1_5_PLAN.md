# Phase 1.5 Plan: Evolution Decision Understanding

## Objective

Phase 1 demonstrated that evolution trajectory contains learnable signals. However, the current MLP controller may partially behave as a mutation-rate optimizer rather than a true evolution policy.

Phase 1.5 aims to verify whether the controller learns dynamic evolution decisions.

## Research Question

Does EvoController learn state-dependent evolution behavior, or does it only discover a globally better mutation probability?

## Tasks

### 1. Expand trajectory dataset

Increase trajectory diversity:

- More random-policy trajectories
- Successful and failed runs
- Wider mutation probability ranges
- Multiple problem characteristics

Goal: reduce overfitting and improve generalization.

### 2. Analyze controller action dynamics

Record controller outputs during evaluation:

- mutation probability trajectory
- operator selection trajectory
- exploration level changes

Compare:

- fixed strategy
- constant strategy
- learned controller

The controller should demonstrate adaptive behavior across generations.

### 3. Expand action space

Move beyond only mutation probability.

Candidate actions:

- mutation probability
- exploration strength
- operator selection

Example:

```json
{
  "mutation_probability": 0.08,
  "operator": "polynomial",
  "exploration_level": "high"
}
```

### 4. Add problem-aware state features

Include problem characteristics:

- dimension
- objective number
- landscape statistics
- convergence difficulty indicators

Goal: enable a single controller to adapt across problems.

## Success Criteria

Phase 1.5 succeeds if:

1. Controller actions vary according to evolution state.
2. Controller behavior differs across optimization problems.
3. Learned policy provides improvement beyond constant mutation baselines.
4. Evidence supports moving to trajectory-aware sequence models (Transformer/Mamba/SSM).

## Restrictions

Do not implement:

- LLM controller
- Mamba controller
- Agent architecture search

Those belong to later phases.

Phase 1.5 focuses only on understanding and validating evolution decision learning.
