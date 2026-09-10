# Phase 2 Experiment B: Search Policy Optimization

## Status

Experiment A passed. Do not implement Mamba/Transformer yet.

## Goal

Move from outcome prediction to decision making.

Research question:

Can a learned outcome predictor select better evolution actions than fixed schedules and imitation-based controllers?

## Core formulation

Given:

```
current evolution state
+
candidate actions
```

Predict:

```
future optimization trajectory
```

Then select:

```
argmax predicted long-horizon reward
```

## Experiments

### B1 Candidate Action Planning

At each generation:

1. Generate candidate actions:
   - mutation multiplier
   - exploration strength
   - operator choice

2. Use Outcome Predictor to estimate future HV trajectory.

3. Execute the highest predicted action.

Compare against:

- fixed NSGA-II
- static full-action baseline
- generation-only schedule
- Phase 1.5/1.75 MLP controller

## B2 Counterfactual Decision Validation

For identical population snapshots:

Evaluate:

- predictor-selected action
- random actions
- oracle best action (offline)

Measure:

- regret
- percentile rank
- future HV gain

## B3 Hard Problem Focus

Prioritize:

- ZDT4
- multimodal problems
- higher dimensional variants

because Phase 1.75 showed closed-loop benefits mainly appear on difficult landscapes.

## Success Criteria

Before Phase 3:

1. Planner-selected actions outperform generation-only schedules.
2. Counterfactual rank is significantly above random.
3. Long-horizon prediction improves actual optimization outcomes.

## Restrictions

Do not add Mamba/SSM until planning based on the predictor demonstrates a need for longer temporal modeling.
