# Phase 2.75D: Final Evaluation Protocol

## Motivation

The dry-run evaluation exposed two important corrections:

1. Target B must use an exact default-action baseline.
2. Oracle-hit must always be interpreted together with action variance and SNR.

The next stage freezes the methodology and performs the final validation.

---

## Current Status

Phase 2.75D:

- Infrastructure: complete
- Checkpoint/resume: complete
- Compact feature pipeline: complete
- Dry-run validation: complete
- Full-scale evaluation: pending

---

# Task 1: Complete Intervention Dataset Scaling

Generate the final intervention dataset.

Requirements:

- Keep compact representation only (64 dimensions).
- Maintain state-level train/validation split.
- Preserve same-state multiple-action intervention structure.
- Avoid problem-context features as primary input.

Priority problems:

- ZDT1
- ZDT2
- ZDT4
- ZDT6

Target scale:

- minimum 300 states/problem
- expand further if signal stability requires it

---

# Task 2: Freeze Advantage Definitions

Compare three targets under identical settings.

## Target A

State-relative advantage:

```
candidate outcome - state candidate mean
```

## Target B

Default-policy advantage:

```
candidate outcome - default NSGA-II action outcome
```

## Target C

Future improvement advantage:

```
future outcome improvement
```

No architecture changes during comparison.

---

# Task 3: Evaluation Metrics

Do not rely on a single metric.

Report:

## Ranking quality

- Spearman correlation
- Kendall tau

## Decision quality

- regret
- oracle gap
- top-k action selection

## Signal diagnostics

- action variance
- replicate noise
- SNR versus horizon

For deceptive landscapes, oracle-hit must be reported together with action variance to avoid tie artifacts.

---

# Task 4: Final Research Decision

## Enter Phase 3 only if:

1. Advantage ranking generalizes to held-out states.
2. Planner improves over generation-only and default baselines.
3. Long-horizon action effects remain measurable.

## If not:

Analyze limits of evolutionary action learning and revise the action/intervention formulation.

---

# Research Principle

Do not increase model complexity before establishing that evolutionary action advantages are predictable and transferable.
