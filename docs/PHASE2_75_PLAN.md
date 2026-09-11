# Phase 2.75: Evolution Intervention Learning

## Motivation

Phase 2B showed that outcome prediction is possible, but action-conditioned planning is not yet reliable.

Current finding:

- OutcomePredictor achieves high prediction accuracy.
- Action shuffle and fixed-state ranking diagnostics indicate predictions are dominated by state information.
- The model does not yet provide reliable estimates of causal action effects.

Therefore Phase 2.75 changes the objective from outcome prediction to action advantage learning.

---

## Research Question

Can we learn the causal contribution of evolutionary actions?

Instead of learning:

```
(state, action) -> future HV
```

learn:

```
(state, action) -> action advantage
```

where:

```
advantage = outcome(action) - outcome(baseline)
```

---

# Tasks

## Task 1: Freeze Phase 2B failure analysis

Preserve the negative result as scientific evidence.

Record:

- action shuffle analysis
- fixed-state ranking evaluation
- action effect SNR
- prediction calibration

The goal is to explain why high prediction accuracy does not imply planning ability.

---

## Task 2: Build intervention dataset

Passive trajectories are insufficient because action variation is weak.

For the same evolution state:

```
Population_t
   |
   +-- Action A -> future outcome
   |
   +-- Action B -> future outcome
   |
   +-- Action C -> future outcome
```

Each branch must start from the same snapshot.

Required properties:

- same initial population
- different action intervention
- independent future rollout
- reproducible seeds

---

## Task 3: Redesign action representation

Investigate whether the current continuous action space hides causal effects.

Consider higher-level actions:

```
exploration
balance
exploitation
```

or:

```
increase diversity
maintain diversity
accelerate convergence
```

Goal:

Increase action effect signal-to-noise ratio.

---

## Task 4: Train Action Advantage Predictor

Input:

```
state history + candidate action
```

Output:

```
expected action advantage
```

Evaluation should focus on decision quality rather than only regression metrics.

Required metrics:

- Spearman correlation
- Kendall correlation
- regret
- oracle hit rate
- oracle gap

---

## Task 5: Advantage-based Planner

Replace:

```
argmax predicted future HV
```

with:

```
argmax predicted action advantage
```

Compare against:

- fixed NSGA-II
- static schedule
- generation-only controller
- Phase 2B planner

---

# Gate Before Mamba / SSM

Do not start memory-model research until:

1. Predictor performance decreases after action ablation.
2. Predicted action ranking correlates with real ranking.
3. Intervention experiments demonstrate positive action advantage.
4. Planner improves over non-causal baselines.

If the gate fails, analyze the fundamental learnability limits of evolutionary actions rather than increasing model complexity.
