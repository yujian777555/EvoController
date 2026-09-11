# Phase 2B Stabilization Plan

## Goal

Before claiming Phase 2B success, ensure that the evaluation protocol matches the research question:

> Can an evolution planner select actions because it understands future consequences, rather than because it learned a static schedule?

---

# Current Status

Phase 2A:
- Outcome predictor validated.
- Input: (state history, candidate action)
- Output: future optimization outcome prediction.

Phase 2B:
- PlanningController implemented.
- Candidate actions are evaluated through the predictor.
- Current evaluation needs protocol alignment.

---

# Rule

Do NOT start Phase 3 (Mamba/SSM) before this phase passes.

The following questions must be answered:

1. Does planning outperform static baselines?
2. Does planning outperform open-loop schedules?
3. Does the selected action have better future consequences?
4. Is improvement caused by planning rather than predictor leakage?

---

# Task 1: Freeze Current Experiment

Do not rerun existing Phase 2B experiments.

Run only:

```bash
python experiments/run_phase2b.py --stage aggregate
```

Verify:

- all runs are present
- metrics are generated
- configuration is saved

---

# Task 2: Statistical Correction

Update Phase 2B comparison scripts.

Requirements:

- add Holm correction
- report raw p-value
- report corrected p-value
- report confidence intervals
- report failed runs

Do not report only mean values.

---

# Task 3: Separate Results Directory

All Phase 2B outputs must be isolated:

```
results/
  phase1_75/
  phase2_outcome/
  phase2b/
```

No experiment may overwrite previous phase results.

---

# Task 4: Redesign Counterfactual Evaluation

Current problem:

Planner optimizes long-horizon future HV, but evaluation may only measure one-step reward.

New evaluation:

For the same population snapshot:

```
state
 |
 + action A -> run k generations -> future HV
 |
 + action B -> run k generations -> future HV
 |
 + action C -> run k generations -> future HV
```

Measure:

- regret
- percentile rank
- predicted vs actual ranking correlation
- oracle gap

Recommended horizons:

- 5 generations
- 10 generations
- 20 generations

---

# Task 5: Candidate Diversity Analysis

Log every candidate action evaluated by PlanningController.

For every generation save:

- candidate action
- predicted future score
- selected action
- actual outcome

Analyze:

- candidate score variance
- prediction uncertainty
- whether argmax decisions are meaningful

---

# Success Criteria

Phase 2B passes if:

1. PlanningController beats fixed/static baselines after correction.
2. Counterfactual evaluation shows selected actions have positive advantage.
3. Predicted ranking correlates with actual future outcomes.
4. Gains are not explained by static schedules.

---

# Failure Handling

If Phase 2B fails:

Analyze:

- action space design
- predictor calibration
- candidate diversity
- reward horizon

Do not add Mamba/Transformer before understanding failure.
