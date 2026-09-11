# Phase 2.75D: Advantage Target Refinement and Intervention Scaling

## Current Status

Phase 2.75C completed a problem-aware advantage learning investigation.

Main findings:

- Adding problem context features increased input dimension but reduced performance.
- The main bottleneck is not missing problem identity information.
- The remaining challenge is learning stable action advantage signals.

Current conclusion:

```
Outcome prediction        SUCCESS
Action advantage learning PARTIALLY SUCCESSFUL
Problem-aware features   NOT HELPFUL
Generalization           NEEDS IMPROVEMENT
```

Phase 3 (Mamba/SSM) is blocked until advantage learning is validated.

---

# Research Question

Can evolutionary action advantages be learned robustly through better targets and denser intervention data?

The focus changes from:

```
Which problem am I solving?
```

to:

```
Under this evolutionary state, which action produces a measurable long-horizon improvement?
```

---

# Task 1: Freeze Phase 2.75C Negative Findings

Create a complete record of:

- problem context feature experiments
- v1 vs v2 input comparison
- cross-problem generalization results
- failure analysis

Do not discard the negative result.

The failure demonstrates that additional context alone does not solve causal decision learning.

---

# Task 2: Return to Compact State Representation

Use the validated v1 representation as the default:

```
state history + action
```

Avoid adding problem identity features unless an ablation proves clear benefit.

Goal:

Prevent shortcut learning from problem labels.

---

# Task 3: Advantage Target Study

Systematically compare:

## Target A

Candidate action minus candidate mean:

```
A = R(action) - mean(R(all actions))
```

## Target B

Candidate action minus default evolutionary policy:

```
A = R(action) - R(default NSGA-II)
```

## Target C

Direct future improvement:

```
A = future_metric - current_metric
```

Evaluation must focus on decision quality:

- Spearman ranking correlation
- Kendall ranking correlation
- regret
- oracle gap
- hit rate

Do not rely only on regression MSE.

---

# Task 4: Scale Intervention Dataset

Current intervention data is useful but limited.

Increase:

```
snapshots: 200 -> 1000+
```

Priority:

- more states per landscape
- more repeated interventions
- stronger coverage of difficult landscapes

Maintain:

```
same state
multiple actions
independent rollouts
```

The intervention property must remain valid.

---

# Task 5: ZDT4/ZDT6 Diagnostic Study

Do not only optimize final HV.

Analyze:

- diversity recovery
- local optimum escape behavior
- exploration duration
- action selection trajectory

Determine whether failures come from:

1. wrong advantage target
2. insufficient intervention data
3. inadequate action space
4. horizon mismatch

---

# Success Gate Before Phase 3

Phase 3 (Memory/SSM) starts only if:

## Gate 1

Advantage predictor reliably ranks actions:

```
Spearman > 0.6
```

## Gate 2

Planner beats simple baselines:

- fixed NSGA-II
- generation-only schedule

## Gate 3

Hard landscapes improve:

Especially:

- ZDT4
- ZDT6

## Gate 4

Cross-problem transfer remains effective without relying on problem identity shortcuts.

---

# Forbidden Changes

Do not:

- introduce Mamba yet
- increase model size without evidence
- add problem embeddings as the default solution
- optimize only easy benchmarks

The current bottleneck is causal advantage identification, not sequence modeling.
