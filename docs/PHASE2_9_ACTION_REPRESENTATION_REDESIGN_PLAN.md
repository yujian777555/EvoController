# Phase 2.9 Action Representation Redesign Plan

## Motivation

Phase 2.8 Task 2.1 established that the failure of action-aware controller learning is not caused by the absence of action effects.

Key finding:

- action intervention signal exists (SNR > 1 at long horizon)
- current action representation cannot expose state-dependent action quality
- increasing model capacity or changing loss alone is unlikely to overcome the representation bottleneck

Therefore Phase 2.9 focuses on redesigning action representation rather than adding larger models.

---

# Scientific Question

Can a better representation of evolutionary actions recover learnable action credit signals?

Current representation:

```
action parameters only
(multiplier, exploration, operator one-hot)
```

Problem:

It describes what action was selected, but not how that action interacts with the current evolutionary state.

---

# Phase 2.9 Tasks

## Task 1: Action Outcome Signature Representation

Goal:

Augment action representation with historical intervention outcomes.

Candidate features:

- previous HV improvement
- convergence contribution
- diversity change
- feasibility change
- recent operator effectiveness

Question:

Does adding action outcome signatures improve action ranking?

---

## Task 2: State-conditioned Action Representation

Compare:

### Baseline

```
concat(state, action)
```

### New representation

```
state encoder
      +
action encoder
      +
state-action interaction
```

Evaluation:

- action sensitivity
- ranking quality
- advantage prediction

---

## Task 3: Pairwise Action Preference Learning

Replace absolute regression:

```
(s, a) -> advantage
```

with:

```
(s, action_A, action_B) -> which action wins
```

Reason:

The final decision problem is ranking actions, not predicting exact reward.

---

# Experimental Protocol

Must preserve:

- Phase 2.75D dataset protocol
- existing state split
- fixed random seeds
- same evolutionary benchmark suite

Do not introduce:

- Mamba
- Transformer
- larger controller architectures

until representation improvement is validated.

---

# Decision Gate

## Success case

If redesigned representation improves:

- pairwise AUC
- ranking correlation
- controller performance

then proceed to final controller evaluation.

## Failure case

If representation redesign cannot recover action ranking:

freeze analysis around the fundamental difficulty of evolutionary action credit assignment.

---

# Expected Deliverables

Code:

```
experiments/action_representation_redesign.py
```

Results:

```
results/phase2_9/
```

Documentation:

```
docs/PHASE2_9_RESULTS.md
```

---

# Next Execution Step

Implement Task 1 first:

Action Outcome Signature Representation.

Do not start model scaling before this experiment answers whether representation is the bottleneck.
