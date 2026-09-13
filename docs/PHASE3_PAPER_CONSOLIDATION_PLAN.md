# Phase 3: Paper Consolidation Plan

## Motivation

Phase 2.8 and Phase 2.9 changed the project direction.

The primary contribution is no longer a stronger evolutionary controller. The evidence suggests that evolutionary action credit assignment itself is difficult:

- outcome prediction is easy but action attribution is hard;
- action effects exist but are difficult to identify;
- improving representation alone has not recovered reliable decision signals.

The next stage focuses on building a rigorous analysis paper rather than continuing uncontrolled model expansion.

---

# Current Scientific Story

## Observation 1: Prediction != Decision

Models can predict future evolutionary performance from state history, but this does not imply that they can choose better interventions.

## Observation 2: Action Effects Exist

Intervention analysis shows that actions influence future outcomes, especially at longer horizons.

## Observation 3: Action Attribution Is the Bottleneck

Existing action representations and outcome signatures cannot reliably rank candidate actions.

---

# Phase 3.1: Recovery Experiment (Recommended)

Goal:

Demonstrate whether the limitation is caused by weak action abstraction rather than complete impossibility.

## Experiment A: Extreme Action Contrast

Compare clearly different interventions:

- low vs high exploration
- weak vs strong mutation
- exploit vs explore operators

Measure:

- pairwise ranking AUC
- regret
- oracle selection accuracy

Purpose:

Test whether stronger intervention separation restores learnability.

---

## Experiment B: Semantic Action Abstraction

Replace raw parameters with higher-level action descriptions:

Examples:

- improve convergence
- increase diversity
- escape stagnation
- exploration recovery

Evaluate whether semantic abstraction improves action credit assignment.

---

# Phase 3.2: Paper Package

Prepare:

- final experimental table
- failure analysis
- mechanism discussion
- reproducibility appendix

Suggested paper contribution:

"Why Learning Evolutionary Decisions Is Hard: A Systematic Study of Action Credit Assignment"

---

# Decision Gate

If recovery experiments succeed:

Present:

failure -> diagnosis -> principled representation fix

If recovery experiments fail:

Freeze as a systematic limitation study.

---

# Restrictions

Do not introduce:

- Mamba
- Transformer
- larger controllers
- architecture search

until action identifiability is resolved.
