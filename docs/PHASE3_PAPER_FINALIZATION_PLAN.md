# Phase 3: Paper Finalization Plan

## Current Status

EvoController has completed:

- Phase 2B Outcome Prediction Analysis
- Phase 2.75D Advantage Learning
- Phase 2.8 Oracle Learnability
- Phase 2.9 Action Representation Redesign
- Phase 3.1 Extreme Action Contrast

Core conclusion:

> Evolutionary action effects exist, but action credit assignment remains difficult under the current intervention protocol.

---

# Final Paper Position

Do not continue:

- larger controllers
- Mamba / Transformer experiments
- architecture search
- blind model scaling

The experiments have already tested multiple failure explanations:

1. model capacity limitation
2. objective design limitation
3. action outcome signature limitation
4. action magnitude / contrast limitation

---

# Paper Contributions

## Contribution 1

A systematic evaluation framework for evolutionary action credit assignment:

- intervention protocol
- oracle learnability analysis
- SNR analysis
- action identifiability evaluation

## Contribution 2

Prediction does not imply decision ability.

Models can predict future optimization outcomes while failing to identify which action caused improvement.

## Contribution 3

A systematic analysis of action credit assignment bottlenecks:

- state dominance
- weak action attribution
- representation limitations

---

# Remaining Optional Experiment

## Semantic Action Abstraction Probe

Optional only.

Test whether high-level search intents are more learnable than low-level evolutionary operators.

Example:

Low-level:

- mutation rate
- operator selection
- exploration parameter

High-level:

- increase diversity
- improve convergence
- escape stagnation

Purpose:

Determine whether the bottleneck comes from low-level action space design or evolutionary decision learning itself.

---

# Final Goal

Freeze experimentation after final validation.

Prepare:

- paper draft
- appendix
- reproducibility package

The project should be presented as a systematic study of the limits of learning evolutionary decisions, rather than only as a controller proposal.
