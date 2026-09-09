# Phase 1.75 Decision Causality & Fair Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that EvoController's gains come from closed-loop, state-dependent evolutionary decisions rather than a stronger action space, a fixed schedule, problem-scale leakage, or test-set luck.

**Architecture:** Keep the current Phase-1.5 MLP multi-head controller and NSGA-II backbone unchanged as the reference implementation. Add matched full-action static/open-loop baselines, normalize the mutation action by the natural NSGA-II scale `1 / n_vars`, enlarge the trajectory dataset, and add counterfactual one-step branch evaluation from identical population states. Phase 1.75 is a scientific-control phase: it must remove confounds before any Transformer/Mamba/SSM work begins.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, pytest, existing EvoController NSGA-II/ZDT infrastructure.

**Spec:** `docs/PHASE1_5_RESULTS.md` and `docs/PHASE1_5_PLAN.md`.

## Global Constraints

- Do **not** implement Mamba, Transformer, SSM, LLM controllers, RL policies, or NeuroEvoScientist integration in this phase.
- All primary comparisons must use the **same three-dimensional action space**: mutation operator, mutation probability, exploration strength.
- No baseline hyperparameter may be tuned on held-out test seeds.
- Keep training/validation/test seeds disjoint and record them in result JSON.
- Primary test evaluation must use at least **20 held-out seeds per problem**.
- Report mean, standard deviation, median paired difference, 95% bootstrap confidence interval, paired Wilcoxon p-value, and Holm-corrected p-value for the five ZDT problems.
- Preserve failed runs; never drop or silently rerun them.
- All new code requires type hints, docstrings, deterministic seeded tests, and `python -m pytest` must pass before completion.

---

## Why Phase 1.75 is required

Phase 1.5 is promising: `mlp2_nopf` beat fixed NSGA-II on all five ZDT problems and beat the old constant-pm baseline on four of five. However, two important confounds remain:

1. The old `constant` baseline controls only mutation probability, while `mlp2_nopf` controls operator + mutation probability + exploration strength. This is not a matched action-space comparison.
2. The current held-out evaluation uses only five seeds, where a one-sided Wilcoxon result of `p=0.031` is the minimum possible p-value. This is useful as a pilot result but insufficient as the main paper claim.

A third ambiguity is causal: action trajectories vary with state, but that alone does not prove that state feedback causes the performance gain. A generation-dependent open-loop schedule could imitate annealing without using population state.

Phase 1.75 must answer the stronger question:

> **Does closed-loop population-state feedback improve evolutionary decisions beyond the best static full-action policy and a generation-only open-loop schedule?**

---

## File map

### Create

- `controller/action_normalization.py` — normalized mutation multiplier representation and conversions.
- `controller/static_full_controller.py` — matched full-action fixed controller.
- `controller/open_loop_controller.py` — generation-only schedule controller with the same full action space.
- `experiments/tune_static_full.py` — training-seed-only tuning of full-action static baselines.
- `experiments/run_phase1_75.py` — 20-seed paired evaluation and ablations.
- `experiments/counterfactual_actions.py` — one-step branch evaluation from identical population states.
- `experiments/analyze_phase1_75.py` — statistics, confidence intervals, Holm correction, and summary tables.
- `docs/PHASE1_75_RESULTS.md` — final scientific record.
- `tests/test_action_normalization.py`
- `tests/test_static_full_controller.py`
- `tests/test_open_loop_controller.py`
- `tests/test_counterfactual_snapshot.py`
- `tests/test_phase1_75_statistics.py`

### Modify

- `controller/dataset.py` — add normalized mutation-multiplier targets without breaking legacy builders.
- `controller/multihead_controller.py` — support normalized mutation-multiplier prediction behind an explicit flag/config; preserve old Phase-1.5 behavior for reproducibility.
- `algorithms/nsga2.py` — add deterministic snapshot/restore or branch-step support needed for counterfactual evaluation.
- `experiments/generate_dataset.py` — support the Phase-1.75 500-trajectory training corpus and record normalized action metadata.
- `README.md` — mark Phase 1.75 complete only after all criteria are evaluated.

---

## Task 1: Normalize mutation action scale

**Files:**
- Create: `controller/action_normalization.py`
- Modify: `controller/dataset.py`
- Modify: `controller/multihead_controller.py`
- Test: `tests/test_action_normalization.py`

**Interfaces:**
- Produces: `mutation_multiplier(pm: float, n_vars: int) -> float`
- Produces: `mutation_probability(multiplier: float, n_vars: int) -> float`
- Produces: `log_mutation_multiplier(pm: float, n_vars: int) -> float`
- Produces: `pm_from_log_multiplier(log_multiplier: float, n_vars: int) -> float`

The natural NSGA-II baseline is `pm_base = 1 / n_vars`. The normalized action is therefore:

```python
multiplier = mutation_probability * n_vars
```

and the regression target is:

```python
log_multiplier = np.log(mutation_probability * n_vars)
```

This removes the trivial `n_vars=10` versus `n_vars=30` mutation-scale difference from the learned target.

- [ ] **Step 1: Write failing conversion round-trip tests**

```python
def test_mutation_multiplier_round_trip() -> None:
    pm = 0.08
    n_vars = 30
    k = mutation_multiplier(pm, n_vars)
    assert k == pytest.approx(2.4)
    assert mutation_probability(k, n_vars) == pytest.approx(pm)


def test_log_multiplier_round_trip() -> None:
    pm = 0.033333333333
    n_vars = 30
    z = log_mutation_multiplier(pm, n_vars)
    assert pm_from_log_multiplier(z, n_vars) == pytest.approx(pm)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_action_normalization.py -v`

Expected: FAIL because the new module/functions do not exist.

- [ ] **Step 3: Implement strict conversion functions**

Reject `n_vars < 1`, `pm <= 0`, and `multiplier <= 0` with `ValueError`. Use `math.log` / `math.exp` and return Python floats.

- [ ] **Step 4: Add a normalized-target path to the multi-head dataset/controller**

Keep existing Phase-1.5 absolute-log-pm behavior as the default compatibility mode. Add an explicit configuration field such as `mutation_target="absolute" | "multiplier"`; Phase 1.75 must use `"multiplier"`.

- [ ] **Step 5: Run focused tests, then full tests**

Run:

```bash
python -m pytest tests/test_action_normalization.py -v
python -m pytest
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add controller/action_normalization.py controller/dataset.py controller/multihead_controller.py tests/test_action_normalization.py
git commit -m "Phase-1.75: Normalize mutation action scale"
```

---

## Task 2: Build matched full-action static baselines

**Files:**
- Create: `controller/static_full_controller.py`
- Create: `experiments/tune_static_full.py`
- Test: `tests/test_static_full_controller.py`

**Interfaces:**
- Produces: `StaticFullController(operator: str, mutation_probability: float, exploration_strength: float)`
- Produces: `predict_action(...) -> dict[str, Any]` returning the same keys as `MultiHeadController.predict_action`.
- Produces tuning artifact: `results/phase1_75/static_full_tuning.json`.

Implement two baselines using **training seeds only**:

1. `static_full_global`: one fixed full-action tuple shared by all ZDT problems.
2. `static_full_per_problem`: one fixed full-action tuple per ZDT problem; this is a strong oracle-style baseline and should be labeled as such in results.

The search space must match Phase 1.5:

- operator in `{polynomial, gaussian}`
- mutation multiplier in `[0.25, 8.0]`
- polynomial `eta_m` in `[2.0, 50.0]`
- gaussian `sigma` in `[0.02, 0.30]`

Use a seeded random search of at least 128 candidate tuples. Evaluate candidates only on designated training/tuning seeds, never test seeds. Rank candidates by mean anytime AUC-HV, with final HV as tie-breaker.

- [ ] **Step 1: Write controller output and validation tests**
- [ ] **Step 2: Verify they fail before implementation**
- [ ] **Step 3: Implement `StaticFullController`**
- [ ] **Step 4: Implement deterministic training-seed-only tuner**
- [ ] **Step 5: Add a test proving test seeds cannot be passed into the tuner when they overlap the declared held-out set**
- [ ] **Step 6: Run tests and commit**

Commit:

```bash
git commit -m "Phase-1.75: Add matched full-action static baselines"
```

---

## Task 3: Build a generation-only open-loop schedule baseline

**Files:**
- Create: `controller/open_loop_controller.py`
- Test: `tests/test_open_loop_controller.py`

**Interfaces:**
- Produces: `OpenLoopScheduleController` with `predict_action(generation: int, max_generations: int, problem_name: str | None = None) -> dict[str, Any]`.

The purpose is to test whether the apparent closed-loop policy can be replaced by a simple annealing schedule that ignores population state.

Build the schedule **from training data/controller actions only**. For normalized generation bins `0.0–0.1, ..., 0.9–1.0`, estimate:

- majority operator
- median mutation multiplier
- median exploration strength conditioned on the selected operator

Provide two variants:

- `open_loop_global`: no problem identity.
- `open_loop_per_problem`: problem identity allowed, but still no population state.

- [ ] **Step 1: Write deterministic binning tests**
- [ ] **Step 2: Run and verify failure**
- [ ] **Step 3: Implement schedule fitting and prediction**
- [ ] **Step 4: Add test that two different histories at the same generation produce identical actions**
- [ ] **Step 5: Run tests and commit**

Commit:

```bash
git commit -m "Phase-1.75: Add open-loop schedule baselines"
```

---

## Task 4: Expand the training trajectory corpus to 500 full-action trajectories

**Files:**
- Modify: `experiments/generate_dataset.py`
- Output: `results/trajectory_phase1_75/` (git ignored)

Generate exactly:

```text
5 ZDT problems × 100 training seeds = 500 trajectories
```

Recommended training seeds: `200..299`. Keep evaluation seeds separate, e.g. `1000..1019`.

Every trajectory must use the Phase-1.5 full random action space and record:

- raw mutation probability
- normalized mutation multiplier
- operator
- exploration strength
- state metrics
- reward metrics
- problem name / n_vars
- seed / runtime / configuration

- [ ] **Step 1: Add CLI support for an explicit seed range and normalized action metadata**
- [ ] **Step 2: Add/extend tests for reproducibility and schema**
- [ ] **Step 3: Run a 2-problem × 2-seed smoke generation and inspect JSON**
- [ ] **Step 4: Generate the 500-trajectory corpus**
- [ ] **Step 5: Write an index summary with success/failure counts by problem**
- [ ] **Step 6: Commit code only; do not commit the large result corpus**

Commit:

```bash
git commit -m "Phase-1.75: Scale full-action trajectory corpus"
```

---

## Task 5: Add deterministic counterfactual one-step branch evaluation

**Files:**
- Modify: `algorithms/nsga2.py`
- Create: `experiments/counterfactual_actions.py`
- Test: `tests/test_counterfactual_snapshot.py`

**Interfaces:**
- Produce a serializable/copyable snapshot object that contains everything required to replay the next generation from the same state: population, objective values/ranks/crowding state if needed, generation index, and RNG state.
- Produce `snapshot_state()` and `restore_state(snapshot)` (or equivalent names, but keep them stable once chosen).

Counterfactual protocol:

1. Sample at least 200 evaluation states per ZDT problem from held-out trajectories, spread across early/mid/late generations.
2. At each state, branch from the exact same snapshot.
3. Evaluate the controller-selected action plus at least 19 alternative full actions.
4. For each action, use at least 5 derived replicate RNG seeds so action ranking is not determined by a single stochastic draw.
5. Score one-step reward as `delta_hv + delta_igd`; also store both components separately.
6. Record the percentile rank of the controller-selected action among the candidate actions.

Primary causal statistic:

```text
mean controller-action percentile rank
```

A value near 0.5 means no better than candidate actions; values clearly above 0.5 support state-conditioned action quality.

- [ ] **Step 1: Write a snapshot replay test showing identical snapshot + identical branch seed + identical action => identical next population/metrics**
- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement snapshot/restore with deep copies and RNG state preservation**
- [ ] **Step 4: Implement counterfactual branch evaluator**
- [ ] **Step 5: Run a 10-state smoke experiment and verify deterministic output**
- [ ] **Step 6: Run the full counterfactual experiment and save `results/phase1_75/counterfactual.json`**
- [ ] **Step 7: Commit**

Commit:

```bash
git commit -m "Phase-1.75: Add counterfactual action evaluation"
```

---

## Task 6: Run the fair closed-loop evaluation with 20 held-out seeds

**Files:**
- Create: `experiments/run_phase1_75.py`

Use held-out seeds `1000..1019` for every problem and every arm. Use identical seeds across arms for paired statistics.

Required arms:

1. `fixed_nsga2`
2. `static_full_global`
3. `static_full_per_problem`
4. `open_loop_global`
5. `open_loop_per_problem`
6. `mlp2_closed_loop_absolute` — old absolute-pm target, same enlarged training data
7. `mlp2_closed_loop_normalized` — normalized multiplier target
8. `generation_only_mlp` — generation + optional problem identity, no population state
9. `state_scrambled_mlp` — same controller architecture/training budget, but state histories are trajectory-shuffled within the training split

All learned arms must use the same 500-trajectory corpus, same train/validation split, same hidden widths, optimizer, epochs, and random seed unless the ablation itself requires otherwise.

Record for every run:

- final HV
- final IGD
- anytime AUC-HV
- runtime
- failure indicator
- full action trajectory

Define a problem-specific failure threshold **before** inspecting test-arm results. Prefer a threshold derived from the training distribution (e.g. lower decile of fixed baseline training HV) and write the derived values into the config artifact.

- [ ] **Step 1: Implement arm factory and shared config**
- [ ] **Step 2: Add tests that all arms receive the same problem/seed pairs**
- [ ] **Step 3: Run a one-problem × two-seed smoke experiment**
- [ ] **Step 4: Run the full 5 × 20 paired evaluation**
- [ ] **Step 5: Save raw results to `results/phase1_75/results.json`**
- [ ] **Step 6: Commit experiment code**

Commit:

```bash
git commit -m "Phase-1.75: Run fair closed-loop baseline evaluation"
```

---

## Task 7: Add paper-grade statistical analysis

**Files:**
- Create: `experiments/analyze_phase1_75.py`
- Test: `tests/test_phase1_75_statistics.py`

For each problem and primary metric, compute:

- mean ± std
- median
- paired median difference vs each primary baseline
- 95% paired bootstrap confidence interval (at least 10,000 bootstrap resamples, deterministic seed)
- paired Wilcoxon signed-rank p-value
- Holm correction across the five ZDT problems for each named comparison
- failure rate

Primary comparisons must be declared in code/config before analysis:

1. `mlp2_closed_loop_normalized` vs `static_full_global`
2. `mlp2_closed_loop_normalized` vs `open_loop_global`
3. `mlp2_closed_loop_normalized` vs `generation_only_mlp`
4. `mlp2_closed_loop_normalized` vs `state_scrambled_mlp`

Treat `static_full_per_problem` and `open_loop_per_problem` as strong secondary/oracle-style references, not as the only success gate.

- [ ] **Step 1: Write unit tests for Holm correction and paired bootstrap on synthetic arrays with known ordering**
- [ ] **Step 2: Verify failure**
- [ ] **Step 3: Implement statistics utilities**
- [ ] **Step 4: Generate Markdown/JSON summary tables from the raw result file**
- [ ] **Step 5: Commit**

Commit:

```bash
git commit -m "Phase-1.75: Add paper-grade statistical analysis"
```

---

## Task 8: Scientific interpretation and Phase-2 gate

**Files:**
- Create: `docs/PHASE1_75_RESULTS.md`
- Modify: `README.md`

The results document must explicitly answer all four questions below without overstating evidence:

1. **Matched action-space:** Does the closed-loop controller beat a fixed policy that has access to operator + pm + exploration?
2. **Feedback vs schedule:** Does closed-loop state feedback beat a generation-only open-loop schedule?
3. **Causality:** Does the controller choose above-random actions in counterfactual one-step branch evaluation?
4. **Scale confound:** Do conclusions persist when predicting normalized mutation multiplier instead of absolute pm?

### Phase-2 go/no-go criteria

Proceed to Phase 2 only if **all** of the following are true:

- Closed-loop normalized controller beats `static_full_global` and `open_loop_global` on at least 3/5 problems after Holm correction, with no statistically significant regression on the remaining problems.
- It beats `generation_only_mlp` or `state_scrambled_mlp` on at least 3/5 problems in final HV or anytime AUC-HV.
- Mean counterfactual controller-action percentile rank is > 0.55 overall, and its 95% CI excludes 0.50.
- Normalized mutation-multiplier modeling does not destroy the Phase-1.5 gains; performance is competitive with or better than the absolute-pm model.
- The 500-trajectory dataset materially reduces the train/validation generalization gap relative to the 125-trajectory Phase-1.5 result, or the remaining gap is explicitly quantified and addressed.

If any gate fails, do **not** implement Mamba. Record the failed criterion and propose a focused corrective experiment.

- [ ] **Step 1: Write `docs/PHASE1_75_RESULTS.md` from saved artifacts**
- [ ] **Step 2: Update README roadmap and concise conclusion**
- [ ] **Step 3: Run `python -m pytest` and verify all tests pass**
- [ ] **Step 4: Commit**

Commit:

```bash
git add docs/PHASE1_75_RESULTS.md README.md
git commit -m "Phase-1.75: Complete decision-causality validation"
```

---

## Expected scientific outcome

Phase 1.5 already shows that adaptive actions correlate with state and can improve optimization. Phase 1.75 is intentionally harder: it asks whether the **feedback loop itself** is necessary. A positive result makes the Phase-2 sequence-model story defensible. A negative result is also valuable because it prevents us from spending GPU time on Mamba when a static or generation-only controller explains the gains.
