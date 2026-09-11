# Phase 2B External Review Verification

Date: 2026-09-11

## Scope

This note verifies the external review against the committed `main` branch only. Git-ignored local results, file mtimes, and currently running background processes cannot be verified from GitHub.

## Repository state correction

The current committed head is `fe12dc5` (`Phase-2B: Complete planning controller with tests and B1 evaluation runner`), which is newer than Experiment-A commit `e148996`.

## Verified findings

### 1. Documentation drift is real

- `README.md` still says Phase 2B core implementation is complete but the experiment is "待运行". If local B1 is already running, this is stale.
- `Phase 1.75 | ❌ 已完成` is semantically understandable but visually ambiguous. Prefer `⚠️ 已完成（gate 未通过）`.
- The README tree omits several current result directories/modules.
- `docs/PHASE2_PLAN.md` is the old Transformer/GRU plan and is not marked superseded by `PHASE2_REDESIGN_PLAN.md`.
- `AGENTS.md` still shows the old Phase 0→4 roadmap and does not describe Phase 1.5 / 1.75 / 2A / 2B.

### 2. Candidate-set seeding concern is real, but the interpretation needs precision

`PlanningController.predict_action()` seeds the candidate generator with:

```python
np.random.PCG64([candidate_seed, len(history)])
```

Therefore runs at the same generation share the same random candidate stream. They share operator choices, exploration draws, and normalized mutation multipliers. Absolute mutation probabilities still scale with `1 / n_vars`, so problems with different dimensionality are not literally identical in absolute pm.

This does not prove that the planner is harmed, but it reduces candidate-set diversity across evaluation runs and can create correlated decisions. Do **not** change this while the current B1 run is in progress; finish and aggregate the current protocol first. A later robustness run should mix `(problem, run seed, generation)` into candidate sampling while retaining deterministic reproducibility.

### 3. Phase-2B multiple-testing correction is missing

`experiments/run_phase2b.py` currently reports raw one-sided paired Wilcoxon tests. It does not apply the Holm-Bonferroni correction already implemented in `experiments/analyze_phase1_75.py`.

For paper reporting, use a pre-declared family. The natural continuation of Phase 1.75 is Holm correction across the five problems separately for each `(metric, comparison-arm)` family, rather than treating all tests as unrelated raw p-values.

Failure rate should also be reported beside mean±std because difficult problems such as ZDT4 can be poorly summarized by means alone.

### 4. The biggest issue: current B2 is not aligned with the planner objective

The Phase-2B planner scores actions using predicted **long-horizon absolute HV** with default horizon weights over `[1, 5, 10, 20]`.

However `experiments/counterfactual_actions.py` is still the Phase-1.75 **one-step** counterfactual evaluator: it branches one generation and scores `delta_hv + delta_igd`.

Therefore a local B2 result such as a percentile rank around 0.50 is evidence that the planner is not especially good at the **one-step Phase-1.75 reward**. It is **not yet a valid refutation of long-horizon planning**, because that is not what the planner optimizes.

Phase 2B's own plan asks for `future HV gain`. B2 should therefore be extended to branch each candidate for horizons matching the predictor (at minimum 5/10/20 generations, with controlled replicated RNG), then measure:

- predicted-vs-real within-state action ranking,
- percentile rank of the planner-selected action,
- oracle regret,
- realized future HV gain,
- early/mid/late breakdown.

### 5. Prediction quality is not yet decision quality

Experiment A's high global R² can be dominated by state/history information. It does not prove that the predictor has learned the *action-dependent* part of the transition.

The suggested "candidate score spread vs model RMSE" diagnostic is useful but insufficient by itself. Add stronger action-conditional tests:

1. **Action ablation/shuffle**: evaluate the predictor after shuffling/zeroing action features while preserving state history. If R² barely changes, the model mostly predicts from state.
2. **Within-state ranking**: for the same snapshot and many candidate actions, compute Spearman/Kendall correlation between predicted long-horizon score and realized long-horizon outcome.
3. **Top-1 regret**: compare predictor-selected action against the oracle best candidate on the same state.
4. **Action-effect SNR**: compare between-action outcome variance against replicate stochastic noise.
5. Prefer predicting an action advantage / future-HV delta relative to a state-only baseline if absolute-HV prediction is dominated by state identity.

### 6. Candidate diagnostics should be logged

Current `PlanningController.predict_action()` returns only the chosen action. The per-generation run artifact therefore cannot reconstruct:

- which 16 candidates were considered,
- their predicted horizon vectors,
- weighted scores,
- score margin between top-1/top-2,
- whether failures come from poor candidates or poor ranking.

Add a diagnostics path (without changing the stable action-return contract) and persist candidate-level scores for Phase 2B analysis.

### 7. Phase-2B counterfactual outputs should not reuse Phase-1.75 output defaults

`experiments/counterfactual_actions.py` still defaults to:

```text
results/phase1_75
```

and writes `counterfactual_{problem}.json`. Running Phase-2B planning counterfactuals with the default output directory can overwrite Phase-1.75 artifacts. New Phase-2B outputs should live under e.g.:

```text
results/phase2b/counterfactual/
```

Snapshots may be reused read-only if desired, but result artifacts must be stage-isolated.

## Not verifiable from GitHub

The following claims may be true from a local-workspace inspection, but GitHub cannot confirm them because `results/` is ignored and process state is local:

- exact progress such as `83/100` B1 runs,
- three currently running shards,
- file mtimes,
- the reported ZDT1 B2 rank `0.494`, regret/stake/noise values,
- whether a particular local Phase-1.75 result was already overwritten.

These should be checked directly in the execution workspace before acting on exact numbers.

## Recommended order

1. **Do not modify the protocol mid-run.** Finish the current B1 shards, then run `--stage aggregate` only.
2. Preserve/copy current artifacts before any B2 rerun; move Phase-2B counterfactual outputs to their own directory.
3. Add Holm-adjusted statistics and failure-rate reporting to the aggregate analysis.
4. Redesign B2 to evaluate the same long horizons the planner optimizes; keep the old one-step result as a secondary diagnostic, not the primary causal test.
5. Add candidate-level prediction logs and action-conditional diagnostics.
6. After the fixed protocol is frozen, run a robustness experiment with run/problem-aware candidate seeds and possibly larger candidate counts (e.g. 16/64/256) to separate candidate-set coverage from ranking quality.
7. Update README, mark `PHASE2_PLAN.md` superseded, and refresh `AGENTS.md` after the experiment status is settled.

## Planner verdict

The external review is **mostly technically correct**, especially on documentation drift, shared candidate sampling, missing multiple-testing correction, action-score logging, and output-path isolation. The main correction is that the reported one-step B2 rank cannot by itself establish failure of the long-horizon planner, because the current B2 evaluator is misaligned with the planner's long-horizon objective. That mismatch should be fixed before making a Phase-2B go/no-go decision.
