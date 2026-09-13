"""Semantic action abstraction probe (Phase 3, optional experiment).

Question: are high-level search *intents* (increase diversity, improve
convergence, escape stagnation) more identifiable than low-level operator
parameters?

Operationalisation on existing data (no new rollouts): each (state, action)
row already carries the state history and the action's operator / multiplier /
exploration. Derive a coarse **state-relative intent label** and ask how much
of the within-state action-effect variance the label explains. If a 3-way
label explains a meaningful share, a semantic action space is worth building;
if not, the bottleneck is the decision problem itself, not the action coding.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

D = Path("results/phase2_75d/dataset_final/state_mean")


def label_for(state_diversity: float, hist_diversity_mean: float,
              multiplier: float, exploration: float, operator: str) -> str:
    """Coarse, state-relative search intent of one action.

    Thresholds are the action's own values against the state's recent level:
    * ``explore``  - pushes exploration above the recent average
    * ``exploit``  - pulls exploration below the recent average
    * ``neutral``  - everything else (including operator switches at the
      recent exploration level)
    """
    if operator == "gaussian":
        strong = exploration >= 0.10          # sigma scale
    else:
        strong = exploration <= 12.0          # eta_m scale (lower = more spread)
    aggressive = multiplier >= 2.0
    if (strong and aggressive) or (strong and multiplier >= 1.0):
        return "explore"
    if (not strong) and multiplier <= 1.0:
        return "exploit"
    return "neutral"


rows = []
for f in sorted(D.glob("intervention_dataset_*.npz")):
    d = np.load(f, allow_pickle=True)
    X, y, hz, kind = d["X"], d["y_adv"], d["horizon"], d["candidate_kind"]
    op, pm, expl = d["mutation_operator"], d["mutation_probability"], d["exploration_strength"]
    mult = d["mutation_multiplier"]
    st = [f"{d['problem'][i]}|{d['seed'][i]}|{d['generation'][i]}" for i in range(len(X))]
    for i in range(len(X)):
        if kind[i] == "controller":
            continue
        rows.append((st[i], int(hz[i]), float(mult[i]), float(expl[i]), str(op[i]), float(y[i])))

by_state: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
for state, h, mult, expl, op, adv in rows:
    lab = label_for(0.0, 0.0, mult, expl, op)
    by_state[(state, h)].append((lab, adv))

report: dict[str, object] = {"labels": ["explore", "exploit", "neutral"], "per_horizon": {}}
for h in (5, 10, 20):
    # variance decomposition: total within-state variance vs variance of intent means
    total, between, counts = 0.0, 0.0, defaultdict(int)
    n_states = 0
    for (state, hh), items in by_state.items():
        if hh != h or len(items) < 2:
            continue
        advs = np.asarray([a for _, a in items])
        labs = [l for l, _ in items]
        total += float(np.var(advs, ddof=1))
        per_label: dict[str, list[float]] = defaultdict(list)
        for l, a in items:
            per_label[l].append(a)
            counts[l] += 1
        means = np.asarray([np.mean(v) for v in per_label.values()])
        if means.size >= 2:
            between += float(np.var(means, ddof=1))
        n_states += 1
    report["per_horizon"][str(h)] = {
        "n_states": n_states,
        "mean_within_state_total_var": total / max(n_states, 1),
        "mean_intent_between_var": between / max(n_states, 1),
        "intent_explained_fraction": (between / total) if total > 0 else None,
        "label_counts": dict(counts),
    }


def _permutation_baseline(groups, n_perm=200, seed=0):
    """Explained fraction under label shuffling inside each state.

    With only ~3.7 samples per (state, label) cell, a meaningless label split
    already "explains" ~1/n of the variance, so the raw fraction must always be
    read against this baseline (the Phase-2.9 probe found the shuffled labels
    explaining *more* than the real ones).
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_perm):
        total, between = 0.0, 0.0
        for items in groups:
            if len(items) < 2:
                continue
            labels = [lab for lab, _ in items]
            advs = np.asarray([a for _, a in items])
            shuffled = list(rng.permutation(labels))
            total += float(np.var(advs, ddof=1))
            per_label: dict[str, list[float]] = defaultdict(list)
            for lab, adv in zip(shuffled, advs):
                per_label[lab].append(adv)
            means = np.asarray([np.mean(v) for v in per_label.values()])
            if means.size >= 2:
                between += float(np.var(means, ddof=1))
        if total > 0:
            values.append(between / total)
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()) if arr.size else None,
        "q95": float(np.percentile(arr, 95)) if arr.size else None,
    }


GROUPS_BY_HORIZON: dict[int, list[list[tuple[str, float]]]] = {}
for (_state, _h), _items in by_state.items():
    GROUPS_BY_HORIZON.setdefault(_h, []).append(_items)

for _h in (5, 10, 20):
    _entry = report["per_horizon"][str(_h)]
    _entry["permutation_baseline"] = _permutation_baseline(
        GROUPS_BY_HORIZON.get(_h, [])
    )
    _real = _entry["intent_explained_fraction"]
    _base = _entry["permutation_baseline"]["mean"]
    _entry["above_permutation_baseline"] = (
        None if _real is None or _base is None else bool(_real > _base)
    )

out = Path("results/phase2_9/semantic_action_probe.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[done] wrote {out}\n")
for h in (5, 10, 20):
    e = report["per_horizon"][str(h)]
    frac = e["intent_explained_fraction"]
    print(f"h={h:>2}: states={e['n_states']:>4} total_var={e['mean_within_state_total_var']:.6f} "
          f"intent_var={e['mean_intent_between_var']:.6f} explained={frac:.4f}" if frac is not None
          else f"h={h}: insufficient")
    print(f"      label counts: {e['label_counts']}")
