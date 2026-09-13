"""Experiment A on existing data: does pairwise AUC rise with action contrast?

Phase 3.1 Experiment A proposes re-running rollouts with deliberately extreme
action contrasts (low vs high exploration, weak vs strong mutation). Before
spending that compute, this script asks the same question of the existing
1200-state intervention data: restrict the pairwise preference task to
high-contrast action pairs and see whether held-out AUC improves.

If AUC is flat across contrast strata, stronger intervention separation cannot
restore learnability and Experiment A is unnecessary.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

D = Path("results/phase2_75d/dataset_final/state_mean")
SPLIT = json.loads(
    Path("results/phase2_75d/target_comparison_final.json").read_text(encoding="utf-8")
)["split"]
VAL = set(SPLIT["val_state_keys"])

X, y, hz, kind, st = [], [], [], [], []
for f in sorted(D.glob("intervention_dataset_*.npz")):
    d = np.load(f, allow_pickle=True)
    X.append(d["X"]); y.append(d["y_adv"]); hz.append(d["horizon"]); kind.append(d["candidate_kind"])
    st += [f"{d['problem'][i]}|{d['seed'][i]}|{d['generation'][i]}" for i in range(len(d["X"]))]
X = np.concatenate(X); y = np.concatenate(y); hz = np.concatenate(hz)
kind = np.concatenate(kind); st = np.array(st)


def contrast(a: int, b: int) -> float:
    xa, xb = X[a, -4:], X[b, -4:]
    dm = abs(np.log(max(xa[0], 1e-6)) - np.log(max(xb[0], 1e-6)))
    de = abs(np.log(max(xa[1], 1e-6)) - np.log(max(xb[1], 1e-6)))
    return float(np.hypot(dm, de) + (1.0 if xa[2] != xb[2] else 0.0))


report: dict[str, object] = {"config": {"horizon": 20, "n_states": len(VAL)}}
for H in (5, 10, 20):
    groups: dict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(hz == H):
        if kind[i] != "controller":
            groups[st[i]].append(int(i))
    tr_d, tr_l, va_d, va_l, va_c = [], [], [], [], []
    for key, idx in groups.items():
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                if y[a] == y[b]:
                    continue
                d = X[a, -4:] - X[b, -4:]
                lab = 1.0 if y[a] > y[b] else 0.0
                c = contrast(a, b)
                if key in VAL:
                    va_d.append(d); va_l.append(lab); va_c.append(c)
                else:
                    tr_d.append(d); tr_l.append(lab)
    tr_d = np.asarray(tr_d); tr_l = np.asarray(tr_l)
    va_d = np.asarray(va_d); va_l = np.asarray(va_l); va_c = np.asarray(va_c)
    if tr_d.size == 0 or va_d.size == 0:
        report[f"h{H}"] = {"n_train": int(tr_d.size), "n_val": int(va_d.size)}
        continue
    entry: dict[str, object] = {"n_train_pairs": int(tr_d.shape[0]), "n_val_pairs": int(va_d.shape[0]), "strata": {}}
    edges = np.quantile(va_c, [0, 0.25, 0.5, 0.75, 1.0])
    for name, model in (
        ("logistic", LogisticRegression(max_iter=1000)),
        ("random_forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    ):
        model.fit(tr_d, tr_l)
        p_all = model.predict_proba(va_d)[:, 1]
        entry[f"auc_all_{name}"] = float(roc_auc_score(va_l, p_all))
        for k in range(4):
            lo, hi = edges[k], edges[k + 1]
            sel = (va_c >= lo) & (va_c <= hi if k == 3 else va_c < hi)
            if sel.sum() < 50 or len(set(va_l[sel])) < 2:
                continue
            entry["strata"][f"q{k+1}_{lo:.2f}_{hi:.2f}"] = {
                "n": int(sel.sum()),
                "auc": float(roc_auc_score(va_l[sel], p_all[sel])),
                "median_contrast": float(np.median(va_c[sel])),
                "median_abs_dy": float(np.median(np.abs(tr_l[:0])) if False else 0.0),
            }
        # train on high-contrast pairs only, test on high-contrast pairs only
        thr = np.quantile(va_c, 0.75)
        ctr = np.asarray([contrast(groups[list(groups)[0]][0], groups[list(groups)[0]][1])]) if False else None
        entry[f"auc_top_quartile_{name}"] = entry["strata"].get(
            f"q4_{edges[3]:.2f}_{edges[4]:.2f}", {}
        ).get("auc")
    report[f"h{H}"] = entry

out = Path("results/phase2_9/extreme_contrast_experimentA.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[done] wrote {out}\n")
for H in (5, 10, 20):
    e = report[f"h{H}"]
    if "strata" not in e:
        print(f"h={H}: {e}"); continue
    print(f"h={H}: AUC(all)={e['auc_all_logistic']:.4f}(log) {e['auc_all_random_forest']:.4f}(rf)")
    for k, v in sorted(e["strata"].items()):
        print(f"    {k:22} n={v['n']:>6} auc={v['auc']:.4f} median_contrast={v['median_contrast']:.2f}")
