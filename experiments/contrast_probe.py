"""Extreme action contrast probe: does pairwise identifiability rise with contrast?"""
import json, sys, glob, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('.').resolve()))

D = Path('results/phase2_75d/dataset_final/state_mean')
X, y, hz, kind, st = [], [], [], [], []
for f in sorted(D.glob('intervention_dataset_*.npz')):
    d = np.load(f, allow_pickle=True)
    X.append(d['X']); y.append(d['y_adv']); hz.append(d['horizon']); kind.append(d['candidate_kind'])
    st += [f"{d['problem'][i]}|{d['seed'][i]}|{d['generation'][i]}" for i in range(len(d['X']))]
X = np.concatenate(X); y = np.concatenate(y); hz = np.concatenate(hz)
kind = np.concatenate(kind); st = np.array(st)

H = 20
groups = collections.defaultdict(list)
for i in np.flatnonzero(hz == H):
    if kind[i] != 'controller':
        groups[st[i]].append(int(i))

# action features = last 4 cols: [multiplier, exploration, poly_onehot, gauss_onehot]
def contrast(a, b):
    """Log-scale distance in (multiplier, exploration) + operator mismatch penalty."""
    xa, xb = X[a, -4:], X[b, -4:]
    dm = abs(np.log(max(xa[0], 1e-6)) - np.log(max(xb[0], 1e-6)))
    de = abs(np.log(max(xa[1], 1e-6)) - np.log(max(xb[1], 1e-6)))
    dop = 1.0 if xa[2] != xb[2] else 0.0
    return float(np.hypot(dm, de) + dop)

rows = []
for key, idx in groups.items():
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            a, b = idx[i], idx[j]
            if y[a] == y[b]:
                continue
            rows.append((contrast(a, b), 1.0 if y[a] > y[b] else 0.0, abs(y[a]-y[b])))
rows = np.array(rows)
print(f'h={H} 状态内不同结果的配对: n={len(rows)}')

# AUC of a trivial "prefer larger multiplier+exploration" rule, by contrast decile
from sklearn.metrics import roc_auc_score
qs = np.quantile(rows[:,0], [0, .2, .4, .6, .8, 1.0])
print(f"\n{'contrast 区间':>22}{'n':>8}{'|Δy| 中位':>12}{'一致性':>10}")
for k in range(5):
    lo, hi = qs[k], qs[k+1]
    sel = (rows[:,0] >= lo) & (rows[:,0] <= hi if k == 4 else rows[:,0] < hi)
    if sel.sum() < 10: continue
    # 无方向先验：用多数类基线（0.5）与 |Δy| 作为"可区分幅度"的代理
    print(f'[{lo:6.3f},{hi:6.3f}){"":>4}{sel.sum():>8}{np.median(rows[sel,2]):>12.5f}{"":>10}')
print()
print('|Δy| 随对比度单调性（Spearman）:', end=' ')
from scipy import stats
print(round(float(stats.spearmanr(rows[:,0], rows[:,2]).statistic), 4))
