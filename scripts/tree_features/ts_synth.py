"""Candidate 2, planted: a high-cardinality categorical whose effects partly
drift, with new categories born over time. Seeds 0,1,2."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import TargetEncoder


def make(seed, n=300_000, levels=20_000):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    av = t + 0.5
    # Zipf ranks, and a population that shifts: a row's category id is rank + birth offset
    r = np.minimum(rng.zipf(1.3, n) - 1, levels - 1)
    shift = (t / n * levels * 0.5).astype(np.int64)  # half the population replaced over the run
    k = r + shift
    nk = k.max() + 1
    e0 = rng.standard_normal(nk)
    drifting = rng.random(nk) < 0.5
    # a drift: a smooth sine with its own phase and a period of ~n/2 rows
    ph = rng.uniform(0, 2 * np.pi, nk)
    eff = e0[k] + np.where(drifting[k], 1.0 * np.sin(2 * np.pi * t / (n / 2) + ph[k]), 0.0)
    x = rng.standard_normal((n, 3))
    y = eff + np.sin(2 * x[:, 0]) + x[:, 1] + rng.standard_normal(n)
    return t, av, k, x, y, eff


def fit_eval(X, y, tr, va, te, cats):
    p = dict(
        objective="l2",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
        cat_smooth=10,
        max_cat_to_onehot=4,
    )
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    return float(np.sqrt(np.mean((pr - y[te]) ** 2)))


out = {}
print(C.load_line())
for seed in (0, 1, 2):
    t, av, k, x, y, eff = make(seed)
    n = len(t)
    tr = t < 0.6 * n
    va = (t >= 0.6 * n) & (t < 0.7 * n)
    te = t >= 0.7 * n
    cut = 0.6 * n
    print(
        f"seed {seed}: levels seen {len(np.unique(k))}, test rows whose level is unseen in train {np.mean(~np.isin(k[te], k[tr])):.3f}; noise floor RMSE 1.0"  # noqa: E501
    )
    row = {}
    row["dropped"] = fit_eval(x, y, tr, va, te, [])
    row["native"] = fit_eval(np.column_stack([x, k.astype(float)]), y, tr, va, te, [3])
    m = y[tr].mean()
    s = np.bincount(k[tr], weights=y[tr], minlength=k.max() + 1)
    c = np.bincount(k[tr], minlength=k.max() + 1)
    row["greedy"] = fit_eval(np.column_stack([x, (s[k] + 10 * m) / (c[k] + 10)]), y, tr, va, te, [])
    enc = TargetEncoder(target_type="continuous", cv=5)
    Xs = np.empty(n)
    Xs[tr] = enc.fit_transform(k[tr, None], y[tr])[:, 0]
    Xs[~tr] = enc.transform(k[~tr, None])[:, 0]
    row["sklearn_crossfit"] = fit_eval(np.column_stack([x, Xs]), y, tr, va, te, [])
    for hl, tag in ((math.inf, "inf"), (20_000.0, "20k")):
        for fz in (cut, None):
            e, w = C.ordered_ts(t, av, k, y, half_life=hl, prior_weight=3.0, freeze_at=fz)
            row[f"ordered_{tag}_{'frozen' if fz else 'cont'}"] = fit_eval(
                np.column_stack([x, e, w]), y, tr, va, te, []
            )
    # bounded memory, decayed, continuing: cap vs count-min at equal doubles
    for K in (512, 4096):
        e, w = C.ordered_ts(t, av, k, y, half_life=20_000.0, prior_weight=3.0, cap=K)
        row[f"cap{K}"] = fit_eval(np.column_stack([x, e, w]), y, tr, va, te, [])
        e, w = C.ordered_ts(t, av, k, y, half_life=20_000.0, prior_weight=3.0, cms=(4, K // 4))
        row[f"cms4x{K // 4}"] = fit_eval(np.column_stack([x, e, w]), y, tr, va, te, [])
        e, w = C.ordered_ts(t, av, k, y, half_life=20_000.0, prior_weight=3.0, cms=(1, K))
        row[f"hash1x{K}"] = fit_eval(np.column_stack([x, e, w]), y, tr, va, te, [])
    for kk, v in row.items():
        print(f"  {kk:28s} test RMSE {v:.4f}", flush=True)
    out[seed] = row
Path(str(C.OUT / "ts_synth.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(C.load_line())
