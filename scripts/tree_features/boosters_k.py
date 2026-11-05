"""v3, review findings 12 and 14b: where refitting leaves stops working, and
an online combination of the last K retrained boosters.

Two planted drifts, 300,000 rows on 3,000 clock steps, seeds 0-2:
- magnitudes: leafrefit.py's synth, the shapes' weights drift and the
  thresholds stay put (the best case for refitting leaves);
- structure: y = sin(2 x0) + 1.5 [x1 > a(s)] + 1.5 x2 [x3 > b(s)] + noise,
  the thresholds moving, a from 0.5 to -1.0 and b from -1.0 to 1.0.
Arms, scored on the second half: the trees trained on the first 40%
(frozen); sgd (adagrad, rate 0.05) on their one-hot leaves; a booster
retrained every 150 steps on the last 300 and served alone; and ewridge, no
intercept, half-life 20 steps, over the last K = 5 retrained boosters'
predictions, one column per age, each booster scoring only steps after its
own training data."""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from leafrefit import bank, leaf_values, rmse, synth, train_trees  # noqa: E402

import polars_online as po  # noqa: E402


def structural(seed, steps=3000, per=100):
    rng = np.random.default_rng(seed)
    n = steps * per
    clock = np.repeat(np.arange(steps, dtype=float), per)
    s = clock / steps
    x = rng.standard_normal((n, 5))
    a = 0.5 - 1.5 * s
    b = -1.0 + 2.0 * s
    y = (
        np.sin(2 * x[:, 0])
        + 1.5 * (x[:, 1] > a)
        + 1.5 * x[:, 2] * (x[:, 3] > b)
        + rng.standard_normal(n)
    )
    return clock, x, y


def booster(X, y, rows):
    p = dict(
        objective="l2",
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=100,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
    )
    return lgb.train(p, lgb.Dataset(X[rows], y[rows]), 100)


def arms(clock, X, y):
    S = clock.max() + 1
    tr, va, te = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S), clock >= 0.5 * S
    out = {}
    b = train_trees(X, y, tr, va, 15, 300)
    base = b.predict(X, num_iteration=b.best_iteration)
    out["frozen trees"] = rmse(base[te], y[te])
    leaves, _ = leaf_values(b, X)
    oh = {
        f"o{m}_{lf}": (leaves[:, m] == lf).astype(float)
        for m in range(leaves.shape[1])
        for lf in range(int(leaves[:, m].max()) + 1)
    }

    def sg(cols):
        return po.spec.sgd(
            "o",
            targets=["y"],
            features=cols,
            clock="c",
            gap_cap=10.0,
            half_life=20.0,
            schedule="adagrad",
            learning_rate=0.05,
            standardize=False,
        )

    p = bank(clock, oh, y, sg)
    out["sgd on one-hot leaves"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
    # retrains: one booster per 150 steps from 0.4 S on, trained on the 300 steps before its start
    K, every, window = 5, 150, 300
    starts = list(range(int(0.4 * S), int(S), every))
    preds = []
    for s0 in starts:
        bb = booster(X, y, (clock >= s0 - window) & (clock < s0))
        preds.append((s0, bb.predict(X)))
    latest = np.full(len(y), np.nan)
    ages = np.full((K, len(y)), np.nan)
    for i, (s0, pr) in enumerate(preds):
        rows = (clock >= s0) & (clock < s0 + every)
        latest[rows] = pr[rows]
        for age in range(K):
            if i - age >= 0:
                ages[age, rows] = preds[i - age][1][rows]
    out["the latest retrained booster"] = rmse(latest[te], y[te])
    ok = clock >= 0.4 * S
    cols = {f"age{k}": np.where(np.isnan(ages[k]), latest, ages[k])[ok] for k in range(K)}

    def ew(cs):
        return po.spec.ewridge(
            "o",
            targets=["y"],
            features=cs,
            fit_intercept=False,
            clock="c",
            gap_cap=10.0,
            half_life=20.0,
            ridge=1e-3,
        )

    p = np.full(len(y), np.nan)
    p[ok] = bank(clock[ok], cols, y[ok], ew)
    out[f"ewridge over the last {K} boosters"] = rmse(
        np.where(np.isnan(p[te]), latest[te], p[te]), y[te]
    )
    return out


print(C.load_line(), flush=True)
res = {}
for kind, make in (("magnitudes", synth), ("structure", structural)):
    for seed in (0, 1, 2):
        clock, X, y = make(seed)
        r = arms(clock, X, y)
        res[f"{kind} {seed}"] = r
        print(f"{kind} seed {seed}: " + ", ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
(C.OUT / "boosters_k.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
