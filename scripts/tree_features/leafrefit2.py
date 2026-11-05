"""Candidate 5, cheaper arms and the batch alternative: sgd (adagrad, O(width)
a row) on one-hot leaves and on leaf values, and LightGBM's own refit
(decay_rate) run every 50 clock steps on the last 50 steps' rows. Planted
drift, seeds 0-2."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from leafrefit import bank, leaf_values, rmse, synth, train_trees  # noqa: E402

import polars_online as po  # noqa: E402

print(C.load_line())
res = {}
for seed in (0, 1, 2):
    clock, X, y = synth(seed)
    S = clock.max() + 1
    tr, va, te = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S), clock >= 0.5 * S
    b = train_trees(X, y, tr, va, 15, 300)
    base = b.predict(X, num_iteration=b.best_iteration)
    leaves, vals = leaf_values(b, X)
    M = vals.shape[1]
    oh = {
        f"o{m}_{l}": (leaves[:, m] == l).astype(float)
        for m in range(M)
        for l in range(int(leaves[:, m].max()) + 1)  # noqa: E741
    }
    r = {"frozen": rmse(base[te], y[te])}
    for lr in (0.05, 0.2):
        sg = lambda cols, lr=lr: po.spec.sgd(  # noqa: E731
            "o",
            targets=["y"],
            features=cols,
            clock="c",
            gap_cap=10.0,
            half_life=20.0,
            schedule="adagrad",
            learning_rate=lr,
            standardize=False,
        )
        t0 = time.perf_counter()
        p = bank(clock, oh, y, sg)
        r[f"sgd adagrad lr={lr} one-hot"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
        r[f"t one-hot lr={lr}"] = time.perf_counter() - t0
        p = bank(clock, {f"v{m}": vals[:, m] for m in range(M)}, y, sg)
        r[f"sgd adagrad lr={lr} leaf values"] = rmse(
            np.where(np.isnan(p[te]), base[te], p[te]), y[te]
        )
    # LightGBM refit every 50 steps on the last 50 steps
    pred = base.copy()
    cur = b
    for s0 in range(int(0.5 * S), int(S), 50):
        win = (clock >= s0 - 50) & (clock < s0)
        cur = cur.refit(X[win], y[win], decay_rate=0.5)
        nxt = (clock >= s0) & (clock < s0 + 50)
        pred[nxt] = cur.predict(X[nxt])
    r["lgb refit every 50 steps, decay 0.5"] = rmse(pred[te], y[te])
    res[seed] = r
    print(seed, json.dumps(r), flush=True)
Path(str(C.OUT / "leafrefit2.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
