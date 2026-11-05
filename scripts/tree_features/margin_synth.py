"""Candidate 1, planted: a common level that random-walks, which no feature
carries, under a nonlinear f(x). Gaussian and Bernoulli targets; seeds 0-2."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import lightgbm as lgb
import numpy as np
import polars as pl

import polars_online as po


def make(seed, steps=2000, per=100):
    rng = np.random.default_rng(seed)
    n = steps * per
    clock = np.repeat(np.arange(steps, dtype=float), per)
    L = np.cumsum(rng.normal(0, 0.08, steps))[clock.astype(int)]
    x = rng.standard_normal((n, 4))
    f = np.sin(2 * x[:, 0]) + (x[:, 1] > 0.5) + 0.5 * x[:, 2] * x[:, 3]
    yg = L + f + rng.standard_normal(n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * L + f - 0.5)))).astype(float)
    return clock, x, yg, yb, L


def online(clock, x, y, kind):
    d = pl.DataFrame({"c": clock, **{f"x{i}": x[:, i] for i in range(4)}, "y": y})
    if kind == "gauss":
        s = po.spec.ewridge(
            "o",
            targets=["y"],
            features=[f"x{i}" for i in range(4)],
            clock="c",
            gap_cap=10.0,
            half_life=5.0,
            ridge=1e-3,
        )
    else:
        s = po.spec.ftrl(
            "o",
            targets=["y"],
            features=[f"x{i}" for i in range(4)],
            clock="c",
            gap_cap=10.0,
            half_life=5.0,
        )
    p = po.ModelBank([s]).fit_predict(d)["o"].struct.field("pred_y").to_numpy()
    return p


def run(obj, x, y, tr, va, te, init):
    p = dict(
        objective=obj,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
    )
    i = (lambda s: None) if init is None else (lambda s: init[s])
    dtr = lgb.Dataset(x[tr], y[tr], init_score=i(tr), free_raw_data=False)
    dva = lgb.Dataset(x[va], y[va], init_score=i(va), reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    raw = b.predict(x[te], num_iteration=b.best_iteration, raw_score=True) + (
        0 if init is None else init[te]
    )
    if obj == "l2":
        return float(np.sqrt(np.mean((raw - y[te]) ** 2)))
    pr = np.clip(1 / (1 + np.exp(-raw)), 1e-12, 1 - 1e-12)
    return float(-np.mean(y[te] * np.log(pr) + (1 - y[te]) * np.log(1 - pr)))


print(C.load_line())
res = {}
for seed in (0, 1, 2):
    clock, x, yg, yb, L = make(seed)
    st = clock
    S = st.max() + 1
    tr, va, te = st < 0.6 * S, (st >= 0.6 * S) & (st < 0.7 * S), st >= 0.7 * S
    print(
        f"seed {seed}: level range train [{L[tr].min():.2f},{L[tr].max():.2f}] test [{L[te].min():.2f},{L[te].max():.2f}]"  # noqa: E501
    )
    r = {}
    m = online(clock, x, yg, "gauss")
    ok = ~np.isnan(m)
    m = np.where(ok, m, 0.0)
    r["gauss online alone RMSE"] = float(np.sqrt(np.mean((m[te] - yg[te]) ** 2)))
    r["gauss trees alone"] = run("l2", x, yg, tr, va, te, None)
    r["gauss trees + pred feature"] = run("l2", np.column_stack([x, m]), yg, tr, va, te, None)
    r["gauss trees on margin"] = run("l2", x, yg, tr, va, te, m)
    p = online(clock, x, yb, "bern")
    p = np.clip(np.where(np.isnan(p), 0.5, p), 1e-6, 1 - 1e-6)
    lm = np.log(p / (1 - p))
    r["bern online alone logloss"] = float(
        -np.mean(yb[te] * np.log(p[te]) + (1 - yb[te]) * np.log(1 - p[te]))
    )
    r["bern trees alone"] = run("binary", x, yb, tr, va, te, None)
    r["bern trees + logit feature"] = run("binary", np.column_stack([x, lm]), yb, tr, va, te, None)
    r["bern trees on logit margin"] = run("binary", x, yb, tr, va, te, lm)
    for k, v in r.items():
        print(f"  {k:32s} {v:.4f}", flush=True)
    res[seed] = r
Path(str(C.OUT / "margin_synth.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
