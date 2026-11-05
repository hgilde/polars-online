"""v3, review finding 2: what an offset buys inside a link, measured.

An offset inside sgd's link is emulated exactly with what the bank has: the
trees' raw margin as a feature whose coefficient is pinned at 1
(coef_min = coef_max = 1, standardize=False; the starting point is
projected, so it holds from the first row). Against it: the trees alone,
the online model alone, and the margin as a free feature. Each arm's
learning rate is chosen from {0.01, 0.05, 0.2} on the validation period;
the test period is scored once.

- BTCUSDT n_trades, Poisson: trees on days 1-40 (validation 41-45), sgd
  Poisson, adagrad, half-life 2 days, on the z-scored lag features.
- flights, arr_delay > 15, logistic: trees on January-August (September),
  sgd logistic, adagrad, half-life 7 days, labels learned at arrival (the
  doubled stream), on the keyless and origin recent rates of the label."""

import glob
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from leafrefit import bank  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import polars_online as po  # noqa: E402


def trees(X, yy, tr, va, obj):
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
    dtr = lgb.Dataset(X[tr], yy[tr], free_raw_data=False)
    b = lgb.train(
        p,
        dtr,
        3000,
        valid_sets=[lgb.Dataset(X[va], yy[va], reference=dtr)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return b.predict(X, num_iteration=b.best_iteration, raw_score=True)


def online(clock, cols, yy, loss, hl, gap, lr, pinned, learn_clock=None):
    names = list(cols)
    lo = [1.0 if c in pinned else -math.inf for c in names]
    hi = [1.0 if c in pinned else math.inf for c in names]

    def spec(cs):
        return po.spec.sgd(
            "o",
            targets=["y"],
            features=cs,
            loss=loss,
            schedule="adagrad",
            learning_rate=lr,
            standardize=False,
            coef_min=lo,
            coef_max=hi,
            clock="c",
            gap_cap=gap,
            half_life=hl,
        )

    return bank(clock, cols, yy, spec, learn_clock)


def poisson_dev(mu, yy):
    mu = np.maximum(mu, 1e-9)
    t = np.where(yy > 0, yy * np.log(np.maximum(yy, 1e-300) / mu), 0.0)
    return float(2 * np.mean(t - (yy - mu)))


def logloss(p, yy):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def arms(name, clock, base_cols, margin, yy, loss, hl, gap, va, te, score, learn_clock=None):
    out = {}
    link = np.exp if loss == "poisson" else (lambda z: 1 / (1 + np.exp(-z)))
    out["trees alone"] = score(link(margin[te]), yy[te])
    cases = {
        "online alone": (base_cols, set()),
        "the margin as a free feature": ({**base_cols, "margin": margin}, set()),
        "the margin as an offset": ({**base_cols, "margin": margin}, {"margin"}),
        "the margin as an offset, intercept alone": ({"margin": margin}, {"margin"}),
    }
    for arm, (cols, pinned) in cases.items():
        best = None
        for lr in (0.01, 0.05, 0.2):
            p = online(clock, cols, yy, loss, hl, gap, lr, pinned, learn_clock)
            p = np.where(np.isnan(p), link(margin), p)
            s = score(p[va], yy[va])
            if best is None or s < best[0]:
                best = (s, lr, p)
        out[arm] = score(best[2][te], yy[te])
        out[arm + " (rate)"] = best[1]
    print(name, json.dumps(out, indent=1), flush=True)
    return out


print(C.load_line(), flush=True)
res = {}
fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
d = (
    pl.concat([pl.read_parquet(f) for f in fs])
    .sort("t")
    .with_columns(
        z=pl.col("n_trades").log1p(),
        r=(pl.col("close") / pl.col("open")).log().abs(),
        mod=((pl.col("t") % 86400) / 60).cast(pl.Float64),
        wd=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.weekday().cast(pl.Float64),
    )
)
d = d.with_columns(
    [pl.col("z").shift(1).alias("z1")]
    + [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (5, 30, 240, 1440)]
    + [pl.col("r").shift(1).alias("r1"), pl.col("r").ewm_mean(half_life=30).shift(1).alias("rm30")]
).slice(1)
F = ["z1", "zm5", "zm30", "zm240", "zm1440", "r1", "rm30", "mod", "wd"]
X = d.select(F).to_numpy()
yy = d["n_trades"].to_numpy()
day = ((d["t"] - d["t"][0]) // 86400).to_numpy()
tr, va, te = day < 40, (day >= 40) & (day < 45), day >= 45
m = trees(X, yy, tr, va, "poisson")
mu, sd = X[tr].mean(0), X[tr].std(0)
zcols = {f"z_{c}": (X[:, j] - mu[j]) / sd[j] for j, c in enumerate(F[:7])}
res["btc poisson deviance"] = arms(
    "BTCUSDT Poisson deviance",
    np.arange(len(yy), dtype=float),
    zcols,
    m,
    yy,
    "poisson",
    2880.0,
    10.0,
    va,
    te,
    poisson_dev,
)
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
yb = (df["y"].to_numpy() > 15).astype(float)
tr, va, te = t < 243 * 1440.0, (t >= 243 * 1440.0) & (t < 273 * 1440.0), t >= 273 * 1440.0
K = {c: C.codes(df[c]) for c in ("carrier", "origin", "dest")}
Xf = np.column_stack(
    [df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")]
    + [K[c].astype(float) for c in K]
)
mf = trees(Xf, yb, tr, va, "binary")
g = C.ordered_ts(t, av, np.zeros(len(t), dtype=np.int64), yb, half_life=60.0, prior_weight=1.0)[0]
o = C.ordered_ts(t, av, K["origin"], yb, half_life=60.0, prior_weight=10.0, prior_half_life=60.0)[0]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
rcols = {"global_rate": lg(g), "origin_rate": lg(o)}
res["flights logloss"] = arms(
    "flights arr_delay > 15, log loss",
    t,
    rcols,
    mf,
    yb,
    "logistic",
    7 * 1440.0,
    600.0,
    va,
    te,
    logloss,
    learn_clock=av,
)
auc = lambda p, yy: -float(roc_auc_score(yy, p))  # noqa: E731
res["flights -AUC"] = arms(
    "flights arr_delay > 15, minus AUC",
    t,
    rcols,
    mf,
    yb,
    "logistic",
    7 * 1440.0,
    600.0,
    va,
    te,
    auc,
    learn_clock=av,
)
(C.OUT / "offset_link.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
