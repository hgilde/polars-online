"""Candidate 6: a batch retrain weighted by the decay a stream gives each row
at the training cut, w = 0.5 ** ((cut - t) / half_life), and the validation
split: the last slice of time (forward) against a shuffled one."""

import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from leafrefit import synth  # noqa: E402


def fit(X, y, w, tr, va, te, obj="l2"):
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
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], weight=w[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    return round(float(np.sqrt(np.mean((pr - y[te]) ** 2))), 4), b.best_iteration


def study(name, X, y, t, cut, test_end, hls, unit):
    trv = t < cut
    te = (t >= cut) & (t < test_end)
    # forward validation: the last 10% of the training time; shuffled: a random 10%
    tv = np.quantile(t[trv], 0.9)
    fwd_tr, fwd_va = trv & (t < tv), trv & (t >= tv)
    rng = np.random.default_rng(0)
    u = rng.random(len(t))
    sh_tr, sh_va = trv & (u >= 0.1), trv & (u < 0.1)
    out = {}
    for hl in hls:
        w = np.ones(len(t)) if hl == math.inf else 0.5 ** ((cut - t) / hl)
        w = np.where(trv, w, 1.0)
        out[f"hl={hl / unit:g}"] = {
            "forward valid": fit(X, y, w, fwd_tr, fwd_va, te),
            "shuffled valid": fit(X, y, w, sh_tr, sh_va, te),
        }
    # a hard window of the same mean age as the middle half-life
    print(name, json.dumps(out), flush=True)
    return out


print(C.load_line())
res = {}
# flights: train to Oct 1, test Oct-Dec
df = C.load_flights()
t = df["t"].to_numpy()
y = df["y"].to_numpy()
K = {c: C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")}
X = np.column_stack(
    [df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")] + list(K.values())
)
D = 1440.0
res["flights"] = study("flights", X, y, t, 273 * D, 1e18, [7 * D, 30 * D, 90 * D, math.inf], D)
# BTC: train days 0-44, test 45-59
fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
b = (
    pl.concat([pl.read_parquet(f) for f in fs])
    .sort("t")
    .with_columns(
        z=pl.col("n_trades").log1p(),
        r=(pl.col("close") / pl.col("open")).log().abs(),
        mod=((pl.col("t") % 86400) / 60).cast(pl.Float64),
        wd=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.weekday().cast(pl.Float64),
    )
)
b = b.with_columns(
    [pl.col("z").shift(1).alias("z1")]
    + [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (5, 30, 240, 1440)]
    + [pl.col("r").shift(1).alias("r1"), pl.col("r").ewm_mean(half_life=30).shift(1).alias("rm30")]
).slice(1)
tb = ((b["t"] - b["t"][0]) / 60).to_numpy()
Xb = b.select(["z1", "zm5", "zm30", "zm240", "zm1440", "r1", "rm30", "mod", "wd"]).to_numpy()
res["btc"] = study(
    "btc", Xb, b["z"].to_numpy(), tb, 45 * D, 1e18, [2 * D, 7 * D, 21 * D, math.inf], D
)
# planted drift: train first half, test the next 10%
for seed in (0, 1, 2):
    clock, Xs, ys = synth(seed)
    res[f"synth{seed}"] = study(
        f"synth{seed}", Xs, ys, clock, 1500.0, 1800.0, [50.0, 200.0, 800.0, math.inf], 1.0
    )
Path(str(C.OUT / "decayw.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
