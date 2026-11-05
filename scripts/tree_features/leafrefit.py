"""Candidate 5: frozen batch trees, their leaves refitted online with decay.
Arms: frozen trees; online ewridge on the M per-tree leaf values; online
ewridge on one-hot leaves (M*L); an online residual corrector (ewridge on the
raw features, target y - tree pred); XGBoost-style periodic refresh is in
refresh.py. Data: planted drift (3 seeds), BTCUSDT 1m."""

import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import lightgbm as lgb
import numpy as np
import polars as pl

import polars_online as po


def train_trees(X, y, tr, va, num_leaves, rounds_cap, obj="l2"):
    p = dict(
        objective=obj,
        learning_rate=0.1,
        num_leaves=num_leaves,
        min_data_in_leaf=100,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
    )
    dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, rounds_cap, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return b


def leaf_values(b, X):
    """(n, M) each tree's output on each row, from pred_leaf and the leaf table."""
    leaves = b.predict(X, pred_leaf=True, num_iteration=b.best_iteration)
    M = leaves.shape[1]
    vals = np.empty(leaves.shape)
    for m in range(M):
        nl = int(leaves[:, m].max()) + 1
        tab = np.array([b.get_leaf_output(m, l) for l in range(nl)])  # noqa: E741
        vals[:, m] = tab[leaves[:, m]]
    return leaves, vals


def bank(clock, cols: dict, y, spec_fn, learn_clock=None):
    """One online model over named columns; pred before each row's update.
    With learn_clock, the stream is doubled: a predict row at clock (no target)
    and a learn row at learn_clock (target), so a label is learned only once
    it is known."""
    d = pl.DataFrame(
        {"c": clock, "y": y, "rid": np.arange(len(y)), "role": np.zeros(len(y), np.int8), **cols}
    )
    if learn_clock is not None:
        d = pl.concat(
            [
                d.with_columns(y=pl.lit(None, pl.Float64)),
                d.with_columns(c=pl.Series(learn_clock), role=pl.lit(1, pl.Int8)),
            ]
        ).sort("c", "role", "rid")
    out = po.ModelBank([spec_fn(list(cols))]).fit_predict(d)
    out = (
        out.with_columns(p=pl.col("o").struct.field("pred_y"))
        .filter(pl.col("role") == 0)
        .sort("rid")
    )
    return out["p"].to_numpy()


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def arms(name, clock, X, y, tr, va, te, hl, gap, learn_clock=None, num_leaves=15, rounds=300):
    r = {}
    b = train_trees(X, y, tr, va, num_leaves, rounds)
    base = b.predict(X, num_iteration=b.best_iteration)
    leaves, vals = leaf_values(b, X)
    M = vals.shape[1]
    r["trees"] = M
    r["frozen trees"] = rmse(base[te], y[te])
    ew = lambda f, ridge=1e-3: (  # noqa: E731
        lambda cols: po.spec.ewridge(
            "o",
            targets=["y"],
            features=cols,
            clock="c",
            gap_cap=gap,
            half_life=hl,
            ridge=ridge,
            standardize=False,
        )
    )
    t0 = time.perf_counter()
    p = bank(clock, {f"v{m}": vals[:, m] for m in range(M)}, y, ew(None), learn_clock)
    r["ewridge on M leaf values"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
    r["t_leafvals_s"] = time.perf_counter() - t0
    # one-hot leaves: M*L columns
    oh = {}
    for m in range(M):
        for l in range(int(leaves[:, m].max()) + 1):  # noqa: E741
            oh[f"o{m}_{l}"] = (leaves[:, m] == l).astype(float)
    r["onehot width"] = len(oh)
    if len(oh) <= 600:
        t0 = time.perf_counter()
        p = bank(clock, oh, y, ew(None, ridge=1e-2), learn_clock)
        r["ewridge on one-hot leaves"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
        r["t_onehot_s"] = time.perf_counter() - t0
    # residual corrector: ewridge on raw features, target y - trees (an offset)
    raw = {f"x{j}": X[:, j] for j in range(X.shape[1])}
    p = bank(clock, raw, y - base, ew(None), learn_clock)
    r["trees + online residual (offset)"] = rmse(base[te] + np.nan_to_num(p[te]), y[te])
    # the online model alone on raw features
    p = bank(clock, raw, y, ew(None), learn_clock)
    r["online alone (raw)"] = rmse(np.nan_to_num(p[te]), y[te])
    # leaf values + offset-free intercept: also on vals + raw
    print(f"{name}:", flush=True)
    for k, v in r.items():
        print(f"  {k:36s} {v:.4f}" if isinstance(v, float) else f"  {k:36s} {v}", flush=True)
    return r


def synth(seed, steps=3000, per=100):
    rng = np.random.default_rng(seed)
    n = steps * per
    clock = np.repeat(np.arange(steps, dtype=float), per)
    s = clock / steps
    x = rng.standard_normal((n, 5))
    a = 1 + np.sin(2 * np.pi * s)  # the weight of each shape drifts
    bb = 1 + 1.5 * s
    c = 1 - 2 * s
    y = (
        a * np.sin(2 * x[:, 0])
        + bb * (x[:, 1] > 0.5)
        + c * x[:, 2] * (x[:, 3] > 0)
        + 0.5 * np.cumsum(rng.normal(0, 0.05, steps))[clock.astype(int)]
        + rng.standard_normal(n)
    )
    return clock, x, y


if __name__ == "__main__":
    print(C.load_line())
    res = {}
    for seed in (0, 1, 2):
        clock, X, y = synth(seed)
        S = clock.max() + 1
        tr, va, te = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S), clock >= 0.5 * S
        res[f"synth{seed}"] = arms(
            f"planted drift seed {seed}", clock, X, y, tr, va, te, hl=20.0, gap=10.0
        )
    # BTC
    fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
    df = (
        pl.concat([pl.read_parquet(f) for f in fs])
        .sort("t")
        .with_row_index("i")
        .with_columns(
            clock=pl.col("i").cast(pl.Float64),
            z=pl.col("n_trades").log1p(),
            r=(pl.col("close") / pl.col("open")).log().abs(),
            mod=((pl.col("t") % 86400) / 60).cast(pl.Float64),
            wd=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.weekday().cast(pl.Float64),
        )
    )
    df = df.with_columns(
        [pl.col("z").shift(1).alias("z1")]
        + [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (5, 30, 240, 1440)]
        + [
            pl.col("r").shift(1).alias("r1"),
            pl.col("r").ewm_mean(half_life=30).shift(1).alias("rm30"),
        ]
    ).slice(1)
    F = ["z1", "zm5", "zm30", "zm240", "zm1440", "r1", "rm30", "mod", "wd"]
    X = df.select(F).to_numpy()
    y = df["z"].to_numpy()
    clock = df["clock"].to_numpy()
    day = ((df["t"] - df["t"][0]) // 86400).to_numpy()
    tr, va, te = day < 40, (day >= 40) & (day < 45), day >= 45
    res["btc"] = arms("BTCUSDT 1m log1p(n_trades)", clock, X, y, tr, va, te, hl=2880.0, gap=10.0)
    Path(str(C.OUT / "leafrefit.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(C.load_line())
