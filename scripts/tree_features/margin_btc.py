"""Candidate 1 on BTCUSDT 1m (60 days): an online model's out-of-sample
prediction as the trees' base margin, against it as a feature and against
either alone. Train days 1-40, valid 41-45, test 46-60."""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

import polars_online as po

fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
df = (
    pl.concat([pl.read_parquet(f) for f in fs])
    .sort("t")
    .with_row_index("i")
    .with_columns(
        pl.col("i").cast(pl.Float64).alias("clock"),
        z=pl.col("n_trades").log1p(),
        r=(pl.col("close") / pl.col("open")).log().abs(),
        mod=((pl.col("t") % 86400) / 60).cast(pl.Float64),
        wd=pl.from_epoch(pl.col("t").cast(pl.Int64), "s").dt.weekday().cast(pl.Float64),
    )
)
lag = (
    [pl.col("z").shift(1).alias("z1")]
    + [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (5, 30, 240, 1440)]
    + [pl.col("r").shift(1).alias("r1"), pl.col("r").ewm_mean(half_life=30).shift(1).alias("rm30")]
)
df = df.with_columns(lag).slice(1)
F = ["z1", "zm5", "zm30", "zm240", "zm1440", "r1", "rm30"]
spec = po.spec.ewridge(
    "on",
    targets=["z"],
    features=F,
    clock="clock",
    gap_cap=10.0,
    half_life=2880.0,
    ridge=1e-4,
    standardize=True,
)
out = po.ModelBank([spec]).fit_predict(df)
m = out["on"].struct.field("pred_z").to_numpy()
ok = ~np.isnan(m)
df = df.with_columns(m=pl.Series(m)).filter(pl.Series(ok))
m = df["m"].to_numpy()
y = df["n_trades"].to_numpy()
z = df["z"].to_numpy()
day = ((df["t"] - df["t"][0]) // 86400).to_numpy()
tr, va, te = day < 40, (day >= 40) & (day < 45), day >= 45
TF = F + ["mod", "wd"]
X = df.select(TF).to_numpy()


def dev(mu, y):  # mean Poisson deviance
    mu = np.maximum(mu, 1e-9)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def run(obj, label, Xm, init):
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
    yy = y if obj == "poisson" else z
    dtr = lgb.Dataset(
        Xm[tr], yy[tr], init_score=None if init is None else init[tr], free_raw_data=False
    )
    dva = lgb.Dataset(Xm[va], yy[va], init_score=None if init is None else init[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    raw = b.predict(Xm[te], num_iteration=b.best_iteration, raw_score=True) + (
        0 if init is None else init[te]
    )
    if obj == "poisson":
        mu = np.exp(raw)
        s = f"Poisson deviance {dev(mu, y[te]):8.3f}  RMSE(log1p) {np.sqrt(np.mean((np.log1p(mu) - z[te]) ** 2)):.4f}"  # noqa: E501
    else:
        s = f"RMSE(log1p) {np.sqrt(np.mean((raw - z[te]) ** 2)):.4f}"
    print(f"{obj:8s} {label:40s} {s}  rounds {b.best_iteration}", flush=True)
    return s


print(C.load_line())
print("rows", len(y), "train/valid/test", tr.sum(), va.sum(), te.sum())
print("mean n_trades train", y[tr].mean().round(1), "test", y[te].mean().round(1))
res = {}
for obj in ("poisson", "l2"):
    res[obj] = {
        "trees alone": run(obj, "trees alone", X, None),
        "trees + online pred as a feature": run(
            obj, "trees + online pred as a feature", np.column_stack([X, m]), None
        ),
        "trees on the online margin": run(obj, "trees on the online pred as base margin", X, m),
        "trees on the margin + it as feature": run(
            obj, "trees on the margin, and it as a feature", np.column_stack([X, m]), m
        ),
    }
mu = np.exp(m[te])
print(
    f"online alone: Poisson deviance {dev(np.expm1(m[te]), y[te]):8.3f}  RMSE(log1p) {np.sqrt(np.mean((m[te] - z[te]) ** 2)):.4f}"  # noqa: E501
)
# XGBoost: does base_margin replace base_score?
d = xgb.DMatrix(X[tr], label=z[tr], base_margin=m[tr])
dt = xgb.DMatrix(X[te], label=z[te], base_margin=m[te])
b = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.1, "nthread": 4}, d, 200)
cfg = json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"]
pt = b.predict(dt)
pn = b.predict(xgb.DMatrix(X[te]), output_margin=True)
print(
    "xgb base_score",
    cfg,
    "| pred(with margin) - margin - pred(no margin) + base_score:",
    float(np.max(np.abs(pt - m[te] - pn + float(cfg.strip("[]"))))),
)
print(f"xgb L2 RMSE(log1p) with margin {np.sqrt(np.mean((pt - z[te]) ** 2)):.4f}")
print(C.load_line())
