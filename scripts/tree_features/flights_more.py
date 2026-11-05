"""Candidates 1 and 5 on nycflights13 (train Jan-Aug, valid Sep, test Oct-Dec).
Labels are learned only once known (avail): the online arms run on a doubled
stream, a predict row at t and a learn row at avail."""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from leafrefit import bank, leaf_values, rmse  # noqa: E402

import polars_online as po  # noqa: E402

df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
cut_v = 243 * 1440.0
cut_t = 273 * 1440.0  # Sep 1, Oct 1 (2013 not a leap year)
tr = t < cut_v
va = (t >= cut_v) & (t < cut_t)
te = t >= cut_t
K = {c: C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")}
X = np.column_stack(
    [df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")]
    + [K[c] for c in ("carrier", "origin", "dest")]
)
print(C.load_line())
res = {}
# margin: the origin's recent delays (2h half-life), an online prediction known before the row
k_origin = C.codes(df["origin"])
m, _ = C.ordered_ts(t, av, k_origin, y, half_life=120.0, prior_weight=1.0, prior_half_life=240.0)


def lgbfit(Xm, init, nl=31, lr=0.05):
    p = dict(
        objective="l2",
        learning_rate=lr,
        num_leaves=nl,
        min_data_in_leaf=100,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
    )
    i = (lambda s: None) if init is None else (lambda s: init[s])
    dtr = lgb.Dataset(Xm[tr], y[tr], init_score=i(tr), free_raw_data=False)
    dva = lgb.Dataset(Xm[va], y[va], init_score=i(va), reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return b, b.predict(Xm, num_iteration=b.best_iteration, raw_score=True) + (
        0 if init is None else init
    )


_, p0 = lgbfit(X, None)
_, p1 = lgbfit(np.column_stack([X, m]), None)
_, p2 = lgbfit(X, m)
_, p3 = lgbfit(np.column_stack([X, m]), m)
res["margin"] = {
    "trees alone": rmse(p0[te], y[te]),
    "online (origin 2h) alone": rmse(m[te], y[te]),
    "trees + it as feature": rmse(p1[te], y[te]),
    "trees on it as margin": rmse(p2[te], y[te]),
    "both": rmse(p3[te], y[te]),
}
print("margin on flights:", json.dumps(res["margin"], indent=1), flush=True)

# leaf refit
b, base = lgbfit(X, None, nl=15, lr=0.1)
leaves, vals = leaf_values(b, X)
M = vals.shape[1]
print("trees", M, "frozen test RMSE", rmse(base[te], y[te]))
r = {"trees": M, "frozen trees": rmse(base[te], y[te])}
hl, gap = 7 * 1440.0, 600.0
ew = lambda ridge: (  # noqa: E731
    lambda cols: po.spec.ewridge(
        "o", targets=["y"], features=cols, clock="c", gap_cap=gap, half_life=hl, ridge=ridge
    )
)
p = bank(t, {f"v{j}": vals[:, j] for j in range(M)}, y, ew(1e-3), learn_clock=av)
r["ewridge on M leaf values"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
oh = {
    f"o{j}_{l}": (leaves[:, j] == l).astype(float)
    for j in range(M)
    for l in range(int(leaves[:, j].max()) + 1)  # noqa: E741
}
r["onehot width"] = len(oh)
if len(oh) <= 700:
    p = bank(t, oh, y, ew(1e-2), learn_clock=av)
    r["ewridge on one-hot leaves"] = rmse(np.where(np.isnan(p[te]), base[te], p[te]), y[te])
raw = {f"x{j}": X[:, j] for j in range(X.shape[1])}
p = bank(t, raw, y - base, ew(1e-3), learn_clock=av)
r["trees + online residual (offset)"] = rmse(base[te] + np.nan_to_num(p[te]), y[te])
# the same with the recent-state margin in the residual model
p = bank(t, {**raw, "m": m}, y - base, ew(1e-3), learn_clock=av)
r["trees + online residual on raw + origin-2h"] = rmse(base[te] + np.nan_to_num(p[te]), y[te])
res["leafrefit"] = r
print("leaf refit on flights:", json.dumps(r, indent=1), encoding="utf-8")
Path(str(C.OUT / "flights_more.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
