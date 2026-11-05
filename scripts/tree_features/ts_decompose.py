"""v3, review finding 1 and 10: flights' gain from target statistics,
decomposed into recent-label means with no key, with low-cardinality keys,
and with high-cardinality keys; every half-life and smoothing chosen on the
earlier months only.

Selection: trees trained on January-July, each candidate scored by its
August validation L2 (LightGBM's early-stopping best). Greedy, a group at a
time, in the order the table reports. Final: trained January-August, early
stopping on September, tested October-December, LightGBM seeds 1, 2, 3
(feature_fraction 0.9). Labels are learned at their arrival (avail)."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

DAY = 1440.0
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
JUL, AUG, SEP = 212 * DAY, 243 * DAY, 273 * DAY
NUM = ["dep_min", "arr_min", "wday", "distance"]
LOW = ["carrier", "origin"]
HIGH = ["tailnum", "dest", "route", "flight_id"]
K = {c: C.codes(df[c]) for c in LOW + HIGH}
K["global"] = np.zeros(len(t), dtype=np.int64)
base = np.column_stack([df[c].to_numpy() for c in NUM] + [K[c].astype(float) for c in LOW])
SHORT = [60.0, 120.0, 240.0, 480.0, 960.0]
LONG = [7 * DAY, 30 * DAY, math.inf]
_cache: dict = {}


def stat(col, hl, a, prior_hl):
    key = (col, hl, a, prior_hl)
    if key not in _cache:
        _cache[key] = C.ordered_ts(
            t, av, K[col], y, half_life=hl, prior_weight=a, prior_half_life=prior_hl
        )[0]
    return _cache[key]


PARAMS = dict(
    objective="l2",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.9,
    verbose=-1,
    num_threads=4,
    deterministic=True,
    max_cat_to_onehot=4,
    cat_smooth=10,
)


def fit(X, tr, va, te=None, seed=1):
    p = dict(PARAMS, seed=seed)
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=[4, 5], free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    if te is None:
        pv = b.predict(X[va], num_iteration=b.best_iteration)
        return float(np.mean((pv - y[va]) ** 2))
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    return (
        float(np.sqrt(np.mean((pr - y[te]) ** 2))),
        float(np.mean(np.abs(pr - y[te]))),
        b.best_iteration,
    )


s_tr, s_va = t < JUL, (t >= JUL) & (t < AUG)
f_tr, f_va, f_te = t < AUG, (t >= AUG) & (t < SEP), t >= SEP


def select(cols, col, grid, a_grid, prior_hl):
    """Best (half-life, smoothing) for `col`, added to `cols`, by August's L2."""
    best = None
    for hl in grid:
        s = fit(np.column_stack([base, *cols, stat(col, hl, a_grid[0], prior_hl)]), s_tr, s_va)
        if best is None or s < best[0]:
            best = (s, hl, a_grid[0])
    for a in a_grid[1:]:
        s = fit(np.column_stack([base, *cols, stat(col, best[1], a, prior_hl)]), s_tr, s_va)
        if s < best[0]:
            best = (s, best[1], a)
    print(
        f"  selected {col}: half-life {best[1]:g} min, smooth {best[2]:g}, August L2 {best[0]:.2f}",
        flush=True,
    )
    return best


print(C.load_line(), flush=True)
chosen: dict = {}
cols: list = []
print(f"base, August L2 {fit(base, s_tr, s_va):.2f}", flush=True)
# the keyless nowcast: its own mean is its prior, so the smoothing does not matter
s, hl, a = select(cols, "global", SHORT, [1.0], None)
chosen["global"] = (hl, a)
prior_hl = hl
cols.append(stat("global", hl, a, None))
groups = {"keyless": list(cols)}
for c in LOW:
    s, hl, a = select(cols, c, SHORT, [1.0, 10.0], prior_hl)
    chosen[c] = (hl, a)
    cols.append(stat(c, hl, a, prior_hl))
groups["low"] = list(cols)
for c in HIGH:
    s, hl, a = select(cols, c, SHORT, [1.0, 10.0], prior_hl)
    chosen[c] = (hl, a)
    cols.append(stat(c, hl, a, prior_hl))
groups["high_short"] = list(cols)
best = None
for hl in LONG:
    long_cols = [stat(c, hl, 10.0, prior_hl) for c in HIGH]
    s = fit(np.column_stack([base, *cols, *long_cols]), s_tr, s_va)
    if best is None or s < best[0]:
        best = (s, hl)
print(
    f"  selected the long statistic of the four: half-life {best[1]:g} min, smooth 10, "
    f"August L2 {best[0]:.2f}",
    flush=True,
)
chosen["long"] = (best[1], 10.0)
cols += [stat(c, best[1], 10.0, prior_hl) for c in HIGH]
groups["all"] = list(cols)
print(C.load_line(), flush=True)

res: dict = {"chosen": {k: list(v) for k, v in chosen.items()}, "arms": {}}
arms = {
    "base": [],
    "+ keyless recent mean": groups["keyless"],
    "+ origin, carrier recent means": groups["low"],
    "+ tailnum, dest, route, flight_id recent means": groups["high_short"],
    "+ their long statistics": groups["all"],
}
for name, cs in arms.items():
    X = np.column_stack([base, *cs]) if cs else base
    rows = [fit(X, f_tr, f_va, f_te, seed) for seed in (1, 2, 3)]
    res["arms"][name] = rows
    print(
        f"{name:48s} RMSE "
        + " / ".join(f"{r[0]:.3f}" for r in rows)
        + "  MAE "
        + " / ".join(f"{r[1]:.3f}" for r in rows)
        + "  rounds "
        + " / ".join(str(r[2]) for r in rows),
        flush=True,
    )
(C.OUT / "ts_decompose.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
