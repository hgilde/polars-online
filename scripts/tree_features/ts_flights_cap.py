"""Candidate 2's cap on real data: ts_flights.py's best arm (30-day statistics
of dest, tailnum, flight_id, route, with short half-lives beside them) with
every statistic's table capped at K levels, the least decayed weight
evicted. tailnum has 4,037 levels and flight_id 5,706."""

import json
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
tr, va, te = t < 243 * DAY, (t >= 243 * DAY) & (t < 273 * DAY), t >= 273 * DAY
NUM = ["dep_min", "arr_min", "wday", "distance"]
LOW = ["carrier", "origin"]
HIGH = ["dest", "tailnum", "flight_id", "route"]
K = {c: C.codes(df[c]) for c in LOW + HIGH}
base = np.column_stack([df[c].to_numpy() for c in NUM] + [K[c].astype(float) for c in LOW])


def fit(X):
    p = dict(
        objective="l2",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.9,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
        max_cat_to_onehot=4,
        cat_smooth=10,
    )
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=[4, 5], free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    return float(np.sqrt(np.mean((pr - y[te]) ** 2))), float(np.mean(np.abs(pr - y[te])))


print(C.load_line())
res = {}
for cap in (None, 1024, 256, 64):
    cols = []
    for c in HIGH:
        e, w = C.ordered_ts(t, av, K[c], y, half_life=30 * DAY, cap=cap)
        cols += [e, w]
    for c, hl in (("tailnum", 360.0), ("origin", 120.0), ("dest", 240.0), ("route", 240.0)):
        cols.append(
            C.ordered_ts(
                t, av, K[c], y, half_life=hl, prior_weight=1.0, prior_half_life=240.0, cap=cap
            )[0]
        )
    r = fit(np.column_stack([base, *cols]))
    res[str(cap)] = r
    print(f"cap {cap}: test RMSE {r[0]:.3f} MAE {r[1]:.3f}", flush=True)
(C.OUT / "ts_flights_cap.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
