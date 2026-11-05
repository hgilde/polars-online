"""Candidate 2, classification form: arr_delay > 15 on nycflights13; the
target statistic of a 0/1 target is a rate. Same split as ts_flights.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import TargetEncoder  # noqa: E402

df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = (df["y"].to_numpy() > 15).astype(float)
cut_v, cut_t = 243 * 1440.0, 273 * 1440.0
tr = t < cut_v
va = (t >= cut_v) & (t < cut_t)
te = t >= cut_t
NUM = ["dep_min", "arr_min", "wday", "distance"]
LOW = ["carrier", "origin"]
HIGH = ["dest", "tailnum", "flight_id", "route"]
K = {c: C.codes(df[c]) for c in LOW + HIGH}
base = np.column_stack([df[c].to_numpy() for c in NUM] + [K[c].astype(float) for c in LOW])


def ll(p, yy):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def fit(X, cats, label):
    p = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        verbose=-1,
        num_threads=4,
        deterministic=True,
        seed=1,
        max_cat_to_onehot=4,
        cat_smooth=10,
    )
    dtr = lgb.Dataset(
        X[tr],
        y[tr],
        categorical_feature=[i for i, n in enumerate(NUM + LOW + HIGH) if n in cats],
        free_raw_data=False,
    )
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    from sklearn.metrics import roc_auc_score

    r = (ll(pr, y[te]), float(roc_auc_score(y[te], pr)))
    print(f"{label:55s} logloss {r[0]:.4f} AUC {r[1]:.4f}", flush=True)
    return r


print(C.load_line())
print(
    f"base rate train {y[tr].mean():.3f} test {y[te].mean():.3f}; constant-rate logloss {ll(np.full(te.sum(), y[tr].mean()), y[te]):.4f}"  # noqa: E501
)
res = {}
res["dropped"] = fit(base, LOW, "high-card dropped")
res["native"] = fit(
    np.column_stack([base] + [K[c].astype(float) for c in HIGH]),
    LOW + HIGH,
    "LightGBM native categorical",
)
H = np.column_stack([K[c] for c in HIGH])
enc = TargetEncoder(target_type="binary", cv=5)
Xs = np.empty((len(t), 4))
Xs[tr] = enc.fit_transform(H[tr], y[tr])
Xs[~tr] = enc.transform(H[~tr])
res["sklearn"] = fit(np.column_stack([base, Xs]), LOW, "sklearn TargetEncoder (cross-fit)")
e30 = [C.ordered_ts(t, av, K[c], y, half_life=30 * 1440.0)[0] for c in HIGH]
res["ots30"] = fit(np.column_stack([base] + e30), LOW, "ordered rate, 30d, continuing, embargoed")
short = [
    C.ordered_ts(t, av, K[c], y, half_life=hl, prior_weight=1.0, prior_half_life=240.0)[0]
    for c, hl in (("tailnum", 360.0), ("origin", 120.0), ("dest", 240.0), ("route", 240.0))
]
res["ots30+recent"] = fit(
    np.column_stack([base] + e30 + short),
    LOW,
    "+ short half-life rates (6h plane, 2h origin, 4h dest/route)",
)
Path(str(C.OUT / "ts_flights_binary.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
