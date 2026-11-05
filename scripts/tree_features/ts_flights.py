"""Candidate 2 on nycflights13: encodings of high-cardinality categoricals for
LightGBM, in time order. Train Jan-Aug, validate (early stopping) Sep, test
Oct-Dec. Target arr_delay (minutes)."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.preprocessing import TargetEncoder

DAY = 1440.0
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
cut_v = (pl.datetime(2013, 9, 1) - pl.datetime(2013, 1, 1)).dt.total_minutes()
cut_v = float(pl.select(cut_v).item())
cut_t = float(
    pl.select((pl.datetime(2013, 10, 1) - pl.datetime(2013, 1, 1)).dt.total_minutes()).item()
)
tr = t < cut_v
va = (t >= cut_v) & (t < cut_t)
te = t >= cut_t
NUM = ["dep_min", "arr_min", "wday", "distance"]
LOW = ["carrier", "origin"]
HIGH = ["dest", "tailnum", "flight_id", "route"]
K = {c: C.codes(df[c]) for c in LOW + HIGH}
base = np.column_stack([df[c].to_numpy() for c in NUM] + [K[c].astype(float) for c in LOW])
base_names = NUM + LOW


def fit_eval(X, names, cats, label):
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
        max_bin=255,
    )
    dtr = lgb.Dataset(
        X[tr], y[tr], feature_name=names, categorical_feature=cats, free_raw_data=False
    )
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pr = b.predict(X[te], num_iteration=b.best_iteration)
    rmse = float(np.sqrt(np.mean((pr - y[te]) ** 2)))
    mae = float(np.mean(np.abs(pr - y[te])))
    print(
        f"{label:58s} test RMSE {rmse:7.3f}  MAE {mae:6.3f}  rounds {b.best_iteration}", flush=True
    )
    return rmse, mae


res = {}
print(C.load_line())
print("rows", len(t), "train", tr.sum(), "valid", va.sum(), "test", te.sum())
print(f"predict-the-train-mean RMSE {np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)):.3f}")
res["none"] = fit_eval(base, base_names, LOW, "V0 high-card dropped")
X = np.column_stack([base] + [K[c].astype(float) for c in HIGH])
res["native"] = fit_eval(X, base_names + HIGH, LOW + HIGH, "V1 LightGBM native categorical")
res["ordinal"] = fit_eval(X, base_names + HIGH, LOW, "V2 codes as numbers")


# greedy (in-sample) TS on train, a = 10
def greedy(c, a=10.0):
    k = K[c]
    m = y[tr].mean()
    s = np.bincount(k[tr], weights=y[tr], minlength=k.max() + 1)
    n = np.bincount(k[tr], minlength=k.max() + 1)
    return (s[k] + a * m) / (n[k] + a)


Xg = np.column_stack([base] + [greedy(c) for c in HIGH])
res["greedy"] = fit_eval(Xg, base_names + HIGH, LOW, "V3 greedy TS, in sample (leaks)")
# sklearn cross-fitted TS
enc = TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=0)
H = np.column_stack([K[c] for c in HIGH])
Xs = np.empty((len(t), len(HIGH)))
Xs[tr] = enc.fit_transform(H[tr], y[tr])
Xs[~tr] = enc.transform(H[~tr])
res["sklearn"] = fit_eval(
    np.column_stack([base, Xs]),
    base_names + HIGH,
    LOW,
    "V4 sklearn TargetEncoder (5-fold cross-fit)",
)


def ots(c, **kw):
    return C.ordered_ts(t, av, K[c], y, **kw)


cols = {}
for hl, tag in ((math.inf, "inf"), (30 * DAY, "30d")):
    for fz in (None, cut_v):
        name = f"ots_{tag}_{'frozen' if fz else 'cont'}"
        enc_cols = []
        w_cols = []
        with C.Timer() as tm:
            for c in HIGH:
                e, w = ots(c, half_life=hl, freeze_at=fz)
                enc_cols.append(e)
                w_cols.append(w)
        cols[name] = (enc_cols, w_cols)
        print(f"  {name}: {tm.s:.1f}s for {len(HIGH)} columns")
        res[name] = fit_eval(
            np.column_stack([base] + enc_cols),
            base_names + HIGH,
            LOW,
            f"V5 ordered TS hl={tag} {'frozen at the cut' if fz else 'continuing'}",
        )
# + weight behind and frequency
enc_cols, w_cols = cols["ots_30d_cont"]
fr = [C.decayed_freq(t, K[c], 30 * DAY)[0] for c in HIGH]
Xf = np.column_stack([base] + enc_cols + w_cols + fr)
nf = base_names + HIGH + [c + "_w" for c in HIGH] + [c + "_freq" for c in HIGH]
res["ots_30d_cont_w_freq"] = fit_eval(
    Xf, nf, LOW, "V6 V5(30d, cont) + label weight + decayed frequency"
)
# recent-state TS: short half-lives
short = []
for c, hl in (("tailnum", 360.0), ("origin", 120.0), ("dest", 240.0), ("route", 240.0)):
    short.append(
        C.ordered_ts(t, av, K[c], y, half_life=hl, prior_weight=1.0, prior_half_life=240.0)[0]
    )
Xr = np.column_stack([Xf] + short)
res["recent"] = fit_eval(
    Xr,
    nf + ["tail_6h", "origin_2h", "dest_4h", "route_4h"],
    LOW,
    "V7 V6 + short half-life TS (6h plane, 2h origin, 4h dest/route)",
)
Path(str(C.OUT / "ts_flights.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
