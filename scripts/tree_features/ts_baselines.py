"""v3, review finding 10: fair baselines on flights for the target statistic.

1. The static arms of ts_flights.py again at learning rate 0.01 with a
   patience of 300 rounds, to show they had converged on the forward split.
2. LightGBM's native categorical tuned: every categorical given 8,192 bins
   so that no level is folded into "other", and cat_smooth,
   min_data_per_group, max_cat_threshold and cat_l2 chosen on August
   (trained January-July) from a seeded sample of the grid.
3. A monthly retrain: at each of October 1, November 1 and December 1, the
   trees and a cross-fitted scikit-learn TargetEncoder refitted on every
   label known by then (early stopping on the last 30 days of them), tested
   on that month; and the same with ts_decompose.py's online statistics as
   features (its chosen half-lives, read from its JSON)."""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import TargetEncoder  # noqa: E402

DAY = 1440.0
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
JUL, AUG, SEP = 212 * DAY, 243 * DAY, 273 * DAY
MONTHS = [(273 * DAY, 304 * DAY), (304 * DAY, 334 * DAY), (334 * DAY, 366 * DAY)]
NUM = ["dep_min", "arr_min", "wday", "distance"]
LOW = ["carrier", "origin"]
HIGH = ["tailnum", "dest", "route", "flight_id"]
K = {c: C.codes(df[c]) for c in LOW + HIGH}
K["global"] = np.zeros(len(t), dtype=np.int64)
base = np.column_stack([df[c].to_numpy() for c in NUM] + [K[c].astype(float) for c in LOW])
H = np.column_stack([K[c] for c in HIGH])
f_tr, f_va, f_te = t < AUG, (t >= AUG) & (t < SEP), t >= SEP
s_tr, s_va = t < JUL, (t >= JUL) & (t < AUG)
BASEP = dict(
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


def run(X, tr, va, te, cats, params=None, patience=100, rounds=3000, score_only=False):
    p = dict(BASEP, **(params or {}))
    dtr = lgb.Dataset(
        X[tr],
        y[tr],
        categorical_feature=cats,
        free_raw_data=False,
        params={k: p[k] for k in ("max_bin_by_feature", "max_bin") if k in p},
    )
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    b = lgb.train(
        p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)]
    )
    if score_only:
        pv = b.predict(X[va], num_iteration=b.best_iteration)
        return float(np.mean((pv - y[va]) ** 2))
    return b.predict(X[te], num_iteration=b.best_iteration), b.best_iteration


def score(pr, te):
    return float(np.sqrt(np.mean((pr - y[te]) ** 2))), float(np.mean(np.abs(pr - y[te])))


print(C.load_line(), flush=True)
res: dict = {}
# 1. converged static arms
Xn = np.column_stack([base, *[K[c].astype(float) for c in HIGH]])
enc = TargetEncoder(target_type="continuous", cv=5)
Xs = np.empty((len(t), 4))
Xs[f_tr] = enc.fit_transform(H[f_tr], y[f_tr])
Xs[~f_tr] = enc.transform(H[~f_tr])
for name, X, cats in (
    ("dropped", base, [4, 5]),
    ("native, defaults", Xn, [4, 5, 6, 7, 8, 9]),
    ("scikit-learn TargetEncoder", np.column_stack([base, Xs]), [4, 5]),
):
    for lr, pat in ((0.05, 100), (0.01, 300)):
        pr, it = run(X, f_tr, f_va, f_te, cats, {"learning_rate": lr}, patience=pat, rounds=20000)
        r = score(pr, f_te)
        res[f"{name}, lr {lr}"] = [*r, it]
        print(f"{name:30s} lr {lr}: RMSE {r[0]:.3f} MAE {r[1]:.3f} rounds {it}", flush=True)
# 2. native tuned on August
cats = [4, 5, 6, 7, 8, 9]
mbf = {"max_bin_by_feature": [255] * 4 + [8192] * 6}
grid = list(itertools.product([1, 10, 50], [10, 50, 200], [32, 128], [1, 10]))
rng = np.random.default_rng(0)
pick = [grid[i] for i in rng.choice(len(grid), 12, replace=False)]
best = None
for cs, mdg, mct, cl2 in pick:
    prm = dict(mbf, cat_smooth=cs, min_data_per_group=mdg, max_cat_threshold=mct, cat_l2=cl2)
    s = run(Xn, s_tr, s_va, None, cats, prm, score_only=True)
    print(
        f"  native cat_smooth {cs} min_data_per_group {mdg} max_cat_threshold {mct} cat_l2 {cl2}: "
        f"August L2 {s:.1f}",
        flush=True,
    )
    if best is None or s < best[0]:
        best = (s, prm)
for seed in (1, 2, 3):
    pr, it = run(Xn, f_tr, f_va, f_te, cats, dict(best[1], seed=seed))
    r = score(pr, f_te)
    res.setdefault("native, tuned on August", []).append([*r, it])
    print(f"native tuned (seed {seed}): RMSE {r[0]:.3f} MAE {r[1]:.3f} rounds {it}", flush=True)
res["native tuned params"] = {k: v for k, v in best[1].items() if k != "max_bin_by_feature"}
# 3. monthly retrain
dec = json.loads((C.OUT / "ts_decompose.json").read_text(encoding="utf-8"))["chosen"]
prior_hl = dec["global"][0]
online = [C.ordered_ts(t, av, K["global"], y, half_life=dec["global"][0], prior_weight=1.0)[0]]
for c in LOW + HIGH:
    hl, a = dec[c]
    online.append(
        C.ordered_ts(t, av, K[c], y, half_life=hl, prior_weight=a, prior_half_life=prior_hl)[0]
    )
lhl, la = dec["long"]
online += [
    C.ordered_ts(t, av, K[c], y, half_life=lhl, prior_weight=la, prior_half_life=prior_hl)[0]
    for c in HIGH
]
for name, extra in (
    ("monthly retrain, static encoder", False),
    ("monthly retrain, static encoder + online statistics", True),
):
    pred = np.full(len(t), np.nan)
    for start, end in MONTHS:
        known = av < start
        tr, va, te = (
            known & (t < start - 30 * DAY),
            known & (t >= start - 30 * DAY),
            (t >= start) & (t < end),
        )
        e = TargetEncoder(target_type="continuous", cv=5)
        Xe = np.empty((len(t), 4))
        Xe[tr] = e.fit_transform(H[tr], y[tr])
        Xe[~tr] = e.transform(H[~tr])
        X = np.column_stack([base, Xe, *online]) if extra else np.column_stack([base, Xe])
        pred[te], _ = run(X, tr, va, te, [4, 5])
    r = score(pred[f_te], f_te)
    res[name] = list(r)
    print(f"{name}: RMSE {r[0]:.3f} MAE {r[1]:.3f}", flush=True)
(C.OUT / "ts_baselines.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
