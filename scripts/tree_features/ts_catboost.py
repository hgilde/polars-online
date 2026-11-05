"""v3, review finding 10: CatBoost, whose ordered target statistic is the
method of section 2, as the batch baseline on flights. has_time=True, so its
statistics follow the rows' order (scheduled departure), the six
categoricals given as categories, trained January-August with early
stopping on September, tested October-December, 4 threads; then the same
with ts_decompose.py's online statistics added as numeric features.
CatBoost is not a dependency: run it in a throwaway environment,

    uv run --no-project --python 3.12 --with catboost --with pandas \
        --with numpy --with polars python -I ts_catboost.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor, Pool  # noqa: E402

DAY = 1440.0
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
AUG, SEP = 243 * DAY, 273 * DAY
NUM = ["dep_min", "arr_min", "wday", "distance"]
CATS = ["carrier", "origin", "tailnum", "dest", "route", "flight_id"]
tr, va, te = t < AUG, (t >= AUG) & (t < SEP), t >= SEP
dec = json.loads((C.OUT / "ts_decompose.json").read_text(encoding="utf-8"))["chosen"]
K = {c: C.codes(df[c]) for c in CATS}
K["global"] = np.zeros(len(t), dtype=np.int64)
prior_hl = dec["global"][0]
online = {"global": C.ordered_ts(t, av, K["global"], y, half_life=prior_hl, prior_weight=1.0)[0]}
for c in ["carrier", "origin", "tailnum", "dest", "route", "flight_id"]:
    hl, a = dec[c]
    online[f"{c}_recent"] = C.ordered_ts(
        t, av, K[c], y, half_life=hl, prior_weight=a, prior_half_life=prior_hl
    )[0]
lhl, la = dec["long"]
for c in ["tailnum", "dest", "route", "flight_id"]:
    online[f"{c}_long"] = C.ordered_ts(
        t, av, K[c], y, half_life=lhl, prior_weight=la, prior_half_life=prior_hl
    )[0]

frame = pd.DataFrame({**{c: df[c].to_numpy() for c in NUM}, **{c: df[c].to_list() for c in CATS}})
print(C.load_line(), flush=True)
res: dict = {}
for name, extra in (
    ("CatBoost, has_time", False),
    ("CatBoost, has_time, + online statistics", True),
):
    X = frame.assign(**online) if extra else frame
    for seed in (1, 2, 3):
        m = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            has_time=True,
            thread_count=4,
            random_seed=seed,
            verbose=0,
            od_type="Iter",
            train_dir=str(C.OUT / "catboost_info"),
            od_wait=200,
            use_best_model=True,
        )
        m.fit(Pool(X[tr], y[tr], cat_features=CATS), eval_set=Pool(X[va], y[va], cat_features=CATS))
        pr = m.predict(Pool(X[te], cat_features=CATS))
        r = [
            float(np.sqrt(np.mean((pr - y[te]) ** 2))),
            float(np.mean(np.abs(pr - y[te]))),
            int(m.get_best_iteration()),
        ]
        res.setdefault(name, []).append(r)
        print(f"{name} seed {seed}: RMSE {r[0]:.3f} MAE {r[1]:.3f} iterations {r[2]}", flush=True)
(C.OUT / "ts_catboost.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
