"""What the screen's pairs buy: XGBoost interaction_constraints and product
terms for a linear model, on the planted data (n=200k: train 140k, valid 10k,
test 50k)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402
from pairs import TRUE, P, anova, binned, edges_of, make, residual_online, tables  # noqa: E402

import polars_online as po  # noqa: E402


def xgb_rmse(x, y, tr, va, te, cons, n_train=None):
    p = {"tree_method": "hist", "max_depth": 6, "eta": 0.1, "nthread": 4, "seed": 1}
    if cons is not None:
        p["interaction_constraints"] = str(cons)
    dtr = xgb.DMatrix(x[tr], y[tr])
    dva = xgb.DMatrix(x[va], y[va])
    dte = xgb.DMatrix(x[te])
    b = xgb.train(p, dtr, 2000, evals=[(dva, "v")], early_stopping_rounds=50, verbose_eval=False)
    pr = b.predict(dte, iteration_range=(0, b.best_iteration + 1))
    return float(np.sqrt(np.mean((pr - y[te]) ** 2))), b.best_iteration


def cons_of(pairs):
    used = set(i for pq in pairs for i in pq)
    return [list(pq) for pq in pairs] + [[i] for i in range(P) if i not in used]


def lin_rmse(x, y, te, pairs):
    cols = (
        {f"x{i}": x[:, i] for i in range(P)}
        | {f"q{i}": x[:, i] ** 2 for i in range(P)}
        | {f"p{i}_{j}": x[:, i] * x[:, j] for i, j in pairs}
    )
    d = pl.DataFrame({"c": np.arange(len(y), dtype=float), "y": y, **cols})
    s = po.spec.ewridge(
        "o", targets=["y"], features=list(cols), clock="c", gap_cap=10.0, half_life=1e9, ridge=1e-4
    )
    p = po.ModelBank([s]).fit_predict(d)["o"].struct.field("pred_y").to_numpy()
    return float(np.sqrt(np.mean((p[te] - y[te]) ** 2)))


print(C.load_line())
res = {}
for seed in (0, 1, 2):
    n = 200_000
    x, y = make(seed, n)
    idx = np.arange(n)
    tr, va, te = idx < 140_000, (idx >= 140_000) & (idx < 150_000), idx >= 150_000
    B = 8
    e = edges_of(x, 2000, B)
    b = binned(x, e)
    r = residual_online(x, y, e, B)
    # the screen sees the training rows only
    T = tables(b[:140_000], r[:140_000], B)
    sc = {k: anova(v) for k, v in T.items()}
    ranked = sorted(sc, key=lambda k: -sc[k])
    row = {}
    row["xgb unconstrained"] = xgb_rmse(x, y, tr, va, te, None)
    for k in (4, 8):
        row[f"xgb top-{k} screened pairs"] = xgb_rmse(x, y, tr, va, te, cons_of(ranked[:k]))
    row["xgb true pairs"] = xgb_rmse(x, y, tr, va, te, cons_of(sorted(TRUE)))
    row["xgb no interactions"] = xgb_rmse(x, y, tr, va, te, cons_of([]))
    row["ewridge x, x^2"] = lin_rmse(x, y, te, [])
    row["ewridge + top-8 screened products"] = lin_rmse(x, y, te, ranked[:8])
    row["ewridge + true products"] = lin_rmse(x, y, te, sorted(TRUE))
    row["top-8"] = [list(k) for k in ranked[:8]]
    res[seed] = row
    print(seed, json.dumps(row), flush=True)
Path(str(C.OUT / "pairs_value.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
