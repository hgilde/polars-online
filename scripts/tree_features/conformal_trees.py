"""Candidate 7 (monitoring/intervals around a tree model): 90% intervals for
frozen trees' predictions in the test period: a static split-conformal radius
from the validation residuals, against the bank's own conformal tracker run
on the residual stream y - tree_pred (an intercept model, a zero feature: what
an offset would give with no features)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings

warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from leafrefit import synth, train_trees  # noqa: E402

import polars_online as po  # noqa: E402


def online_interval(clock, r, hl, gap, learn_clock=None):
    d = pl.DataFrame(
        {
            "c": clock,
            "y": r,
            "zero": np.zeros(len(r)),
            "rid": np.arange(len(r)),
            "role": np.zeros(len(r), np.int8),
        }
    )
    if learn_clock is not None:
        d = pl.concat(
            [
                d.with_columns(y=pl.lit(None, pl.Float64)),
                d.with_columns(c=pl.Series(learn_clock), role=pl.lit(1, pl.Int8)),
            ]
        ).sort("c", "role", "rid")
    s = po.spec.ewridge(
        "o",
        targets=["y"],
        features=["zero"],
        clock="c",
        gap_cap=gap,
        half_life=hl,
        conformal=0.9,
        ridge=1e-6,
    )
    o = po.ModelBank([s]).fit_predict(d).filter(pl.col("role") == 0).sort("rid")
    f = o["o"].struct
    return f.field("pred_y").to_numpy(), f.field("lo_y").to_numpy(), f.field("hi_y").to_numpy()


def report(name, y, base, va, te, clock, hl, gap, learn_clock=None):
    q = np.quantile(np.abs(y[va] - base[va]), 0.9)
    cov_static = float(np.mean(np.abs(y[te] - base[te]) <= q))
    p, lo, hi = online_interval(clock, y - base, hl, gap, learn_clock)
    ok = te & ~np.isnan(lo)
    cov_online = float(np.mean((y[ok] - base[ok] >= lo[ok]) & (y[ok] - base[ok] <= hi[ok])))
    w_online = float(np.mean(hi[ok] - lo[ok]) / 2)
    # coverage by quarter of the test period: does the static one hold up?
    qs = np.array_split(np.where(te)[0], 4)
    by_q = [round(float(np.mean(np.abs(y[i] - base[i]) <= q)), 3) for i in qs]
    by_qo = [
        round(float(np.mean((y[i] - base[i] >= lo[i]) & (y[i] - base[i] <= hi[i]))), 3) for i in qs
    ]
    r = {
        "static coverage": cov_static,
        "static half-width": float(q),
        "static by quarter": by_q,
        "online coverage": cov_online,
        "online mean half-width": w_online,
        "online by quarter": by_qo,
        "rmse frozen": float(np.sqrt(np.mean((y[te] - base[te]) ** 2))),
        "rmse frozen + online bias": float(
            np.sqrt(np.mean((y[te] - base[te] - np.nan_to_num(p[te])) ** 2))
        ),
    }
    print(name, json.dumps(r), flush=True)
    return r


print(C.load_line())
res = {}
for seed in (0, 1, 2):
    clock, X, y = synth(seed)
    S = clock.max() + 1
    tr, va, te = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S), clock >= 0.5 * S
    b = train_trees(X, y, tr, va, 15, 300)
    base = b.predict(X, num_iteration=b.best_iteration)
    res[f"synth{seed}"] = report(f"synth{seed}", y, base, va, te, clock, 20.0, 10.0)
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
yf = df["y"].to_numpy()
tr, va, te = t < 243 * 1440.0, (t >= 243 * 1440.0) & (t < 273 * 1440.0), t >= 273 * 1440.0
K = [C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")]
X = np.column_stack([df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")] + K)
b = train_trees(X, yf, tr, va, 31, 1000)
base = b.predict(X, num_iteration=b.best_iteration)
res["flights"] = report("flights", yf, base, va, te, t, 7 * 1440.0, 600.0, learn_clock=av)
Path(str(C.OUT / "conformal_trees.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line())
