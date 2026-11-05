"""v3: the bank's own `embargo` already does what label_clock= needs, for a
constant delay. Each held row keeps the prediction and the conformal radius
it was served with, and its release scores the label against them. Flights'
residual stream y - trees, an intercept model (a column of zeros), 90%
conformal: the doubled stream (labels at their arrival) against one stream
under embargo = 8 hours, which is later than 99.5% of the labels arrive."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from leafrefit import train_trees  # noqa: E402

import polars_online as po  # noqa: E402

print(C.load_line(), flush=True)
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
yf = df["y"].to_numpy()
tr, va, te = t < 243 * 1440.0, (t >= 243 * 1440.0) & (t < 273 * 1440.0), t >= 273 * 1440.0
K = [C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")]
X = np.column_stack([df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")] + K)
b = train_trees(X, yf, tr, va, 31, 1000)
r = yf - b.predict(X, num_iteration=b.best_iteration)
n = len(r)
print(f"labels later than 8 hours: {np.mean(av - t > 480):.4f}")


def spec(**kw):
    return po.spec.ewridge(
        "o",
        targets=["y"],
        features=["zero"],
        clock="c",
        gap_cap=600.0,
        half_life=7 * 1440.0,
        conformal=0.9,
        ridge=1e-6,
        **kw,
    )


one = pl.DataFrame({"c": t, "y": r, "zero": np.zeros(n)})
for kw in ({}, {"conformal_rate": 0.01}, {"conformal_rate": 0.002}, {"resid_quantiles": [0.9]}):
    o = po.ModelBank([spec(embargo=480.0, **kw)]).fit_predict(one)
    f = o["o"].struct
    p = f.field("pred_y").to_numpy()
    if "resid_quantiles" in kw:
        q = f.field("abs_resid_q0.9_y").to_numpy()
        lo, hi = p - q, p + q
    else:
        lo, hi = f.field("lo_y").to_numpy(), f.field("hi_y").to_numpy()
    ok = te & ~np.isnan(lo)
    cov = np.mean((r[ok] >= lo[ok]) & (r[ok] <= hi[ok]))
    half = np.mean(hi[ok] - lo[ok]) / 2
    err = np.sqrt(np.mean((r[ok] - p[ok]) ** 2))
    print(
        f"one stream, embargo 8 h, {kw or 'conformal_rate 0.05'}: served coverage {cov:.4f}, "
        f"half-width {half:.2f}, RMSE {err:.3f}",
        flush=True,
    )
print(C.load_line(), flush=True)
