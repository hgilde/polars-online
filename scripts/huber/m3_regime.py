# A level shift at row 10,000 (+20 on y, no outliers): rows until the prediction error's
# EW mean returns within 0.5 of the noise sd, huber vs ewridge, half_life 500.
import warnings

import numpy as np
import polars as pl

import polars_online as po

warnings.simplefilter("ignore")
rng = np.random.default_rng(1)
n = 20_000
x = rng.standard_normal(n)
y = 2.0 * x + 0.5 * rng.standard_normal(n)
y[10_000:] += 20.0
df = pl.DataFrame({"x": x, "y": y, "t": np.arange(n, dtype=float)})
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0)
for name, spec in [
    ("ewridge", po.spec.ewridge("m", **common)),
    ("huber", po.spec.huber("m", **common)),
    ("huber delta=3", po.spec.huber("m", huber_delta=3.0, **common)),
]:
    out = po.ModelBank([spec]).fit_predict(df)
    pred = out.select(pl.col("m").struct.field("pred_y")).to_series().to_numpy()
    err = np.abs(y - pred)[10_000:]
    # first row after the shift from which a trailing 200-row mean |err| stays under 1.0
    m = np.convolve(err, np.ones(200) / 200, mode="valid")
    back = next((i for i in range(len(m)) if np.all(m[i : i + 500] < 1.0)), None)
    print(
        f"{name:>14}: rows to re-converge after a +20 level shift: {back}; "
        f"mean |err| rows 0-200 after shift {err[:200].mean():.2f}, "
        f"200-400 {err[200:400].mean():.2f}, 1000-1200 {err[1000:1200].mean():.2f}"
    )
