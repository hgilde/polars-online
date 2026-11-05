"""v3, review finding 14c: input drift of a deployed GBDT, read against
po.gbdt's own cuts, as a recipe.

The training matrix's cuts (QuantileDMatrix.get_quantile_cut) bin each
feature; the training rows' bin shares are the reference; a serving bin's
share is decayed on the clock (half-life h, labels not needed). The
population stability index PSI = sum (p - q) ln(p / q), each share floored
at 1e-4, is read per feature. BTCUSDT: the lag features against days 1-40,
read daily over days 41-60. Planted: one feature's mean shifts by 0.5
standard deviations at step 2,000 of 3,000; how many steps until its PSI
passes 0.1 and 0.2, and the largest PSI of an unshifted feature."""

import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import polars_online as po  # noqa: E402


def cuts_of(frame, label, features):
    m = po.gbdt.xgboost_matrix(frame, label=label, features=features, max_bin=32, engine="python")
    ptr, cuts = m.get_quantile_cut()
    return [np.asarray(cuts[ptr[j] + 1 : ptr[j + 1]]) for j in range(len(features))]


def bins(x, c):
    return np.searchsorted(c, x, side="right")


def decayed_psi(clock, b, ref, h, every):
    """PSI of the decayed serving shares against `ref`, read at each row index in `every`."""
    k = len(ref)
    hist = np.zeros(k)
    out = {}
    last = clock[0]
    ln2 = math.log(2.0)
    want = set(every)
    for i in range(len(clock)):
        hist *= math.exp(-ln2 * (clock[i] - last) / h)
        last = clock[i]
        hist[b[i]] += 1.0
        if i in want:
            q = np.maximum(hist / hist.sum(), 1e-4)
            p = np.maximum(ref, 1e-4)
            out[i] = float(np.sum((q - p) * np.log(q / p)))
    return out


print(C.load_line(), flush=True)
res = {}
fs = sorted(glob.glob(str(C.SLOT / ".cache" / "BTCUSDT-1m-2024-*.parquet")))[:60]
d = pl.concat([pl.read_parquet(f) for f in fs]).sort("t").with_columns(z=pl.col("n_trades").log1p())
d = d.with_columns(
    [pl.col("z").ewm_mean(half_life=h).shift(1).alias(f"zm{h}") for h in (30, 1440)]
).slice(1)
F = ["zm30", "zm1440"]
day = ((d["t"] - d["t"][0]) // 86400).to_numpy()
train = d.filter(pl.Series(day < 40))
cuts = cuts_of(train, "z", F)
clock = np.arange(d.height, dtype=float)
read = [int(np.searchsorted(day, k)) - 1 for k in range(41, 61)]
for j, f in enumerate(F):
    x = d[f].to_numpy()
    b = bins(x, cuts[j])
    ref = np.bincount(b[day < 40], minlength=len(cuts[j]) + 1) / (day < 40).sum()
    psi = decayed_psi(clock, b, ref, 1440.0, read)
    res[f"btc {f}"] = [round(v, 3) for v in psi.values()]
    print(
        f"BTCUSDT {f}, PSI at the end of days 41-60 (half-life 1 day): {res[f'btc {f}']}",
        flush=True,
    )
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    steps, per = 3000, 100
    clock = np.repeat(np.arange(steps, dtype=float), per)
    x = rng.standard_normal((steps * per, 3))
    x[clock >= 2000, 0] += 0.5
    tr = clock < 1000
    frame = pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": rng.standard_normal(len(clock))}
    )
    cuts = cuts_of(frame.filter(pl.Series(tr)), "y", ["x0", "x1", "x2"])
    every = list(range(int(1500 * per), len(clock), per))
    first = {}
    worst_null = 0.0
    for j in range(3):
        b = bins(x[:, j], cuts[j])
        ref = np.bincount(b[tr], minlength=len(cuts[j]) + 1) / tr.sum()
        psi = decayed_psi(clock, b, ref, 20.0, every)
        if j == 0:
            for thr in (0.1, 0.2):
                hit = [clock[i] for i, v in psi.items() if v > thr and clock[i] >= 2000]
                first[thr] = (hit[0] - 2000) if hit else None
            pre = max(v for i, v in psi.items() if clock[i] < 2000)
        else:
            worst_null = max(worst_null, max(psi.values()))
    res[f"planted {seed}"] = {
        "steps to PSI > 0.1": first[0.1],
        "steps to PSI > 0.2": first[0.2],
        "largest PSI of x0 before the shift": pre,
        "largest PSI of an unshifted feature": worst_null,
    }
    print(f"planted {seed}: {res[f'planted {seed}']}", flush=True)
(C.OUT / "psi_drift.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
