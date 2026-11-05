"""Is a decayed, embargoed ordered TS already a recipe with po.ewm_sum?
A doubled stream (a predict row at t with y null, a learn row at avail with
y), with_windows(group=category, closed="left"); compared with the
prototype's per-row loop."""

import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np
import polars as pl

import polars_online as po

df = C.load_flights()
hl = 30 * 1440.0
base = df.select("t", "avail", "y", "tailnum").with_row_index("rid")
pred_rows = base.select(
    "rid",
    "tailnum",
    c=pl.col("t"),
    y=pl.lit(None, pl.Float64),
    one=pl.lit(None, pl.Float64),
    role=pl.lit(0, pl.Int8),
)
learn_rows = base.select(
    "rid", "tailnum", c=pl.col("avail"), y=pl.col("y"), one=pl.lit(1.0), role=pl.lit(1, pl.Int8)
)
stream = pl.concat([pred_rows, learn_rows]).sort("c", "role", "rid")
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
out = stream.online.with_windows(
    S=po.ewm_sum("y", half_life=hl, closed="left"),
    W=po.ewm_sum("one", half_life=hl, closed="left"),
    clock="c",
    gap_cap=1e12,
    group="tailnum",
)
el = time.perf_counter() - t0
got = out.filter(pl.col("role") == 0).sort("rid")
# reference: the prototype's loop, unshrunk sums
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
y = df["y"].to_numpy()
k = C.codes(df["tailnum"])
order = np.argsort(av, kind="stable")
j = 0
tab = {}
Wr = np.empty(len(t))
Sr = np.empty(len(t))
ln2 = math.log(2)
for i in range(len(t)):
    while j < len(t) and av[order[j]] < t[i]:
        r = order[j]
        e = tab.get(k[r])
        if e is None:
            tab[k[r]] = [1.0, y[r], av[r]]
        else:
            f = math.exp(-ln2 * (av[r] - e[2]) / hl)
            e[0] = e[0] * f + 1
            e[1] = e[1] * f + y[r]
            e[2] = av[r]
        j += 1
    e = tab.get(k[i])
    if e is None:
        Wr[i] = 0
        Sr[i] = 0
    else:
        f = math.exp(-ln2 * (t[i] - e[2]) / hl)
        Wr[i] = e[0] * f
        Sr[i] = e[1] * f
W = got["W"].fill_null(0.0).to_numpy()
S = got["S"].fill_null(0.0).to_numpy()
print(
    f"groups {df['tailnum'].n_unique()}, stream rows {stream.height}, with_windows {el:.2f}s ({el / stream.height * 1e9:.0f} ns a row)"  # noqa: E501
)
print(
    "max |W - ref| / (1+W):",
    float(np.max(np.abs(W - Wr) / (1 + Wr))),
    " max |S - ref| / (1+|S|):",
    float(np.max(np.abs(S - Sr) / (1 + np.abs(Sr)))),
)
print(
    "null W rows (no label yet):",
    int(got["W"].null_count()),
    "ref zero-W rows:",
    int((Wr == 0).sum()),
)
