"""(a) ewridge with emit_breaks at k=20 and k=64: release vs main, ABBA."""
import sys
import time

import numpy as np
import polars as pl

import polars_online as po
from _abba import main

ROWS = {20: 150_000, 64: 15_000}


def make(rows, k):
    rng = np.random.default_rng(0)
    d = {f"x{j}": rng.standard_normal(rows) for j in range(k)}
    x = np.column_stack(list(d.values()))
    d["y"] = x @ rng.standard_normal(k) + 0.5 * rng.standard_normal(rows)
    d["t"] = np.cumsum(rng.exponential(1.0, rows))
    return pl.DataFrame(d)


def case(k):
    df = make(ROWS[k], k)
    spec = po.spec.ewridge("m", targets=["y"], features=[f"x{j}" for j in range(k)], clock="t",
                           half_life=1000.0, gap_cap=100.0, min_weight=float(k), emit_breaks=True)

    def run():
        b = po.ModelBank([spec])
        t = time.perf_counter()
        b.fit_predict(df)
        return time.perf_counter() - t

    return run, ROWS[k]


if __name__ == "__main__":
    main(__file__, lambda: {f"emit_breaks k={k}": case(k) for k in ROWS})
