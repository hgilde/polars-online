"""(f) kmeans re-check: 4 features K=8 and k=20 K=8, as in scripts/benchmark.py."""
import time

import numpy as np
import polars as pl

import polars_online as po
from _abba import main

ROWS = 2_000_000


def make_cases():
    rng = np.random.default_rng(0)
    d = {f"x{j}": rng.standard_normal(ROWS) for j in range(20)}
    lab = rng.integers(0, 3, ROWS)
    for j in range(4):
        d[f"b{j}"] = d[f"x{j}"] + 6.0 * (lab == j % 3)
    d["t"] = np.cumsum(rng.exponential(1.0, ROWS))
    df = pl.DataFrame(d)
    base = dict(clock="t", half_life=1000.0, gap_cap=100.0, min_weight=20.0)
    s4 = po.spec.kmeans("m", n_clusters=8, features=[f"b{j}" for j in range(4)], **base)
    s20 = po.spec.kmeans("m", n_clusters=8, features=[f"x{j}" for j in range(20)], **base)

    def mk(spec):
        def run():
            b = po.ModelBank([spec])
            t = time.perf_counter()
            b.fit_predict(df)
            return time.perf_counter() - t
        return run, ROWS

    return {"kmeans 4 features K=8": mk(s4), "kmeans k=20 K=8": mk(s20)}


if __name__ == "__main__":
    main(__file__, make_cases)
