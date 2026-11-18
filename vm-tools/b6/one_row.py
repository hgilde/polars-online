"""(e) one-row latency: fit_predict on a one-row frame, median of 2,000 calls."""
import statistics
import time

import polars as pl

import polars_online as po
from _abba import main

CALLS = 2000


def make_cases():
    k = 5
    feats = [f"x{j}" for j in range(k)]
    spec = po.spec.ewridge("m", targets=["y"], features=feats, clock="t", gap_cap=10.0,
                           half_life=1000.0, min_weight=5.0)
    bank = po.ModelBank([spec])
    warm = pl.DataFrame({**{f: [0.1 * (i % 7) for i in range(200)] for f in feats},
                         "y": [float(i % 5) for i in range(200)], "t": [float(i) for i in range(200)]})
    bank.fit_predict(warm)
    state = {"t": 200.0}

    def run():
        ts = []
        for _ in range(CALLS):
            state["t"] += 1.0
            row = pl.DataFrame({**{f: [0.3] for f in feats}, "y": [1.0], "t": [state["t"]]})
            t0 = time.perf_counter()
            bank.fit_predict(row)
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    return {"one-row fit_predict (median of 2000)": (run, None)}


if __name__ == "__main__":
    main(__file__, make_cases)
