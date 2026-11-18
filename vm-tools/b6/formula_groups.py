"""(c) a formula target (look-ahead window mean) over 1,000 groups through the bank."""
import time

import numpy as np
import polars as pl

import polars_online as po
from _abba import main

ROWS, G = 2_000_000, 1000


def make_cases():
    rng = np.random.default_rng(0)
    d = {f"x{j}": rng.standard_normal(ROWS) for j in range(5)}
    d["y0"] = d["x0"] + 0.5 * rng.standard_normal(ROWS)
    d["t"] = np.arange(float(ROWS))
    d["g"] = np.arange(ROWS) % G
    df = pl.DataFrame(d)
    fwd = po.ewm_mean("y0", half_life=20.0, window_size=40.0, reverse=True).alias("fwd")
    spec = po.spec.ewridge("m", targets=[fwd], features=[f"x{j}" for j in range(5)], clock="t",
                           gap_cap=5000.0, half_life=100000.0, group="g", embargo=40.0 * G,
                           min_weight=5.0)

    def run():
        b = po.ModelBank([spec])
        t = time.perf_counter()
        b.fit_predict(df)
        return time.perf_counter() - t

    return {"formula target, 1000 groups": (run, ROWS)}


if __name__ == "__main__":
    main(__file__, make_cases)
