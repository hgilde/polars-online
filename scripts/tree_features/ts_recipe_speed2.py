import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np
import polars as pl

import polars_online as po

print(C.load_line())
rng = np.random.default_rng(0)
for G in (10, 100, 1000, 10000):
    for n in (100_000, 400_000):
        s = pl.DataFrame(
            {
                "c": np.arange(n, dtype=float),
                "g": rng.integers(0, G, n),
                "y": rng.standard_normal(n),
            }
        )
        t0 = time.perf_counter()
        s.online.with_windows(
            S=po.ewm_sum("y", half_life=1000.0, closed="left"), clock="c", gap_cap=1e12, group="g"
        )
        el = time.perf_counter() - t0
        print(f"G={G:6d} n={n:7d}: {el:6.2f}s {el / n * 1e9:8.0f} ns a row", flush=True)
