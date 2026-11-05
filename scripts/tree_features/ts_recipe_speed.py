import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import polars as pl

import polars_online as po

df = C.load_flights()
print(C.load_line())
for g in ("origin", "dest", "tailnum", "flight_id"):
    base = df.select("t", "avail", "y", g).with_row_index("rid")
    s = pl.concat(
        [
            base.select(
                "rid",
                g,
                c=pl.col("t"),
                y=pl.lit(None, pl.Float64),
                one=pl.lit(None, pl.Float64),
                role=pl.lit(0, pl.Int8),
            ),
            base.select(
                "rid", g, c=pl.col("avail"), y=pl.col("y"), one=pl.lit(1.0), role=pl.lit(1, pl.Int8)
            ),
        ]
    ).sort("c", "role", "rid")
    for lazy in (False, True):
        t0 = time.perf_counter()
        src = s.lazy() if lazy else s
        o = src.online.with_windows(
            S=po.ewm_sum("y", half_life=43200.0, closed="left"),
            W=po.ewm_sum("one", half_life=43200.0, closed="left"),
            clock="c",
            gap_cap=1e12,
            group=g,
        )
        if lazy:
            o = o.collect()
        el = time.perf_counter() - t0
        print(
            f"group={g:9s} groups {df[g].n_unique():5d} {'lazy ' if lazy else 'eager'}: {el:6.2f}s, {el / s.height * 1e9:7.0f} ns a row",  # noqa: E501
            flush=True,
        )
