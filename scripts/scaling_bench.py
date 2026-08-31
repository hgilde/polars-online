"""Thread-scaling row for the CI benchmark summary (docs/PERFORMANCE.md P8).

Throughput alone hides the thing most likely to regress: a change that
serializes the fan-out looks fine on one thread and costs 4x on ten. This
re-runs one grouped workload at several `RAYON_NUM_THREADS` in subprocesses --
the pool size is fixed at first use, so it cannot be varied in-process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

WORKLOAD = """
import os, time, numpy as np, polars as pl, polars_online as po
rows, k, groups = {rows}, 20, 64
rng = np.random.default_rng(0)
d = {{"t": np.arange(float(rows))}}
for i in range(k):
    d[f"x{{i}}"] = rng.standard_normal(rows)
d["y"] = rng.standard_normal(rows)
d["g"] = np.arange(rows) % groups
df = pl.DataFrame(d)
spec = po.spec.ewridge("m", targets=["y"], features=[f"x{{i}}" for i in range(k)],
                       clock="t", max_dclock=10.0, halflife=1000.0,
                       min_periods=25.0, group="g")
po.ModelBank([spec]).fit_predict(df)
b = po.ModelBank([spec])
t = time.perf_counter(); b.fit_predict(df); dt = time.perf_counter() - t
print(rows / dt)
"""


def run(rows: int, threads: int) -> float:
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    out = subprocess.run(
        [sys.executable, "-c", WORKLOAD.format(rows=rows)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    counts = [1, 2, 4, min(8, os.cpu_count() or 8)]
    counts = sorted(set(c for c in counts if c >= 1))
    base = None
    rowsps: list[tuple[int, float]] = []
    for n in counts:
        r = run(args.rows, n)
        rowsps.append((n, r))
        if n == 1:
            base = r

    if args.markdown:
        print("| threads | rows/sec | speedup |")
        print("|---|---|---|")
        for n, r in rowsps:
            sp = f"{r / base:.1f}x" if base else "-"
            print(f"| {n} | {r:,.0f} | {sp} |")
    else:
        for n, r in rowsps:
            print(f"threads={n}: {r:,.0f} rows/s")


if __name__ == "__main__":
    main()
