"""The README's Parallelism figures (docs/PERFORMANCE.md §12, re-run in §28).

    uv run python scripts/parallel_bench.py [OUTDIR]

Three workloads, each timed in a fresh process, since both thread pools are
fixed at their first use:

1. the ticks grid: the README's six `ewridge` specs (three factor sets, with
   and without `standardize`) over 2.56M rows in 64 groups, through
   `lf.online.fit_predict(chunk_rows=200_000, save_state=...)` into
   `sink_parquet`, at `POLARS_ONLINE_MAX_THREADS` 1 and 14;
2. eight single-group specs, k=20 over 300k rows, `half_life=1000*j`,
   `gap_cap=10`: one bank, against one bank per spec in turn;
3. the two knobs: 12M rows over 64 groups, one spec of four features and two
   half-lives, 200k chunks, at (`POLARS_MAX_THREADS`,
   `POLARS_ONLINE_MAX_THREADS`) of (14, 14) and (4, 14): wall time, and the
   peak memory footprint `/usr/bin/time -l` reports (macOS).

The input is generated, seeded, into OUTDIR (default `.cache/parallel_bench`,
which git ignores), and reused by later runs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

GRID = """
import time, polars as pl, polars_online as po
from itertools import product
factors = {{"mkt": ["x0"], "mkt-sz": ["x0", "x1"], "mkt-sz-val": ["x0", "x1", "x2"]}}
def spec(name, features, standardize):
    return po.spec.ewridge(f"{{name}}-std{{standardize:d}}", targets=["y"], features=features,
                           clock="t", gap_cap=300.0, group="stock_id", session="session",
                           session_gap=60.0, half_life=[100.0, 1000.0], ridge=[1e-3, 0.1],
                           standardize=standardize)
specs = [spec(n, f, s) for (n, f), s in product(factors.items(), [False, True])]
t0 = time.perf_counter()
(pl.scan_parquet("{src}").online.fit_predict(specs, chunk_rows=200_000, save_state="{state}")
   .sink_parquet("{dst}"))
print(time.perf_counter() - t0)
"""

EIGHT = """
import time, numpy as np, polars as pl, polars_online as po
rng = np.random.default_rng(0)
rows, k = 300_000, 20
d = {"t": np.arange(float(rows))}
for i in range(k):
    d[f"x{i}"] = rng.standard_normal(rows)
d["y"] = rng.standard_normal(rows)
df = pl.DataFrame(d)
specs = [po.spec.ewridge(f"m{j}", targets=["y"], features=[f"x{i}" for i in range(k)],
                         clock="t", gap_cap=10.0, half_life=1000.0 * j) for j in range(1, 9)]
po.ModelBank(specs).fit_predict(df)   # the pool is built, and the pages are warm
best_one, best_each = 1e9, 1e9
for _ in range(3):
    b = po.ModelBank(specs)
    t0 = time.perf_counter(); b.fit_predict(df); best_one = min(best_one, time.perf_counter() - t0)
    t0 = time.perf_counter()
    for s in specs:
        po.ModelBank([s]).fit_predict(df)
    best_each = min(best_each, time.perf_counter() - t0)
print(best_one, best_each)
"""

KNOBS = """
import time, polars as pl, polars_online as po
spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1", "x2", "x3"], clock="t",
                       gap_cap=10.0, half_life=[1000.0, 5000.0], group="stock_id")
t0 = time.perf_counter()
pl.scan_parquet("{src}").online.fit_predict([spec], chunk_rows=200_000).sink_parquet("{dst}")
print(time.perf_counter() - t0)
"""


def make_ticks(path: Path, rows: int, k: int, groups: int) -> None:
    """Interleaved groups on one clock, ten sessions, a linear target."""
    if path.exists():
        return
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, k))
    beta = rng.standard_normal(k)
    data: dict[str, object] = {
        "t": np.arange(float(rows)),
        "stock_id": [f"s{i % groups}" for i in range(rows)],
        "session": np.arange(rows) // (rows // 10),
    }
    for j in range(k):
        data[f"x{j}"] = x[:, j]
    data["y"] = x @ beta + 0.5 * rng.standard_normal(rows)
    pl.DataFrame(data).write_parquet(path, row_group_size=100_000)


def timed(code: str, env: dict[str, str]) -> tuple[str, float]:
    """Run `code` in a fresh interpreter: its stdout, and its peak memory in
    GB, NaN where the platform does not report it. macOS's BSD `time -l`
    gives the peak memory footprint; GNU `time -v` on Linux gives the peak
    resident set, a close but not identical measure; Windows has neither
    (task 160, SC4: `-l` alone failed on Linux)."""
    if sys.platform == "darwin":
        prefix, pattern, unit = ["/usr/bin/time", "-l"], r"(\d+)\s+peak memory footprint", 1.0
    elif sys.platform.startswith("linux") and Path("/usr/bin/time").exists():
        prefix, pattern, unit = (
            ["/usr/bin/time", "-v"],
            r"Maximum resident set size \(kbytes\):\s*(\d+)",
            1024.0,
        )
    else:
        prefix, pattern, unit = [], "", 0.0
    r = subprocess.run(
        [*prefix, sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(pattern, r.stderr) if pattern else None
    return r.stdout.strip(), (int(m.group(1)) * unit / 1e9 if m else float("nan"))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/parallel_bench")
    out.mkdir(parents=True, exist_ok=True)
    ticks, wide = out / "ticks.parquet", out / "wide.parquet"
    make_ticks(ticks, 2_560_000, 3, 64)
    make_ticks(wide, 12_000_000, 4, 64)

    print("## the ticks grid, six specs over 2.56M rows (best of 3)")
    for threads in (1, 14):
        runs = [
            float(
                timed(
                    GRID.format(src=ticks, state=out / "grid.state", dst=out / "grid.parquet"),
                    {"POLARS_ONLINE_MAX_THREADS": str(threads)},
                )[0]
            )
            for _ in range(3)
        ]
        shown = ", ".join(f"{r:.2f}" for r in runs)
        print(f"bank threads {threads}: {min(runs):.2f} s  (runs {shown})")

    print("## eight single-group specs, k=20, 300k rows (best of 3, in-process)")
    one, each = (float(v) for v in timed(EIGHT, {})[0].split())
    print(f"one bank {one * 1000:.0f} ms, one at a time {each * 1000:.0f} ms")

    print("## two knobs: 12M rows, 64 groups, k=4, two half_lives, 200k chunks (best of 2)")
    for polars_threads, bank_threads in ((14, 14), (4, 14)):
        env = {
            "POLARS_MAX_THREADS": str(polars_threads),
            "POLARS_ONLINE_MAX_THREADS": str(bank_threads),
        }
        runs = []
        for _ in range(2):
            text, gb = timed(KNOBS.format(src=wide, dst=out / "knobs.parquet"), env)
            runs.append((float(text), gb))
        secs, gb = min(runs)
        print(f"polars {polars_threads} / bank {bank_threads}: {secs:.2f} s, peak {gb:.2f} GB")


if __name__ == "__main__":
    main()
