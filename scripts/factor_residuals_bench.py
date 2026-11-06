"""Factor ridges whose out-of-sample residuals feed a windowed ridge with a
forward target, in one streaming query: its time a row and peak memory in
rows, in groups and in the window's length (docs/PERFORMANCE.md §47).

    uv run python scripts/factor_residuals_bench.py [OUTDIR] [--runs N] [--only AXIS]

The pipeline, per instrument (the group column ``g``): three ``ewridge``
factor ridges, each fitting its dependent ``d<f>`` on four inputs
``x<f>_0..3`` at a half-life of 2 h; each one's residual ``resid_d<f>``
(the dependent less its prediction, made before the row is learned) is a
feature of a second ``ewridge`` under ``window_size`` with
``half_life=float("inf")``, a flat window, whose target is the forward
EWMA of ``y`` less ``y`` (half-life 30 s over 150 s ahead), learned 150 s
after its row. Both stages are grouped by ``g``; one row a second a group,
the groups' rows interleaved in clock order.

The input is generated, seeded, into OUTDIR (default
``.cache/factor_residuals_bench``, which git ignores) and reused. Each case
runs in a fresh process, parquet in and ``sink_parquet`` out, best wall time
of ``--runs`` (default 3), and prints the most of its runs' peak physical
footprints (the most memory the process held at once, which leaves out the
clean pages of the mapped input file; macOS, RSS elsewhere) and peak RSS
and, for the pipeline, the size of the second stage's saved state:

- ``pipeline``: the whole query;
- ``factors``: the factor ridges alone, their residuals written;
- ``scan and sink``: the file read and written, the floor under both.

Three axes, ``--only`` picks one:

- ``rows``: one group, a 2 h window, 250,000 to 4,000,000 rows;
- ``groups``: 4,000,000 rows over 1, 100 and 1,000 groups, a 2 h window;
- ``window``: 100 groups of 10 h each (3,600,000 rows), windows of 30 min,
  2 h and 8 h, so each group's window is full: the bytes a snapshot are
  the growth of the memory and of the state over the snapshots added;
- ``cadence``: the same 2 h window with a snapshot every 10 s and every
  60 s (``snapshot_every``) rather than every row;
- ``parts``: one group of 1,000,000 rows, the pipeline beside the same with
  its second stage unwindowed (``no window``: a 2 h half-life), with ``y``
  as its target rather than the forward window (``plain target``: no
  embargo), and the factor ridges alone, so each part's share of the time
  shows.

``--released`` runs every case with the allocator told to return the pages
it frees at once (``_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0``),
so a peak is the memory that was live rather than pages kept for reuse.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

K_F, K_X = 3, 4  # factor ridges; inputs a factor ridge

#: (axis, groups, seconds a group, window), in the order they run.
RUNS = [
    ("rows", 1, 250_000, "2h"),
    ("rows", 1, 1_000_000, "2h"),
    ("rows", 1, 4_000_000, "2h"),
    ("groups", 100, 40_000, "2h"),
    ("groups", 1_000, 4_000, "2h"),
    ("window", 100, 36_000, "30m"),
    ("window", 100, 36_000, "2h"),
    ("window", 100, 36_000, "8h"),
    ("cadence", 100, 36_000, "2h/10s"),
    ("cadence", 100, 36_000, "2h/60s"),
    ("parts", 1, 1_000_000, "2h"),
]
CASES = {"rows": ("pipeline", "factors", "scan and sink"), "groups": ("pipeline", "scan and sink")}
CASES["window"] = CASES["cadence"] = ("pipeline",)
CASES["parts"] = ("pipeline", "no window", "plain target", "factors")


def make(path: Path, groups: int, seconds: int) -> None:
    """``groups`` instruments, a row a second each for ``seconds``, written
    in clock order with the groups' rows interleaved."""
    import numpy as np
    import polars as pl
    from scipy.signal import lfilter

    rng = np.random.default_rng(3)
    shape = (seconds, groups)  # row s * groups + g is group g at second s
    phi = 0.5 ** (1 / 30)
    cols: dict[str, object] = {}
    transients = []
    for f in range(K_F):
        xs = [rng.standard_normal(shape) for _ in range(K_X)]
        beta = rng.normal(0, 1, K_X)
        # A transient the factor's inputs do not see: AR(1), half-life 30 s.
        e = lfilter([1.0], [1.0, -phi], rng.normal(0, 0.3, shape), axis=0)
        transients.append(e)
        cols.update({f"x{f}_{j}": x.ravel() for j, x in enumerate(xs)})
        cols[f"d{f}"] = (sum(b * x for b, x in zip(beta, xs, strict=True)) + e).ravel()
    # The series whose forward EWMA is the target moves with the transients.
    y = np.cumsum(rng.normal(0, 0.01, shape), axis=0) + 0.5 * sum(transients)
    cols["y"] = y.ravel()
    second = np.repeat(np.arange(seconds, dtype=np.int64), groups)
    names = np.array([f"i{g:04d}" for g in range(groups)])
    pl.DataFrame(
        {
            "ts": pl.Series(second * 1_000_000_000, dtype=pl.Int64).cast(pl.Datetime("ns")),
            "g": names[np.tile(np.arange(groups), seconds)],
            **cols,
        }
    ).write_parquet(path)


def specs(window: str, case: str = "pipeline") -> tuple[list[dict[str, object]], dict[str, object]]:
    """The factor ridges and the second stage under ``window``: a length, or
    ``length/spacing`` for a snapshot every ``spacing`` of the clock. Under
    ``case`` ``"no window"`` the second stage decays at a 2 h half-life
    instead, and under ``"plain target"`` its target is ``y`` itself, with no
    forward window and no embargo."""
    import polars as pl

    import polars_online as po

    size, _, every = window.partition("/")
    cadence = {"snapshot_every": every} if every else {}

    clock = {"clock": "ts", "gap_cap": "5m", "group": "g"}
    factors = [
        po.spec.ewridge(
            f"fac{f}",
            targets=[f"d{f}"],
            features=[f"x{f}_{j}" for j in range(K_X)],
            half_life="2h",
            **clock,
        )
        for f in range(K_F)
    ]
    fwd = po.ewm_mean("y", half_life="30s", window_size="150s", reverse=True) - pl.col("y")
    target: dict[str, object] = {"targets": [fwd.alias("B")], "embargo": "150s"}
    if case == "plain target":
        target = {"targets": ["y"]}
    fit: dict[str, object] = {"window_size": size, "half_life": float("inf"), **cadence}
    if case == "no window":
        fit = {"half_life": "2h"}
    stage2 = po.spec.ewridge(
        "stage2",
        features=[f"resid_d{f}" for f in range(K_F)],
        **target,
        **fit,
        min_weight=500.0,
        **clock,
    )
    return factors, stage2


def one(src: Path, out: Path, case: str, window: str) -> None:
    """Run one case in this process and print its seconds, peak RSS and the
    second stage's saved bytes, tab-separated."""
    import polars as pl

    factors, stage2 = specs(window, case)
    lf = pl.scan_parquet(src)
    resid = [pl.col(f"fac{f}").struct.field(f"resid_d{f}") for f in range(K_F)]
    state = out / "stage2.state"
    state.unlink(missing_ok=True)
    start = time.perf_counter()
    if case in ("pipeline", "no window", "plain target"):
        plan = (
            lf.online.fit_predict(factors)
            .with_columns(resid)
            .online.fit_predict([stage2], save_state=state)
            .select("ts", "g", *(f"resid_d{f}" for f in range(K_F)), "stage2")
            .online.unnest([stage2])
        )
    elif case == "factors":
        plan = lf.online.fit_predict(factors).select("ts", "g", *resid)
    else:
        plan = lf
    plan.sink_parquet(out / "out.parquet")
    secs = time.perf_counter() - start
    saved = state.stat().st_size if state.exists() else 0
    print(f"{secs}\t{peak_footprint()}\t{peak_rss()}\t{saved}", flush=True)


def peak_rss() -> float:
    """The most resident memory the process held; NaN where `resource`, which
    is POSIX-only, is missing. It counts the pages of the parquet file Polars
    maps into memory while they are resident, which grow with the file."""
    try:
        import resource
    except ImportError:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss if sys.platform == "darwin" else rss * 1024)


def peak_footprint() -> float:
    """The most physical memory the process held at once, as Activity Monitor
    reads it (``proc_pid_rusage``'s ``lifetime_max_phys_footprint``, from
    ``struct rusage_info_v4`` in ``<sys/resource.h>``): it leaves out the clean
    pages of a mapped file, which the system can drop at will. macOS only, so
    elsewhere the peak RSS."""
    if sys.platform != "darwin":
        return peak_rss()
    import ctypes
    import os

    # The fields of `rusage_info_v4` up to the one read, all 64-bit.
    before = 28  # user_time ... logical_writes

    class Info(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("head", ctypes.c_uint64 * before)] + [
            ("lifetime_max_phys_footprint", ctypes.c_uint64),
            ("tail", ctypes.c_uint64 * 6),
        ]

    call = ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Info)]
    info = Info()
    if call(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return float(info.lifetime_max_phys_footprint)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=".cache/factor_residuals_bench")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", choices=sorted(CASES))
    ap.add_argument(
        "--released",
        action="store_true",
        help="tell the allocator to return freed pages at once, so a peak is what was live",
    )
    ap.add_argument("--one", nargs=3, metavar=("SRC", "CASE", "WINDOW"))
    args = ap.parse_args()
    out = Path(args.outdir)
    if args.one:
        src, case, window = args.one
        one(Path(src), out, case, window)
        return
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header, load

    print(header(), flush=True)
    env = dict(os.environ)
    if args.released:
        env["_RJEM_MALLOC_CONF"] = "dirty_decay_ms:0,muzzy_decay_ms:0"
        print("allocator: freed pages returned at once", flush=True)
    print(
        "axis    groups  rows        window  case           us a row  peak MB   RSS MB"
        "  stage-2 state"
    )
    for axis, groups, seconds, window in RUNS:
        if args.only and axis != args.only:
            continue
        rows = groups * seconds
        src = out / f"input_{groups}x{seconds}.parquet"
        if not src.exists():
            make(src, groups, seconds)
        for case in CASES[axis]:
            if case == "scan and sink" and window != "2h":
                continue
            best, peak, rss, saved = float("inf"), 0.0, 0.0, 0
            for _ in range(args.runs):
                line = subprocess.run(
                    [sys.executable, __file__, str(out), "--one", str(src), case, window],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                ).stdout.split()
                secs, saved = float(line[0]), int(line[3])
                peak, rss = max(peak, float(line[1])), max(rss, float(line[2]))
                best = min(best, secs)
            print(
                f"{axis:7} {groups:>6,}  {rows:>10,}  {window:>6}  {case:13} "
                f"{best / rows * 1e6:9.2f}  {peak / 1e6:7.0f}  {rss / 1e6:7.0f}  {saved:>13,}",
                flush=True,
            )
    print(f"load at the end: {load()}", flush=True)


if __name__ == "__main__":
    main()
