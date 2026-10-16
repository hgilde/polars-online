"""Training LightGBM and XGBoost from Arrow files, by every route there is:
the libraries' usual paths from Polars, Python iterators over Arrow batches,
cuts placed outside the libraries, and the Rust driver in
`scripts/gbdt_driver`, which reads the files and drives their C APIs.

docs/GBDT-ADAPTER.md (PLAN task 235) is this script's output, read. Both
libraries are in the dev group since task 242 (XGBoost 3.4.1 and LightGBM
4.7.0, the versions the document measured), whose `po-*` routes measure
`po.gbdt` with the extension built:

    uv run python scripts/gbdt_adapter.py generate .cache/gbdt/r1e7_f50 --rows 1e7 --features 50
    uv run python scripts/gbdt_adapter.py to-ipc .cache/gbdt/r1e7_f50 .cache/gbdt/r1e7_f50_ipc
    uv run python scripts/gbdt_adapter.py sweep .cache/gbdt/r1e7_f50_ipc \\
        xgb-iter rust-xgb-columnar-sample po-xgb-native --rounds 50

`run` measures one route in this process and prints one JSON line; `sweep`
runs each route it is given in a fresh interpreter, so that each peak is
that route's own. The `rust-*` routes run the driver that `GBDT_DRIVER`
names (`cargo build --release` in `scripts/gbdt_driver`) on a directory of
Arrow IPC parts, and score the model it saves. On macOS both libraries'
wheels load `@rpath/libomp.dylib`: Homebrew's `libomp` provides it, and so
does the copy scikit-learn's wheel bundles
(`DYLD_LIBRARY_PATH=.../site-packages/sklearn/.dylibs`).

The data are generated (hard rule 1): seeded parquet parts under the
directory given, gitignored under `.cache/`. Each part holds `PART_VALUES`
values: rows of `features` Float64 columns (a fifth of them with 5% nulls), one
`pl.Enum` column of 32 levels, and a regression target. `valid.parquet`
beside them, from another seed, scores every route the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

#: Values a part holds: 2.5 million rows at 50 features, 250,000 at 500.
PART_VALUES = 125_000_000
VALID_ROWS = 1_000_000
LEVELS = [f"c{i:02d}" for i in range(32)]
CAT = pl.Enum(LEVELS)
#: Rows each route samples to place its bins: LightGBM's own default,
#: `bin_construct_sample_cnt`, used for the XGBoost reference too.
SAMPLE_ROWS = 200_000
#: Values a batch holds, 102 MB of Float64: 250,000 rows of 51 columns and
#: 25,449 of 501, so that a batch is the same size whatever the width.
CHUNK_VALUES = 12_750_000
CHUNK_ROWS = 250_000  # set by `run` from CHUNK_VALUES and the data's width
#: The seed of the sample bins are placed from (`--sample-seed`): an XGBoost
#: reference's, and LightGBM's own (`data_random_seed` is one more).
SAMPLE_SEED = 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _part(rows: int, features: int, seed: int) -> pl.DataFrame:
    """One seeded block: mixed marginals (normal, lognormal, uniform, a grid
    of 0.1), 5% nulls in every fifth column, a skewed Enum, and a target that
    needs trees (a sine, a product, a threshold, a category effect)."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray | pl.Series] = {}
    for j in range(features):
        kind = j % 4
        if kind == 0:
            v = rng.standard_normal(rows)
        elif kind == 1:
            v = rng.lognormal(0.0, 1.0, rows)
        elif kind == 2:
            v = rng.uniform(-3.0, 3.0, rows)
        else:
            v = np.round(rng.standard_normal(rows), 1)
        cols[f"x{j}"] = v
    codes = np.minimum(rng.geometric(0.15, rows) - 1, len(LEVELS) - 1)
    effect = np.linspace(-1.0, 1.0, len(LEVELS))[codes]
    x = [cols[f"x{j}"] for j in range(6)]
    y = (
        np.sin(x[0])
        + 0.5 * x[2] * x[3]
        + np.where(x[4] > 0.3, x[5], -x[5])
        + np.log1p(x[1])
        + effect
    ) + 0.5 * rng.standard_normal(rows)
    df = pl.DataFrame(cols)
    nulls = [f"x{j}" for j in range(0, features, 5)]
    masks = {c: pl.Series(rng.random(rows) < 0.05) for c in nulls}
    df = df.with_columns(pl.when(masks[c]).then(None).otherwise(pl.col(c)).alias(c) for c in nulls)
    cats = pl.Series("cat", np.array(LEVELS)[codes]).cast(CAT)
    return df.with_columns(cats, pl.Series("y", y))


def _write_part(args: tuple[str, int, int, int]) -> str:
    path, rows, features, seed = args
    _part(rows, features, seed).write_parquet(path)
    return path


def generate(out: Path, rows: int, features: int, seed: int, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    part_rows = PART_VALUES // features
    for i, start in enumerate(range(0, rows, part_rows)):
        n = min(part_rows, rows - start)
        jobs.append((str(out / f"part-{i:04d}.parquet"), n, features, seed * 1_000_003 + i))
    valid_rows = min(VALID_ROWS, max(rows // 10, 1), part_rows)
    jobs.append((str(out / "valid.parquet"), valid_rows, features, seed * 1_000_003 + 999_983))
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for path in pool.map(_write_part, jobs):
            print(path, flush=True)
    size = sum(p.stat().st_size for p in out.glob("*.parquet"))
    print(
        json.dumps(
            {
                "generated": str(out),
                "rows": rows,
                "features": features,
                "bytes": size,
                "seconds": round(time.perf_counter() - t, 1),
            }
        )
    )


def _parts(data: Path) -> tuple[list[str], bool]:
    """The parts of a data directory: Arrow IPC files if it holds any, else
    parquet."""
    ipc = sorted(str(p) for p in data.glob("part-*.arrow"))
    return (ipc, True) if ipc else (sorted(str(p) for p in data.glob("part-*.parquet")), False)


def features_of(data: Path) -> list[str]:
    parts, ipc = _parts(data)
    scan = pl.scan_ipc if ipc else pl.scan_parquet
    names = scan(parts[0]).collect_schema().names()
    return [n for n in names if n.startswith("x")] + ["cat"]


def source(data: Path) -> pl.LazyFrame:
    """The training stream: every part, as one query."""
    parts, ipc = _parts(data)
    return pl.scan_ipc(parts) if ipc else pl.scan_parquet(parts)


def to_ipc(data: Path, out: Path) -> None:
    """The same rows as uncompressed Arrow IPC files, one a parquet part, in
    record batches of `CHUNK_VALUES` values: what a reader of Arrow files
    reads with no decoding. `valid.parquet` is copied beside them."""
    out.mkdir(parents=True, exist_ok=True)
    rows = max(1, CHUNK_VALUES // (len(features_of(data)) + 1))
    t = time.perf_counter()
    for part in sorted(data.glob("part-*.parquet")):
        pl.read_parquet(part).write_ipc(
            out / (part.stem + ".arrow"), compression="uncompressed", record_batch_size=rows
        )
    shutil.copy(data / "valid.parquet", out / "valid.parquet")
    size = sum(p.stat().st_size for p in out.glob("part-*.arrow"))
    took = round(time.perf_counter() - t, 1)
    print(json.dumps({"ipc": str(out), "bytes": size, "seconds": took}))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


_RUSAGE_V4 = [
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
    "cpu_time_qos_default",
    "cpu_time_qos_maintenance",
    "cpu_time_qos_background",
    "cpu_time_qos_utility",
    "cpu_time_qos_legacy",
    "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive",
    "billed_system_time",
    "serviced_system_time",
    "logical_writes",
    "lifetime_max_phys_footprint",
    "instructions",
    "cycles",
    "billed_energy",
    "serviced_energy",
    "interval_max_phys_footprint",
    "runnable_time",
]


def _footprint():
    """A reader of this process's physical footprint and its lifetime peak,
    on macOS (`proc_pid_rusage`, `struct rusage_info_v4` in
    `<sys/resource.h>`); None elsewhere. The footprint is what Activity
    Monitor calls Memory: it leaves out clean file-backed pages, so a
    memory-mapped spill, which RSS counts while it is resident and the
    system can drop at will, is not charged to the route."""
    if sys.platform != "darwin":
        return None
    import ctypes

    class Info(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RUSAGE_V4]

    call = ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Info)]
    pid = os.getpid()

    def read() -> tuple[int, int]:
        info = Info()
        if call(pid, 4, ctypes.byref(info)) != 0:
            raise OSError("proc_pid_rusage failed")
        return info.phys_footprint, info.lifetime_max_phys_footprint

    return read


class Meter:
    """Wall time and peak memory per phase. A thread samples every 20 ms:
    the physical footprint on macOS, and RSS everywhere. The process's own
    lifetime peaks bound the whole run."""

    def __init__(self) -> None:
        import psutil

        self._proc = psutil.Process()
        self._fp = _footprint()
        self.phases: dict[str, dict[str, float]] = {}
        self._phase = "start"
        self._t = time.perf_counter()
        self._peak = self._now()
        self._stop = False
        self._base = self._peak
        threading.Thread(target=self._sample, daemon=True).start()

    def _now(self) -> tuple[int, int]:
        rss = self._proc.memory_info().rss
        return (self._fp()[0] if self._fp else rss), rss

    def _sample(self) -> None:
        while not self._stop:
            fp, rss = self._now()
            self._peak = (max(self._peak[0], fp), max(self._peak[1], rss))
            time.sleep(0.02)

    def phase(self, name: str) -> None:
        now = time.perf_counter()
        fp, rss = self._now()
        rec = self.phases.setdefault(self._phase, {"s": 0.0, "peak_gb": 0.0, "rss_gb": 0.0})
        rec["s"] += now - self._t
        rec["peak_gb"] = max(rec["peak_gb"], max(self._peak[0], fp) / 1e9)
        rec["rss_gb"] = max(rec["rss_gb"], max(self._peak[1], rss) / 1e9)
        self._phase, self._t, self._peak = name, now, (fp, rss)

    def done(self) -> dict:
        self.phase("end")
        self._stop = True
        self.phases.pop("start", None)
        self.phases.pop("end", None)
        for rec in self.phases.values():
            for k in rec:
                rec[k] = round(rec[k], 2)
        # macOS reports ru_maxrss in bytes, Linux in KiB.
        scale = 1 if sys.platform == "darwin" else 1024
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        return {
            "phases": self.phases,
            "peak_gb": round((self._fp()[1] if self._fp else rss) / 1e9, 2),
            "peak_rss_gb": round(rss / 1e9, 2),
            "base_gb": round(self._base[0] / 1e9, 2),
        }


def _load() -> list[float]:
    return [round(v, 1) for v in os.getloadavg()]


class Counter:
    """Time spent producing batches (the source: Polars reading and
    decoding) and handing them over (the library's own conversion), and the
    passes a library makes over the source."""

    def __init__(self) -> None:
        self.source_s = 0.0
        self.hand_s = 0.0
        self.batches = 0
        self.passes = 0
        self.replays = 0
        self.replay_s = 0.0
        self.t0 = time.perf_counter()
        self.pass_starts: list[float] = []


def sample_rows(lf: pl.LazyFrame, n: int, seed: int = 0, slices: int = 40) -> pl.DataFrame:
    """`n` rows in `slices` contiguous runs at random offsets: a parquet
    scan reads only the row groups a slice touches."""
    total = lf.select(pl.len()).collect().item()
    run = max(n // slices, 1)
    rng = np.random.default_rng(seed)
    offsets = np.sort(rng.choice(max(total - run, 1), size=slices, replace=False))
    return pl.concat([lf.slice(int(o), run).collect() for o in offsets])


class Batches:
    """`collect_batches` over a query, restarted on demand: what a library's
    re-iterable input needs when the source is a query and not a list."""

    def __init__(self, lf: pl.LazyFrame, cols: list[str], counter: Counter) -> None:
        self.lf, self.c = lf.select(cols), counter
        self._it = None

    def restart(self) -> None:
        self._it = None

    def next(self) -> pl.DataFrame | None:
        t = time.perf_counter()
        if self._it is None:
            self._it = iter(self.lf.collect_batches(chunk_size=CHUNK_ROWS))
            self.c.passes += 1
            self.c.pass_starts.append(round(t - self.c.t0, 2))
        b = next(self._it, None)
        self.c.source_s += time.perf_counter() - t
        if b is not None:
            self.c.batches += 1
        return b


class PaBatches:
    """The same parts read by `pyarrow.dataset`, each batch a pyarrow
    `Table` handed to XGBoost as it is (option B). XGBoost takes a string
    dictionary only, so a `string_view` one (what Polars writes to Arrow IPC)
    is cast, which touches the 32 labels and not the codes."""

    def __init__(self, data: Path, cols: list[str], counter: Counter) -> None:
        import pyarrow.dataset as ds

        parts, ipc = _parts(data)
        self.ds, self.cols, self.c = (
            ds.dataset(parts, format="ipc" if ipc else "parquet"),
            cols,
            counter,
        )
        self._it = None

    def restart(self) -> None:
        self._it = None

    def next(self):
        import pyarrow as pa

        t = time.perf_counter()
        if self._it is None:
            self._it = iter(self.ds.to_batches(columns=self.cols, batch_size=CHUNK_ROWS))
            self.c.passes += 1
            self.c.pass_starts.append(round(t - self.c.t0, 2))
        rb = next(self._it, None)
        self.c.source_s += time.perf_counter() - t
        if rb is None:
            return None
        self.c.batches += 1
        table = pa.Table.from_batches([rb])
        i = table.schema.get_field_index("cat")
        want = pa.dictionary(table.schema.field(i).type.index_type, pa.string())
        if table.schema.field(i).type != want:
            table = table.set_column(i, "cat", table.column(i).cast(want))
        return table


class Replay:
    """The adapter's one read of the source, replayed for each pass a library
    asks for: every batch held in memory, or spilled to an uncompressed Arrow
    IPC file and read back, a batch at a time, on each replay (polars 2.0's
    `read_ipc` copies the file into memory, so a spill held whole would hold
    the data whole). `take(start, batch)` sees each batch as it passes, so a
    sample is drawn in the same read."""

    def __init__(
        self,
        lf: pl.LazyFrame,
        cols: list[str],
        counter: Counter,
        spill: Path | None,
        f32: bool,
        take=None,
    ) -> None:
        self.c, self.chunks = counter, []
        q = lf.select(cols)
        if f32:
            q = q.with_columns(pl.col(pl.Float64).exclude("y").cast(pl.Float32))
        t, start = time.perf_counter(), 0
        for i, b in enumerate(q.collect_batches(chunk_size=CHUNK_ROWS)):
            if take is not None:
                take(start, b)
            start += b.height
            if spill is not None:
                path = spill / f"chunk-{i:06d}.arrow"
                b.write_ipc(path, compression="uncompressed")
                self.chunks.append(path)
            else:
                self.chunks.append(b)
        self.rows = start
        self.c.source_s += time.perf_counter() - t
        self.c.passes += 1
        self._i = 0

    def restart(self) -> None:
        self._i = 0

    def next(self) -> pl.DataFrame | None:
        if self._i == 0:
            self.c.replays += 1
            self.c.pass_starts.append(round(time.perf_counter() - self.c.t0, 2))
        if self._i == len(self.chunks):
            return None
        b = self.chunks[self._i]
        self._i += 1
        if isinstance(b, Path):
            t = time.perf_counter()
            b = pl.read_ipc(b)
            self.c.replay_s += time.perf_counter() - t
        return b


class BernoulliTake:
    """A uniform sample drawn as the batches pass: each row kept with
    probability `n / total`, `total` read from the query (for a parquet
    scan, from the files' metadata)."""

    def __init__(self, lf: pl.LazyFrame, n: int, seed: int) -> None:
        total = lf.select(pl.len()).collect().item()
        self.p, self.rng, self.parts = min(n / total, 1.0), np.random.default_rng(seed), []

    def __call__(self, start: int, b: pl.DataFrame) -> None:
        self.parts.append(b.filter(pl.Series(self.rng.random(b.height) < self.p)))

    def frame(self) -> pl.DataFrame:
        return pl.concat(self.parts)


class IndexTake:
    """The rows at sorted global indices, and the whole label column, as the
    batches pass."""

    def __init__(self, idx: np.ndarray, feats: list[str]) -> None:
        self.idx, self.feats, self.rows, self.ys = idx, feats, [], []
        self.start = 0

    def __call__(self, start: int, b: pl.DataFrame) -> None:
        lo, hi = np.searchsorted(self.idx, [start, start + b.height])
        if hi > lo:
            self.rows.append(_lgb_rows(b[self.idx[lo:hi] - start], self.feats))
        self.ys.append(b["y"].to_numpy())

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.rows).astype(np.float64), np.concatenate(self.ys)


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


def _xgb_params(threads: int) -> dict:
    return {
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "max_bin": 256,
        "nthread": threads,
        "objective": "reg:squarederror",
        "seed": 0,
    }


def _xgb_iter(feats: list[str], batches, counter: Counter, numpy: bool = False):
    """A `DataIter` over batches. A Polars batch reaches XGBoost through
    pyarrow without a copy; `numpy` hands a NaN-filled copy instead, which
    is what a caller without pyarrow is left with."""
    import xgboost as xgb

    types = ["q"] * (len(feats) - 1) + ["c"]

    class It(xgb.DataIter):
        def next(self, input_data) -> bool:
            b = batches.next()
            if b is None:
                return False
            t = time.perf_counter()
            if numpy:
                x = b.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64))
                input_data(
                    data=x.to_numpy(),
                    label=b["y"].to_numpy(),
                    feature_names=feats,
                    feature_types=types,
                )
            else:
                input_data(data=b.select(feats), label=b["y"])
            counter.hand_s += time.perf_counter() - t
            return True

        def reset(self) -> None:
            batches.restart()

    return It


def _xgb_iter_pa(feats: list[str], batches, counter: Counter):
    """A `DataIter` handing XGBoost each pyarrow `Table` as it is."""
    import xgboost as xgb

    class It(xgb.DataIter):
        def next(self, input_data) -> bool:
            b = batches.next()
            if b is None:
                return False
            t = time.perf_counter()
            input_data(data=b.select(feats), label=b.column("y").to_numpy())
            counter.hand_s += time.perf_counter() - t
            return True

        def reset(self) -> None:
            batches.restart()

    return It


def _xgb_score(bst, valid: pl.DataFrame, feats: list[str]) -> float:
    import xgboost as xgb

    d = xgb.DMatrix(valid.select(feats), enable_categorical=True)
    pred = bst.predict(d)
    return float(np.sqrt(np.mean((pred - valid["y"].to_numpy()) ** 2)))


def _xgb_trees(bst) -> str:
    return hashlib.sha256("".join(bst.get_dump(dump_format="json")).encode()).hexdigest()[:12]


def _xgb_cuts(d) -> str:
    ptr, vals = d.get_quantile_cut()
    return hashlib.sha256(np.asarray(ptr).tobytes() + np.asarray(vals).tobytes()).hexdigest()[:12]


def route_xgb(
    name: str, data: Path, m: Meter, rounds: int, threads: int, spill: Path | None, f32: bool
) -> dict:
    import xgboost as xgb

    feats = features_of(data)
    lf = source(data)
    c = Counter()
    out: dict = {}
    kw = {"max_bin": 256, "nthread": threads, "enable_categorical": True}
    if name == "xgb-numpy":
        m.phase("collect")
        df = lf.collect()
        m.phase("convert")
        types = ["q"] * (len(feats) - 1) + ["c"]
        X = df.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64)).to_numpy()
        y = df["y"].to_numpy()
        del df
        m.phase("bin")
        d = xgb.QuantileDMatrix(X, label=y, feature_names=feats, feature_types=types, **kw)
        del X
    elif name == "xgb-pandas":
        m.phase("collect")
        df = lf.collect()
        m.phase("convert")
        pdf = df.to_pandas()
        del df
        m.phase("bin")
        d = xgb.QuantileDMatrix(pdf[feats], label=pdf["y"], **kw)
        del pdf
    elif name == "xgb-polars":
        m.phase("collect")
        df = lf.collect()
        m.phase("bin")
        d = xgb.QuantileDMatrix(df.select(feats), label=df["y"], **kw)
        del df
    elif name == "xgb-iter-pa":
        m.phase("bin")
        batches = PaBatches(data, feats + ["y"], c)
        It = _xgb_iter_pa(feats, batches, c)
        d = xgb.QuantileDMatrix(It(), **kw)
    elif name in ("xgb-iter", "xgb-iter-numpy", "xgb-iter-ref", "xgb-extmem"):
        ref = None
        if name == "xgb-iter-ref":
            m.phase("sample")
            s = sample_rows(lf, SAMPLE_ROWS, SAMPLE_SEED)
            ref = xgb.QuantileDMatrix(s.select(feats), label=s["y"], **kw)
        m.phase("bin")
        batches = Batches(lf, feats + ["y"], c)
        It = _xgb_iter(feats, batches, c, numpy=name == "xgb-iter-numpy")
        if name == "xgb-extmem":
            cache = Path(tempfile.mkdtemp(dir=spill))
            d = xgb.ExtMemQuantileDMatrix(It(cache_prefix=str(cache / "cache")), ref=ref, **kw)
        else:
            d = xgb.QuantileDMatrix(It(), ref=ref, **kw)
    elif name.startswith("xgb-replay"):
        # The source read once; XGBoost's passes replay what that read held.
        m.phase("read")
        sp = Path(tempfile.mkdtemp(dir=spill)) if "-spill" in name else None
        take = BernoulliTake(lf, SAMPLE_ROWS, SAMPLE_SEED) if name.endswith("-ref") else None
        rep = Replay(lf, feats + ["y"], c, sp, f32, take)
        ref = None
        if take is not None:
            m.phase("sample")
            s = take.frame()
            ref = xgb.QuantileDMatrix(s.select(feats), label=s["y"], **kw)
        m.phase("bin")
        It = _xgb_iter(feats, rep, c)
        d = xgb.QuantileDMatrix(It(), ref=ref, **kw)
        del rep
    else:
        raise SystemExit(f"unknown route {name}")
    out["cuts"] = _xgb_cuts(d) if name != "xgb-extmem" else None
    m.phase("boost")
    bst = xgb.train(_xgb_params(threads), d, num_boost_round=rounds)
    m.phase("score")
    out["valid_rmse"] = round(_xgb_score(bst, pl.read_parquet(data / "valid.parquet"), feats), 6)
    out["trees"] = _xgb_trees(bst)
    out.update(_counted(c))
    return out


def _counted(c: Counter) -> dict:
    return {
        "source_s": round(c.source_s, 2),
        "handover_s": round(c.hand_s, 2),
        "source_passes": c.passes,
        "replays": c.replays,
        "replay_s": round(c.replay_s, 2),
        "batches": c.batches,
        "pass_starts_s": c.pass_starts,
    }


# --------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------


def _lgb_params(threads: int) -> dict:
    return {
        "objective": "regression",
        "num_leaves": 63,
        "learning_rate": 0.1,
        "max_bin": 255,
        "num_threads": threads,
        "seed": 0,
        # Explicit, because the Python package drops `seed` when it samples a
        # Sequence (`Dataset.get_params` keeps only dataset parameters) while
        # the C++ derives the sample's seed from it: with `seed` alone a
        # Sequence and a numpy array sample different rows (4.7.0).
        "data_random_seed": 1 + SAMPLE_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }


def _lgb_frame(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """LightGBM reads integer category codes: a dictionary column (a polars
    Enum) ends the process from inside its Arrow reader (4.7.0)."""
    return df.select(pl.col(feats[:-1]), pl.col("cat").to_physical())


def _lgb_score(bst, valid: pl.DataFrame, feats: list[str]) -> float:
    X = valid.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64)).to_numpy()
    pred = bst.predict(X)
    return float(np.sqrt(np.mean((pred - valid["y"].to_numpy()) ** 2)))


def _lgb_trees(bst) -> str:
    trees = json.dumps(bst.dump_model()["tree_info"], sort_keys=True)
    return hashlib.sha256(trees.encode()).hexdigest()[:12]


def _lgb_rows(b: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """A batch as LightGBM's row pushes take it: row-major, codes as floats."""
    return _lgb_frame(b, feats).to_numpy(order="c")


def _lgb_sample_indices(params: dict, total: int) -> np.ndarray:
    """The rows LightGBM samples to place its bins, from LightGBM itself
    (`LGBM_SampleIndices`, seeded by `data_random_seed`): an adapter that
    hands it the same rows gets the same bins as `Dataset(numpy)`."""
    import lightgbm as lgb

    return lgb.Dataset(np.empty((1, 1)), params=params)._create_sample_indices(total_nrow=total)


def _lgb_sequence(batches, feats: list[str], total: int, f32: bool, sample: np.ndarray | None):
    """A `lightgbm.Sequence` read in order. LightGBM asks a Sequence for
    single rows only to sample it, in ascending order, and `sample` holds
    those rows already; then it pushes `seq[start:end]` in turn. With a
    `reference` Dataset it asks for no single rows at all."""
    import lightgbm as lgb

    class Seq(lgb.Sequence):
        batch_size = CHUNK_ROWS

        def __init__(self) -> None:
            self._buf: list[np.ndarray] = []
            self._have = 0
            self._next = 0
            self._k = 0

        def __len__(self) -> int:
            return total

        def __getitem__(self, idx):
            if isinstance(idx, (int, np.integer)) and sample is not None:
                row = sample[self._k]
                self._k += 1
                return row
            if not isinstance(idx, slice) or idx.start != self._next:
                raise IndexError(f"read in order only: {idx}")
            want = idx.stop - idx.start
            while self._have < want:
                b = batches.next()
                if b is None:
                    break
                t = time.perf_counter()
                x = _lgb_rows(b, feats).astype(np.float32 if f32 else np.float64, copy=False)
                batches.c.hand_s += time.perf_counter() - t
                self._buf.append(x)
                self._have += x.shape[0]
            x = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
            out, rest = x[:want], x[want:]
            self._buf = [rest] if rest.shape[0] else []
            self._have = rest.shape[0]
            self._next = idx.stop
            return np.ascontiguousarray(out)

    return Seq()


def route_lgb(
    name: str, data: Path, m: Meter, rounds: int, threads: int, spill: Path | None, f32: bool
) -> dict:
    import lightgbm as lgb

    feats = features_of(data)
    lf = source(data)
    c = Counter()
    params = _lgb_params(threads)
    cat = [len(feats) - 1]
    if name == "lgb-numpy":
        m.phase("collect")
        df = lf.collect()
        m.phase("convert")
        X = df.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64)).to_numpy()
        y = df["y"].to_numpy()
        del df
        m.phase("bin")
        d = lgb.Dataset(X, label=y, params=params, categorical_feature=cat).construct()
        del X
    elif name == "lgb-pandas":
        m.phase("collect")
        df = lf.collect()
        m.phase("convert")
        pdf = df.to_pandas()
        del df
        m.phase("bin")
        d = lgb.Dataset(
            pdf[feats], label=pdf["y"], params=params, categorical_feature=cat
        ).construct()
        del pdf
    elif name == "lgb-polars":
        m.phase("collect")
        df = lf.collect()
        m.phase("bin")
        frame = _lgb_frame(df, feats)
        d = lgb.Dataset(frame, label=df["y"], params=params, categorical_feature=cat).construct()
        del df, frame
    elif name == "lgb-stream-ref":
        # The fewest reads: the label column alone, a sample from 40 slices,
        # one pass of pushes. The sample is not LightGBM's, so neither are
        # the bins.
        m.phase("label")
        y = lf.select("y").collect()["y"].to_numpy()
        m.phase("sample")
        s = sample_rows(lf, SAMPLE_ROWS, SAMPLE_SEED)
        ref = lgb.Dataset(
            _lgb_frame(s, feats), label=s["y"], params=params, categorical_feature=cat
        ).construct()
        m.phase("bin")
        batches = Batches(lf, feats + ["y"], c)
        seq = _lgb_sequence(batches, feats, len(y), f32, None)
        d = lgb.Dataset(seq, label=y, reference=ref, params=params, categorical_feature=cat)
        d.construct()
    elif name in ("lgb-stream", "lgb-replay", "lgb-replay-spill"):
        # LightGBM's own sample rows, so its own bins: drawn in a first pass
        # over the source (stream), or as the one read passes (replay). Both
        # need the row count first, which a parquet scan reads from metadata.
        total = lf.select(pl.len()).collect().item()
        take = IndexTake(_lgb_sample_indices(params, total), feats)
        if name == "lgb-stream":
            m.phase("sample")
            batches = Batches(lf, feats + ["y"], c)
            while (b := batches.next()) is not None:
                take(take.start, b)
                take.start += b.height
            batches.restart()
        else:
            m.phase("read")
            sp = Path(tempfile.mkdtemp(dir=spill)) if name == "lgb-replay-spill" else None
            batches = Replay(lf, feats + ["y"], c, sp, f32, take)
        rows, y = take.result()
        m.phase("bin")
        seq = _lgb_sequence(batches, feats, total, f32, rows)
        d = lgb.Dataset(seq, label=y, params=params, categorical_feature=cat)
        d.construct()
        del batches
    else:
        raise SystemExit(f"unknown route {name}")
    m.phase("boost")
    bst = lgb.train(params, d, num_boost_round=rounds)
    m.phase("score")
    out = {"valid_rmse": round(_lgb_score(bst, pl.read_parquet(data / "valid.parquet"), feats), 6)}
    out["trees"] = _lgb_trees(bst)
    out.update(_counted(c))
    return out


# --------------------------------------------------------------------------
# Rust: options C and D
# --------------------------------------------------------------------------

#: The Rust routes: the library, how the batches are fed, and where the cuts
#: come from (none: the library's own; `cuts`: the t-digest at compression
#: 1,000; `cuts200`: at the library's own 200; `sample`: 200,000 rows).
RUST_ROUTES = {
    "rust-xgb-dense": ("xgb", "dense", None),
    "rust-xgb-columnar": ("xgb", "columnar", None),
    "rust-xgb-dense-cuts": ("xgb", "dense", ("digest", 1000)),
    "rust-xgb-dense-cuts200": ("xgb", "dense", ("digest", 200)),
    "rust-xgb-dense-sample": ("xgb", "dense", ("sample", 0)),
    "rust-xgb-columnar-cuts": ("xgb", "columnar", ("digest", 1000)),
    "rust-xgb-columnar-sample": ("xgb", "columnar", ("sample", 0)),
    "rust-xgb-codes": ("xgb", "codes", ("digest", 1000)),
    "rust-lgb-f64": ("lgb", "f64", None),
    "rust-lgb-f32": ("lgb", "f32", None),
    "rust-lgb-forced": ("lgb", "f64", ("digest", 1000)),
    "rust-lgb-codes": ("lgb", "codes", ("digest", 1000)),
}


def _driver(*args: str) -> dict:
    """Run `scripts/gbdt_driver` (built with `cargo build --release`), named
    by `GBDT_DRIVER`, and read its one JSON line."""
    exe = os.environ.get("GBDT_DRIVER")
    if not exe:
        raise SystemExit("set GBDT_DRIVER to the built scripts/gbdt_driver binary")
    res = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8")
    if res.returncode:
        raise SystemExit(f"driver failed: {res.stderr[-2000:]}")
    return json.loads([ln for ln in res.stdout.splitlines() if ln.startswith("{")][-1])


def _quantize(valid: pl.DataFrame, feats: list[str], cuts: list[np.ndarray], miss: float):
    """The validation rows as the bins a model fed codes was trained on."""
    cols = []
    for j, name in enumerate(feats[:-1]):
        x = valid[name].to_numpy()
        c = np.searchsorted(cuts[j], x, side="right").astype(np.float64)
        c[np.isnan(x)] = miss
        cols.append(c)
    cols.append(valid["cat"].to_physical().cast(pl.Float64).to_numpy())
    return np.column_stack(cols)


def route_rust(
    name: str, data: Path, m: Meter, rounds: int, threads: int, spill: Path | None, f32: bool
) -> dict:
    """Options C and D: the Rust driver reads the Arrow IPC files, computes
    the cuts when the route asks for them, and drives the library's C API
    from the shared library its wheel ships; this process only scores the
    model it saves."""
    lib, feed, sketch = RUST_ROUTES[name]
    feats = features_of(data)
    rows = source(data).select(pl.len()).collect().item()
    tmp = Path(tempfile.mkdtemp(dir=spill))
    common = ["--threads", str(threads), "--rounds", str(rounds)]
    phases: dict = {}
    peak = 0.0
    out: dict = {}
    cuts_file = tmp / "cuts.txt"
    if sketch is not None:
        how, comp = sketch
        r = _driver(
            "sketch", str(data), "--out", str(cuts_file), "--sketch", how,
            "--compression", str(comp or 1000), "--rows", str(rows), "--threads", str(threads),
        )  # fmt: skip
        phases["sketch"] = {"s": round(sum(v["s"] for v in r["phases"].values()), 2),
                            "peak_gb": r["peak_gb"]}  # fmt: skip
        peak = max(peak, r["peak_gb"])
    model = tmp / ("model.json" if lib == "xgb" else "model.txt")
    if lib == "xgb":
        import xgboost as xgb

        args = ["xgb", str(data), "--lib", xgb.build_info()["libxgboost"], "--feed", feed]
    else:
        import lightgbm as lgb

        args = ["lgb", str(data), "--lib", lgb.basic._LIB._name, "--feed", feed]
        args += ["--rows", str(rows)]
        if name == "rust-lgb-forced":
            args += ["--forced"]
    if sketch is not None:
        args += ["--cuts", str(cuts_file)]
    r = _driver(*args, "--model", str(model), *common)
    for k, v in r["phases"].items():
        phases[k] = v
    peak = max(peak, r["peak_gb"])
    for k in ("read_s", "convert_s", "push_s", "source_passes", "batches", "pass_ends_s"):
        if k in r:
            out[k] = r[k]
    m.phase("score")
    valid = pl.read_parquet(data / "valid.parquet")
    codes = feed == "codes"
    cuts = (
        [
            np.array(ln.split(), dtype=float)
            for ln in cuts_file.read_text(encoding="utf-8").splitlines()
        ]
        if codes
        else []
    )
    if lib == "xgb":
        import xgboost as xgb

        bst = xgb.Booster(model_file=str(model))
        types = ["q"] * (len(feats) - 1) + ["c"]
        if codes:
            X, miss = _quantize(valid, feats, cuts, 255.0), 255.0
        else:
            X = valid.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64))
            X, miss = X.to_numpy(), np.nan
        d = xgb.DMatrix(X, missing=miss, feature_names=feats, feature_types=types,
                        enable_categorical=True)  # fmt: skip
        pred = bst.predict(d)
        out["trees"] = _xgb_trees(bst)
    else:
        import lightgbm as lgb

        bst = lgb.Booster(model_file=str(model))
        if codes:
            X = _quantize(valid, feats, cuts, np.nan)
        else:
            X = valid.select(pl.col(feats[:-1]), pl.col("cat").to_physical().cast(pl.Float64))
            X = X.to_numpy()
        pred = bst.predict(X)
        out["trees"] = _lgb_trees(bst)
    out["valid_rmse"] = round(float(np.sqrt(np.mean((pred - valid["y"].to_numpy()) ** 2))), 6)
    out["rust_phases"] = phases
    out["rust_peak_gb"] = peak
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# --------------------------------------------------------------------------
# po.gbdt, the built component (PLAN task 242)
# --------------------------------------------------------------------------

#: The routes through `po.gbdt`: the library, the engine, and its options.
#: A route is handed the part files (which the native engine reads itself
#: when they are Arrow IPC), or with `-lazy` the parts as one `LazyFrame`,
#: read through `collect_batches`.
PO_ROUTES = {
    "po-xgb-native": ("xgb", "native", {}),
    "po-xgb-native-lazy": ("xgb", "native", {}),
    "po-xgb-python": ("xgb", "python", {}),
    "po-xgb-native-sketch": ("xgb", "native", {"sample_rows": None}),
    "po-lgb-native": ("lgb", "native", {"in_memory": False}),
    "po-lgb-native-lazy": ("lgb", "native", {"in_memory": False}),
    "po-lgb-native-memory": ("lgb", "native", {"in_memory": True}),
    "po-lgb-python": ("lgb", "python", {"in_memory": False}),
}


def route_po(
    name: str, data: Path, m: Meter, rounds: int, threads: int, spill: Path | None, f32: bool
) -> dict:
    """The package's own builder over the parts as one query, with the
    research's training parameters, so its trees compare by hash."""
    import polars_online as po

    lib, engine, opts = PO_ROUTES[name]
    feats = features_of(data)
    lf = source(data) if name.endswith("-lazy") else _parts(data)[0]
    # The library imported before the clock starts, as the other routes do.
    import lightgbm as lgb
    import xgboost as xgb

    m.phase("bin")
    if lib == "xgb":
        d = po.gbdt.xgboost_matrix(
            lf, label="y", features=feats, threads=threads, engine=engine, **opts
        )
        out: dict = {"cuts": _xgb_cuts(d), "native": dict(po.gbdt._LAST_STATS)}
        m.phase("boost")
        bst = xgb.train(_xgb_params(threads), d, num_boost_round=rounds)
        m.phase("score")
        out["valid_rmse"] = round(
            _xgb_score(bst, pl.read_parquet(data / "valid.parquet"), feats), 6
        )
        out["trees"] = _xgb_trees(bst)
        return out
    params = _lgb_params(threads)
    dataset = {k: params[k] for k in ("data_random_seed", "verbose")}
    d = po.gbdt.lightgbm_dataset(
        lf, label="y", features=feats, params=dataset, threads=threads, engine=engine, **opts
    )
    native = dict(po.gbdt._LAST_STATS)
    m.phase("boost")
    bst = lgb.train(params, d, num_boost_round=rounds)
    m.phase("score")
    out = {"valid_rmse": round(_lgb_score(bst, pl.read_parquet(data / "valid.parquet"), feats), 6)}
    out["trees"] = _lgb_trees(bst)
    out["native"] = native
    return out


ROUTES = [
    "source-pass",
    "xgb-numpy",
    "xgb-pandas",
    "xgb-polars",
    "xgb-iter",
    "xgb-iter-numpy",
    "xgb-iter-pa",
    "xgb-iter-ref",
    "xgb-extmem",
    "xgb-replay",
    "xgb-replay-spill",
    "xgb-replay-ref",
    "xgb-replay-spill-ref",
    "lgb-numpy",
    "lgb-pandas",
    "lgb-polars",
    "lgb-stream",
    "lgb-stream-ref",
    "lgb-replay",
    "lgb-replay-spill",
    *RUST_ROUTES,
    *PO_ROUTES,
]


def route_source(
    name: str, data: Path, m: Meter, rounds: int, threads: int, spill: Path | None, f32: bool
) -> dict:
    """One pass of the query with nothing downstream: what each pass a
    library makes over the source costs on its own."""
    c = Counter()
    m.phase("read")
    b = Batches(source(data), features_of(data) + ["y"], c)
    rows = 0
    while (x := b.next()) is not None:
        rows += x.height
    return {"rows": rows, **_counted(c)}


def header() -> str:
    """What the numbers ran on: the libraries, the machine and its load."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    libs = []
    for n in ("xgboost", "lightgbm", "polars", "numpy", "pyarrow"):
        try:
            libs.append(f"{n} {version(n)}")
        except PackageNotFoundError:
            libs.append(f"{n} absent")
    return (
        f"{'; '.join(libs)}; Python {platform.python_version()}; {platform.system()} "
        f"{platform.machine()}; {os.cpu_count()} cpus; load {_load()}"
    )


def run(data: Path, name: str, rounds: int, threads: int, spill: Path | None, f32: bool) -> dict:
    global CHUNK_ROWS
    CHUNK_ROWS = max(1, CHUNK_VALUES // (len(features_of(data)) + 1))
    load0 = _load()
    m = Meter()
    t = time.perf_counter()
    kinds = {
        "xgb": route_xgb,
        "lgb": route_lgb,
        "source": route_source,
        "rust": route_rust,
        "po": route_po,
    }
    fn = kinds[name.split("-")[0]]
    out = fn(name, data, m, rounds, threads, spill, f32)
    total = time.perf_counter() - t
    rec = {"route": name, "data": data.name, "rounds": rounds, "threads": threads, "f32": f32}
    rec["sample_seed"] = SAMPLE_SEED
    rec["chunk_rows"] = CHUNK_ROWS
    rec.update(out)
    rec.update(m.done())
    rec["total_s"] = round(total, 2)
    rec["load"] = [load0, _load()]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the seeded parquet parts")
    g.add_argument("out", type=Path)
    g.add_argument("--rows", type=float, required=True)
    g.add_argument("--features", type=int, required=True)
    g.add_argument("--seed", type=int, default=235)
    g.add_argument("--workers", type=int, default=4)
    i = sub.add_parser("to-ipc", help="the same parts as uncompressed Arrow IPC files")
    i.add_argument("data", type=Path)
    i.add_argument("out", type=Path)
    for cmd in ("run", "sweep"):
        r = sub.add_parser(cmd, help="measure here" if cmd == "run" else "a process per route")
        r.add_argument("data", type=Path)
        r.add_argument("routes", nargs="+", choices=ROUTES)
        r.add_argument("--rounds", type=int, default=50)
        r.add_argument("--threads", type=int, default=8)
        r.add_argument("--spill", type=Path, default=None, help="for external memory and spills")
        r.add_argument("--f32", action="store_true", help="the adapter casts features to Float32")
        r.add_argument("--sample-seed", type=int, default=0, help="for an XGBoost reference")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.out, int(a.rows), a.features, a.seed, a.workers)
        return
    if a.cmd == "to-ipc":
        to_ipc(a.data, a.out)
        return
    if a.spill is not None:
        a.spill.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        global SAMPLE_SEED
        SAMPLE_SEED = a.sample_seed
        for name in a.routes:
            print(json.dumps(run(a.data, name, a.rounds, a.threads, a.spill, a.f32)), flush=True)
        return
    env = dict(os.environ, POLARS_MAX_THREADS=str(a.threads))
    print(json.dumps({"header": header()}), flush=True)
    for name in a.routes:
        cmd = [sys.executable, __file__, "run", str(a.data), name, "--rounds", str(a.rounds)]
        cmd += ["--threads", str(a.threads)]
        cmd += ["--spill", str(a.spill)] if a.spill is not None else []
        cmd += ["--f32"] if a.f32 else []
        cmd += ["--sample-seed", str(a.sample_seed)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
        line = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
        if res.returncode or not line:
            fail = {"route": name, "failed": res.returncode, "stderr": res.stderr[-2000:]}
            print(json.dumps(fail), flush=True)
        else:
            print(line[-1], flush=True)
        if a.spill is not None:
            for p in a.spill.glob("tmp*"):
                shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
