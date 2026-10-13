"""The memory budgets in ``budgets.toml``: bytes per row of every output
field, and the peak resident memory of a few standard runs.

The bytes come from ``estimated_size()`` of each field of every model at its
fullest -- every output it can emit switched on, ``coef`` on every row --
over the same 20,000 rows, so a field that grows a byte a row, or a dtype
that turns dense (``coef`` as ``Array(Float64, k)`` held 1.6 GiB at a
million rows, where the list holds 7.8 MiB), fails here against a fixed
number rather than at a release. ``estimated_size()`` of the same rows is
the same number on every platform.

The peak resident memory is per platform: each run is a child interpreter,
and its budget is the peak over the baseline after its input is read, in
MiB, which the allocator's arenas move by a few percent from run to run.

Both are held by ``scripts/budgets.py``'s rules: past the budget by more
than the tolerance fails, and so does under it, naming the number to
tighten it to. ``POLARS_ONLINE_BUDGETS=measure`` makes a measuring run,
which reports and never fails; ``POLARS_ONLINE_BUDGETS_OUT=<dir>`` writes
the measurements there as JSON for ``scripts/budgets.py apply``. A run in
GitHub Actions writes its table to the job summary.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import tomllib
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import child
import polars_online as po

TIER = "mixed"
REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


budgets = _load("budgets")

#: The rows every field is measured over.
ROWS = 20_000


def stream(n: int = ROWS, seed: int = 7) -> pl.DataFrame:
    """Three interleaved groups on an irregular clock with two gaps past the
    cap, a session change halfway, null features and targets, zero weights,
    a 0/1 label, a class label, and a block key that only increases."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.5, 1.0, 1.5, 2.0], size=n))
    t[n // 4 :] += 40.0
    t[3 * n // 4 :] += 40.0
    x = rng.standard_normal((n, 3))
    y = 0.5 + x @ np.array([1.0, -0.5, 0.0]) + 0.3 * rng.standard_normal(n)
    z = 0.2 * y + x[:, 2] + 0.3 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.choice(n, n // 25, replace=False)] = 0.0
    x0 = x[:, 0].copy()
    x0[rng.choice(n, n // 100, replace=False)] = np.nan
    yy = y.copy()
    yy[rng.choice(n, n // 100, replace=False)] = np.nan
    return pl.DataFrame(
        {
            "t": t,
            "g": np.array(["a", "b", "c"])[np.arange(n) % 3],
            "s": np.where(np.arange(n) < n // 2, "am", "pm"),
            "block": np.arange(n) // 500,
            "x0": x0,
            "x1": x[:, 1],
            "x2": x[:, 2],
            "y": yy,
            "z": z,
            "w": w,
            "hit": (y > 0.5).astype(float),
            "lab": np.where(y > 0.5, "up", "down"),
        }
    ).with_columns(pl.col("x0", "y").fill_nan(None))


CLOCK = dict(clock="t", gap_cap=6.0, weight="w")
FIT = dict(targets=["y", "z"], features=["x0", "x1", "x2"], group="g", half_life=25.0, **CLOCK)

#: Every output a regression can emit, which every regression below takes
#: but for the ones its model has not (``EXCEPT``). ``coef_every=0`` puts
#: ``coef`` on every row, its largest form.
EVERY_OUTPUT = dict(
    coef_every=0,
    emit_clocks=True,
    emit_sigma=True,
    emit_zscore=True,
    emit_metrics=True,
    conformal=0.9,
    resid_quantiles=[0.1, 0.9],
    emit_autocorr=True,
    emit_drift=True,
    drift_threshold=20.0,
    emit_calibration=True,
    emit_breaks=True,
    emit_specification=True,
    emit_tails=True,
    emit_feature_health=True,
    emit_averaged=True,
    emit_error_inflation=True,
    emit_se_coef=True,
    emit_robust_se=True,
    emit_influence=True,
    emit_selected=True,
)

#: What each regression has not, as the spec checks refuse it.
EXCEPT = {
    "ewridge": {"emit_selected"},
    "rls": {"emit_selected"},
    "lasso": {"emit_error_inflation", "emit_se_coef", "emit_robust_se", "emit_influence"},
    "kalman": {"emit_selected", "emit_robust_se"},
    "huber": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence"},
    "quantile": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence"},
    "ftrl": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence"},
    "sgd": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence"},
    "pa": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence"},
    "holt": {"emit_selected", "emit_error_inflation", "emit_se_coef", "emit_robust_se"}
    | {"emit_influence", "emit_feature_health"},
}


def fullest(builder: str, **kw) -> dict:
    """``kw`` with every output ``builder``'s model can emit switched on."""
    every = {k: v for k, v in EVERY_OUTPUT.items() if k not in EXCEPT[builder]}
    if "emit_drift" not in every:
        every.pop("drift_threshold")
    return {**kw, **every}


#: (name, builder, keyword arguments, the specs it reads): every model at
#: its fullest. The models that are not regressions emit what they emit,
#: with ``emit_clocks`` and, where they have one, ``coef`` on every row.
WORKLOAD: list[tuple[str, str, dict, list[str]]] = [
    ("ewridge", "ewridge", fullest("ewridge", **FIT), []),
    ("rls", "rls", fullest("rls", **FIT), []),
    ("lasso", "lasso", fullest("lasso", **FIT, penalties=[0.2, 0.05, 0.0]), []),
    ("kalman", "kalman", fullest("kalman", **FIT, coef_half_life=50.0), []),
    ("huber", "huber", fullest("huber", **FIT), []),
    ("quantile", "quantile", fullest("quantile", **FIT, quantile=0.8), []),
    ("ftrl", "ftrl", fullest("ftrl", **dict(FIT, targets=["hit"])), []),
    ("sgd", "sgd", fullest("sgd", **FIT, learning_rate=0.01), []),
    ("pa", "pa", fullest("pa", **FIT), []),
    ("holt", "holt", fullest("holt", targets=["y"], group="g", half_life=25.0, **CLOCK), []),
    (
        "ew_cov",
        "ew_cov",
        dict(
            features=["x0", "x1", "x2"],
            stats=["mean", "var", "std", "cov", "corr", "partial_corr", "mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.99],
            pca=2,
            group="g",
            half_life=25.0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    (
        "ew_cov_lags",
        "ew_cov",
        dict(
            features=["x1", "x2"],
            stats=["corr", "lag_corr"],
            lags=[1, 3],
            half_life=25.0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    (
        "kmeans",
        "kmeans",
        dict(features=["x1", "x2"], n_clusters=4, half_life=25.0, coef_every=0, emit_clocks=True)
        | CLOCK,
        [],
    ),
    (
        "micro",
        "micro",
        dict(features=["x1", "x2"], eps=0.5, half_life=25.0, coef_every=0, emit_clocks=True)
        | CLOCK,
        [],
    ),
    (
        "ew_class",
        "ew_class",
        dict(
            features=["x1", "x2"],
            label="lab",
            classes=["down", "up"],
            precision_prior=1.0,
            half_life=25.0,
            coef_every=0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    (
        "seqtest",
        "seqtest",
        dict(targets=["y"], a="ewridge", b="kalman", emit_clocks=True),
        ["ewridge", "kalman"],
    ),
    (
        "marginal",
        "marginal",
        dict(
            targets=["y"],
            features=["x1", "x2"],
            lags=[1],
            bins=4,
            half_life=25.0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    (
        "deco",
        "deco",
        dict(features=["x0", "x1", "x2"], half_life=25.0, coef_every=0, emit_clocks=True, **CLOCK),
        [],
    ),
    (
        "corrchange",
        "corrchange",
        dict(features=["x1", "x2"], window_rows=40, emit_clocks=True, **CLOCK),
        [],
    ),
    (
        "bocpd",
        "bocpd",
        dict(features=["x1", "x2"], covariance_prior=[1.0], emit_clocks=True, **CLOCK),
        [],
    ),
    (
        "hmm",
        "hmm",
        dict(
            features=["x1", "x2"],
            n_components=2,
            precision_prior=0.1,
            half_life=25.0,
            coef_every=0,
            emit_clocks=True,
            **CLOCK,
        ),
        [],
    ),
    (
        "rcov",
        "rcov",
        dict(
            features=["x1", "x2"],
            group="block",
            group_close="monotone",
            block_rows=300,
            emit_clocks=True,
        ),
        [],
    ),
    (
        "audit",
        "audit",
        dict(columns=["x0", "x1", "y"], group="g", pairs=True, clock="t", gap_cap=6.0),
        [],
    ),
]


def measure_fields() -> dict[str, float]:
    """Bytes per row of every output field of every spec in ``WORKLOAD``,
    keyed ``memory.<spec>.<field>``."""
    df = stream()
    built: dict[str, dict] = {}
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.ReadinessWarning)
        for name, builder, kw, reads in WORKLOAD:
            built[name] = getattr(po.spec, builder)(name, **kw)
            result = po.ModelBank([built[r] for r in reads] + [built[name]]).fit_predict(df)
            col = result[name]
            for field in col.struct.fields:
                size = col.struct.field(field).estimated_size()
                out[f"memory.{name}.{field}"] = size / df.height
    return out


def _held(kind: str, measured: dict[str, float], measure: bool = False) -> None:
    """Hold ``measured`` to the budgets, report it, and write it out; with
    ``measure``, or in a measuring run, report without failing."""
    doc = budgets.load()
    measure = measure or budgets.measuring()
    verdicts = budgets.judge_all(doc, measured, measure)
    title = f"{kind} budgets ({sys.platform}, Python {sys.version_info.major}."
    title += f"{sys.version_info.minor})" + (" -- measuring run" if measure else "")
    budgets.write_summary(budgets.report(doc, verdicts, title=title))
    out = os.environ.get("POLARS_ONLINE_BUDGETS_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        name = kind if kind == "memory" else f"{kind}-{PLATFORM}"
        (Path(out) / f"{name}.json").write_text(
            json.dumps(measured, indent=1, sort_keys=True), encoding="utf-8"
        )
    failed = [v.message for v in verdicts if v.failed]
    assert not failed, "\n".join(failed)


# `estimated_size()` is py-polars' count of the buffers: another Polars may count
# them otherwise, so the canary, which unpins it, leaves this out.
@pytest.mark.pins
def test_every_field_of_every_model_at_its_fullest_within_its_budget():
    measured = measure_fields()
    # Every model of the workload measured, at its fullest: a regression's
    # diagnostics are there, so a field that stopped being emitted shows.
    specs = {k.split(".")[1] for k in measured}
    assert specs == {name for name, *_ in WORKLOAD}
    for name, field in (
        ("ewridge", "coef"),
        ("ewridge", "lo_y"),
        ("ewridge", "influence_y"),
        ("ewridge", "se_coef_hc0"),
        ("lasso", "selected_y"),
        ("kalman", "se_coef"),
        ("ew_cov", "mahal_q0.99"),
        ("ew_cov", "pc1_score"),
    ):
        assert f"memory.{name}.{field}" in measured, sorted(k for k in measured if name in k)
    # `coef` on every row: the dense form, which the budget is for.
    assert measured["memory.ewridge.coef"] > 8 * 4
    budgeted = {k for k in budgets.entries(budgets.load()) if k.startswith("memory.")}
    stale = sorted(budgeted - set(measured))
    assert not stale, f"budgets for fields the workload no longer emits: {stale}"
    _held("memory", measured)


# --- peak resident memory ----------------------------------------------------------

PLATFORM = {"linux": "linux", "darwin": "darwin", "win32": "windows"}[sys.platform]

#: The child: read the input, return the allocator's freed memory, take the
#: baseline as the process's *current* resident size, run while a thread
#: samples the current resident size, and report the peak over the baseline in
#: MiB. It is not `ru_maxrss` or Windows' `peak_wset`: those are high-water
#: marks that never fall, so reading the input at a transient peak above the
#: run's own (multi-threaded decompression does that on Linux) left the run
#: no way to raise it, and the delta read 0.0 (CI, 2026-10-12). The sampler
#: sees a sustained peak, which is what a run's output and state are; a spike
#: shorter than its 2 ms period is not what this budget holds.
CHILD = textwrap.dedent(
    """
    import gc, json, sys, threading, time
    import polars as pl, polars_online as po, psutil

    proc = psutil.Process()
    case, src, dst, spec = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    df = None if case.startswith("streaming") else pl.read_parquet(src)
    gc.collect()
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # freed arenas back to the system
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    if case.startswith("streaming"):
        pl.scan_parquet(src).online.fit_predict([spec]).sink_parquet(dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
    else:
        out = po.ModelBank([spec]).fit_predict(df)
        rows = out.height
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    sampler.join()
    print(json.dumps({"rows": rows, "mib": (peak[0] - base) / 2**20}))
    """
)


def _frame(rows: int, k: int, m: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k))
    y = x @ rng.standard_normal((k, m)) + 0.5 * rng.standard_normal((rows, m))
    data = {f"x{j}": x[:, j] for j in range(k)} | {f"y{j}": y[:, j] for j in range(m)}
    return pl.DataFrame(data | {"t": np.cumsum(rng.exponential(1.0, rows))})


def _spec(case: str) -> tuple[dict, pl.DataFrame]:
    if case == "ewridge_10_targets":
        df = _frame(200_000, 20, 10)
        spec = po.spec.ewridge(
            "m", targets=[f"y{j}" for j in range(10)], features=[f"x{j}" for j in range(20)],
            clock="t", half_life=1000.0, gap_cap=100.0,
        )  # fmt: skip
    elif case == "ew_cov_wide":
        df = _frame(200_000, 20, 1)
        spec = po.spec.ew_cov(
            "m", features=[f"x{j}" for j in range(20)], clock="t", half_life=1000.0, gap_cap=100.0
        )
    else:
        df = _frame(1_000_000, 5, 1)
        spec = po.spec.ewridge(
            "m", targets=["y0"], features=[f"x{j}" for j in range(5)], clock="t",
            half_life=1000.0, gap_cap=100.0,
        )  # fmt: skip
    return spec, df


#: The standard runs: a ten-target regression over 200,000 rows (the shape
#: whose dense `coef` cost +3.9 GiB), 230 moments a row over 200,000, and a
#: million rows streamed from a parquet scan to a parquet sink, with Polars
#: reading one row group ahead on four threads. Polars' default read-ahead
#: grows with a file this size: a scan to a sink with no model peaked at 227
#: MiB for a million rows and 473 for three million, and at 95 and 110 with
#: the prefetch at one (2026-10-13), so the bounded run measures this
#: library, about 50-65 MiB of it.
RSS_CASES = ("ewridge_10_targets", "ew_cov_wide", "streaming_bounded")
#: Each case's environment beyond the child's own.
RSS_ENV = {"streaming_bounded": {"POLARS_ROW_GROUP_PREFETCH_SIZE": "1", "POLARS_MAX_THREADS": "4"}}


def _floor_python() -> tuple[int, int]:
    """The oldest Python the package supports, from `requires-python`."""
    spec = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    major, minor = spec["project"]["requires-python"].removeprefix(">=").split(".")[:2]
    return int(major), int(minor)


@pytest.mark.extended(reason="memory: three child interpreters, up to a million rows each")
@pytest.mark.serial
@pytest.mark.pins
def test_the_peak_memory_of_the_standard_runs_within_its_budget(tmp_path):
    measured = {}
    for case in RSS_CASES:
        spec, df = _spec(case)
        src, dst = tmp_path / f"{case}.parquet", tmp_path / f"{case}.out.parquet"
        df.write_parquet(src)
        r = subprocess.run(
            [sys.executable, "-c", CHILD, case, str(src), str(dst), json.dumps(spec)],
            capture_output=True, text=True, encoding="utf-8",
            env=child.env(**RSS_ENV.get(case, {})), timeout=600,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr[-3000:]
        got = json.loads(r.stdout.strip().splitlines()[-1])
        # The run happened, over every row, and took memory to do it.
        assert got["rows"] == df.height, (case, got)
        assert got["mib"] > 1.0, (case, got)
        measured[f"rss.{case}.{PLATFORM}"] = got["mib"]
    budgeted = {k for k in budgets.entries(budgets.load()) if k.startswith("rss.")}
    stale = sorted(k for k in budgeted if k.split(".")[1] not in RSS_CASES)
    assert not stale, f"budgets for runs this test no longer makes: {stale}"
    # Held where the budget is measured, CI's leg at each OS's floor Python,
    # and reported everywhere else. The interpreter and the machine each move
    # the peak by about a fifth by themselves (2026-10-12): on CI's
    # macOS runner 3.14 read 18-20% under 3.12, and the streaming run read
    # 291 MiB on a 14-core M4 Pro against 237 on that runner, one build.
    held_here = os.environ.get("CI") == "true" and sys.version_info[:2] == _floor_python()
    _held("rss", measured, measure=not held_here)
