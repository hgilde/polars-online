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
the measurements there as JSON for ``scripts/budgets.py apply`` -- the
resident memory only on the leg that holds its budget, so no other leg's
peaks can be applied to it. A run in GitHub Actions writes its table to the
job summary.
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

#: What each regression has not, as the spec checks refuse it. A single slot
#: has nothing to choose or average across, so ``emit_selected`` and
#: ``emit_averaged`` go wherever there is no grid (lasso's path is one; the
#: refusal of ``emit_averaged`` on one slot is task 278's).
ONE_SLOT = {"emit_selected", "emit_averaged"}
#: And what a model with no least-squares fit has not.
NOT_LEAST_SQUARES = {"emit_error_inflation", "emit_se_coef", "emit_robust_se", "emit_influence"}
EXCEPT = {
    "ewridge": ONE_SLOT,
    "rls": ONE_SLOT,
    "lasso": NOT_LEAST_SQUARES,
    "kalman": ONE_SLOT | {"emit_robust_se"},
    "huber": ONE_SLOT | NOT_LEAST_SQUARES,
    "quantile": ONE_SLOT | NOT_LEAST_SQUARES,
    "ftrl": ONE_SLOT | NOT_LEAST_SQUARES,
    "sgd": ONE_SLOT | NOT_LEAST_SQUARES,
    "pa": ONE_SLOT | NOT_LEAST_SQUARES,
    "holt": ONE_SLOT | NOT_LEAST_SQUARES | {"emit_feature_health"},
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
    # The resident memory only where its budget is held: every leg wrote it
    # under one name per platform, so `apply` could take a 3.14 leg's file.
    if out and (kind != "rss" or _rss_held_here()):
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
    df = None if case.startswith(("streaming", "collect")) else pl.read_parquet(src)
    if case.startswith("predict"):
        # Fitted before the baseline: the run is the scoring alone.
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
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
    elif case.startswith("collect"):
        # The negative control: the source read whole inside the run, and
        # then handed to the same plan.
        pl.read_parquet(src).lazy().online.fit_predict([spec]).sink_parquet(dst)
        rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
    elif case.startswith("predict"):
        out = bank.predict(df)
        rows = out.height
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
    elif case == "predict_wide":
        df = _frame(20_000, 1_000, 1)
        spec = po.spec.sgd(
            "m", targets=["y0"], features=[f"x{j}" for j in range(1_000)], clock="t",
            half_life=1000.0, gap_cap=100.0, learning_rate=0.001,
        )  # fmt: skip
    elif case == "ewridge_window":
        df = _frame(200_000, 20, 1)
        spec = po.spec.ewridge(
            "m", targets=["y0"], features=[f"x{j}" for j in range(20)], clock="t",
            half_life=1000.0, gap_cap=100.0, window_size=20_000.0,
        )  # fmt: skip
    elif case == "bocpd_runs":
        df = _frame(50_000, 4, 1)
        spec = po.spec.bocpd(
            "m", features=[f"x{j}" for j in range(4)], clock="t", gap_cap=100.0,
            prune_below=1e-6, max_run=200,
        )  # fmt: skip
    else:
        df = _frame(STREAMED_ROWS, 5, 1)
        spec = po.spec.ewridge(
            "m", targets=["y0"], features=[f"x{j}" for j in range(5)], clock="t",
            half_life=1000.0, gap_cap=100.0,
        )  # fmt: skip
    return spec, df


#: Rows of the streamed run: five times what a budget near 200 MiB could
#: hold whole. The frame is 267 MiB in memory and 251 MiB as parquet.
STREAMED_ROWS = 5_000_000

#: The standard runs: a ten-target regression over 200,000 rows (the shape
#: whose dense `coef` cost +3.9 GiB), 230 moments a row over 200,000, an eager
#: `predict` of 1,000 features over 20,000 rows (it copies the task's
#: features into rows, `n * k * 8` bytes), a windowed `ewridge` (its ring of
#: snapshots), `bocpd` holding up to 200 runs, and five million rows streamed
#: from a parquet scan to a parquet sink.
#:
#: The streamed run is what holds the plan to a bounded peak, so its input is
#: larger than any budget it could pass whole (review round 9, J-1: at a
#: million rows, 53 MiB, the plan fed a frame collected first peaked at 185.8
#: MiB, under the streaming run's own 192.8 and inside the budget). It rules
#: out a source read whole, a read-ahead without limit and chunks kept
#: instead of yielded: `collect_then_plan`, the same plan on the frame read
#: whole inside the run, must exceed the streamed run's budget by more than
#: its tolerance (`test_the_peak_memory_of_the_standard_runs_within_its_budget`).
#: Polars reads one row group ahead on four threads, and the allocator gives
#: back freed pages at once, so the peak is the live memory: on 2026-10-23
#: (macOS, load 21-23) 131 / 157 / 163 MiB streamed at 1M / 3M / 5M rows,
#: against 148 / 265 / 383 collected first. With the allocator's default page
#: retention the streamed peak grows with the file, 208 / 330 / 340 MiB, and
#: at a million rows the two could not be told apart (208 against 207).
RSS_CASES = (
    "ewridge_10_targets",
    "ew_cov_wide",
    "predict_wide",
    "ewridge_window",
    "bocpd_runs",
    "streaming_5m",
)
#: Each case's environment beyond the child's own: the streamed run's and its
#: control's. `_RJEM_MALLOC_CONF` is Polars' jemalloc on Linux and macOS, and
#: `MIMALLOC_PURGE_DELAY` its mimalloc on Windows.
STREAMED_ENV = {
    "POLARS_ROW_GROUP_PREFETCH_SIZE": "1",
    "POLARS_MAX_THREADS": "4",
    "_RJEM_MALLOC_CONF": "dirty_decay_ms:0,muzzy_decay_ms:0",
    "MIMALLOC_PURGE_DELAY": "0",
}
RSS_ENV = {"streaming_5m": STREAMED_ENV, "collect_then_plan": STREAMED_ENV}


def _floor_python() -> tuple[int, int]:
    """The oldest Python the package supports, from `requires-python`."""
    spec = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    major, minor = spec["project"]["requires-python"].removeprefix(">=").split(".")[:2]
    return int(major), int(minor)


#: Set to ``false`` by CI on a leg whose peak is reported but not held: the
#: second Windows image, `windows-2022`, beside `windows-latest`, which
#: holds the one ``rss.*.windows`` budget (ci.yml's pytest step says why).
RSS_HELD_VARIABLE = "POLARS_ONLINE_RSS_HELD"


def _rss_held_here() -> bool:
    """Whether this run holds the resident memory to its budget: CI's leg at
    each OS's floor Python, on the one image per OS that holds it."""
    return (
        os.environ.get("CI") == "true"
        and sys.version_info[:2] == _floor_python()
        and os.environ.get(RSS_HELD_VARIABLE, "true") != "false"
    )


def _peak(case: str, spec: dict, src: Path, dst: Path, rows: int) -> float:
    """The peak over the baseline of ``case`` in a child interpreter, MiB."""
    r = subprocess.run(
        [sys.executable, "-c", CHILD, case, str(src), str(dst), json.dumps(spec)],
        capture_output=True, text=True, encoding="utf-8",
        env=child.env(**RSS_ENV.get(case, {})), timeout=600,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    # The run happened, over every row, and took memory to do it.
    assert got["rows"] == rows, (case, got)
    assert got["mib"] > 1.0, (case, got)
    return got["mib"]


@pytest.mark.extended(reason="memory: seven child interpreters, up to five million rows each")
@pytest.mark.serial
@pytest.mark.pins
def test_the_peak_memory_of_the_standard_runs_within_its_budget(tmp_path):
    measured = {}
    for case in RSS_CASES:
        spec, df = _spec(case)
        src, dst = tmp_path / f"{case}.parquet", tmp_path / f"{case}.out.parquet"
        df.write_parquet(src)
        measured[f"rss.{case}.{PLATFORM}"] = _peak(case, spec, src, dst, df.height)
        if case == "streaming_5m":
            # The negative control: the same rows read whole and then planned
            # must fail the streamed run's budget, or the budget cannot see
            # the regression it is for.
            control = _peak("collect_then_plan", spec, src, dst, df.height)
            doc = budgets.load()
            tolerance = doc["meta"]["tolerance"]["rss"]
            budget = doc["rss"].get("streaming_5m", {}).get(PLATFORM, {}).get("budget", 0.0)
            ceiling = (1 + tolerance) * max(measured[f"rss.{case}.{PLATFORM}"], budget)
            assert control > ceiling, (control, ceiling, measured)
        src.unlink()
    budgeted = {k for k in budgets.entries(budgets.load()) if k.startswith("rss.")}
    stale = sorted(k for k in budgeted if k.split(".")[1] not in RSS_CASES)
    assert not stale, f"budgets for runs this test no longer makes: {stale}"
    # Held where the budget is measured, CI's leg at each OS's floor Python,
    # and reported everywhere else. The interpreter and the machine each move
    # the peak by about a fifth by themselves (2026-10-12): on CI's
    # macOS runner 3.14 read 18-20% under 3.12, and the streaming run read
    # 291 MiB on a 14-core M4 Pro against 237 on that runner, one build.
    _held("rss", measured, measure=not _rss_held_here())


@pytest.mark.parametrize(
    ("floor", "image_held"), [(True, None), (True, "true"), (True, "false"), (False, None)]
)
def test_the_resident_memory_is_written_out_only_where_its_budget_is_held(
    tmp_path, monkeypatch, floor, image_held
):
    """Every CI leg wrote ``rss-<platform>.json`` under one name, so
    ``budgets.py apply`` given a 3.14 leg's file would set the floor's
    budgets from peaks up to 7% apart (review 8, E-5). Only the leg that
    holds the budget writes it. Of the two Windows images at the floor,
    `windows-2022` is told it does not (task 273 (a2)): the two read the
    six runs up to 8% apart on one build, and each failed the budget as
    one to tighten."""
    module = sys.modules[__name__]
    here = sys.version_info[:2]
    monkeypatch.setattr(module, "_floor_python", lambda: here if floor else (here[0], here[1] + 1))
    monkeypatch.setenv("CI", "true")
    if image_held is None:
        monkeypatch.delenv(RSS_HELD_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(RSS_HELD_VARIABLE, image_held)
    held = floor and image_held != "false"
    monkeypatch.setenv("POLARS_ONLINE_BUDGETS_OUT", str(tmp_path))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    measured = {f"rss.{case}.{PLATFORM}": 1.0 for case in RSS_CASES}
    _held("rss", measured, measure=True)
    written = tmp_path / f"rss-{PLATFORM}.json"
    assert written.exists() == held
    if held:
        assert json.loads(written.read_text(encoding="utf-8")) == measured
    # The bytes per row are the same number on every leg, written by each.
    _held("memory", {"memory.ewridge.pred_y": 8.125}, measure=True)
    assert (tmp_path / "memory.json").exists()
