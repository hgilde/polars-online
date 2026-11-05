"""How the Python surfaces' cost grows with the specs on one stream and with
the stream's length (docs/PERFORMANCE.md, "How cost grows").

The Rust cells (``crates/online-bench/tests/scaling.rs``) hold the bank, the
runner and the models by their allocations and saved bytes as well as their
time. What the Python surfaces add on top is their own per-call work: the
chunked :class:`~polars_online.ModelBank` call, and the IO source behind
``LazyFrame.online.fit_predict``, which reads the plan's batches and builds
each output frame. Each case here times one workload at two sizes, the least
of several runs interleaved, and holds the ratio of the time per row to the
growth declared for it, times ``STEP ** 0.5``: time is the coarse measure,
and a wrong growth is a whole power of the step past the declared one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "extended"
pytestmark = pytest.mark.extended(reason="timing: each case runs two sizes five times")

#: The step between a case's two sizes, and how far past its declared growth
#: a time per row may read.
STEP = 8
SLACK = STEP**0.5
RUNS = 5


def _frame(n: int, k: int = 4, t: int = 1, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, k))
    cols = {f"x{j}": x[:, j] for j in range(k)}
    for j in range(t):
        cols[f"y{j}"] = x @ (1.0 / np.arange(1, k + 1)) + j + 0.3 * rng.uniform(-1, 1, n)
    cols["t"] = np.arange(n, dtype=np.float64)
    return pl.DataFrame(cols)


def _specs(n: int) -> list[dict[str, Any]]:
    return [
        po.spec.ewridge(
            f"m{s}",
            targets=["y0"],
            features=["x0", "x1", "x2", "x3"],
            clock="t",
            half_life=1000.0,
            gap_cap=100.0,
        )
        for s in range(n)
    ]


def _seconds(fn: Callable[[], object]) -> float:
    at = time.perf_counter()
    fn()
    return time.perf_counter() - at


def _hold(name: str, small: Callable[[], object], large: Callable[[], object], bound: float, rows: tuple[int, int]) -> None:
    """The least time per row of ``large`` over ``small``'s, within ``bound``
    times the slack."""
    best_s, best_l = float("inf"), float("inf")
    for _ in range(RUNS):
        best_s = min(best_s, _seconds(small) / rows[0])
        best_l = min(best_l, _seconds(large) / rows[1])
    ratio = best_l / best_s
    allowed = bound * SLACK
    print(f"{name}: {ratio:.2f}x per row (allowed {allowed:.2f}x)")
    assert ratio <= allowed, f"{name}: a row's time grew {ratio:.2f}x, past {allowed:.2f}x"


@pytest.mark.parametrize(("small", "chunk"), [(1, 2048), (8, 2048), (1, 4), (8, 4)])
def test_a_chunked_bank_costs_each_spec_the_same(small: int, chunk: int) -> None:
    """``ModelBank.fit_predict`` over one stream, a call a chunk, with
    ``small`` and ``8 * small`` specs over the same columns: linear in the
    specs, a row at a time (``chunk`` 2,048) and a call at a time (4), where
    what a call costs a spec -- its columns looked up, its output built --
    is the cost."""
    rows = 4096 if chunk > 4 else 512
    df = _frame(2048 + rows)
    warm, block = df.head(2048), df.slice(2048)
    banks = {}
    for n in (small, STEP * small):
        bank = po.ModelBank(_specs(n), threads=1)
        bank.fit_predict(warm)
        banks[n] = bank
    chunks = [block.slice(i, chunk) for i in range(0, rows, chunk)]

    def run(n: int) -> Callable[[], object]:
        return lambda: [banks[n].fit_predict(c) for c in chunks]

    _hold(f"bank specs {small}->{STEP * small}, chunk {chunk}", run(small), run(STEP * small), STEP, (rows, rows))


@pytest.mark.parametrize(("small", "chunk"), [(1, 2048), (8, 2048), (8, 16)])
def test_the_io_source_costs_each_spec_the_same(small: int, chunk: int) -> None:
    """``LazyFrame.online.fit_predict``, collected, with ``small`` and
    ``8 * small`` specs: the IO source's batches and their output frames,
    linear in the specs, at a large chunk and at a small one."""
    rows = 8192 if chunk > 16 else 1024
    lf = _frame(rows).lazy()

    def run(n: int) -> Callable[[], object]:
        specs = _specs(n)
        return lambda: lf.online.fit_predict(specs, chunk_size=chunk, threads=1).collect()

    _hold(f"io source specs {small}->{STEP * small}, chunk {chunk}", run(small), run(STEP * small), STEP, (rows, rows))


def test_the_io_source_costs_a_row_the_same_however_long_the_stream() -> None:
    """``LazyFrame.online.fit_predict`` over ``n`` and ``8n`` rows: a row's
    time must not grow with the rows before it."""
    n = 16_384

    def run(rows: int) -> Callable[[], object]:
        lf = _frame(rows).lazy()
        specs = _specs(1)
        return lambda: lf.online.fit_predict(specs, chunk_size=2048, threads=1).collect()

    _hold(f"io source rows {n}->{STEP * n}", run(n), run(STEP * n), 1.0, (n, STEP * n))
