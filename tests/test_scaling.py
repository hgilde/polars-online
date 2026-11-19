"""How the Python surfaces' cost grows with the specs on one stream and with
the stream's length (docs/PERFORMANCE.md, "How cost grows with size").

The Rust cells (``crates/rheon-bench/tests/scaling.rs``) hold the bank, the
runner and the models by their allocations and saved bytes as well as their
time. What the Python surfaces add on top is their own per-call work: the
chunked :class:`~rheon.polars.ModelBank` call, and the IO source behind
``LazyFrame.online.fit_predict``, which reads the plan's batches and builds
each output frame. Each case here times one workload at several sizes, the
least of several rounds interleaved, in CPU time. A stream's length is held
by the ratio of the time per row at two sizes to 1, times ``STEP ** 0.5``:
time is the coarse measure, and a wrong growth is a whole power of the step
past the declared one. The specs are held by three sizes and the ratio of
the cost of the last increment to the first's, which is the same for any
cost a stream pays whatever its specs, so that a cost quadratic in the specs
shows even beside a large fixed one. ``_hold`` and ``_hold_linear`` are
themselves tested at the end of the file, on busy loops of known cost and on
a clock too coarse to time them.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "extended"
pytestmark = pytest.mark.extended(reason="timing: each case runs two sizes five times")

#: The step between a case's two sizes, and how far past its declared growth
#: a time per row may read.
STEP = 8
SLACK = STEP**0.5
RUNS = 5
#: A case that reads over its bound after ``RUNS`` rounds takes more, up to
#: this many: the least only falls with more rounds, so a busy machine's
#: noise washes out, while a wrong growth stays past the bound however many.
MAX_RUNS = 20


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
        rp.spec.ewridge(
            f"m{s}",
            targets=["y0"],
            features=["x0", "x1", "x2", "x3"],
            clock="t",
            half_life=1000.0,
            gap_cap=100.0,
        )
        for s in range(n)
    ]


# Windows counts process CPU time in 15.6 ms ticks (though it reports a
# 100 ns resolution), so a run shorter than a tick reads as 0 and a ratio of
# two of them divides by zero; there the wall clock, whose ticks are under a
# microsecond, is the finer measure.
_CLOCK = time.perf_counter if sys.platform == "win32" else time.process_time

#: A reading must span this many ticks of the clock to be a measure.
MIN_TICKS = 10


def _seconds(fn: Callable[[], object]) -> float:
    """The CPU time ``fn`` took, all this process's threads: on a busy machine
    the wall clock also counts the time a run spent waiting for a core, which
    differs between a short run and a long one and read as growth. Where the
    CPU clock is too coarse to time a run (Windows), the wall clock."""
    at = _CLOCK()
    fn()
    return _CLOCK() - at


def _floor() -> float:
    """The shortest reading that is a measure: ``MIN_TICKS`` of the clock's
    real step, the least positive difference between successive readings
    (the 100 ns a clock reports can be a 15.6 ms step in fact), found by
    reading it until it moves. Infinite where it never moves in half a
    second, so that nothing is long enough."""
    gaps: list[float] = []
    last = _CLOCK()
    end = time.perf_counter() + 0.5
    while len(gaps) < 3 and time.perf_counter() < end:
        now = _CLOCK()
        if now != last:
            gaps.append(now - last)
            last = now
    return MIN_TICKS * min(gaps) if gaps else float("inf")


def _resolved(name: str, floor: float, *readings: float) -> None:
    """Fail where a reading is under ``floor``: a run the clock cannot
    resolve reads 0 or one whole tick, and a ratio of two such readings is a
    division by zero or a number of ticks, never a measure."""
    shortest = min(readings)
    assert shortest >= floor, (
        f"{name}: a run of {shortest * 1e3:.3f} ms is too short to time, "
        f"under {MIN_TICKS} ticks of the clock ({floor * 1e3:.3f} ms). Give the case more work."
    )


def _hold(
    name: str,
    small: Callable[[], object],
    large: Callable[[], object],
    bound: float,
    rows: tuple[int, int],
) -> None:
    """The least time per row of ``large`` over ``small``'s, within ``bound``
    times the slack."""
    floor = _floor()
    best_s, best_l = float("inf"), float("inf")
    allowed = bound * SLACK
    ratio = float("inf")
    for run in range(MAX_RUNS):
        # One round: both sizes back to back, so a change in load meets both.
        best_s = min(best_s, _seconds(small))
        best_l = min(best_l, _seconds(large))
        if best_s < floor or best_l < floor:
            continue
        ratio = (best_l / rows[1]) / (best_s / rows[0])
        if run + 1 >= RUNS and ratio <= allowed:
            break
    _resolved(name, floor, best_s, best_l)
    print(f"{name}: {ratio:.2f}x per row (allowed {allowed:.2f}x)")
    assert ratio <= allowed, f"{name}: a row's time grew {ratio:.2f}x, past {allowed:.2f}x"


#: The sizes of a case that grows with the specs, in units of its smallest:
#: three, so that the cost a stream pays whatever its specs (a call, a frame,
#: a collect) cancels from the difference of two readings.
GROWTH = (1, 4, 16)
#: How far past linear the ratio of the increments may read. Linear gives
#: ``(16 - 4) / (4 - 1)`` = 4 whatever the fixed cost; quadratic gives 16.
#: Twice linear, 8: the bank's case of one spec at 4 rows a call read 4.6 to
#: 5.0 quiet and past 6 under a full gate's load (14 workers), while a
#: quadratic cost reads 16.
INCREMENT_SLACK = 2.0
#: The smaller increment counts at least this share of the smallest run, so
#: that where a spec costs little beside the fixed cost, noise in the two
#: smallest readings does not make a ratio of it. At 5% the bank's case of one
#: spec at 4 rows a call read 7 to 11 on a loaded machine (load 15 to 50),
#: and 4.8 to 5.0 on a quiet one. A quadratic cost still reads 16 at a fixed
#: cost of 0, 5 or 50 specs, since its first increment is past a quarter of
#: the smallest run there.
INCREMENT_FLOOR = 0.25


def _hold_linear(name: str, runs: tuple[Callable[[], object], ...]) -> None:
    """The cost of ``runs``, at ``GROWTH`` times their smallest's specs, grows
    linearly in the specs: the least times ``t1 < t2 < t3`` have
    ``(t3 - t2) / (t2 - t1)`` within ``INCREMENT_SLACK`` of the linear
    ratio. A time that is a fixed cost ``F`` and ``c`` a spec, ``F + c n``,
    gives that ratio at any ``F``; ``F + c n ** 2`` reads 16, up to an ``F``
    of several hundred ``c``."""
    floor = _floor()
    best = [float("inf")] * len(runs)
    linear = (GROWTH[2] - GROWTH[1]) / (GROWTH[1] - GROWTH[0])
    allowed = linear * INCREMENT_SLACK
    ratio = float("inf")
    for run in range(MAX_RUNS):
        # One round: every size back to back, so a change in load meets all.
        for i, fn in enumerate(runs):
            best[i] = min(best[i], _seconds(fn))
        if min(best) < floor:
            continue
        t1, t2, t3 = best
        ratio = (t3 - t2) / max(t2 - t1, INCREMENT_FLOOR * t1)
        if run + 1 >= RUNS and ratio <= allowed:
            break
    _resolved(name, floor, *best)
    print(f"{name}: increments {ratio:.2f}x (linear {linear:.0f}x, allowed {allowed:.1f}x)")
    assert ratio <= allowed, (
        f"{name}: the cost of the last {GROWTH[2] - GROWTH[1]} units of specs is {ratio:.2f} "
        f"times the first {GROWTH[1] - GROWTH[0]}'s, past {allowed:.1f} (linear is {linear:.0f})"
    )


@pytest.mark.parametrize(("small", "chunk"), [(1, 2048), (8, 2048), (1, 4), (8, 4)])
def test_a_chunked_bank_costs_each_spec_the_same(small: int, chunk: int) -> None:
    """``ModelBank.fit_predict`` over one stream, a call a chunk, with
    ``small``, ``4 * small`` and ``16 * small`` specs over the same columns: linear in the
    specs, a row at a time (``chunk`` 2,048) and a call at a time (4), where
    what a call costs a spec -- its columns looked up, its output built --
    is the cost."""
    rows = 4096 if chunk > 4 else 512
    df = _frame(2048 + MAX_RUNS * rows)
    warm = df.head(2048)
    banks = {}
    for n in (g * small for g in GROWTH):
        bank = rp.ModelBank(_specs(n), threads=1)
        bank.fit_predict(warm)
        banks[n] = bank
    # Each run feeds the next block of the stream, so a bank's clock moves on.
    blocks = [df.slice(2048 + r * rows, rows) for r in range(MAX_RUNS)]
    at = {n: 0 for n in banks}

    def run(n: int) -> Callable[[], object]:
        def go() -> object:
            block = blocks[at[n]]
            at[n] += 1
            return [banks[n].fit_predict(block.slice(i, chunk)) for i in range(0, rows, chunk)]

        return go

    _hold_linear(
        f"bank specs {small}-{GROWTH[-1] * small}, chunk {chunk}",
        tuple(run(g * small) for g in GROWTH),
    )


@pytest.mark.parametrize(("small", "chunk"), [(1, 2048), (8, 2048), (8, 16)])
def test_the_io_source_costs_each_spec_the_same(small: int, chunk: int) -> None:
    """``LazyFrame.online.fit_predict``, collected, with ``small``,
    ``4 * small`` and ``16 * small`` specs: the IO source's batches and their output frames,
    linear in the specs, at a large chunk and at a small one."""
    rows = 8192 if chunk > 16 else 1024
    lf = _frame(rows).lazy()

    def run(n: int) -> Callable[[], object]:
        specs = _specs(n)
        return lambda: lf.online.fit_predict(specs, chunk_size=chunk, threads=1).collect()

    _hold_linear(
        f"io source specs {small}-{GROWTH[-1] * small}, chunk {chunk}",
        tuple(run(g * small) for g in GROWTH),
    )


def test_the_io_source_costs_a_row_the_same_however_long_the_stream() -> None:
    """``LazyFrame.online.fit_predict`` over ``n`` and ``8n`` rows: a row's
    time must not grow with the rows before it."""
    n = 16_384

    def run(rows: int) -> Callable[[], object]:
        lf = _frame(rows).lazy()
        specs = _specs(1)
        return lambda: lf.online.fit_predict(specs, chunk_size=2048, threads=1).collect()

    _hold(f"io source rows {n}->{STEP * n}", run(n), run(STEP * n), 1.0, (n, STEP * n))


def _spin(seconds: float) -> Callable[[], object]:
    """A run that holds the clock for ``seconds`` of CPU time, so that its
    reading is the cost the case declares."""

    def go() -> None:
        end = _CLOCK() + seconds
        while _CLOCK() < end:
            pass

    return go


#: One unit of a self-test's cost, a spec: long enough for a clock of
#: microseconds, short enough that five rounds of the largest run take about
#: a second.
_UNIT = 0.0005


def _busy(fixed: float, power: int) -> tuple[Callable[[], object], ...]:
    """Runs at ``GROWTH`` of a case costing ``fixed`` units and ``n ** power``
    for ``n`` specs."""
    return tuple(_spin(_UNIT * (fixed + g**power)) for g in GROWTH)


@pytest.mark.parametrize("fixed", [0, 5, 50])
def test_the_growth_check_passes_a_linear_cost_at_any_fixed_cost(fixed: int) -> None:
    """C3: a fixed cost ``F`` of 0, 5 and 50 specs and one spec each, the
    cost the cases declare, passes."""
    _hold_linear(f"busy linear, fixed {fixed}", _busy(fixed, 1))


@pytest.mark.parametrize("fixed", [0, 5, 50])
def test_the_growth_check_fails_a_quadratic_cost_at_any_fixed_cost(fixed: int) -> None:
    """C3: a cost of ``F + n ** 2`` units fails, at ``F`` of 0, 5 and 50: the
    two-size ratio this replaces read ``(F + 64) / (F + 1)`` against 22.6 and
    passed a quadratic cost from an ``F`` of 5."""
    with pytest.raises(AssertionError, match="past"):
        _hold_linear(f"busy quadratic, fixed {fixed}", _busy(fixed, 2))


def test_a_stream_length_check_holds_a_row_the_same_and_fails_one_that_grows() -> None:
    """The two-size check, on a cost per row that is constant and on one that
    grows eightfold with the stream's length."""
    flat = (_spin(_UNIT * 8), _spin(_UNIT * 64))
    _hold("busy flat", *flat, 1.0, (8, 64))
    grows = (_spin(_UNIT * 8), _spin(_UNIT * 64 * 8))
    with pytest.raises(AssertionError, match="a row's time grew"):
        _hold("busy growing", *grows, 1.0, (8, 64))


def test_a_clock_too_coarse_for_a_run_fails_by_name_and_does_not_pass_on_zero() -> None:
    """C2: on a clock that steps 15.6 ms, as Windows' CPU clock does, runs of
    1 and 8 ms read 0 (or a whole tick), and their ratio was a
    ``ZeroDivisionError`` where the run's two readings were both 0 and a pass
    where they were. Both checks fail, naming the clock."""

    def stepped() -> float:
        return math.floor(time.perf_counter() / 0.0156) * 0.0156

    def real_spin(seconds: float) -> Callable[[], object]:
        def go() -> None:
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                pass

        return go

    clock = pytest.MonkeyPatch()
    clock.setattr(sys.modules[__name__], "_CLOCK", stepped)
    try:
        with pytest.raises(AssertionError, match="too short to time"):
            _hold("coarse", real_spin(0.001), real_spin(0.008), STEP, (1, 1))
        with pytest.raises(AssertionError, match="too short to time"):
            _hold_linear("coarse", tuple(real_spin(0.001 * g) for g in GROWTH))
        # The same two runs on a fine clock are no trouble to read.
    finally:
        clock.undo()
    _hold("fine", real_spin(0.001), real_spin(0.008), STEP, (1, 1))
