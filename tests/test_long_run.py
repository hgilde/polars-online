"""Task 278 (c), class 9: long-run stability, at 1e5 and 1e6 rows.

The weekly soak (`tests/test_long_run_soak.py`) runs the same cases at 1e8
rows; the measurements and their oracles are in `tests/long_run.py`. A case
is judged on every checkpoint, not its end, and against a bound that grows
no faster than ``sqrt(n)`` rounding errors: rounding that does not
accumulate systematically stays inside it, and a drift that grows with the
rows (a bias per row) leaves it by the soak's 1e8 rows.

Each bound is the measurement with headroom, measured 2026-10-25 on macOS
arm64 at 1e8 rows (the soak's), the worst checkpoint quoted.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import polars as pl
import pytest

import long_run as lr
import polars_online as po
from public_benchmarks import numacc_or_skip

TIER = "mixed"

EPS = 2.0**-52


def _within(points: list, column: int, a: float, b: float) -> list:
    """The checkpoints whose error in ``column`` exceeds ``a + b * sqrt(n) * EPS``."""
    return [p for p in points if p[column] > a + b * math.sqrt(p[0]) * EPS]


# --- NIST StRD NumAcc1-4: certified moments at a large offset ----------------


@pytest.mark.parametrize("i", [1, 2, 3, 4])
def test_numacc_moments_are_the_certified_ones(i):
    """``ew_cov`` with no decay over each NumAcc set. The oracle is the exact
    mean and Bessel-corrected standard deviation of the doubles fed
    (fractions), checked against NIST's certificate within the rounding of
    the decimals to doubles. Measured: mean within 9.3e-17 relative, sd
    within 1.5e-15 (NumAcc4, level 1e7, spread 0.1). Bounds 4 and 32
    roundings."""
    x, cert_mean, cert_sd = numacc_or_skip(i)
    exact = [Fraction(v) for v in x]
    mean = sum(exact) / len(exact)
    var = sum((e - mean) ** 2 for e in exact) / (len(exact) - 1)
    sd = math.sqrt(var)
    # The doubles are the certificate's decimals to within their rounding.
    assert abs(mean - cert_mean) <= abs(cert_mean) * EPS
    assert abs(sd - float(cert_sd)) <= 2 * float(np.max(np.abs(x))) * EPS / float(cert_sd)

    spec = po.spec.ew_cov("c", features=["x"], half_life=float("inf"), stats=["mean", "std"])
    out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x": np.append(x, x[0])})).unnest("c")
    got_mean, got_sd = out["mean_x"][-1], out["std_x"][-1]
    assert abs(Fraction(got_mean) - mean) <= abs(mean) * 4 * EPS, (got_mean, float(mean))
    assert abs(got_sd - sd) <= sd * 32 * EPS, (got_sd, sd)

    if i >= 3:
        # The hard path: the textbook E[x^2] - E[x]^2 in doubles loses the
        # spread at this offset, which is what the certificate tests.
        naive = math.sqrt(max(0.0, (np.mean(x * x) - np.mean(x) ** 2) * len(x) / (len(x) - 1)))
        assert abs(naive - sd) > 1e-3 * sd, naive


@pytest.mark.oracle_grid(reason="a million rows of NumAcc4 through the bank, a second or more")
def test_numacc4_cycled_a_million_rows_does_not_drift():
    """NumAcc4 repeated to 1e6 rows. Measured at 1e8: the mean within 9.3e-17
    at every checkpoint; the sd within 1.4e-13 (2e7 to 1e8 rows) and 1.7e-14
    at 1e6, about 0.15 ``sqrt(n)`` roundings. Bounds: the mean 4 roundings
    flat, the sd ``16 + 1.0 sqrt(n)`` roundings."""
    x, _, _ = numacc_or_skip(4)
    points = lr.numacc_cycled(x, 1_000_000)
    assert points[-1][0] >= 999_000
    assert not _within(points, 1, 4 * EPS, 0.0), points
    assert not _within(points, 2, 16 * EPS, 1.0), points


# --- The bank's accumulators against a recomputation from scratch ------------

#: Each case's bound on its error columns, ``(column, a, b)`` for ``a + b
#: sqrt(n) EPS``, with the 1e8-row measurement each was set from.
BOUNDS = {
    # mean 1.2e-16, sd 5.8e-16 at every checkpoint: flat, as a decayed
    # mean forgets its own rounding.
    "decayed": [(1, 4 * EPS, 0.0), (2, 32 * EPS, 0.0)],
    # mean 2.3e-16; sd 2.0e-12 (at 2,000 rows) and no larger at 1e8: the
    # window is a subtraction, and at a level of 1e6 over a spread of 1 the
    # discarded snapshot's moments cancel about six digits of the level.
    "windowed": [(1, 4 * EPS, 0.0), (2, 1e-11, 0.0)],
    # 8.2e-13 at 2e7 rows, 1.6 sqrt(n) roundings; the recomputation itself
    # is a chunked fsum and a solve, good to about 1e-14.
    "no_decay": [(1, 1e-13, 4.0)],
    # 1.9e-14 at 2e6 rows, flat.
    "decayed_ridge": [(1, 1e-13, 0.0)],
    # weight_sum at 4.1e-14 (float clock) and 1.4e-13 (Datetime clock),
    # flat from the steady state on: each row's rounding is held for about
    # 1 / (1 - a) = 1,443 rows, so the bound is four times eps / (1 - a).
    "float": [(1, 4 * EPS * 1443, 0.0)],
    "temporal": [(1, 4 * EPS * 1443, 0.0)],
}

CASES = {
    "decayed": lambda rows: lr.ew_cov_decayed(rows),
    "windowed": lambda rows: lr.ew_cov_decayed(rows, window=150.0),
    "no_decay": lr.ewridge_no_decay,
    "decayed_ridge": lr.ewridge_decayed,
    "float": lambda rows: lr.clock_steady_state(rows, temporal=False),
    "temporal": lambda rows: lr.clock_steady_state(rows, temporal=True),
}


def check_case(name: str, rows: int) -> list:
    points = CASES[name](rows)
    assert points and points[-1][0] >= rows // 2, points
    for column, a, b in BOUNDS[name]:
        assert not _within(points, column, a, b), (name, column, points)
    return points


@pytest.mark.parametrize("name", ["float", "temporal"])
def test_the_clocks_weight_reaches_its_steady_state_and_stays(name):
    check_case(name, 100_000)


@pytest.mark.oracle_grid(reason="a million rows a case, against a recomputation from scratch")
@pytest.mark.parametrize("name", sorted(CASES))
def test_a_million_rows_do_not_drift(name):
    check_case(name, 1_000_000)


@pytest.mark.oracle_grid(reason="a window over a million rows, a second or more")
def test_the_window_discards_what_it_should():
    """Evidence the windowed case exercises the subtraction: its weight is the
    window's, well under the unwindowed steady state ``h / ln 2`` of about 72
    rows at a half-life of 50, three half-lives of window keeping 7/8 of it."""
    spec = po.spec.ew_cov(
        "c",
        features=["x0"],
        clock="t",
        gap_cap=1e9,
        half_life=50.0,
        window_size=150.0,
        stats=["mean"],
    )
    chunk = next(lr._stream(20_000, 0, 1, 1e6, None))
    ws = po.ModelBank([spec]).fit_predict(chunk).unnest("c")["weight_sum"].to_numpy()[-1000:]
    assert 0.8 * 72.1 < ws.mean() < 0.9 * 72.1, ws.mean()


@pytest.mark.oracle_grid(reason="a million rows through a Kalman filter, its P read from the state")
def test_kalman_p_stays_positive_definite_and_bounded():
    """``P`` symmetric to the bit, positive definite, and inside a factor of
    two of its value at 10,000 rows at every checkpoint. Measured at 1e8: the
    eigenvalues within [6.4e-5, 7.4e-5] and [2.2e-4, 2.6e-4] from 5,000 rows
    on, the asymmetry exactly 0."""
    check_kalman(1_000_000)


def check_kalman(rows: int) -> None:
    points = lr.kalman_p(rows)
    ref = next(p for p in points if p[0] == 10_000)
    for n, lo, hi, asym in points:
        assert asym <= 1e-12, (n, asym)
        assert lo > 0, (n, lo)
        if n >= 10_000:
            assert 0.5 * ref[1] <= lo <= 2 * ref[1], (n, lo, ref)
            assert 0.5 * ref[2] <= hi <= 2 * ref[2], (n, hi, ref)
