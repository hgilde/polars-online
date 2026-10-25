"""Task 278 (c), class 9: long-run stability at 1e8 rows, weekly.

The cases and bounds of `tests/test_long_run.py`, a hundred times longer:
1e8 rows a case, about ten minutes in all on a developer Mac (2026-10-25:
NumAcc4 34 s, the decayed and windowed moments 13 s and 77 s, the ridge
without and with decay 227 s and 134 s, the Kalman filter 42 s, each clock
18-20 s). Deselected everywhere else by pyproject's addopts and run by the
weekly leak-check workflow, `uv run pytest -m soak`, as `tests/test_soak.py`
is.
"""

from __future__ import annotations

import pytest

import long_run as lr
from public_benchmarks import numacc_or_skip
from test_long_run import CASES, EPS, _within, check_case, check_kalman

TIER = "soak"

pytestmark = pytest.mark.soak

ROWS = 100_000_000


def test_numacc4_cycled_to_1e8_rows_does_not_drift():
    x, _, _ = numacc_or_skip(4)
    points = lr.numacc_cycled(x, ROWS)
    assert points[-1][0] >= ROWS - len(x)
    assert not _within(points, 1, 4 * EPS, 0.0), points
    assert not _within(points, 2, 16 * EPS, 1.0), points


@pytest.mark.parametrize("name", sorted(CASES))
def test_1e8_rows_do_not_drift(name):
    check_case(name, ROWS)


def test_kalman_p_stays_positive_definite_and_bounded_over_1e8_rows():
    check_kalman(ROWS)
