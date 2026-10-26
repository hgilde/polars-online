"""Class 4's full sweep: contamination short of the breakdown point captures
no seed (docs/PLAN.md task 278; the harness is ``tests/convergence.py``).

``tests/test_convergence.py`` runs a representative cell in the essentials
and 50 seeds at 20,000 rows in the extended tier. This sweeps 200 seeds at
50,000 rows, with and without decay, and is deselected from every run but
the weekly leak-check workflow's ``pytest -m soak``. Every default cell
fails today on the defect ``test_convergence.BREAKDOWN_DEFECTS`` states,
and is marked so; the narrow band's 5% cell holds.

A seed is *captured* when its slope ends more than 0.05 from the robust
oracle's on the same rows: task 277's threshold, at which one online Huber
pass was captured on 13 of 40 seeds while the median gap read 0.001.

``quantile`` is the median regression. Its contamination is one-sided and
concentrated where ``x0 > 0`` (``convergence.contaminate``), so a share
``s`` of all rows is ``2 s`` of the rows on that side: the conditional
median there breaks at ``s = 25%``, which docs/MODELS.md states as the
point where every estimator that reads only residuals breaks (30% of the
rows, 60% of those with ``x`` above 0, for ``huber``). The sweep stops
short of it, at 20%. ``huber``'s and ``sgd(loss="huber")``'s sweep, with
the refit of their first rows (task 277), is
``test_convergence.py``'s ``test_huber_is_never_captured_short_of_breakdown``.
"""

from __future__ import annotations

import dataclasses

import pytest

import convergence as cv
from test_convergence import BREAKDOWN_DEFECTS

TIER = "soak"

pytestmark = pytest.mark.soak

#: The shares swept, all short of the 25% at which the conditional median on
#: the contaminated side breaks.
SHARES = (0.05, 0.10, 0.15, 0.20)


ROWS = 50_000
SEEDS = 200


def _params() -> list:
    out = []
    for share in SHARES:
        for h in (cv.INF, 5000.0):
            marks = []
            if (share, h) in BREAKDOWN_DEFECTS:
                marks.append(pytest.mark.xfail(strict=True, reason=BREAKDOWN_DEFECTS[(share, h)]))
            out.append(pytest.param(share, h, marks=marks, id=f"{share:.0%}-{h:g}"))
    return out


@pytest.mark.parametrize("share, half_life", _params())
def test_quantile_is_never_captured_short_of_breakdown(share, half_life):
    reg = cv.REGISTRY["quantile"]
    captured, t, wild = cv.captures(reg, share, SEEDS, ROWS, half_life)
    print(f"quantile, {share:.0%} wild, half-life {half_life:g}: {captured}/{SEEDS}; {t}")
    assert wild > 0.5 * share * ROWS, "the contamination is there"
    assert captured == 0, t


def test_a_narrow_band_is_never_captured_at_five_percent():
    narrow = dataclasses.replace(
        cv.REGISTRY["quantile"],
        build=cv._spec("quantile", quantile=0.5, solve_every=0.0, quantile_eps=0.02),
    )
    captured, t, wild = cv.captures(narrow, 0.05, SEEDS, ROWS, cv.INF)
    print(f"quantile, quantile_eps 0.02, 5% wild: {captured}/{SEEDS}; {t}")
    assert wild > 0.025 * ROWS
    assert captured == 0, t
