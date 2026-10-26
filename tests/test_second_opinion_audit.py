"""An ``audit``'s pairs against numpy (task 278, batch 29).

``pairs=True`` keeps, for every pair of columns, the correlation and the
count of equal values over the rows where *both* are usable: not null, not
NaN, not infinite and no larger than ``1e100`` in magnitude. numpy computes
both from the raw rows once those are picked out: ``np.corrcoef`` and an
elementwise ``==`` (which reads ``-0.0 == 0.0``). Each column below is
unusable on its own rows, of every kind, so a pair's rows are an
intersection no single column's count gives; and the groups and their
pooled merge are held to the same computation on their rows.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

N = 3000


def _frame(seed: int = 7) -> pl.DataFrame:
    """``b`` correlated with ``a``; ``c`` is ``a`` rounded to a tenth, so it
    repeats; ``d`` equals ``c`` on about 70% of the rows and is ``-0.0``
    wherever ``c`` is ``0.0``. Then ``a`` null every 11th row, ``b`` NaN,
    ``+inf`` and ``1e200`` on their own rows, ``c`` ``-inf`` every 17th; two
    interleaved groups."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=N)
    b = 0.6 * a + 0.8 * rng.normal(size=N)
    c = np.round(a, 1)
    d = c.copy()
    d[rng.random(N) < 0.3] += 0.5
    d[c == 0.0] = -0.0
    i = pl.int_range(pl.len())
    return pl.DataFrame({"a": a, "b": b, "c": c, "d": d, "g": np.arange(N) % 2}).with_columns(
        a=pl.when(i % 11 == 0).then(None).otherwise(pl.col("a")),
        b=pl.when(i % 13 == 1)
        .then(float("nan"))
        .when(i % 101 == 5)
        .then(float("inf"))
        .when(i % 97 == 3)
        .then(1e200)
        .otherwise(pl.col("b")),
        c=pl.when(i % 17 == 2).then(-float("inf")).otherwise(pl.col("c")),
    )


def _usable(s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    v = s.to_numpy().astype(float)
    return np.isfinite(v) & (np.abs(v) <= 1e100), v


def _check(pairs: pl.DataFrame, frame: pl.DataFrame) -> int:
    """Each pair's ``count``, ``corr`` and ``equal`` against numpy on the
    rows both columns hold; returns how many rows of equal values the
    ``-0.0`` / ``0.0`` rows contributed, so a caller sees they were read."""
    signed_zeros = 0
    for row in pairs.iter_rows(named=True):
        ua, va = _usable(frame[row["column_a"]])
        ub, vb = _usable(frame[row["column_b"]])
        both = ua & ub
        assert row["count"] == int(both.sum()), row
        want = np.corrcoef(va[both], vb[both])[0, 1]
        # Measured 2026-10-25: at most 2.2e-15 apart.
        assert row["corr"] == pytest.approx(want, rel=1e-13, abs=1e-15), row
        assert row["equal"] == int((va[both] == vb[both]).sum()), row
        signed_zeros += int(
            (
                (va[both] == 0.0)
                & (vb[both] == 0.0)
                & (np.signbit(va[both]) != np.signbit(vb[both]))
            ).sum()
        )
    return signed_zeros


def test_the_pairs_are_numpys_over_the_rows_both_columns_hold():
    frame = _frame()
    bank = po.ModelBank([po.spec.audit("au", columns=["a", "b", "c", "d"], pairs=True)])
    bank.fit_predict(frame)
    pairs = bank.audit(table="pairs")
    assert pairs.height == 6
    # The intersections differ from pair to pair, so no column's own count
    # stands in for one.
    assert pairs["count"].n_unique() == 6
    # Measured 2026-10-25: 52 rows of 0.0 beside -0.0 compared, all equal.
    assert _check(pairs, frame) > 20, "no -0.0 / 0.0 row was compared"
    cd = pairs.filter(pl.col("column_a") == "c", pl.col("column_b") == "d").row(0, named=True)
    assert cd["equal"] > 1500


def test_each_groups_pairs_and_the_pooled_merge_are_numpys():
    frame = _frame(seed=8)
    spec = po.spec.audit("au", columns=["a", "b", "c", "d"], pairs=True, group="g")
    bank = po.ModelBank([spec])
    for part in frame.iter_slices(700):
        bank.fit_predict(part)
    pairs = bank.audit(table="pairs")
    for g in (0, 1):
        _check(pairs.filter(pl.col("group") == str(g)), frame.filter(pl.col("g") == g))
    pooled = bank.audit(table="pairs", pooled=True)
    assert pooled.height == 6
    _check(pooled, frame)
