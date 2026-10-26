"""The bank's residual autocorrelation weighs each residual by its row's
weight, as ``sigma``, the residual quantiles and the metrics do
(docs/PLAN.md task 278, batch 17). It gave every residual a weight of 1, so
under uneven weights it described another series than the spread beside it.

The reference is the definition written out here, from the update equations
(``EwAutoCorr::update_weighted``): with ``lam = 2^(-d/h)`` the row's decay,
a row with a residual and a positive weight ``w`` folds it,

    W' = lam W + w,  a = lam W / W',  b = w / W',  d = r - mean
    var' = a var + a b d^2,  mean' = mean + b d
    (a pair, once the residual ``lag`` back exists)
    Wc' = lam Wc + w,  cross' = (lam Wc / Wc') cross + a (w / Wc') d (r_lag - mean)

and every other row only ages ``W`` and ``Wc``; the field on a row is read
before the row's own residual (``cross / var``, clamped to [-1, 1], once a
pair has weight and the variance is positive). The stream's residual series
alternates two regimes in blocks of 150 rows: weight 1 at ``phi = +0.8``, and
weight 0.02 at ``phi = -0.8``, with a zero-weight row now and then. Weighed,
the autocorrelation is the heavy regime's; at a weight of 1 a residual, the
old rule, it sits near 0.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

import polars_online as po

TIER = "essential"

N = 3000
HALF_LIFE = 400.0


def _frame() -> pl.DataFrame:
    r = np.random.default_rng(278_170)
    x = r.normal(size=(N, 2))
    heavy = (np.arange(N) // 150) % 2 == 0
    e = np.zeros(N)
    for t in range(1, N):
        e[t] = (0.8 if heavy[t] else -0.8) * e[t - 1] + r.normal(scale=0.5)
    w = np.where(heavy, 1.0, 0.02)
    w[r.random(N) < 0.03] = 0.0
    y = x @ np.array([1.0, -0.5]) + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


def _hand(resid: list[float | None], weights: list[float], *, weighed: bool) -> list[float | None]:
    """The field as the update equations above give it, row by row, at each
    row's weight (``weighed``) or at 1 (the old rule)."""
    big_w = mean = var = w_pairs = cross = 0.0
    previous: float | None = None
    lam = 2.0 ** (-1.0 / HALF_LIFE)
    out: list[float | None] = []
    for t, (r, w) in enumerate(zip(resid, weights, strict=True)):
        out.append(max(-1.0, min(1.0, cross / var)) if w_pairs > 0.0 and var > 0.0 else None)
        step = 1.0 if t == 0 else lam  # the first row has no clock step
        if r is None or not math.isfinite(r) or w <= 0.0:
            big_w *= step
            w_pairs *= step
            continue
        v = w if weighed else 1.0
        w_new = step * big_w + v
        a, b = step * big_w / w_new, v / w_new
        d = r - mean
        var = a * var + a * b * d * d
        if previous is not None:
            wp_new = step * w_pairs + v
            cross = (step * w_pairs / wp_new) * cross + a * (v / wp_new) * d * (previous - mean)
            w_pairs = wp_new
        else:
            w_pairs *= step
        mean += b * d
        big_w = w_new
        previous = r
    return out


def _fit() -> pl.DataFrame:
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALF_LIFE,
        weight="w",
        emit_autocorr=True,
        resid_autocorr_lag=1,
    )
    return po.ModelBank([spec], threads=1).fit_predict(_frame())["m"].struct.unnest()


def test_the_residual_autocorrelation_weighs_each_residual_by_its_row():
    out = _fit()
    resid, got = out["resid_y"].to_list(), out["autocorr_y"].to_list()
    weights = _frame()["w"].to_list()
    want, old = _hand(resid, weights, weighed=True), _hand(resid, weights, weighed=False)
    assert [g is None for g in got] == [h is None for h in want], "nulls on other rows"
    gap = max(abs(g - h) for g, h in zip(got, want, strict=True) if g is not None and h is not None)
    print(f"largest gap: {gap:.1e}; at the end {got[-1]:.3f}, the old rule's {old[-1]:.3f}")
    # The compensated mean the bank keeps and the plain one here part by a
    # few ulps a row; measured 2026-10-26 at 3.3e-16.
    assert gap < 1e-10, gap
    # What moved: the heavy regime's +0.8 against the old rule's mix.
    assert got[-1] is not None and old[-1] is not None
    assert got[-1] > 0.5 and abs(old[-1]) < 0.3, (got[-1], old[-1])


def test_at_a_weight_of_one_the_residual_autocorrelation_is_the_unweighed_one():
    """With every row at weight 1 the two rules are one, to the bit: the
    rows that move are only those an uneven weight reaches."""
    df = _frame().with_columns(w=pl.lit(1.0))
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALF_LIFE,
        weight="w",
        emit_autocorr=True,
    )
    plain = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], half_life=HALF_LIFE, emit_autocorr=True
    )
    a = po.ModelBank([spec], threads=1).fit_predict(df)["m"].struct.field("autocorr_y")
    b = po.ModelBank([plain], threads=1).fit_predict(df)["m"].struct.field("autocorr_y")
    assert a.is_not_null().sum() > N // 2
    assert a.equals(b, null_equal=True) and (a.fill_null(0.0) == b.fill_null(0.0)).all()
