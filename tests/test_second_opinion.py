"""Second opinions from independent libraries (code review 2026-09-12).

``tests/test_river.py`` holds the models to river where the two implement
the same recursion. This file does the same for the fixes that came out of
the code review of 2026-09-12 (``docs/REVIEW-2026-09-12.md``; what has been
done about each finding is ``docs/REVIEW-2026-09-12-PROGRESS.md``): every
test compares a number the bank reports with the same number computed from
the raw rows by numpy, scipy or river, independently of this library's own
arithmetic. Two tiers, as there: **exact**, where the library computes the
same quantity (a weighted mean, a two-pass covariance, a least-squares
solve), and **statistical**, where it differs by design and agrees only in
the limit.

The windows here run on a row-count clock, so a row's age is the number of
rows between it and the reference row. A window keeps the rows whose age is
**at most** ``window`` -- a row exactly ``window`` old is not older than the
window, which is how ``crates/online-core/src/window.rs`` states the
boundary -- each at weight ``0.5 ** (age / halflife)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _window(
    ages: np.ndarray, halflife: float, window: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """The rows a window keeps, and their weights."""
    keep = np.ones(ages.shape, bool) if window is None else ages <= window
    return keep, 0.5 ** (ages[keep] / halflife)


class TestWindowAtALargeOffset:
    """C17, the review's T-S10 (i). A window's moments are a weighted mean and
    a population covariance of the rows inside it, which ``numpy.average`` and
    ``numpy.cov(aweights=..., ddof=0)`` compute two-pass -- centre, then square
    -- and so correctly at any offset. The truncation used to go back through
    ``E[x x'] = C + m m'``, subtract and re-centre, which at a level of ``1e8``
    left the variance with a resolution of about 2. The unwindowed
    accumulator (Welford) never had the defect, which makes ``window=None``
    the negative control in every test here.

    Spreads are unit-scale, so at ``1e8`` an absolute error of ``1e-6``
    separates the fix (near ``1e-9``, the live accumulator's own error at that
    level) from the defect (errors of order 1)."""

    HALFLIFE = 40.0

    @pytest.mark.parametrize("window", [None, 25.0, 60.0])
    @pytest.mark.parametrize("offset", [0.0, 1e8])
    def test_ew_cov_is_numpy_cov_of_the_rows_inside_the_window(self, offset, window):
        rng = np.random.default_rng(7)
        n = 300
        x = rng.normal(0.0, 1.0, (n, 2))
        x[:, 1] += 0.5 * x[:, 0]
        x += offset
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=["mean", "var", "cov"],
            halflife=self.HALFLIFE,
            min_periods=0,
            window=window,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        tol = 1e-12 if offset == 0.0 else 1e-6
        for i in range(100, n, 11):
            # The report on row i is the state after the rows before it, with
            # ages counted from row i - 1: the statistic is read before the
            # row is learned.
            keep, w = _window((i - 1) - np.arange(i), self.HALFLIFE, window)
            rows = x[:i][keep]
            mean = np.average(rows, axis=0, weights=w)
            cov = np.cov(rows.T, aweights=w, ddof=0)
            got_cov = np.array(
                [
                    [out["var_x0"][i], out["cov_x0_x1"][i]],
                    [out["cov_x0_x1"][i], out["var_x1"][i]],
                ]
            )
            assert out["n_eff"][i] == pytest.approx(w.sum(), rel=1e-10)
            np.testing.assert_allclose(
                [out["mean_x0"][i], out["mean_x1"][i]], mean, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(got_cov, cov, rtol=0.0, atol=tol * np.abs(cov).max())

    @pytest.mark.parametrize("window", [None, 70.0])
    @pytest.mark.parametrize("offset", [0.0, 1e8])
    def test_a_marginal_pair_is_numpy_of_the_rows_inside_the_window(self, offset, window):
        rng = np.random.default_rng(11)
        n, halflife = 400, 25.0
        u = rng.normal(0.0, 1.0, n)
        x = offset + u
        y = offset + 2.0 * u + rng.normal(0.0, 0.2, n)
        spec = po.spec.marginal(
            "m", features=["x"], targets=["y"], halflife=halflife, window=window
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        pair = bank.marginal("m").row(0, named=True)
        # A readout is referenced at the last learned row, which is inside.
        keep, w = _window((n - 1) - np.arange(n), halflife, window)
        xs, ys = x[keep], y[keep]
        cov = np.cov(np.vstack([xs, ys]), aweights=w, ddof=0)
        tol = (1e-12 if offset == 0.0 else 1e-6) * np.abs(cov).max()
        assert pair["n_eff"] == pytest.approx(w.sum(), rel=1e-10)
        assert pair["mean_x"] == pytest.approx(np.average(xs, weights=w), rel=1e-12, abs=1e-12)
        assert pair["mean_y"] == pytest.approx(np.average(ys, weights=w), rel=1e-12, abs=1e-12)
        assert abs(pair["var_x"] - cov[0, 0]) <= tol
        assert abs(pair["var_y"] - cov[1, 1]) <= tol
        assert abs(pair["cov"] - cov[0, 1]) <= tol


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray, intercept: bool = True) -> np.ndarray:
    """Weighted least squares by ``numpy.linalg.lstsq`` on rows scaled by
    ``sqrt(w)`` -- the fit a mean-form accumulator solves, computed from the
    rows. With an intercept the first coefficient is it."""
    z = np.column_stack([np.ones(len(y)), x]) if intercept else x
    sw = np.sqrt(w)
    return np.linalg.lstsq(z * sw[:, None], y * sw, rcond=None)[0]


class TestWindowedFit:
    """C1: a windowed ``ewridge`` is the weighted least-squares fit of the rows
    inside the window, each at ``0.5 ** (age / halflife)`` -- ``numpy``'s
    ``lstsq`` on those rows is the second opinion. The standardized solve with
    an intercept read its centred Gram and means from the live accumulator
    while its right-hand side came from the window, so the fit mixed two
    histories; the slope changes at row 250 so the two disagree. The other
    three combinations never had the defect and are the controls. The
    penalty is ``1e-10`` on the standardized scale, far below the tolerance,
    so ``lstsq``'s unpenalized fit is the reference."""

    @pytest.mark.parametrize("window", [None, 60.0])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_pred_is_the_numpy_fit_of_the_rows_inside_the_window(self, standardize, window):
        rng = np.random.default_rng(3)
        n, halflife = 400, 25.0
        # Offset so that centring (and so the intercept branch) matters.
        x = rng.normal(0.0, 1.0, (n, 2)) + 3.0
        y = np.where(
            np.arange(n) < 250,
            1.0 + 2.0 * x[:, 0] - x[:, 1],
            -1.0 + 0.5 * x[:, 0] + 1.5 * x[:, 1],
        ) + rng.normal(0.0, 0.1, n)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=halflife,
            ridge=1e-10,
            standardize=standardize,
            window=window,
            solve_every=1e-9,  # solve on every row, so every pred is comparable
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        for i in range(260, n, 9):
            # Out of sample: the fit on the rows before row i predicts row i.
            keep, w = _window((i - 1) - np.arange(i), halflife, window)
            b = _wls(x[:i][keep], y[:i][keep], w)
            assert pred[i] == pytest.approx(b[0] + x[i] @ b[1:], abs=1e-8)
