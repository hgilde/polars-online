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


class TestTheWindowIsAllTheModelSees:
    """Pattern D and C2. Under a ``window`` every model reports, gates and fits
    on the weight *inside* it -- ``Σ 0.5 ** (age / halflife)`` over the rows
    the window keeps, which ``numpy`` sums from the rows. Before the fixes,
    ``lasso`` (C9), ``ew_class`` (S14) and ``marginal`` (S17) reported the
    whole history's weight while fitting on the window; ``ewridge`` and
    ``ew_cov`` already reported the window's, and are the controls.

    C2: ``ewridge`` and ``lasso`` built their windowed view all-or-nothing,
    so a target with no rows left inside the window -- a sparse label beside a
    dense one -- made the whole view fall back to the live accumulators, and
    every *other* target was then solved on the whole history. Now the empty
    target reports nothing and the rest stay windowed."""

    HALFLIFE, WINDOW = 25.0, 60.0

    def _rows(self, n: int = 320):
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 1.0, (n, 2)) + 3.0
        y0 = np.where(
            np.arange(n) < 200,
            1.0 + 2.0 * x[:, 0] - x[:, 1],
            -1.0 + 0.5 * x[:, 0] + 1.5 * x[:, 1],
        ) + rng.normal(0.0, 0.1, n)
        y1 = 0.3 * x[:, 0] + rng.normal(0.0, 0.1, n)
        labels = np.where(x[:, 0] + rng.normal(0.0, 0.5, n) > 3.0, "a", "b")
        return x, y0, y1, labels

    def _spec(self, kind: str, window: float | None):
        common = {"halflife": self.HALFLIFE, "window": window}
        feats = ["x0", "x1"]
        if kind == "ewridge":
            return po.spec.ewridge("m", targets=["y0"], features=feats, ridge=1e-10, **common)
        if kind == "lasso":
            return po.spec.lasso("m", targets=["y0"], features=feats, lasso_path=[0.0], **common)
        if kind == "marginal":
            return po.spec.marginal("m", targets=["y0"], features=feats, **common)
        if kind == "ew_class":
            return po.spec.ew_class(
                "m", features=feats, label="c", classes=["a", "b"], precision_prior=1e-6, **common
            )
        return po.spec.ew_cov("m", features=feats, stats=["mean"], **common)

    @pytest.mark.parametrize("window", [None, 60.0])
    @pytest.mark.parametrize("kind", ["ewridge", "lasso", "marginal", "ew_class", "ew_cov"])
    def test_n_eff_is_the_weight_inside_the_window(self, kind, window):
        x, y0, _, labels = self._rows()
        n = len(y0)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y0, "c": labels})
        n_eff = po.ModelBank([self._spec(kind, window)]).fit_predict(df)["m"].struct.field("n_eff")
        for i in range(100, n, 13):
            # The n_eff a row reports is the weight before it, with ages
            # counted from the row before.
            _, w = _window((i - 1) - np.arange(i), self.HALFLIFE, window)
            assert n_eff[i] == pytest.approx(w.sum(), rel=1e-10), (kind, i)

    @pytest.mark.parametrize("kind", ["ewridge", "lasso"])
    def test_a_target_that_leaves_the_window_does_not_unwindow_the_others(self, kind):
        x, y0, y1, _ = self._rows()
        n = len(y0)
        # y1 is observed on the first 100 rows only; its last row is out of the
        # window from row 161 on, and from then on it has nothing to fit.
        y1_sparse = [float(v) if i < 100 else None for i, v in enumerate(y1)]
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y0, "y1": y1_sparse})
        feats, common = ["x0", "x1"], {"halflife": self.HALFLIFE, "window": self.WINDOW}
        if kind == "ewridge":
            spec = po.spec.ewridge(
                "m", targets=["y0", "y1"], features=feats, ridge=1e-10, solve_every=1e-9, **common
            )
            f0, f1, tol = "pred_y0", "pred_y1", 1e-8
        else:
            spec = po.spec.lasso(
                "m",
                targets=["y0", "y1"],
                features=feats,
                lasso_path=[0.0],
                solve_every=1e-9,
                **common,
            )
            f0, f1, tol = "pred_y0__l0", "pred_y1__l0", 1e-6
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        for i in range(170, n, 7):
            keep, w = _window((i - 1) - np.arange(i), self.HALFLIFE, self.WINDOW)
            b = _wls(x[:i][keep], y0[:i][keep], w)
            assert out[f0][i] == pytest.approx(b[0] + x[i] @ b[1:], abs=tol), (kind, i)
            # y1 has no row inside the window: it reports nothing, rather than
            # a fit from rows the window excludes.
            assert out[f1][i] is None or np.isnan(out[f1][i]), (kind, i)


def _regime_rows(n: int, seed: int) -> np.ndarray:
    """Two features whose covariance changes at row ``n // 2`` -- a wide,
    tilted cloud, then a narrow one -- so the whole history's covariance and
    a window's disagree, which is what a live read under a window shows up
    as."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 2))
    early = np.arange(n) < n // 2
    x[early] = x[early] @ np.array([[3.0, 1.2], [0.0, 1.5]])
    return x


class TestWindowedGaussian:
    """C12 and C14, the review's T-S6 and T-S8. At ``halflife = inf`` every row
    inside the window weighs 1 and the precision prior does not fade, so a
    windowed ``ew_class`` and ``ew_cov`` are plain Gaussian computations on the
    rows the window keeps: ``scipy.stats.multivariate_normal`` for the class
    densities, ``scipy.spatial.distance.mahalanobis`` for the distance, and
    ``numpy.linalg.eigh`` for the components -- each fed the in-window mean
    and population covariance from ``numpy``.

    C12: the ``full`` shape factorized the *live* class covariance while
    scoring against the windowed mean. C14: ``mahal`` and the PCA refresh read
    the live accumulator. ``diagonal`` and ``shared`` (which read the view)
    and ``window=None`` are the controls. The covariance changes halfway, so
    the whole history and the window disagree by construction."""

    WINDOW, PRIOR = 80.0, 1e-9

    @pytest.mark.parametrize("window", [None, 80.0])
    @pytest.mark.parametrize("shape", ["full", "diagonal", "shared"])
    def test_ew_class_is_the_scipy_gaussian_of_the_rows_inside_the_window(self, shape, window):
        from scipy.stats import multivariate_normal

        n = 400
        x = _regime_rows(n, seed=21)
        labels = np.where(x[:, 0] + 0.5 * x[:, 1] > 0.0, "a", "b")
        spec = po.spec.ew_class(
            "m",
            features=["x0", "x1"],
            label="c",
            classes=["a", "b"],
            covariance=shape,
            precision_prior=self.PRIOR,
            halflife=float("inf"),
            window=window,
            min_periods=0,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "c": labels})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        for i in range(250, n, 17):
            keep, _ = _window((i - 1) - np.arange(i), float("inf"), window)
            rows, lab = x[:i][keep], labels[:i][keep]
            means, covs, priors = [], [], []
            for c in ("a", "b"):
                rc = rows[lab == c]
                means.append(rc.mean(axis=0))
                covs.append(np.cov(rc.T, ddof=0) + self.PRIOR * np.eye(2))
                priors.append(len(rc) / len(rows))
            if shape == "diagonal":
                covs = [np.diag(np.diag(cv)) for cv in covs]
            elif shape == "shared":
                pooled = priors[0] * covs[0] + priors[1] * covs[1]
                covs = [pooled, pooled]
            logp = np.array(
                [
                    np.log(priors[c]) + multivariate_normal(means[c], covs[c]).logpdf(x[i])
                    for c in range(2)
                ]
            )
            p = np.exp(logp - logp.max())
            p /= p.sum()
            assert out["p_a"][i] == pytest.approx(p[0], abs=1e-9), (shape, i)

    @pytest.mark.parametrize("window", [None, 80.0])
    def test_ew_cov_mahal_and_components_are_scipy_and_numpy_of_the_window(self, window):
        from scipy.spatial.distance import mahalanobis

        n = 400
        x = _regime_rows(n, seed=22)
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=["mean", "mahal"],
            precision_prior=self.PRIOR,
            pca=1,
            pca_every=1,
            halflife=float("inf"),
            window=window,
            min_periods=0,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]}))
        out = out["m"].struct.unnest()
        for i in range(250, n, 17):
            keep, _ = _window((i - 1) - np.arange(i), float("inf"), window)
            rows = x[:i][keep]
            mean, cov = rows.mean(axis=0), np.cov(rows.T, ddof=0)
            want = mahalanobis(x[i], mean, np.linalg.inv(cov + self.PRIOR * np.eye(2)))
            assert out["mahal"][i] == pytest.approx(want, rel=1e-9), i
            # The components a row is read with were refreshed after the row
            # before it, from the same window.
            assert out["pc0_var"][i] == pytest.approx(np.linalg.eigh(cov)[0][-1], rel=1e-9), i


class TestMahalQuantiles:
    """C15: ``mahal_quantiles`` found the ``mahal`` slot with arithmetic of its
    own, which counted ``lagcorr`` as ``k(k-1)/2`` slots instead of
    ``len(lags)·k²`` -- so beside a ``lagcorr`` the quantile estimators were
    fed a lagged correlation. The reference is ``numpy.quantile`` of the
    ``mahal`` column the bank itself emitted. Statistical tier: the P²
    estimator is approximate, so 10% -- the defect puts it at a quantile of a
    correlation, in ``[-1, 1]``, against a distance near 1.2."""

    @pytest.mark.parametrize("stats", [["mahal"], ["lagcorr", "mahal"]])
    def test_the_quantile_tracks_the_mahal_column(self, stats):
        rng = np.random.default_rng(23)
        n = 3000
        x = rng.normal(0.0, 1.0, (n, 2))
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=stats,
            lags=[1] if "lagcorr" in stats else None,
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.9],
            halflife=float("inf"),
            min_periods=5,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]}))
        out = out["m"].struct.unnest()
        mahal = out["mahal"].drop_nulls().drop_nans().to_numpy()
        assert out["mahal_q0.5"][-1] == pytest.approx(np.quantile(mahal[:-1], 0.5), rel=0.1)
        assert out["mahal_q0.9"][-1] == pytest.approx(np.quantile(mahal[:-1], 0.9), rel=0.1)
