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

import functools
from typing import Any

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


class TestWindowedSpread:
    """S1: under a ``window`` the bank's ``sigma`` and ``resid_z`` are the
    window's -- the EW root mean square of the out-of-sample residuals
    inside it, ``sqrt(Σ λ^age r² / Σ λ^age)``, over the rows the fit is read
    from. They were the stream's own EW mean over the whole history, so a
    burst of errors the window had dropped still widened ``sigma`` for as
    long as the halflife remembered it -- the spread of a history the fit no
    longer sees. ``numpy`` over the residuals the bank itself emitted is the
    second opinion (review 2026-09-12, S1; the user's decision of
    2026-09-15)."""

    H, W = 40.0, 60.0

    def _frame(self, n=500, seed=21):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.1, n)
        # A burst of large errors: the window drops it at row 320 + W, long
        # before a halflife of 40 forgets it.
        y[300:320] += rng.normal(0.0, 5.0, 20)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})

    def _spec(self, kind, **kw):
        common = dict(
            targets=["y"],
            features=["x0", "x1"],
            halflife=self.H,
            window=self.W,
            solve_every=1e-9,
            emit_sigma=True,
            emit_resid_z=True,
            **kw,
        )
        if kind == "lasso":
            return po.spec.lasso("m", lasso_path=[0.01], **common)
        return po.spec.ewridge("m", **common)

    @staticmethod
    def _field(out, spec, prefix):
        name = next(n for n in po.spec.output_fields(spec) if n.startswith(prefix))
        return out["m"].struct.field(name).to_numpy().astype(float)

    @pytest.mark.parametrize("kind", ["ewridge", "lasso"])
    def test_sigma_is_the_spread_of_the_residuals_inside_the_window(self, kind):
        df = self._frame()
        spec = self._spec(kind)
        out = po.ModelBank([spec]).fit_predict(df)
        resid = self._field(out, spec, "resid_y")
        sigma = self._field(out, spec, "sigma_y")
        z = self._field(out, spec, "resid_z_y")
        ok = np.isfinite(resid)
        checked = 0
        for i in range(1, len(resid)):
            # Row i is scored against the rows learned before it, inside the
            # window as it stood at row i - 1: the fit's own boundary.
            ages = (i - 1) - np.arange(i)
            keep = (ages <= self.W) & ok[:i]
            if not keep.any():
                assert not np.isfinite(sigma[i]), i
                continue
            w = 0.5 ** (ages[keep] / self.H)
            want = np.sqrt(np.sum(w * resid[:i][keep] ** 2) / np.sum(w))
            assert sigma[i] == pytest.approx(want, rel=1e-9), (kind, i)
            if ok[i]:
                assert z[i] == pytest.approx(resid[i] / want, rel=1e-9), (kind, i)
            checked += 1
        assert checked > 400
        # The review's reading: once the burst is older than the window, the
        # spread is back where it was before it -- from row 320 + 2W, since
        # the fit keeps the burst until its own window drops it at 380, and
        # the residuals of the fit it bent stay in the spread's window one
        # window more.
        assert np.mean(sigma[460:]) < 1.5 * np.mean(sigma[200:300])

    def test_the_window_survives_chunks_a_save_and_scoring(self, tmp_path):
        # The spread's ring is state like the fit's: chunked, saved and
        # scored, it gives the numbers one pass gives -- every field but
        # `coef`, which each chunk's last row reports. E31: scoring a row
        # against the bank fitted to the row before is that row's
        # fit_predict value.
        df = self._frame()
        spec = self._spec("ewridge")
        cols = ["pred_y", "resid_y", "sigma_y", "resid_z_y", "n_eff"]
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df[:97]), bank.fit_predict(df[97:180])]
        path = tmp_path / "bank.bin"
        bank.save(path)
        parts.append(po.ModelBank.load(path, specs=[spec]).fit_predict(df[180:]))
        got = pl.concat(parts)["m"].struct.unnest().select(cols)
        assert got.equals(whole["m"].struct.unnest().select(cols), null_equal=True)
        scorer = po.ModelBank([spec])
        scorer.fit_predict(df[:250])
        scored = scorer.predict(df[250:251])["m"].struct.field("sigma_y")[0]
        assert scored == whole["m"].struct.field("sigma_y")[250]

    def test_the_spread_s_ring_is_held_to_the_window_budget(self):
        """The spread's snapshots are a ring like the fit's, a pair of floats
        a slot, so a grid of many slots over one feature outgrows the fit's
        own ring: the window's budget bounds it too (review 2026-09-12, P4).
        Two hundred ridges over a window of 2000 rows is about 6 MB of
        spread, where the fit's ring holds a few hundred KB."""
        rng = np.random.default_rng(4)
        n = 3000
        x = rng.normal(0.0, 1.0, n)
        df = pl.DataFrame({"x0": x, "y": x + rng.normal(0.0, 0.1, n)})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            halflife=500.0,
            window=2000.0,
            ridge=[10.0 ** (-k / 20) for k in range(200)],
            emit_sigma=True,
            window_budget={"refuse": 1.0},
        )
        with pytest.raises(ValueError, match="window_budget"):
            po.ModelBank([spec]).fit_predict(df)


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


class TestSessionShrinkBlend:
    """C16 and C3, the review's T-S10 (ii). A ``session_shrink`` blend at ``f``
    mixes the fast accumulators (``halflife``) with their slow twin
    (``long_halflife``), which saw the *same* rows -- so the blend is itself
    one weighted accumulator, each row at ``(1 - f)·λ_h^age + f·λ_H^age`` (the
    mean-form definitions cancel the weights). ``numpy.average`` and
    ``numpy.cov(aweights=..., ddof=0)`` with that weight vector are the
    blended moments exactly, two-pass and so right at any offset.

    C16: both blends went back through raw second moments, which at a level of
    ``1e8`` leave nothing of a unit variance; ``offset = 0`` is the control.
    C3: the blend never re-solved, so the first rows of a new session were
    predicted with the coefficients from before it -- the rows the feature is
    for. At ``f = 1`` the blended state *is* the slow twin, whose fit is the
    weighted least squares of every row at ``λ_H^age``."""

    H_FAST, H_SLOW = 40.0, 400.0

    @pytest.mark.parametrize("offset", [0.0, 1e8])
    def test_the_blended_moments_are_the_rows_at_the_blended_weights(self, offset):
        rng = np.random.default_rng(31)
        n, f = 600, 0.5
        u = rng.normal(0.0, 1.0, (n, 2))
        x = offset + u
        y = offset + 2.0 * u[:, 0] - u[:, 1] + rng.normal(0.0, 0.1, n)
        # One session, then a second that starts on the last row: the blend
        # runs as that row arrives, and the row is learned after it.
        session = np.where(np.arange(n) < n - 1, 1, 2)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=self.H_FAST,
            session="s",
            session_gap=1.0,
            session_shrink=f,
            long_halflife=self.H_SLOW,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "s": session}))
        g = bank.gram("m")[0]
        # Rows before the last: blended at their ages, then aged one step at
        # the fast rate by the last row's `session_gap`. The last row: weight 1.
        age = (n - 2) - np.arange(n - 1)
        lam_f, lam_s = 0.5 ** (1.0 / self.H_FAST), 0.5 ** (1.0 / self.H_SLOW)
        w = np.append(lam_f * ((1.0 - f) * lam_f**age + f * lam_s**age), 1.0)
        cols = [g["columns"].index("x0"), g["columns"].index("x1")]
        cov = np.cov(x.T, aweights=w, ddof=0)
        got = np.asarray(g["comoments"])[np.ix_(cols, cols)]
        tol = (1e-12 if offset == 0.0 else 1e-6) * np.abs(cov).max()
        assert g["n_eff"] == pytest.approx(w.sum(), rel=1e-12)
        np.testing.assert_allclose(
            np.asarray(g["means"])[cols], np.average(x, axis=0, weights=w), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(got, cov, rtol=0.0, atol=tol)
        ybar = np.average(y, weights=w)
        yvar = np.average((y - ybar) ** 2, weights=w)
        assert g["target_means"][0] == pytest.approx(ybar, rel=1e-12, abs=1e-12)
        assert abs(g["target_vars"][0] - yvar) <= (1e-12 if offset == 0.0 else 1e-6) * yvar

    @pytest.mark.parametrize("long_halflife", [2000.0, float("inf")])
    def test_the_first_row_of_a_session_is_predicted_from_the_blend(self, long_halflife):
        rng = np.random.default_rng(32)
        n1, n2 = 4300, 10
        n = n1 + n2
        x = rng.normal(0.0, 1.0, n)
        slope = np.where(np.arange(n) < 4000, 1.0, -1.0)
        y = 0.5 + slope * x + rng.normal(0.0, 0.05, n)
        session = np.where(np.arange(n) < n1, 1, 2)
        df = pl.DataFrame({"x": x, "y": y, "s": session})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            ridge=1e-10,
            halflife=100.0,
            session="s",
            session_gap=1.0,
            session_shrink=1.0,  # the blend is the slow twin itself
            long_halflife=long_halflife,
        )
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y")
        # The slow twin's fit: every row of session 1 at 0.5 ** (age / H). At
        # H = inf the weights are numpy's unit ones: the long run is the whole
        # history, which the builder refused (review 2026-09-12, S27).
        age = (n1 - 1) - np.arange(n1)
        b = _wls(x[:n1, None], y[:n1], 0.5 ** (age / long_halflife))
        assert pred[n1] == pytest.approx(b[0] + b[1] * x[n1], abs=1e-8)
        # E31: scoring that row against the bank fitted to the end of session 1
        # gives the same number, from a blended copy.
        bank = po.ModelBank([spec])
        bank.fit_predict(df[:n1])
        scored = bank.predict(df[n1 : n1 + 1])["m"].struct.field("pred_y")[0]
        assert scored == pytest.approx(pred[n1], abs=1e-12)


def _no_intercept_rows(n: int, seed: int):
    """Features at a level of 5 and a target with a true intercept of 2: a
    no-intercept fit is then a different regression from the centred one, so
    a model that centres anyway lands on the wrong coefficients and its hidden
    intercept, ``-Σ b_i m_i / s_i``, is of order 10."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 2)) + 5.0
    y = 2.0 + 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0.0, 0.1, n)
    return x, y


class TestNoInterceptIsNotCentred:
    """Pattern B: C8 (``lasso``), C10 (``kalman``), C11 (``robust``), C13
    (``sgd``). With ``add_intercept=False`` and standardization on, these
    models centred the features anyway -- ``ewridge`` alone scaled by the raw
    second moment -- so each solved a system that is neither the centred
    problem (which needs an intercept) nor the raw one, and ``coef`` could not
    reproduce ``pred``. Without an intercept there is nothing to centre on:
    the reference is ``numpy.linalg.lstsq`` on the raw features, no constant
    column. ``add_intercept=True`` (and, where the model has it,
    ``standardize=False``) is the control."""

    def test_lasso_at_zero_penalty_is_numpy_least_squares(self):
        # C8, exact: at a zero penalty the elastic net is least squares.
        x, y = _no_intercept_rows(400, seed=41)
        for intercept in (False, True):
            spec = po.spec.lasso(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                lasso_path=[0.0],
                halflife=float("inf"),
                solve_every=1e-9,
                max_cd_iters=100_000,
                cd_tol=1e-14,
                add_intercept=intercept,
            )
            df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
            pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y__l0")
            for i in range(100, 400, 23):
                b = _wls(x[:i], y[:i], np.ones(i), intercept=intercept)
                want = b[0] + x[i] @ b[1:] if intercept else x[i] @ b
                assert pred[i] == pytest.approx(want, abs=1e-6), (intercept, i)

    @pytest.mark.parametrize("delta", [1e9, float("inf")])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_huber_with_no_outliers_is_numpy_least_squares(self, standardize, delta):
        # C11, exact: with a delta no residual reaches, every IRLS weight is 1
        # and the Huber fit is least squares. At `inf` that is the definition
        # (review 2026-09-12, S27), which the builder refused, so it was 1e9.
        x, y = _no_intercept_rows(400, seed=42)
        spec = po.spec.huber(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            huber_delta=delta,
            ridge=1e-12,
            standardize=standardize,
            halflife=float("inf"),
            solve_every=1e-9,
            add_intercept=False,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y")
        for i in range(100, 400, 23):
            b = _wls(x[:i], y[:i], np.ones(i), intercept=False)
            assert pred[i] == pytest.approx(x[i] @ b, abs=1e-8), (standardize, delta, i)

    def test_kalman_coefficients_reproduce_its_predictions(self):
        # C10. The exact contract every coefficient model meets: the `coef`
        # reported after row i-1 is the fit row i is predicted with.
        # Centring without an intercept broke it by the hidden intercept,
        # about 16 here. Then the library check, statistical: with no process
        # noise the filter is recursive least squares, and after 5000 rows it
        # sits at numpy's no-intercept fit.
        x, y = _no_intercept_rows(5000, seed=43)
        spec = po.spec.kalman(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            coef_halflife=50.0,
            q=[0.0, 0.0],
            p0=1e6,
            obs_var=0.01,
            halflife=float("inf"),
            coef_every=1,
            add_intercept=False,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        coef = np.array(out["coef"].to_list(), dtype=float)
        pred = out["pred_y"].to_numpy()
        np.testing.assert_allclose(np.sum(coef[99:-1] * x[100:], axis=1), pred[100:], rtol=1e-9)
        b = _wls(x, y, np.ones(len(y)), intercept=False)
        np.testing.assert_allclose(coef[-1], b, atol=1e-2)

    def test_sgd_converges_to_numpy_least_squares_and_its_coefficients_follow_it(self):
        # C13. sgd standardizes a row against moments that include it, and
        # reports `coef` through the scaler as it stood after the row before,
        # so its coefficients reproduce a prediction to within a scaler step,
        # not to rounding -- about 1% here, with or without an intercept once
        # nothing is centred (the review's D5). Centring without an intercept
        # put the two thousands of times apart. The library check is the
        # statistical one: the fit settles at numpy's no-intercept least
        # squares.
        x, y = _no_intercept_rows(20_000, seed=44)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            scale_features=True,
            learning_rate=0.01,
            halflife=1e6,
            coef_every=1,
            add_intercept=False,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        coef = np.array(out["coef"].to_list(), dtype=float)
        pred = out["pred_y"].to_numpy()
        via_coef = np.sum(coef[999:-1] * x[1000:], axis=1)
        assert np.max(np.abs(via_coef - pred[1000:]) / np.abs(pred[1000:])) < 0.05
        b = _wls(x, y, np.ones(len(y)), intercept=False)
        np.testing.assert_allclose(coef[-1], b, atol=5e-2)


class TestAZeroWeightRowIsNotSeen:
    """S28. ``weight = 0`` means the row is scored, the clock advances, and
    nothing is learned -- everywhere, the residual diagnostics included.
    ``sigma`` got that right, while the residual quantiles, the
    autocorrelation and the drift detector folded a zero-weight row's
    residual in at full weight, so the row the user had weighted out could
    move a quantile, fire the detector, and under ``drift_action = "reset"``
    restart the model.

    The exact check needs no library: a zero-weight row's *target* cannot
    change anything on the rows after it, so a run where that row's target
    jumps a hundredfold must equal a run where it does not, field by field,
    from the next row on. The library check is the detector itself: river's
    ``PageHinkley`` with ``alpha = 1`` and ``mode = "up"`` is ours flag for
    flag (verified on its own before this test was written), so fed the
    bank's own ``|resid| / sigma`` series with the zero-weight row left out,
    it must reproduce the bank's ``drift`` column."""

    def _run(self, jump: float, drift_action: str = "flag"):
        rng = np.random.default_rng(51)
        n, k0 = 1200, 700
        x = rng.normal(0.0, 1.0, n)
        y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, n)
        w = np.ones(n)
        w[k0] = 0.0
        y[k0] += jump
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            halflife=200.0,
            weight="w",
            emit_sigma=True,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
            emit_drift=True,
            drift_delta=0.05,
            drift_threshold=20.0,
            drift_action=drift_action,
            emit_metrics=True,
        )
        df = pl.DataFrame({"x": x, "y": y, "w": w})
        return po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest(), k0, w

    @pytest.mark.parametrize("drift_action", ["flag", "reset"])
    def test_its_target_changes_nothing_after_it(self, drift_action):
        clean, k0, _ = self._run(0.0, drift_action)
        jumped, _, _ = self._run(100.0, drift_action)
        after = slice(k0 + 1, None)
        for col in clean.columns:
            a, b = clean[col][after], jumped[col][after]
            assert a.equals(b, null_equal=True), col

    def test_the_drift_column_is_rivers_page_hinkley_without_it(self):
        river_drift = pytest.importorskip("river.drift")
        out, _, w = self._run(100.0)
        resid = out["resid_y"].to_numpy()
        sigma = out["sigma_y"].to_numpy()
        flags = out["drift_y"].to_numpy()
        ph = river_drift.PageHinkley(
            min_instances=0, delta=0.05, threshold=20.0, alpha=1.0, mode="up"
        )
        fed = 0
        for t in range(len(resid)):
            e = abs(resid[t]) / sigma[t] if sigma[t] > 0 else float("nan")
            if w[t] > 0 and np.isfinite(e):
                ph.update(float(e))
                fed += 1
                assert bool(flags[t]) == bool(ph.drift_detected), t
            else:
                assert not flags[t], t
        assert fed > 1000


class TestLabelDelayFoldsWhatWasScored:
    """C21 and C5, the review's T-S16. Under ``label_delay`` a row is scored
    when it arrives and learned from ``delay`` rows later -- exactly what
    river's progressive validation does with ``delay``: it predicts each row
    with a model that has learned only the rows at least ``delay`` behind it.
    With no features and no forgetting, the bank's model is river's
    ``StatisticRegressor(stats.Mean())``, so river gives every prediction the
    frame should carry, independently.

    C21: the residual diagnostics folded the residual of the *replay* -- the
    model's prediction at release, after it had learned every row before
    this one -- so ``sigma`` described a prediction nobody was shown, one
    that had seen the labels the delay says were not yet available. Now the
    replay folds the prediction the row was scored with, so ``sigma[t]^2`` is
    the mean of the residuals the frame carries, over the rows whose labels
    have matured by row ``t``. The level flips every ``2·delay`` rows, which
    is where the two definitions part most.

    C5 has no library oracle -- the review's numpy count needs a closed
    group, which ``label_delay`` refuses -- but the queue C21 keeps must stay
    aligned with the rows waiting, and a reset on a skipped row used to leave
    them waiting. Its test is the definition: after such a reset the model
    is the model a fresh bank fed the new session builds."""

    DELAY = 10

    def _rows(self, n: int = 600):
        rng = np.random.default_rng(61)
        level = np.where((np.arange(n) // (2 * self.DELAY)) % 2 == 0, 1.0, -1.0)
        return level + rng.normal(0.0, 0.3, n)

    def _spec(self, **kw):
        return po.spec.ewridge(
            "m",
            targets=["y"],
            features=["zero"],  # no information: the fit is the running mean
            ridge=1e-10,
            halflife=float("inf"),
            solve_every=1e-9,
            min_periods=1,
            emit_sigma=True,
            label_delay=float(self.DELAY),
            **kw,
        )

    def test_the_predictions_are_rivers_delayed_mean_and_sigma_is_their_residuals(self):
        river = pytest.importorskip("river")
        from river import dummy, evaluate, metrics, stats

        y = self._rows()
        n = len(y)
        df = pl.DataFrame({"zero": np.zeros(n), "y": y})
        out = po.ModelBank([self._spec()]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y"].to_numpy()
        resid = out["resid_y"].to_numpy()
        sigma = out["sigma_y"].to_numpy()
        ds = [({"zero": 0.0}, float(v)) for v in y]
        steps = evaluate.iter_progressive_val_score(
            ds,
            dummy.StatisticRegressor(stats.Mean()),
            metrics.MSE(),
            delay=self.DELAY,
            step=1,
            yield_predictions=True,
        )
        want = np.array([s["Prediction"] for s in steps])
        scored = np.isfinite(pred)
        assert scored.sum() > n - 2 * self.DELAY
        # Every prediction the frame carries is river's delayed prediction.
        np.testing.assert_allclose(pred[scored], want[scored], rtol=1e-12, atol=1e-12)
        # sigma at row t is read before row t, after the rows released as it
        # arrived: every row at least `delay` behind it.
        for t in range(3 * self.DELAY, n, 7):
            matured = resid[: t - self.DELAY + 1]
            matured = matured[np.isfinite(matured)]
            assert sigma[t] ** 2 == pytest.approx(np.mean(matured**2), rel=1e-9), t
        assert river is not None

    def test_the_record_survives_a_save_in_the_middle(self):
        y = self._rows()
        n = len(y)
        df = pl.DataFrame({"zero": np.zeros(n), "y": y})
        whole = po.ModelBank([self._spec()]).fit_predict(df)["m"].struct.field("sigma_y")
        bank = po.ModelBank([self._spec()])
        first = bank.fit_predict(df[: n // 2 + 3])["m"].struct.field("sigma_y")
        bank = po.ModelBank.load_bytes(bank.save_bytes())
        second = bank.fit_predict(df[n // 2 + 3 :])["m"].struct.field("sigma_y")
        assert pl.concat([first, second]).equals(whole, null_equal=True)

    def test_a_reset_on_a_skipped_row_drops_the_rows_still_waiting(self):
        rng = np.random.default_rng(62)
        n1, n2 = 60, 60
        x = rng.normal(0.0, 1.0, n1 + n2)
        y = np.where(np.arange(n1 + n2) < n1, 5.0, -5.0) + 2.0 * x + rng.normal(0.0, 0.1, n1 + n2)
        x_col = [float(v) for v in x]
        x_col[n1] = None  # the first row of session 2 is skipped
        df = pl.DataFrame({"x": x_col, "y": y, "s": [1] * n1 + [2] * n2})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            halflife=float("inf"),
            session="s",
            session_gap="reset",
            label_delay=5.0,
            min_periods=2,
        )
        whole = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        fresh = po.ModelBank([spec]).fit_predict(df[n1:])["m"].struct.unnest()
        tail = whole[n1:]
        for col in ("pred_y", "n_eff"):
            assert tail[col].equals(fresh[col], null_equal=True), col
        assert whole["coef"][-1].to_list() == fresh["coef"][-1].to_list()


class TestPcaSignsRunAlongOneGroup:
    """S20. A closed ``ew_cov(pca = r)`` row's loadings are signed against the
    previous close's, so a component does not flip sign from one close to
    the next for no reason. Under ``group_close = "session"`` every group
    closes every session, and the previous close was looked up by (spec,
    instance) alone -- whichever group closed last. That chains every close
    to the one before it, and the chain only misleads when a group's
    component moves *across* the other group's between closes: then its sign
    is set by an overlap that changes sign, and flips while the group itself
    moved smoothly. (Not for any fixed pair of directions, which the review's
    example -- two clouds that are reflections of each other -- is: the chain
    keeps those consistent. Measured, before this test was written.) Here A
    lies along ``x0`` and B alternates between two directions either side of
    ``x1``, which overlap each other by 0.81 but A by +0.3 and -0.3. The
    reference is ``numpy.linalg.eigh`` on each closed row's own co-moments,
    sign-aligned with the previous close *of the same group*, the rule the
    docstring states."""

    def test_the_loadings_keep_their_sign_along_each_groups_closes(self):
        rng = np.random.default_rng(71)
        parts = []
        # Six sessions, the last short: a group's span closes when its
        # session changes, so sessions 1-5 close and 6 stays open.
        for session, n in ((1, 150), (2, 150), (3, 150), (4, 150), (5, 150), (6, 10)):
            b_dir = (0.3, 0.9539392) if session % 2 else (-0.3, 0.9539392)
            for group, (d0, d1) in (("A", (1.0, 0.0)), ("B", b_dir)):
                z = 3.0 * rng.normal(0.0, 1.0, n)
                parts.append(
                    pl.DataFrame(
                        {
                            "g": [group] * n,
                            "s": [session] * n,
                            "x0": d0 * z + 0.2 * rng.normal(0.0, 1.0, n),
                            "x1": d1 * z + 0.2 * rng.normal(0.0, 1.0, n),
                        }
                    )
                )
        df = pl.concat(parts)
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=["mean"],
            pca=1,
            halflife=float("inf"),
            group="g",
            session="s",
            group_close="session",
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        closed = bank.closed_groups()
        assert closed.height == 10
        for group in ("A", "B"):
            rows = closed.filter(pl.col("group") == group).sort("session")
            prev = None
            for row in rows.iter_rows(named=True):
                c00, c01, c11 = row["comoments"]  # the upper triangle, row by row
                vals, vecs = np.linalg.eigh(np.array([[c00, c01], [c01, c11]]))
                v = vecs[:, -1]
                got = np.asarray(row["eig_vecs"])
                # The first close's sign is the model's own; after that, the
                # same group's previous close decides it.
                ref = got if prev is None else prev
                v = v * np.sign(v @ ref)
                np.testing.assert_allclose(got, v, atol=1e-9, err_msg=f"{group} {row['session']}")
                assert row["eig_vals"][0] == pytest.approx(vals[-1], rel=1e-9)
                prev = v


class TestTheGramIsTheWindowsToo:
    """S19, the review's T-S1 both ways. Under a ``window`` the fit is solved
    from the truncated accumulators, and ``bank.coef()`` reports it -- but
    ``bank.gram()`` and a closed row's Gram read the *live* ones, so
    ``po.gram.solve`` on them gave the whole history's fit beside a ``coef``
    solved on the window: two histories behind one spec. The reference is
    ``numpy.linalg.lstsq`` on the rows the last fit was read from, at
    ``0.5 ** (age / halflife)``: both ``coef()`` and the solve of the Gram must
    land on it. ``window=None`` is the control, where the two histories are
    the same one. Under a window the Gram carries no target moments -- the
    window's snapshots do not keep them -- and says so with ``None``, as it
    does for a state written before they existed."""

    @pytest.mark.parametrize("window", [None, 60.0])
    def test_the_gram_solves_to_the_fit_the_bank_reports(self, window):
        rng = np.random.default_rng(81)
        n, halflife = 400, 25.0
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
            window=window,
            solve_every=1e-9,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        # The last fit was solved after the last row: ages count from it.
        keep, w = _window((n - 1) - np.arange(n), halflife, window)
        want = _wls(x[keep], y[keep], w)
        coef = bank.coef("m")["coef"].to_numpy()
        np.testing.assert_allclose(coef, want, atol=1e-6)
        g = bank.gram("m")[0]
        assert g["n_eff"] == pytest.approx(w.sum(), rel=1e-10)
        np.testing.assert_allclose(po.gram.solve(g, ridge=1e-10), want, atol=1e-6)
        if window is not None:
            assert g["target_means"] is None and g["target_vars"] is None


class TestTheCrossMomentsAreCentred:
    """N1, found while fixing C1 and not in the review. ``ewridge`` kept each
    target's cross-moments raw, ``E_w[z·y]``, and formed the standardized
    solve's right-hand side as ``E[z·y] − m·ȳ``: two numbers the size of
    ``level²`` subtracted to leave one the size of a covariance -- pattern
    E, at the one site the review's grep did not reach. The unstandardized
    solve read the raw normal equations, whose conditioning falls the same
    way. A feature and a target at a common level ``L`` (a price regressed on
    prices) lost the fit as ``L`` grew: at ``1e8`` the prediction was off by
    about ten. Each target now keeps the means of the feature row and of the
    target over the rows it was present on, and a centred cross-moment, and
    both solves read the centred system.

    The reference is ``numpy.linalg.lstsq`` on the rows centred at their own
    weighted means, which is exact at any level. The tolerance is ``1e-14·L``
    on top of ``1e-10``: the data is resolved to ``L·ε`` before either side
    touches it. Offset 0 is the control."""

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_a_level_regressed_on_levels_is_the_numpy_fit(self, standardize, offset):
        rng = np.random.default_rng(97)
        n, halflife = 600, 200.0
        u = rng.normal(0.0, 1.0, (n, 2))
        x = offset + u
        # y = 2·x0 − x1 + noise: the target sits at the level too.
        y = offset + 2.0 * u[:, 0] - u[:, 1] + rng.normal(0.0, 0.1, n)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=halflife,
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        pred = po.ModelBank([spec]).fit_predict(frame)["m"].struct.field("pred_y").to_numpy()
        worst = 0.0
        for t in range(100, n, 50):
            # Row t is predicted by the fit of the rows before it.
            w = 0.5 ** (((t - 1) - np.arange(t)) / halflife)
            xm = w @ x[:t] / w.sum()
            ym = w @ y[:t] / w.sum()
            slopes = _wls(x[:t] - xm, y[:t] - ym, w, intercept=False)
            worst = max(worst, abs(pred[t] - (ym + (x[t] - xm) @ slopes)))
        assert worst <= 1e-10 + 1e-14 * offset, f"worst |pred - numpy| = {worst:.3e}"


def _level_rows(n: int, offset: float, seed: int):
    """Two features and a target at a common level ``offset``, the target
    ``2·x0 − x1`` plus noise: a price regressed on prices."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, (n, 2))
    x = offset + u
    y = offset + 2.0 * u[:, 0] - u[:, 1] + rng.normal(0.0, 0.1, n)
    return x, y


def _centred_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``numpy.linalg.lstsq`` on the rows centred at their means, the
    intercept recovered from them: exact at any level."""
    xm, ym = x.mean(axis=0), y.mean()
    slopes = np.linalg.lstsq(x - xm, y - ym, rcond=None)[0]
    return np.concatenate([[ym - xm @ slopes], slopes])


class TestTheGramIsCentredToo:
    """N4, found while fixing N1 and half done with task 81. The models solve
    from centred cross-moments, but ``bank.gram()`` exported them raw,
    ``E[z·y]``, so ``po.gram.solve`` -- and ``lasso_path`` and
    ``coef_stats`` -- formed the right-hand side as ``E[z·y] − m·ȳ``: the
    subtraction N1 took out of the model, which at a level ``L`` keeps
    ``L²·ε`` of a covariance. The export carries the centred cross-moments
    the model holds, a closed row and a merge carry them on, and the offline
    solves read them. ``numpy.linalg.lstsq`` on centred rows is the
    reference, exact at any level; the tolerance is N1's, ``1e-10`` plus
    ``1e-14·L`` for the data's own resolution, on what a level resolves:
    the slopes and the predictions they give. Offset 0 is the control."""

    @staticmethod
    def _spec(standardize=False, **kw):
        return po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=float("inf"),
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
            **kw,
        )

    @staticmethod
    def _tol(offset: float) -> float:
        return 1e-10 + 1e-14 * offset

    @staticmethod
    def _same_fit(got, want, x, tol, what):
        """Two fits agree where a fit at a level is resolved: the slopes, and
        the predictions they give at the rows. An intercept at a level is
        not: it is ``ybar - m @ b``, so a slope known to ``δ`` moves it by
        ``L·δ``, ``L²·ε`` in all -- in numpy's fit as in any other."""
        assert np.max(np.abs(got[1:] - want[1:])) <= tol, (what, "slopes", got, want)
        gap = np.max(np.abs((got[0] + x @ got[1:]) - (want[0] + x @ want[1:])))
        assert gap <= tol, (what, "predictions", gap)

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_solve_on_the_gram_is_the_numpy_fit_at_any_level(self, standardize, offset):
        x, y = _level_rows(600, offset, seed=98)
        spec = self._spec(standardize)
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        got = po.gram.solve(bank.gram("m")[0], standardize=standardize)
        self._same_fit(got, _centred_fit(x, y), x, self._tol(offset), "numpy")
        # And the model's own fit, which read the centred system all along.
        coef = bank.coef("m")["coef"].to_numpy()
        self._same_fit(got, coef, x, self._tol(offset), "coef")

    @pytest.mark.parametrize("offset", [0.0, 1e8])
    def test_a_merge_a_closed_row_and_the_other_solves_keep_it(self, offset):
        x, y = _level_rows(1200, offset, seed=99)
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        tol = self._tol(offset)
        # A merge of two shards is the Gram of the union: their centred
        # cross-moments pool with the gap between their means, as the
        # co-moments do.
        halves = []
        for part in (frame.head(700), frame.tail(500)):
            bank = po.ModelBank([self._spec()])
            bank.fit_predict(part)
            halves.append(bank.gram("m")[0])
        merged = po.gram.merge(halves)
        want = _centred_fit(x, y)
        self._same_fit(po.gram.solve(merged), want, x, tol, "merge")
        # A closed row is the Gram it was written from.
        spec = self._spec(group="g", group_close="monotone")
        bank = po.ModelBank([spec])
        bank.fit_predict(frame.with_columns(g=pl.Series(["a"] * 700 + ["b"] * 500)))
        (row,) = bank.closed_groups().iter_rows(named=True)
        got = po.gram.solve(po.gram.from_row(row))
        self._same_fit(got, _centred_fit(x[:700], y[:700]), x[:700], tol, "closed row")
        # The lasso path at no penalty is the same fit, and the residual
        # variance coef_stats reads is numpy's.
        g = halves[0]
        want_h = _centred_fit(x[:700], y[:700])
        path = po.gram.lasso_path(g, [0.0], max_iter=100_000, tol=1e-15)[0]
        self._same_fit(path, want_h, x[:700], 1e3 * tol, "lasso_path")
        stats = po.gram.coef_stats(g, want_h)
        z = np.column_stack([np.ones(700), x[:700]])
        resid = y[:700] - z @ want_h
        # The data resolves a variance of 0.01 to about 1e-6 of itself at 1e8.
        rel = 1e-9 + 1e-12 * offset
        assert stats["resid_var"] == pytest.approx(np.mean(resid**2), rel=rel, abs=1e-12)


def _holt(y: np.ndarray, halflife: float, trend_halflife: float) -> tuple[np.ndarray, float, float]:
    """``holt``'s ``pred`` on a row-count clock, and its final level and
    trend. ``NaN`` in ``y`` is a null target."""
    spec = po.spec.holt(
        "h", targets=["y"], halflife=halflife, trend_halflife=trend_halflife, min_periods=0.0
    )
    bank = po.ModelBank([spec])
    frame = pl.DataFrame(
        {"y": [None if np.isnan(v) else float(v) for v in y]}, schema={"y": pl.Float64}
    )
    pred = bank.fit_predict(frame)["h"].struct.field("pred_y").to_numpy()
    coef = dict(bank.coef("h").select("term", "coef").iter_rows())
    return pred, coef["level"], coef["trend"]


class TestHoltAcrossAMissingObservation:
    """C22. ``holt`` did not move its level across a row whose target was
    null or whose weight was zero, so the row after it forecast one trend
    step short, and its level update read a two-step move as a one-step
    slope. Each target now keeps its clock since its last observation, and
    extrapolates over it and forms its rates from it, so a row the model
    cannot learn from is *transparent*: the same numbers as if it were absent
    and its clock folded into the next row's.

    ``statsmodels`` is the second opinion twice. Its ``Holt`` is the textbook
    recursion, whose gains are fixed; ``holt``'s level and trend are weighted
    means (review 2026-09-12, S29/S30), whose gains ``w / (lam·W + w)`` start
    at 1 and fall to those fixed ones as the weight saturates. So the two part
    over the first rows by design and agree exactly from row ``SETTLED`` on a
    row-count clock -- the control that pins the mapping from halflives to
    smoothing weights (the review's T-S14). Its state-space
    ``ExponentialSmoothing`` takes ``NaN`` in the
    series and answers it with the prediction step alone, so the row after a
    missing one forecasts ``l + 2b`` from the ``l`` and ``b`` that stood
    before it, a number the gain does not enter (T-S17). Past that row the two
    part by design: its gain is fixed, and ours grows with the clock since the
    last observation, which is what a halflife in clock units means."""

    H_LEVEL, H_TREND = 6.0, 25.0
    #: The row from which the weighted means' gains equal the fixed ones to
    #: the last digit this test reads: ``0.5 ** (1000 / H_TREND)`` is 1e-12.
    SETTLED = 1000

    @staticmethod
    def series(n: int) -> np.ndarray:
        rng = np.random.default_rng(5)
        return 3.0 + 0.4 * np.arange(n) + np.cumsum(rng.normal(0.0, 0.5, n))

    def test_the_recursion_is_statsmodels_holt(self):
        holtwinters = pytest.importorskip("statsmodels.tsa.holtwinters")
        y = self.series(self.SETTLED + 200)
        pred, level, trend = _holt(y, self.H_LEVEL, self.H_TREND)
        res = holtwinters.Holt(
            y, initialization_method="known", initial_level=y[0], initial_trend=0.0
        ).fit(
            smoothing_level=1.0 - 0.5 ** (1.0 / self.H_LEVEL),
            smoothing_trend=1.0 - 0.5 ** (1.0 / self.H_TREND),
            optimized=False,
        )
        fitted = np.asarray(res.fittedvalues)
        # The first rows part by design: a weighted mean's first gains are
        # larger than the fixed ones, so it follows the series sooner.
        assert not np.allclose(pred[1:50], fitted[1:50], rtol=1e-6)
        np.testing.assert_allclose(pred[self.SETTLED :], fitted[self.SETTLED :], rtol=1e-12)
        assert level == pytest.approx(res.level[-1], rel=1e-12)
        assert trend == pytest.approx(res.trend[-1], rel=1e-12)

    def test_the_row_after_a_missing_one_forecasts_two_trend_steps(self):
        es = pytest.importorskip("statsmodels.tsa.statespace.exponential_smoothing")
        y = self.series(self.SETTLED + 200)
        alpha = 1.0 - 0.5 ** (1.0 / self.H_LEVEL)
        beta = 1.0 - 0.5 ** (1.0 / self.H_TREND)

        def smooth(series: np.ndarray) -> np.ndarray:
            model = es.ExponentialSmoothing(
                series,
                trend=True,
                initialization_method="known",
                initial_level=series[0],
                initial_trend=0.0,
            )
            # The innovations form: the trend's gain is alpha·beta.
            return np.asarray(model.smooth([alpha, alpha * beta]).fittedvalues)

        # With nothing missing, every settled row: the control that pins the
        # mapping.
        s = self.SETTLED
        pred, _, _ = _holt(y, self.H_LEVEL, self.H_TREND)
        np.testing.assert_allclose(pred[s:], smooth(y)[s:], rtol=1e-9)
        missing = s + 100
        gappy = y.copy()
        gappy[missing] = np.nan
        pred, _, _ = _holt(gappy, self.H_LEVEL, self.H_TREND)
        # Every settled row up to the one after the gap, which is `l + 2b`.
        np.testing.assert_allclose(pred[s : missing + 2], smooth(gappy)[s : missing + 2], rtol=1e-9)

    @pytest.mark.parametrize("how", ["null", "zero weight"])
    def test_a_row_it_cannot_learn_from_is_as_if_absent(self, how):
        n = 150
        y = self.series(n)
        skip = np.arange(n) % 3 == 2
        full = pl.DataFrame(
            {
                "t": np.arange(n, dtype=float),
                "y": [
                    None if s and how == "null" else float(v) for v, s in zip(y, skip, strict=True)
                ],
                "w": np.where(skip & (how == "zero weight"), 0.0, 1.0),
            },
            schema={"t": pl.Float64, "y": pl.Float64, "w": pl.Float64},
        )
        spec = po.spec.holt(
            "h",
            targets=["y"],
            halflife=self.H_LEVEL,
            trend_halflife=self.H_TREND,
            clock="t",
            max_dclock=10.0,
            weight="w",
            min_periods=0.0,
        )
        got = po.ModelBank([spec]).fit_predict(full)["h"].struct.field("pred_y").to_numpy()
        # The same stream with those rows removed: the clock folds their
        # deltas into the next row's.
        kept = full.filter(pl.Series(~skip))
        want = po.ModelBank([spec]).fit_predict(kept)["h"].struct.field("pred_y").to_numpy()
        np.testing.assert_allclose(got[~skip], want, rtol=1e-12)


class TestBocpdAtALevel:
    """C23. ``bocpd`` kept each run's sums raw, ``Σw·x`` and ``Σw·x²``, and
    formed the scatter as ``Σw·x² − n·x̄²`` on every row, for every run --
    pattern E in the representation itself, not at a blend or a truncation.
    At a level ``L`` the scatter's error is ``n·L²·ε``, and at ``1e8``
    unit-variance noise read as a changepoint on every row. The runs now keep
    Welford means and centred scatters.

    The second opinion is ``bayesian_changepoint_detection``, whose
    ``StudentT`` runs the conjugate update centred, ``β += κ(x − μ)²/(2(κ +
    1))``: the normal-inverse-gamma model ``emission = "diag"`` is at one
    feature, with ``alpha = prior_nu/2``, ``beta = prior_scale/2``, ``kappa =
    prior_kappa`` and ``mu = prior_mean``, through Adams and MacKay's
    recursion at a constant hazard with nothing pruned. ``R[0, t+1] + R[1,
    t+1]`` is our ``p_change`` on row ``t``, ``argmax R[:, t]`` our
    ``run_mode`` and ``Σ r·R[r, t]`` our ``run_mean``. At ``1e6`` and ``1e8``
    both sides are shifted, the package through ``mu``; the tolerance grows
    as ``1e-14·L`` because the data is resolved to ``L·ε`` before either
    side reads it. Offset 0 is the control."""

    @pytest.mark.parametrize("offset", [0.0, 1e6, 1e8])
    def test_the_run_length_posterior_is_the_packages(self, offset):
        bcd = pytest.importorskip("bayesian_changepoint_detection.online_changepoint_detection")
        rng = np.random.default_rng(23)
        n, hazard = 120, 40.0
        nu0, psi0, kappa0 = 2.0, 1.0, 1.0
        x = offset + np.where(np.arange(n) < 60, 0.0, 3.0) + rng.normal(0.0, 1.0, n)
        spec = po.spec.bocpd(
            "b",
            features=["x"],
            hazard=hazard,
            emission="diag",
            prior_mean=[offset],
            prior_kappa=kappa0,
            prior_nu=nu0,
            prior_scale=[psi0],
            prune_below=0.0,
            max_run=n + 2,
            min_periods=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x}))["b"]
        r, maxes = bcd.online_changepoint_detection(
            x,
            functools.partial(bcd.constant_hazard, hazard),
            bcd.StudentT(nu0 / 2.0, psi0 / 2.0, kappa0, offset),
        )
        tol = 1e-9 + 1e-14 * offset
        np.testing.assert_allclose(
            out.struct.field("p_change").to_numpy(), r[0, 1:] + r[1, 1:], rtol=0.0, atol=tol
        )
        np.testing.assert_allclose(
            out.struct.field("run_mean").to_numpy(), np.arange(n + 1) @ r[:, :n], rtol=tol
        )
        np.testing.assert_array_equal(out.struct.field("run_mode").to_numpy(), maxes[:n])


class TestKalmanZeroWeightRow:
    """S9. ``kalman``'s per-target weights -- ``wj``, which gates the
    prediction, and ``wsig``, the memory of the residual variance ``σ²`` that
    sets both the observation noise ``σ²/w`` and the process noise ``σ²·(ln 2
    / coef_halflife)²`` -- decayed on a row whose target was null and not on
    one whose target was present at weight zero. The filter treats the two
    alike, a prediction step and no update, so ``σ²`` remembered more across
    one than the other.

    The second opinion is ``filterpy``'s ``KalmanFilter`` beside a ``numpy``
    recursion for ``σ²``: ``predict(Q)`` on every row, ``update(y, R = σ²/w,
    H = z)`` only where there is a target and a positive weight, and ``σ²``
    the EW mean of the squared out-of-sample residuals with its weight
    decayed on every row. Unstandardized, so ``kf.x`` is our coefficient
    vector; ``filterpy`` updates ``P`` in Joseph form and ``kalman`` in the
    simple form, so the two agree to rounding. The same rows with the target
    null instead of the weight zero are the control."""

    @staticmethod
    def filterpy_pred(
        kalman: Any, x: np.ndarray, y: np.ndarray, w: np.ndarray, halflife: float, coef_hl: float
    ) -> np.ndarray:
        n, k = x.shape
        kf = kalman.KalmanFilter(dim_x=k + 1, dim_z=1)
        kf.x = np.zeros((k + 1, 1))
        kf.P = np.eye(k + 1)
        kf.F = np.eye(k + 1)
        sig2 = wsig = wj = 0.0
        pred = np.full(n, np.nan)
        for i in range(n):
            d = 0.0 if i == 0 else 1.0
            lam = 0.5 ** (d / halflife)
            s2 = sig2 if sig2 > 0.0 else 1.0
            kf.predict(Q=np.eye(k + 1) * s2 * (np.log(2.0) / coef_hl) ** 2 * d)
            z = np.concatenate(([1.0], x[i]))
            if wj > 0.0:
                pred[i] = z @ kf.x[:, 0]
            if np.isnan(y[i]) or w[i] <= 0.0:
                # A prediction step and no update, and time passes for both
                # weights.
                wj *= lam
                wsig *= lam
                continue
            kf.update(y[i], R=s2 / w[i], H=z[None, :])
            if not np.isnan(pred[i]):
                r = y[i] - pred[i]
                ws_new = lam * wsig + w[i]
                sig2 = (lam * wsig * sig2 + w[i] * r * r) / ws_new
                wsig = ws_new
            wj = lam * wj + w[i]
        return pred

    @pytest.mark.parametrize("how", ["null", "zero weight"])
    def test_the_filter_is_filterpy_with_sigma_decayed_on_every_row(self, how):
        kalman = pytest.importorskip("filterpy.kalman")
        rng = np.random.default_rng(41)
        n, halflife, coef_hl = 300, 30.0, 50.0
        x = rng.normal(0.0, 1.0, (n, 2))
        drift = np.arange(n) / n
        y = 0.5 + (1.5 - drift) * x[:, 0] + (-0.8 + 2.0 * drift) * x[:, 1]
        y = y + rng.normal(0.0, 0.3, n)
        skip = (np.arange(n) % 9 == 4) & (np.arange(n) > 20)
        w = np.where(skip & (how == "zero weight"), 0.0, 1.0)
        y_seen = np.where(skip & (how == "null"), np.nan, y)
        frame = pl.DataFrame(
            {
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": [None if np.isnan(v) else float(v) for v in y_seen],
                "w": w,
            },
            schema={"x0": pl.Float64, "x1": pl.Float64, "y": pl.Float64, "w": pl.Float64},
        )
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            coef_halflife=coef_hl,
            standardize=False,
            p0=1.0,
            weight="w",
            halflife=halflife,
            min_periods=0.0,
        )
        got = po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("pred_y").to_numpy()
        want = self.filterpy_pred(kalman, x, y_seen, w, halflife, coef_hl)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


class TestAHopelessSerialFactorSaysSo:
    """S18. ``marginal``'s ``serial_rule = "truncated"`` corrects the count
    behind ``t`` by ``1 + 2·Σ ρ_x(l)·ρ_y(l)``, and two series whose
    autocorrelations have opposite signs take that below zero: an estimate
    outside the parameter space. The rule floored it at ``f64::MIN_POSITIVE``,
    so ``n_serial`` came out ``+inf`` and ``t_serial`` ``±inf`` -- infinite
    evidence from a correction that had failed. It is NaN now, the answer
    ``"geometric"`` already gives a factor it cannot form.

    ``statsmodels``' ``acf`` is the second opinion on the precondition: its
    own lag-1 autocorrelations of this pair, about ``+0.8`` and ``-0.8``, put
    the factor below zero, so the case is the pair's and not an artefact of
    our estimator (which centres each leg at the pre-row mean, and agrees with
    ``acf`` to the statistical tier the review's T-S11 gives, ``0.02`` at this
    length). Two series that are both positively autocorrelated are the
    control, where the count is ``n_kish`` over the factor."""

    @pytest.mark.parametrize("phi_y", [0.8, -0.8])
    def test_the_count_is_nan_where_the_factor_is_not_positive(self, phi_y):
        stattools = pytest.importorskip("statsmodels.tsa.stattools")
        rng = np.random.default_rng(13)
        n = 5000

        def ar1(phi: float) -> np.ndarray:
            e = rng.normal(0.0, 1.0, n)
            out = np.empty(n)
            out[0] = e[0]
            for t in range(1, n):
                out[t] = phi * out[t - 1] + e[t]
            return out

        x, y = ar1(0.8), ar1(phi_y)
        spec = po.spec.marginal(
            "m",
            targets=["y"],
            features=["x"],
            lags=[1],
            serial_rule="truncated",
            halflife=float("inf"),
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        row = bank.marginal("m").row(0, named=True)
        rho_x = stattools.acf(x, nlags=1)[1]
        rho_y = stattools.acf(y, nlags=1)[1]
        assert row["lagcorr_xx"][0] == pytest.approx(rho_x, abs=0.02)
        assert row["lagcorr_yy"][0] == pytest.approx(rho_y, abs=0.02)
        if 1.0 + 2.0 * rho_x * rho_y > 0.0:
            factor = 1.0 + 2.0 * row["lagcorr_xx"][0] * row["lagcorr_yy"][0]
            assert row["n_serial"] == pytest.approx(row["n_kish"] / factor, rel=1e-12)
            assert np.isfinite(row["t_serial"])
        else:
            # NaN in the model, null in the frame, as `phi_x` is.
            assert row["n_serial"] is None, row["n_serial"]
            assert row["t_serial"] is None, row["t_serial"]


def _gappy(n: int, seed: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Two features, and a target at ``level`` present only where ``x0 >
    -0.5``: gaps tied to a feature, so the features' means and spreads over
    the target's rows are not their means and spreads over every row."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 2))
    y = level + 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.1, n)
    return x, np.where(x[:, 0] > -0.5, y, np.nan)


def _gappy_frame(x: np.ndarray, **targets: np.ndarray) -> pl.DataFrame:
    """The features and targets as a frame, ``NaN`` in a target as null."""
    cols: dict[str, Any] = {"x0": x[:, 0], "x1": x[:, 1]}
    for name, v in targets.items():
        cols[name] = [None if np.isnan(t) else float(t) for t in v]
    return pl.DataFrame(cols, schema={c: pl.Float64 for c in cols})


def _own_rows_pred(x: np.ndarray, y: np.ndarray, t: int, halflife: float) -> float:
    """Row ``t`` predicted by the weighted least-squares fit of the rows
    before it on which the target is present, each at ``0.5 ** (age /
    halflife)`` -- the age counted in rows, the target's missing ones
    included, since the clock runs on every row."""
    keep = ~np.isnan(y[:t])
    w = 0.5 ** (((t - 1) - np.arange(t)) / halflife)
    b = _wls(x[:t][keep], y[:t][keep], w[keep])
    return float(b[0] + x[t] @ b[1:])


def _pairwise_pred(x: np.ndarray, y: np.ndarray, t: int, halflife: float) -> float:
    """Row ``t`` predicted from pairwise-complete weighted moments of the rows
    before it: the features' covariance over every row, their covariance
    with the target and every mean over the rows the target is present on,
    each by ``numpy.cov`` with the rows' weights as ``aweights``."""
    keep = ~np.isnan(y[:t])
    w = 0.5 ** (((t - 1) - np.arange(t)) / halflife)
    cxx = np.cov(x[:t].T, aweights=w, ddof=0)
    joint = np.cov(np.column_stack([x[:t][keep], y[:t][keep]]).T, aweights=w[keep], ddof=0)
    slopes = np.linalg.solve(cxx, joint[:2, 2])
    mx = np.average(x[:t][keep], axis=0, weights=w[keep])
    my = np.average(y[:t][keep], weights=w[keep])
    return float(my + (x[t] - mx) @ slopes)


class TestATargetWithGaps:
    """N3, found while fixing N1, and ``target_gaps`` (docs/PLAN.md task 81).
    With a target null on some rows, ``ewridge`` and ``lasso`` read the Gram
    over every row and the target's cross-moments over its own rows, so a
    slope moved with the target's level, by ``(m_j - m)·ȳ_j / Var(x)`` --
    ``m_j`` a feature's mean over the target's rows, ``m`` its mean over all
    of them. Here the target is present only where ``x0 > -0.5``, which puts
    ``m_j - m`` near 0.5, and the target sits at a level.

    ``target_gaps="own_rows"``, the default, fits each target on exactly the
    rows it is present on: ``numpy.linalg.lstsq`` on those rows is the second
    opinion. ``"pairwise"`` reads pairwise-complete moments: pandas'
    ``DataFrame.cov`` without decay, ``numpy.cov`` with the rows' weights
    under it. The two answers differ here because the gaps are tied to
    ``x0``, and each test checks that too, so neither passes by accident.
    The tolerances are an order above what the solves round to at these
    levels."""

    @staticmethod
    def _pred(spec: dict[str, Any], frame: pl.DataFrame, field: str) -> np.ndarray:
        out = po.ModelBank([spec]).fit_predict(frame)
        return out[spec["name"]].struct.field(field).to_numpy()

    @pytest.mark.parametrize("level", [0.0, 50.0])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_own_rows_is_the_numpy_fit_of_the_targets_rows(self, standardize, level):
        n, h = 600, 40.0
        x, y = _gappy(n, 7, level)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=h,
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y")
        worst, apart = 0.0, 0.0
        for t in range(100, n, 23):
            want = _own_rows_pred(x, y, t, h)
            worst = max(worst, abs(pred[t] - want))
            apart = max(apart, abs(want - _pairwise_pred(x, y, t, h)))
        assert worst <= 1e-9 * (1.0 + level), f"worst |pred - numpy| = {worst:.3e}"
        assert apart > 1e-2, f"the two answers should differ here: {apart:.3e}"

    @pytest.mark.parametrize("standardize", [False, True])
    def test_each_target_is_fitted_on_its_own_rows(self, standardize):
        """Three targets with three patterns of missing rows: present on
        every row, where ``x0 > -0.5``, and where ``x1 < 0.3``. Each takes its
        own Gram at its first gap, and each fit is ``lstsq`` on its rows."""
        rng = np.random.default_rng(17)
        n, h = 600, 60.0
        x = rng.normal(0.0, 1.0, (n, 2))
        base = 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.1, n)
        ys = {
            "ya": 3.0 + base,
            "yb": np.where(x[:, 0] > -0.5, 5.0 + base, np.nan),
            "yc": np.where(x[:, 1] < 0.3, -4.0 + 0.5 * base, np.nan),
        }
        spec = po.spec.ewridge(
            "m",
            targets=list(ys),
            features=["x0", "x1"],
            halflife=h,
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
        )
        out = po.ModelBank([spec]).fit_predict(_gappy_frame(x, **ys))["m"]
        for name, y in ys.items():
            pred = out.struct.field(f"pred_{name}").to_numpy()
            worst = max(abs(pred[t] - _own_rows_pred(x, y, t, h)) for t in range(100, n, 29))
            assert worst <= 1e-9, f"{name}: worst |pred - numpy| = {worst:.3e}"

    def test_pairwise_is_pandas_pairwise_covariance(self):
        """Without decay the pairwise moments are pandas' pairwise-complete
        covariance, which divides each pair by its own count less one; the
        model's moments are means, so each pair is rescaled by its count."""
        pd = pytest.importorskip("pandas")
        n = 500
        x, y = _gappy(n, 11, 0.0)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=float("inf"),
            ridge=0.0,
            target_gaps="pairwise",
            max_rows_between_solves=1,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y")
        worst, apart = 0.0, 0.0
        for t in range(60, n, 37):
            keep = ~np.isnan(y[:t])
            cov = pd.DataFrame({"x0": x[:t, 0], "x1": x[:t, 1], "y": y[:t]}).cov()
            n_y = int(keep.sum())
            cxx = cov.loc[["x0", "x1"], ["x0", "x1"]].to_numpy() * (t - 1) / t
            cxy = cov.loc[["x0", "x1"], "y"].to_numpy() * (n_y - 1) / n_y
            slopes = np.linalg.solve(cxx, cxy)
            want = y[:t][keep].mean() + (x[t] - x[:t][keep].mean(axis=0)) @ slopes
            worst = max(worst, abs(pred[t] - want))
            apart = max(apart, abs(want - _own_rows_pred(x, y, t, float("inf"))))
        assert worst <= 1e-10, f"worst |pred - pandas| = {worst:.3e}"
        assert apart > 1e-2, f"the two answers should differ here: {apart:.3e}"

    @pytest.mark.parametrize("level", [0.0, 50.0])
    def test_pairwise_under_decay_is_numpy_weighted_covariance(self, level):
        n, h = 600, 40.0
        x, y = _gappy(n, 13, level)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=h,
            ridge=0.0,
            target_gaps="pairwise",
            solve_every=1e-9,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y")
        worst = max(abs(pred[t] - _pairwise_pred(x, y, t, h)) for t in range(100, n, 41))
        assert worst <= 1e-9 * (1.0 + level), f"worst |pred - numpy| = {worst:.3e}"

    @pytest.mark.parametrize("target_gaps", ["own_rows", "pairwise"])
    def test_lasso_at_zero_penalty_is_the_same_fit(self, target_gaps):
        """``lasso`` reads the same accumulators, so at a zero penalty its
        path point is the same least-squares fit; its coordinate descent is
        run to convergence here. At a level it also keeps its cross-moments
        centred now, as ``ewridge`` does since N1 (N2)."""
        n, h = 500, 50.0
        x, y = _gappy(n, 19, 20.0)
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            lasso_path=[0.5, 0.0],
            halflife=h,
            target_gaps=target_gaps,
            max_rows_between_solves=1,
            max_cd_iters=10_000,
            cd_tol=1e-15,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y__l0")
        ref = _own_rows_pred if target_gaps == "own_rows" else _pairwise_pred
        worst = max(abs(pred[t] - ref(x, y, t, h)) for t in range(100, n, 31))
        assert worst <= 1e-8, f"worst |pred - numpy| = {worst:.3e}"

    @pytest.mark.parametrize("target_gaps", ["own_rows", "pairwise"])
    def test_a_window_with_gaps_is_the_fit_of_the_rows_inside_it(self, target_gaps):
        """A ``window`` over a target with gaps, beside one without, so that
        under ``"own_rows"`` the two part inside the window's history and
        each Gram is truncated against the one its target read at the
        boundary. The fit is of the rows inside the window, each at ``0.5 **
        (age / halflife)``: the target's own by ``numpy.linalg.lstsq`` under
        ``"own_rows"``, and under ``"pairwise"`` every row's feature
        covariance against the target's own cross-covariance, both by
        ``numpy.cov``."""
        n, h, window = 500, 40.0, 60.0
        x, y = _gappy(n, 41, 3.0)
        rng = np.random.default_rng(43)
        ya = 2.0 + x @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.1, n)
        spec = po.spec.ewridge(
            "m",
            targets=["ya", "y"],
            features=["x0", "x1"],
            halflife=h,
            ridge=0.0,
            window=window,
            target_gaps=target_gaps,
            solve_every=1e-9,
        )
        pred = self._pred(spec, _gappy_frame(x, ya=ya, y=y), "pred_y")
        worst = 0.0
        for t in range(150, n, 29):
            keep, w = _window((t - 1) - np.arange(t), h, window)
            xs, ys = x[:t][keep], y[:t][keep]
            own = ~np.isnan(ys)
            if target_gaps == "own_rows":
                b = _wls(xs[own], ys[own], w[own])
                want = b[0] + x[t] @ b[1:]
            else:
                cxx = np.cov(xs.T, aweights=w, ddof=0)
                joint = np.cov(np.column_stack([xs[own], ys[own]]).T, aweights=w[own], ddof=0)
                slopes = np.linalg.solve(cxx, joint[:2, 2])
                mx = np.average(xs[own], axis=0, weights=w[own])
                want = np.average(ys[own], weights=w[own]) + (x[t] - mx) @ slopes
            worst = max(worst, abs(pred[t] - want))
        assert worst <= 1e-8, f"worst |pred - numpy| = {worst:.3e}"

    @pytest.mark.parametrize("standardize", [False, True])
    def test_own_rows_is_statsmodels_wls_on_the_targets_rows(self, standardize):
        """The same fit from a second library: ``statsmodels``' ``WLS`` on the
        rows the target is present on, each at the weight the decay gives
        it, with the target at a level."""
        sm = pytest.importorskip("statsmodels.api")
        n, h, level = 500, 30.0, 40.0
        x, y = _gappy(n, 23, level)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=h,
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y")
        worst = 0.0
        for t in range(100, n, 37):
            keep = ~np.isnan(y[:t])
            w = 0.5 ** (((t - 1) - np.arange(t)) / h)
            fit = sm.WLS(y[:t][keep], sm.add_constant(x[:t][keep]), weights=w[keep]).fit()
            worst = max(worst, abs(pred[t] - fit.params @ np.r_[1.0, x[t]]))
        assert worst <= 1e-9 * (1.0 + level), f"worst |pred - statsmodels| = {worst:.3e}"

    @pytest.mark.parametrize("standardize", [False, True])
    def test_a_ridge_on_its_own_rows_is_statsmodels_ridge(self, standardize):
        """With a penalty. ``statsmodels``' ridge -- ``fit_regularized`` with
        ``L1_wt=0`` -- minimizes ``RSS / (2n) + alpha / 2 * |b|^2``, which on
        rows centred at their means is the model's ``(C + ridge * I) b = c``
        with ``alpha = ridge``: on the features' own scale, or on their
        standard deviations with ``standardize``. No decay, so the target's
        rows weigh the same, and ``n`` is their count."""
        sm = pytest.importorskip("statsmodels.api")
        n, ridge = 400, 0.3
        x, y = _gappy(n, 29, 10.0)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=float("inf"),
            ridge=ridge,
            standardize=standardize,
            max_rows_between_solves=1,
        )
        pred = self._pred(spec, _gappy_frame(x, y=y), "pred_y")
        worst, shrunk = 0.0, 0.0
        for t in range(60, n, 31):
            keep = ~np.isnan(y[:t])
            xs, ys = x[:t][keep], y[:t][keep]
            mx, my = xs.mean(axis=0), ys.mean()
            scale = xs.std(axis=0) if standardize else np.ones(2)
            fit = sm.OLS(ys - my, (xs - mx) / scale).fit_regularized(alpha=ridge, L1_wt=0.0)
            want = my + (x[t] - mx) @ (np.asarray(fit.params) / scale)
            worst = max(worst, abs(pred[t] - want))
            shrunk = max(shrunk, abs(want - _own_rows_pred(x, y, t, float("inf"))))
        assert worst <= 1e-9 * 11.0, f"worst |pred - statsmodels| = {worst:.3e}"
        assert shrunk > 1e-3, f"the penalty should move the fit: {shrunk:.3e}"

    @pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
    def test_lasso_on_its_own_rows_is_statsmodels_elastic_net(self, l1_ratio):
        """A penalized path point from a second library. ``statsmodels``'
        elastic net minimizes ``RSS / (2n) + alpha * ((1 - L1_wt) / 2 * |b|^2
        + L1_wt * |b|_1)``, which is the ``lasso`` model's objective on the
        target's rows standardized by their own spread, with ``alpha`` the
        path's penalty and ``L1_wt`` its ``l1_ratio``. A third feature is
        noise, so the penalty zeroes a coefficient as well as shrinking the
        others. No decay, and both coordinate descents run to convergence."""
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(31)
        n, lam = 500, 0.1
        x = rng.normal(0.0, 1.0, (n, 3))
        y = 5.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.3, n)
        y = np.where(x[:, 0] > -0.5, y, np.nan)
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            lasso_path=[lam],
            l1_ratio=l1_ratio,
            halflife=float("inf"),
            max_rows_between_solves=1,
            max_cd_iters=100_000,
            cd_tol=1e-15,
        )
        frame = pl.DataFrame(
            {
                "x0": x[:, 0],
                "x1": x[:, 1],
                "x2": x[:, 2],
                "y": [None if np.isnan(v) else float(v) for v in y],
            },
            schema={c: pl.Float64 for c in ("x0", "x1", "x2", "y")},
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        got = bank.coef("m")["coef"].to_numpy()
        keep = ~np.isnan(y)
        xs, ys = x[keep], y[keep]
        mx, my, sd = xs.mean(axis=0), ys.mean(), xs.std(axis=0)
        fit = sm.OLS(ys - my, (xs - mx) / sd).fit_regularized(
            method="elastic_net", alpha=lam, L1_wt=l1_ratio, maxiter=10_000, cnvrg_tol=1e-14
        )
        b = np.asarray(fit.params) / sd
        want = np.r_[my - mx @ b, b]
        assert got == pytest.approx(want, abs=1e-7), np.max(np.abs(got - want))
        if l1_ratio == 1.0:
            assert got[3] == 0.0 and want[3] == 0.0, "the noise feature is out of the fit"
