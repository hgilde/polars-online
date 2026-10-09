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
**less than** ``window`` -- a row exactly ``window`` old has left, under the
default ``closed="right"``, as Polars' ``rolling_*_by`` and
``crates/online-core/src/window.rs`` state the boundary (docs/PLAN.md task
196) -- each at weight ``0.5 ** (age / half_life)``. ``closed="both"`` is
held to its definition in ``tests/test_window.py``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: A level that crosses zero (docs/PLAN.md task 209 (d)): the oracle tests
#: that take a ``level`` take this too, a level running from +1,000 to
#: -1,000 in equal steps over the rows.
CROSSING = "crossing"


def _level(level: float | str, n: int) -> Any:
    """``level`` itself, or ``CROSSING``'s levels over ``n`` rows."""
    return np.linspace(1000.0, -1000.0, n) if level == CROSSING else level


def _magnitude(level: float | str) -> float:
    """``1 +`` the largest ``|level|``: what an absolute tolerance scales by,
    the numbers carrying the level's rounding."""
    return 1001.0 if level == CROSSING else 1.0 + abs(float(level))


def _window(
    ages: np.ndarray, half_life: float, window: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """The rows a window keeps, under the default ``closed="right"``, and
    their weights."""
    keep = np.ones(ages.shape, bool) if window is None else ages < window
    return keep, 0.5 ** (ages[keep] / half_life)


class TestWindowAtALargeOffset:
    """C17, the review's T-S10 (i). A window's moments are a weighted mean and
    a population covariance of the rows inside it, which ``numpy.average`` and
    ``numpy.cov(aweights=..., ddof=0)`` compute two-pass -- centre, then square
    -- and so correctly at any offset. The truncation used to go back through
    ``E[x x'] = C + m m'``, subtract and re-centre, which at a level of ``1e8``
    left the variance with a resolution of about 2. The unwindowed
    accumulator (Welford) never had the defect, which makes ``window_size=None``
    the negative control in every test here.

    Spreads are unit-scale, so at ``1e8`` an absolute error of ``1e-6``
    separates the fix (near ``1e-9``, the live accumulator's own error at that
    level) from the defect (errors of order 1)."""

    HALFLIFE = 40.0

    @pytest.mark.parametrize("window", [None, 25.0, 60.0])
    @pytest.mark.parametrize("offset", [0.0, 1e8, -1e8])
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
            half_life=self.HALFLIFE,
            min_weight=0,
            window_size=window,
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
            assert out["weight_sum"][i] == pytest.approx(w.sum(), rel=1e-10)
            np.testing.assert_allclose(
                [out["mean_x0"][i], out["mean_x1"][i]], mean, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(got_cov, cov, rtol=0.0, atol=tol * np.abs(cov).max())

    @pytest.mark.parametrize("window", [None, 70.0])
    @pytest.mark.parametrize("offset", [0.0, 1e8, -1e8])
    def test_a_marginal_pair_is_numpy_of_the_rows_inside_the_window(self, offset, window):
        rng = np.random.default_rng(11)
        n, half_life = 400, 25.0
        u = rng.normal(0.0, 1.0, n)
        x = offset + u
        y = offset + 2.0 * u + rng.normal(0.0, 0.2, n)
        spec = po.spec.marginal(
            "m", features=["x"], targets=["y"], half_life=half_life, window_size=window
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        pair = bank.marginal("m").row(0, named=True)
        # A readout is referenced at the last learned row, which is inside.
        keep, w = _window((n - 1) - np.arange(n), half_life, window)
        xs, ys = x[keep], y[keep]
        cov = np.cov(np.vstack([xs, ys]), aweights=w, ddof=0)
        tol = (1e-12 if offset == 0.0 else 1e-6) * np.abs(cov).max()
        assert pair["weight_sum"] == pytest.approx(w.sum(), rel=1e-10)
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
    inside the window, each at ``0.5 ** (age / half_life)`` -- ``numpy``'s
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
        n, half_life = 400, 25.0
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
            half_life=half_life,
            ridge=1e-10,
            standardize=standardize,
            window_size=window,
            solve_every=1e-9,  # solve on every row, so every pred is comparable
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        for i in range(260, n, 9):
            # Out of sample: the fit on the rows before row i predicts row i.
            keep, w = _window((i - 1) - np.arange(i), half_life, window)
            b = _wls(x[:i][keep], y[:i][keep], w)
            assert pred[i] == pytest.approx(b[0] + x[i] @ b[1:], abs=1e-8)


class TestTheWindowIsAllTheModelSees:
    """Pattern D and C2. Under a ``window`` every model reports, gates and fits
    on the weight *inside* it -- ``Σ 0.5 ** (age / half_life)`` over the rows
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
        common = {"half_life": self.HALFLIFE, "window_size": window}
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
        weight_sum = (
            po.ModelBank([self._spec(kind, window)]).fit_predict(df)["m"].struct.field("weight_sum")
        )
        for i in range(100, n, 13):
            # The weight_sum a row reports is the weight before it, with ages
            # counted from the row before.
            _, w = _window((i - 1) - np.arange(i), self.HALFLIFE, window)
            assert weight_sum[i] == pytest.approx(w.sum(), rel=1e-10), (kind, i)

    @pytest.mark.parametrize("kind", ["ewridge", "lasso"])
    def test_a_target_that_leaves_the_window_does_not_unwindow_the_others(self, kind):
        x, y0, y1, _ = self._rows()
        n = len(y0)
        # y1 is observed on the first 100 rows only; its last row is out of the
        # window from row 161 on, and from then on it has nothing to fit.
        y1_sparse = [float(v) if i < 100 else None for i, v in enumerate(y1)]
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y0, "y1": y1_sparse})
        feats, common = ["x0", "x1"], {"half_life": self.HALFLIFE, "window_size": self.WINDOW}
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
    """S1: under a ``window`` the bank's ``sigma`` and ``zscore`` are the
    window's -- the EW root mean square of the out-of-sample residuals
    inside it, ``sqrt(Σ λ^age r² / Σ λ^age)``, over the rows the fit is read
    from. They were the stream's own EW mean over the whole history, so a
    burst of errors the window had dropped still widened ``sigma`` for as
    long as the half-life remembered it -- the spread of a history the fit no
    longer sees. ``numpy`` over the residuals the bank itself emitted is the
    second opinion (review 2026-09-12, S1; the user's decision of
    2026-09-15)."""

    H, W = 40.0, 60.0

    def _frame(self, n=500, seed=21):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.1, n)
        # A burst of large errors: the window drops it at row 320 + W, long
        # before a half-life of 40 forgets it.
        y[300:320] += rng.normal(0.0, 5.0, 20)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})

    def _spec(self, kind, **kw):
        common = dict(
            targets=["y"],
            features=["x0", "x1"],
            half_life=self.H,
            window_size=self.W,
            solve_every=1e-9,
            emit_sigma=True,
            emit_zscore=True,
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
        z = self._field(out, spec, "zscore_y")
        ok = np.isfinite(resid)
        checked = 0
        for i in range(1, len(resid)):
            # Row i is scored against the rows learned before it, inside the
            # window as it stood at row i - 1: the fit's own boundary.
            ages = (i - 1) - np.arange(i)
            keep = (ages < self.W) & ok[:i]
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
        cols = ["pred_y", "resid_y", "sigma_y", "zscore_y", "weight_sum"]
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
            half_life=500.0,
            window_size=2000.0,
            ridge=[10.0 ** (-k / 20) for k in range(200)],
            emit_sigma=True,
            window_budget={"refuse": 1.0},
        )
        with pytest.raises(ValueError, match="window_budget"):
            po.ModelBank([spec]).fit_predict(df)


class TestWindowedGaussian:
    """C12 and C14, the review's T-S6 and T-S8. At ``half_life = inf`` every row
    inside the window weighs 1 and the precision prior does not fade, so a
    windowed ``ew_class`` and ``ew_cov`` are plain Gaussian computations on the
    rows the window keeps: ``scipy.stats.multivariate_normal`` for the class
    densities, ``scipy.spatial.distance.mahalanobis`` for the distance, and
    ``numpy.linalg.eigh`` for the components -- each fed the in-window mean
    and population covariance from ``numpy``.

    C12: the ``full`` shape factorized the *live* class covariance while
    scoring against the windowed mean. C14: ``mahal`` and the PCA refresh read
    the live accumulator. ``diagonal`` and ``shared`` (which read the view)
    and ``window_size=None`` are the controls. The covariance changes halfway, so
    the whole history and the window disagree by construction."""

    WINDOW, PRIOR = 80.0, 1e-9

    @pytest.mark.parametrize("window", [None, 80.0])
    @pytest.mark.parametrize("shape", ["full", "diagonal", "shared"])
    def test_ew_class_is_the_scipy_gaussian_of_the_rows_inside_the_window(self, shape, window):
        """And ``coef``, the class means, is ``numpy``'s mean of each class's
        rows inside the window, on the row before: the whole history's
        means stood there beside the windowed densities (review round 4,
        CE1)."""
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
            half_life=float("inf"),
            window_size=window,
            min_weight=0,
            coef_every=0,
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
            # `coef` after row i - 1 is the class means the row i was scored on.
            np.testing.assert_allclose(
                out["coef"][i - 1].to_list(), np.concatenate(means), rtol=0, atol=1e-9
            )

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
            half_life=float("inf"),
            window_size=window,
            min_weight=0,
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
    own, which counted ``lag_corr`` as ``k(k-1)/2`` slots instead of
    ``len(lags)·k²`` -- so beside a ``lag_corr`` the quantile estimators were
    fed a lagged correlation. The reference is ``numpy.quantile`` of the
    ``mahal`` column the bank itself emitted. Statistical tier: the P²
    estimator is approximate, so 10% -- the defect puts it at a quantile of a
    correlation, in ``[-1, 1]``, against a distance near 1.2."""

    @pytest.mark.parametrize("stats", [["mahal"], ["lag_corr", "mahal"]])
    def test_the_quantile_tracks_the_mahal_column(self, stats):
        rng = np.random.default_rng(23)
        n = 3000
        x = rng.normal(0.0, 1.0, (n, 2))
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            stats=stats,
            lags=[1] if "lag_corr" in stats else None,
            precision_prior=1e-6,
            mahal_quantiles=[0.5, 0.9],
            half_life=float("inf"),
            min_weight=5,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]}))
        out = out["m"].struct.unnest()
        mahal = out["mahal"].drop_nulls().drop_nans().to_numpy()
        assert out["mahal_q0.5"][-1] == pytest.approx(np.quantile(mahal[:-1], 0.5), rel=0.1)
        assert out["mahal_q0.9"][-1] == pytest.approx(np.quantile(mahal[:-1], 0.9), rel=0.1)


class TestSessionShrinkBlend:
    """C16 and C3, the review's T-S10 (ii), and task 145. A ``session_shrink``
    blend at ``f`` mixes the fast accumulators (``half_life``) with their slow
    twin (``long_half_life``), which saw the *same* rows, as two data sets: ``1
    - f`` of today's and ``f`` of the long run's, at today's weight. So the
    blend is itself one weighted accumulator, each row at ``W_h · ((1 - f)
    λ_h^age / W_h + f λ_H^age / W_H)``, each kernel normalised by its weight.
    ``numpy.average`` and ``numpy.cov(aweights=..., ddof=0)`` with that weight
    vector are the blended moments exactly, two-pass and so right at any
    offset.

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
            half_life=self.H_FAST,
            session="s",
            session_gap=1.0,
            session_shrink=f,
            long_half_life=self.H_SLOW,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "s": session}))
        g = bank.gram("m")[0]
        # Rows before the last: blended at their ages, then aged one step at
        # the fast rate by the last row's `session_gap`. The last row: weight 1.
        age = (n - 2) - np.arange(n - 1)
        lam_f, lam_s = 0.5 ** (1.0 / self.H_FAST), 0.5 ** (1.0 / self.H_SLOW)
        kf, ks = lam_f**age, lam_s**age
        blended = kf.sum() * ((1.0 - f) * kf / kf.sum() + f * ks / ks.sum())
        w = np.append(lam_f * blended, 1.0)
        cols = [g["columns"].index("x0"), g["columns"].index("x1")]
        cov = np.cov(x.T, aweights=w, ddof=0)
        got = np.asarray(g["comoments"])[np.ix_(cols, cols)]
        tol = (1e-12 if offset == 0.0 else 1e-6) * np.abs(cov).max()
        assert g["weight_sum"] == pytest.approx(w.sum(), rel=1e-12)
        np.testing.assert_allclose(
            np.asarray(g["means"])[cols], np.average(x, axis=0, weights=w), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(got, cov, rtol=0.0, atol=tol)
        ybar = np.average(y, weights=w)
        yvar = np.average((y - ybar) ** 2, weights=w)
        assert g["target_means"][0] == pytest.approx(ybar, rel=1e-12, abs=1e-12)
        assert abs(g["target_vars"][0] - yvar) <= (1e-12 if offset == 0.0 else 1e-6) * yvar

    @pytest.mark.parametrize("long_half_life", [2000.0, float("inf")])
    def test_the_first_row_of_a_session_is_predicted_from_the_blend(self, long_half_life):
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
            half_life=100.0,
            session="s",
            session_gap=1.0,
            session_shrink=1.0,  # the blend is the slow twin itself
            long_half_life=long_half_life,
        )
        pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y")
        # The slow twin's fit: every row of session 1 at 0.5 ** (age / H). At
        # H = inf the weights are numpy's unit ones: the long run is the whole
        # history, which the builder refused (review 2026-09-12, S27).
        age = (n1 - 1) - np.arange(n1)
        b = _wls(x[:n1, None], y[:n1], 0.5 ** (age / long_half_life))
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
    (``sgd``). With ``fit_intercept=False`` and standardization on, these
    models centred the features anyway -- ``ewridge`` alone scaled by the raw
    second moment -- so each solved a system that is neither the centred
    problem (which needs an intercept) nor the raw one, and ``coef`` could not
    reproduce ``pred``. Without an intercept there is nothing to centre on:
    the reference is ``numpy.linalg.lstsq`` on the raw features, no constant
    column. ``fit_intercept=True`` (and, where the model has it,
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
                half_life=float("inf"),
                solve_every=1e-9,
                max_iter=100_000,
                tol=1e-14,
                fit_intercept=intercept,
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
            half_life=float("inf"),
            solve_every=1e-9,
            fit_intercept=False,
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
        # noise the filter is recursive least squares, and after 20,000 rows
        # it sits at numpy's no-intercept fit. Not exactly: over the
        # standardizer's warm-up its first 22 rows were learned in coordinates
        # read again as the scales settled, and the filter, which forgets
        # nothing here, keeps them as about 22 rows of a prior that is off,
        # gone as 1/n. Measured 3.3e-3 here (1.3e-3 at 50,000 rows); at 5,000
        # rows 1.25e-2, where the build before task 206, reading its state
        # through the moments as they stood, missed by 7.3e-3.
        x, y = _no_intercept_rows(20_000, seed=43)
        spec = po.spec.kalman(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            q=[0.0, 0.0],
            p0=1e6,
            obs_var=0.01,
            half_life=float("inf"),
            coef_every=0,
            fit_intercept=False,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        coef = np.array(out["coef"].to_list(), dtype=float)
        pred = out["pred_y"].to_numpy()
        np.testing.assert_allclose(np.sum(coef[99:-1] * x[100:], axis=1), pred[100:], rtol=1e-9)
        b = _wls(x, y, np.ones(len(y)), intercept=False)
        np.testing.assert_allclose(coef[-1], b, atol=1e-2)

    def test_sgd_converges_to_numpy_least_squares_and_its_coefficients_follow_it(self):
        # C13. sgd standardizes a row against moments that include it. Past
        # its scaler's warm-up it holds `coef` in the caller's units, the
        # scaler shaping the step only (docs/PLAN.md task 206), so `coef`
        # after the row before reproduces a prediction to rounding. Read
        # through the scaler as it stood, it missed by a scaler step, about
        # 1% here (the review's D5); centring without an intercept put the
        # two thousands of times apart. The library check is the statistical
        # one: the fit settles at numpy's no-intercept least squares.
        x, y = _no_intercept_rows(20_000, seed=44)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            standardize=True,
            learning_rate=0.01,
            half_life=1e6,
            coef_every=0,
            fit_intercept=False,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        coef = np.array(out["coef"].to_list(), dtype=float)
        pred = out["pred_y"].to_numpy()
        via_coef = np.sum(coef[999:-1] * x[1000:], axis=1)
        assert np.max(np.abs(via_coef - pred[1000:]) / np.abs(pred[1000:])) < 1e-13
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
    flag when nothing decays, so fed the bank's own ``|resid| / sigma``
    series with the zero-weight row left out, it must reproduce the bank's
    ``drift`` column. Under a half-life it is not: ours keeps the error's
    mean at the model's half-life and river keeps a plain mean, and with a
    level shift of 3 the two fire a row apart.

    The target's level shifts by ten noise standard deviations at row 900,
    after the zero-weight row, so the detector fires in every leg. Without
    it no leg ever fired, and the reset leg and river's comparison checked
    nothing (review 2026-10-05, TA2)."""

    SHIFT_AT = 900

    def _run(self, jump: float, drift_action: str = "flag", half_life: float = 200.0):
        rng = np.random.default_rng(51)
        n, k0 = 1200, 700
        x = rng.normal(0.0, 1.0, n)
        y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, n)
        y[self.SHIFT_AT :] += 5.0
        w = np.ones(n)
        w[k0] = 0.0
        y[k0] += jump
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            half_life=half_life,
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
        fired = np.flatnonzero(jumped["drift_y"].fill_null(False).to_numpy())
        assert len(fired) >= 1, "the detector never fired, so the legs compared nothing"
        assert (fired > k0).all() and (fired >= self.SHIFT_AT).any(), fired
        if drift_action == "reset":
            # The model starts over after the row that fired: the next row
            # sees no accumulated weight.
            ws = jumped["weight_sum"].to_numpy()
            for t in fired:
                assert ws[t] > 100.0 and ws[t + 1] == 0.0, (t, ws[t], ws[t + 1])

    def test_the_drift_column_is_rivers_page_hinkley_without_it(self):
        import river.drift as river_drift

        out, _, w = self._run(100.0, half_life=float("inf"))
        resid = out["resid_y"].to_numpy()
        sigma = out["sigma_y"].to_numpy()
        flags = out["drift_y"].to_numpy()
        ph = river_drift.PageHinkley(
            min_instances=0, delta=0.05, threshold=20.0, alpha=1.0, mode="up"
        )
        fed, rivers = 0, []
        for t in range(len(resid)):
            e = abs(resid[t]) / sigma[t] if sigma[t] > 0 else float("nan")
            if w[t] > 0 and np.isfinite(e):
                ph.update(float(e))
                fed += 1
                assert bool(flags[t]) == bool(ph.drift_detected), t
                if ph.drift_detected:
                    rivers.append(t)
            else:
                assert not flags[t], t
        assert fed > 1000
        # River fired, on the bank's rows, and again after re-arming.
        assert rivers == np.flatnonzero(flags).tolist()
        assert any(t >= self.SHIFT_AT for t in rivers), rivers


class TestLabelDelayFoldsWhatWasScored:
    """C21 and C5, the review's T-S16. Under ``embargo`` a row is scored
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
    group, which ``embargo`` refuses -- but the queue C21 keeps must stay
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
            half_life=float("inf"),
            solve_every=1e-9,
            min_weight=1,
            emit_sigma=True,
            embargo=float(self.DELAY),
            **kw,
        )

    def test_the_predictions_are_rivers_delayed_mean_and_sigma_is_their_residuals(self):
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

    def test_r2_is_sklearns_over_the_matured_rows(self):
        """C21's other half: ``r2_y`` folds the prediction the row was scored
        with, released with its label, so at row ``t`` -- read before the row,
        under no forgetting -- it is scikit-learn's ``r2_score`` of the scored
        predictions over every row at least ``delay`` behind (docs/PLAN.md
        task 112)."""
        from sklearn.metrics import r2_score

        y = self._rows()
        n = len(y)
        df = pl.DataFrame({"zero": np.zeros(n), "y": y})
        out = po.ModelBank([self._spec(emit_metrics=True)]).fit_predict(df)["m"].struct.unnest()
        pred, r2 = out["pred_y"].to_numpy(), out["r2_y"].to_numpy()
        checked = 0
        for t in range(3 * self.DELAY, n, 7):
            rows = np.arange(t - self.DELAY + 1)
            rows = rows[np.isfinite(pred[rows])]
            want = r2_score(y[rows], pred[rows])
            assert r2[t] == pytest.approx(want, rel=1e-9, abs=1e-12), t
            checked += 1
        assert checked > 50

    def test_coverage_is_the_share_of_matured_rows_inside_their_interval(self):
        """And ``coverage_y``, with ``conformal``: at row ``t`` it is the share of
        the matured rows whose target fell inside the interval the frame
        showed for them, ``[lo_y, hi_y]`` as scored, not as the replay would
        have drawn it (C21; docs/PLAN.md task 112)."""
        y = self._rows()
        n = len(y)
        df = pl.DataFrame({"zero": np.zeros(n), "y": y})
        out = po.ModelBank([self._spec(conformal=0.9)]).fit_predict(df)["m"].struct.unnest()
        lo, hi, cov = (out[c].to_numpy() for c in ("lo_y", "hi_y", "coverage_y"))
        inside = (lo <= y) & (y <= hi)
        checked = 0
        for t in range(3 * self.DELAY, n, 7):
            rows = np.arange(t - self.DELAY + 1)
            rows = rows[np.isfinite(lo[rows]) & np.isfinite(hi[rows])]
            if len(rows) == 0:
                continue
            assert cov[t] == pytest.approx(inside[rows].mean(), rel=1e-9, abs=1e-12), t
            checked += 1
        assert checked > 50

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
            half_life=float("inf"),
            session="s",
            session_gap="reset",
            embargo=5.0,
            min_weight=2,
        )
        whole = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        fresh = po.ModelBank([spec]).fit_predict(df[n1:])["m"].struct.unnest()
        tail = whole[n1:]
        for col in ("pred_y", "weight_sum"):
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
            half_life=float("inf"),
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
    ``0.5 ** (age / half_life)``: both ``coef()`` and the solve of the Gram must
    land on it. ``window_size=None`` is the control, where the two histories are
    the same one. The target moments are the window's too (task 136): the
    weighted mean and variance of the in-window targets are ``statsmodels``'
    ``DescrStatsW``, and Kish's count ``(Σw)² / Σw²``. Under a window they
    were ``None``, since the snapshots held none."""

    @pytest.mark.parametrize("window", [None, 60.0])
    def test_the_gram_solves_to_the_fit_the_bank_reports(self, window):
        rng = np.random.default_rng(81)
        n, half_life = 400, 25.0
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
            half_life=half_life,
            ridge=1e-10,
            window_size=window,
            solve_every=1e-9,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        # The last fit was solved after the last row: ages count from it.
        keep, w = _window((n - 1) - np.arange(n), half_life, window)
        want = _wls(x[keep], y[keep], w)
        coef = bank.coef("m")["coef"].to_numpy()
        np.testing.assert_allclose(coef, want, atol=1e-6)
        g = bank.gram("m")[0]
        assert g["weight_sum"] == pytest.approx(w.sum(), rel=1e-10)
        np.testing.assert_allclose(po.gram.solve(g, ridge=1e-10), want, atol=1e-6)
        from statsmodels.stats.weightstats import DescrStatsW

        moments = DescrStatsW(y[keep], weights=w)
        assert g["target_means"] == pytest.approx([moments.mean], rel=1e-9)
        assert g["target_vars"] == pytest.approx([moments.var], rel=1e-8)
        assert g["target_n_kish"] == pytest.approx([w.sum() ** 2 / (w**2).sum()], rel=1e-9)


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

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8, -1e8])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_a_level_regressed_on_levels_is_the_numpy_fit(self, standardize, offset):
        rng = np.random.default_rng(97)
        n, half_life = 600, 200.0
        u = rng.normal(0.0, 1.0, (n, 2))
        x = offset + u
        # y = 2·x0 − x1 + noise: the target sits at the level too.
        y = offset + 2.0 * u[:, 0] - u[:, 1] + rng.normal(0.0, 0.1, n)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=half_life,
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        pred = po.ModelBank([spec]).fit_predict(frame)["m"].struct.field("pred_y").to_numpy()
        worst = 0.0
        for t in range(100, n, 50):
            # Row t is predicted by the fit of the rows before it.
            w = 0.5 ** (((t - 1) - np.arange(t)) / half_life)
            xm = w @ x[:t] / w.sum()
            ym = w @ y[:t] / w.sum()
            slopes = _wls(x[:t] - xm, y[:t] - ym, w, intercept=False)
            worst = max(worst, abs(pred[t] - (ym + (x[t] - xm) @ slopes)))
        assert worst <= 1e-10 + 1e-14 * abs(offset), f"worst |pred - numpy| = {worst:.3e}"


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
            half_life=float("inf"),
            ridge=0.0,
            standardize=standardize,
            solve_every=1e-9,
            **kw,
        )

    @staticmethod
    def _tol(offset: float) -> float:
        return 1e-10 + 1e-14 * abs(offset)

    @staticmethod
    def _same_fit(got, want, x, tol, what):
        """Two fits agree where a fit at a level is resolved: the slopes, and
        the predictions they give at the rows. An intercept at a level is
        not: it is ``ybar - m @ b``, so a slope known to ``δ`` moves it by
        ``L·δ``, ``L²·ε`` in all -- in numpy's fit as in any other."""
        assert np.max(np.abs(got[1:] - want[1:])) <= tol, (what, "slopes", got, want)
        gap = np.max(np.abs((got[0] + x @ got[1:]) - (want[0] + x @ want[1:])))
        assert gap <= tol, (what, "predictions", gap)

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8, -1e8])
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

    @pytest.mark.parametrize("offset", [0.0, 1e8, -1e8])
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
        rel = 1e-9 + 1e-12 * abs(offset)
        assert stats["resid_var"] == pytest.approx(np.mean(resid**2), rel=rel, abs=1e-12)


def _holt(
    y: np.ndarray, half_life: float, trend_half_life: float
) -> tuple[np.ndarray, float, float]:
    """``holt``'s ``pred`` on a row-count clock, and its final level and
    trend. ``NaN`` in ``y`` is a null target."""
    spec = po.spec.holt(
        "h", targets=["y"], half_life=half_life, trend_half_life=trend_half_life, min_weight=0.0
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
    row-count clock -- the control that pins the mapping from half-lives to
    smoothing weights (the review's T-S14). Its state-space
    ``ExponentialSmoothing`` takes ``NaN`` in the
    series and answers it with the prediction step alone, so the row after a
    missing one forecasts ``l + 2b`` from the ``l`` and ``b`` that stood
    before it, a number the gain does not enter (T-S17). Past that row the two
    part by design: its gain is fixed, and ours grows with the clock since the
    last observation, which is what a half-life in clock units means."""

    H_LEVEL, H_TREND = 6.0, 25.0
    #: The row from which the weighted means' gains equal the fixed ones to
    #: the last digit this test reads: ``0.5 ** (1000 / H_TREND)`` is 1e-12.
    SETTLED = 1000

    @staticmethod
    def series(n: int, drift: float | str = 0.0) -> np.ndarray:
        """A rising trend, or with ``drift`` its walk about a level running
        from +1,000 to -1,000 (`CROSSING`)."""
        rng = np.random.default_rng(5)
        rising = 3.0 + 0.4 * np.arange(n) + np.cumsum(rng.normal(0.0, 0.5, n))
        return rising if drift == 0.0 else _level(drift, n) + rising - 0.4 * np.arange(n)

    @pytest.mark.parametrize("drift", [0.0, CROSSING])
    def test_the_recursion_is_statsmodels_holt(self, drift):
        """And on a series that falls through zero from +1,000 to -1,000,
        200 rows after the gains have settled (docs/PLAN.md task 209 (d))."""
        import statsmodels.tsa.holtwinters as holtwinters

        n = self.SETTLED + (1400 if drift == CROSSING else 200)
        y = self.series(n, drift)
        if drift == CROSSING:
            assert y[self.SETTLED :].min() < 0 < y[self.SETTLED :].max(), "the case: it crosses"
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
        settled = slice(self.SETTLED, None)
        atol = 1e-12 * _magnitude(drift)
        np.testing.assert_allclose(pred[settled], fitted[settled], rtol=1e-12, atol=atol)
        assert level == pytest.approx(res.level[-1], rel=1e-12)
        assert trend == pytest.approx(res.trend[-1], rel=1e-12)

    def test_the_row_after_a_missing_one_forecasts_two_trend_steps(self):
        import statsmodels.tsa.statespace.exponential_smoothing as es

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
            half_life=self.H_LEVEL,
            trend_half_life=self.H_TREND,
            clock="t",
            gap_cap=10.0,
            weight="w",
            min_weight=0.0,
        )
        got = po.ModelBank([spec]).fit_predict(full)["h"].struct.field("pred_y").to_numpy()
        # The same stream with those rows removed: the clock folds their
        # deltas into the next row's.
        kept = full.filter(pl.Series(~skip))
        want = po.ModelBank([spec]).fit_predict(kept)["h"].struct.field("pred_y").to_numpy()
        np.testing.assert_allclose(got[~skip], want, rtol=1e-12)


class TestALevelOnlyHoltIsAnEwMean:
    """``holt(trend=False)`` holds the trend at zero (docs/PLAN.md task 115,
    S30), so the forecast is flat and the level is the exponentially weighted
    mean of the target's observations. pandas' ``ewm(times=, adjust=True)``
    computes that mean on an irregular clock, passing a null by its time as
    ``holt`` does; statsmodels' ``DescrStatsW`` computes the weighted one,
    each observation at its weight times ``0.5 ** (age / half_life)``. The
    row's prediction is the mean through the row before it. Measured:
    ``5.6e-16`` against ``DescrStatsW`` and ``2.2e-12`` against pandas, whose
    time clock carries an error of its own; with the trend on, the forecasts
    part from the mean by up to 1.2."""

    HALFLIFE = 7.0

    @staticmethod
    def rows(weighted, n=300, seed=21):
        rng = np.random.default_rng(seed)
        t = np.cumsum(rng.uniform(0.2, 3.0, n))
        y = 5.0 + 0.02 * t + rng.normal(0.0, 1.0, n)
        y[rng.random(n) < 0.1] = np.nan
        w = rng.uniform(0.0, 2.0, n) if weighted else np.ones(n)
        return t, y, w

    def fit(self, t, y, w, **kw):
        frame = pl.DataFrame({"t": t, "y": y, "w": w}).with_columns(pl.col("y").fill_nan(None))
        spec = po.spec.holt(
            "m",
            targets=["y"],
            clock="t",
            gap_cap=1e9,
            half_life=self.HALFLIFE,
            min_weight=0.0,
            weight="w",
            coef_every=0,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct
        return out.field("pred_y").to_numpy(), np.array(out.field("coef").to_list(), float)

    def test_the_level_is_pandas_ewm(self):
        import pandas as pd

        t, y, w = self.rows(weighted=False)
        pred, coef = self.fit(t, y, w, trend=False)
        times = pd.to_datetime(np.round(t * 1e9).astype("int64"), unit="ns")
        mean = (
            pd.Series(y)
            .ewm(halflife=pd.Timedelta(self.HALFLIFE, "s"), times=times, adjust=True)
            .mean()
            .to_numpy()
        )
        # pandas' own error on a time clock is about 1e-9 (docs/TESTING.md).
        np.testing.assert_allclose(pred[1:], mean[:-1], rtol=1e-8)
        assert np.isnan(pred[0])
        assert (coef[1:, 1] == 0.0).all(), "the trend is held at zero"
        # The control: with the trend on, the same rows forecast elsewhere.
        with_trend, _ = self.fit(t, y, w)
        assert np.nanmax(np.abs(with_trend[1:] - mean[:-1])) > 0.1

    def test_a_weighted_level_is_descrstatsw(self):
        from statsmodels.stats.weightstats import DescrStatsW

        t, y, w = self.rows(weighted=True)
        pred, _ = self.fit(t, y, w, trend=False)
        seen = ~np.isnan(y) & (w > 0.0)
        for i in range(1, len(y)):
            before = seen[:i]
            if not before.any():
                assert np.isnan(pred[i])
                continue
            age = t[i - 1] - t[:i][before]
            stats = DescrStatsW(y[:i][before], weights=w[:i][before] * 0.5 ** (age / self.HALFLIFE))
            assert pred[i] == pytest.approx(stats.mean, rel=1e-12), i


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

    @pytest.mark.parametrize("offset", [0.0, 1e6, 1e8, -1e8])
    def test_the_run_length_posterior_is_the_packages(self, offset):
        import bayesian_changepoint_detection.online_changepoint_detection as bcd

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
            min_weight=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x}))["b"]
        r, maxes = bcd.online_changepoint_detection(
            x,
            functools.partial(bcd.constant_hazard, hazard),
            bcd.StudentT(nu0 / 2.0, psi0 / 2.0, kappa0, offset),
        )
        tol = 1e-9 + 1e-14 * abs(offset)
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
    · d / coef_half_life)²`` of a step ``d`` clock units long -- decayed on a
    row whose target was null and not on
    one whose target was present at weight zero. The filter treats the two
    alike, a prediction step and no update, so ``σ²`` remembered more across
    one than the other.

    The second opinion is ``filterpy``'s ``KalmanFilter`` beside a ``numpy``
    recursion for ``σ²``: ``predict`` on every row, its process noise only
    on a row that observes the target and for the whole clock since the last
    one (docs/PLAN.md task 211), ``update(y, R = σ²/w,
    H = z)`` only where there is a target and a positive weight, and ``σ²``
    the EW mean of the squared out-of-sample residuals with its weight
    decayed on every row. Until ``σ²`` exists, the row's innovation squared
    stands in for it, and ``P`` is unsized until the first row with a noise
    sets it to ``p0 = 1`` times that noise (review 2026-10-05, CC4).
    Unstandardized, so ``kf.x``
    is our coefficient vector; ``filterpy`` updates ``P`` in Joseph form and
    ``kalman`` in the simple form, so the two agree to rounding. The same
    rows with the target null instead of the weight zero are the control."""

    @staticmethod
    def filterpy_pred(
        kalman: Any,
        t: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        half_life: float,
        coef_hl: float,
        F: list[np.ndarray] | None = None,
        usable: np.ndarray | None = None,
    ) -> np.ndarray:
        """filterpy's predictions over the regressor rows ``Z``, the
        intercept's column first: ``[1, x]`` unstandardized, the rows
        standardized otherwise, each row's predict step taking ``F[i]`` as
        its transition where ``F`` is given (the identity otherwise) and the
        process noise of the clock since the last observation on a row that
        observes the target, ``sigma^2 * (ln 2 * D / coef_half_life)^2``, as
        the docstring states it (task 211). The noise is ``sigma^2``, and
        before there is one, the row's own innovation squared, before the
        update; none where that is 0 or the row has no innovation: no process
        noise and no update (review 2026-10-05, CC4). Unstandardized the
        prior is ``p0 = 1`` times the first noise; standardized each slot's
        waits for its scale (``usable``, `_filterpy_run`)."""
        return _filterpy_run(
            kalman, t, Z, y, w, half_life, coef_half_life=coef_hl, F=F, usable=usable
        )[0]

    @pytest.mark.parametrize("how", ["null", "zero weight"])
    def test_the_filter_is_filterpy_with_sigma_decayed_on_every_row(self, how):
        import filterpy.kalman as kalman

        rng = np.random.default_rng(41)
        n, half_life, coef_hl = 300, 30.0, 50.0
        x = rng.normal(0.0, 1.0, (n, 2))
        drift = np.arange(n) / n
        y = 0.5 + (1.5 - drift) * x[:, 0] + (-0.8 + 2.0 * drift) * x[:, 1]
        y = y + rng.normal(0.0, 0.3, n)
        skip = (np.arange(n) % 9 == 4) & (np.arange(n) > 20)
        w = np.where(skip & (how == "zero weight"), 0.0, 1.0)
        y_seen = np.where(skip & (how == "null"), np.nan, y)
        # Steps of 1, one gap of 4 and then half steps, so the process noise's
        # d**2 is not d (review 2026-10-05, TA5).
        steps = np.ones(n)
        steps[150] = 4.0
        steps[230:] = 0.5
        t = np.cumsum(steps) - steps[0]
        frame = pl.DataFrame(
            {
                "t": t,
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": [None if np.isnan(v) else float(v) for v in y_seen],
                "w": w,
            },
            schema={
                "t": pl.Float64,
                "x0": pl.Float64,
                "x1": pl.Float64,
                "y": pl.Float64,
                "w": pl.Float64,
            },
        )
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1e9,
            coef_half_life=coef_hl,
            standardize=False,
            p0=1.0,
            weight="w",
            half_life=half_life,
            min_weight=0.0,
        )
        got = po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("pred_y").to_numpy()
        Z = np.column_stack([np.ones(n), x])
        want = self.filterpy_pred(kalman, t, Z, y_seen, w, half_life, coef_hl)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


class TestAStandardizedKalmanIsFilterpy:
    """The review's TC1 (2026-10-05): the library second opinions held
    `kalman` only at ``standardize = False``, and the default is ``True``,
    which `kalman_ref` held by restating the core. Standardized, the filter
    runs on ``z = [1, (x - m) / s]``, with ``m`` and ``s`` the EW means and
    standard deviations of the features over the rows *before* this one:
    at the spec's half-life on the clock, every processed row at its weight
    (a row whose target is null included), and a scale of 1 where the
    variance is 0 (`kalman.rs`'s module doc; `EwDiag`). Standardized here in
    numpy, by sums over the raw rows, and fed to filterpy as
    `TestKalmanZeroWeightRow` feeds it the raw ones.

    From the first row the filter's state follows the moments to their new
    coordinates after every row that moves them (docs/PLAN.md tasks 206 and
    211): ``b <- A b`` and ``P <- A P A'``, with ``A_00 = 1``, ``A_0i =
    (m'_i - m_i) / s_i`` and ``A_ii = s'_i / s_i`` from the moments the row
    was read at to the ones after it. That is a Kalman predict step with
    ``F = A``, so filterpy takes it as its transition on the next row,
    beside the process noise. The filter holds its state at an anchor and
    re-maps only when the moments drift from it, the same filter to
    rounding. Each slot's prior waits for its scale, and is ``p0`` times the
    mean squared innovation over three observed rows (task 211).

    The coefficients are read out in the original units with the moments
    *after* the row, so ``coef`` after one row applied to the next row's
    features is the next row's prediction (docs/PLAN.md task 97): asserted
    here directly, where `tests/test_oracles_rls_kalman_paths.py` stated
    it."""

    @staticmethod
    def standardized(
        t: np.ndarray, x: np.ndarray, w: np.ndarray, half_life: float
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """The rows standardized against the moments before each, the
        transition each row's predict step takes, and which slots' scales
        each row reads are usable (`_standardizer_moves`)."""
        Z, F, _, usable = _standardizer_moves(t, x, w, half_life)
        return Z, F, usable

    @pytest.mark.parametrize("level", [0.0, 1e3, -1e3, CROSSING])
    @pytest.mark.parametrize("how", ["null", "zero weight"])
    def test_the_filter_is_filterpy_on_the_standardized_rows(self, how, level):
        """At a target level of ±1,000 too (docs/PLAN.md task 209 (c)): the
        intercept travels there from its prior at 0, on the noise the
        filter learns from innovations the level makes a thousand times
        the noise's; and at a level running from +1,000 to -1,000 across the
        rows, through zero (task 209 (d))."""
        import filterpy.kalman as kalman

        rng = np.random.default_rng(43)
        n, half_life, coef_hl = 300, 30.0, 50.0
        raw = rng.normal(0.0, 1.0, (n, 2))
        drift = np.arange(n) / n
        y = _level(level, n) + 0.5 + (1.5 - drift) * raw[:, 0] + (-0.8 + 2.0 * drift) * raw[:, 1]
        y = y + rng.normal(0.0, 0.3, n)
        # Features at levels and scales of their own, so standardizing matters.
        x = raw * np.array([5.0, 0.2]) + np.array([3.0, -40.0])
        skip = (np.arange(n) % 9 == 4) & (np.arange(n) > 20)
        w = np.where(skip & (how == "zero weight"), 0.0, rng.uniform(0.5, 1.5, n))
        y_seen = np.where(skip & (how == "null"), np.nan, y)
        steps = np.ones(n)
        steps[150] = 4.0
        steps[230:] = 0.5
        t = np.cumsum(steps) - steps[0]
        frame = pl.DataFrame(
            {
                "t": t,
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": [None if np.isnan(v) else float(v) for v in y_seen],
                "w": w,
            },
            schema={c: pl.Float64 for c in ("t", "x0", "x1", "y", "w")},
        )
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1e9,
            coef_half_life=coef_hl,
            p0=1.0,
            weight="w",
            half_life=half_life,
            min_weight=0.0,
            coef_every=0,
        )
        out = po.ModelBank([spec]).fit_predict(frame)["k"].struct.unnest()
        got = out["pred_y"].to_numpy()
        Z, F, usable = self.standardized(t, x, w, half_life)
        want = TestKalmanZeroWeightRow.filterpy_pred(
            kalman, t, Z, y_seen, w, half_life, coef_hl, F, usable
        )
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12 * _magnitude(level))
        # Without the change of coordinates filterpy is the filter read
        # through the moments as they stood, which this one no longer is.
        stale = TestKalmanZeroWeightRow.filterpy_pred(
            kalman, t, Z, y_seen, w, half_life, coef_hl, None, usable
        )
        assert np.nanmax(np.abs(stale - got)) > 1e-3
        assert np.isfinite(got[1:]).all()
        # Task 97: coef after row i, applied to row i + 1, is row i + 1's pred.
        coef = out["coef"].to_list()
        via = np.array([np.dot(coef[i], [1.0, *x[i + 1]]) for i in range(1, n - 1)])
        np.testing.assert_allclose(via, got[2:], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("level", [1e3, -1e3])
    def test_a_target_at_a_level_settles_where_one_at_zero_does(self, level):
        """What filterpy's agreement above does not say: how well the
        filter predicts once the intercept has travelled (docs/PLAN.md task
        209 (c)). The first innovation is the whole level, and the noise the
        filter learns from it -- in ``R`` and in the process noise -- decays
        on the half-life, so the fit at ±1,000 runs behind the fit at 0 for a
        while: measured, up to 23 off over rows 20 to 100 (the noise is 0.3),
        0.37 over 100 to 500, where its error is 0.320 against 0.312, and
        0.05 from 500 on, where its error is 0.3095 against 0.3090. Held
        there: within 1% of the fit at 0, and a prediction at every row."""
        rng = np.random.default_rng(43)
        n = 2000
        raw = rng.normal(0.0, 1.0, (n, 2))
        noise = rng.normal(0.0, 0.3, n)
        x = raw * np.array([5.0, 0.2]) + np.array([3.0, -40.0])

        def fit(level):
            y = level + 0.5 + 1.5 * raw[:, 0] - 0.8 * raw[:, 1] + noise
            spec = po.spec.kalman(
                "k",
                targets=["y"],
                features=["x0", "x1"],
                coef_half_life=50.0,
                half_life=30.0,
                min_weight=0.0,
            )
            frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
            out = po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("pred_y")
            return out.to_numpy().astype(float), y

        def rmse(p, y):
            return float(np.sqrt(np.mean((p[500:] - y[500:]) ** 2)))

        p0, y0 = fit(0.0)
        p, y = fit(level)
        assert np.isfinite(p[1:]).all()
        assert rmse(p, y) <= 1.01 * rmse(p0, y0), (rmse(p, y), rmse(p0, y0))


class TestASharedCovarianceIsFilterpy:
    """Review round 4, CC3, and task 204. Under ``share_p`` the targets keep
    one ``P`` and each its own coefficients. ``P``'s recursion never reads
    ``y``, so each observed target's update is ``filterpy``'s
    ``KalmanFilter.update`` from the ``P`` the row found, and ``P`` takes the
    row once: the ``P`` those updates leave, the same for each. (Task 204:
    each update started from the ``P`` the one before left, which counted
    the row once per target.) Every one of them reads the same noise, ``R = σ²/w``
    with ``σ²`` the targets' mean residual variance as the row arrives (before
    there is one, the mean squared innovation over the targets the row
    observes; `kalman.rs`'s module doc, CC4), and the process noise
    ``σ²·(ln 2 · D / coef_half_life)²`` goes in once, before the first update,
    on a row that observes a target, ``D`` the clock since the last such row
    (docs/PLAN.md task 211).
    The filter read ``σ²`` inside its per-target loop, after the targets
    before had moved theirs on the row, so the second target's ``R`` held the
    first's residual of the same row and the order of ``targets`` changed
    every prediction. Two targets an order of magnitude apart in noise, the
    second missing one row in five, a zero-weight row one in nine, uneven
    clock steps, unstandardized so ``kf.x`` is the coefficients; ``filterpy``
    updates ``P`` in Joseph form and ``kalman`` in the simple form, so the two
    agree to rounding."""

    @staticmethod
    def filterpy_pred(
        kalman: Any,
        t: np.ndarray,
        Z: np.ndarray,
        ys: list[np.ndarray],
        w: np.ndarray,
        half_life: float,
        coef_hl: float,
    ) -> list[np.ndarray]:
        n, k1 = Z.shape
        m = len(ys)
        kfs = [kalman.KalmanFilter(dim_x=k1, dim_z=1) for _ in range(m)]
        for kf in kfs:
            kf.x = np.zeros((k1, 1))
            kf.F = np.eye(k1)
        P = np.zeros((k1, k1))
        sized = False
        gap = 0.0
        sig2, wsig, wj = np.zeros(m), np.zeros(m), np.zeros(m)
        preds = [np.full(n, np.nan) for _ in range(m)]
        for i in range(n):
            d = 0.0 if i == 0 else t[i] - t[i - 1]
            lam = 0.5 ** (d / half_life)
            gap += d
            z = Z[i]
            zb = [z @ kf.x[:, 0] for kf in kfs]
            for j in range(m):
                if wj[j] > 0.0:
                    preds[j][i] = zb[j]
            seen = [not np.isnan(ys[j][i]) and w[i] > 0.0 for j in range(m)]
            # The noise, read once for the row: the mean residual variance,
            # or before there is one the mean squared innovation.
            s2 = sig2.mean()
            if not s2 > 0.0:
                e2 = [(ys[j][i] - zb[j]) ** 2 for j in range(m) if seen[j]]
                s2 = float(np.mean(e2)) if e2 else 0.0
            # The process noise on a row that observes a target, for the
            # whole clock since the last such row (task 211).
            if any(seen):
                if sized:
                    P = P + np.eye(k1) * s2 * (np.log(2.0) * gap / coef_hl) ** 2
                elif s2 > 0.0:
                    P, sized = np.eye(k1) * s2, True
                gap = 0.0
            after = None
            for j in range(m):
                if not seen[j]:
                    wj[j] *= lam
                    wsig[j] *= lam
                    continue
                if s2 > 0.0:
                    kfs[j].P = P
                    kfs[j].update(ys[j][i], R=s2 / w[i], H=z[None, :])
                    after = kfs[j].P
                aged = lam * wsig[j]
                wsig[j] = aged
                if not np.isnan(preds[j][i]):
                    r = ys[j][i] - preds[j][i]
                    sig2[j] = (aged * sig2[j] + w[i] * r * r) / (aged + w[i])
                    wsig[j] = aged + w[i]
                wj[j] = lam * wj[j] + w[i]
            if after is not None:
                P = after
        return preds

    @pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
    def test_the_shared_filter_is_filterpy_reading_one_noise_a_row(self, order):
        import filterpy.kalman as kalman

        rng = np.random.default_rng(47)
        n, half_life, coef_hl = 300, 30.0, 50.0
        x = rng.normal(0.0, 1.0, (n, 2))
        drift = np.arange(n) / n
        targets = {
            "a": 1.0 + (2.0 - drift) * x[:, 0] - x[:, 1] + rng.normal(0.0, 0.3, n),
            "b": -0.5 + 0.5 * x[:, 0] + (3.0 * drift) * x[:, 1] + rng.normal(0.0, 1.5, n),
        }
        targets["b"] = np.where(np.arange(n) % 5 == 3, np.nan, targets["b"])
        w = np.where(np.arange(n) % 9 == 4, 0.0, rng.uniform(0.5, 1.5, n))
        steps = np.ones(n)
        steps[150] = 4.0
        steps[230:] = 0.5
        t = np.cumsum(steps) - steps[0]
        frame = pl.DataFrame(
            {
                "t": t,
                "x0": x[:, 0],
                "x1": x[:, 1],
                **{c: [None if np.isnan(v) else float(v) for v in y] for c, y in targets.items()},
                "w": w,
            },
            schema={c: pl.Float64 for c in ("t", "x0", "x1", "a", "b", "w")},
        )
        spec = po.spec.kalman(
            "k",
            targets=list(order),
            features=["x0", "x1"],
            clock="t",
            gap_cap=1e9,
            coef_half_life=coef_hl,
            standardize=False,
            share_p=True,
            p0=1.0,
            weight="w",
            half_life=half_life,
            min_weight=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(frame)["k"].struct.unnest()
        Z = np.column_stack([np.ones(n), x])
        want = self.filterpy_pred(kalman, t, Z, [targets[c] for c in order], w, half_life, coef_hl)
        for c, p in zip(order, want, strict=True):
            got = out[f"pred_{c}"].to_numpy()
            assert np.isfinite(got).sum() > 250, c
            np.testing.assert_allclose(got, p, rtol=1e-9, atol=1e-12, err_msg=c)


class TestASharedCovariancesStandardErrorsAreEachTargets:
    """Docs/PLAN.md task 211 (the review of 2026-10-08, T8). Under
    ``share_p`` the one ``P`` is driven by the targets' mean noise ``σ̄²``.
    With a noise ``σ²_j`` of its own, a target's own filter keeps ``P_j =
    σ²_j P̃`` where the shared one keeps ``σ̄² P̃``: ``P̃``'s recursion, ``P̃ -
    P̃ z zᵀ P̃ / (zᵀ P̃ z + 1)`` with the noise ``(ln 2 / h)²`` and the prior
    ``p0``, reads no noise at all. So target ``j``'s ``se_coef`` is the
    shared ``P``'s times ``σ_j / σ̄``. Read off ``P`` as it stands, two
    targets of noise 0.01 and 1 had standard errors 7.3 times too large and
    1.39 times too small.

    The second opinion is each target's own filter, filterpy's
    ``KalmanFilter`` at the target's true noise and the process noise its
    ``coef_half_life`` implies, ``R_j (ln 2 / h)²`` per unit row:
    unstandardized, so ``P`` is the coefficients' covariance. The bank's
    noises are EW estimates of the true ones, so the two agree to a few
    percent past the first rows, where the old reading parted by 7.3 and
    1.39 times."""

    def test_each_target_reads_its_own_noise(self):
        import filterpy.kalman as kalman

        rng = np.random.default_rng(50)
        n, h = 6000, 50.0
        noise = (0.01, 1.0)
        x = rng.normal(size=(n, 2))
        Z = np.column_stack([np.ones(n), x])
        ys = {}
        for j, r in enumerate(noise):
            q = r * (np.log(2.0) / h) ** 2
            b = np.array([0.5, -0.3, 0.2]) + np.cumsum(np.sqrt(q) * rng.normal(size=(n, 3)), 0)
            ys[f"y{j}"] = (Z * b).sum(axis=1) + np.sqrt(r) * rng.normal(size=n)
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], **ys})
        spec = po.spec.kalman(
            "k",
            targets=["y0", "y1"],
            features=["x0", "x1"],
            coef_half_life=h,
            half_life=500.0,
            min_weight=0.0,
            standardize=False,
            share_p=True,
            emit_se_coef=True,
            coef_every=1,
        )
        se = np.array(
            po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("se_coef").to_list(),
            dtype=float,
        )
        for j, r in enumerate(noise):
            kf = kalman.KalmanFilter(dim_x=3, dim_z=1)
            kf.x = np.zeros((3, 1))
            kf.P = np.eye(3) * r
            own = np.empty((n, 3))
            for i in range(n):
                kf.predict(Q=np.eye(3) * r * (np.log(2.0) / h) ** 2)
                kf.update(ys[f"y{j}"][i], R=r, H=Z[i][None, :])
                own[i] = np.sqrt(np.diag(kf.P))
            ratio = se[2000:, 3 * j : 3 * j + 3] / own[2000:]
            assert np.median(ratio) == pytest.approx(1.0, abs=0.05), (j, np.median(ratio))
            assert np.quantile(ratio, [0.01, 0.99]) == pytest.approx([1.0, 1.0], abs=0.15), j


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
        import statsmodels.tsa.stattools as stattools

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
            half_life=float("inf"),
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        row = bank.marginal("m").row(0, named=True)
        rho_x = stattools.acf(x, nlags=1)[1]
        rho_y = stattools.acf(y, nlags=1)[1]
        assert row["lag_corr_xx"][0] == pytest.approx(rho_x, abs=0.02)
        assert row["lag_corr_yy"][0] == pytest.approx(rho_y, abs=0.02)
        if 1.0 + 2.0 * rho_x * rho_y > 0.0:
            factor = 1.0 + 2.0 * row["lag_corr_xx"][0] * row["lag_corr_yy"][0]
            assert row["n_serial"] == pytest.approx(row["n_kish"] / factor, rel=1e-12)
            assert np.isfinite(row["t_serial"])
        else:
            # NaN in the model, null in the frame, as `phi_x` is.
            assert row["n_serial"] is None, row["n_serial"]
            assert row["t_serial"] is None, row["t_serial"]


class TestAWindowedLagIsTheWindows:
    """C18, the review's T-S11, and task 137. Under a ``window`` the lag
    moments are truncated as the pair's are, so ``lag_corr_xx`` is the
    autocorrelation of the rows inside the window. The reference is
    ``statsmodels``' ``acf`` of those rows, to the statistical tier T-S11
    gives, 0.02: each windowed lag moment is the increments made inside the
    window, centred at the mean as it stood, which ``acf``'s full-sample
    mean does not reproduce to the bit. An AR(1) ``x`` changes ``phi`` from
    0.2 to 0.8 at row 4500 of 5000 and the window is the last 500 rows, so
    the whole history's autocorrelation is far from the window's: the
    reading sat between the two before (live co-moment over windowed
    variance), and the pair is refused without ``window_lags=True``."""

    def test_lag_corr_is_the_acf_of_the_rows_inside_the_window(self):
        import statsmodels.tsa.stattools as stattools

        rng = np.random.default_rng(137)
        n, change, window = 5000, 4500, 500
        e = rng.normal(0.0, 1.0, n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = (0.2 if t < change else 0.8) * x[t - 1] + e[t]
        y = rng.normal(0.0, 1.0, n)
        spec = po.spec.marginal(
            "m",
            targets=["y"],
            features=["x"],
            lags=[1, 2, 3],
            half_life=float("inf"),
            clock="t",
            gap_cap=1.0,
            window_size=float(window),
            window_lags=True,
        )
        bank = po.ModelBank([spec])
        frame = pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y})
        bank.fit_predict(frame)
        row = bank.marginal("m").row(0, named=True)
        inside = stattools.acf(x[-window:], nlags=3)[1:]
        whole = stattools.acf(x, nlags=3)[1:]
        np.testing.assert_allclose(row["lag_corr_xx"], inside, atol=0.02)
        assert np.all(np.abs(inside - whole) > 0.15), (inside, whole)
        assert row["weight_sum"] == pytest.approx(window, rel=1e-12)


class TestTheBartlettSerialFactor:
    """Task 135 (S18's other half). ``serial_rule = "bartlett"`` weights lag
    ``l`` by Newey and West's ``1 - l/(L + 1)``, ``L`` the longest kept lag.
    ``statsmodels`` supplies both halves of the second opinion: the weights
    (``sandwich_covariance.weights_bartlett``, the kernel its HAC estimator
    uses) and the autocorrelations (``acf``), which our lag moments match to
    the 0.02 tier T-S11 gives. The factor is then ``n_kish / n_serial``, to
    rounding, from our own lag correlations and the library's weights. On
    S18's pair, where ``"truncated"`` has no factor, this one has one."""

    @pytest.mark.parametrize("phi_y", [0.8, -0.8])
    def test_the_factor_takes_statsmodels_bartlett_weights(self, phi_y):
        import statsmodels.stats.sandwich_covariance as sandwich
        import statsmodels.tsa.stattools as stattools

        rng = np.random.default_rng(135)
        n, lags = 5000, [1, 2, 3, 4]

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
            lags=lags,
            serial_rule="bartlett",
            half_life=float("inf"),
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        row = bank.marginal("m").row(0, named=True)
        acf_x = stattools.acf(x, nlags=4)[1:]
        acf_y = stattools.acf(y, nlags=4)[1:]
        np.testing.assert_allclose(row["lag_corr_xx"], acf_x, atol=0.02)
        np.testing.assert_allclose(row["lag_corr_yy"], acf_y, atol=0.02)
        w = sandwich.weights_bartlett(max(lags))[1:]  # lags 1..L; the library's kernel
        np.testing.assert_allclose(w, [1 - lag / 5 for lag in lags], rtol=1e-15)
        factor = 1.0 + 2.0 * float(
            np.sum(w * np.array(row["lag_corr_xx"]) * np.array(row["lag_corr_yy"]))
        )
        assert factor > 0.0, factor
        assert row["n_serial"] == pytest.approx(row["n_kish"] / factor, rel=1e-12)
        assert np.isfinite(row["t_serial"])


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


def _own_rows_pred(x: np.ndarray, y: np.ndarray, t: int, half_life: float) -> float:
    """Row ``t`` predicted by the weighted least-squares fit of the rows
    before it on which the target is present, each at ``0.5 ** (age /
    half-life)`` -- the age counted in rows, the target's missing ones
    included, since the clock runs on every row."""
    keep = ~np.isnan(y[:t])
    w = 0.5 ** (((t - 1) - np.arange(t)) / half_life)
    b = _wls(x[:t][keep], y[:t][keep], w[keep])
    return float(b[0] + x[t] @ b[1:])


def _pairwise_pred(x: np.ndarray, y: np.ndarray, t: int, half_life: float) -> float:
    """Row ``t`` predicted from pairwise-complete weighted moments of the rows
    before it: the features' covariance over every row, their covariance
    with the target and every mean over the rows the target is present on,
    each by ``numpy.cov`` with the rows' weights as ``aweights``."""
    keep = ~np.isnan(y[:t])
    w = 0.5 ** (((t - 1) - np.arange(t)) / half_life)
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
            half_life=h,
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
            half_life=h,
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
        import pandas as pd

        n = 500
        x, y = _gappy(n, 11, 0.0)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
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
            half_life=h,
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
            half_life=h,
            target_gaps=target_gaps,
            max_rows_between_solves=1,
            max_iter=10_000,
            tol=1e-15,
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
        (age / half-life)``: the target's own by ``numpy.linalg.lstsq`` under
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
            half_life=h,
            ridge=0.0,
            window_size=window,
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
        import statsmodels.api as sm

        n, h, level = 500, 30.0, 40.0
        x, y = _gappy(n, 23, level)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=h,
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
        import statsmodels.api as sm

        n, ridge = 400, 0.3
        x, y = _gappy(n, 29, 10.0)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
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
        import statsmodels.api as sm

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
            half_life=float("inf"),
            max_rows_between_solves=1,
            max_iter=100_000,
            tol=1e-15,
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


class _EpsilonInsensitive:
    """The loss river's ``PARegressor`` docstring names, ``max(|y - p| - eps,
    0)``."""

    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, y_true: float, y_pred: float) -> float:
        return max(abs(y_pred - y_true) - self.eps, 0.0)


def _river_pa(**kw):
    """river's ``PARegressor``, scoring a row with the epsilon-insensitive
    loss. river 0.26.1's ``EpsilonInsensitiveHinge`` opens with a
    classification line, ``y_true = y_true * 2 - 1``, so it scores a target
    of 2 as 3; the model gets the loss it names, and keeps its own once a
    release has fixed it."""
    from river import linear_model

    model = linear_model.PARegressor(**kw)
    eps = kw.get("eps", 0.1)
    if model.loss(2.0, 0.0) != pytest.approx(2.0 - eps):
        model.loss = _EpsilonInsensitive(eps)
    return model


class TestPassiveAggressiveIsRivers:
    """T-S18, from pass 10 of the review: ``pa`` is river's ``PARegressor`` --
    the same ``tau`` in all three modes and the same step -- on every row,
    without an intercept and at unit weight, the two conditions the mapping
    states (D10). Measured: ``2e-15`` on the predictions and ``9e-16`` on
    the coefficients over 500 rows, once river is given the loss it
    documents (see :func:`_river_pa`). River's ``eps`` is in the target's
    units and ours in the target's own EW std (docs/PLAN.md task 202), so
    the two are held at ``eps = 0``, which is no tube in either; and ours on
    the raw features, as river reads them."""

    @pytest.mark.parametrize("level", [0.0, -1e3, CROSSING])
    @pytest.mark.parametrize(("mode", "river_mode"), [("pa", 0), ("pa1", 1), ("pa2", 2)])
    def test_every_row_is_rivers_without_an_intercept(self, mode, river_mode, level):
        """At a level of -1,000 too, both features and the target all
        negative (docs/PLAN.md task 209 (b)), where the numbers are a
        thousand times larger and agree to 1e-12 of them; and at a level
        running from +1,000 to -1,000 across the rows (task 209 (d))."""
        rng = np.random.default_rng(11)
        n, c, eps = 500, 0.3, 0.0
        x = rng.normal(0.0, 1.0, (n, 2)) + np.reshape(_level(level, n), (-1, 1))
        y = 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0.0, 0.3, n)
        spec = po.spec.pa(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            mode=mode,
            c=c,
            eps=eps,
            fit_intercept=False,
            half_life=float("inf"),
            min_weight=0.0,
            coef_every=0,
            standardize=False,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct
        pred = out.field("pred_y").to_numpy()
        coef = np.array(out.field("coef").to_list(), dtype=float)
        river = _river_pa(C=c, mode=river_mode, eps=eps, learn_intercept=False)
        for i in range(n):
            row = {"x0": x[i, 0], "x1": x[i, 1]}
            # The prediction the row is scored with, then the fit it leaves.
            scale = _magnitude(level)
            want_p = river.predict_one(row)
            assert pred[i] == pytest.approx(want_p, abs=1e-12 * scale), (mode, i)
            river.learn_one(row, y[i])
            want = [river.weights["x0"], river.weights["x1"]]
            assert coef[i] == pytest.approx(want, abs=1e-12), (mode, i)

    def test_the_intercept_is_inside_the_norm_here_and_outside_in_river(self):
        """The one difference the mapping names (D10): here the intercept is a
        column of ``z``, so ``‖z‖²`` counts its 1 and a plain step lands on the
        tube's edge, the target itself at ``eps = 0``; river adds the same
        ``tau`` to its bias outside the norm, and overshoots by ``ℓ/‖x‖²``."""
        x, y, eps = {"x0": 1.0, "x1": 0.5}, 2.0, 0.0
        frame = pl.DataFrame({"x0": [1.0, 1.0], "x1": [0.5, 0.5], "y": [y, y]})
        spec = po.spec.pa(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            mode="pa",
            eps=eps,
            half_life=float("inf"),
            min_weight=0.0,
            standardize=False,
        )
        again = po.ModelBank([spec]).fit_predict(frame)["m"].struct.field("pred_y")[1]
        assert again == pytest.approx(y - eps, abs=1e-12)
        river = _river_pa(mode=0, eps=eps, learn_intercept=True)
        river.learn_one(x, y)
        loss, sq_norm = y - eps, 1.0**2 + 0.5**2  # from a first prediction of 0
        assert river.predict_one(x) == pytest.approx(y - eps + loss / sq_norm, abs=1e-12)


class TestTheEwMomentsArePandas:
    """T-S9: pandas' ``ewm`` keeps the same adjusted, centred, mean-form EW
    recursion, so ``ew_cov``'s ``mean``, ``var``, ``cov`` and ``corr`` on a
    row clock are its ``mean()``, ``var(bias=True)``, ``cov(bias=True)``
    and ``corr()`` -- one row behind, since a row's statistics are read
    before it is folded in -- at any offset. The tolerance is absolute: at
    ``1e8`` the data are resolved to ``L·ε``, and both sides are within a
    few units in the last place of an exact reference built from the
    deviations (measured: ``6e-8`` on the mean, ``2e-8`` on the rest), where
    a relative one means nothing for a covariance of two independent
    columns, which crosses 0. ``bias=False`` would be Kish's correction,
    ``n_kish``, not ``weight_sum``."""

    H = 25.0

    @pytest.mark.parametrize("offset", [0.0, 1e8])
    def test_ew_cov_is_pandas_ewm(self, offset):
        import pandas as pd

        rng = np.random.default_rng(7)
        n = 400
        x = offset + rng.normal(0.0, 1.0, (n, 2))
        spec = po.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            half_life=self.H,
            stats=["mean", "var", "cov", "corr"],
            min_weight=0.0,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct.unnest()
        pdf = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
        ew, pair = pdf.ewm(halflife=self.H), pdf["x0"].ewm(halflife=self.H)
        want = {
            "mean_x0": ew.mean()["x0"].to_numpy(),
            "var_x0": ew.var(bias=True)["x0"].to_numpy(),
            "cov_x0_x1": pair.cov(pdf["x1"], bias=True).to_numpy(),
            "corr_x0_x1": pair.corr(pdf["x1"]).to_numpy(),
        }
        for name, ref in want.items():
            # Row i's statistics are pandas' at row i - 1.
            got, ref = out[name].to_numpy().astype(float)[1:], ref[:-1]
            live = np.isfinite(got)
            assert live.sum() > n - 10, name
            tol = (4e-15 if name == "mean_x0" else 1e-15) * offset + 1e-12
            assert np.max(np.abs(got[live] - ref[live])) <= tol, (name, offset)

    def test_on_an_irregular_clock_the_mean_is_pandas_with_times(self):
        """``times=`` is the clock model, a weight of ``0.5 ** (age / h)`` in
        clock units under the ``adjust=True`` it forces, as ours is. pandas
        takes ``times`` for ``mean()`` alone, and rounds them to nanoseconds,
        which is the ``1e-10`` (measured ``8e-12``)."""
        import pandas as pd

        rng = np.random.default_rng(8)
        n = 300
        t = np.cumsum(rng.uniform(0.2, 3.0, n))
        x = rng.normal(0.0, 1.0, n)
        spec = po.spec.ew_cov(
            "m",
            features=["x0"],
            half_life=self.H,
            clock="t",
            # A cap no gap here reaches (they are under 3): pandas has none.
            gap_cap=1e9,
            stats=["mean"],
            min_weight=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"t": t, "x0": x}))
        got = out["m"].struct.field("mean_x0").to_numpy().astype(float)[1:]
        times = pd.to_datetime(t, unit="s")
        ref = pd.Series(x).ewm(halflife=pd.Timedelta(seconds=self.H), times=times).mean()
        ref = ref.to_numpy()[:-1]
        live = np.isfinite(got)
        assert live.sum() > n - 10
        assert np.max(np.abs(got[live] - ref[live])) <= 1e-10

    def test_the_target_moments_and_a_marginal_pair_are_pandas_too(self):
        """The same call on ``y`` alone is the target moments a Gram carries,
        and on ``[x, y]`` it is ``marginal``'s pair, at the end of the stream."""
        import pandas as pd

        rng = np.random.default_rng(9)
        n = 600
        x = rng.normal(0.0, 1.0, n)
        y = 0.7 * x + rng.normal(0.0, 1.0, n)
        frame = pl.DataFrame({"x0": x, "y": y})
        bank = po.ModelBank(
            [po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=self.H)]
        )
        bank.fit_predict(frame)
        g = bank.gram("m")[0]
        ys = pd.Series(y).ewm(halflife=self.H)
        assert g["target_means"][0] == pytest.approx(ys.mean().iloc[-1], rel=1e-12)
        assert g["target_vars"][0] == pytest.approx(ys.var(bias=True).iloc[-1], rel=1e-12)
        spec = po.spec.marginal("m", targets=["y"], features=["x0"], half_life=self.H)
        pair = po.ModelBank([spec])
        pair.fit_predict(frame)
        corr = pd.Series(x).ewm(halflife=self.H).corr(pd.Series(y)).iloc[-1]
        assert pair.marginal("m")["corr"][0] == pytest.approx(corr, rel=1e-12)


class TestSgdQuantileIsQuantReg:
    """T-S4's quantile half, met by the model that can meet it. ``quantile``
    fitted by IRLS on each row's prior residual until N9, its weights frozen
    as the rows arrived, and did not settle on ``statsmodels``' ``QuantReg``
    at any length measured; it takes a Newton step on the smoothed check loss
    now, and :class:`TestQuantileIsQuantReg` holds it there.
    ``sgd(loss="quantile")`` takes the pinball loss's subgradient,
    and under ``inv_scaling`` it is ``QuantReg``'s fit to within ``0.03`` at
    100 000 rows (measured ``0.022`` and ``0.027``), at the median and the 0.9
    quantile of a skewed noise, where the mean's fit is 0.3 and 1.3 away."""

    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_sgd_quantile_settles_on_quantreg(self, tau):
        import statsmodels.api as sm

        rng = np.random.default_rng(3)
        n = 100_000
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            loss="quantile",
            quantile=tau,
            schedule="inv_scaling",
            learning_rate=0.5,
            half_life=float("inf"),
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        got = bank.coef("m")["coef"].to_numpy()
        z = sm.add_constant(x)
        want = np.asarray(sm.QuantReg(y, z).fit(q=tau).params)
        assert np.max(np.abs(got - want)) < 0.06, (got, want)
        # The check tells the quantile from the mean: least squares is far off.
        assert np.max(np.abs(np.linalg.lstsq(z, y, rcond=None)[0] - want)) > 0.25


class TestQuantileIsQuantReg:
    """T-S4's quantile half, and what N9 became. The model fits by one-step
    Newton on the kernel-smoothed check loss: a row inside the band of
    half-width ``h = quantile_eps * sigma`` is a least-squares row with target
    ``y + 2h(tau - 1/2)``, and a row outside it nudges the cross-moment by
    ``2h * psi_tau(r) * z`` and weighs nothing in the Gram. That settles on
    ``statsmodels``' ``QuantReg``, where the frozen IRLS weights it replaced
    did not: those were 0.164 off at the median and 0.477 at the 0.9 quantile
    after 20 000 rows, against ``QuantReg``'s own standard errors of 0.007 and
    0.021, where this one is 0.005 and 0.006 off (N9)."""

    @staticmethod
    def _rows(n, seed=3):
        """Exponential noise, so the median's fit and the mean's part by 0.3
        in the intercept and the check can tell them apart."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 2))
        return x, 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, n)

    @staticmethod
    def _fit(x, y, tau, **kw):
        spec = po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            quantile=tau,
            max_rows_between_solves=1,
            **kw,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        return bank.coef("m")["coef"].to_numpy()

    @pytest.mark.parametrize(("tau", "tol"), [(0.5, 0.05), (0.9, 0.08)])
    def test_the_fit_is_quantregs(self, tau, tol):
        import statsmodels.api as sm

        x, y = self._rows(20_000)
        got = self._fit(x, y, tau, half_life=float("inf"))
        z = sm.add_constant(x)
        want = np.asarray(sm.QuantReg(y, z).fit(q=tau).params)
        assert np.max(np.abs(got - want)) <= tol, (got, want)
        # The quantile it was asked for, not the mean: least squares is far off.
        assert np.max(np.abs(np.linalg.lstsq(z, y, rcond=None)[0] - want)) > 0.25

    @pytest.mark.parametrize("level", [1e3, -1e3])
    @pytest.mark.parametrize("tau", [0.25, 0.75])
    def test_the_fit_is_quantregs_on_a_target_of_one_sign(self, tau, level):
        """Every target above zero, at 1e3, or every one below, at -1e3, at a
        level on each side of the median (docs/PLAN.md task 209 (c)): both
        sides of the check function, its scale ``s`` -- its first residuals
        are the whole level -- and the least-squares rows before it, which
        the first rows are. The fit is ``QuantReg``'s, 0.011 off at 0.25 and
        0.010 at 0.75 measured, and the fit of the same rows at 0, shifted by
        the level, to 5.8e-15."""
        import statsmodels.api as sm

        x, y = self._rows(20_000)
        y_level = y + level
        assert (np.sign(y_level) == np.sign(level)).all(), "the case: one sign"
        got = self._fit(x, y_level, tau, half_life=float("inf"))
        z = sm.add_constant(x)
        want = np.asarray(sm.QuantReg(y_level, z).fit(q=tau).params)
        assert np.max(np.abs(got - want)) <= 0.03, (got, want)
        at_zero = self._fit(x, y, tau, half_life=float("inf"))
        shifted = at_zero + np.array([level, 0.0, 0.0])
        np.testing.assert_allclose(got, shifted, rtol=0.0, atol=1e-11)

    def test_it_is_the_batch_smoothed_fit_at_the_same_bandwidth(self):
        """The estimator the stream approximates, computed by hand: Newton to
        convergence on the same smoothed loss, at the same bandwidth. A long
        stream settles ``sigma``, so the band is the residual scale's."""
        x, y = self._rows(20_000)
        tau, eps = 0.9, 0.2
        got = self._fit(x, y, tau, half_life=float("inf"), quantile_eps=eps)
        z = np.column_stack([np.ones(len(y)), x])
        b = np.linalg.lstsq(z, y, rcond=None)[0]
        h = eps * np.std(y - z @ b)
        for _ in range(200):
            r = y - z @ b
            near = np.abs(r) < h
            psi = np.where(near, tau - 0.5 + r / (2.0 * h), tau - (r < 0))
            step = np.linalg.solve((z[near].T @ z[near]) / (2.0 * h), z.T @ psi)
            b = b + step
            if np.max(np.abs(step)) < 1e-12:
                break
        assert np.max(np.abs(got - b)) <= 0.05, (got, b)

    def test_it_follows_a_shift_the_frozen_weights_lagged(self):
        """Under a half-life the fit has to move when the level does. The
        weights the frozen IRLS left on old rows pulled it back toward the
        fits those rows were scored by: 600 rows after a jump of 3 -- three
        half-lives -- it had covered 1.5 of it, where the quantile regression
        of the rows then in the window had moved the whole way."""
        import statsmodels.api as sm

        x, y = self._rows(10_000)
        y[5000:] += 3.0
        stop, window = 5600, 600
        got = self._fit(x[:stop], y[:stop], 0.5, half_life=200.0)
        lo = stop - window
        want = np.asarray(sm.QuantReg(y[lo:stop], sm.add_constant(x[lo:stop])).fit(q=0.5).params)
        assert np.max(np.abs(got - want)) <= 0.35, (got, want)


class TestHuberAgainstScikitLearn:
    """T-S4's huber half (``docs/REVIEW-2026-09-12.md``), with scikit-learn a
    live oracle since task 121. Two tiers, as the review sets them. Exact: at
    ``huber_delta = 1e9`` no row is down-weighted, so the fit is least
    squares, and without an intercept and with ``standardize`` it is
    ``LinearRegression(fit_intercept=False)`` on the raw columns -- the limit
    case the review names for pattern B. Statistical: with one row in fifty a
    gross error, ``huber`` at its default ``huber_delta = 1.345`` and
    ``HuberRegressor()`` at its default ``epsilon = 1.35``, the same constant
    rounded, land together where least squares does not. The
    algorithms differ -- ours reweights each row by its prior residual in
    units of the residuals' EW spread, theirs solves for the scale jointly --
    so they agree to a tolerance: 0.044 at this seed, at most 0.08 over eight,
    against 0.38 to 0.47 for least squares. The spread ours measures in is
    not itself robust (review D4, ``robust.rs``), so at one row in ten the
    cut widens and the intercept sits halfway to least squares: 1.5 against
    theirs at 0.66, the truth at 0.5 and least squares at 2.7."""

    @staticmethod
    def fit(df, **kw):
        spec = po.spec.huber(
            "h",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            max_rows_between_solves=1,
            min_weight=0.0,
            **kw,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        return np.array(bank.coef("h")["coef"].to_list())

    def test_a_huge_delta_is_least_squares(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(3)
        n = 600
        x = rng.normal(0.0, 1.0, (n, 2)) + np.array([3.0, -1.0])
        y = 0.8 * x[:, 0] - 1.7 * x[:, 1] + rng.normal(0.0, 0.5, n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        got = self.fit(df, huber_delta=1e9, ridge=1e-12, standardize=True, fit_intercept=False)
        want = LinearRegression(fit_intercept=False).fit(x, y).coef_
        np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_under_outliers_it_lands_where_huber_regressor_does(self):
        from sklearn.linear_model import HuberRegressor, LinearRegression

        rng = np.random.default_rng(2)
        n = 4000
        x = rng.normal(0.0, 1.0, (n, 2))
        truth = np.array([0.5, 1.2, -0.7])
        y = truth[0] + x @ truth[1:] + rng.normal(0.0, 1.0, n)
        # One row in fifty carries a gross error, all to one side.
        bad = rng.random(n) < 0.02
        y[bad] += rng.uniform(15.0, 30.0, bad.sum())
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        # `huber`'s default `huber_delta` is the 95%-efficiency constant
        # 1.345 (Huber 1981), statsmodels' `HuberT`; scikit-learn rounds it to
        # 1.35, its own default (docs/PLAN.md task 195, U3).
        ours = self.fit(df)
        theirs = HuberRegressor(alpha=0.0, max_iter=1000).fit(x, y)
        theirs = np.concatenate(([theirs.intercept_], theirs.coef_))
        ols = LinearRegression().fit(x, y)
        ols = np.concatenate(([ols.intercept_], ols.coef_))
        np.testing.assert_allclose(ours, theirs, atol=0.1)
        # The control: least squares is pulled away by the same rows.
        assert np.abs(ols - theirs).max() > 4 * np.abs(ours - theirs).max()


class TestTheBinsAgainstScipyAndAStump:
    """T-S12 (``docs/REVIEW-2026-09-12.md``). With given edges, ``half_life =
    inf`` and unit weights, ``marginal``'s bins are
    ``scipy.stats.binned_statistic``'s -- the count, the mean -- and each
    bin's population variance, exactly: the bins are half-open ``[a, b)``
    in both, a value on an edge going to the bin above. The best split is a
    regression stump's among the edges: scikit-learn's
    ``DecisionTreeRegressor(max_depth = 1)`` chooses among every midpoint,
    so its impurity decrease over ``var(y)`` bounds ``split_gain`` from
    above, with equality when its threshold falls between the same two data
    points as ``split_at``."""

    def test_the_bins_are_binned_statistic_and_the_split_a_stump(self):
        from scipy.stats import binned_statistic
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.default_rng(5)
        n = 3000
        edges = [-1.0, 0.0, 0.5, 1.0]
        x = rng.uniform(-2.0, 2.0, n)
        # Values exactly on each edge, which go to the bin above in both.
        x[:40] = np.repeat(edges, 10)
        y = np.where(x >= 0.5, 2.0, 0.0) + 0.3 * rng.standard_normal(n)
        spec = po.spec.marginal(
            "m",
            targets=["y"],
            features=["x"],
            half_life=float("inf"),
            bin_edges=[edges],
            min_weight=2.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(pl.DataFrame({"x": x, "y": y}))
        row = bank.marginal("m").row(0, named=True)

        full = [-np.inf, *edges, np.inf]
        count = binned_statistic(x, y, "count", bins=full).statistic
        mean = binned_statistic(x, y, "mean", bins=full).statistic
        var = [np.var(y[(x >= lo) & (x < hi)]) for lo, hi in zip(full[:-1], full[1:], strict=True)]
        np.testing.assert_array_equal(row["bin_n"], count)
        np.testing.assert_allclose(row["bin_mean_y"], mean, rtol=1e-12)
        np.testing.assert_allclose(row["bin_var_y"], var, rtol=1e-10)

        tree = DecisionTreeRegressor(max_depth=1).fit(x[:, None], y).tree_
        n_root, (n_l, n_r) = tree.n_node_samples[0], tree.n_node_samples[1:3]
        drop = tree.impurity[0] - (n_l * tree.impurity[1] + n_r * tree.impurity[2]) / n_root
        stump_gain = drop / tree.impurity[0]
        assert stump_gain >= row["split_gain"] - 1e-12
        # The stump cut between the same two points as the edge at 0.5, so
        # the two gains are one number.
        below, above = x[x < row["split_at"]].max(), x[x >= row["split_at"]].min()
        assert row["split_at"] == 0.5 and below < tree.threshold[0] < above
        assert stump_gain == pytest.approx(row["split_gain"], rel=1e-9)


class TestAMeanRevertingKalmanIsFilterpy:
    """T-S5 in full (``docs/REVIEW-2026-09-12.md``): with ``revert_half_life``
    finite the transition is ``F = diag(2^(-d / r_i))`` and the process noise
    is added after it, which is ``filterpy``'s ``predict`` with that ``F``; on
    a reverting slot the noise of a gap ``d`` is ``q_i ((1 - 2^(-d / r_i)) /
    theta_i)**2``, ``theta_i = ln 2 / r_i`` (docs/PLAN.md task 214).
    Unstandardized, so the pull is toward zero in the columns' own units and
    ``kf.x`` is our coefficient vector; a scalar ``r`` pulls every slot, and
    ``[inf, r, r]`` leaves the intercept a random walk. Rows one clock unit
    apart, and a gap of five, so ``F`` is not the same on every row."""

    @pytest.mark.parametrize("revert", [25.0, [float("inf"), 40.0, 40.0]])
    def test_the_reverting_filter_is_filterpy_with_a_decaying_transition(self, revert):
        import filterpy.kalman as kalman

        rng = np.random.default_rng(17)
        n, half_life, coef_hl = 300, 30.0, 50.0
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(0.0, 0.3, n)
        t = np.arange(n, dtype=float)
        t[150:] += 5.0
        frame = pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "y": y})
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            coef_half_life=coef_hl,
            revert_half_life=revert,
            standardize=False,
            p0=1.0,
            half_life=half_life,
            gap_cap=1e9,
            min_weight=0.0,
        )
        got = po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("pred_y").to_numpy()

        r = np.broadcast_to(np.asarray(revert, dtype=float), (3,))
        kf = kalman.KalmanFilter(dim_x=3, dim_z=1)
        kf.x = np.zeros((3, 1))
        sized = False
        sig2 = wsig = wj = 0.0
        want = np.full(n, np.nan)
        for i in range(n):
            d = 0.0 if i == 0 else t[i] - t[i - 1]
            lam = 0.5 ** (d / half_life)
            kf.F = np.diag(0.5 ** (d / r))
            z = np.array([1.0, x[i, 0], x[i, 1]])
            # Before there is a sigma^2, the row's innovation squared against
            # the coefficients after the transition (CC4), which also sizes
            # the prior, p0 = 1 times it.
            s2 = sig2 if sig2 > 0.0 else (y[i] - z @ (kf.F @ kf.x)[:, 0]) ** 2
            if sized:
                # `d` on a walk, and on a reverting slot the gap the
                # reversion damps, `(1 - 2^(-d/r)) / theta` (task 214).
                g = np.array(
                    [d if np.isinf(ri) else (1.0 - 0.5 ** (d / ri)) * ri / np.log(2.0) for ri in r]
                )
                kf.predict(Q=np.diag(s2 * (np.log(2.0) / coef_hl) ** 2 * g**2))
            elif s2 > 0.0:
                kf.P, sized = np.eye(3) * s2, True
            if wj > 0.0:
                want[i] = z @ kf.x[:, 0]
            if s2 > 0.0:
                kf.update(y[i], R=s2, H=z[None, :])
            if not np.isnan(want[i]):
                res = y[i] - want[i]
                ws_new = lam * wsig + 1.0
                sig2 = (lam * wsig * sig2 + res * res) / ws_new
                wsig = ws_new
            wj = lam * wj + 1.0
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


# The fits are small on purpose; a readiness notice about one says nothing here.
@pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
class TestEwRidgeIsSklearnsRidge:
    """Every ``ewridge`` solve against scikit-learn's ``Ridge``: a second
    opinion on the whole weighted, penalized fit, where the Rust oracle in
    ``ewridge.rs`` (``every_solve_is_its_closed_form_across_targets_and_ridges``)
    checks the same forms with ``faer``'s solver (docs/PLAN.md task 113).

    With no decay, ``ewridge``'s mean-form system at ridge ``λ`` is
    ``Ridge``'s at ``alpha = λ·Σw`` with the same ``sample_weight``; the
    standardized penalty is the same fit on features divided by their
    deviation (centred) or root mean square (through the origin); a
    ``coef_prior`` ``c0`` is the ridge fit of ``y − X·c0``, shifted back by
    ``c0``; and ``ridge_scale``'s sum-scale system is ``Ridge`` at
    ``alpha = λ`` with the intercept column penalized as a feature."""

    @staticmethod
    def rows(n=300, seed=5):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 3)) * [1.0, 2.0, 0.5] + [0.5, -1.0, 3.0]
        y = 1.0 + x @ np.array([2.0, -1.0, 0.5]) + rng.normal(0.0, 0.3, n)
        return x, y, rng.uniform(0.5, 2.0, n)

    @pytest.mark.parametrize("with_prior", [False, True])
    @pytest.mark.parametrize("lam", [0.7, 4.0])
    @pytest.mark.parametrize(
        "mode", ["centred", "centred standardized", "origin", "origin standardized", "ridge_scale"]
    )
    def test_the_fit_is_sklearns_ridge(self, mode, lam, with_prior):
        from sklearn.linear_model import Ridge

        x, y, w = self.rows()
        intercept = mode.startswith("centred") or mode == "ridge_scale"
        standardize = "standardized" in mode
        prior = np.array([0.4, -1.0, 2.0, 0.5]) if with_prior else np.zeros(4)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "w": w})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            ridge=lam,
            standardize=standardize,
            ridge_scale="sum" if mode == "ridge_scale" else "mean",
            fit_intercept=intercept,
            coef_prior=[list(prior if intercept else prior[1:])] if with_prior else None,
            half_life=float("inf"),
            weight="w",
            min_weight=0.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        got = bank.coef("m").sort("position")["coef"].to_numpy()
        sw = w.sum()
        c0 = prior[1:]
        if mode == "ridge_scale":
            z = np.column_stack([np.ones(len(y)), x])
            fit = Ridge(alpha=lam, fit_intercept=False).fit(z, y - z @ prior, sample_weight=w)
            want = prior + fit.coef_
        elif intercept:
            mean = (w[:, None] * x).sum(0) / sw
            sd = np.sqrt((w[:, None] * (x - mean) ** 2).sum(0) / sw) if standardize else np.ones(3)
            fit = Ridge(alpha=lam * sw).fit(x / sd, y - x @ c0, sample_weight=w)
            want = np.concatenate(([fit.intercept_], c0 + fit.coef_ / sd))
        else:
            rms = np.sqrt((w[:, None] * x**2).sum(0) / sw) if standardize else np.ones(3)
            fit = Ridge(alpha=lam * sw, fit_intercept=False).fit(
                x / rms, y - x @ c0, sample_weight=w
            )
            want = c0 + fit.coef_ / rms
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


class TestFtrlIsVowpalWabbits:
    """``ftrl`` without a half-life against Vowpal Wabbit's ``--ftrl``, on
    every row: a second FTRL-proximal beside river's (T-R1), reaching the
    options river's comparison does not (docs/PLAN.md task 115; the user:
    "find an alternative oracle for ftrl that supports more options"). VW's
    ``ftrl.cc`` runs the recursion ``ftrl.rs`` states, and predicts as
    McMahan et al.'s Algorithm 1 does, from weights recomputed after the last
    update, where river's ``LogisticRegression`` predicts with the step
    before's. So the whole model is compared here: ``pred`` and ``coef`` on
    every row, under both losses, with and without the intercept and row
    weights (zeros among them), with null targets, two targets, and ``l1``
    and ``l2`` together.

    The mapping: ``--ftrl_alpha``, ``--ftrl_beta``, ``--l1`` and ``--l2`` are
    ours by name. VW's constant feature is the intercept (``--noconstant``
    without one). A row's weight is VW's importance, which scales the
    gradient as ours does. VW's squared loss is ``(p - y)²``, with derivative
    ``2(p - y)`` where ours is ``p - y``, so a row goes in at half its weight;
    and VW clips a squared-loss prediction to the labels seen so far unless
    it is given bounds, so the bounds are set wide. Its logistic loss takes
    labels ``±1`` and has the same derivative in the margin as ours. A null
    target is a VW prediction with no update.

    VW computes in single precision, so the agreement is to its rounding:
    measured at most ``3e-7`` on a probability, ``2.6e-6`` on a squared-loss
    prediction and ``9e-7`` on a coefficient over 400 rows, against ``1e-5``
    here. The control shows a slip in the mapping, the squared loss at its
    full weight, missing by more than ``0.2``. None of the libraries checked
    (river, VW, Keras's ``Ftrl``) forgets as a half-life does, so the finite
    half-life stays with ``tests/reference.py``'s ``ftrl_ref`` and the
    longhand and closed forms in ``ftrl.rs``'s tests (review 2026-09-12,
    C24)."""

    FEATURES = ("x0", "x1", "x2")
    TARGETS = ("y0", "y1")
    #: ``VW::details::CONSTANT``, the constant feature's hash
    #: (``vw/core/constant.h``); pyvw does not expose it.
    CONSTANT = 11650396
    #: ``(alpha, beta, l1, l2)``: the defaults, and an ``l1`` that holds the
    #: noise feature ``x2`` at exactly zero on about half the rows.
    PENALTIES = {"defaults": (0.1, 1.0, 0.0, 1.0), "sparse": (0.5, 0.3, 2.0, 0.2)}
    TOL = 1e-5

    @staticmethod
    def rows(loss, n=400, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 3))
        if loss == "logistic":
            p0 = 1.0 / (1.0 + np.exp(-(0.4 + 1.5 * x[:, 0] - 0.5 * x[:, 1])))
            p1 = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x[:, 1])))
            y0 = (rng.random(n) < p0).astype(float)
            y1 = (rng.random(n) < p1).astype(float)
        else:
            y0 = 0.4 + 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0.0, 0.3, n)
            y1 = -1.0 + 0.8 * x[:, 1] + rng.normal(0.0, 0.5, n)
        w = rng.uniform(0.0, 2.0, n)
        w[::17] = 0.0
        return pl.DataFrame(
            {
                "x0": x[:, 0],
                "x1": x[:, 1],
                "x2": x[:, 2],
                # Each target null on its own rows, so one learns where the
                # other only predicts.
                "y0": [None if i % 11 == 5 else v for i, v in enumerate(y0.tolist())],
                "y1": [None if i % 7 == 3 else v for i, v in enumerate(y1.tolist())],
                "w": w,
            },
            schema_overrides={"y0": pl.Float64, "y1": pl.Float64},
        )

    def ours(self, frame, loss, intercept, weighted, penalties):
        alpha, beta, l1, l2 = self.PENALTIES[penalties]
        spec = po.spec.ftrl(
            "m",
            targets=list(self.TARGETS),
            features=list(self.FEATURES),
            loss=loss,
            fit_intercept=intercept,
            alpha=alpha,
            beta=beta,
            l1=l1,
            l2=l2,
            half_life=float("inf"),
            min_weight=0.0,
            coef_every=0,
            **({"weight": "w"} if weighted else {}),
        )
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct
        pred = np.column_stack([out.field(f"pred_{t}").to_numpy() for t in self.TARGETS])
        coef = np.array(out.field("coef").to_list(), dtype=float)
        return pred, coef.reshape(frame.height, len(self.TARGETS), -1)

    def vws(self, frame, target, loss, intercept, weighted, penalties, *, half=True):
        """VW's prediction on each row, before the row, and its weights after
        it, intercept first as in ``coef``."""
        import vowpalwabbit

        alpha, beta, l1, l2 = self.PENALTIES[penalties]
        args = [
            "--ftrl",
            f"--ftrl_alpha {alpha!r}",
            f"--ftrl_beta {beta!r}",
            f"--l1 {l1!r}",
            f"--l2 {l2!r}",
            "--quiet",
            "-b 18",
        ]
        if loss == "logistic":
            args += ["--loss_function logistic", "--link logistic"]
        else:
            args += ["--loss_function squared", "--min_prediction -1e9", "--max_prediction 1e9"]
        if not intercept:
            args.append("--noconstant")
        vw = vowpalwabbit.Workspace(" ".join(args))
        space = vw.hash_space(" ")
        slots = [vw.hash_feature(f, space) for f in self.FEATURES]
        if intercept:
            slots.insert(0, self.CONSTANT)
        # Hashing could put two features on one weight; these do not.
        assert len({s % (1 << 18) for s in slots}) == len(slots)
        pred, coef = [], []
        for row in frame.iter_rows(named=True):
            features = "| " + " ".join(f"{f}:{row[f]!r}" for f in self.FEATURES)
            pred.append(vw.predict(features))
            y = row[target]
            if y is not None:
                importance = row["w"] if weighted else 1.0
                if loss == "logistic":
                    label = "1" if y == 1.0 else "-1"
                else:
                    label = repr(y)
                    if half:
                        importance /= 2.0
                vw.learn(f"{label} {importance!r} {features}")
            coef.append([vw.get_weight(s) for s in slots])
        vw.finish()
        return np.array(pred), np.array(coef)

    @pytest.mark.parametrize("penalties", ["defaults", "sparse"])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("intercept", [False, True])
    @pytest.mark.parametrize("loss", ["logistic", "squared"])
    def test_every_row_is_vws(self, loss, intercept, weighted, penalties):
        frame = self.rows(loss)
        pred, coef = self.ours(frame, loss, intercept, weighted, penalties)
        for j, target in enumerate(self.TARGETS):
            want_pred, want_coef = self.vws(frame, target, loss, intercept, weighted, penalties)
            np.testing.assert_allclose(pred[:, j], want_pred, rtol=0.0, atol=self.TOL)
            np.testing.assert_allclose(coef[:, j], want_coef, rtol=0.0, atol=self.TOL)
            if penalties == "sparse":
                # The l1 branch ran: the noise feature sat at exactly zero on
                # 49 to 242 of the 350 rows after the 50th, by case.
                assert (coef[50:, j, -1] == 0.0).sum() >= 40, (target, loss)

    def test_a_slip_in_the_mapping_misses(self):
        """The control: VW's squared loss at the row's full weight, where the
        mapping halves it, is off by more than 0.2, so ``1e-5`` separates a
        right mapping from a wrong one by four orders of magnitude."""
        frame = self.rows("squared")
        pred, _ = self.ours(frame, "squared", True, True, "defaults")
        slip, _ = self.vws(frame, "y0", "squared", True, True, "defaults", half=False)
        assert np.abs(pred[:, 0] - slip).max() > 0.2


class TestPoEvalIsSklearnsMetrics:
    """Review round 4 (YB19): ``po.eval``'s metrics were held to numpy
    formulas re-typed beside them, and ``ic`` and ``mse`` to nothing but
    ``from_sums`` agreeing with ``metrics``. scikit-learn and scipy compute
    every one: ``r2_score``, ``mean_squared_error``, ``scipy.stats.pearsonr``
    for ``ic``, ``accuracy_score`` for both readings of ``hit_rate`` (the sign
    of ``y`` and of ``pred``, rows with ``y == 0`` left out; or a 0.5
    threshold) and ``log_loss``. The weighted sums are the same calls with
    ``sample_weight``, and numpy's weighted covariance for ``ic``."""

    @staticmethod
    def regression() -> tuple[pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(21)
        n = 1500
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 0.8 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0.0, 1.0, n)
        y[::37] = 0.0  # signless rows, which the sign test leaves out
        df = pl.DataFrame(
            {
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": y,
                "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
                "w": rng.uniform(0.2, 3.0, n),
            }
        )
        spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=200.0)
        out = po.ModelBank([spec]).fit_predict(df)
        scored = out.with_columns(pred=pl.col("m").struct.field("pred_y")).drop_nulls("pred")
        return out, scored

    def test_the_regression_metrics(self):
        from scipy.stats import pearsonr
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

        out, scored = self.regression()
        got = po.eval.metrics(out, "m", group=["g"], min_samples=1)
        assert got["g"].to_list() == ["a", "b"]
        for row in got.iter_rows(named=True):
            part = scored.filter(pl.col("g") == row["g"])
            y, pred = part["y"].to_numpy(), part["pred"].to_numpy()
            signed = y != 0.0
            assert row["n"] == len(y) and (~signed).sum() > 5
            assert row["r2"] == pytest.approx(r2_score(y, pred), rel=1e-12)
            assert row["mse"] == pytest.approx(mean_squared_error(y, pred), rel=1e-12)
            assert row["ic"] == pytest.approx(pearsonr(pred, y).statistic, rel=1e-12)
            want = accuracy_score(np.sign(y[signed]), np.sign(pred[signed]))
            assert row["hit_rate"] == pytest.approx(want, rel=1e-12)

    def test_the_weighted_sums(self):
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

        out, scored = self.regression()
        got = po.eval.from_sums(po.eval.sums(out, "m", weight="w"), min_samples=1).row(
            0, named=True
        )
        y, pred, w = (scored[c].to_numpy() for c in ("y", "pred", "w"))
        signed = y != 0.0
        assert got["r2"] == pytest.approx(r2_score(y, pred, sample_weight=w), rel=1e-12)
        assert got["mse"] == pytest.approx(mean_squared_error(y, pred, sample_weight=w), rel=1e-12)
        c = np.cov(pred, y, aweights=w)
        assert got["ic"] == pytest.approx(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]), rel=1e-12)
        want = accuracy_score(np.sign(y[signed]), np.sign(pred[signed]), sample_weight=w[signed])
        assert got["hit_rate"] == pytest.approx(want, rel=1e-12)

    def test_the_binary_reading(self):
        from scipy.stats import pearsonr
        from sklearn.metrics import accuracy_score, log_loss, r2_score

        rng = np.random.default_rng(22)
        n = 6000
        x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
        label = (rng.random(n) < 1.0 / (1.0 + np.exp(-(1.2 * x0 - 0.8 * x1)))).astype(float)
        df = pl.DataFrame({"x0": x0, "x1": x1, "y": label})
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            loss="logistic",
            learning_rate=0.05,
            half_life=float("inf"),
            min_weight=50.0,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        got = po.eval.metrics(out, "m", binary=True, min_samples=1).row(0, named=True)
        scored = out.with_columns(pred=pl.col("m").struct.field("pred_y")).drop_nulls("pred")
        y, p = scored["y"].to_numpy(), scored["pred"].to_numpy()
        assert p.min() > 1e-15 and p.max() < 1.0 - 1e-15, "the clip is not in play"
        assert got["hit_rate"] == pytest.approx(accuracy_score(y, p > 0.5), rel=1e-12)
        assert got["log_loss"] == pytest.approx(log_loss(y, p), rel=1e-12)
        assert got["r2"] == pytest.approx(r2_score(y, p), rel=1e-12), "the Brier skill score"
        assert got["ic"] == pytest.approx(pearsonr(p, y).statistic, rel=1e-12)
        assert 0.6 < got["hit_rate"] < 0.95


class TestShrinkToTheIdentityIsSklearnsLedoitWolf:
    """Review round 4 (YB4): ``shrink(target="identity")`` kept a term of the
    constant-correlation target's optimal intensity, ``sum_i pi_ii``, which
    belongs to that target's diagonal ``f_ii = s_ii``. The identity target's
    diagonal is the mean variance, and Ledoit and Wolf (2004, JMVA) take
    ``rho = 0`` for it: ``alpha = pi / (T gamma)``, clipped to ``[0, 1]``.
    ``sklearn.covariance.ledoit_wolf`` computes exactly that, and the
    intensities were 16-36% below it on the four samples here."""

    def test_the_intensity_and_the_shrunk_matrix_are_sklearns(self):
        from sklearn.covariance import ledoit_wolf

        rng = np.random.default_rng(2)
        alphas = []
        for n, k in ((120, 6), (60, 5), (30, 10), (500, 4), (8, 12)):
            f = rng.standard_normal((n, 2))
            e = rng.standard_normal((n, k))
            x = np.column_stack(
                [0.9 * f[:, 0] + 0.436 * e[:, i] for i in range(k // 2)]
                + [0.3 * f[:, 1] + 0.954 * e[:, i] for i in range(k // 2, k)]
            )
            d = x - x.mean(axis=0)
            got, alpha = po.corr.shrink(d.T @ d / n, target="identity", x=x)
            want, want_alpha = ledoit_wolf(x)
            assert alpha == pytest.approx(want_alpha, rel=1e-12), (n, k)
            np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-13)
            alphas.append(alpha)
        assert min(alphas) > 0.0 and max(alphas) <= 1.0
        assert sum(0.0 < a < 1.0 for a in alphas) >= 4, f"strictly inside, mostly: {alphas}"


class TestNearestIsStatsmodelsCorrNearest:
    """Review round 4 (YB7, YB19): ``statsmodels.stats.correlation_tools.
    corr_nearest`` runs Higham's alternating projections with Dykstra's
    correction too, and stops on another rule: no eigenvalue clipped. An
    iterate converging to the boundary of the cone keeps a tiny negative
    eigenvalue, so it runs its iteration limit (``n_fact`` times ``k``
    sweeps), far past convergence at a factor of about 3 a sweep; its
    ``IterationLimitWarning`` says so and is silenced here. It takes the
    input's diagonal as already 1 -- given an input that is PSD it returns
    it unchanged, as Higham's 4x4, whose diagonal is 2, is -- so the inputs
    here are almost-correlation matrices, the case both exist for."""

    def test_highams_three_by_three_and_random_unit_diagonals(self):
        import warnings

        from statsmodels.stats.correlation_tools import corr_nearest
        from statsmodels.tools.sm_exceptions import IterationLimitWarning

        rng = np.random.default_rng(1)
        inputs = [np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])]
        for _ in range(12):
            k = int(rng.integers(2, 8))
            m = rng.standard_normal((k, k))
            m = (m + m.T) / 2
            np.fill_diagonal(m, 1.0)
            inputs.append(m)
        repaired = 0
        for a in inputs:
            ours, _, iters = po.corr.nearest(a, tol=1e-12, max_iter=10_000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IterationLimitWarning)
                theirs = corr_nearest(a, n_fact=300)
            np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-10)
            repaired += iters > 1
        assert repaired >= 8, "most inputs were not correlation matrices to begin with"


def _standardizer_moves(
    t: np.ndarray, x: np.ndarray, w: np.ndarray, half_life: float, fit_intercept: bool = True
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """What `kalman`'s standardizer does to the rows, from the moments'
    definition (`kalman.rs`'s module doc; docs/PLAN.md tasks 206 and 211):
    each row standardized against the EW moments of the rows before it,
    every processed row at its weight, a scale of 1 where there is no spread
    -- ``[1, (x - m) / s]``, or through the origin ``x / s`` with ``s`` the
    root mean square; and, from the first row, the change of coordinates
    each row of positive weight makes from the moments before it to those
    after it: ``A_00 = 1``, ``A_0i = (m'_i - m_i) / s_i`` and ``A_ii = s'_i /
    s_i`` (``A = diag(s' / s)`` through the origin). ``F[i]`` is the change
    the row before row ``i`` made, the transition row ``i``'s predict step
    takes, ``I`` where there was none; ``remapped[i]`` whether row ``i`` made
    one; ``usable[i]`` which slots' scales row ``i`` reads are usable, the
    intercept's always. A spread of the rounding a mean leaves on one row,
    1e-34 at these levels where the filter's is exactly 0, is none."""
    n, k = x.shape
    off = 1 if fit_intercept else 0
    Z = np.ones((n, k + off))
    usable = np.ones((n, k + off), dtype=bool)
    moments = []
    for i in range(n + 1):
        u = w[:i] * 0.5 ** ((t[i - 1] - t[:i]) / half_life) if i else np.zeros(0)
        m, s, ok = np.zeros(k), np.ones(k), np.zeros(k, dtype=bool)
        if u.sum() > 0.0:
            if fit_intercept:
                m = (u[:, None] * x[:i]).sum(axis=0) / u.sum()
                v = (u[:, None] * (x[:i] - m) ** 2).sum(axis=0) / u.sum()
                ok = v > 1e-20 * (m * m + 1.0)
            else:
                v = (u[:, None] * x[:i] ** 2).sum(axis=0) / u.sum()
                ok = v > 0.0
            s = np.where(ok, np.sqrt(np.maximum(v, 0.0)), 1.0)
        moments.append((m, s))
        if i < n:
            Z[i, off:] = (x[i] - m) / s
            usable[i, off:] = ok
    F = [np.eye(k + off) for _ in range(n)]
    remapped = w > 0.0
    for i in range(n - 1):
        if remapped[i]:
            (m, s), (m2, s2) = moments[i], moments[i + 1]
            if fit_intercept:
                F[i + 1][0, 1:] = (m2 - m) / s
            F[i + 1][off:, off:] = np.diag(s2 / s)
    return Z, F, remapped, usable


def _median_of_three(rows: list[tuple[float, float]]) -> float:
    """The noise a standardized ``kalman`` prior is sized from (task 214):
    the weighted median of the first three rows' squared innovations, by
    ``numpy``'s sort and cumulative weight, over ``scipy``'s median of a
    chi-squared of one degree; 0 before there are three."""
    from scipy import stats

    if len(rows) < 3:
        return 0.0
    v, w = (np.array(c) for c in zip(*rows, strict=True))
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    median = v[order][np.searchsorted(cum, 0.5 * w.sum())]
    return float(median / stats.chi2(1).median())


def _filterpy_run(
    kalman: Any,
    t: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    half_life: float,
    *,
    coef_half_life: float | list[float],
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float = 1.0,
    F: list[np.ndarray] | None = None,
    usable: np.ndarray | None = None,
    revert: float | list[float] = float("inf"),
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """filterpy's run of `kalman`'s documented recursion (`kalman.rs`'s
    module doc; docs/PLAN.md task 211) over the regressor rows ``Z``, one
    target: per row its prediction, its readiness statistic and the
    posterior ``P`` after it (``None`` while unsized), which after a row
    that observes nothing carries the noise of the clock since the last
    observation as the next one will charge it.

    Every row runs filterpy's ``predict`` with the transition ``F = diag(2^(-d
    / r)) A``, ``A`` the change of coordinates the row before made (``F[i]``
    here, `_standardizer_moves`; the identity unstandardized), and the
    process noise only on a row that observes the target: ``Q D**2``, ``D``
    the clock since the last such row (on a reverting slot ``((1 - 2**(-D /
    r)) / theta)**2`` in place of ``D**2``, task 214), ``Q`` from ``coef_half_life`` --
    ``sigma**2 (ln 2 / h)**2`` per coefficient, 0 at ``inf`` -- or ``q``. The
    noise is ``obs_var``, else the residual variance, else the row's own
    innovation squared, read after the transition. Unstandardized
    (``usable`` None) the first row with a noise sizes ``P = p0 * R * I``,
    and ``obs_var`` sizes it before the first row; standardized each slot's
    ``P_ii = p0 * R`` is set on the first observed row its scale is usable
    (``usable[i]``), ``R`` the weighted median of the first three observed
    rows' squared innovations over the median of a chi-squared of one degree
    (task 214), or ``obs_var``, the intercept's then
    from the start; an unsized slot takes no noise. The readiness statistic
    is ``sqrt(1 + z' P⁻ z / R)``, ``P⁻`` the prior with the noise of the
    clock since the last observation, this row's included, ``R`` the noise
    as the state holds it: infinite while ``P`` is unsized or there is none
    (docs/PLAN.md task 116)."""
    n, k1 = Z.shape
    hl = np.broadcast_to(np.asarray(coef_half_life, dtype=float), (k1,))
    r = np.broadcast_to(np.asarray(revert, dtype=float), (k1,))
    standardize = usable is not None
    kf = kalman.KalmanFilter(dim_x=k1, dim_z=1)
    kf.x = np.zeros((k1, 1))
    kf.P = np.zeros((k1, k1))
    sized = np.zeros(k1, dtype=bool)
    if obs_var is not None:
        # Every slot unstandardized; standardized the intercept, whose
        # column is all ones.
        sized[:] = not standardize
        sized[0] |= bool(np.all(Z[:, 0] == 1.0))
        kf.P = np.diag(np.where(sized, p0 * obs_var, 0.0))
    # The first three observed rows' squared innovations and weights (task
    # 214).
    basis: list[tuple[float, float]] = []
    gap = 0.0
    sig2 = wsig = wj = 0.0
    pred = np.full(n, np.nan)
    infl = np.full(n, np.inf)
    posts: list[np.ndarray | None] = []

    def noise_of(s2: float) -> np.ndarray:
        if q is not None:
            return np.asarray(q, dtype=float)
        finite = np.where(np.isinf(hl), 1.0, hl)
        return np.where(np.isinf(hl), 0.0, s2 * (np.log(2.0) / finite) ** 2)

    def gap2(g: float) -> np.ndarray:
        # `D**2`, and on a reverting slot the square of the gap the
        # reversion damps, `((1 - 2**(-D/r)) / theta)**2` (task 214).
        finite = np.where(np.isinf(r), 1.0, r)
        damped = ((1.0 - 0.5 ** (g / finite)) * finite / np.log(2.0)) ** 2
        return np.where(np.isinf(r), g * g, damped)

    for i in range(n):
        d = 0.0 if i == 0 else t[i] - t[i - 1]
        lam = 0.5 ** (d / half_life)
        gap += d
        Fi = np.diag(0.5 ** (d / r)) @ (np.eye(k1) if F is None else F[i])
        z = Z[i]
        seen = not np.isnan(y[i]) and w[i] > 0.0
        x_prior = Fi @ kf.x[:, 0]
        e2 = (y[i] - z @ x_prior) ** 2 if seen else 0.0
        # The row's noise, and the noise the state holds.
        held = obs_var if obs_var is not None else sig2
        s2 = held if obs_var is not None or sig2 > 0.0 else e2
        P_prior = Fi @ kf.P @ Fi.T
        if sized.any() and held > 0.0:
            prior = P_prior + np.diag(np.where(sized, noise_of(held) * gap2(gap), 0.0))
            infl[i] = np.sqrt(1.0 + z @ prior @ z / held)
        Q = np.diag(np.where(sized, noise_of(s2) * gap2(gap), 0.0)) if seen else 0.0 * P_prior
        kf.predict(Q=Q, F=Fi)
        if seen:
            gap = 0.0
            if standardize:
                if obs_var is None and e2 > 0.0 and len(basis) < 3:
                    basis.append((e2, w[i]))
                size = obs_var if obs_var is not None else _median_of_three(basis)
                for a in range(k1):
                    if not sized[a] and usable[i][a] and size > 0.0:
                        kf.P[a, a] = p0 * size
                        sized[a] = True
            elif not sized.any() and s2 > 0.0:
                kf.P = np.eye(k1) * p0 * s2
                sized[:] = True
        if wj > 0.0:
            pred[i] = z @ kf.x[:, 0]
        if not seen:
            wj *= lam
            wsig *= lam
            if not sized.any():
                posts.append(None)
                continue
            # `P` as it stands carries the noise of the clock since the
            # last observation, in the coordinates of the moments after this
            # row, which the next row's transition maps to: here, before it,
            # mapped back by that change.
            now = np.diag(np.where(sized, noise_of(held) * gap2(gap), 0.0)) if held > 0.0 else 0.0
            A = np.linalg.inv(F[i + 1]) if F is not None and i + 1 < n else np.eye(k1)
            posts.append(kf.P + A @ now @ A.T)
            continue
        if s2 > 0.0:
            kf.update(y[i], R=s2 / w[i], H=z[None, :])
        aged = lam * wsig
        wsig = aged
        if not np.isnan(pred[i]):
            res = y[i] - pred[i]
            sig2 = (aged * sig2 + w[i] * res * res) / (aged + w[i])
            wsig = aged + w[i]
        wj = lam * wj + w[i]
        posts.append(kf.P.copy() if sized.any() else None)
    return pred, infl, posts


def _filterpy_kalman(
    kalman: Any,
    t: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    half_life: float,
    *,
    coef_half_life: float | list[float],
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float = 1.0,
    F: list[np.ndarray] | None = None,
    usable: np.ndarray | None = None,
) -> np.ndarray:
    """filterpy's predictions over the regressor rows ``Z`` under `kalman`'s
    documented recursion (`_filterpy_run`), with the three settings
    `TestKalmanZeroWeightRow` leaves at their defaults: an explicit ``q``
    makes the process noise ``diag(q) D**2`` whatever the residuals; an
    ``obs_var`` is the observation noise on every row and sizes the prior
    before the first row; and a ``coef_half_life`` per coefficient, ``inf``
    pinning one, gives each its own ``sigma**2 (ln 2 D / h)**2``, 0 at
    ``inf``. ``F[i]`` and ``usable``, where given, are the standardizer's
    (`_standardizer_moves`)."""
    return _filterpy_run(
        kalman,
        t,
        Z,
        y,
        w,
        half_life,
        coef_half_life=coef_half_life,
        q=q,
        obs_var=obs_var,
        p0=p0,
        F=F,
        usable=usable,
    )[0]


class TestKalmanSettingsAreFilterpy:
    """The `kalman` settings `kalman_ref` alone held, which restates the
    core (review 2026-10-06, TA4): an explicit ``q``, a fixed ``obs_var`` with
    ``p0``, a ``coef_half_life`` per coefficient with ``inf`` pinning, and no
    intercept under the default ``standardize``. Each is a plain
    configuration of filterpy's ``KalmanFilter``. The first three run
    unstandardized, so ``kf.x`` is the coefficient vector; the last on rows
    scaled, not centred, by each feature's EW root mean square over the rows
    before (``kalman.rs``'s module doc), computed here in numpy by sums over
    the raw rows. Measured: 1.1e-15 relative or less, 1.0e-14 with no
    intercept. ``share_p`` is task 186's."""

    TOL = 1e-11

    @staticmethod
    def rows(seed: int, scaled: bool):
        rng = np.random.default_rng(seed)
        n = 300
        raw = rng.normal(0.0, 1.0, (n, 2))
        drift = np.arange(n) / n
        y = 0.5 + (1.5 - drift) * raw[:, 0] + (-0.8 + 2.0 * drift) * raw[:, 1]
        y = y + rng.normal(0.0, 0.3, n)
        # Levels and scales of their own where standardizing is the point.
        x = raw * np.array([5.0, 0.2]) + np.array([3.0, -40.0]) if scaled else raw
        late = np.arange(n) > 20
        y_seen = np.where((np.arange(n) % 9 == 4) & late, np.nan, y)
        w = np.where((np.arange(n) % 11 == 7) & late, 0.0, rng.uniform(0.5, 1.5, n))
        steps = np.ones(n)
        steps[150] = 4.0
        steps[230:] = 0.5
        t = np.cumsum(steps) - steps[0]
        frame = pl.DataFrame(
            {
                "t": t,
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": [None if np.isnan(v) else float(v) for v in y_seen],
                "w": w,
            },
            schema={c: pl.Float64 for c in ("t", "x0", "x1", "y", "w")},
        )
        return frame, t, x, y_seen, w

    @staticmethod
    def ours(frame: pl.DataFrame, **kw: Any) -> np.ndarray:
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1e9,
            weight="w",
            half_life=30.0,
            min_weight=0.0,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("pred_y").to_numpy()

    CASES = {
        # `q` alone: it is the process noise `coef_half_life` would derive,
        # and `kalman` refuses the two together (review round 4, PC6); the
        # oracle reads `q` and leaves the half-life unread.
        "explicit q": (
            dict(q=[0.0, 0.01, 0.02], standardize=False, p0=1.0),
            dict(coef_half_life=50.0, q=[0.0, 0.01, 0.02]),
        ),
        "obs_var and p0": (
            dict(coef_half_life=50.0, obs_var=0.25, p0=4.0, standardize=False),
            dict(coef_half_life=50.0, obs_var=0.25, p0=4.0),
        ),
        "a half-life per coefficient, two pinned": (
            dict(coef_half_life=[float("inf"), 30.0, float("inf")], standardize=False, p0=1.0),
            dict(coef_half_life=[float("inf"), 30.0, float("inf")]),
        ),
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_the_filter_is_filterpys(self, case):
        import filterpy.kalman as kalman

        ours, theirs = self.CASES[case]
        frame, t, x, y, w = self.rows(43, scaled=False)
        got = self.ours(frame, **ours)
        Z = np.column_stack([np.ones(len(t)), x])
        want = _filterpy_kalman(kalman, t, Z, y, w, 30.0, **theirs)
        assert np.isfinite(want[1:]).all()
        np.testing.assert_allclose(got, want, rtol=self.TOL, atol=1e-13)

    def test_with_no_intercept_the_rows_are_scaled_and_not_centred(self):
        import filterpy.kalman as kalman

        frame, t, x, y, w = self.rows(44, scaled=True)
        got = self.ours(frame, coef_half_life=50.0, fit_intercept=False, p0=1.0)
        # No moments before the first row: no usable scale, and no prior.
        # From then the filter follows each move of the scales, `F = diag(s'
        # / s)` (task 211).
        Z, F, _, usable = _standardizer_moves(t, x, w, 30.0, fit_intercept=False)
        want = _filterpy_kalman(kalman, t, Z, y, w, 30.0, coef_half_life=50.0, F=F, usable=usable)
        np.testing.assert_allclose(got, want, rtol=self.TOL, atol=1e-13)

    def test_a_slip_in_the_mapping_misses(self):
        """The control: filterpy with one entry of ``q`` a tenth off parts
        from the bank by far more than the tolerance."""
        import filterpy.kalman as kalman

        frame, t, x, y, w = self.rows(43, scaled=False)
        got = self.ours(frame, q=[0.0, 0.01, 0.02], standardize=False, p0=1.0)
        Z = np.column_stack([np.ones(len(t)), x])
        slip = _filterpy_kalman(kalman, t, Z, y, w, 30.0, coef_half_life=50.0, q=[0.0, 0.011, 0.02])
        assert np.nanmax(np.abs(got - slip)) > 1e-6


class TestSgdIsScikitLearnsSgd:
    """`sgd`'s losses and schedules against scikit-learn's ``SGDRegressor``
    and ``SGDClassifier(loss="log_loss")``, ``partial_fit`` one row at a
    time, the prediction read before each (review 2026-10-06, TA4). Both
    take one gradient step per row, ``b -= lr * d * z * w`` with the
    intercept unpenalised and the weight scaling the step, so under
    ``learning_rate="constant"`` they are the same recursion: Huber's
    clamped residual, the epsilon-insensitive sign outside the tube, and
    the logistic ``sigmoid(eta) - y``, which scikit-learn writes ``-y / (1 +
    exp(y eta))`` on labels of +-1. ``inv_scaling``'s ``lr / (1 +
    weight_sum) ** power`` is scikit-learn's ``eta0 / t ** power_t`` at unit
    weights with no decay, ``t`` counting rows from 1. Measured: 1.0e-15
    relative or less. The Poisson loss and the AdaGrad schedule have no
    scikit-learn twin and stay with ``sgd_ref``. Nor, since task 195, has a
    Huber cut or a tube of positive width: our cut is in units of the
    residual's EW std (U1) and our tube in the target's own EW std (task
    202), where scikit-learn's ``epsilon`` is a fixed number in the target's
    units, so the epsilon-insensitive loss is held at ``eps = 0`` -- the sign of the
    residual, which is the step's whole shape outside a tube -- and the
    clamp to ``sgd_ref``. Ours steps on the raw features here, as
    scikit-learn's ``partial_fit`` does."""

    TOL = 1e-12
    LR = 0.02

    @staticmethod
    def rows(n: int = 400, seed: int = 3) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * rng.standard_t(3, n)
        p = 1.0 / (1.0 + np.exp(-(0.3 + x @ np.array([1.2, -0.7]))))
        yb = (rng.random(n) < p).astype(float)
        w = rng.uniform(0.5, 1.5, n)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "yb": yb, "w": w})

    @staticmethod
    def ours(frame: pl.DataFrame, target: str, weighted: bool, **kw: Any) -> np.ndarray:
        spec = po.spec.sgd(
            "s",
            targets=[target],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=0.0,
            weight="w" if weighted else None,
            standardize=False,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(frame)["s"].struct
        return out.field(f"pred_{target}").to_numpy().astype(float)

    @staticmethod
    def sklearns(est: Any, frame: pl.DataFrame, target: str, weighted: bool) -> np.ndarray:
        x = frame.select("x0", "x1").to_numpy()
        y, w = frame[target].to_numpy(), frame["w"].to_numpy()
        classifier = hasattr(est, "predict_proba")
        pred = np.full(len(y), 0.5 if classifier else 0.0)
        for i in range(len(y)):
            row = x[i : i + 1]
            if i:
                pred[i] = est.predict_proba(row)[0, 1] if classifier else est.predict(row)[0]
            kw: dict[str, Any] = {"sample_weight": [w[i]]} if weighted else {}
            if classifier:
                kw["classes"] = [0.0, 1.0]
            est.partial_fit(row, y[i : i + 1], **kw)
        return pred

    @pytest.mark.parametrize("weighted", [False, True], ids=["unit weights", "weighted"])
    @pytest.mark.parametrize(
        ("loss", "level"),
        [
            ("squared", 0.0),
            ("squared", -1e3),
            ("squared", CROSSING),
            ("epsilon_insensitive", 0.0),
            ("epsilon_insensitive", -1e3),
            ("epsilon_insensitive", CROSSING),
            ("logistic", 0.0),
        ],
    )
    def test_every_row_is_scikit_learns(self, loss, weighted, level):
        """A regression's target at a level of -1,000 too, every one below
        zero (docs/PLAN.md task 209 (c)), and at a level running from +1,000
        to -1,000 across the rows (task 209 (d)), with the gradient's clip
        lifted: scikit-learn has none, and the first residuals are the level.
        The logistic loss's target is a label, 0 or 1, with no level to
        take."""
        from sklearn.linear_model import SGDClassifier, SGDRegressor

        frame = self.rows()
        frame = frame.with_columns(pl.col("y") + _level(level, frame.height))
        if level == -1e3:
            assert (frame["y"] < 0).all(), "the case: every target below zero"
        sk: dict[str, Any] = dict(
            penalty=None, learning_rate="constant", eta0=self.LR, shuffle=False
        )
        clip = {"clip_gradient": float("inf")} if level else {}
        est: Any
        if loss == "squared":
            got = self.ours(frame, "y", weighted, loss=loss, learning_rate=self.LR, **clip)
            est, target = SGDRegressor(loss="squared_error", **sk), "y"
        elif loss == "epsilon_insensitive":
            got = self.ours(frame, "y", weighted, loss=loss, eps=0.0, learning_rate=self.LR, **clip)
            est, target = SGDRegressor(loss="epsilon_insensitive", epsilon=0.0, **sk), "y"
        else:
            got = self.ours(frame, "yb", weighted, loss=loss, learning_rate=self.LR)
            est, target = SGDClassifier(loss="log_loss", **sk), "yb"
        want = self.sklearns(est, frame, target, weighted)
        np.testing.assert_allclose(got, want, rtol=self.TOL, atol=1e-15 * _magnitude(level))

    def test_inv_scaling_at_unit_weights_is_scikit_learns_invscaling(self):
        from sklearn.linear_model import SGDRegressor

        frame = self.rows()
        got = self.ours(
            frame,
            "y",
            False,
            loss="squared",
            learning_rate=0.1,
            schedule="inv_scaling",
            power=0.5,
        )
        est = SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="invscaling",
            eta0=0.1,
            power_t=0.5,
            shuffle=False,
        )
        want = self.sklearns(est, frame, "y", False)
        np.testing.assert_allclose(got, want, rtol=self.TOL, atol=1e-15)

    def test_a_slip_in_the_mapping_misses(self):
        """The control: scikit-learn's epsilon-insensitive loss at a tube of
        0.05 where ours has none parts from the bank by far more than the
        tolerance."""
        from sklearn.linear_model import SGDRegressor

        frame = self.rows()
        got = self.ours(
            frame, "y", False, loss="epsilon_insensitive", eps=0.0, learning_rate=self.LR
        )
        est = SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=0.05,
            penalty=None,
            learning_rate="constant",
            eta0=self.LR,
            shuffle=False,
        )
        assert np.abs(got - self.sklearns(est, frame, "y", False)).max() > 1e-6


class TestHmmIsHmmlearns:
    """`hmm` at fixed parameters against hmmlearn's ``GaussianHMM`` (review
    2026-10-06, TA4): the Hamilton filter's probabilities and each row's
    log-likelihood, where the longhand in ``tests/test_hmm.py`` restated the
    recursion. ``learn = False`` with given means, covariances and
    transition: each state's emission is the Gaussian of its covariance plus
    ``precision_prior`` on the diagonal, and the filter starts uniform, so
    hmmlearn's start is ``[1/2, 1/2] @ Pi``. ``filtered_<k>`` on a row is
    read before the row, so it is hmmlearn's filtered probability at the row
    before, the last of ``predict_proba`` over the rows up to it;
    ``predicted_<k>`` is that times ``Pi``; and ``loglik`` is ``log p(x_t |
    x_<t)``, the difference of ``score`` over the rows up to it and up to the
    row before. Measured: 1.4e-13, the probabilities absolute and the
    log-likelihoods relative."""

    TOL = 1e-10

    def test_the_filter_and_the_loglik_are_hmmlearns(self):
        from hmmlearn.hmm import GaussianHMM

        rng = np.random.default_rng(23)
        n = 500
        state = (np.arange(n) // 40) % 2
        mix = np.array([[1.0, 0.3], [0.0, 0.8]])
        x = np.where(state[:, None] == 0, -1.0, 1.0) + rng.normal(size=(n, 2)) @ mix
        means = [-1.0, -0.8, 1.2, 0.9]
        covs = [1.0, 0.3, 0.3, 0.8, 1.4, -0.2, -0.2, 0.7]
        pi = np.array([[0.93, 0.07], [0.12, 0.88]])
        ridge = 1e-2
        spec = po.spec.hmm(
            "h",
            features=["x0", "x1"],
            k=2,
            precision_prior=ridge,
            half_life=1e9,
            learn=False,
            means=means,
            covs=covs,
            transition=pi.flatten().tolist(),
            transition_prior=1.0,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
        out = po.ModelBank([spec]).fit_predict(frame)["h"].struct.unnest()

        model = GaussianHMM(n_components=2, covariance_type="full", init_params="", params="")
        model.startprob_ = np.array([0.5, 0.5]) @ pi
        model.transmat_ = pi
        model.means_ = np.array(means).reshape(2, 2)
        model.covars_ = np.array(covs).reshape(2, 2, 2) + ridge * np.eye(2)
        filtered = np.array([model.predict_proba(x[: t + 1])[-1] for t in range(n)])
        scores = np.array([model.score(x[: t + 1]) for t in range(n)])
        loglik = np.diff(scores, prepend=0.0)

        got = out.select("filtered_0", "filtered_1").to_numpy()
        np.testing.assert_allclose(got[0], [0.5, 0.5], rtol=0, atol=0)
        np.testing.assert_allclose(got[1:], filtered[:-1], rtol=0, atol=self.TOL)
        predicted = out.select("predicted_0", "predicted_1").to_numpy()
        np.testing.assert_allclose(predicted[1:], filtered[:-1] @ pi, rtol=0, atol=self.TOL)
        np.testing.assert_allclose(out["loglik"].to_numpy(), loglik, rtol=self.TOL, atol=self.TOL)
        # And the states are told apart: the filter is no flat 1/2.
        assert np.abs(got[1:, 0] - 0.5).max() > 0.45


class TestRlsIsPadasips:
    """`rls` against padasip's ``FilterRLS`` (review 2026-10-06, TA4), where
    only ``rls_ref`` and ``rls == ewridge(ridge_scale="sum")`` held it.
    ``FilterRLS(n, mu, eps)`` keeps ``R = A^-1`` from ``A_0 = eps I`` with
    ``A_t = mu A_{t-1} + z z'`` and steps ``w += R z (y - w'z)``: the classic
    exponentially weighted recursion, which at unit weights on a row clock is
    `rls` at ``mu = 2 ** (-1 / half_life)``. The bank's first row takes no
    decay (its step is 0), where ``FilterRLS`` decays ``A_0`` on every row,
    so ``eps = ridge / mu`` gives both the same ``A`` after it. `rls`
    withholds its first row, before any row is learned; padasip predicts 0
    there. Measured: 1.6e-14 relative."""

    TOL = 1e-11

    @pytest.mark.parametrize("level", [0.0, CROSSING])
    @pytest.mark.parametrize(
        ("half_life", "ridge"), [(50.0, 1.0), (200.0, 0.1), (float("inf"), 1.0)]
    )
    def test_every_row_is_padasips(self, half_life, ridge, level):
        """And with the target's level running from +1,000 to -1,000 across
        the rows, through zero (docs/PLAN.md task 209 (d))."""
        import padasip

        rng = np.random.default_rng(31)
        n = 400
        x = rng.normal(size=(n, 2))
        y = _level(level, n) + 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.3 * rng.normal(size=n)
        spec = po.spec.rls(
            "r",
            targets=["y"],
            features=["x0", "x1"],
            half_life=half_life,
            delta=ridge,
            min_weight=0.0,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        got = po.ModelBank([spec]).fit_predict(frame)["r"].struct.field("pred_y").to_numpy()
        mu = 0.5 ** (1.0 / half_life)
        rls = padasip.filters.FilterRLS(n=3, mu=mu, eps=ridge / mu, w="zeros")
        want = np.empty(n)
        for i in range(n):
            z = np.array([1.0, *x[i]])
            want[i] = rls.predict(z)
            rls.adapt(y[i], z)
        assert np.isnan(got[0]) and np.isfinite(got[1:]).all()
        atol = 1e-13 * _magnitude(level)
        np.testing.assert_allclose(got[1:], want[1:], rtol=self.TOL, atol=atol)


def _filterpy_kalman_readiness(
    kalman: Any,
    t: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    half_life: float,
    *,
    coef_half_life: float,
    revert: float | list[float] = float("inf"),
    obs_var: float | None = None,
    p0: float = 1.0,
    remap: list[np.ndarray] | None = None,
    usable: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """filterpy's run of `kalman`'s documented recursion (`_filterpy_run`),
    with the transition ``F = diag(2^(-d / r))`` of
    `TestAMeanRevertingKalmanIsFilterpy` -- after ``remap[i]``, the change of
    coordinates the row before made, where given (`_standardizer_moves`) --
    returning per row what docs/PLAN.md task 116 reads from it: the
    readiness statistic ``sqrt(1 + z' P⁻ z / R)`` before the row's update --
    ``P⁻`` filterpy's ``P`` carried through the transition with the process
    noise of the clock since the last observation, this row's included
    (task 211), ``R`` the noise as the state holds it, ``obs_var`` or else
    the residual variance, infinite while ``P`` is unsized or the target has
    no residual variance -- and the posterior ``P`` after the row, ``None``
    while unsized."""
    _, infl, posts = _filterpy_run(
        kalman,
        t,
        Z,
        y,
        w,
        half_life,
        coef_half_life=coef_half_life,
        obs_var=obs_var,
        p0=p0,
        F=remap,
        usable=usable,
        revert=revert,
    )
    return infl, posts


class TestKalmansErrorInflationIsFilterpysPrior:
    """docs/PLAN.md task 116 (A): `kalman`'s ``error_inflation`` is
    ``sqrt(1 + z' P⁻ z / R)``, the prior predictive variance of the row over
    its noise: ``P⁻`` the covariance carried through the transition and the
    process noise the row's clock gap adds, which is filterpy's
    ``KalmanFilter.predict``, and ``R`` the noise the state holds. Null
    targets, zero weights, uneven steps and a reversion are the rows
    `TestKalmanSettingsAreFilterpy` runs; standardized, the rows are
    `TestAStandardizedKalmanIsFilterpy`'s. Its covariance after the row is the
    coefficients' (F): ``se_coef**2`` is the diagonal of ``T P Tᵀ``, ``T`` the
    map `coef` reads the coefficients out by, the identity unstandardized."""

    TOL = 1e-9

    CASES = {
        "random walk": dict(standardize=False),
        "a reversion, the intercept a walk": dict(
            standardize=False, revert_half_life=[float("inf"), 40.0, 40.0]
        ),
        "obs_var and p0": dict(standardize=False, obs_var=0.25, p0=4.0),
        "standardized": dict(standardize=True),
    }

    @staticmethod
    def run(case: dict[str, Any], se: bool = False):
        import filterpy.kalman as kalman

        frame, t, x, y, w = TestKalmanSettingsAreFilterpy.rows(47, scaled=case["standardize"])
        spec = po.spec.kalman(
            "k",
            targets=["y"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=1e9,
            weight="w",
            half_life=30.0,
            min_weight=0.0,
            coef_half_life=50.0,
            p0=case.get("p0", 1.0),
            obs_var=case.get("obs_var"),
            revert_half_life=case.get("revert_half_life"),
            standardize=case["standardize"],
            emit_error_inflation=True,
            coef_every=0,
            **({"emit_se_coef": True} if se else {}),
        )
        out = po.ModelBank([spec]).fit_predict(frame)["k"].struct.unnest()
        remap, usable = None, None
        if case["standardize"]:
            # One row before has no spread, and the filter takes a scale of 1
            # (`variance_is_usable`), where the sums here leave a variance of
            # rounding, 1e-34 at these levels: `_standardizer_moves` reads
            # that as none.
            Z, remap, _, usable = _standardizer_moves(t, x, w, 30.0)
        else:
            Z = np.column_stack([np.ones(len(t)), x])
        infl, posts = _filterpy_kalman_readiness(
            kalman,
            t,
            Z,
            y,
            w,
            30.0,
            coef_half_life=50.0,
            revert=case.get("revert_half_life", float("inf")),
            obs_var=case.get("obs_var"),
            p0=case.get("p0", 1.0),
            remap=remap,
            usable=usable,
        )
        return out, infl, posts, t, x, w

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_the_row_field_is_the_prior_predictive_variance_over_the_noise(self, case):
        out, want, _, _, _, _ = self.run(self.CASES[case])
        got = out["error_inflation_y"].to_numpy()
        finite = np.isfinite(want)
        # Infinite until P is sized and the target has a noise: a row or two.
        assert finite.sum() > len(want) - 5, finite.sum()
        np.testing.assert_array_equal(np.isfinite(got), finite)
        np.testing.assert_allclose(got[finite], want[finite], rtol=self.TOL)
        assert (got[finite] > 1.0).all()

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_se_coef_is_the_posterior_covariance_in_coefs_units(self, case):
        out, _, posts, t, x, w = self.run(self.CASES[case], se=True)
        se = out["se_coef"].to_list()
        n = len(t)
        if self.CASES[case]["standardize"]:
            remapped = _standardizer_moves(t, x, w, 30.0)[2]
        for i in range(n):
            if posts[i] is None:
                assert se[i] is None or all(v is None for v in se[i]), (i, se[i])
                continue
            P = posts[i]
            if self.CASES[case]["standardize"]:
                # The coefficients are read out with the moments after the
                # row: c_i = b_i / s_i, c_0 = b_0 - sum_i c_i m_i. `P` has
                # followed them there, `A P A'`, so read out with them it is
                # filterpy's posterior read out with the moments before the
                # row (tasks 206 and 211). The first rows are unsized.
                j = i if remapped[i] else i + 1
                u = w[:j] * 0.5 ** ((t[j - 1] - t[:j]) / 30.0)
                m = (u[:, None] * x[:j]).sum(axis=0) / u.sum()
                v = (u[:, None] * (x[:j] - m) ** 2).sum(axis=0) / u.sum()
                # One row has no spread: the filter's is exactly 0 there,
                # the sums' a rounding (as in `_standardizer_moves`).
                s = np.where(v > 1e-20 * (m * m + 1.0), np.sqrt(np.maximum(v, 0.0)), 1.0)
                T = np.zeros((3, 3))
                T[0, 0] = 1.0
                T[0, 1:] = -m / s
                T[1:, 1:] = np.diag(1.0 / s)
                P = T @ P @ T.T
            np.testing.assert_allclose(se[i], np.sqrt(np.diag(P)), rtol=1e-8, err_msg=str(i))


class TestRlsErrorInflationIsPadasips:
    """docs/PLAN.md task 116 (A): `rls`'s per-row ``error_inflation`` is
    ``sqrt(1 + z' A⁻¹ z · s₂ / s₁)``, the leverage of the row against the
    decayed information matrix, scaled by the Kish ratio of the rows the fit
    learned (`rls.rs`'s module doc). padasip's ``FilterRLS`` keeps ``A⁻¹`` as
    its ``R``, on `TestRlsIsPadasips`'s mapping; ``s₁ = Σ λ^i`` and
    ``s₂ = Σ λ^(2i)`` are written here from their definition. After the row,
    ``se_coef = sigma · sqrt(s₂ / s₁ · diag(A⁻¹))`` (F), ``sigma`` the row's
    own EW residual std. At ``delta`` small and not, decayed and not."""

    CASES = [(50.0, 1.0), (50.0, 1e-6), (float("inf"), 1.0)]

    @staticmethod
    def run(half_life: float, ridge: float, se: bool):
        import padasip

        rng = np.random.default_rng(37)
        n = 400
        x = rng.normal(size=(n, 2))
        y = 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.3 * rng.normal(size=n)
        spec = po.spec.rls(
            "r",
            targets=["y"],
            features=["x0", "x1"],
            half_life=half_life,
            delta=ridge,
            min_weight=0.0,
            emit_error_inflation=True,
            emit_sigma=True,
            coef_every=0,
            **({"emit_se_coef": True} if se else {}),
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(frame)["r"].struct.unnest()
        mu = 0.5 ** (1.0 / half_life)
        rls = padasip.filters.FilterRLS(n=3, mu=mu, eps=ridge / mu, w="zeros")
        return out, x, y, mu, rls

    @staticmethod
    def kish(mu: float, i: int) -> tuple[float, float]:
        """The weight and the squared weight of ``i`` unit rows, decayed."""
        ages = np.arange(i)[::-1]
        return float((mu**ages).sum()), float((mu ** (2 * ages)).sum())

    @pytest.mark.parametrize(("half_life", "ridge"), CASES)
    def test_the_row_field_is_padasips_leverage(self, half_life, ridge):
        out, x, y, mu, rls = self.run(half_life, ridge, se=False)
        # Infinite before a learned row, which is written as null.
        got = out["error_inflation_y"].fill_null(np.inf).to_numpy()
        assert np.isinf(got[0])
        for i in range(len(y)):
            z = np.array([1.0, *x[i]])
            if i > 0:
                s1, s2 = self.kish(mu, i)
                want = np.sqrt(1.0 + z @ rls.R @ z * s2 / s1)
                np.testing.assert_allclose(got[i], want, rtol=1e-9, err_msg=str(i))
            rls.adapt(y[i], z)

    @pytest.mark.parametrize(("half_life", "ridge"), CASES)
    def test_se_coef_is_sigma_times_padasips_inverse(self, half_life, ridge):
        out, x, y, mu, rls = self.run(half_life, ridge, se=True)
        se = out["se_coef"].to_list()
        sigma = out["sigma_y"].to_numpy()
        checked = 0
        for i in range(len(y)):
            rls.adapt(y[i], np.array([1.0, *x[i]]))
            s1, s2 = self.kish(mu, i + 1)
            if np.isfinite(sigma[i]):
                want = sigma[i] * np.sqrt(s2 / s1 * np.diag(rls.R))
                np.testing.assert_allclose(se[i], want, rtol=1e-8, err_msg=str(i))
                checked += 1
        assert checked > len(y) - 5


class TestEwRidgeSeCoefIsStatsmodels:
    """docs/PLAN.md task 116 (F): `ewridge`'s ``se_coef`` is
    ``sigma · sqrt(diag(T M Tᵀ))`` with ``M = Σ̂⁻¹ / n_kish`` (WARMUP §7.1),
    which with no decay, unit weights and a vanishing ridge is the
    least-squares covariance over the noise, ``(X'X)⁻¹`` -- statsmodels'
    ``WLS(...).fit().cov_params() / scale``, the intercept's included, which
    the unstandardizing map ``T`` carries. Standardized or not, the solve's
    space is mapped out, so both read the same."""

    @pytest.mark.parametrize("standardize", [False, True])
    def test_se_over_sigma_is_the_least_squares_covariance(self, standardize):
        import statsmodels.api as sm

        rng = np.random.default_rng(53)
        n = 200
        x = rng.normal(size=(n, 2)) * np.array([3.0, 0.5]) + np.array([10.0, -2.0])
        y = 1.0 + 0.5 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(size=n)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            ridge=1e-10,
            standardize=standardize,
            emit_se_coef=True,
            emit_sigma=True,
            coef_every=1,
            max_rows_between_solves=1,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct.unnest()
        se = out["se_coef"].to_list()
        sigma = out["sigma_y"].to_numpy()
        checked = 0
        for i in (20, 60, n - 1):
            X = sm.add_constant(x[: i + 1])
            fit = sm.WLS(y[: i + 1], X, weights=np.ones(i + 1)).fit()
            want = np.sqrt(np.diag(fit.cov_params() / fit.scale))
            np.testing.assert_allclose(np.asarray(se[i]) / sigma[i], want, rtol=1e-6)
            checked += 1
        assert checked == 3


class TestVarianceInflation:
    """``po.gram.vif`` against statsmodels' ``variance_inflation_factor``, which
    regresses each column on the rest and the constant (task 223: the
    ``collinear`` check reads it)."""

    @pytest.mark.parametrize("noise", [1e-3, 0.1, 1.0])
    def test_vif_is_statsmodels(self, noise):
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        rng = np.random.default_rng(61)
        n = 1500
        x = rng.normal(size=(n, 3)) * np.array([1.0, 4.0, 0.2]) + np.array([5.0, -1.0, 0.0])
        x[:, 2] = x[:, 0] + noise * rng.normal(size=n)
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": rng.normal(size=n)})
        spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1", "x2"], lam=1.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        got = po.gram.vif(bank.gram("m")[0])
        design = np.c_[np.ones(n), x]
        want = [variance_inflation_factor(design, j) for j in (1, 2, 3)]
        np.testing.assert_allclose(got, want, rtol=1e-7)


class TestLarsPathIsSklearns:
    """``po.gram.lars_path`` and ``po.gram.lasso_path`` (docs/PLAN.md task
    226) against scikit-learn, from the raw rows: ``lars_path_gram`` on the
    standardized rows' Gram for the knots and their active sets, and
    ``enet_path`` / ``lasso_path`` for the grid. The rows are standardized
    by their population spread and the target centred, which is the
    correlation form the Gram's ``lam=1.0`` accumulator reaches by its own
    arithmetic; ``n_samples`` puts scikit-learn's ``alpha`` in the same
    units as the penalty here.

    The designs mix their columns' signs, which is what makes a column
    leave the path: seeds 0, 1, 3 and 11 have four to seven knots where one
    does (counted below, so the lasso modification is exercised), seed 2
    none. scikit-learn leaves a coefficient that has left at a rounding
    residue (up to ``1.1e-16`` here) where this path writes 0, so a
    coefficient counts as active above ``1e-12``."""

    N, K = 200, 10

    def rows(self, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        mix = np.eye(self.K) + rng.standard_normal((self.K, self.K))
        x = rng.standard_normal((self.N, self.K)) @ mix + 5.0
        beta = rng.standard_normal(self.K)
        y = x @ beta + 1.0 + 2.0 * rng.standard_normal(self.N)
        features = [f"x{i}" for i in range(self.K)]
        spec = po.spec.ewridge("m", targets=["y"], features=features, lam=1.0, min_weight=5.0)
        frame = pl.DataFrame({f: x[:, i] for i, f in enumerate(features)}).with_columns(
            y=pl.Series(y)
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        return x, y, bank.gram("m")[0]

    @staticmethod
    def standardized(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sd = x.std(axis=0)
        return (x - x.mean(axis=0)) / sd, y - y.mean(), sd

    LEAVES = {0: 7, 1: 4, 2: 0, 3: 4, 11: 4}

    @pytest.mark.parametrize("seed", list(LEAVES))
    def test_the_knots_and_their_active_sets_are_lars_path_grams(self, seed):
        from sklearn.linear_model import lars_path_gram

        x, y, g = self.rows(seed)
        xs, yc, sd = self.standardized(x, y)
        alphas, _, coefs = lars_path_gram(
            xs.T @ yc, xs.T @ xs, n_samples=self.N, method="lasso", eps=1e-15
        )
        path = po.gram.lars_path(g)
        assert len(path["penalties"]) == len(alphas)
        np.testing.assert_allclose(path["penalties"], alphas, rtol=1e-9, atol=1e-12)
        ours = path["coef"][:, 1:] * sd  # back to the standardized basis
        np.testing.assert_allclose(ours, coefs.T, rtol=1e-8, atol=1e-10)
        # The same active set at every knot: the nonzero coefficients.
        support = [list(np.flatnonzero(np.abs(c) > 1e-12)) for c in coefs.T]
        assert [list(np.flatnonzero(r)) for r in ours] == support
        sets = path["active"]
        leaves = sum(len(b) < len(a) for a, b in zip(sets, sets[1:], strict=False))
        assert leaves == self.LEAVES[seed]

    def test_penalty_weights_are_lars_on_rescaled_columns(self):
        from sklearn.linear_model import lars_path_gram

        x, y, g = self.rows(3)
        xs, yc, sd = self.standardized(x, y)
        w = np.linspace(0.5, 2.0, self.K)
        xw = xs / w
        alphas, _, coefs = lars_path_gram(
            xw.T @ yc, xw.T @ xw, n_samples=self.N, method="lasso", eps=1e-15
        )
        path = po.gram.lars_path(g, penalty_weights=w)
        np.testing.assert_allclose(path["penalties"], alphas, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(path["coef"][:, 1:] * sd, (coefs / w[:, None]).T, atol=1e-10)

    @pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
    def test_the_grid_is_enet_paths(self, l1_ratio):
        from sklearn.linear_model import enet_path

        x, y, g = self.rows(7)
        xs, yc, sd = self.standardized(x, y)
        penalties = [0.8, 0.4, 0.2, 0.1, 0.05, 0.01]
        _, coefs, _ = enet_path(
            xs, yc, l1_ratio=l1_ratio, alphas=penalties, tol=1e-14, max_iter=100_000
        )
        ours = po.gram.lasso_path(g, penalties, l1_ratio=l1_ratio, tol=1e-14, max_iter=100_000)
        np.testing.assert_allclose(ours[:, 1:] * sd, coefs.T, atol=1e-8)
        # The intercept: the target's mean less the slopes at the column means.
        np.testing.assert_allclose(ours[:, 0], y.mean() - ours[:, 1:] @ x.mean(axis=0), atol=1e-8)


class TestSolveSubsetsIsStatsmodelsOnThePooledRows:
    """``po.gram.solve_subsets`` (docs/PLAN.md task 227) against statsmodels'
    OLS on the rows of each subset of blocks, pooled: the merge, the solve
    and the statistics at once. With every row at weight 1 the Kish size is
    the row count, and ``sigma2 = resid_var * n / (n - p)`` and ``se`` are
    OLS's ``scale`` and ``bse``. The levels sit at 1,000, where re-centring
    raw moments would lose six digits."""

    def test_coef_se_t_and_r2_are_ols_on_the_subsets_rows(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(71)
        n_blocks, per, k = 5, 300, 3
        x = rng.standard_normal((n_blocks * per, k)) + 1000.0
        y = x @ np.array([0.5, -1.0, 0.25]) + 3.0 + rng.standard_normal(n_blocks * per)
        block = np.repeat(np.arange(n_blocks), per)
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "block": block})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=float("inf"),
            group="block",
            min_weight=5.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        grams = bank.gram("m")
        assert [g["group"] for g in grams] == ["0", "1", "2", "3", "4"]
        subsets = [[0, 2], [1, 3, 4], [4]]
        fits = po.gram.solve_subsets(grams, subsets)
        for subset, (fit,) in zip(subsets, fits, strict=True):
            rows = np.isin(block, subset)
            ols = sm.OLS(y[rows], sm.add_constant(x[rows])).fit()
            np.testing.assert_allclose(fit["coef"], ols.params, rtol=1e-8)
            np.testing.assert_allclose(fit["se"][1:], ols.bse[1:], rtol=1e-8)
            np.testing.assert_allclose(fit["t"][1:], ols.tvalues[1:], rtol=1e-8)
            assert fit["sigma2"] == pytest.approx(ols.scale, rel=1e-8)
            assert fit["r2"] == pytest.approx(ols.rsquared, rel=1e-8)
            assert fit["n"] == rows.sum()


class TestAuditIsScipyAndStatsmodels:
    """An ``audit``'s column statistics against the libraries that compute
    them from the raw values (task 223 (b)): scipy's biased skew and excess
    kurtosis, statsmodels' Dickey-Fuller statistic with a constant and no
    lags, numpy's correlation of a column with its own lag, and numpy's
    median and median absolute deviation, exact from exact counts and to a
    rank of about 1% from the digest past the cap."""

    @staticmethod
    def _audit(x: np.ndarray, **kw: Any) -> dict[str, Any]:
        frame = pl.DataFrame({"t": np.arange(len(x), dtype=float), "x": x})
        spec = po.spec.audit("a", columns=["x"], clock="t", gap_cap=10.0, **kw)
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        return bank.audit().row(0, named=True)

    @pytest.mark.parametrize("level", [0.0, 1e4])
    def test_the_moments_are_scipys(self, level):
        from scipy.stats import kurtosis, skew

        rng = np.random.default_rng(71)
        x = level + rng.standard_t(5, size=3000)
        got = self._audit(x)
        assert got["mean"] == pytest.approx(np.mean(x), rel=1e-14, abs=1e-12)
        assert got["std"] == pytest.approx(np.std(x, ddof=1), rel=1e-10)
        assert got["skew"] == pytest.approx(skew(x), rel=1e-8)
        assert got["kurtosis"] == pytest.approx(kurtosis(x), rel=1e-8)
        assert (got["min"], got["max"]) == (x.min(), x.max())

    @pytest.mark.parametrize("phi", [0.0, 0.9, 1.0])
    def test_the_persistence_is_statsmodels_dickey_fuller(self, phi):
        from statsmodels.tsa.stattools import adfuller

        rng = np.random.default_rng(72)
        e = rng.normal(size=2000)
        x = np.empty_like(e)
        v = 0.0
        for i, ei in enumerate(e):
            v = phi * v + ei
            x[i] = v
        got = self._audit(x)
        tau = adfuller(x, maxlag=0, autolag=None, regression="c", result_object=False)[0]
        assert got["unit_root_t"] == pytest.approx(tau, rel=1e-8)
        assert got["autocorr"] == pytest.approx(np.corrcoef(x[:-1], x[1:])[0, 1], rel=1e-10)

    def test_the_median_and_mad_are_numpys_from_exact_counts(self):
        rng = np.random.default_rng(73)
        x = rng.integers(-20, 30, size=1001).astype(float)
        got = self._audit(x)
        med = np.median(x)
        assert got["median"] == med
        assert got["mad"] == np.median(np.abs(x - med))
        assert got["distinct"] == len(np.unique(x))

    @pytest.mark.parametrize("level", [0.0, 1e6])
    def test_the_digest_reads_the_median_and_mad_to_a_rank(self, level):
        rng = np.random.default_rng(74)
        x = level + rng.lognormal(size=20_000)
        got = self._audit(x)
        assert got["distinct"] is None, "past the cap: the digest answers"
        lo, hi = np.quantile(x, [0.49, 0.51])
        assert lo <= got["median"] <= hi
        dev = np.abs(x - np.median(x))
        dlo, dhi = np.quantile(dev, [0.48, 0.52])
        assert dlo <= got["mad"] <= dhi


def _calibration_rows(n: int, seed: int) -> pl.DataFrame:
    """A fit whose predictions are miscalibrated in warm-up and drift, with
    uneven weights and a zero now and then."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 0.4 + x @ np.array([1.2, -0.8]) + rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    w[rng.random(n) < 0.05] = 0.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


class TestCalibrationIsMincerZarnowitz:
    """Task 221 (a): ``calibration_*`` is the least-squares regression of
    ``y`` on the scored ``pred`` (Mincer and Zarnowitz 1969), and its Wald
    statistic the joint test of slope 1 and intercept 0. Run once
    (``calibration_half_life=inf``) with unit weights it is ``statsmodels``'
    ``OLS`` and twice its ``f_test``'s F; with a memory and weights its
    slope and intercept are ``WLS`` at each scored row's weight times its
    decay, and the statistic is the definition at Kish's size, written by
    hand from ``WLS``'s residuals. Each row's fields read the rows before
    it, never the row."""

    def test_run_once_is_ols_and_its_f_test(self):
        import statsmodels.api as sm

        df = _calibration_rows(600, 11).drop("w")
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=20.0,
            emit_calibration=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y = out["pred_y"].to_numpy(), df["y"].to_numpy()
        checked = 0
        for t in (60, 250, 599):
            ok = np.isfinite(pred[:t])
            fit = sm.OLS(y[:t][ok], sm.add_constant(pred[:t][ok])).fit()
            f = float(fit.f_test((np.eye(2), np.array([0.0, 1.0]))).fvalue)
            row = out.row(t, named=True)
            np.testing.assert_allclose(row["calibration_intercept_y"], fit.params[0], rtol=1e-9)
            np.testing.assert_allclose(row["calibration_slope_y"], fit.params[1], rtol=1e-9)
            np.testing.assert_allclose(row["calibration_wald_y"], 2.0 * f, rtol=1e-8)
            checked += 1
        assert checked == 3

    def test_a_memory_and_weights_are_wls_at_kish_size(self):
        import statsmodels.api as sm

        h_model, h_calib = 40.0, 120.0
        df = _calibration_rows(700, 12)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=h_model,
            weight="w",
            min_weight=10.0,
            emit_calibration=True,
            calibration_half_life=h_calib,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y, w = out["pred_y"].to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()
        checked = 0
        for t in (80, 300, 699):
            s = np.arange(t)
            # Row s folded at its weight, aged by every row after it up to t - 1.
            omega = w[:t] * 0.5 ** ((t - 1 - s) / h_calib)
            ok = np.isfinite(pred[:t]) & (omega > 0)
            X = sm.add_constant(pred[:t][ok])
            fit = sm.WLS(y[:t][ok], X, weights=omega[ok]).fit()
            om = omega[ok]
            n_kish = om.sum() ** 2 / (om**2).sum()
            s2 = (om * fit.resid**2).sum() / om.sum() * n_kish / (n_kish - 2.0)
            d = fit.params - np.array([0.0, 1.0])
            m = (X * om[:, None]).T @ X / om.sum()
            wald = n_kish * d @ m @ d / s2
            row = out.row(t, named=True)
            np.testing.assert_allclose(row["calibration_intercept_y"], fit.params[0], rtol=1e-8)
            np.testing.assert_allclose(row["calibration_slope_y"], fit.params[1], rtol=1e-8)
            np.testing.assert_allclose(row["calibration_wald_y"], wald, rtol=1e-7)
            checked += 1
        assert checked == 3

    def test_an_embargo_folds_the_prediction_each_row_was_scored_with(self):
        """Under ``embargo=3`` on a row clock, row ``s`` is released as row
        ``s + 3`` arrives, before that row is scored: row ``t``'s fields are
        ``OLS`` over the rows up to ``t - 3``, each at the prediction it was
        shown (C21), not the one the model gives it at its release."""
        import statsmodels.api as sm

        df = _calibration_rows(400, 13).drop("w")
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=10.0,
            emit_calibration=True,
            embargo=3.0,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y = out["pred_y"].to_numpy(), df["y"].to_numpy()
        for t in (100, 399):
            upto = t - 2
            ok = np.isfinite(pred[:upto])
            fit = sm.OLS(y[:upto][ok], sm.add_constant(pred[:upto][ok])).fit()
            np.testing.assert_allclose(out["calibration_slope_y"][t], fit.params[1], rtol=1e-9)
            np.testing.assert_allclose(
                out["calibration_intercept_y"][t], fit.params[0], rtol=1e-8, atol=1e-12
            )


def _break_rows(n: int, seed: int, shift: float = 0.0, at: int | None = None) -> pl.DataFrame:
    """Two features and a target; the intercept moves by ``shift`` noise
    standard deviations from row ``at`` on."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    level = 0.5 + (shift * (np.arange(n) >= at) if at is not None else 0.0)
    y = level + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


class TestBreaksAreBrownDurbinAndEvans:
    """Task 221 (b). Run once with no ridge, ``resid / error_inflation`` is
    the recursive residual of Brown, Durbin and Evans (1975), which
    ``statsmodels``' ``recursive_olsresiduals`` computes by its own update of
    the inverse Gram; ``studentized`` is it over the spread of those before
    it, so ``statsmodels``' residuals give it back through the definition.
    ``cusum`` on row ``t`` is the CUSUM path of the studentized residuals
    before ``t`` over the square root of their count, and
    ``breaks_cusumolsresid`` reads the same statistic from both libraries'
    recursive residuals. ``break_wald``'s two fits are ``WLS`` at the two
    memories."""

    @staticmethod
    def _run(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            emit_breaks=True,
            emit_error_inflation=True,
            **{"half_life": float("inf"), **kw},
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    def test_the_studentized_residual_is_the_recursive_residual_over_its_spread(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import breaks_cusumolsresid, recursive_olsresiduals

        df = _break_rows(500, 21)
        out = self._run(df)
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        ols = sm.OLS(df["y"].to_numpy(), X).fit()
        scaled = recursive_olsresiduals(ols, skip=3)[4]
        mine = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        np.testing.assert_allclose(mine[3:], scaled[3:], rtol=0, atol=1e-8)
        # Each studentized residual against the spread of those before it,
        # once that spread has ten rows.
        z = out["studentized_y"].to_numpy().astype(float)
        checked = 0
        for t in range(3 + 10, 500):
            want = scaled[t] / np.sqrt(np.mean(scaled[3:t] ** 2))
            np.testing.assert_allclose(z[t], want, rtol=1e-7)
            checked += 1
        assert np.isnan(z[: 3 + 10]).all()
        assert checked == 487
        # `cusum` on row t: the path before t over its count's square root.
        cusum = out["cusum_y"].to_numpy().astype(float)
        ok = np.isfinite(z)
        path, count = np.cumsum(np.where(ok, z, 0.0)), np.cumsum(ok)
        np.testing.assert_allclose(
            cusum[15:], path[14:-1] / np.sqrt(count[14:-1]), rtol=1e-10, atol=1e-12
        )
        # Ploberger and Kramer's statistic, from either library's residuals.
        theirs = breaks_cusumolsresid(scaled[3:], ddof=0)[0]
        ours = breaks_cusumolsresid(mine[3:], ddof=0)[0]
        np.testing.assert_allclose(ours, theirs, rtol=1e-8)

    def test_the_cusum_crosses_where_statsmodels_does(self):
        """An intercept break of a noise sd at row 400 of 800, and none: the
        CUSUM path crosses Brown, Durbin and Evans' 5% boundary on the same
        streams as ``statsmodels``' ``rcusum`` does against its
        ``rcusumci`` -- every break stream, and the one stream with none of
        the five that both flag, a false alarm at the 5% level."""
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import recursive_olsresiduals

        flagged = []
        for seed, shift in (
            (31, 1.0),
            (34, 1.0),
            (32, 0.0),
            (33, 0.0),
            (35, 0.0),
            (36, 0.0),
            (37, 0.0),
        ):
            df = _break_rows(800, seed, shift, 400)
            z = self._run(df)["studentized_y"].to_numpy().astype(float)
            ok = np.isfinite(z)
            path, r = np.cumsum(np.where(ok, z, 0.0)), np.cumsum(ok)
            t = r[-1]
            ours = bool((ok & (np.abs(path) > 0.948 * (np.sqrt(t) + 2 * r / np.sqrt(t)))).any())
            X = sm.add_constant(df.select("x0", "x1").to_numpy())
            rr = recursive_olsresiduals(sm.OLS(df["y"].to_numpy(), X).fit(), skip=3)
            theirs = bool((np.abs(rr[5][1:]) > rr[6][1]).any())
            assert ours == theirs, (seed, ours, theirs)
            flagged.append(ours)
        assert flagged == [True, True, False, True, False, False, False]

    def test_the_wald_distance_is_two_wls_fits(self):
        import statsmodels.api as sm

        h = 60.0
        df = _break_rows(900, 41, 1.0, 600)
        out = self._run(df, half_life=h)
        x = df.select("x0", "x1").to_numpy()
        y = df["y"].to_numpy()
        X = sm.add_constant(x)
        checked = 0
        for t in (300, 650, 899):
            age = (t - 1) - np.arange(t)
            wf, ws = 0.5 ** (age / h), 0.5 ** (age / (4 * h))
            fast = sm.WLS(y[:t], X[:t], weights=wf).fit()
            slow = sm.WLS(y[:t], X[:t], weights=ws).fit()
            nf, ns = wf.sum() ** 2 / (wf**2).sum(), ws.sum() ** 2 / (ws**2).sum()
            c = 1 / nf + 1 / ns - 2 * (wf * ws).sum() / (wf.sum() * ws.sum())
            s2 = (ws * slow.resid**2).sum() / ws.sum() * ns / (ns - 3)
            d = fast.params - slow.params
            g = (X[:t] * ws[:, None]).T @ X[:t] / ws.sum()
            want = d @ g @ d / (s2 * c)
            np.testing.assert_allclose(out["break_wald_y"][t], want, rtol=1e-6)
            checked += 1
        assert checked == 3
        assert out["break_wald_y"][650] > 21.1, "the break passes chi2(3)'s 0.01% value"


def _sandwich_rows(n: int, seed: int) -> pl.DataFrame:
    """Two features far from zero and a residual whose spread grows with the
    first: heteroskedastic, as HC0 is for."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([3.0, -1.0])
    e = rng.normal(size=n) * (0.5 + np.abs(x[:, 0] - 3.0))
    y = 0.5 + x @ np.array([1.0, -2.0]) + e
    w = rng.uniform(0.5, 1.5, size=n)
    w[rng.random(n) < 0.05] = 0.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


class TestRobustStandardErrorsAreTheSandwich:
    """Task 221 (c). ``se_coef_hc0`` and ``se_coef_hac`` are the
    least-squares sandwich ``(X'ΩX)⁻¹ S (X'ΩX)⁻¹`` with ``S``
    ``statsmodels``' own HAC kernel, ``S_hac_simple`` at ``nlags = 0`` and
    at ``robust_se_lags``, over the scores ``ω e x`` of the bank's
    out-of-sample residuals: run once, and under a memory and weights, each
    row at its present weight. Against ``OLS.fit(cov_type="HC0" / "HAC")``,
    which reads the final fit's in-sample residuals, they agree to the
    estimation error the out-of-sample residuals carry."""

    @staticmethod
    def _run(df: pl.DataFrame, lags: int, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            min_weight=10.0,
            emit_robust_se=True,
            robust_se_lags=lags,
            coef_every=0,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    @pytest.mark.parametrize(("half_life", "weighted"), [(float("inf"), False), (150.0, True)])
    def test_the_sandwich_is_statsmodels_kernel_on_the_out_of_sample_residuals(
        self, half_life, weighted
    ):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        lags = 4
        df = _sandwich_rows(900, 51)
        kw: dict[str, Any] = {"half_life": half_life}
        if weighted:
            kw["weight"] = "w"
        out = self._run(df if weighted else df.drop("w"), lags, **kw)
        resid = out["resid_y"].to_numpy()
        w = df["w"].to_numpy() if weighted else np.ones(df.height)
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        checked = 0
        for t in (120, 500, 899):
            age = t - np.arange(t + 1)
            omega = w[: t + 1] * (0.5 ** (age / half_life))
            keep = np.isfinite(resid[: t + 1]) & (omega > 0)
            Xk, ek, om = X[: t + 1][keep], resid[: t + 1][keep], omega[keep]
            bi = np.linalg.inv((Xk * om[:, None]).T @ Xk)
            for nlags, field in ((0, "se_coef_hc0"), (lags, "se_coef_hac")):
                S = S_hac_simple(Xk * (om * ek)[:, None], nlags=nlags)
                want = np.sqrt(np.diag(bi @ S @ bi))
                np.testing.assert_allclose(out[field][t].to_numpy(), want, rtol=1e-8)
                checked += 1
        assert checked == 6

    def test_run_once_it_is_statsmodels_robust_fit_to_the_estimation_error(self):
        import statsmodels.api as sm

        df = _sandwich_rows(1500, 52).drop("w")
        out = self._run(df, 3, half_life=float("inf"))
        ok = np.isfinite(out["resid_y"].to_numpy())
        X = sm.add_constant(df.select("x0", "x1").to_numpy())[ok]
        y = df["y"].to_numpy()[ok]
        for cov, kw, field in (
            ("HC0", {}, "se_coef_hc0"),
            ("HAC", {"maxlags": 3}, "se_coef_hac"),
        ):
            fit = sm.OLS(y, X).fit(cov_type=cov, cov_kwds=kw or None)
            ratio = out[field][-1].to_numpy() / fit.bse
            # Out of sample: a little larger, by the estimation error.
            assert ((ratio > 1.0) & (ratio < 1.03)).all(), (cov, ratio)

    def test_overlapping_labels_put_back_about_the_square_root_of_the_horizon(self):
        """A target summing the next 10 rows' shocks against an AR(1)
        feature at 0.95, learned under ``embargo=10``: across 300 streams the
        coefficient's spread is about ``sqrt(10)`` times ``se_coef``, and
        ``se_coef_hac`` at its default lags, twice the horizon, reads it to
        within 15%; HC0, which assumes no correlation, does not."""
        h, n, g = 10, 3000, 300
        rng = np.random.default_rng(221)
        frames = []
        for gi in range(g):
            u = rng.normal(size=n + 200)
            x = np.zeros(n + 200)
            for t in range(1, n + 200):
                x[t] = 0.95 * x[t - 1] + np.sqrt(1 - 0.95**2) * u[t]
            eps = rng.normal(size=n + 200 + h)
            noise = np.convolve(eps, np.ones(h), "valid")[1 : n + 201]
            frames.append(
                pl.DataFrame({"g": np.full(n, gi), "x0": x[200:], "y": 0.2 * x[200:] + noise[200:]})
            )
        df = pl.concat(frames)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=float("inf"),
            group="g",
            embargo=float(h),
            min_weight=20.0,
            emit_robust_se=True,
            emit_se_coef=True,
            emit_sigma=True,
            coef_every=0,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        last = out.group_by("g", maintain_order=True).last()
        slope = np.array([c[1] for c in last["coef"].to_list()])
        se = np.array([c[1] for c in last["se_coef"].to_list()]).mean()
        hc0 = np.array([c[1] for c in last["se_coef_hc0"].to_list()]).mean()
        hac = np.array([c[1] for c in last["se_coef_hac"].to_list()]).mean()
        spread = slope.std(ddof=1)
        assert 0.6 * np.sqrt(h) < spread / se < 1.2 * np.sqrt(h), spread / se
        assert abs(hac / spread - 1.0) < 0.15, hac / spread
        assert hc0 / spread < 0.5, hc0 / spread


def _spec_rows(
    n: int, seed: int, curve: float = 0.0, lookahead: int = 0, hetero: bool = True
) -> pl.DataFrame:
    """Two features; noise whose spread grows with the first; a square of
    it the fit may miss; and, with ``lookahead``, a target summing that many
    rows' shocks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    u = rng.normal(size=n + max(lookahead, 1))
    e = np.convolve(u, np.ones(lookahead), "valid")[1 : n + 1] if lookahead else u[:n]
    if hetero:
        e = e * np.maximum(0.2, 1.0 + 0.6 * x[:, 0])
    y = 0.5 + x @ np.array([1.0, -1.0]) + curve * x[:, 0] ** 2 + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


class TestSpecificationTestsAreStatsmodels:
    """Task 221 (d). Run once, on the bank's out-of-sample residuals of the
    rows before each row: ``ljung_box`` is ``acorr_ljungbox``,
    ``breusch_pagan`` is ``het_breuschpagan``'s ``n R2``, and ``reset`` is
    ``compare_lm_test`` of the residual on ``p, p2, p3`` against ``p``.
    Past a horizon ``ljung_box`` is the definition with Bartlett's
    covariance, from ``statsmodels``' ``acf``; on planted curvature it
    decides as ``linear_reset`` does on the in-sample fit."""

    @staticmethod
    def _run(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=10.0,
            emit_specification=True,
            ljung_box_lags=6,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    def test_the_three_statistics_are_statsmodels(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

        df = _spec_rows(700, 61, curve=0.2)
        out = self._run(df)
        r, p = out["resid_y"].to_numpy(), out["pred_y"].to_numpy()
        x = df.select("x0", "x1").to_numpy()
        for t in (150, 699):
            ok = np.isfinite(r[:t])
            rt, pt = r[:t][ok], p[:t][ok]
            lb = acorr_ljungbox(rt, lags=[6])["lb_stat"].iloc[0]
            np.testing.assert_allclose(out["ljung_box_y"][t], lb, rtol=1e-9)
            bp = het_breuschpagan(rt, sm.add_constant(x[:t][ok]))[0]
            np.testing.assert_allclose(out["breusch_pagan_y"][t], bp, rtol=1e-8)
            q = pt - pt[0]
            full = sm.OLS(rt, sm.add_constant(np.column_stack([q, q**2, q**3]))).fit()
            base = sm.OLS(rt, sm.add_constant(q)).fit()
            np.testing.assert_allclose(out["reset_y"][t], full.compare_lm_test(base)[0], rtol=1e-7)

    def test_past_a_horizon_it_reads_bartletts_covariance(self):
        from statsmodels.tsa.stattools import acf

        h = 4
        df = _spec_rows(900, 62, lookahead=h)
        out = self._run(df, embargo=float(h))
        r = out["resid_y"].to_numpy()
        t = 899
        # Under the embargo, row t has learned the rows up to t - h.
        released = r[: t - h + 1]
        rt = released[np.isfinite(released)]
        n = len(rt)
        rho = acf(rt, nlags=h - 1 + 6, fft=False)
        lags = np.arange(h, h + 6)
        rr = rho[lags] * np.sqrt((n + 2) / (n - lags))
        s = h - 1

        def at(j: int) -> float:
            return float(rho[abs(j)]) if abs(j) <= s else 0.0

        def gamma(v: int) -> float:
            return sum(at(j) * at(j + v) for j in range(-s, s + 1))

        c = np.array([[gamma(abs(a - b)) for b in range(6)] for a in range(6)])
        want = n * rr @ np.linalg.solve(c, rr)
        np.testing.assert_allclose(out["ljung_box_y"][t], want, rtol=1e-8)

    def test_reset_decides_as_linear_reset(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import linear_reset

        for seed, curve in ((63, 0.4), (64, 0.0), (65, 0.4), (66, 0.0)):
            df = _spec_rows(1500, seed, curve=curve, hetero=False)
            ours = self._run(df)["reset_y"][-1] > 5.991
            X = sm.add_constant(df.select("x0", "x1").to_numpy())
            fit = sm.OLS(df["y"].to_numpy(), X).fit()
            theirs = linear_reset(fit, power=3, test_type="fitted", use_f=False).statistic > 5.991
            assert ours == theirs == (curve > 0), (seed, ours, theirs)


class TestTailsAreScipyAndJarqueBera:
    """Task 221 (e). Run once, the tails of the recursive residuals before
    each row, ``resid / error_inflation``, are ``scipy.stats``' biased
    ``skew`` and ``kurtosis`` and ``statsmodels``' ``jarque_bera``; with a
    memory and weights, the same definitions at each row's present weight
    and Kish's size, written by hand from ``numpy.average``."""

    def test_run_once_and_windowed(self):
        from scipy import stats
        from statsmodels.stats.stattools import jarque_bera

        rng = np.random.default_rng(71)
        n = 1200
        x = rng.normal(size=(n, 2))
        w = rng.uniform(0.5, 1.5, size=n)
        w[rng.random(n) < 0.05] = 0.0
        y = x @ np.array([1.0, -1.0]) + rng.standard_t(5, size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})
        common = dict(
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            min_weight=3.0,
            emit_tails=True,
            emit_error_inflation=True,
        )
        once = po.ModelBank([po.spec.ewridge("m", half_life=float("inf"), **common)])
        out = once.fit_predict(df.drop("w")).unnest("m")
        v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        for t in (200, 1199):
            vt = v[:t][np.isfinite(v[:t])]
            np.testing.assert_allclose(out["skew_y"][t], stats.skew(vt), rtol=1e-9)
            np.testing.assert_allclose(out["kurtosis_y"][t], stats.kurtosis(vt), rtol=1e-9)
            np.testing.assert_allclose(out["jarque_bera_y"][t], jarque_bera(vt)[0], rtol=1e-9)
        h = 150.0
        spec = po.spec.ewridge("m", half_life=60.0, tails_half_life=h, weight="w", **common)
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        for t in (300, 1199):
            om = w[:t] * 0.5 ** (((t - 1) - np.arange(t)) / h)
            keep = np.isfinite(v[:t]) & (om > 0)
            vt, ot = v[:t][keep], om[keep]
            mu = np.average(vt, weights=ot)
            m = [np.average((vt - mu) ** j, weights=ot) for j in (2, 3, 4)]
            skew, kurt = m[1] / m[0] ** 1.5, m[2] / m[0] ** 2 - 3
            n_kish = ot.sum() ** 2 / (ot**2).sum()
            np.testing.assert_allclose(out["skew_y"][t], skew, rtol=1e-8)
            np.testing.assert_allclose(out["kurtosis_y"][t], kurt, rtol=1e-8)
            np.testing.assert_allclose(
                out["jarque_bera_y"][t], n_kish / 6 * (skew**2 + kurt**2 / 4), rtol=1e-8
            )


class TestInfluenceIsDffitsAtTheNewestRow:
    """Task 221 (f). Run once with no ridge, ``influence`` on row ``t`` is
    ``statsmodels``' ``OLSInfluence(...).dffits`` of the fit over rows
    ``0..=t`` at its last row: that row's in-sample leverage is ``h/(1+h)``
    and its externally studentized residual the recursive residual over
    the recursive residuals' mean square before it, which is the fit
    without it's ``RSS/(n-k)``. Where the definitions part -- an earlier
    row's in-sample DFFITS reads the rows after it -- nothing is compared."""

    def test_each_rows_influence_is_its_prefixs_last_dffits(self):
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence

        rng = np.random.default_rng(81)
        n = 300
        x = rng.normal(size=(n, 2)) + np.array([4.0, -2.0])
        x[::37, 0] += 5.0  # rows of high leverage
        y = 0.5 + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            ridge=1e-12,
            min_weight=3.0,
            emit_influence=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        X = sm.add_constant(x)
        checked = 0
        for t in range(8, n, 5):
            want = OLSInfluence(sm.OLS(y[: t + 1], X[: t + 1]).fit()).dffits[0][t]
            np.testing.assert_allclose(out["influence_y"][t], want, rtol=1e-8)
            checked += 1
        assert checked == 59
