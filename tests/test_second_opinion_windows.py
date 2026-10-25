"""Windowed models held to third-party libraries at the window's edge
(task 278, batch 10).

At ``half_life = inf`` every row inside a window weighs its own weight and
no prior fades, so a windowed model is a batch computation on a set of rows
a library can be handed:

- ``ewridge`` at ridge 0 is weighted least squares over the rows inside,
  which statsmodels' ``RollingWLS`` computes over a fixed count of rows.
  On the row-number clock (no clock column) a row's age is the number of
  rows behind the last row learned, so ``window_size = W`` keeps the last
  ``W`` rows under ``closed="right"`` (age less than ``W``) and the last
  ``W + 1`` under ``"both"`` (age at most ``W``): ``RollingWLS`` with
  ``window=W`` or ``W + 1``.
- ``ew_class`` is a Gaussian per class over the rows inside:
  ``scipy.stats.multivariate_normal`` of numpy's class means and
  population covariances.
- ``marginal``'s table is numpy's weighted mean, covariance, correlation and
  slope of the rows inside.

Each test runs ``closed="right"`` and ``"both"`` on an integer clock, where a
row sits exactly one window back on every row, and shows that the two
oracles part by far more than the tolerance, so the edge is what it reads.
A row is scored with the state the rows before it left: the window as it
stood after row ``i - 1``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import multivariate_normal
from statsmodels.regression.rolling import RollingWLS

import polars_online as po

TIER = "mixed"

GRID = pytest.mark.extended(reason="a grid over the window's edge, size and shape")
CLOSED = ["right", "both"]


def _rows_inside(window: float, closed: str) -> int:
    """How many rows a window of ``window`` rows keeps on the row-number
    clock, once the stream is longer than it."""
    return int(window) + (1 if closed == "both" else 0)


def _ages_inside(ages: np.ndarray, window: float, closed: str) -> np.ndarray:
    return ages < window if closed == "right" else ages <= window


class TestEwRidgeIsRollingWLS:
    """``ewridge`` at ``half_life = inf`` and ``ridge = 0``, solved after
    every row, against ``RollingWLS`` on the same weighted rows: the
    prediction on row ``i`` is the rolling fit ending at row ``i - 1``
    applied to row ``i``'s features. Checked from the first row whose window
    is full, so both are the fit of the same count of rows."""

    N = 400

    def _frame(self, seed: int = 5) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((self.N, 2)) * [1.0, 3.0] + [0.5, -2.0]
        # A slope that drifts, so each window's fit is its own and the row
        # at the edge moves it.
        slope = np.linspace(-1.0, 2.0, self.N)
        y = 1.0 + slope * x[:, 0] - 0.5 * x[:, 1] + 0.4 * rng.standard_normal(self.N)
        w = rng.uniform(0.5, 2.0, self.N)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})

    def _oracle(self, df: pl.DataFrame, rows: int) -> np.ndarray:
        """Row ``i``'s prediction from the rolling fit of the ``rows`` rows
        ending at ``i - 1``; NaN before the first full window."""
        z = np.column_stack([np.ones(df.height), df.select(["x0", "x1"]).to_numpy()])
        params = RollingWLS(df["y"].to_numpy(), z, window=rows, weights=df["w"].to_numpy())
        params = params.fit(params_only=True).params
        pred = np.full(df.height, np.nan)
        pred[1:] = np.einsum("ik,ik->i", params[:-1], z[1:])
        return pred

    @pytest.mark.parametrize(
        ("window", "closed", "standardize"),
        [
            pytest.param(37.0, "both", False, id="37-both"),
            pytest.param(37.0, "right", False, id="37-right", marks=GRID),
            pytest.param(8.0, "both", False, id="8-both", marks=GRID),
            pytest.param(8.0, "right", False, id="8-right", marks=GRID),
            pytest.param(120.0, "both", True, id="120-both-std", marks=GRID),
            pytest.param(120.0, "right", True, id="120-right-std", marks=GRID),
        ],
    )
    def test_pred_is_the_rolling_wls_of_the_rows_inside(self, window, closed, standardize):
        df = self._frame()
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=0.0,
            half_life=float("inf"),
            window_size=window,
            closed=closed,
            standardize=standardize,
            solve_every=0,
            weight="w",
            min_weight=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        rows = _rows_inside(window, closed)
        want = self._oracle(df, rows)
        got = out["pred_y"].to_numpy()
        full = np.arange(df.height) >= rows + 1
        # weight_sum is the weight of the rows inside, read before the row.
        w = df["w"].to_numpy()
        held = np.array([w[max(0, i - rows) : i].sum() for i in range(df.height)])
        np.testing.assert_allclose(out["weight_sum"].to_numpy(), held, rtol=1e-12, atol=1e-12)
        # Measured 2026-10-25: at most 2.6e-13 relative to 1 + |pred| over the
        # grid (8-both); 1e-9 is the tolerance.
        err = np.abs(got[full] - want[full]) / (1.0 + np.abs(want[full]))
        print(f"window {window} {closed} standardize={standardize}: max err {err.max():.2e}")
        assert err.max() < 1e-9, err.max()
        # The other edge is another fit: the row exactly a window back moves
        # the prediction by far more than the tolerance.
        other = self._oracle(df, _rows_inside(window, "right" if closed == "both" else "both"))
        both = full & np.isfinite(other)
        assert np.abs(other[both] - want[both]).max() > 1e-4

    def test_no_row_on_the_edge_makes_both_edges_one(self):
        """On a clock where no row is ever exactly one window behind a
        learned row, ``"both"`` and ``"right"`` keep the same rows: every
        output is the same to the bit."""
        df = self._frame().with_columns(t=pl.int_range(pl.len()).cast(pl.Float64) * 0.7 + 0.05)
        t = df["t"].to_numpy()
        window = 26.0
        assert not any((t[i] - t[:i] == window).any() for i in range(len(t)))

        def run(closed: str) -> pl.DataFrame:
            spec = po.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                ridge=0.0,
                clock="t",
                gap_cap=1e6,
                half_life=40.0,
                window_size=window,
                closed=closed,
                solve_every=0,
                weight="w",
                min_weight=0.0,
            )
            return po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()

        right, both = run("right"), run("both")
        for f in ("pred_y", "resid_y", "weight_sum"):
            assert right[f].equals(both[f]), f


def _class_rows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two features whose covariance changes halfway, and a label from a
    tilted line, so each class's moments inside a window are their own."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 2))
    early = np.arange(n) < n // 2
    x[early] = x[early] @ np.array([[3.0, 1.2], [0.0, 1.5]])
    labels = np.where(x[:, 0] + 0.5 * x[:, 1] + 0.3 * rng.standard_normal(n) > 0.0, "a", "b")
    return x, labels


class TestEwClassIsScipysGaussianAtTheEdge:
    """``ew_class`` at ``half_life = inf`` with a tiny precision prior: each
    row's ``p_a`` is the posterior of the two classes' Gaussians over the
    rows inside the window, by ``scipy.stats.multivariate_normal``, with the
    class prior each class's share of those rows. ``closed="both"`` keeps
    the row exactly a window back, which on this stream moves ``p_a`` by
    up to about 0.1 (the inventory's probe: 0.0897)."""

    PRIOR = 1e-12
    WINDOW = 80.0

    @pytest.mark.parametrize(
        ("shape", "closed"),
        [
            pytest.param("full", "both", id="full-both"),
            *[
                pytest.param(s, c, id=f"{s}-{c}", marks=GRID)
                for s in ("full", "diag", "tied")
                for c in CLOSED
                if (s, c) != ("full", "both")
            ],
        ],
    )
    def test_p_is_the_scipy_posterior_of_the_rows_inside(self, shape, closed):
        n = 400
        x, labels = _class_rows(n, seed=21)
        spec = po.spec.ew_class(
            "m",
            features=["x0", "x1"],
            label="c",
            classes=["a", "b"],
            covariance_type=shape,
            precision_prior=self.PRIOR,
            half_life=float("inf"),
            window_size=self.WINDOW,
            closed=closed,
            min_weight=0,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "c": labels})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()

        def posterior(i: int, edge: str) -> float:
            keep = _ages_inside((i - 1) - np.arange(i), self.WINDOW, edge)
            rows, lab = x[:i][keep], labels[:i][keep]
            means, covs, priors = [], [], []
            for c in ("a", "b"):
                rc = rows[lab == c]
                means.append(rc.mean(axis=0))
                covs.append(np.cov(rc.T, ddof=0) + self.PRIOR * np.eye(2))
                priors.append(len(rc) / len(rows))
            if shape == "diag":
                covs = [np.diag(np.diag(cv)) for cv in covs]
            elif shape == "tied":
                pooled = priors[0] * covs[0] + priors[1] * covs[1]
                covs = [pooled, pooled]
            logp = np.array(
                [
                    np.log(priors[c]) + multivariate_normal(means[c], covs[c]).logpdf(x[i])
                    for c in range(2)
                ]
            )
            p = np.exp(logp - logp.max())
            return float(p[0] / p.sum())

        other = "right" if closed == "both" else "both"
        apart = 0.0
        for i in range(100, n, 3):
            want = posterior(i, closed)
            # Measured 2026-10-25: at most 6.9e-13 from scipy over the grid,
            # an error that scales with the prior (6.9e-10 at a prior of
            # 1e-9, where the precision prior's form is what parts them);
            # 1e-10 is the bound.
            assert out["p_a"][i] == pytest.approx(want, abs=1e-10), (shape, closed, i)
            apart = max(apart, abs(posterior(i, other) - want))
        print(f"{shape} {closed}: the other edge moves p_a by up to {apart:.3g}")
        assert apart > 1e-4, apart


class TestMarginalTableIsNumpyAtTheEdge:
    """``marginal``'s table, read after the last row, is numpy's weighted
    mean, population variance, covariance, correlation and slope of the rows
    inside the window, each at ``w * 0.5 ** (age / half_life)``. Under
    ``closed="both"`` the row exactly a window behind the last is one of
    them; before task 278 (batch 10) only ``weight_sum`` was checked there."""

    @pytest.mark.parametrize(
        ("window", "closed"),
        [
            pytest.param(70.0, "both", id="70-both"),
            pytest.param(70.0, "right", id="70-right", marks=GRID),
            pytest.param(9.0, "both", id="9-both", marks=GRID),
            pytest.param(9.0, "right", id="9-right", marks=GRID),
        ],
    )
    def test_the_table_is_numpy_of_the_rows_inside(self, window, closed):
        rng = np.random.default_rng(11)
        n, half_life = 400, 25.0
        u = rng.normal(0.0, 1.0, n)
        x = 3.0 + u
        y = -1.0 + 2.0 * u * np.linspace(0.2, 1.5, n) + rng.normal(0.0, 0.4, n)
        wt = rng.uniform(0.5, 2.0, n)
        df = pl.DataFrame({"x": x, "y": y, "w": wt})
        spec = po.spec.marginal(
            "m",
            features=["x"],
            targets=["y"],
            half_life=half_life,
            window_size=window,
            closed=closed,
            weight="w",
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        pair = bank.marginal("m").row(0, named=True)

        def table(edge: str) -> dict[str, float]:
            ages = (n - 1) - np.arange(n)
            keep = _ages_inside(ages, window, edge)
            w = wt[keep] * 0.5 ** (ages[keep] / half_life)
            cov = np.cov(np.vstack([x[keep], y[keep]]), aweights=w, ddof=0)
            return {
                "weight_sum": w.sum(),
                "mean_x": np.average(x[keep], weights=w),
                "mean_y": np.average(y[keep], weights=w),
                "var_x": cov[0, 0],
                "var_y": cov[1, 1],
                "cov": cov[0, 1],
                "corr": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                "beta": cov[0, 1] / cov[0, 0],
            }

        want, other = table(closed), table("both" if closed == "right" else "right")
        for key, value in want.items():
            # Measured 2026-10-25: at most 4.4e-15 relative; 1e-11 is the bound.
            assert pair[key] == pytest.approx(value, rel=1e-11, abs=1e-12), key
        moved = max(abs(other[k] - want[k]) / (1 + abs(want[k])) for k in want)
        print(f"window {window} {closed}: the other edge moves the table by up to {moved:.3g}")
        assert moved > 1e-4, moved
