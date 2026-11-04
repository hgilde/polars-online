"""``pa`` with its own intercept and a tube of positive width, and ``sgd``
under the Poisson loss and AdaGrad, against river row by row.

``tests/test_second_opinion.py`` holds ``pa`` to river's ``PARegressor``
without an intercept and at ``eps = 0``, and ``sgd`` to scikit-learn's
``SGDRegressor`` under the losses and schedules it shares. The forms here
have no scikit-learn twin: river computes each, given the mapping each class
states.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


class _Tube:
    """The epsilon-insensitive loss with a tube set before each row."""

    eps = 0.0

    def __call__(self, y_true: float, y_pred: float) -> float:
        return max(abs(y_pred - y_true) - self.eps, 0.0)


class TestAPassiveAggressiveInterceptAndTubeAreRivers:
    """``pa`` with its own intercept and a tube of positive width, held to
    river's ``PARegressor`` on every row in every mode. The intercept is a
    column of ``z`` here, inside ``‖z‖²``, so river is given a constant-1
    feature and no intercept of its own: the augmented form. The tube is
    ``eps·σ_y``, ``σ_y`` the target's own EW standard deviation before the
    row (biased, the row's ``y`` joining after it is judged), where river's
    ``eps`` is a number in the target's units; so river's tube is set before
    each row to ``eps`` times pandas' ``ewm(halflife=).var(bias=True)``
    shifted one row (an ``expanding().var(ddof=0)`` without decay), 0 before
    two rows. The coefficients do not decay under a half-life, as river's
    do not; only ``σ_y`` reads it. Measured 2026-11-04: 2.7e-15 on the
    predictions and 1.3e-15 on the coefficients over 600 rows, with about a
    third of the rows inside the tube."""

    @pytest.mark.parametrize("half_life", [float("inf"), 50.0])
    @pytest.mark.parametrize(("mode", "river_mode"), [("pa", 0), ("pa1", 1), ("pa2", 2)])
    def test_every_row_is_rivers_on_a_constant_feature(self, mode, river_mode, half_life):
        import pandas as pd
        from river import linear_model

        rng = np.random.default_rng(11)
        n, c, eps = 600, 0.3, 0.1
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 3.0 + 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(0.0, 0.3, n)
        spec = po.spec.pa(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            mode=mode,
            **({} if mode == "pa" else {"c": c}),
            eps=eps,
            fit_intercept=True,
            half_life=half_life,
            min_weight=0.0,
            coef_every=1,
            standardize=False,
        )
        frame = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        out = po.ModelBank([spec]).fit_predict(frame)["m"].struct
        pred = out.field("pred_y").to_numpy()
        coef = np.array(out.field("coef").to_list(), dtype=float)
        ys = pd.Series(y)
        if np.isinf(half_life):
            var = ys.expanding().var(ddof=0)
        else:
            var = ys.ewm(halflife=half_life, adjust=True).var(bias=True)
        tube = eps * np.sqrt(var.shift(1).fillna(0.0).to_numpy())
        river = linear_model.PARegressor(C=c, mode=river_mode, eps=eps, learn_intercept=False)
        river.loss = _Tube()
        inside = 0
        for i in range(n):
            row = {"one": 1.0, "x0": x[i, 0], "x1": x[i, 1]}
            river.loss.eps = tube[i]
            want_p = river.predict_one(row)
            assert pred[i] == pytest.approx(want_p, abs=1e-13), (mode, i)
            inside += abs(y[i] - want_p) <= tube[i]
            river.learn_one(row, y[i])
            want = [river.weights.get(k, 0.0) for k in ("one", "x0", "x1")]
            assert coef[i] == pytest.approx(want, abs=1e-13), (mode, i)
        # The tube held rows passive, and the intercept took the level.
        assert 100 < inside < n - 100, inside
        assert coef[-1][0] == pytest.approx(3.0, abs=0.2), coef[-1]


class TestSgdPoissonAndAdaGradAreRivers:
    """``sgd`` under the Poisson loss and under AdaGrad, held to river's
    ``LinearRegression`` row by row: the prediction read before each row,
    the coefficients after it. Unstandardized, without the gradient's clip
    (river clips the loss's gradient, ours each coordinate's), and with the
    intercept as a constant-1 feature: river's ``intercept_lr=0`` keeps its
    own at 0, and ours without ``fit_intercept`` takes no running-mean step
    for a level, which river does not have. Weighted rows too: river's
    ``learn_one(x, y, w)`` scales the loss's gradient by ``w``, as ours
    does.

    - **Poisson, a constant rate:** river's ``Poisson`` loss is ``exp(η) −
      y η``, its gradient ``exp(η) − y`` and its mean ``exp(η)``, ours
      inside the link's ``±30`` clamp, which no row here reaches. Measured:
      0 at unit weights and 4e-16 relative weighted.
    - **AdaGrad:** ours steps ``lr·g/(√G + ε)``, river's ``lr·g/√(G + ε')``,
      the epsilon inside the root. River's is set to ``1e-300``, which moves
      no step past rounding and keeps a zero gradient's step 0 where 0 would
      divide by 0; ours is ``1e-8`` (in ``s_y`` under the squared loss). So
      the two part by about ``ε/√G`` of each step, and the tolerance is
      ``1e-7`` relative: measured 2.7e-9 to 8.6e-9 over 600 rows.
    - **Decayed AdaGrad** (a half-life): ours keeps ``G ← λG + g²``, which
      is river's ``RMSProp`` accumulator ``v ← ρv + (1 − ρ)g²`` over ``1 −
      λ`` at ``ρ = λ``, so river's ``RMSProp(lr·√(1 − λ), rho=λ)`` takes
      the same steps on a row clock, ``λ = 2^(−1/h)``, under the same
      epsilon."""

    LR = 0.05

    @staticmethod
    def rows(n: int = 600, seed: int = 7) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = 0.5 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * rng.standard_t(3, n)
        counts = rng.poisson(np.exp(0.3 + 0.4 * x[:, 0] - 0.3 * x[:, 1])).astype(float)
        w = rng.uniform(0.5, 1.5, n)
        return pl.DataFrame(
            {"one": np.ones(n), "x0": x[:, 0], "x1": x[:, 1], "y": y, "c": counts, "w": w}
        )

    @staticmethod
    def ours(
        frame: pl.DataFrame, target: str, half_life: float, weighted: bool, **kw: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        spec = po.spec.sgd(
            "s",
            targets=[target],
            features=["one", "x0", "x1"],
            fit_intercept=False,
            half_life=half_life,
            min_weight=0.0,
            weight="w" if weighted else None,
            standardize=False,
            clip_gradient=float("inf"),
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(frame)["s"].struct
        coef = np.array(out.field("coef").to_list(), dtype=float)
        return out.field(f"pred_{target}").to_numpy(), coef

    @staticmethod
    def rivers(
        frame: pl.DataFrame, target: str, weighted: bool, optimizer: Any, loss: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        from river import linear_model

        model = linear_model.LinearRegression(
            optimizer=optimizer, loss=loss, l2=0.0, intercept_lr=0.0, clip_gradient=float("inf")
        )
        names = ("one", "x0", "x1")
        x = frame.select(names).to_numpy()
        y, w = frame[target].to_numpy(), frame["w"].to_numpy()
        pred, coef = np.zeros(len(y)), np.zeros((len(y), 3))
        for i in range(len(y)):
            row = dict(zip(names, x[i], strict=True))
            pred[i] = model.predict_one(row)
            model.learn_one(row, y[i], w[i] if weighted else 1.0)
            coef[i] = [model.weights.get(k, 0.0) for k in names]
        return pred, coef

    @pytest.mark.parametrize("weighted", [False, True], ids=["unit weights", "weighted"])
    def test_poisson_at_a_constant_rate(self, weighted):
        from river import optim

        frame = self.rows()
        got_p, got_c = self.ours(
            frame, "c", float("inf"), weighted, loss="poisson", learning_rate=0.01
        )
        want_p, want_c = self.rivers(frame, "c", weighted, optim.SGD(0.01), optim.losses.Poisson())
        np.testing.assert_allclose(got_p, want_p, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(got_c, want_c, rtol=0.0, atol=1e-14)
        # The rate moved: the fit is near the counts' log-linear truth.
        assert got_c[-1] == pytest.approx([0.3, 0.4, -0.3], abs=0.2), got_c[-1]

    @pytest.mark.parametrize("weighted", [False, True], ids=["unit weights", "weighted"])
    @pytest.mark.parametrize("half_life", [float("inf"), 50.0], ids=["adagrad", "decayed"])
    @pytest.mark.parametrize("loss", ["squared", "poisson"])
    def test_adagrad(self, loss, half_life, weighted):
        from river import optim

        frame = self.rows()
        target = "c" if loss == "poisson" else "y"
        got_p, got_c = self.ours(
            frame,
            target,
            half_life,
            weighted,
            loss=loss,
            learning_rate=self.LR,
            schedule="adagrad",
        )
        if np.isinf(half_life):
            opt = optim.AdaGrad(self.LR, eps=1e-300)
        else:
            lam = 0.5 ** (1.0 / half_life)
            opt = optim.RMSProp(self.LR * np.sqrt(1.0 - lam), rho=lam, eps=1e-300)
        river_loss = optim.losses.Poisson() if loss == "poisson" else optim.losses.Squared()
        want_p, want_c = self.rivers(frame, target, weighted, opt, river_loss)
        np.testing.assert_allclose(got_p, want_p, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(got_c, want_c, rtol=1e-7, atol=1e-7)
        # The control: the epsilon's placement is the whole gap. River at
        # the rate ten parts in a million off parts from ours by far more.
        slip = 1.00001
        if np.isinf(half_life):
            off = optim.AdaGrad(self.LR * slip, eps=1e-300)
        else:
            off = optim.RMSProp(self.LR * slip * np.sqrt(1.0 - lam), rho=lam, eps=1e-300)
        far, _ = self.rivers(frame, target, weighted, off, river_loss)
        assert np.max(np.abs(far - got_p) / (1.0 + np.abs(far))) > 1e-6
