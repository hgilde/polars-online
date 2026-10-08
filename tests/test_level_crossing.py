"""A level that crosses zero (docs/PLAN.md task 209 (d)).

A target, and a feature beside it, whose level drifts from +1,000 to -1,000
across the stream: every regression on it, scored by the bank's running
``hit_rate`` -- the sign test about zero -- and by its error on each side of
the crossing; and the four whose oracle is a fit computed from the rows held
to it, row by row once they have settled. The models with a library of their
own as the oracle meet the crossing in its test: ``kalman`` filterpy's,
``rls`` padasip's, ``sgd`` scikit-learn's, ``pa`` and ``ftrl`` river's and
``holt`` statsmodels' (`test_second_opinion.py`, `test_river.py`).
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_model_registry import REGRESSIONS

N = 4000
HALF_LIFE = 200.0
#: The rows each side of the crossing is scored over: settled, and as far
#: from the crossing (row 2000) as from the stream's ends.
ABOVE = slice(800, 1600)
BELOW = slice(2400, 3200)

#: Every regression, with what it needs on this stream. ``ftrl`` fits the
#: target with the squared loss, its logistic default taking a label;
#: ``holt`` reads the target alone and follows the drift as its trend.
MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"delta": 1.0}),
    ("kalman", {"coef_half_life": 100.0}),
    ("lasso", {"lasso_path": [0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {"loss": "squared"}),
    ("sgd", {}),
    ("pa", {}),
    ("holt", {"features": []}),
]
IDS = [m for m, _ in MODELS]


def test_every_regression_crosses():
    assert set(IDS) == REGRESSIONS


def crossing(n: int = N, seed: int = 17) -> pl.DataFrame:
    """``y = level + 2 x0 - x1 + 0.3 e`` with ``level`` running from
    +1,000 to -1,000 in equal steps, and ``x2`` the level itself: one row a
    clock unit, the noise of `y` about ``x2 + 2 x0 - x1`` 0.3."""
    rng = np.random.default_rng(seed)
    level = np.linspace(1000.0, -1000.0, n)
    x = rng.normal(size=(n, 2))
    y = level + 2.0 * x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=n)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": level, "y": y})


def build(model: str, extra: dict, **kw) -> dict:
    opts = dict(
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=HALF_LIFE,
        min_weight=10.0,
        emit_metrics=True,
    )
    opts.update(extra)
    opts.update(kw)
    return getattr(po.spec, model)("m", **opts)


def running_hit_rate(pred: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """The bank's ``hit_rate`` from its definition (README, "Per-row
    diagnostics"): the exponentially weighted share of the rows before this
    one whose prediction and outcome fall on the same side of zero, a row
    where either is zero or missing on neither and aged all the same; null
    before any row is scored. One row a clock unit at weight 1."""
    out = np.full(len(y), np.nan)
    hits, weight = 0.0, 0.0
    for i in range(len(y)):
        if weight > 0.0:
            out[i] = hits
        p, v = pred[i], y[i]
        if np.isfinite(p) and np.isfinite(v) and p != 0.0 and v != 0.0:
            hit = float(np.sign(p) == np.sign(v))
            new = lam * weight + 1.0
            hits = (lam * weight * hits + hit) / new
            weight = new
        else:
            weight *= lam
    return out


def _fit(model: str, extra: dict, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The prediction and the running ``hit_rate`` on each row."""
    out = po.ModelBank([build(model, extra)]).fit_predict(df)["m"].struct.unnest()
    pred = next(c for c in out.columns if c.startswith("pred_"))
    hit = next(c for c in out.columns if c.startswith("hit_rate_"))
    return out[pred].to_numpy().astype(float), out[hit].to_numpy().astype(float)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
def test_the_hit_rate_is_right_on_both_sides_of_zero(model, extra):
    """The bank's ``hit_rate`` is the sign test's running share from its
    definition on every row (`running_hit_rate`, to 1e-12), and right on
    each side: every row predicted on the side's stretch, the running share
    at least 0.99 there.

    And nothing in a fit reads which side of zero it sits on: the same
    stream with the target and every feature negated, which crosses
    upward, is fitted as the mirror of this one -- the predictions negated,
    the hit rate the same -- to the last bit. Each step is odd in the
    target and the features together, the intercept's column of ones aside,
    which flips the intercept and leaves each slope; a spread, a cut and a
    band are even, and the check at the median is odd. The errors on the two
    sides differ, by each model's start from zero rather than by the sign:
    `pa` and `sgd` travel to the level over the first side (`pa` 1,800 off
    at row 800 and 3 at row 1,600, its steps pushed along the level
    feature's lag behind its mean), and `ftrl`'s error grows with the level
    it does not centre."""
    df = crossing()
    pred, got = _fit(model, extra, df)
    y = df["y"].to_numpy()
    lam = 0.5 ** (1.0 / HALF_LIFE)
    want = running_hit_rate(pred, y, lam)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12, equal_nan=True)
    for side in (ABOVE, BELOW):
        assert np.isfinite(pred[side]).all(), (model, side)
        assert got[side].min() >= 0.99, (model, side, got[side].min())
    mirrored = df.select(-pl.all())
    pred_m, got_m = _fit(model, extra, mirrored)
    np.testing.assert_array_equal(pred_m, -pred)
    np.testing.assert_array_equal(got_m, got)


def _decayed_least_squares(df: pl.DataFrame, rows: range) -> np.ndarray:
    """Row ``t``'s prediction by the exponentially weighted least-squares
    fit of the rows before it, at ``HALF_LIFE``: ``numpy.linalg.lstsq`` on
    the rows centred at their weighted means, which is exact at any level
    (`test_second_opinion.TestTheCrossMomentsAreCentred`)."""
    x = df.select("x0", "x1", "x2").to_numpy()
    y = df["y"].to_numpy()
    out = []
    for t in rows:
        w = 0.5 ** (((t - 1) - np.arange(t)) / HALF_LIFE)
        xm, ym = w @ x[:t] / w.sum(), w @ y[:t] / w.sum()
        sw = np.sqrt(w)[:, None]
        slopes = np.linalg.lstsq((x[:t] - xm) * sw, (y[:t] - ym) * sw[:, 0], rcond=None)[0]
        out.append(ym + (x[t] - xm) @ slopes)
    return np.array(out)


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        ("ewridge", {"ridge": 0.0, "max_rows_between_solves": 1}),
        ("lasso", {"lasso_path": [0.0], "max_rows_between_solves": 1, "tol": 1e-14}),
        ("huber", {"huber_delta": 1e9, "ridge": 0.0, "max_rows_between_solves": 1}),
    ],
    ids=["ewridge", "lasso", "huber"],
)
def test_a_least_squares_fit_is_numpys_across_the_crossing(model, extra):
    """``ewridge`` with no ridge, ``lasso`` at no penalty and ``huber`` with
    a cut no residual reaches are each the decayed least-squares fit, and
    are numpy's on every 50th row from the 200th, on both sides of the
    crossing (row 1,200) and through it: 5.7e-13 at most, measured, each,
    2.6e-13 beside the crossing."""
    df = crossing(n=2400, seed=19)
    rows = range(200, 2400, 50)
    out = po.ModelBank([build(model, extra)]).fit_predict(df)["m"].struct.unnest()
    pred_field = next(c for c in out.columns if c.startswith("pred_"))
    got = out[pred_field].to_numpy()[list(rows)]
    want = _decayed_least_squares(df, rows)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-11)


def test_a_quantile_fit_is_quantregs_across_the_crossing():
    """``quantile`` without decay, at 0.25 and 0.75 over the whole stream,
    the crossing inside it: statsmodels' ``QuantReg`` of the same rows, the
    level feature's slope and the intercept included (0.011 and 0.0028 off,
    measured)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(23)
    n = 20_000
    level = np.linspace(1000.0, -1000.0, n)
    x = rng.normal(size=(n, 2))
    y = level + 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, n)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": level, "y": y})
    z = sm.add_constant(np.column_stack([x, level]))
    for tau in (0.25, 0.75):
        spec = po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            quantile=tau,
            half_life=float("inf"),
            max_rows_between_solves=1,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        got = bank.coef("m")["coef"].to_numpy()
        want = np.asarray(sm.QuantReg(y, z).fit(q=tau).params)
        assert np.max(np.abs(got - want)) <= 0.03, (tau, got, want)
