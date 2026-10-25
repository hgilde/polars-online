"""Convergence class 6: equivariance (docs/PLAN.md task 278).

A fit must transform as the theory of its estimator says when the data are
transformed: scale the target and the predictions scale with it, shift it by
``x . c`` and the coefficients move by ``c``, rescale or shift a feature and
the intercept and that slope absorb it. Each of these is a statement about
this library's arithmetic alone, so the oracle is the model on the
untransformed stream, and what the class looks for is a constant in the
target's or a feature's units that should have been relative: a tolerance, a
clip, an epsilon, a floor.

**Powers of two are exact.** Multiplying by ``2^k`` is exact in IEEE 754
arithmetic while nothing overflows or goes subnormal, and ``+``, ``-``,
``*``, ``/`` and ``sqrt`` are correctly rounded, so
``fl(2^k a + 2^k b) = 2^k fl(a + b)``, ``fl(2^k a * b) = 2^k fl(a * b)``,
``fl(2^k a / 2^k b) = fl(a / b)``, ``sqrt(4^k a) = 2^k sqrt(a)``, and every
comparison keeps its answer. A computation built from these with no constant
in the target's units therefore returns bitwise ``2^k`` times its outputs in
the target's units, ``4^k`` times those in its squared units, and the same
bits for everything unit-free: ``weight_sum``, ``settled_frac``, a gate's
``withheld_reason``, a z-score, a p-value. A ratio's bits are unchanged too,
so ``exp`` and ``ln`` of one see the same input on every platform. The tests
scale by ``2^±10``, ``2^±20`` and ``2^±30`` (about ``1e±3``, ``1e±6`` and
``1e±9``); the squares stay far from overflow and the subnormals. Every
deviation from the bits is therefore a constant reached by the target, the
defect this class exists for, and the tests hold the bits, not a tolerance.

**A parameter in the target's units is scaled with it.** Theory maps such a
parameter, and the builder documents its units: ``lasso``'s penalties and
its ``tol``, ``pa1``'s ``c``, ``kalman``'s ``obs_var`` (squared), ``ftrl``'s
``alpha`` and ``beta``, ``sgd``'s ``clip_gradient`` (the squared loss's
gradient is the residual times the feature) and its learning rate under the
sign-valued losses (a step of ``lr * psi * z`` moves a coefficient by the
rate whatever the target's scale). ``Y_SCALE`` builds every spec at a scale
``c`` with those parameters mapped, so the test checks the map as well as
the arithmetic.

**Decimal scales and shifts are rounding.** ``fl(10^k y) = 10^k y (1 +
delta)`` with ``|delta| <= u``, and a shift ``y + L`` is carried by the
target's mean, so neither is exact; their bounds are derived per test below,
in units of ``u = 2^-53``.

What is not equivariant by design is stated, not tested as if it were:
``lasso``'s L1 penalty pulls toward zero, so a shift ``x . c`` that flips an
active coefficient's sign moves the fit by the subgradient
(``TestRegressionShift``); an unstandardized ridge is in the features'
squared units (``TestAffineFeatures``); ``ftrl`` has no feature scaling with
an intercept (its builder says so); a model whose coefficients start at zero
(``rls``'s prior, ``kalman``, ``sgd``, ``pa``) moves to a shifted target
only as fast as its own dynamics forget the start, which
``TestRegressionShift`` holds to the closed forms the theory gives.

The tables are plain dictionaries of builders so that they fold into the
convergence harness's registry (one entry per model, its classes listed).
``huber`` and ``sgd(loss="huber")`` wait in ``AFTER_TASK_277``: task 277
changes their warm-up, and they join ``Y_SCALE`` when it lands.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "mixed"

U = 2.0**-53
K = 3
FEATURES = [f"x{i}" for i in range(K)]
#: Rows of the essentials' stream and of the full grid's.
N_ESSENTIAL = 400
N_FULL = 1500
#: The full grid's seeds: a tail over 50 streams, never a median (task 278's
#: rules).
SEEDS = range(50)
#: The scales, as powers of two.
POWERS = (-30, -20, -10, 10, 20, 30)

_COMMON: dict[str, Any] = dict(targets=["y"], features=FEATURES, half_life=300.0, coef_every=0)


def stream(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Three features at different scales and levels, a target with an
    intercept and heavy-tailed noise (t with 5 degrees of freedom), so the
    intercept, the centring and the robust paths all have work to do."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, K)) * np.array([1.0, 3.0, 0.5]) + np.array([0.5, -2.0, 4.0])
    y = 0.3 + x @ np.array([1.5, -0.7, 2.0]) + 0.5 * rng.standard_t(5, n)
    return x, y


def fit(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """The spec's output over one chunk, its struct unnested."""
    cols: dict[str, Any] = {f: x[:, i] for i, f in enumerate(FEATURES)}
    cols["y"] = y
    with warnings.catch_warnings():
        # A ReadinessWarning about a ridge-held coefficient is the spec's,
        # and the same at every scale; it is not what is tested here.
        warnings.simplefilter("ignore", po.ReadinessWarning)
        return po.ModelBank([spec]).fit_predict(pl.DataFrame(cols))["m"].struct.unnest()


def floats(s: pl.Series) -> np.ndarray:
    """A numeric or list field as one float array, nulls as NaN."""
    if isinstance(s.dtype, (pl.List, pl.Array)):
        vals = s.to_list()
        width = max((len(v) for v in vals if v is not None), default=0)
        return np.array(
            [
                np.full(width, np.nan) if v is None else [np.nan if q is None else q for q in v]
                for v in vals
            ],
            float,
        ).ravel()
    return s.cast(pl.Float64).to_numpy()


def preds(out: pl.DataFrame) -> list[str]:
    """The prediction fields, in the order the spec declares them (a lasso
    path's names carry its penalties, which a scaled spec scales)."""
    return [c for c in out.columns if c.startswith("pred_")]


# ---------------------------------------------------------------------------
# The models. Each builder takes the target's scale `c` and maps every
# parameter in the target's units by it, as the builder's docstring states
# the units.

Builder = Callable[..., dict[str, Any]]

Y_SCALE: dict[str, Builder] = {
    "ewridge": lambda c, **kw: po.spec.ewridge("m", **_COMMON, **kw),
    "ewridge_standardized": lambda c, **kw: po.spec.ewridge("m", standardize=True, **_COMMON, **kw),
    "ewridge_window": lambda c, **kw: po.spec.ewridge("m", window_size=200.0, **_COMMON, **kw),
    "rls": lambda c, **kw: po.spec.rls("m", **_COMMON, **kw),
    # The L1 threshold and the descent's `tol` are in the target's units.
    "lasso": lambda c, **kw: po.spec.lasso(
        "m", penalties=[0.1 * c, 0.01 * c, 0.0], tol=1e-10 * c, **_COMMON, **kw
    ),
    "kalman": lambda c, **kw: po.spec.kalman("m", coef_half_life=200.0, **_COMMON, **kw),
    "kalman_obs_var": lambda c, **kw: po.spec.kalman(
        "m", coef_half_life=200.0, obs_var=0.25 * c * c, **_COMMON, **kw
    ),
    "kalman_raw": lambda c, **kw: po.spec.kalman(
        "m", coef_half_life=200.0, standardize=False, **_COMMON, **kw
    ),
    "quantile_median": lambda c, **kw: po.spec.quantile("m", quantile=0.5, **_COMMON, **kw),
    "quantile_tail_standardized": lambda c, **kw: po.spec.quantile(
        "m", quantile=0.9, standardize=True, **_COMMON, **kw
    ),
    # The squared loss's gradient is the residual times the feature: the
    # clip is in the target's units (`TestKnownDefects` pins its default).
    "sgd_squared": lambda c, **kw: po.spec.sgd(
        "m", loss="squared", clip_gradient=1e3 * c, **_COMMON, **kw
    ),
    "sgd_inv_scaling": lambda c, **kw: po.spec.sgd(
        "m",
        loss="squared",
        schedule="inv_scaling",
        learning_rate=0.05,
        clip_gradient=1e3 * c,
        **_COMMON,
        **kw,
    ),
    # A sign-valued loss's gradient has no target units, its step does: the
    # rate is the target over the feature squared.
    "sgd_quantile": lambda c, **kw: po.spec.sgd(
        "m", loss="quantile", quantile=0.7, learning_rate=0.01 * c, **_COMMON, **kw
    ),
    "sgd_epsilon_insensitive": lambda c, **kw: po.spec.sgd(
        "m", loss="epsilon_insensitive", learning_rate=0.01 * c, **_COMMON, **kw
    ),
    "pa": lambda c, **kw: po.spec.pa("m", mode="pa", **_COMMON, **kw),
    "pa1": lambda c, **kw: po.spec.pa("m", mode="pa1", c=1.0 * c, **_COMMON, **kw),
    "pa2": lambda c, **kw: po.spec.pa("m", mode="pa2", **_COMMON, **kw),
    "ftrl": lambda c, **kw: po.spec.ftrl(
        "m", loss="squared", alpha=0.1 * c, beta=1.0 * c, **_COMMON, **kw
    ),
    "holt": lambda c, **kw: po.spec.holt("m", targets=["y"], half_life=300.0, **kw),
}

#: Measured bitwise exact on main at `fe98f01` under `2^±20` (2026-10-25);
#: their decimal scalings move predictions by up to 5e-4 of the target's
#: spread, the resolution of the octave-bucketed MAD sketch whose buckets
#: are powers of two. Registered in `Y_SCALE` once task 277 lands.
AFTER_TASK_277: dict[str, Builder] = {
    "huber": lambda c, **kw: po.spec.huber("m", **_COMMON, **kw),
    "sgd_huber": lambda c, **kw: po.spec.sgd(
        "m", loss="huber", clip_gradient=1e3 * c, **_COMMON, **kw
    ),
}

#: Every diagnostic and gate a regression takes, switched on together, and
#: thresholds set so each gate withholds rows: `min_weight` 40 and
#: `max_error_inflation` 1.02 (`ewridge`'s `sqrt(1 + edf / n)` reaches it
#: near 100 rows), so a gate that read the target's units would withhold a
#: different set of rows.
DIAGNOSTICS: dict[str, Any] = dict(
    emit_sigma=True,
    emit_zscore=True,
    emit_metrics=True,
    conformal=0.1,
    resid_quantiles=[0.1, 0.9],
    emit_autocorr=True,
    emit_drift=True,
    emit_calibration=True,
    emit_breaks=True,
    emit_robust_se=True,
    emit_specification=True,
    emit_tails=True,
    emit_influence=True,
    emit_feature_health=True,
    emit_se_coef=True,
    emit_error_inflation=True,
    min_weight=40.0,
    max_error_inflation=1.02,
    min_settled_frac=0.2,
)


def with_diagnostics(builder: Builder, c: float) -> dict[str, Any]:
    """The model's spec at scale `c` with every diagnostic in `DIAGNOSTICS`
    it accepts: one it refuses by name is left out (a model with no residual
    variance has no `emit_se_coef`, one with no least-squares fit no
    `emit_robust_se`)."""
    extra = dict(DIAGNOSTICS)
    for _ in range(len(DIAGNOSTICS) + 1):
        try:
            return builder(c, **extra)
        except (ValueError, TypeError) as e:
            refused = [k for k in extra if k in str(e)]
            if not refused:
                raise
            for k in refused:
                extra.pop(k)
    raise AssertionError("every diagnostic refused")


# ---------------------------------------------------------------------------


def assert_scaled(base: pl.DataFrame, other: pl.DataFrame, c: float, what: str) -> None:
    """`other` is `base` with the target scaled by a power of two `c`: the
    predictions and residuals bitwise `c` times, the coefficients too (an
    intercept in the target's units, a slope in the target's over the
    feature's), and the weight, the settled share and the gates bitwise the
    same."""
    for f, g in zip(preds(base), preds(other), strict=True):
        np.testing.assert_array_equal(floats(base[f]) * c, floats(other[g]), err_msg=f"{what}: {f}")
    # The same layout at every scale; a path's names carry its penalties.
    for f, g in zip(base.columns, other.columns, strict=True):
        if f.startswith("resid_") or f == "coef":
            np.testing.assert_array_equal(
                floats(base[f]) * c, floats(other[g]), err_msg=f"{what}: {f}"
            )
    for f in ("weight_sum", "settled_frac"):
        if f in base.columns:
            np.testing.assert_array_equal(floats(base[f]), floats(other[f]), err_msg=f"{what}: {f}")
    if "withheld_reason" in base.columns:
        assert (
            base["withheld_reason"].cast(pl.String).equals(other["withheld_reason"].cast(pl.String))
        ), what


class TestPowerOfTwoTargetScale:
    """Every model's predictions, residuals and coefficients scale bitwise
    with the target, and its weight, settled share and gates do not move,
    the module docstring's argument."""

    @pytest.mark.parametrize("model", sorted(Y_SCALE))
    def test_the_fit_scales_to_the_bit(self, model):
        x, y = stream(0, N_ESSENTIAL)
        base = fit(Y_SCALE[model](1.0), x, y)
        for k in (-20, 20):
            c = 2.0**k
            assert_scaled(base, fit(Y_SCALE[model](c), x, y * c), c, f"{model} at 2^{k}")

    @pytest.mark.extended(reason="a grid: every model at six scales over five streams")
    @pytest.mark.parametrize("model", sorted(Y_SCALE))
    def test_the_fit_scales_to_the_bit_across_scales_and_streams(self, model):
        for seed in range(5):
            x, y = stream(seed, N_FULL)
            base = fit(Y_SCALE[model](1.0), x, y)
            for k in POWERS:
                c = 2.0**k
                assert_scaled(base, fit(Y_SCALE[model](c), x, y * c), c, f"{model} {seed} 2^{k}")


class TestEveryFieldKeepsItsUnits:
    """Every field of a spec with every diagnostic and gate on comes out of
    a power-of-two target scale bitwise in its own units: the same (a
    z-score, a p-value, a share, a gate), or scaled by `c` (a prediction, a
    spread, a band, a standard error) or by `c^2` (a variance). A diagnostic
    whose threshold or floor sat in the target's units would leave the bits
    here. The gates are set to withhold rows (`DIAGNOSTICS`), and the stream
    breaks halfway so the drift and break statistics move: both are printed
    as evidence that the paths ran."""

    MODELS = (
        "ewridge",
        "rls",
        "lasso",
        "kalman",
        "quantile_median",
        "sgd_squared",
        "pa",
        "ftrl",
        "holt",
    )

    @pytest.mark.parametrize("model", MODELS)
    def test_every_field_is_in_its_units_to_the_bit(self, model):
        x, y = stream(1, N_FULL)
        y[N_FULL // 2 :] += 3.0  # a break, for the drift and break statistics
        base = fit(with_diagnostics(Y_SCALE[model], 1.0), x, y)
        withheld = base["withheld_reason"].cast(pl.String).value_counts()
        print(model, len(base.columns), "fields;", withheld.to_dicts())
        assert base["withheld_reason"].null_count() < N_FULL, f"{model}: no gate withheld a row"
        if "drift_y" in base.columns:
            print(model, "drift flags:", int(base["drift_y"].sum()))
        for k in (-20, 20):
            c = 2.0**k
            other = fit(with_diagnostics(Y_SCALE[model], c), x, y * c)
            assert_scaled(base, other, c, f"{model} at 2^{k}")
            for f, g in zip(base.columns, other.columns, strict=True):
                a, b = base[f], other[g]
                if not a.dtype.is_numeric() and not isinstance(a.dtype, (pl.List, pl.Array)):
                    assert a.cast(pl.String).equals(b.cast(pl.String)), f"{model} 2^{k}: {f}"
                    continue
                fa, fb = floats(a), floats(b)
                units = [m for m in (1.0, c, c * c) if np.array_equal(fa * m, fb, equal_nan=True)]
                assert units, f"{model} at 2^{k}: {f} is in no unit of the target's"


#: Two backward-stable recursions over the same rows part only by their
#: rounding. A mean-form accumulator rounds a few times a row at the running
#: magnitude and keeps about `n_eff = 1 / (1 - lam)` rows of it (433 at a
#: half-life of 300 rows), so each run's forward error is at most a few
#: `n_eff * u` of the largest |target|, and the solve's condition (the
#: features' correlation matrix, below 10 here) multiplies it. A decimal
#: scale is a power-of-two scale (exact, above) of a mantissa in [1, 2), so
#: the two runs differ by twice that: `64 * n_eff * u * max|y| / sd(y)` in
#: units of the target's spread, 64 for the per-row operations, the
#: condition and the two runs. Measured 2026-10-25 over 50 streams and the
#: scales 1e-6, 1e-3, 1e3, 1e6: worst 6.6e-12 (`kalman`, seed 19; p95
#: 3.5e-12), 1.5e-13 for every mean-form model (seed 25), 2.2e-14 for the
#: gradient models; the bound is about 2e-11 on these streams, so the
#: worst sits at 0.31 of it (p95 0.16) and the mean-form models below 0.01.
DECIMAL_FACTOR = 64.0
N_EFF = 1.0 / (1.0 - 0.5 ** (1.0 / 300.0))


def worst_decimal_deviation(model: str, seed: int, n: int, scales) -> tuple[float, float]:
    """The largest |pred at scale c - c * pred| over the rows and `scales`,
    in units of `c * sd(y)`, and the bound above for the stream."""
    x, y = stream(seed, n)
    sd = float(np.std(y))
    base = fit(Y_SCALE[model](1.0), x, y)
    p0 = np.column_stack([floats(base[f]) for f in preds(base)])
    worst = 0.0
    for c in scales:
        out = fit(Y_SCALE[model](c), x, y * c)
        p1 = np.column_stack([floats(out[f]) for f in preds(out)])
        assert np.array_equal(np.isfinite(p0), np.isfinite(p1)), f"{model}: rows withheld differ"
        worst = max(worst, float(np.nanmax(np.abs(p1 - p0 * c))) / (sd * c))
    return worst, DECIMAL_FACTOR * N_EFF * U * float(np.max(np.abs(y))) / sd


class TestDecimalTargetScale:
    """A decimal scale is rounding, held to `DECIMAL_FACTOR`'s bound."""

    @pytest.mark.parametrize("model", sorted(Y_SCALE))
    def test_a_decimal_scale_moves_predictions_by_rounding(self, model):
        worst, bound = worst_decimal_deviation(model, 0, N_ESSENTIAL, (1e-6, 1e6))
        assert worst <= bound, (model, worst, bound)

    @pytest.mark.extended(reason="a grid: every model over 50 streams and four scales")
    @pytest.mark.parametrize("model", sorted(Y_SCALE))
    def test_a_decimal_scale_moves_predictions_by_rounding_over_50_streams(self, model):
        tail = []
        for seed in SEEDS:
            worst, bound = worst_decimal_deviation(model, seed, N_FULL, (1e-6, 1e-3, 1e3, 1e6))
            tail.append(worst / bound)
            assert worst <= bound, (model, seed, worst, bound)
        print(model, "worst / bound:", max(tail), "p95:", float(np.quantile(tail, 0.95)))


#: The models whose intercept absorbs a level exactly in theory: the level
#: reaches only the target's running mean (and the intercept recovered from
#: it, `ybar - m . b`), or, for `pa` under its default standardization and
#: `holt`, a level the first row sets.
LEVEL_EXACT = (
    "ewridge",
    "ewridge_standardized",
    "ewridge_window",
    "lasso",
    "quantile_median",
    "quantile_tail_standardized",
    "pa",
    "holt",
)

#: A level `L` is carried in the target's running mean, which rounds a few
#: times a row at the magnitude `|L| + max|y|`; an EW mean of quantities each
#: within a few `u` of that magnitude is within the same, so the error does
#: not accumulate with the rows. The intercept's recovery and the prediction
#: add a few roundings more, and `holt`'s trend takes the difference of two
#: levels each carrying that error. Measured 2026-10-25 over 50 streams and
#: `L` in 1e3, 1e6, -1e9: worst 16 (the mean-form models, seed 25), 7.3
#: (`pa`), 49 (`holt`, seed 21; p95 47) units of `u * (|L| + max|y|)`.
LEVEL_FACTOR = 128.0


def worst_level_deviation(model: str, seed: int, n: int, levels) -> float:
    """The largest |pred of y + L - (pred + L)|, in `u * (|L| + max|y|)`."""
    x, y = stream(seed, n)
    base = fit(Y_SCALE[model](1.0), x, y)
    p0 = np.column_stack([floats(base[f]) for f in preds(base)])
    worst = 0.0
    for level in levels:
        out = fit(Y_SCALE[model](1.0), x, y + level)
        p1 = np.column_stack([floats(out[f]) for f in preds(out)])
        assert np.array_equal(np.isfinite(p0), np.isfinite(p1)), f"{model}: rows withheld differ"
        scale = U * (abs(level) + float(np.max(np.abs(y))))
        worst = max(worst, float(np.nanmax(np.abs(p1 - (p0 + level)))) / scale)
    return worst


class TestLevelShift:
    """The target shifted by a level: the intercept absorbs it, to the
    rounding `LEVEL_FACTOR` derives, and nothing else moves."""

    @pytest.mark.parametrize("model", LEVEL_EXACT)
    def test_the_intercept_absorbs_a_level(self, model):
        assert worst_level_deviation(model, 0, N_ESSENTIAL, (1e6,)) <= LEVEL_FACTOR

    @pytest.mark.extended(reason="a grid: eight models over 50 streams and three levels")
    @pytest.mark.parametrize("model", LEVEL_EXACT)
    def test_the_intercept_absorbs_a_level_over_50_streams(self, model):
        tail = [worst_level_deviation(model, s, N_FULL, (1e3, 1e6, -1e9)) for s in SEEDS]
        print(model, "worst:", max(tail), "seed", int(np.argmax(tail)))
        assert max(tail) <= LEVEL_FACTOR, (model, max(tail), int(np.argmax(tail)))


def coefs(out: pl.DataFrame) -> np.ndarray:
    """`coef` as a row per stream row (every row carries one: `coef_every=0`)."""
    return floats(out["coef"]).reshape(len(out), -1)


#: The shift `y + z . SHIFT`, `z = [1, x]`: a level and a slope change per
#: feature, the slopes larger than the stream's own coefficients so that a
#: model that kept part of the old fit is far off.
SHIFT = np.array([7.0, 10.0, -20.0, 5.0])


class TestRegressionShift:
    """`y + z . c`: the fit moves by `c`, exactly where theory says it does,
    and by the closed form of its start where the model starts from zero.

    - A ridge (`ewridge`, `rls`) shrinks toward its prior, so the shifted
      problem with the prior shifted by `c` is the original one moved by `c`:
      `(S + R) b' = S b_y + S c + R (p + c)` gives `b' = b + c`.
    - A quantile regression is regression-equivariant (Koenker and Bassett
      1978): the rows' residuals, and so the band and the steps, are the
      same; at ridge 0 nothing pulls toward zero.
    - The lasso's stationarity `C b - d + l * s = 0`, `s` the signs, moves
      with `c` only while every coefficient keeps its sign and none is zero;
      a shift that flips one moves the fit by the subgradient, `2 l` in that
      coordinate's equation. That is the penalty, not a defect, and the test
      states it.
    - A Kalman filter with a fixed noise, no process noise and no
      standardization is Bayesian regression with prior mean 0 and
      covariance `P0 = p0 * R * I`, so the shifted fit is `b' = b + c - P_t
      P0^-1 c`, with `P_t^-1 = P0^-1 + sum z z' / R`: the start, forgotten as
      the data arrive.
    - `pa` with `eps = 0` projects the coefficients onto each row's
      hyperplane `y = z . b`, so the difference of the two runs, less `c`,
      is projected too, `d' = (I - z z' / |z|^2) d`, and its norm never
      grows. `sgd`'s squared loss multiplies it by `I - lr * w * z z'`,
      whose eigenvalues lie in `[-1, 1]` while `lr * w * |z|^2 <= 2`.
    """

    def _bound(self, x: np.ndarray, y: np.ndarray, shifted: np.ndarray) -> float:
        # As `DECIMAL_FACTOR`: two recursions at the shifted target's
        # magnitude, in units of the original's spread. Measured 2026-10-25
        # at about 2e-13 of the spread, about 1/1000 of the bound.
        return DECIMAL_FACTOR * N_EFF * U * float(np.max(np.abs(shifted))) / float(np.std(y))

    @pytest.mark.parametrize(
        "model",
        ["ewridge", "ewridge_standardized", "rls", "quantile_median", "quantile_tail"],
    )
    def test_the_fit_moves_by_the_shift(self, model):
        x, y = stream(2, N_FULL)
        z = np.column_stack([np.ones(len(y)), x])
        shifted = y + z @ SHIFT
        prior = [[0.0, *SHIFT[1:]]]
        base_spec, shifted_spec = {
            "ewridge": (
                po.spec.ewridge("m", **_COMMON),
                po.spec.ewridge("m", coef_prior=prior, **_COMMON),
            ),
            "ewridge_standardized": (
                po.spec.ewridge("m", standardize=True, **_COMMON),
                po.spec.ewridge("m", standardize=True, coef_prior=prior, **_COMMON),
            ),
            # `rls`'s prior penalizes the intercept too, so it shifts as well.
            "rls": (
                po.spec.rls("m", **_COMMON),
                po.spec.rls("m", coef_prior=[list(SHIFT)], **_COMMON),
            ),
            "quantile_median": (
                po.spec.quantile("m", quantile=0.5, ridge=0.0, **_COMMON),
                po.spec.quantile("m", quantile=0.5, ridge=0.0, **_COMMON),
            ),
            "quantile_tail": (
                po.spec.quantile("m", quantile=0.9, ridge=0.0, **_COMMON),
                po.spec.quantile("m", quantile=0.9, ridge=0.0, **_COMMON),
            ),
        }[model]
        p0 = floats(fit(base_spec, x, y)["pred_y"])
        p1 = floats(fit(shifted_spec, x, shifted)["pred_y"])
        dev = float(np.nanmax(np.abs(p1 - (p0 + z @ SHIFT)))) / float(np.std(y))
        assert dev <= self._bound(x, y, shifted), (model, dev)

    def test_the_lasso_moves_by_the_shift_while_no_sign_changes_and_not_after(self):
        x, y = stream(2, N_FULL)
        spec = po.spec.lasso("m", penalties=[0.1], tol=1e-14, max_iter=10_000, **_COMMON)
        z = np.column_stack([np.ones(len(y)), x])
        base = coefs(fit(spec, x, y))
        moved = coefs(fit(spec, x, y + z @ SHIFT))
        # Rows whose fit has every slope away from zero, each keeping its
        # sign under the shift: the stationarity conditions move with `c`.
        keep = np.all(
            (base[:, 1:] != 0) & (np.sign(base[:, 1:]) == np.sign(base[:, 1:] + SHIFT[1:])), axis=1
        )
        keep[:50] = False  # the warm-up's first solves, before the moments settle
        assert keep.sum() > N_FULL // 2, "the sign-keeping path did not run"
        np.testing.assert_allclose(moved[keep], base[keep] + SHIFT, rtol=0, atol=1e-9)
        # A shift that flips the first slope's sign: the penalty's subgradient
        # changes sign in that coordinate's equation, and the fit moves by
        # about `2 l / s_0` in the standardized coordinates, never by `c`.
        flip = np.array([0.0, -3.0, 0.0, 0.0])
        flipped = coefs(fit(spec, x, y + z @ flip))
        gap = np.abs(flipped[-1] - (base[-1] + flip))
        s0 = float(np.std(x[:, 0]))
        print(
            "lasso: the flipped slope misses the shift by", gap[1], "against 2 l / s0 =", 0.2 / s0
        )
        # `2 l (C^-1)_00 / s_0` in theory, `(C^-1)_00` near 1 for these
        # nearly uncorrelated features; measured 0.2019 against 0.1999.
        assert 0.5 < gap[1] / (0.2 / s0) < 2.0

    def test_a_kalman_filter_forgets_its_start_as_the_closed_form_says(self):
        x, y = stream(3, 600)
        z = np.column_stack([np.ones(len(y)), x])
        noise, p0 = 0.25, 1.0
        spec = po.spec.kalman("m", q=[0.0] * 4, obs_var=noise, p0=p0, standardize=False, **_COMMON)
        d = coefs(fit(spec, x, y + z @ SHIFT)) - coefs(fit(spec, x, y)) - SHIFT
        info = np.eye(4) / (p0 * noise)
        want = np.empty_like(d)
        for i in range(len(y)):
            info += np.outer(z[i], z[i]) / noise
            want[i] = -np.linalg.solve(info, SHIFT / (p0 * noise))
        # Both sides solve the same normal equations, the filter by `n`
        # rank-one updates of `P` and numpy by an LU of `P_n^-1`; each is
        # within `gamma_n * kappa` of the exact solution (Higham, Theorem
        # 7.2), `kappa` the condition of `P_n^-1`, in units of |c|. Measured
        # 2026-10-25: 1.2e-13 against a bound of about 1e-8.
        kappa = float(np.linalg.cond(info))
        bound = 2 * kappa * len(y) * U / (1 - len(y) * U) * float(np.linalg.norm(SHIFT))
        assert float(np.max(np.abs(d - want))) <= bound
        assert float(np.max(np.abs(d))) > 1.0, "the start was never far from the shift"

    @pytest.mark.parametrize("model", ["pa", "sgd"])
    def test_a_projection_never_grows_the_distance_to_the_shifted_fit(self, model):
        x, y = stream(4, N_FULL)
        z = np.column_stack([np.ones(len(y)), x])
        lr = 0.002
        spec = (
            po.spec.pa("m", mode="pa", eps=0.0, standardize=False, **_COMMON)
            if model == "pa"
            else po.spec.sgd(
                "m",
                loss="squared",
                learning_rate=lr,
                clip_gradient=float("inf"),
                standardize=False,
                **_COMMON,
            )
        )
        if model == "sgd":
            assert float(np.max(lr * np.sum(z * z, axis=1))) <= 2.0, "the contraction's premise"
        base = coefs(fit(spec, x, y))
        d = np.linalg.norm(coefs(fit(spec, x, y + z @ SHIFT)) - base - SHIFT, axis=1)
        # Each step rounds a few times at the coefficients' magnitude.
        slack = 16 * U * (np.linalg.norm(base, axis=1)[1:] + np.linalg.norm(SHIFT))
        assert np.all(d[1:] <= d[:-1] + slack), model
        print(model, "distance to the shifted fit:", d[0], "->", d[-1])
        assert d[-1] < d[0] / 3, "the contraction never acted"


#: A power of two per feature, from 1e-6 to 1e6, different for each so no
#: common factor can hide a feature that is treated in absolute units.
FEATURE_POWERS = np.array([-20, 10, 20])

#: The models whose fit is free of the features' units: standardized ones,
#: and unstandardized ones with nothing in the features' units (ridge 0).
X_SCALE: dict[str, Builder] = {
    "ewridge_standardized": Y_SCALE["ewridge_standardized"],
    "ewridge_ridge_0": lambda c, **kw: po.spec.ewridge("m", ridge=0.0, **_COMMON, **kw),
    "lasso": Y_SCALE["lasso"],
    "kalman": Y_SCALE["kalman"],
    "kalman_obs_var": Y_SCALE["kalman_obs_var"],
    "quantile_ridge_0": lambda c, **kw: po.spec.quantile(
        "m", quantile=0.5, ridge=0.0, **_COMMON, **kw
    ),
    "quantile_tail_standardized": Y_SCALE["quantile_tail_standardized"],
    "sgd_squared": Y_SCALE["sgd_squared"],
    "sgd_quantile": Y_SCALE["sgd_quantile"],
    "sgd_epsilon_insensitive": Y_SCALE["sgd_epsilon_insensitive"],
    "pa": Y_SCALE["pa"],
    "pa1": Y_SCALE["pa1"],
    "pa2": Y_SCALE["pa2"],
}

#: A uniform scale `a` of every feature, with each parameter in the
#: features' units mapped as its builder states them: a ridge in their
#: squared units (`* a^2`), `rls`'s prior strength likewise, and, without an
#: intercept (whose 1 does not scale), `ftrl`'s four (`alpha / a`, `beta *
#: a`, `l1 * a`, `l2 * a^2`) and the raw Kalman prior and process noise,
#: in one over the features' squared units.
UNIFORM_X_SCALE: dict[str, Callable[[float], dict[str, Any]]] = {
    "ewridge": lambda a: po.spec.ewridge("m", ridge=1e-6 * a * a, **_COMMON),
    "quantile": lambda a: po.spec.quantile("m", quantile=0.5, ridge=1e-6 * a * a, **_COMMON),
    "rls_no_intercept": lambda a: po.spec.rls(
        "m", delta=1.0 * a * a, fit_intercept=False, **_COMMON
    ),
    "ftrl_no_intercept": lambda a: po.spec.ftrl(
        "m",
        loss="squared",
        alpha=0.1 / a,
        beta=1.0 * a,
        l1=0.01 * a,
        l2=0.1 * a * a,
        fit_intercept=False,
        **_COMMON,
    ),
    "kalman_raw_no_intercept": lambda a: po.spec.kalman(
        "m",
        q=[1e-6 / (a * a)] * K,
        p0=1.0 / (a * a),
        standardize=False,
        fit_intercept=False,
        **_COMMON,
    ),
}


def assert_same_fit_in_new_units(
    base: pl.DataFrame, other: pl.DataFrame, scale: np.ndarray, intercept: bool, what: str
) -> None:
    """`other` fitted the features times `scale` (powers of two): the same
    predictions to the bit, each slope divided by its feature's scale to
    the bit, the intercept unchanged."""
    for f, g in zip(preds(base), preds(other), strict=True):
        np.testing.assert_array_equal(floats(base[f]), floats(other[g]), err_msg=f"{what}: {f}")
    # From the row whose centred Gram first has full rank on: before it the
    # solve is singular, and an unpenalized fit picks a point of the null
    # space through the solver's jitter, `eps * trace / k`, whose trace
    # mixes the features' units (`solve.rs`). Those coefficients predict
    # nothing (the gates withhold the rows) and are arbitrary by design.
    a, b = coefs(base)[FULL_RANK:], coefs(other)[FULL_RANK:]
    per = np.concatenate([[1.0], 1.0 / scale]) if intercept else 1.0 / scale
    per = np.tile(per, a.shape[1] // len(per))
    np.testing.assert_array_equal(a * per, b, err_msg=f"{what}: coef")


#: The first row whose fit is determined: the features and the intercept
#: need `K + 1` rows, and their centred Gram one more.
FULL_RANK = K + 2


class TestAffineFeatures:
    """A feature rescaled or shifted: the slope and the intercept absorb it.

    A ridge in the features' squared units is not equivariant by design
    (the builders say so), so an unstandardized ridge enters `X_SCALE` only
    at 0 and `UNIFORM_X_SCALE` with the ridge scaled; `ftrl` has no feature
    scaling at all with an intercept, whose 1 does not scale.
    """

    @pytest.mark.parametrize("model", sorted(X_SCALE))
    def test_a_power_of_two_per_feature_changes_nothing_but_the_slopes(self, model):
        x, y = stream(5, N_ESSENTIAL)
        scale = 2.0**FEATURE_POWERS
        base = fit(X_SCALE[model](1.0), x, y)
        other = fit(X_SCALE[model](1.0), x * scale, y)
        assert_same_fit_in_new_units(base, other, scale, True, model)

    @pytest.mark.parametrize("model", sorted(UNIFORM_X_SCALE))
    def test_a_uniform_power_of_two_with_its_parameters_changes_nothing_but_the_slopes(self, model):
        x, y = stream(5, N_ESSENTIAL)
        base = fit(UNIFORM_X_SCALE[model](1.0), x, y)
        for k in (-20, 20):
            a = 2.0**k
            other = fit(UNIFORM_X_SCALE[model](a), x * a, y)
            intercept = "no_intercept" not in model
            assert_same_fit_in_new_units(base, other, np.full(K, a), intercept, f"{model} 2^{k}")

    @pytest.mark.extended(reason="a grid: thirteen models over ten streams and three scalings")
    @pytest.mark.parametrize("model", sorted(X_SCALE))
    def test_powers_of_two_per_feature_across_streams(self, model):
        for seed in range(10):
            x, y = stream(seed, N_FULL)
            base = fit(X_SCALE[model](1.0), x, y)
            for powers in ([-30, 0, 30], [10, -10, 20], [-20, 10, 20]):
                scale = 2.0 ** np.array(powers)
                other = fit(X_SCALE[model](1.0), x * scale, y)
                assert_same_fit_in_new_units(base, other, scale, True, f"{model} {seed} {powers}")

    @pytest.mark.parametrize("model", sorted(X_SCALE))
    def test_a_shifted_feature_is_absorbed_by_the_intercept(self, model):
        worst = worst_feature_shift_deviation(model, 6, N_ESSENTIAL)
        assert worst <= FEATURE_SHIFT_FACTOR, (model, worst)

    @pytest.mark.extended(reason="a grid: thirteen models over 50 streams")
    @pytest.mark.parametrize("model", sorted(X_SCALE))
    def test_a_shifted_feature_is_absorbed_over_50_streams(self, model):
        tail = [worst_feature_shift_deviation(model, s, N_FULL) for s in SEEDS]
        print(model, "worst:", max(tail), "seed", int(np.argmax(tail)))
        assert max(tail) <= FEATURE_SHIFT_FACTOR, (model, max(tail), int(np.argmax(tail)))


#: The shifts: three levels, one of them a million times the feature's
#: spread.
FEATURE_SHIFT = np.array([1e3, -1e6, 1e4])
#: A shift `b_j` reaches the prediction as `sum_j beta_j x_j` with `x_j`
#: near `b_j`, terms of size `|beta_j b_j|` that cancel against the
#: intercept down to the target's size: even exact coefficients give an
#: error of a few `u * sum_j |beta_j| (|b_j| + max|x_j|)` there, and the
#: centred moments (a Welford update about the running mean) add a few
#: more of the same. The bound is that sum, with the stream's true
#: coefficients, times 64. Measured 2026-10-25 over 50 streams: worst 33
#: (`kalman`, seed 25; its anchor re-maps the state once the shifted means
#: move), 29 (`sgd`'s epsilon-insensitive loss, seed 27), 22 or less for the
#: rest.
FEATURE_SHIFT_FACTOR = 64.0
TRUE_BETA = np.array([1.5, -0.7, 2.0])


def worst_feature_shift_deviation(model: str, seed: int, n: int) -> float:
    """The largest |pred with x + b - pred| in units of
    `u * sum_j |beta_j| (|b_j| + max|x_j|)`."""
    x, y = stream(seed, n)
    base = fit(X_SCALE[model](1.0), x, y)
    other = fit(X_SCALE[model](1.0), x + FEATURE_SHIFT, y)
    scale = U * float(np.sum(np.abs(TRUE_BETA) * (np.abs(FEATURE_SHIFT) + np.max(np.abs(x), 0))))
    worst = 0.0
    for f, g in zip(preds(base), preds(other), strict=True):
        a, b = floats(base[f]), floats(other[g])
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{model}: rows withheld differ"
        worst = max(worst, float(np.nanmax(np.abs(a - b))) / scale)
    return worst


class TestKnownDefects:
    """Constants in the target's units that this class found (task 278 (b),
    2026-10-25), each pinned as it stands so that the fix flips the test
    rather than passing it unnoticed. The equivariant configuration beside
    each is held in `Y_SCALE`. When a defect is fixed, its test here
    becomes the bitwise check of `TestPowerOfTwoTargetScale` for the
    default configuration and moves there.
    """

    def test_sgds_default_gradient_clip_is_in_the_targets_units(self):
        # `sgd`'s docstring: "the squared and Huber losses fit a target scaled
        # by c as the unscaled one, scaled by c". The default `clip_gradient`
        # of 1e3 is in the target's units times the feature's, so a target
        # in the millions has every gradient clipped and the fit is not the
        # unscaled one scaled: on every row of the second half it is off by
        # at least half the target's spread, 3.4 on average (seed 0, 2^20:
        # 0.59 to 6.5).
        x, y = stream(0, N_FULL)
        c = 2.0**20
        spec = po.spec.sgd("m", loss="squared", **_COMMON)
        p0 = floats(fit(spec, x, y)["pred_y"])
        p1 = floats(fit(spec, x, y * c)["pred_y"])
        off = np.abs(p1 - p0 * c)[N_FULL // 2 :] / (c * float(np.std(y)))
        print("sgd at 2^20 with the default clip: off by", float(off.min()), "to", float(off.max()))
        assert float(off.min()) > 0.5
        assert float(off.mean()) > 2.0

    def test_adagrads_epsilon_is_in_the_gradients_units(self):
        # `lr / (sqrt(G_i) + 1e-8)`: the 1e-8 is in the gradient's units,
        # so a target near 1e-6 has it at about 1% of `sqrt(G_i)`, and the
        # fit is not the unscaled one scaled even with every unit-bearing
        # parameter mapped (seed 0: 4e-4 of the spread at 2^-20, 4e-10 at
        # 2^20).
        x, y = stream(0, N_FULL)

        def spec(c: float) -> dict[str, Any]:
            return po.spec.sgd(
                "m",
                loss="squared",
                schedule="adagrad",
                learning_rate=0.1 * c,
                clip_gradient=1e3 * c,
                **_COMMON,
            )

        c = 2.0**-20
        p0 = floats(fit(spec(1.0), x, y)["pred_y"])
        p1 = floats(fit(spec(c), x, y * c)["pred_y"])
        off = float(np.nanmax(np.abs(p1 - p0 * c))) / (c * float(np.std(y)))
        print("adagrad at 2^-20: off by", off)
        assert off > 1e-5

    def test_lassos_tol_is_in_the_targets_units_and_stops_early_in_silence(self):
        # The descent stops when no standardized coefficient `beta_i * s_i`,
        # which is in the target's units, moves by `tol` (default 1e-10). A
        # target near 1e-9 meets that after one sweep, so the path is
        # wherever the warm start left it: 0.20 of the spread off at 2^-30
        # and 1e-3 at 2^-20, with no descent counted unconverged in
        # `solve_failures`. (At 2^20 and up the same `tol` is below the
        # coefficients' rounding, and more descents run out of sweeps:
        # 16 counted against 10 at scale 1.)
        x, y = stream(0, N_FULL)

        def run(c: float) -> tuple[pl.DataFrame, int]:
            spec = po.spec.lasso("m", penalties=[0.1 * c, 0.01 * c, 0.0], **_COMMON)
            bank = po.ModelBank([spec])
            cols: dict[str, Any] = {f: x[:, i] for i, f in enumerate(FEATURES)}
            out = bank.fit_predict(pl.DataFrame({**cols, "y": y * c}))["m"].struct.unnest()
            return out, int(bank.solve_failures()["solve_failures"].sum())

        base, _ = run(1.0)
        c = 2.0**-30
        small, failures = run(c)
        off = max(
            float(np.nanmax(np.abs(floats(small[g]) - floats(base[f]) * c)))
            / (c * float(np.std(y)))
            for f, g in zip(preds(base), preds(small), strict=True)
        )
        print("lasso at 2^-30 with the default tol: off by", off, "with", failures, "counted")
        assert off > 0.1
        assert failures == 0
