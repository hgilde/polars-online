"""Task 14: online logistic regression / FTRL-proximal (docs/PLAN.md 4.6)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import ftrl_ref

TIER = "mixed"


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def _binary_df(n=6000, seed=21, noise=0.0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    logit = 1.5 * x0 - 0.5 * x1
    if noise:
        logit = logit + noise * rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return pl.DataFrame({"x0": x0, "x1": x1, "y0": y})


def _spec(**kw):
    d = dict(targets=["y0"], features=["x0", "x1"], half_life=1e9, min_weight=50.0)
    d.update(kw)
    return po.spec.ftrl("m", **d)


def test_predictions_are_probabilities():
    out = po.ModelBank([_spec()]).fit_predict(_binary_df())
    p = _pred(out)
    finite = p[np.isfinite(p)]
    assert finite.size > 1000
    assert (finite >= 0).all() and (finite <= 1).all()


def test_beats_the_base_rate_on_a_learnable_target():
    df = _binary_df()
    out = po.ModelBank([_spec()]).fit_predict(df)
    p = _pred(out)
    y = df["y0"].to_numpy()
    m = np.isfinite(p)
    # Log loss must beat always predicting the base rate.
    eps = 1e-12
    ll = -np.mean(y[m] * np.log(p[m] + eps) + (1 - y[m]) * np.log(1 - p[m] + eps))
    base = y[m].mean()
    ll_base = -np.mean(y[m] * np.log(base) + (1 - y[m]) * np.log(1 - base))
    assert ll < ll_base, f"log loss {ll} should beat base rate {ll_base}"


def test_resid_is_y_minus_p():
    df = _binary_df(n=2000)
    out = po.ModelBank([_spec()]).fit_predict(df)
    p = _pred(out)
    r = out["m"].struct.field("resid_y0").to_numpy().astype(float)
    y = df["y0"].to_numpy()
    m = np.isfinite(p) & np.isfinite(r)
    np.testing.assert_allclose(r[m], y[m] - p[m], atol=1e-12)


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(22)
    n = 5000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y0": (rng.random(n) < 0.5).astype(float),
        }
    )
    out = po.ModelBank([_spec(half_life=2000.0)]).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06, f"IC {ic}: predictions are not out-of-sample"


def test_forgets_a_regime_flip_on_the_clock():
    rng = np.random.default_rng(23)
    n = 4000
    x0 = rng.standard_normal(2 * n)
    y = np.concatenate([(x0[:n] > 0).astype(float), (x0[n:] < 0).astype(float)])
    df = pl.DataFrame({"x0": x0, "x1": np.zeros(2 * n), "y0": y})
    out = po.ModelBank([_spec(half_life=300.0)]).fit_predict(df)
    p = _pred(out)
    late = slice(2 * n - 500, 2 * n)
    hi = p[late][x0[late] > 0.5]
    assert np.nanmean(hi) < 0.4, "model did not follow the flipped regime"


def _plumbing_case():
    df, _ = synthetic(seed=64, n_groups=2, n_rows=200, k=2, null_frac=0.0)
    df = df.with_columns(y0=(pl.col("y0") > 0).cast(pl.Float64))
    kw = dict(
        features=["x0", "x1"],
        half_life=300.0,
        clock="t",
        gap_cap=50.0,
        min_weight=10.0,
    )
    return df, kw, po.spec.ftrl("m", targets=["y0"], group="group", **kw)


def test_chunk_invariance():
    df, _, spec = _plumbing_case()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_strict_binary_refuses_a_target_other_than_0_or_1():
    """``strict_binary`` was documented as an error and ran as a silent skip
    that still counted the row toward ``weight_sum`` (review 2026-09-12, S31). The
    chunk is refused, naming the row and the value, before any stream is
    touched -- as a label outside ``ew_class``'s classes is -- so the bank is
    left as it was and the corrected chunk can be fed. A null is no value,
    and passes."""
    df = pl.DataFrame(
        {"x0": [1.0, 2.0, 3.0, 4.0], "x1": [0.0] * 4, "y0": [0.0, None, 0.5, 1.0]},
        schema={"x0": pl.Float64, "x1": pl.Float64, "y0": pl.Float64},
    )
    bank = po.ModelBank([_spec(strict_binary=True, min_weight=0.0)])
    before = bank.save_bytes()
    with pytest.raises(ValueError, match=r"strict_binary.*row 2") as exc:
        bank.fit_predict(df)
    assert "0.5" in str(exc.value)
    assert bank.save_bytes() == before, "the refused chunk touched the bank"
    fixed = df.with_columns(y0=pl.when(pl.col("y0") == 0.5).then(None).otherwise(pl.col("y0")))
    assert _pred(bank.fit_predict(fixed)).size == 4


class TestStrictBinaryEdges:
    """``strict_binary``'s edges (task 278, batch 18): where nothing is
    refused it is the default to the bit, ``-0.0`` is the label 0, a NaN
    label is a null, and the label is checked whatever the row's weight."""

    def test_on_labels_of_zero_and_one_it_is_the_default(self):
        df = _binary_df(n=2000).with_columns(
            y0=pl.when(pl.int_range(pl.len()) % 9 == 4).then(None).otherwise(pl.col("y0"))
        )
        assert df["y0"].null_count() > 200
        strict = po.ModelBank([_spec(strict_binary=True)]).fit_predict(df)
        assert strict.equals(po.ModelBank([_spec()]).fit_predict(df), null_equal=True)

    def test_negative_zero_is_the_label_zero_and_nan_is_a_null(self):
        df = _binary_df(n=2000)
        signed = df.with_columns(y0=pl.when(pl.col("y0") == 0.0).then(-0.0).otherwise(pl.col("y0")))
        assert np.signbit(signed["y0"].to_numpy()).sum() > 500
        spec = _spec(strict_binary=True)
        out = po.ModelBank([spec]).fit_predict(signed)["m"]
        assert out.equals(po.ModelBank([spec]).fit_predict(df)["m"], null_equal=True)
        nan = df.with_columns(y0=pl.when(pl.col("x1") > 1.5).then(np.nan).otherwise(pl.col("y0")))
        null = nan.with_columns(pl.col("y0").fill_nan(None))
        assert null["y0"].null_count() > 50
        got = po.ModelBank([spec]).fit_predict(nan)["m"]
        assert got.equals(po.ModelBank([spec]).fit_predict(null)["m"], null_equal=True)

    def test_a_zero_weight_row_with_a_bad_label_is_refused_too(self):
        """The check reads the label, not the weight: a row of weight 0
        learns nothing, yet a label of 0.5 on it refuses the chunk, and the
        message says to null it. Pinned as the decision it is."""
        df = _binary_df(n=200).with_columns(
            w=pl.when(pl.int_range(pl.len()) == 17).then(0.0).otherwise(1.0),
            y0=pl.when(pl.int_range(pl.len()) == 17).then(0.5).otherwise(pl.col("y0")),
        )
        bank = po.ModelBank([_spec(strict_binary=True, weight="w")])
        before = bank.save_bytes()
        with pytest.raises(ValueError, match=r"has the value 0.5 at row 17.*null the rows"):
            bank.fit_predict(df)
        assert bank.save_bytes() == before


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no-intercept"])
@pytest.mark.parametrize("half_life", [float("inf"), 200.0], ids=["no-decay", "half_life"])
@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_no_smoothing_no_penalty_and_a_zero_weight_first_row_stay_finite(
    loss, half_life, fit_intercept
):
    """``beta = l1 = l2 = 0`` leaves a coordinate's rate ``sqrt(n_i) /
    alpha``, 0 until the coordinate first sees a gradient (hard rule 9): a
    zero-weight first row, a feature that is 0 for its first 50 rows and a
    hundred zero weights later must leave every prediction and coefficient
    finite, and the fit ``ftrl_ref``'s, written from the docstring (task
    278, batch 18). Measured 2026-10-26: at most 2.7e-15 from it."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal((n, 2))
    x[:50, 1] = 0.0
    w = np.ones(n)
    w[0] = 0.0
    w[rng.choice(np.arange(1, n), 100, replace=False)] = 0.0
    eta = 0.5 + 1.5 * x[:, 0] - x[:, 1]
    if loss == "logistic":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = eta + 0.3 * rng.standard_normal(n)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y, "w": w})
    kw = dict(half_life=half_life, beta=0.0, l1=0.0, l2=0.0, loss=loss, min_weight=0.0)
    spec = _spec(weight="w", fit_intercept=fit_intercept, coef_every=1, **kw)
    out = po.ModelBank([spec]).fit_predict(df)
    pred = _pred(out)
    coef = np.array(out["m"].struct.field("coef").to_list(), float)
    assert np.isfinite(pred).all() and np.isfinite(coef).all()
    assert (coef[:, -1] != 0.0).sum() > n - 100, "the late feature never learned"
    ref = ftrl_ref(x, y[:, None], np.ones(n), w, fit_intercept=fit_intercept, **kw)["pred"][:, 0]
    assert np.max(np.abs(pred - ref) / (1.0 + np.abs(ref))) < 1e-12


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="alpha"):
        _spec(alpha=0.0)
    with pytest.raises(ValueError, match="l1"):
        _spec(l1=-1.0)


class TestSquaredLoss:
    """E18: FTRL with a squared loss — sparse linear regression, no solves."""

    def _spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0", "x1", "x2"],
            loss="squared",
            alpha=0.5,
            l1=0.0,
            l2=0.01,
            half_life=float("inf"),
            min_weight=10.0,
        )
        d.update(kw)
        return po.spec.ftrl("m", **d)

    def _data(self, n=20000, seed=0, noise=0.1):
        rng = np.random.default_rng(seed)
        x0, x1, x2 = (rng.standard_normal(n) for _ in range(3))
        # x2 is pure noise
        y = 1.5 * x0 - 0.5 * x1 + noise * rng.standard_normal(n)
        return pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y0": y})

    def test_recovers_the_coefficients(self):
        df = self._data()
        out = po.ModelBank([self._spec(coef_every=1)]).fit_predict(df)
        c = np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)
        assert c[1] == pytest.approx(1.5, abs=0.1)
        assert c[2] == pytest.approx(-0.5, abs=0.1)

    def test_l1_zeroes_the_noise_feature(self):
        df = self._data()
        out = po.ModelBank([self._spec(l1=0.5, coef_every=1)]).fit_predict(df)
        c = np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)
        assert abs(c[3]) < 0.05, f"noise feature survived L1: {c[3]}"
        assert abs(c[1]) > 1.0, "the signal feature should survive"

    def test_predictions_are_not_squashed_into_a_probability(self):
        # The logistic link would cap these at 1.
        rng = np.random.default_rng(5)
        n = 5000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "x2": np.zeros(n), "y0": 20.0 * x})
        out = po.ModelBank([self._spec(min_weight=5.0)]).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert np.nanmax(p) > 5.0

    def test_logistic_is_still_the_default(self):
        df = self._data(n=2000)
        df = df.with_columns(y0=(pl.col("y0") > 0).cast(pl.Float64))
        out = po.ModelBank(
            [
                po.spec.ftrl(
                    "m",
                    targets=["y0"],
                    features=["x0", "x1", "x2"],
                    half_life=float("inf"),
                    min_weight=10.0,
                )
            ]
        ).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        finite = p[np.isfinite(p)]
        assert ((finite >= 0) & (finite <= 1)).all()

    def test_out_of_sample_on_noise(self):
        rng = np.random.default_rng(6)
        n = 5000
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        out = po.ModelBank([self._spec(half_life=2000.0)]).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p)
        ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
        assert abs(ic) < 0.06

    def test_chunk_invariance(self):
        df = self._data(n=400, seed=7)
        spec = self._spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_strict_binary_is_rejected(self):
        with pytest.raises(ValueError, match="strict_binary"):
            self._spec(strict_binary=True)

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(
            ValueError, match='loss must be one of "logistic", "squared"; got "hinge"'
        ):
            self._spec(loss="hinge")


@pytest.mark.parametrize("loss", ["logistic", "squared"])
def test_a_feature_scale_is_the_parameters_scaled_without_an_intercept(loss):
    """Task 239: the docstring's units paragraph says every feature scaled by
    ``c`` fits the same model at ``alpha / c``, ``beta * c``, ``l1 * c`` and
    ``l2 * c**2``. That holds through the origin, where every coordinate
    scales; with an intercept, whose 1 does not, it does not, and the
    docstring says so."""
    df = _binary_df(n=3000)
    c = 8.0
    kw = dict(fit_intercept=False, half_life=200.0, loss=loss, min_weight=0.0)
    base = dict(alpha=0.1, beta=1.0, l1=0.05, l2=0.5)
    scaled = dict(alpha=0.1 / c, beta=1.0 * c, l1=0.05 * c, l2=0.5 * c**2)
    wide = df.with_columns(pl.col("x0") * c, pl.col("x1") * c)
    pa = _pred(po.ModelBank([_spec(**kw, **base)]).fit_predict(df))
    pb = _pred(po.ModelBank([_spec(**kw, **scaled)]).fit_predict(wide))
    ok = np.isfinite(pa)
    assert ok.sum() > 2900 and np.array_equal(ok, np.isfinite(pb))
    assert np.max(np.abs(pa[ok] - pb[ok])) < 1e-12
    kw["fit_intercept"] = True
    qa = _pred(po.ModelBank([_spec(**kw, **base)]).fit_predict(df))
    qb = _pred(po.ModelBank([_spec(**kw, **scaled)]).fit_predict(wide))
    both = np.isfinite(qa) & np.isfinite(qb)
    assert np.max(np.abs(qa[both] - qb[both])) > 1e-3


# The penalties each move the fit one way (task 278, batch 18). Three seeds
# in the essentials, seventeen more in the extended tier; each assertion
# holds on every seed.
SEEDS = [0, 1, 2] + [
    pytest.param(seed, marks=pytest.mark.extended(reason="the monotone grids' further seeds"))
    for seed in range(3, 20)
]


def _logistic(seed: int, n: int = 4000, noise_features: int = 0) -> tuple[pl.DataFrame, list]:
    """Two informative features and ``noise_features`` that are not."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2 + noise_features))
    eta = 0.25 + 1.5 * x[:, 0] - 0.5 * x[:, 1]
    frame = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    frame["y0"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return pl.DataFrame(frame), [f"x{i}" for i in range(x.shape[1])]


def _every_coef(df, features, **kw) -> np.ndarray:
    spec = _spec(features=features, half_life=float("inf"), min_weight=0.0, coef_every=1, **kw)
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    return np.array(out.struct.field("coef").to_list(), float)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_larger_l1_zeroes_more_coefficients(seed):
    """A coordinate whose accumulated evidence ``|z_i|`` is at most ``l1``
    is exactly 0, so the count of exact zeros never falls as ``l1`` rises:
    none at 0 past the first rows, and most of the three noise features'
    entries at 10. Counted over the slopes of rows 100 to 4,000. Measured
    2026-10-26 on 20 seeds: 0, 56-161, 606-1,556 and 4,113-10,026."""
    df, features = _logistic(seed, noise_features=3)
    zeros = []
    for l1 in (0.0, 0.1, 1.0, 10.0):
        coef = _every_coef(df, features, l1=l1)
        zeros.append(int((coef[100:, 1:] == 0.0).sum()))
    assert zeros[0] == 0, zeros
    assert (np.diff(zeros) > 0).all(), zeros


@pytest.mark.parametrize("seed", SEEDS)
def test_a_larger_l2_shrinks_every_coefficient(seed):
    """``l2`` adds to every coordinate's rate, so the fit is pulled harder
    to 0: each slope's mean magnitude over the last 2,000 rows falls as
    ``l2`` rises over 0, 1, 10, 100 and 1,000. Measured 2026-10-26 on 20
    seeds: the first slope from 1.27-1.49 to 0.46-0.51, every step down at
    least 0.17% of the magnitude before it (the smallest, from 0 to 1)."""
    df, features = _logistic(seed)
    magnitude = np.array(
        [
            np.abs(_every_coef(df, features, l2=l2)[-2000:, 1:]).mean(axis=0)
            for l2 in (0, 1, 10, 100, 1000)
        ]
    )
    assert (np.diff(magnitude, axis=0) < 0).all(), magnitude
