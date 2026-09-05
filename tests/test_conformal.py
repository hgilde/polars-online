"""E36: adaptive conformal intervals -- `lo_<slot>`, `hi_<slot>` and
`coverage_<slot>` from `conformal=<coverage>`.

The interval is `pred ± q` with `q` a tracked quantile of the conformity
score `|resid|`:

    err = 1{|resid| > q}
    q  <- max(0, q + rate·sigma·w·(err − α)),      α = 1 − coverage

read before the row, so the interval a row gets never saw that row (its
own score moves `q` afterwards). `sigma` is the slot's EW residual standard
deviation before the row, which makes `conformal_rate` unit-free; the
radius starts at the Gaussian one, `sigma · Φ⁻¹(1 − α/2)`, on the first row
that has both, and tracks from there. `coverage` is the EW fraction of
scored rows whose interval held the target.

Three things are established here:

1. **The oracle.** A longhand Python replay of the recursion over the bank's
   own `pred`, `resid` and `sigma` reproduces `lo`, `hi` and `coverage` bit
   for bit, for every regression model, with a grid, nulls in features and
   targets, zero and varying weights, an irregular clock and groups.
2. **The guarantee, at scale.** Telescoping the recursion gives, for scores
   in `[0, B]` and steps `η_t`,

       Σ η_t (err_t − α) ≤ B + max η        (and ≥ −(B + max η) unless
                                             the clamp at zero ever bit)

   for *every* sequence of residuals -- no distribution, no stationarity.
   That inequality is checked as a hard bound, and the empirical coverage
   sits at the target within 1% on 200k rows of Gaussian, fat-tailed,
   heteroskedastic and regime-shifting residuals, where `pred ± z·sigma`
   over-covers by several points on the fat-tailed and heteroskedastic
   ones.
3. **The edges.** Opt-in fields; validation by name; refused for the
   unsupervised models; nulls, zero weights and warmup; chunk invariance;
   save/load; predict mode; a drift reset restarts the radius; the sigma
   grid names one interval per slot.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

# --- the oracle ----------------------------------------------------------------

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.38357751867269e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p: float) -> float:
    """Acklam's inverse normal, the operations in the order
    `online_core::norm_ppf` performs them, so the warm start is bit-exact."""
    a, b, c, d = _A, _B, _C, _D

    def tail(q):
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p < 0.02425:
        return tail(math.sqrt(-2.0 * math.log(p)))
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return -tail(math.sqrt(-2.0 * math.log(1.0 - p)))


def replay(pred, resid, sigma, lam, w, accepted, *, coverage, rate=0.05):
    """`lo`, `hi`, `coverage` for one slot from the bank's own `pred`,
    `resid` and `sigma` (NaN for null), the per-row decay factor and weight,
    and which rows the stream accepted at all."""
    alpha = 1.0 - coverage
    n = len(pred)
    lo, hi, cov_out = (np.full(n, np.nan) for _ in range(3))
    q, cov, cov_w = math.nan, 0.0, 0.0
    for i in range(n):
        if not accepted[i]:
            continue
        p = pred[i]
        if math.isfinite(q):
            lo[i], hi[i] = p - q, p + q
        if cov_w > 0.0:
            cov_out[i] = cov
        r, s, lam_i, w_i = resid[i], sigma[i], lam[i], w[i]
        if not math.isfinite(r) or w_i <= 0.0:
            cov_w *= lam_i
            continue
        usable = math.isfinite(s) and s > 0.0
        if not math.isfinite(q):
            if usable:
                q = s * norm_ppf(1.0 - 0.5 * alpha)
            cov_w *= lam_i
            continue
        miss = abs(r) > q
        cw = lam_i * cov_w + w_i
        cov = (lam_i * cov_w * cov + w_i * (0.0 if miss else 1.0)) / cw
        cov_w = cw
        eta = rate * s if usable else 0.0
        q = max(q + eta * w_i * ((1.0 if miss else 0.0) - alpha), 0.0)
    return lo, hi, cov_out


def row_plan(df, *, clock, max_dclock, halflife, weight, features):
    """The stream's per-row decay factor and weight, and which rows it
    accepts: a null or non-finite feature or weight skips the row, and its
    clock delta is folded into the next accepted row's."""
    n = df.height
    t = df[clock].to_numpy().astype(float) if clock else np.arange(n, dtype=float)
    wcol = df[weight].to_list() if weight else [1.0] * n
    feats = df.select(features).to_numpy().astype(float) if features else np.zeros((n, 0))
    lam = np.full(n, np.nan)
    w = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    pending = 0.0
    for i in range(n):
        d = 0.0 if i == 0 else min(t[i] - t[i - 1], max_dclock)
        pending += d
        wi = wcol[i]
        ok = all(math.isfinite(v) for v in feats[i]) and wi is not None and math.isfinite(wi)
        if not ok:
            continue
        accepted[i] = True
        lam[i] = math.exp2(-(pending / halflife))
        w[i] = float(wi)
        pending = 0.0
    return lam, w, accepted


def field(out, name, col="m"):
    return out[col].struct.field(name).to_numpy().astype(float)


# --- data ----------------------------------------------------------------------


def messy(n=3000, seed=0, *, groups=1):
    """A stream with every input path in it: an irregular clock with gaps
    past `max_dclock`, nulls in a feature and in the target, zero, varying
    and null weights, two targets, and a shock."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    y0 = 1.5 * x0 - 0.5 * x1 + 0.3 * rng.standard_normal(n)
    y1 = -x0 + 0.2 * x1 + 0.8 * rng.standard_normal(n)
    y0[n // 2] += 25.0
    dt = rng.choice([1.0, 2.0, 3.0, 5.0, 40.0], size=n, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    dt[0] = 0.0
    t = np.cumsum(dt)
    w = rng.choice([0.0, 0.5, 1.0, 2.0], size=n, p=[0.05, 0.3, 0.5, 0.15]).tolist()
    x1 = x1.tolist()
    y0 = y0.tolist()
    for i in range(n):
        if i % 53 == 7:
            x1[i] = None
        if i % 41 == 3:
            y0[i] = None
        if i % 97 == 5:
            w[i] = None
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": pl.Series(x1, dtype=pl.Float64),
            "y0": pl.Series(y0, dtype=pl.Float64),
            "y1": y1,
            "w": pl.Series(w, dtype=pl.Float64),
            "g": [f"g{i % groups}" for i in range(n)],
        }
    )


MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("ewridge", {"ridge": [1e-6, 1.0], "feature_sets": {"a": ["x0"], "b": ["x0", "x1"]}}),
    ("rls", {"ridge": 1.0}),
    ("kalman", {"coef_halflife": 100.0}),
    ("lasso", {"lasso_path": [0.1, 0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("sgd", {"learning_rate": 0.02}),
    ("pa", {}),
    ("ftrl", {"loss": "squared", "alpha": 0.5}),
    ("holt", {}),
]
IDS = [m if not extra.get("ridge") else "ewridge-grid" for m, extra in MODELS]


def build(model, extra, **kw):
    d = dict(
        targets=["y0", "y1"],
        clock="t",
        max_dclock=10.0,
        halflife=200.0,
        weight="w",
        min_periods=5.0,
        emit_sigma=True,
        conformal=0.9,
    )
    if model != "holt":
        d["features"] = ["x0", "x1"]
    d.update(extra)
    d.update(kw)
    return getattr(po.spec, model)("m", **d)


def slots(spec, kind="lo"):
    idx = po.spec.output_index(spec)
    return idx.filter(pl.col("kind") == kind)


# --- 1. the oracle ---------------------------------------------------------------


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestOracle:
    def test_lo_hi_and_coverage_are_the_replayed_recursion(self, model, extra):
        df = messy(n=3000, seed=1)
        spec = build(model, extra)
        out = po.ModelBank([spec]).fit_predict(df)
        features = spec["features"]
        lam, w, accepted = row_plan(
            df, clock="t", max_dclock=10.0, halflife=200.0, weight="w", features=features
        )
        idx = po.spec.output_index(spec)
        lo_rows = idx.filter(pl.col("kind") == "lo").to_dicts()
        assert lo_rows, "no interval fields"
        for row in lo_rows:
            slot = row["field"][len("lo_") :]
            pred, resid, sigma = (field(out, f"{k}_{slot}") for k in ("pred", "resid", "sigma"))
            lo, hi, cov = replay(pred, resid, sigma, lam, w, accepted, coverage=0.9)
            np.testing.assert_array_equal(field(out, f"lo_{slot}"), lo, err_msg=slot)
            np.testing.assert_array_equal(field(out, f"hi_{slot}"), hi, err_msg=slot)
            np.testing.assert_array_equal(field(out, f"coverage_{slot}"), cov, err_msg=slot)
            assert np.isfinite(lo).sum() > 2000, f"{slot}: too few intervals to mean anything"

    def test_the_rate_and_level_are_honoured(self, model, extra):
        df = messy(n=1500, seed=2)
        spec = build(model, extra, conformal=0.75, conformal_rate=0.2)
        out = po.ModelBank([spec]).fit_predict(df)
        lam, w, accepted = row_plan(
            df, clock="t", max_dclock=10.0, halflife=200.0, weight="w", features=spec["features"]
        )
        row = slots(spec).to_dicts()[-1]
        slot = row["field"][len("lo_") :]
        pred, resid, sigma = (field(out, f"{k}_{slot}") for k in ("pred", "resid", "sigma"))
        lo, hi, cov = replay(pred, resid, sigma, lam, w, accepted, coverage=0.75, rate=0.2)
        np.testing.assert_array_equal(field(out, f"lo_{slot}"), lo)
        np.testing.assert_array_equal(field(out, f"hi_{slot}"), hi)
        np.testing.assert_array_equal(field(out, f"coverage_{slot}"), cov)


class TestOracleOnTheStreamPlumbing:
    """The same replay across the plumbing the row plan can vary."""

    def test_groups_are_independent_streams(self):
        df = messy(n=3000, seed=3, groups=3)
        spec = build("ewridge", {"max_rows_between_solves": 1}, group="g")
        out = po.ModelBank([spec]).fit_predict(df)
        for g in ("g0", "g1", "g2"):
            mask = (df["g"] == g).to_numpy()
            sub = df.filter(pl.col("g") == g)
            lam, w, accepted = row_plan(
                sub, clock="t", max_dclock=10.0, halflife=200.0, weight="w", features=["x0", "x1"]
            )
            for tgt in ("y0", "y1"):
                pred, resid, sigma = (
                    field(out, f"{k}_{tgt}")[mask] for k in ("pred", "resid", "sigma")
                )
                lo, hi, cov = replay(pred, resid, sigma, lam, w, accepted, coverage=0.9)
                np.testing.assert_array_equal(field(out, f"lo_{tgt}")[mask], lo)
                np.testing.assert_array_equal(field(out, f"hi_{tgt}")[mask], hi)
                np.testing.assert_array_equal(field(out, f"coverage_{tgt}")[mask], cov)

    def test_row_count_clock_and_a_halflife_grid(self):
        df = messy(n=2000, seed=4).drop("t", "w")
        spec = po.spec.rls(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            ridge=1.0,
            halflife=[50.0, 400.0],
            min_periods=3.0,
            emit_sigma=True,
            conformal=0.95,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for h in (50.0, 400.0):
            lam, w, accepted = row_plan(
                df, clock=None, max_dclock=math.inf, halflife=h, weight=None, features=["x0", "x1"]
            )
            slot = f"y0@h{h:g}"
            pred, resid, sigma = (field(out, f"{k}_{slot}") for k in ("pred", "resid", "sigma"))
            lo, hi, cov = replay(pred, resid, sigma, lam, w, accepted, coverage=0.95)
            np.testing.assert_array_equal(field(out, f"lo_{slot}"), lo)
            np.testing.assert_array_equal(field(out, f"hi_{slot}"), hi)
            np.testing.assert_array_equal(field(out, f"coverage_{slot}"), cov)
        # Two halflives, two radii: the slow one has a wider, steadier interval.
        wide = field(out, "hi_y0@h400") - field(out, "lo_y0@h400")
        narrow = field(out, "hi_y0@h50") - field(out, "lo_y0@h50")
        assert not np.allclose(np.nan_to_num(wide), np.nan_to_num(narrow))

    def test_the_expression_matches_the_bank(self):
        df = messy(n=800, seed=5).drop("w")
        bank = po.ModelBank(
            [
                po.spec.ewridge(
                    "m",
                    targets=["y0"],
                    features=["x0", "x1"],
                    clock="t",
                    max_dclock=10.0,
                    halflife=100.0,
                    min_periods=3.0,
                    conformal=0.9,
                    max_rows_between_solves=1,
                )
            ]
        ).fit_predict(df)
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("y0")
                .online.ewridge(
                    ["x0", "x1"],
                    clock="t",
                    max_dclock=10.0,
                    halflife=100.0,
                    min_periods=3.0,
                    conformal=0.9,
                    max_rows_between_solves=1,
                )
                .alias("m")
            )
        for k in ("lo_y0", "hi_y0", "coverage_y0"):
            np.testing.assert_array_equal(field(expr, k), field(bank, k))


# --- 2. the guarantee, at scale ----------------------------------------------------


def _stream(kind, n=200_000, seed=0):
    """Four residual regimes; the model sees `x0`, `x1` and a fixed relation
    except where the regime changes it."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    if kind == "gaussian":
        eps = 0.5 * rng.standard_normal(n)
        beta = np.full(n, 2.0)
    elif kind == "fat_tailed":
        eps = 0.3 * rng.standard_t(2.5, size=n)
        beta = np.full(n, 2.0)
    elif kind == "heteroskedastic":
        # A lognormal scale mixture: a single sigma over-covers the calm
        # rows and misses the wild ones.
        eps = np.exp(x1) * rng.standard_normal(n)
        beta = np.full(n, 2.0)
    elif kind == "shifting":
        # The slope flips at n/2 and the noise triples at 3n/4: the model
        # relearns with a lag, and so does sigma.
        beta = np.where(np.arange(n) < n // 2, 2.0, -2.0)
        scale = np.where(np.arange(n) < 3 * n // 4, 0.5, 1.5)
        eps = scale * rng.standard_normal(n)
    else:  # pragma: no cover
        raise ValueError(kind)
    y = beta * x0 + 0.5 * x1 + eps
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y})


def _fit_large(df, coverage=0.9, rate=0.05, halflife=2000.0):
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        halflife=halflife,
        min_periods=20.0,
        emit_sigma=True,
        conformal=coverage,
        conformal_rate=rate,
    )
    return po.ModelBank([spec]).fit_predict(df)


class TestTheGuaranteeAtScale:
    @pytest.mark.parametrize("kind", ["gaussian", "fat_tailed", "heteroskedastic", "shifting"])
    def test_the_telescoped_bound_holds_exactly(self, kind):
        """`Σ η_t (err_t − α) ≤ B + max η`, and the same from below unless the
        clamp at zero ever bit: a theorem about the recursion, so it is asserted
        as a hard inequality on the bank's own outputs."""
        df = _stream(kind)
        out = _fit_large(df)
        y = df["y"].to_numpy()
        lo, hi, sigma, resid = (field(out, f"{k}_y") for k in ("lo", "hi", "sigma", "resid"))
        m = np.isfinite(lo) & np.isfinite(resid)
        q = (hi[m] - lo[m]) / 2
        assert (q > 0).all(), "the clamp at zero bit; the lower bound needs restating"
        err = (y[m] < lo[m]) | (y[m] > hi[m])
        eta = 0.05 * sigma[m]
        b = np.abs(resid[m]).max()
        total = float(np.sum(eta * (err - 0.1)))
        bound = b + eta.max()
        assert abs(total) <= bound, f"{kind}: {total} vs {bound}"

    @pytest.mark.parametrize("kind", ["gaussian", "fat_tailed", "heteroskedastic", "shifting"])
    def test_long_run_coverage_is_the_target(self, kind):
        df = _stream(kind)
        out = _fit_large(df)
        y = df["y"].to_numpy()
        lo, hi = field(out, "lo_y"), field(out, "hi_y")
        m = np.isfinite(lo)
        assert m.sum() > 199_000
        covered = np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))
        assert abs(covered - 0.9) < 0.01, f"{kind}: empirical coverage {covered}"
        # The EW `coverage` field says the same thing about the recent past.
        cov = field(out, "coverage_y")
        assert abs(np.nanmean(cov[-20_000:]) - 0.9) < 0.02, f"{kind}: {np.nanmean(cov[-20_000:])}"

    @pytest.mark.parametrize("kind", ["fat_tailed", "heteroskedastic"])
    def test_beats_the_gaussian_interval_where_the_residuals_are_not_gaussian(self, kind):
        """`pred ± Φ⁻¹(0.95)·sigma` is the interval `emit_sigma` implies. On
        fat tails sigma is inflated by the outliers and the interval
        over-covers the bulk; under a scale mixture likewise. The tracked
        radius lands on the target on both. (Through a regime shift with
        Gaussian noise both are right in the long run -- the shifting stream
        is in the coverage test, not here.)"""
        df = _stream(kind)
        out = _fit_large(df)
        y = df["y"].to_numpy()
        pred, sigma, lo, hi = (field(out, f"{k}_y") for k in ("pred", "sigma", "lo", "hi"))
        m = np.isfinite(lo) & np.isfinite(sigma)
        z = norm_ppf(0.95)
        gauss = np.mean(np.abs(y[m] - pred[m]) <= z * sigma[m])
        conf = np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))
        assert abs(conf - 0.9) < abs(gauss - 0.9), f"{kind}: conformal {conf}, gaussian {gauss}"
        assert abs(gauss - 0.9) > 0.02, f"{kind}: the Gaussian interval was fine here ({gauss})"

    def test_the_radius_tracks_a_scale_shift(self):
        df = _stream("shifting")
        out = _fit_large(df)
        lo, hi = field(out, "lo_y"), field(out, "hi_y")
        q = (hi - lo) / 2
        n = df.height
        before = np.nanmedian(q[n // 2 : 3 * n // 4])
        after = np.nanmedian(q[-n // 8 :])
        # noise 0.5 -> 1.5: the 90% radius goes from ~0.82 to ~2.47.
        assert before == pytest.approx(0.5 * norm_ppf(0.95), rel=0.1), before
        assert after == pytest.approx(1.5 * norm_ppf(0.95), rel=0.1), after

    def test_every_level_is_met(self):
        df = _stream("fat_tailed", n=100_000)
        y = df["y"].to_numpy()
        for level in (0.5, 0.8, 0.99):
            out = _fit_large(df, coverage=level)
            lo, hi = field(out, "lo_y"), field(out, "hi_y")
            m = np.isfinite(lo)
            covered = np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))
            assert abs(covered - level) < 0.012, f"level {level}: {covered}"


# --- 3. the edges ----------------------------------------------------------------


def _small(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 2 * x + 0.5 * rng.standard_normal(n)
    return pl.DataFrame({"x0": x, "y0": y, "t": np.arange(float(n))})


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0"],
        halflife=100.0,
        min_periods=10.0,
        max_rows_between_solves=1,
        conformal=0.9,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


class TestFields:
    def test_absent_by_default(self):
        fields = po.spec.output_fields(_spec(conformal=None))
        assert not any(f.startswith(("lo_", "hi_", "coverage_")) for f in fields)

    def test_present_when_requested_and_placed_after_the_prediction(self):
        assert po.spec.output_fields(_spec()) == [
            "pred_y0",
            "resid_y0",
            "lo_y0",
            "hi_y0",
            "coverage_y0",
            "n_eff",
            "coef",
        ]

    def test_one_interval_per_slot_in_a_grid(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0", "x1"],
            ridge=[1e-6, 0.5],
            feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
            halflife=[100.0, 500.0],
            conformal=0.9,
        )
        idx = po.spec.output_index(spec)
        for kind in ("lo", "hi", "coverage"):
            rows = idx.filter(pl.col("kind") == kind)
            assert rows.height == 2 * 2 * 2 * 2
            assert set(rows["target"]) == {"y0", "y1"}
            assert set(rows["ridge"]) == {1e-6, 0.5}
            assert set(rows["feature_set"]) == {"a", "b"}
            assert set(rows["halflife"]) == {100.0, 500.0}
            assert set(rows["dtype"]) == {"f64"}
        assert "lo_y1__b_r0.5@h500" in idx["field"].to_list()

    def test_the_names_are_not_predictions(self):
        """`eval.unpack` finds predictions by the `pred_` prefix, so the
        bounds must not wear it."""
        out = po.ModelBank([_spec()]).fit_predict(_small())
        long = po.eval.unpack(out, "m")
        assert set(long["slot"]) == {"pred_y0"}

    def test_the_kwargs_class_mirrors_the_builder(self):
        from polars_online._kwargs import CommonKwargs

        assert {"conformal", "conformal_rate"} <= set(CommonKwargs.__annotations__)


class TestValidation:
    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"conformal": 0.0}, "conformal must be a coverage level strictly between 0 and 1"),
            ({"conformal": 1.0}, "conformal must be a coverage level strictly between 0 and 1"),
            ({"conformal": -0.5}, "conformal must be a coverage level strictly between 0 and 1"),
            ({"conformal": 1.5}, "conformal must be a coverage level strictly between 0 and 1"),
            ({"conformal": float("nan")}, "conformal must not be NaN"),
            ({"conformal": float("inf")}, "conformal must be finite"),
            ({"conformal": 0.9, "conformal_rate": 0.0}, "conformal_rate must be finite and > 0"),
            ({"conformal": 0.9, "conformal_rate": -1.0}, "conformal_rate must be finite and > 0"),
            ({"conformal": 0.9, "conformal_rate": float("inf")}, "conformal_rate must be finite"),
            ({"conformal": None, "conformal_rate": 0.1}, "conformal_rate needs conformal"),
        ],
    )
    def test_bad_values_name_the_parameter(self, kw, msg):
        with pytest.raises(ValueError, match=msg) as exc:
            _spec(**kw)
        assert str(exc.value).startswith('spec "m"')

    @pytest.mark.parametrize("kw", [{"conformal": "0.9"}, {"conformal_rate": [0.1]}])
    def test_bad_types_name_the_parameter(self, kw):
        with pytest.raises(TypeError, match=next(iter(kw))):
            _spec(**kw)

    def test_a_dict_edited_by_hand_is_checked_at_use(self):
        spec = _spec()
        spec["conformal"] = 2.0
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            po.ModelBank([spec])

    @pytest.mark.parametrize(
        ("builder", "kw"),
        [
            (po.spec.ew_cov, {"features": ["x0"], "halflife": 10.0}),
            (po.spec.kmeans, {"features": ["x0"], "k": 2, "halflife": 10.0}),
            (po.spec.micro, {"features": ["x0"], "eps": 0.3, "halflife": 10.0}),
        ],
        ids=["ew_cov", "kmeans", "micro"],
    )
    def test_refused_for_the_unsupervised_models(self, builder, kw):
        with pytest.raises(ValueError, match="conformal does not apply to"):
            builder("m", conformal=0.9, **kw)


class TestRowSemantics:
    def test_null_until_the_radius_exists(self):
        """Prediction first (`min_periods`), then a residual, then a sigma,
        then a radius: the interval appears two rows after the first
        residual, and `coverage` one row after that."""
        out = po.ModelBank([_spec(min_periods=20.0)]).fit_predict(_small())
        pred, lo, cov = (
            out["m"].struct.field(k).to_list() for k in ("pred_y0", "lo_y0", "coverage_y0")
        )
        first_pred = next(i for i, v in enumerate(pred) if v is not None)
        first_lo = next(i for i, v in enumerate(lo) if v is not None)
        first_cov = next(i for i, v in enumerate(cov) if v is not None)
        assert first_lo == first_pred + 2
        assert first_cov == first_lo + 1
        assert all(v is None for v in lo[:first_lo])

    def test_the_interval_is_out_of_sample(self):
        """A shock must not widen the interval reported for its own row."""
        df = _small(n=600)
        y = df["y0"].to_numpy().copy()
        y[400] += 30.0
        df = df.with_columns(pl.Series("y0", y))
        out = po.ModelBank([_spec()]).fit_predict(df)
        q = (field(out, "hi_y0") - field(out, "lo_y0")) / 2
        # Row 400's radius moved from row 399's by one ordinary step at most.
        assert abs(q[400] - q[399]) <= 0.05 * 0.6 * 0.9 + 1e-12
        assert q[401] > q[400], "the miss should widen the next interval"
        assert lo_hi_contains(out, y, 399)
        assert not lo_hi_contains(out, y, 400), "a 30-sigma shock is outside the interval"

    def test_a_null_target_row_learns_nothing(self):
        df = _small()
        y = df["y0"].to_list()
        y[200] = None
        df = df.with_columns(pl.Series("y0", y, dtype=pl.Float64))
        out = po.ModelBank([_spec()]).fit_predict(df)
        q = (field(out, "hi_y0") - field(out, "lo_y0")) / 2
        cov = field(out, "coverage_y0")
        assert np.isfinite(q[200]), "the prediction and its interval are still emitted"
        # `(hi - lo) / 2` recovers the radius to a rounding of `pred ± q`;
        # a step would be ~0.05 * sigma * 0.1 = 2e-3 at the smallest.
        assert abs(q[201] - q[200]) < 1e-12, "the radius did not move"
        assert cov[201] == cov[200], "the coverage did not move"

    def test_a_zero_weight_row_learns_nothing(self):
        df = _small().with_columns(pl.Series("w", [0.0 if i == 200 else 1.0 for i in range(400)]))
        out = po.ModelBank([_spec(weight="w")]).fit_predict(df)
        q = (field(out, "hi_y0") - field(out, "lo_y0")) / 2
        cov = field(out, "coverage_y0")
        assert abs(q[201] - q[200]) < 1e-12 and cov[201] == cov[200]
        # And a weight of 2 takes twice the step a weight of 1 does.
        one = po.ModelBank([_spec()]).fit_predict(df.drop("w"))
        two = po.ModelBank([_spec(weight="w")]).fit_predict(
            df.with_columns(pl.Series("w", [2.0 if i == 200 else 1.0 for i in range(400)]))
        )
        q1 = (field(one, "hi_y0") - field(one, "lo_y0")) / 2
        q2 = (field(two, "hi_y0") - field(two, "lo_y0")) / 2
        assert q2[200] == q1[200]
        assert q2[201] - q2[200] == pytest.approx(2 * (q1[201] - q1[200]), abs=1e-12)
        assert abs(q1[201] - q1[200]) > 1e-4, "row 200 should have taken a step"

    def test_a_null_feature_row_is_skipped_entirely(self):
        df = _small()
        x = df["x0"].to_list()
        x[200] = None
        df = df.with_columns(pl.Series("x0", x, dtype=pl.Float64))
        # On a clock, so dropping the row is the same stream as skipping it.
        out = po.ModelBank([_spec(clock="t", max_dclock=10.0)]).fit_predict(df)
        row = out["m"][200]
        assert all(v is None for k, v in row.items() if k != "n_eff")
        ref = po.ModelBank([_spec(clock="t", max_dclock=10.0)]).fit_predict(
            _small().filter(pl.arange(0, 400) != 200)
        )
        for k in ("lo_y0", "hi_y0", "coverage_y0"):
            got = np.delete(field(out, k), 200)
            np.testing.assert_array_equal(got, field(ref, k))

    def test_a_perfect_fit_gives_a_vanishing_interval(self):
        df = _small().with_columns((2.0 * pl.col("x0")).alias("y0"))
        out = po.ModelBank([_spec()]).fit_predict(df)
        q = (field(out, "hi_y0") - field(out, "lo_y0")) / 2
        # The default ridge leaves residuals of ~1e-6; the radius follows.
        assert np.isfinite(q[-1]) and 0.0 <= q[-1] < 1e-4
        cov = field(out, "coverage_y0")
        assert np.isfinite(cov[-1])

    def test_the_coverage_is_an_ew_mean_on_the_models_clock(self):
        df = _small(n=2000)
        out = po.ModelBank([_spec(halflife=20.0)]).fit_predict(df)
        cov = field(out, "coverage_y0")
        # With a 20-row halflife the field moves quickly and sits near the
        # target on average, never exactly on it.
        tail = cov[-500:]
        assert abs(tail.mean() - 0.9) < 0.05
        assert tail.std() > 0.01


def lo_hi_contains(out, y, i):
    return field(out, "lo_y0")[i] <= y[i] <= field(out, "hi_y0")[i]


class TestStreamContract:
    def test_chunk_invariance(self):
        df = messy(n=1200, seed=6)
        spec = build("ewridge", {"max_rows_between_solves": 1})
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, 7)) for i in range(0, df.height, 7)]
        chunked = pl.concat(parts)
        for k in ("lo_y0", "hi_y0", "coverage_y0", "lo_y1", "hi_y1", "coverage_y1"):
            np.testing.assert_array_equal(field(chunked, k), field(whole, k), err_msg=k)

    def test_survives_save_load(self, tmp_path):
        df = messy(n=1000, seed=7)
        spec = build("kalman", {"coef_halflife": 100.0})
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        head = bank.fit_predict(df.slice(0, 500))
        bank.save(tmp_path / "state.msgpack")
        loaded = po.ModelBank.load(tmp_path / "state.msgpack", [spec])
        tail = loaded.fit_predict(df.slice(500))
        got = pl.concat([head, tail])
        for k in ("lo_y0", "hi_y0", "coverage_y0"):
            np.testing.assert_array_equal(field(got, k), field(whole, k), err_msg=k)

    def test_predict_reads_the_radius_and_moves_nothing(self):
        df = messy(n=600, seed=8)
        spec = build("ewridge", {"max_rows_between_solves": 1})
        bank = po.ModelBank([spec])
        bank.fit_predict(df.slice(0, 400))
        snap = bank.save_bytes()
        scored = bank.predict(df.slice(400))
        assert bank.save_bytes() == snap
        q = (field(scored, "hi_y0") - field(scored, "lo_y0")) / 2
        cov = field(scored, "coverage_y0")
        finite = np.isfinite(q)
        assert finite.sum() > 150
        assert np.ptp(q[finite]) < 1e-12, "the radius must not move while scoring"
        assert np.unique(cov[np.isfinite(cov)]).size == 1
        # The first scored row is what the next learned row would have got.
        learned = bank.fit_predict(df.slice(400, 1))
        for k in ("lo_y0", "hi_y0", "coverage_y0"):
            np.testing.assert_array_equal(field(scored, k)[:1], field(learned, k))

    def test_a_drift_reset_restarts_the_radius(self):
        rng = np.random.default_rng(9)
        n = 1200
        x = rng.standard_normal(n)
        y = np.where(np.arange(n) < 600, 2.0, -2.0) * x + 0.3 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": y})
        out = po.ModelBank(
            [_spec(emit_drift=True, drift_action="reset", drift_threshold=5.0)]
        ).fit_predict(df)
        drift = out["m"].struct.field("drift_y0").to_numpy()
        fired = np.flatnonzero(drift)
        assert fired.size and 600 <= fired[0] < 700, fired[:3]
        d = fired[0]
        lo = field(out, "lo_y0")
        pred = field(out, "pred_y0")
        assert np.isfinite(lo[d]), "the interval on the firing row was read before the reset"
        next_pred = d + 1 + int(np.argmax(np.isfinite(pred[d + 1 :])))
        next_lo = d + 1 + int(np.argmax(np.isfinite(lo[d + 1 :])))
        assert next_pred > d + 5, "the reset model waits for min_periods again"
        assert next_lo == next_pred + 2, "and the radius restarts with it"

    def test_the_runner_agrees_with_the_bank(self, tmp_path):
        df = messy(n=800, seed=10, groups=2)
        spec = build("huber", {"max_rows_between_solves": 1}, group="g")
        want = po.ModelBank([spec]).fit_predict(df)
        src = tmp_path / "in.parquet"
        dst = tmp_path / "out.parquet"
        df.write_parquet(src)
        po.run(input=str(src), output=str(dst), specs=[spec])
        got = pl.read_parquet(dst)
        for k in ("lo_y0", "hi_y0", "coverage_y0"):
            np.testing.assert_array_equal(field(got, k), field(want, k), err_msg=k)
