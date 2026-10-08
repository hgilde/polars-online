"""Readiness beyond ``ewridge`` (docs/PLAN.md task 116;
docs/WARMUP-AND-CONVERGENCE.md §5.8).

The noise statistic ``error_inflation = sqrt(1 + estimation variance / noise)``
on ``kalman`` (exact, per row: ``z' P⁻ z / R``), ``rls`` (the gate's bound
``k / n_kish``, the row's leverage in sum form) and ``lasso`` (the active count
over ``n_kish``, gate only); ``support_coef`` on the robust models; the
coefficients' standard errors ``se_coef``; and the support warning held until
the stream has settled. Every gate here is off unless set, and every new row
field is opt-in: no default moves (D8). The library second opinions -- filterpy,
padasip, statsmodels -- are in ``tests/test_second_opinion.py``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _polars_online as native
from polars_online import _spec

LIMIT_REASON = "above_max_error_inflation"


def resolved_limit(spec: dict) -> object:
    """The ``max_error_inflation`` a spec resolves to in the bank, as the API
    snapshot reads it (``_polars_online.resolved_defaults``)."""
    import json

    return json.loads(native.resolved_defaults(_spec._json(spec)))["stream"]["max_error_inflation"]


def frame(n: int = 400, k: int = 2, seed: int = 0, noise: float = 0.5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    y = 1.0 + x @ np.arange(1, k + 1, dtype=float) + noise * rng.standard_normal(n)
    cols: dict[str, Any] = {f"x{j}": x[:, j] for j in range(k)}
    cols["y"] = y
    cols["w"] = rng.uniform(0.5, 1.5, n)
    return pl.DataFrame(cols)


def build(kind: str, **kw: Any) -> dict:
    base: dict[str, Any] = dict(targets=["y"], features=["x0", "x1"], half_life=20.0)
    base.update(kw)
    if kind == "kalman":
        base.setdefault("coef_half_life", 30.0)
    if kind == "lasso":
        base.setdefault("lasso_path", [0.05])
    return getattr(po.spec, kind)("m", **base)


def unnest(out: pl.DataFrame) -> pl.DataFrame:
    return out["m"].struct.unnest()


def reasons(out: pl.DataFrame) -> list[str | None]:
    return unnest(out)["withheld_reason"].cast(pl.String).to_list()


def pred_field(kind: str) -> str:
    return "pred_y__l0.05" if kind == "lasso" else "pred_y"


# ---------------------------------------------------------------------------
# A. error_inflation for kalman, rls, lasso


class TestTheGatesAreOffUnlessSet:
    @pytest.mark.parametrize("kind", ["kalman", "rls", "lasso"])
    def test_no_row_is_withheld_for_noise_by_default(self, kind):
        """D8: no default moves. The gate is off on these models until
        ``max_error_inflation`` is set, and their ``min_weight`` keeps its
        default."""
        out = po.ModelBank([build(kind)]).fit_predict(frame(200))
        assert LIMIT_REASON not in set(reasons(out))
        assert resolved_limit(build(kind)) == "inf"

    def test_ewridge_keeps_root_two(self):
        assert resolved_limit(build("ewridge")) == pytest.approx(math.sqrt(2.0))


class TestKalman:
    def test_the_gate_withholds_exactly_the_rows_whose_own_value_reaches_it(self):
        """``kalman``'s gate runs on the per-row value, which the field
        shows: a row is withheld for noise if and only if its own
        ``error_inflation`` is at or above the limit."""
        limit = 1.05
        df = frame(400)
        s = build("kalman", max_error_inflation=limit, emit_error_inflation=True, min_weight=0.0)
        out = unnest(po.ModelBank([s]).fit_predict(df))
        # Infinite -- `P` unsized, no noise yet -- is written as null.
        infl = out["error_inflation_y"].fill_null(math.inf).to_numpy()
        why = out["withheld_reason"].cast(pl.String).to_list()
        pred = out["pred_y"].to_list()
        gated = [r == LIMIT_REASON for r in why]
        assert gated == [bool(v >= limit) for v in infl]
        assert any(gated) and not all(gated)
        assert all(p is None for p, g in zip(pred, gated, strict=True) if g)
        # Rows a gate withholds later in the stream: per row, not per stream.
        assert any(gated[200:]) and not all(gated[200:])

    def test_is_calibrated_where_the_model_is_exactly_specified(self):
        """Data generated from the random walk the filter assumes, with the
        filter's own ``q`` and ``obs_var`` and a prior it is drawn from: the
        one-step error has variance ``R (1 + h)``, so ``e² / (R (1 + h))`` is
        a chi-square of one degree of freedom and its mean over ``n`` rows is
        1 within ``4 sqrt(2 / n)`` (four standard errors; 0.089 at 4000)."""
        rng = np.random.default_rng(5)
        n, k1, R, q, p0 = 4000, 3, 0.5, 0.002, 2.0
        beta = rng.normal(0.0, math.sqrt(p0 * R), k1)
        x = rng.normal(size=(n, 2))
        y = np.empty(n)
        for i in range(n):
            if i > 0:
                beta = beta + rng.normal(0.0, math.sqrt(q), k1)
            y[i] = beta @ np.array([1.0, *x[i]]) + rng.normal(0.0, math.sqrt(R))
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        s = po.spec.kalman(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=50.0,
            q=[q] * k1,
            obs_var=R,
            p0=p0,
            standardize=False,
            min_weight=0.0,
            emit_error_inflation=True,
        )
        out = unnest(po.ModelBank([s]).fit_predict(df))
        e = out["resid_y"].to_numpy()
        infl = out["error_inflation_y"].to_numpy()
        ok = np.isfinite(e)
        assert ok.sum() == n - 1
        ratio = e[ok] ** 2 / (R * infl[ok] ** 2)
        assert abs(ratio.mean() - 1.0) < 4.0 * math.sqrt(2.0 / ok.sum()), ratio.mean()
        # And the estimation variance is visible: not a ratio of 1.
        assert np.median(infl[ok]) > 1.01

    def test_the_summary_reads_the_mean_field(self):
        """Without a row to read it at, ``summary`` reports the per-row
        value's mean field, ``sqrt(1 + Σ_i P_ii E[z_i²] / R)``: finite and
        above 1 once the filter has a noise."""
        bank = po.ModelBank([build("kalman")])
        bank.fit_predict(frame(300))
        v = bank.summary("m")["error_inflation"][0]
        assert v is not None and 1.0 < v < 1.5


class TestRls:
    def test_the_gate_opens_where_its_bound_says(self):
        """The gate reads ``sqrt(1 + k_total / n_kish)``: on unit rows, a row
        clock and ``λ = 2^(-1/h)``, ``n_kish`` before row ``i`` is
        ``(1 - λ^i)² (1 - λ²) / ((1 - λ)² (1 - λ^(2i)))``, so the first row it
        lets through is the first where ``1 + 3 / n_kish < limit²``."""
        h, limit = 20.0, 1.3
        lam = 2.0 ** (-1.0 / h)

        def n_kish(i: int) -> float:
            return (1 - lam**i) ** 2 * (1 - lam**2) / ((1 - lam) ** 2 * (1 - lam ** (2 * i)))

        want = next(i for i in range(1, 400) if 1.0 + 3.0 / n_kish(i) < limit**2)
        s = build("rls", half_life=h, max_error_inflation=limit, min_weight=0.0)
        why = reasons(po.ModelBank([s]).fit_predict(frame(200)))
        assert why[:want] == [LIMIT_REASON] * want
        assert all(r is None for r in why[want:]), why[want : want + 5]

    def test_weights_enter_through_kish_n(self):
        """Scaling every weight by a constant moves nothing; one heavy row is
        barely one observation to Kish's ``n``, and withholds for longer
        than unit weights."""
        df = frame(300)
        # A prior of 1e-9: `delta` is in the sums' units, so it scales with
        # nothing and must be negligible for the weights' scale to cancel.
        s = build("rls", max_error_inflation=1.2, emit_error_inflation=True, weight="w", delta=1e-9)
        a = unnest(po.ModelBank([s]).fit_predict(df))
        b = unnest(po.ModelBank([s]).fit_predict(df.with_columns(pl.col("w") * 7.0)))
        # From the third row, a row per coefficient: before it, a direction
        # no row has excited is the prior's alone, which scales with nothing.
        np.testing.assert_allclose(
            a["error_inflation_y"][3:].to_numpy(), b["error_inflation_y"][3:].to_numpy(), rtol=1e-6
        )
        unit = df.with_columns(pl.lit(1.0).alias("w"))
        heavy = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(50.0).otherwise(1.0).alias("w")
        )
        held = [
            reasons(po.ModelBank([s]).fit_predict(d)).count(LIMIT_REASON) for d in (unit, heavy)
        ]
        assert held[1] > held[0], held


class TestLasso:
    def test_the_degrees_of_freedom_are_the_active_count(self):
        """``lasso``'s gate reads ``sqrt(1 + df / n_kish)`` per path point,
        ``df`` the coefficients the last solve left active plus the
        intercept (Zou, Hastie and Tibshirani 2007). Two of five features
        carry the signal, so a penalty that zeroes the rest leaves ``df = 3``;
        the summary reads it against the emitted ``coef``'s non-zeros."""
        rng = np.random.default_rng(3)
        n, k = 500, 5
        x = rng.standard_normal((n, k))
        y = 1.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * rng.standard_normal(n)
        df = pl.DataFrame({**{f"x{j}": x[:, j] for j in range(k)}, "y": y})
        feats = [f"x{j}" for j in range(k)]
        h = 40.0
        s = po.spec.lasso(
            "m", targets=["y"], features=feats, half_life=h, lasso_path=[0.3], coef_every=0
        )
        bank = po.ModelBank([s])
        out = unnest(bank.fit_predict(df))
        coef = out["coef"][-1].to_list()
        active = sum(1 for c in coef[1:] if c != 0.0)
        assert active == 2, coef
        lam = 2.0 ** (-1.0 / h)
        nk = (1 - lam**n) ** 2 * (1 - lam**2) / ((1 - lam) ** 2 * (1 - lam ** (2 * n)))
        got = bank.summary("m")["error_inflation"][0]
        assert got == pytest.approx(math.sqrt(1.0 + (active + 1) / nk), rel=1e-12)

    def test_the_gate_opens_where_the_formula_says(self):
        """Per row, from the state before it: the active count after the row
        before over Kish's ``n`` before the row."""
        h, limit = 20.0, 1.2
        lam = 2.0 ** (-1.0 / h)
        s = build("lasso", half_life=h, max_error_inflation=limit, coef_every=1, min_weight=0.0)
        out = unnest(po.ModelBank([s]).fit_predict(frame(300)))
        coef = out["coef"].to_list()
        why = out["withheld_reason"].cast(pl.String).to_list()
        for i in range(1, 300):
            prev = coef[i - 1]
            if prev is None or any(c is None or math.isnan(c) for c in prev):
                continue
            df_ = 1 + sum(1 for c in prev[1:] if c != 0.0)
            nk = (1 - lam**i) ** 2 * (1 - lam**2) / ((1 - lam) ** 2 * (1 - lam ** (2 * i)))
            ratio = math.sqrt(1.0 + df_ / nk)
            if abs(ratio - limit) < 1e-9:
                continue
            assert (why[i] == LIMIT_REASON) == (ratio >= limit), (i, ratio, why[i])

    def test_emit_error_inflation_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="keeps no factor"):
            build("lasso", emit_error_inflation=True)


class TestRefusals:
    def test_the_scalar_models_say_why_they_have_no_noise_gate(self):
        with pytest.raises(ValueError, match="a mean's variance"):
            po.spec.ew_cov("m", features=["x0", "x1"], half_life=10.0, max_error_inflation=2.0)

    def test_a_gradient_model_has_no_noise_gate(self):
        with pytest.raises(ValueError, match="max_error_inflation"):
            build("sgd", learning_rate=0.01, max_error_inflation=2.0)


ROW_FIELD_KINDS = ["kalman", "rls"]
GATE_KINDS = ["kalman", "rls", "lasso"]


def gated_spec(kind: str) -> dict:
    kw: dict[str, Any] = dict(max_error_inflation=1.25, weight="w", min_weight=0.0)
    if kind in ROW_FIELD_KINDS:
        kw["emit_error_inflation"] = True
    return build(kind, **kw)


class TestTheHardRules:
    @pytest.mark.parametrize("kind", GATE_KINDS)
    def test_the_rows_own_target_moves_nothing_it_is_scored_with(self, kind):
        """Hard rule 2: the statistic is read from the state before the
        row's target. Perturbing row ``i``'s target moves neither its field,
        to the bit, nor whether its gate withheld it."""
        df = frame(200)
        s = gated_spec(kind)
        base = unnest(po.ModelBank([s]).fit_predict(df))
        for i in (5, 60, 150):
            moved = df.with_columns(
                pl.when(pl.int_range(pl.len()) == i)
                .then(pl.col("y") + 100.0)
                .otherwise(pl.col("y"))
                .alias("y")
            )
            got = unnest(po.ModelBank([s]).fit_predict(moved))
            assert got["withheld_reason"][i] == base["withheld_reason"][i], i
            if kind in ROW_FIELD_KINDS:
                a = base["error_inflation_y"][i]
                b = got["error_inflation_y"][i]
                assert np.float64(a).tobytes() == np.float64(b).tobytes(), (i, a, b)

    @pytest.mark.parametrize("kind", GATE_KINDS)
    def test_one_seven_or_six_hundred_chunks_give_the_same_bits(self, kind):
        """Hard rule 3: the field and the gate's withholding, to the bit."""
        df = frame(600).with_columns(
            # A null target and a zero-weight row, every so often.
            pl.when(pl.int_range(pl.len()) % 13 == 4).then(None).otherwise(pl.col("y")).alias("y"),
            pl.when(pl.int_range(pl.len()) % 17 == 9).then(0.0).otherwise(pl.col("w")).alias("w"),
        )
        s = gated_spec(kind)
        one = unnest(po.ModelBank([s]).fit_predict(df))
        for n_chunks in (7, 600):
            bank = po.ModelBank([s])
            step = -(-600 // n_chunks)
            many = unnest(
                pl.concat([bank.fit_predict(df[i : i + step]) for i in range(0, 600, step)])
            )
            assert one["withheld_reason"].to_list() == many["withheld_reason"].to_list()
            p = pred_field(kind)
            assert one[p].equals(many[p])
            if kind in ROW_FIELD_KINDS:
                a = one["error_inflation_y"].to_numpy()
                b = many["error_inflation_y"].to_numpy()
                assert a.tobytes() == b.tobytes(), n_chunks
        assert LIMIT_REASON in set(one["withheld_reason"].cast(pl.String).to_list())

    @pytest.mark.parametrize("kind", ROW_FIELD_KINDS)
    def test_a_zero_weight_row_and_a_null_target_learn_nothing(self, kind):
        """A zero-weight row and a null target teach the statistic nothing:
        the next row reads the same either way. (Unstandardized, since a
        null target's features still teach `kalman`'s standardizer.)"""
        df = frame(120)
        zero = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 50).then(0.0).otherwise(pl.col("w")).alias("w")
        )
        null = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 50).then(None).otherwise(pl.col("y")).alias("y")
        )
        extra = {"standardize": False} if kind == "kalman" else {}
        s = build(
            kind,
            max_error_inflation=1.25,
            weight="w",
            min_weight=0.0,
            emit_error_inflation=True,
            **extra,
        )
        a = unnest(po.ModelBank([s]).fit_predict(zero))["error_inflation_y"].to_numpy()
        b = unnest(po.ModelBank([s]).fit_predict(null))["error_inflation_y"].to_numpy()
        np.testing.assert_array_equal(a[:52], b[:52])
        assert np.isfinite(a[51])


# ---------------------------------------------------------------------------
# B. support_coef for huber and quantile


def robust(kind: str, **kw: Any) -> dict:
    base: dict[str, Any] = dict(targets=["y"], features=["x0", "x1"], half_life=math.inf)
    if kind == "quantile":
        base["quantile"] = 0.5
    base.update(kw)
    return getattr(po.spec, kind)("m", **base)


class TestRobustSupport:
    @pytest.mark.parametrize("kind", ["huber", "quantile"])
    def test_a_duplicated_column_reads_half(self, kind):
        """``support_coef = diag(G_raw · G⁻¹)`` on the system each solve
        inverts, the band-reweighted Gram plus the mean-form ridge: a
        duplicated pair splits its coefficient evenly, a clean column is
        all data (WARMUP §2.2)."""
        df = frame(300).with_columns(pl.col("x0").alias("x2"))
        s = robust(kind, features=["x0", "x1", "x2"], ridge=1e-8, coef_every=0)
        out = unnest(po.ModelBank([s]).fit_predict(df))
        support = out["support_coef"][-1].to_list()
        assert support[0] is None, "the intercept is not a share"
        assert support[1:] == pytest.approx([0.5, 1.0, 0.5], abs=1e-3)

    @pytest.mark.parametrize("standardize", [False, True])
    def test_huber_without_outliers_is_the_ridges_shrinkage(self, standardize):
        """With ``huber_delta`` far past every residual no row is
        down-weighted, and the band Gram is the EW Gram: the share is
        ``1 − λ ((Σ + λI)⁻¹)_jj``, ``Σ`` the centred mean-form covariance
        (in correlation form under ``standardize``), from numpy."""
        rng = np.random.default_rng(9)
        n, ridge = 300, 0.4
        x = rng.normal(size=(n, 2)) * np.array([1.5, 0.6])
        x[:, 1] += 0.5 * x[:, 0]
        y = 1.0 + x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        s = robust("huber", huber_delta=1e6, ridge=ridge, standardize=standardize, coef_every=0)
        out = unnest(po.ModelBank([s]).fit_predict(df))
        got = out["support_coef"][-1].to_list()
        cov = np.cov(x.T, bias=True)
        if standardize:
            sd = np.sqrt(np.diag(cov))
            cov = cov / np.outer(sd, sd)
        want = 1.0 - ridge * np.diag(np.linalg.inv(cov + ridge * np.eye(2)))
        assert got[0] is None
        np.testing.assert_allclose(got[1:], want, rtol=1e-9)

    def test_the_summary_and_the_warning_name_the_feature(self):
        df = frame(300).with_columns(pl.col("x0").alias("x2"))
        s = robust("huber", features=["x0", "x1", "x2"], ridge=1e-8, half_life=20.0, coef_every=1)
        bank = po.ModelBank([s])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit_predict(df)
        got = [str(w.message) for w in caught if issubclass(w.category, po.ReadinessWarning)]
        assert len(got) == 1 and "support_coef" in got[0], got
        summary = bank.summary("m")
        assert summary["min_support_coef"][0] < 0.5
        assert summary["min_support_coef_feature"][0] in ("x0", "x2")


# ---------------------------------------------------------------------------
# F. se_coef


class TestSeCoef:
    @pytest.mark.parametrize(
        ("kind", "why"),
        [
            ("lasso", "post-selection"),
            ("huber", "sandwich"),
            ("quantile", "sandwich"),
            ("sgd", "second moment"),
            ("pa", "second moment"),
            ("ftrl", "second moment"),
        ],
    )
    def test_is_refused_where_no_covariance_is_kept_with_the_reason(self, kind, why):
        kw: dict[str, Any] = {}
        if kind == "quantile":
            kw["quantile"] = 0.5
        if kind == "sgd":
            kw["learning_rate"] = 0.01
        with pytest.raises(ValueError, match=why):
            build(kind, emit_se_coef=True, **kw)

    @pytest.mark.parametrize("kind", ["ewridge", "rls", "kalman"])
    def test_rides_on_the_coef_schedule_and_unnests_like_coef(self, kind):
        s = build(kind, emit_se_coef=True)
        out = po.ModelBank([s]).fit_predict(frame(100))
        f = unnest(out)
        assert f["se_coef"].null_count() == f["coef"].null_count()
        assert len(f["se_coef"][-1]) == len(f["coef"][-1]) == 3
        assert all(v > 0.0 for v in f["se_coef"][-1].to_list())
        cols = po.unnest(out, [s]).columns
        assert {"se_coef_y_intercept", "se_coef_y_x0", "se_coef_y_x1"} <= set(cols), cols

    @pytest.mark.parametrize("kind", ["ewridge", "rls", "kalman"])
    def test_is_chunk_invariant_on_every_coef_row(self, kind):
        df = frame(210)
        s = build(kind, emit_se_coef=True, coef_every=1)
        one = unnest(po.ModelBank([s]).fit_predict(df))
        bank = po.ModelBank([s])
        many = unnest(pl.concat([bank.fit_predict(df[i : i + 30]) for i in range(0, 210, 30)]))
        assert one["se_coef"].equals(many["se_coef"])

    def test_is_null_until_there_is_a_residual_spread(self):
        """``ewridge``'s and ``rls``'s are ``sigma · sqrt(M_jj)``: no
        ``sigma`` yet, no standard error."""
        f = unnest(
            po.ModelBank([build("rls", emit_se_coef=True, coef_every=1)]).fit_predict(frame(30))
        )
        se = f["se_coef"].to_list()
        assert se[0] is None or all(v is None for v in se[0])
        assert all(v is not None for v in se[-1])

    def test_shrinks_as_the_sample_grows(self):
        """Without decay the standard errors fall like one over the root of
        the rows."""
        s = build("ewridge", emit_se_coef=True, coef_every=1, half_life=math.inf)
        se = unnest(po.ModelBank([s]).fit_predict(frame(1600)))["se_coef"]
        ratio = np.array(se[1599].to_list()) / np.array(se[399].to_list())
        np.testing.assert_allclose(ratio, 0.5, rtol=0.15)
