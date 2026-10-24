"""``huber``'s feature sets, sum-scale prior, emitted scale and delayed
labels: a downstream project's request for Huber regression through the
bank (task 272 (a); the analysis is ``docs/HUBER.md`` §8, request 1).

Each is held to a reference built from a model that already has the
property: ``ewridge`` fed the rows at the weights the largest set's
residuals give, ``rls`` and ``ewridge(ridge_scale="sum")`` at
``huber_delta=inf``, the same stream without the delay. The core's own
references are ``crates/online-core/src/robust/options_tests.rs``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

FEATURES = ["x0", "x1", "x2"]


def _frame(n: int = 1200, seed: int = 4) -> pl.DataFrame:
    """A level, outliers the cut binds on (one-sided, on rows with
    ``x0 > 0``), a second target null on a tenth of the rows, uneven
    weights with a zero every 11 rows, and an irregular clock."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    y0 = 500.0 + x @ np.array([2.0, -1.0, 0.5]) + 0.4 * rng.standard_normal(n)
    y0[(rng.random(n) < 0.06) & (x[:, 0] > 0)] += 40.0
    y1 = -2.0 + x[:, 1] + rng.standard_t(3, n)
    y1[rng.random(n) < 0.1] = np.nan
    w = np.where(np.arange(n) % 11 == 5, 0.0, rng.uniform(0.5, 1.5, n))
    t = np.cumsum(rng.integers(1, 4, n)).astype(float)
    return pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y0": y0, "y1": y1, "w": w, "t": t}
    ).with_columns(pl.col("y1").fill_nan(None))


COMMON = dict(
    targets=["y0", "y1"],
    features=FEATURES,
    clock="t",
    gap_cap=20.0,
    half_life=300.0,
    weight="w",
    coef_every=1,
)
SETS = {"one": ["x1"], "all": FEATURES, "two": ["x2", "x0"]}


def _field(out: pl.DataFrame, name: str) -> np.ndarray:
    return out["m"].struct.field(name).to_numpy().astype(float)


class TestFeatureSets:
    def test_the_fields_are_ewridges(self):
        spec = po.spec.huber("m", feature_sets=SETS, **COMMON)
        fields = po.spec.output_index(spec)["field"].to_list()
        for t in ("y0", "y1"):
            for name in SETS:
                assert f"pred_{t}__{name}" in fields and f"resid_{t}__{name}" in fields
        assert "pred_y0" not in fields
        one = po.spec.huber("m", feature_sets={"a": ["x0"]}, **COMMON)
        assert "pred_y0" in po.spec.output_index(one)["field"].to_list()

    def test_the_largest_set_is_the_plain_fit_to_the_bit(self):
        """The rows enter the sums at the largest set's Huber weight, so the
        largest set's slot is the fit of a ``huber`` on those features, to
        the bit; a smaller set is a fit of its own from the same sums."""
        df = _frame()
        sets = po.ModelBank([po.spec.huber("m", feature_sets=SETS, **COMMON)]).fit_predict(df)
        plain = po.ModelBank([po.spec.huber("m", **COMMON)]).fit_predict(df)
        for t in ("y0", "y1"):
            a, b = _field(sets, f"pred_{t}__all"), _field(plain, f"pred_{t}")
            assert np.array_equal(a, b, equal_nan=True), t
            assert not np.allclose(_field(sets, f"pred_{t}__one"), b, equal_nan=True), (
                "a smaller set is not the full fit"
            )

    def test_a_largest_set_short_of_every_feature_needs_the_same_min_weight(self):
        """The largest set is the plain fit of its features to the bit when
        it is every feature, or under one ``min_weight``: the default counts
        every feature of the spec, a coefficient each with the intercept, so
        a largest set short of them predicts later than a ``huber`` on its
        features alone, its first Huber weights and scale differ, and the
        two part until the decay forgets those rows (task 272 (e), R5-1)."""
        df = _frame()
        base = {k: v for k, v in COMMON.items() if k != "features"}
        sets = {"a": ["x0", "x1"], "b": ["x2"]}

        def pair(**kw):
            wide = po.spec.huber(
                "m", features=FEATURES, feature_sets=sets, emit_huber_scale=True, **base, **kw
            )
            alone = po.spec.huber("m", features=["x0", "x1"], emit_huber_scale=True, **base, **kw)
            return (po.ModelBank([wide]).fit_predict(df), po.ModelBank([alone]).fit_predict(df))

        wide, alone = pair(min_weight=6.0)
        for t in ("y0", "y1"):
            for a, b in ((f"pred_{t}__a", f"pred_{t}"), (f"huber_scale_{t}", f"huber_scale_{t}")):
                assert np.array_equal(_field(wide, a), _field(alone, b), equal_nan=True), (t, a)
        wide, alone = pair()
        a, b = _field(wide, "pred_y0__a"), _field(alone, "pred_y0")
        first = lambda v: int(np.nonzero(np.isfinite(v))[0][0])  # noqa: E731
        assert first(a) > first(b), "the wider spec's default waits for more weight"
        gap = np.abs(a - b)[first(a) :]
        early, late = np.nanmax(gap[:50]), np.nanmax(gap[-200:])
        assert early > 1e-3, early
        assert late < early / 100, (early, late)

    def test_each_set_is_weighed_by_the_largest_sets_residual(self):
        """A smaller set's fit is the weighted ridge fit of its features at
        the row's weight times the Huber weight of the largest set's
        residual, against the scale the cut was drawn in: ``ewridge`` on the
        set, fed those weights, built here from a plain ``huber``'s
        residuals under ``huber_scale = "rms"``, whose scale a recursion in
        numpy gives, solved on every row by both."""
        df = _frame(900).drop("y1").with_columns(pl.col("t").cast(pl.Float64))
        c = dict(COMMON, targets=["y0"], half_life=200.0, solve_every=0.0)
        c.pop("coef_every")
        sets = {"one": ["x1"], "all": FEATURES}
        hub = dict(huber_scale="rms", huber_delta=1.345, ridge=1e-4)
        out = po.ModelBank([po.spec.huber("m", feature_sets=sets, **hub, **c)]).fit_predict(df)
        plain = po.ModelBank([po.spec.huber("m", **hub, **c)]).fit_predict(df)
        pred = _field(plain, "pred_y0")
        y, w, t = (df[col].to_numpy() for col in ("y0", "w", "t"))
        # The scale's recursion (robust.rs): its weight ages by the decay on
        # every row, and a predicted row of positive weight adds its
        # residual at its own weight.
        d = np.diff(t, prepend=t[0]).clip(max=20.0)
        lam = 0.5 ** (d / 200.0)
        s2, ws, w_rob = 0.0, 0.0, np.ones(len(y))
        for i in range(len(y)):
            ws *= lam[i]
            if np.isfinite(pred[i]):
                r = y[i] - pred[i]
                if s2 > 0.0:
                    cut = 1.345 * np.sqrt(s2)
                    w_rob[i] = 1.0 if abs(r) <= cut else cut / abs(r)
                if w[i] > 0.0:
                    s2 = (ws * s2 + w[i] * r * r) / (ws + w[i])
                    ws += w[i]
        assert (w_rob < 1.0).sum() > 30, "the cut must bind"
        ref = po.ModelBank(
            [
                po.spec.ewridge(
                    "m",
                    ridge=1e-4,
                    **dict(c, features=["x1"], weight="wr"),
                    max_rows_between_solves=1,
                )
            ]
        ).fit_predict(df.with_columns(wr=pl.Series(w * w_rob)))
        a, b = _field(out, "pred_y0__one"), _field(ref, "pred_y0")
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 850
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("rows", [1, 7, 600])
    def test_one_chunk_or_many_give_one_run(self, rows):
        df = _frame()
        spec = po.spec.huber("m", feature_sets=SETS, **COMMON)
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
        assert pl.concat(parts).equals(whole, null_equal=True), rows

    def test_a_bank_saved_with_sets_resumes_to_the_bit(self):
        df = _frame()
        spec = po.spec.huber("m", feature_sets=SETS, **COMMON)
        whole = po.ModelBank([spec]).fit_predict(df)
        for cut in (3, 600):
            bank = po.ModelBank([spec])
            head = bank.fit_predict(df.slice(0, cut))
            loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
            tail = loaded.fit_predict(df.slice(cut))
            assert pl.concat([head, tail]).equals(whole, null_equal=True), cut

    def test_a_set_outside_the_features_is_refused(self):
        with pytest.raises(ValueError, match='feature set "a" references unknown feature "z"'):
            po.spec.huber("m", feature_sets={"a": ["z"]}, **COMMON)
        with pytest.raises(ValueError, match="feature_sets names no set"):
            po.spec.huber("m", feature_sets={}, **COMMON)
        with pytest.raises(ValueError, match='feature set "a" lists "x0" more than once'):
            po.spec.huber("m", feature_sets={"a": ["x0", "x0"]}, **COMMON)


class TestRidgeScale:
    """``huber(ridge_scale="sum")`` is the Huber form of ``rls``, as
    ``ewridge(ridge_scale="sum")`` is its least-squares form: at
    ``huber_delta=inf``, solved on every row, all three give the same
    predictions (``tests/test_rls.py`` holds the last two to each other)."""

    @staticmethod
    def _common(**kw):
        return dict(
            targets=["y0"],
            features=["x0", "x1", "x2"],
            clock="t",
            half_life=300.0,
            gap_cap=50.0,
            weight="w",
            min_weight=5.0,
            **kw,
        )

    def _three(self, df, common, ridge):
        hub = po.spec.huber(
            "m", huber_delta=float("inf"), ridge=ridge, ridge_scale="sum", solve_every=0.0, **common
        )
        rls = po.spec.rls("m", delta=ridge, **common)
        ew = po.spec.ewridge(
            "m", ridge=ridge, ridge_scale="sum", max_rows_between_solves=1, **common
        )
        return [_field(po.ModelBank([s]).fit_predict(df), "pred_y0") for s in (hub, rls, ew)]

    def test_at_an_infinite_cut_it_is_rls_and_ewridge(self):
        from data import synthetic

        df, _ = synthetic(seed=21, n_groups=1, n_rows=400, k=3, null_frac=0.0)
        h, r, e = self._three(df, self._common(), 0.7)
        both = np.isfinite(h) & np.isfinite(r) & np.isfinite(e)
        assert both.sum() > 350
        assert np.max(np.abs(h[both] - r[both])) < 1e-9
        assert np.max(np.abs(h[both] - e[both])) < 1e-9

    def test_leading_zero_weight_rows_age_the_prior_as_rls_does(self):
        rng = np.random.default_rng(3)
        n = 60
        x = rng.standard_normal((n, 3))
        df = pl.DataFrame(
            {
                "t": np.arange(float(n)),
                "x0": x[:, 0],
                "x1": x[:, 1],
                "x2": x[:, 2],
                "y0": x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(n),
                "w": [0.0, 0.0, 0.0] + [1.0] * (n - 3),
            }
        )
        h, r, _ = self._three(df, self._common() | {"half_life": 5.0, "gap_cap": 10.0}, 5.0)
        both = np.isfinite(h) & np.isfinite(r)
        assert both.sum() > 40
        assert np.max(np.abs(h[both] - r[both])) < 1e-9

    def test_the_sum_scale_prior_fades_as_the_rows_arrive(self):
        """Under ``"sum"`` the prior reaches the intercept, at a weight that
        falls against the rows' as they arrive (here with no decay, as
        ``ridge / W``), so the early fit sits far below the mean form's on a
        target at 500 and the late one closer: 129 below in rows 10-60 and
        within 3.7 in the last 200, as ``ewridge``'s two forms part."""
        df = _frame()
        c = dict(COMMON, targets=["y0"], half_life=float("inf"))
        mean = _field(po.ModelBank([po.spec.huber("m", ridge=1.0, **c)]).fit_predict(df), "pred_y0")
        summed = _field(
            po.ModelBank([po.spec.huber("m", ridge=1.0, ridge_scale="sum", **c)]).fit_predict(df),
            "pred_y0",
        )
        early = np.nanmax(np.abs(mean[10:60] - summed[10:60]))
        late = np.nanmax(np.abs(mean[-200:] - summed[-200:]))
        assert early > 5.0 and late < early / 10, (early, late)

    def test_it_is_refused_where_ewridges_is(self):
        with pytest.raises(ValueError, match="ridge_scale is incompatible with standardize"):
            po.spec.huber("m", ridge_scale="sum", standardize=True, **COMMON)
        with pytest.raises(ValueError, match="two or more feature sets"):
            po.spec.huber("m", ridge_scale="sum", feature_sets=SETS, **COMMON)
        with pytest.raises(ValueError, match='ridge_scale must be one of "mean", "sum"'):
            po.spec.huber("m", ridge_scale="median", **COMMON)  # type: ignore[arg-type]


class TestEmittedScale:
    """``emit_huber_scale``: each target's residual scale as the row's cut
    reads it, before the row's own residual, as ``huber_scale_<t>``."""

    def test_the_field_is_there_only_when_asked(self):
        on = po.spec.output_index(po.spec.huber("m", emit_huber_scale=True, **COMMON))
        off = po.spec.output_index(po.spec.huber("m", **COMMON))
        assert {"huber_scale_y0", "huber_scale_y1"} <= set(on["field"].to_list())
        assert not any(f.startswith("huber_scale") for f in off["field"].to_list())
        sets = po.spec.huber("m", emit_huber_scale=True, feature_sets=SETS, **COMMON)
        fields = po.spec.output_index(sets)["field"].to_list()
        assert [f for f in fields if f.startswith("huber_scale")] == [
            "huber_scale_y0",
            "huber_scale_y1",
        ], "one scale per target, the largest set's, whatever the sets"

    def test_it_reads_no_row_its_own(self):
        """A wild value on a row leaves that row's scale as it was, and
        enters the next row's: the field is causal by construction. Under
        the root mean square, which a wild residual moves by its size; the
        MAD's median moves by one row's weight, often within its bucket."""
        df = _frame()
        spec = po.spec.huber("m", emit_huber_scale=True, huber_scale="rms", **COMMON)
        at = 700
        wild = df.with_columns(
            y0=pl.when(pl.int_range(pl.len()) == at).then(1e6).otherwise(pl.col("y0"))
        )
        a = _field(po.ModelBank([spec]).fit_predict(df), "huber_scale_y0")
        b = _field(po.ModelBank([spec]).fit_predict(wild), "huber_scale_y0")
        assert np.isnan(a[0]) and np.isfinite(a[50:]).all()
        assert np.array_equal(a[: at + 1], b[: at + 1], equal_nan=True)
        assert b[at + 1] > a[at + 1]

    def test_a_row_with_an_unusable_feature_has_no_scale(self):
        """A row the model does not read, a feature NaN or null, reports no
        scale, as ``lasso``'s ``lam_selected``: the field is null there and
        the rows around it read the scale as it stands (task 272 (e),
        R5-3)."""
        df = _frame()
        spec = po.spec.huber("m", emit_huber_scale=True, **COMMON)
        idx = pl.int_range(pl.len())
        bad = df.with_columns(
            x0=pl.when(idx == 150).then(float("nan")).when(idx == 151).then(None).otherwise("x0")
        )
        s = _field(po.ModelBank([spec]).fit_predict(bad), "huber_scale_y0")
        assert np.isfinite(s[149]) and np.isfinite(s[152])
        assert np.isnan(s[150]) and np.isnan(s[151])

    def test_each_set_is_weighed_by_the_largest_sets_residual_under_the_mad(self):
        """The feature sets' reference of ``TestFeatureSets`` under the
        default scale, the MAD, whose sketch no recursion here reproduces:
        the scale is the plain fit's ``huber_scale_y0``, emitted before each
        row, and the weight is formed from it and the plain fit's residual."""
        df = _frame(900).drop("y1")
        c = dict(COMMON, targets=["y0"], half_life=200.0, solve_every=0.0)
        c.pop("coef_every")
        sets = {"one": ["x1"], "all": FEATURES}
        out = po.ModelBank([po.spec.huber("m", feature_sets=sets, ridge=1e-4, **c)]).fit_predict(df)
        plain = po.ModelBank(
            [po.spec.huber("m", emit_huber_scale=True, ridge=1e-4, **c)]
        ).fit_predict(df)
        pred, scale = _field(plain, "pred_y0"), _field(plain, "huber_scale_y0")
        r = np.abs(df["y0"].to_numpy() - pred)
        cut = 1.345 * scale
        w_rob = np.where(np.isfinite(r) & np.isfinite(cut) & (r > cut), cut / r, 1.0)
        assert (w_rob < 1.0).sum() > 30, "the cut must bind"
        assert np.array_equal(_field(out, "pred_y0__all"), pred, equal_nan=True)
        ref = po.ModelBank(
            [
                po.spec.ewridge(
                    "m",
                    ridge=1e-4,
                    **dict(c, features=["x1"], weight="wr"),
                    max_rows_between_solves=1,
                )
            ]
        ).fit_predict(df.with_columns(wr=pl.Series(df["w"].to_numpy() * w_rob)))
        a, b = _field(out, "pred_y0__one"), _field(ref, "pred_y0")
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 850
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("rows", [1, 7, 600])
    def test_one_chunk_or_many_give_one_run(self, rows):
        df = _frame()
        spec = po.spec.huber("m", emit_huber_scale=True, feature_sets=SETS, **COMMON)
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
        assert pl.concat(parts).equals(whole, null_equal=True), rows

    def test_it_is_refused_on_quantile(self):
        with pytest.raises(TypeError):
            po.spec.quantile("m", quantile=0.5, emit_huber_scale=True, **COMMON)  # type: ignore[call-arg]


class TestDelayedLabels:
    """Under ``embargo`` a row is scored where it arrives and learned once
    its label matures, the held rows in arrival order
    (``crates/online-polars/src/stream/label_delay.rs``), each stepping the
    model as an undelayed row would (``Robust::step``). Its Huber weight is
    therefore read when it is learned, from the fit of every row before it
    and of none after it, which is the fit an undelayed run weighs it by,
    not from the prediction made at its own row. Pinned by the fit: the
    delayed run's ``coef`` at row ``r`` is the undelayed run's at ``r - E``,
    to the bit, on a stream whose cut binds and whose delayed predictions
    differ from the undelayed ones; were a row weighed by the prediction
    made where it was scored, the two fits would part."""

    E = 25

    def _runs(self):
        df = _frame().drop("t")
        c = dict(COMMON, clock=None, gap_cap=None)
        c = {k: v for k, v in c.items() if v is not None}
        spec = dict(feature_sets=SETS, emit_huber_scale=True, **c)
        plain = po.ModelBank([po.spec.huber("m", **spec)]).fit_predict(df)
        held = po.ModelBank([po.spec.huber("m", embargo=self.E, **spec)]).fit_predict(df)
        return df, plain, held

    def test_the_fit_is_the_undelayed_fit_a_delay_behind(self):
        df, plain, held = self._runs()
        a = plain["m"].struct.field("coef").to_list()
        b = held["m"].struct.field("coef").to_list()
        compared = 0
        for r in range(self.E, df.height):
            if b[r] is not None and a[r - self.E] is not None:
                assert b[r] == a[r - self.E], r
                compared += 1
        assert compared > 1000
        # The two runs score from different fits, so a weight read from the
        # scored prediction would differ, and the cut binds.
        p, q = _field(plain, "pred_y0__all"), _field(held, "pred_y0__all")
        both = np.isfinite(p) & np.isfinite(q)
        assert (np.abs(p[both] - q[both]) > 1e-6).mean() > 0.9
        r = np.abs(df["y0"].to_numpy() - p)
        assert (r > 1.345 * _field(plain, "huber_scale_y0")).sum() > 30

    def test_the_weight_is_out_of_sample_for_its_row(self):
        """The fit a held row is weighed by holds no row at or after it: the
        delayed run's prediction at a row is the undelayed fit a delay
        behind, applied to the row's features."""
        df, plain, held = self._runs()
        coef = plain["m"].struct.field("coef").to_list()
        x = df.select(FEATURES).to_numpy()
        q = _field(held, "pred_y0__all")
        compared = 0
        for r in range(self.E, df.height):
            c = coef[r - self.E]
            if c is None or not np.isfinite(q[r]):
                continue
            # Slots are target-major then set: y0's "all" set is the second.
            k = len(FEATURES) + 1
            beta = np.asarray(c[k : 2 * k])
            assert abs(q[r] - (beta[0] + x[r] @ beta[1:])) < 1e-9 * (1 + abs(q[r])), r
            compared += 1
        assert compared > 1000
