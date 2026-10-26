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

import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po
from reference import robust_ref

TIER = "essential"

FEATURES = ["x0", "x1", "x2"]
INF = float("inf")


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
        """The largest set is the plain fit of its features to the bit: the
        model's own floor counts the largest set's coefficients, and
        ``min_weight`` withholds the output alone (task 278, D10). Under one
        ``min_weight`` every field is the same; under each spec's default,
        which counts every feature of the spec, a coefficient each with the
        intercept, the wider spec withholds more rows, and the rows both
        report are the same. Before, the gate held learning back too, so
        the two parted until the decay forgot the first rows (task 272 (e),
        R5-1)."""
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
        # Under each spec's default the wider one withholds more rows, and
        # its gate withholds the output alone (task 278, D10): wherever both
        # report, the same bits.
        wide, alone = pair()
        a, b = _field(wide, "pred_y0__a"), _field(alone, "pred_y0")
        first = lambda v: int(np.nonzero(np.isfinite(v))[0][0])  # noqa: E731
        assert first(a) > first(b), "the wider spec's default waits for more weight"
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 500
        assert np.array_equal(a[both], b[both])

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
        # The recursion below is the one pass: the refit (task 277) is off.
        # Every prediction the fit learns from is reported: min_weight 0
        # (it withholds the output alone, task 278, D10).
        hub = dict(
            huber_scale="rms", huber_delta=1.345, ridge=1e-4, huber_refit_rows=0, min_weight=0.0
        )
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
        # The recursion below is the one pass: the refit (task 277) is off.
        one = dict(ridge=1e-4, huber_refit_rows=0)
        out = po.ModelBank([po.spec.huber("m", feature_sets=sets, **one, **c)]).fit_predict(df)
        plain = po.ModelBank([po.spec.huber("m", emit_huber_scale=True, **one, **c)]).fit_predict(
            df
        )
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


def _coefs(out: pl.DataFrame, width: int) -> np.ndarray:
    """``coef`` on every row as an array, NaN where it is null."""
    return np.array(
        [[np.nan] * width if r is None else r for r in out["m"].struct.field("coef").to_list()],
        dtype=float,
    )


class TestParameterRelations:
    """``huber``'s own keywords held to relations that need no oracle, each
    to the bit, and its ridge to ``robust_ref`` (task 278, batch 19): an
    infinite cut is ``ewridge``, the emitted scale moves nothing, the clock
    cadence solves where its docstring says, and a feature scaled by a power
    of two changes only its own coefficient. ``huber_scale`` and
    ``huber_scale_half_life`` are ``tests/test_huber_scale.py``'s."""

    @pytest.mark.parametrize("ridge", [1e-6, 5.0])
    @pytest.mark.parametrize(
        "schedule",
        [{}, {"max_rows_between_solves": 1}, {"max_rows_between_solves": 5}],
        ids=["weight cadence", "every row", "five rows"],
    )
    @pytest.mark.parametrize(
        "targets", [["y0"], ["y0", "y1"]], ids=["one target", "two targets with gaps"]
    )
    def test_an_infinite_cut_is_ewridge_to_the_bit(self, targets, schedule, ridge):
        """At ``huber_delta=inf`` every Huber weight is 1, so the sums are
        ``ewridge``'s and so is every field, on a stream with gaps past
        ``gap_cap``, rows of weight 0 and a second target null on a tenth of
        its rows, at the default ridge and at 5. The refit of the first rows
        is off there by default (an infinite cut has no outliers to refit
        for), so it moves nothing either. ``min_weight`` is given: the two
        models' defaults differ (``huber``'s is a row a coefficient,
        ``ewridge``'s 0 behind its error-inflation gate), and so would
        ``withheld_reason`` on the first rows. Under ``solve_every`` the two
        part on the first solve, which ``huber`` takes on the row its target
        has a row a coefficient and ``ewridge`` on the row its weight
        reaches that many, then each schedules from its own first."""
        df = _frame()
        c = dict(COMMON, targets=targets, min_weight=4.0, ridge=ridge, **schedule)
        with warnings.catch_warnings():
            # A ridge of 5 sets most of x2's coefficient, as the notice says.
            warnings.simplefilter("ignore", po.ReadinessWarning)
            hub = po.ModelBank([po.spec.huber("m", huber_delta=INF, **c)]).fit_predict(df)
            ew = po.ModelBank([po.spec.ewridge("m", **c)]).fit_predict(df)
            off = po.spec.huber("m", huber_delta=INF, huber_refit_rows=0, **c)
            assert po.ModelBank([off]).fit_predict(df).equals(hub, null_equal=True)
        assert hub.equals(ew, null_equal=True)
        # Learned rows past the refit's 500, and a finite cut that moves the
        # fit, so the refit could have run and the relation is not vacuous.
        assert int((df["w"] > 0).sum()) > 1000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            finite = po.ModelBank([po.spec.huber("m", **c)]).fit_predict(df)
        assert not np.allclose(_field(finite, "pred_y0"), _field(hub, "pred_y0"), equal_nan=True)

    def test_an_infinite_cut_refuses_a_refit(self):
        with pytest.raises(
            ValueError, match="huber_refit_rows is for huber with a finite huber_delta"
        ):
            po.ModelBank([po.spec.huber("m", huber_delta=INF, huber_refit_rows=10, **COMMON)])

    @pytest.mark.parametrize(
        "kw",
        [{}, {"huber_scale": "rms"}, {"feature_sets": SETS}, {"embargo": 25.0}],
        ids=["mad", "rms", "feature sets", "embargo"],
    )
    def test_emitting_the_scale_moves_no_other_field(self, kw):
        """``emit_huber_scale`` adds ``huber_scale_<t>`` and nothing else:
        every other field the same to the bit, the refit of the first rows
        included (a target here passes its 500 learned rows)."""
        df = _frame()
        plain = po.ModelBank([po.spec.huber("m", **COMMON, **kw)]).fit_predict(df)
        on = po.ModelBank([po.spec.huber("m", emit_huber_scale=True, **COMMON, **kw)]).fit_predict(
            df
        )
        added = [f for f in on["m"].struct.fields if f not in plain["m"].struct.fields]
        assert added == ["huber_scale_y0", "huber_scale_y1"]
        kept = on.with_columns(
            m=pl.struct([pl.col("m").struct.field(f) for f in plain["m"].struct.fields])
        )
        assert kept.equals(plain, null_equal=True)
        assert np.isfinite(_field(on, "huber_scale_y0")[50:]).all()

    @pytest.mark.parametrize("ridge", [0.5, 5.0])
    @pytest.mark.parametrize("scale", ["mad", "rms"])
    def test_a_ridge_is_robust_refs(self, scale, ridge):
        """``robust_ref`` at two ridges other than the default, under each
        scale, row by row on two targets with gaps and rows of weight 0.
        The reference is the one-pass recursion, so the refit of the first
        rows is off (it is held to statsmodels in ``test_huber_refit.py``).
        Measured 2026-10-26: at most 1.9e-15 of ``1 + |pred|`` over the four
        cases; the bound is 100 times that, rounded up to a power of ten.
        Each ridge moves the predictions from the default ridge's, by 3.4 to
        8.8 on ``y0`` (measured the same day)."""
        df = _frame()
        x = df.select(FEATURES).to_numpy()
        y = df.select("y0", "y1").to_numpy().astype(float)
        t = df["t"].to_numpy()
        dc = np.concatenate(([0.0], np.diff(t)))
        common = dict(COMMON, coef_every=None, max_rows_between_solves=1, min_weight=5.0)
        common = {k: v for k, v in common.items() if v is not None}

        def bank(r):
            spec = po.spec.huber("m", ridge=r, huber_scale=scale, huber_refit_rows=0, **common)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", po.ReadinessWarning)
                return po.ModelBank([spec]).fit_predict(df)

        out = bank(ridge)
        ref = robust_ref(
            x,
            y,
            dc,
            df["w"].to_numpy(),
            half_life=300.0,
            ridge=ridge,
            min_weight=5.0,
            gap_cap=20.0,
            huber_scale=scale,
        )
        worst = 0.0
        for j, name in enumerate(("y0", "y1")):
            got, exp = _field(out, f"pred_{name}"), ref["pred"][:, j]
            assert np.array_equal(np.isnan(got), np.isnan(exp)), name
            ok = np.isfinite(exp)
            assert ok.sum() > 1000, name
            worst = max(worst, float(np.max(np.abs(got[ok] - exp[ok]) / (1.0 + np.abs(exp[ok])))))
            resid = _field(out, f"resid_{name}")
            ok = np.isfinite(ref["resid"][:, j])
            np.testing.assert_allclose(resid[ok], ref["resid"][ok, j], rtol=1e-12, atol=1e-12)
        print(f"ridge {ridge}, {scale}: worst {worst:.2e} of 1 + |pred|")
        assert worst <= 1e-12, worst
        moved = np.nanmax(np.abs(_field(out, "pred_y0") - _field(bank(1e-6), "pred_y0")))
        assert moved > 1.0, moved


class TestHuberSolveEvery:
    """``solve_every`` through the bank, on a clock whose steps are quarters
    (so every elapsed clock is exact, and a solve falls on the edge itself
    often). The fit is first solved on the row its target has been present
    on a row per coefficient (the third), then on the first row whose clock
    is at least ``solve_every`` past the last solve's, and the coefficients
    hold in between, ``coef`` on every row showing each solve. The refit of
    the first rows solves the target once more, on the row its 500th learned
    row arrives, and leaves the schedule as it stood (``robust.rs``,
    ``refit_target``). Under the Huber loss a cadence moves the trajectory
    (the weights read the fit), so a scheduled run is not the every-row run
    held between solves; the schedule and the predictions read from the
    coefficients reported are what is pinned."""

    N, EVERY = 800, 5.0

    @classmethod
    def frame(cls) -> pl.DataFrame:
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 1.0, (cls.N, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + 0.5 * rng.standard_normal(cls.N)
        y[(rng.random(cls.N) < 0.1) & (x[:, 0] > 0)] += 20.0
        t = np.cumsum(rng.integers(1, 9, cls.N) * 0.25)
        return pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "y": y})

    @classmethod
    def spec(cls, **kw) -> dict:
        kw = {
            "clock": "t",
            "gap_cap": 10.0,
            "half_life": 100.0,
            "min_weight": 3.0,
            "solve_every": cls.EVERY,
            **kw,
        }
        return po.spec.huber(
            "m", targets=["y"], features=["x0", "x1"], max_rows_between_coefs=1, **kw
        )

    def schedule(self, t: np.ndarray, first: int) -> list[int]:
        want, last = [first], first
        for i in range(first + 1, self.N):
            if t[i] - t[last] >= self.EVERY:
                want.append(i)
                last = i
        return want

    @pytest.mark.parametrize("refit", [None, 0], ids=["the default refit", "no refit"])
    def test_the_fit_moves_on_the_scheduled_rows_and_holds_between(self, refit):
        df = self.frame()
        kw = {} if refit is None else {"huber_refit_rows": refit}
        out = po.ModelBank([self.spec(**kw)]).fit_predict(df).unnest("m")
        t, coef = df["t"].to_numpy(), out["coef"].to_list()
        moved = [
            i for i in range(self.N) if coef[i] is not None and (i == 0 or coef[i] != coef[i - 1])
        ]
        want = self.schedule(t, 2)
        on_the_edge = sum(t[b] - t[a] == self.EVERY for a, b in zip(want, want[1:], strict=False))
        assert on_the_edge > 10, on_the_edge
        # Every row has weight, so the 500th learned row is row 499, off
        # the schedule: the refit's one solve, and no other row more.
        refit_row = 499
        assert refit_row not in want
        extra = [] if refit == 0 else [refit_row]
        assert moved == sorted(want + extra)
        # Each prediction is the coefficients the row before reported.
        pred = out["pred_y"].to_numpy()
        z = np.column_stack([np.ones(self.N), df["x0"].to_numpy(), df["x1"].to_numpy()])
        rows = [i for i in range(1, self.N) if coef[i - 1] is not None and np.isfinite(pred[i])]
        assert len(rows) > self.N - 10
        want_pred = np.array([z[i] @ np.array(coef[i - 1]) for i in rows])
        np.testing.assert_array_equal(pred[rows], want_pred)

    def test_zero_is_every_row(self):
        """``solve_every = 0`` solves every row: the row cap of 1."""
        df = self.frame()
        zero = po.ModelBank([self.spec(solve_every=0.0)]).fit_predict(df)
        cap = po.ModelBank([self.spec(solve_every=1e18, max_rows_between_solves=1)]).fit_predict(df)
        assert zero.equals(cap, null_equal=True)
        assert not zero.equals(po.ModelBank([self.spec()]).fit_predict(df), null_equal=True)

    def test_chunks_and_a_save_between_solves_keep_the_schedule(self):
        """The clock since the last solve is state, across the refit's row
        too: chunks of 1, 7 and 600 rows, and a bank saved and loaded on
        either side of a solve and of the refit, give the one run to the
        bit."""
        df, spec = self.frame(), self.spec()
        whole = po.ModelBank([spec]).fit_predict(df)
        for rows in (1, 7, 600):
            bank = po.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
            assert pl.concat(parts).equals(whole, null_equal=True), rows
        for cut in (3, 101, 499, 500, 650):
            bank = po.ModelBank([spec])
            head = bank.fit_predict(df.slice(0, cut))
            loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
            tail = loaded.fit_predict(df.slice(cut))
            assert pl.concat([head, tail]).equals(whole, null_equal=True), cut


#: Task 278 (b19), a defect: the refit's stopping rule reads the
#: coefficients in the caller's units (`huber_refit::converged`: no
#: coefficient moved by more than 1e-10 of `max|beta|`), so a feature in
#: small units, whose coefficient is large, loosens the rule for every other
#: coefficient and the refit stops a pass earlier or later. The fit after the
#: refit then differs from the unscaled run's by up to 3.7e-8 of a prediction
#: (x0 times 2^-10, measured 2026-10-26), where the one pass is exact.
REFIT_UNITS = pytest.mark.xfail(
    strict=True,
    reason="task 278 (b19): the refit's convergence rule is in the features' units",
)


class TestPowersOfTwo:
    """Scaling ``x0`` by ``2^k`` is exact, and so is every step of the fit
    that reads it under correlation form (the ridge is dimensionless there)
    or with no ridge: every field to the bit, and ``x0``'s coefficient
    divided by ``2^k`` to the bit (``quantile``'s is
    ``tests/test_robust.py``'s)."""

    @staticmethod
    def check(df, **kw):
        def run(frame):
            with warnings.catch_warnings():
                # A ridge of 1 sets half of x0's coefficient, as the notice says.
                warnings.simplefilter("ignore", po.ReadinessWarning)
                return po.ModelBank([po.spec.huber("m", **COMMON, **kw)]).fit_predict(frame)

        base = run(df)
        coef = _coefs(base, 8)
        x0 = [1, 5]  # the slot of x0 in each target's [intercept, x0, x1, x2]
        rest = [i for i in range(8) if i not in x0]
        # `coef` and `support_coef` from the row the weight reaches a row a
        # coefficient: before it, at ridge 0, the design is short of rank
        # and both are the factor's jitter's, which has units.
        determined = int(np.argmax(_field(base, "weight_sum") >= 4.0))
        for k in (10, -10, 30):
            out = run(df.with_columns(pl.col("x0") * 2.0**k))
            got = _coefs(out, 8)
            fields = [f for f in base["m"].struct.fields if f not in ("coef", "support_coef")]
            m = pl.col("m").struct.field
            a, b = out.select(m(*fields)), base.select(m(*fields))
            assert a.equals(b, null_equal=True), k
            a, b = (o.select(m("support_coef")).slice(determined) for o in (out, base))
            assert a.equals(b, null_equal=True), k
            got, want = got[determined:], coef[determined:]
            np.testing.assert_array_equal(got[:, x0] * 2.0**k, want[:, x0], err_msg=f"k = {k}")
            np.testing.assert_array_equal(got[:, rest], want[:, rest], err_msg=f"k = {k}")

    @pytest.mark.parametrize(
        "kw",
        [
            {"standardize": True},
            {"standardize": True, "ridge": 1.0, "huber_scale": "rms"},
            {"ridge": 0.0},
        ],
        ids=["standardized, default ridge", "standardized, ridge 1, rms", "plain, ridge 0"],
    )
    def test_a_feature_scaled_by_a_power_of_two_changes_only_its_coefficient(self, kw):
        """The one pass, the refit of the first rows off."""
        self.check(_frame(), huber_refit_rows=0, **kw)

    def test_with_the_refit_of_the_first_rows_it_holds_until_the_refit(self):
        """The default, the refit on: target ``y0`` passes its 500th learned
        row at row 549 (``y1`` at row 615), and on every row before it the
        relation holds to the bit."""
        df = _frame()
        assert int(np.argmax(np.cumsum(df["w"].to_numpy() > 0) >= 500)) == 549
        self.check(df.head(549), standardize=True)

    @REFIT_UNITS
    def test_and_so_after_the_refit(self):
        """From the refit's row, its coefficients, and from the next row its
        predictions, part (``REFIT_UNITS``)."""
        self.check(_frame(), standardize=True)
