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
