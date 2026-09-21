"""Not using a model before it is ready (docs/WARMUP-AND-CONVERGENCE.md).

Two gates, both stated as intent rather than as a number that needs a formula
in the user's head:

- ``min_settled_frac`` -- how far the decay window has filled toward steady
  state, ``settled_frac = 1 - 2 ** (-T / halflife)`` with ``T`` the decay time
  the models have actually seen. Off by default: under a stationary process a
  mean-form fit is unbiased from its first row, so what this guards is a
  history that does not represent the process, which only the user can judge.
- ``max_error_inflation`` -- how much estimation error is expected to inflate
  a prediction's error over the noise floor, ``sqrt(1 + edf / n_kish)``.
  Default ``sqrt(2)``: withhold while the estimation variance exceeds the
  noise being fitted. It replaces the ``k + 1`` floor ``min_periods`` gave
  ``ewridge`` and, unlike it, tracks the model and reads Kish's sample size,
  so uneven weights withhold for longer.

Every withheld row says why (``withheld_reason``), every row says how settled
the stream is (``settled_frac``), and a coefficient the ridge determined more
than the data did is named (``support_coef``). Chunking cannot move any of it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po

REASONS = {"below_min_settled_frac", "above_max_error_inflation", "below_min_periods"}


def frame(n: int = 300, k: int = 2, seed: int = 0, noise: float = 0.5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    beta = np.arange(1, k + 1, dtype=float)
    y = 1.0 + x @ beta + noise * rng.standard_normal(n)
    cols = {f"x{j}": x[:, j] for j in range(k)}
    cols["y"] = y
    cols["t"] = np.arange(float(n))
    return pl.DataFrame(cols)


def spec(**kw) -> dict:
    base = dict(targets=["y"], features=["x0", "x1"], halflife=20.0, max_rows_between_solves=1)
    base.update(kw)
    return po.spec.ewridge("m", **base)


def field(out: pl.DataFrame, name: str) -> pl.Series:
    return out["m"].struct.field(name)


def reasons(out: pl.DataFrame) -> list[str | None]:
    return field(out, "withheld_reason").cast(pl.String).to_list()


def first_present(s: pl.Series) -> int:
    """The row of the first non-null value: where a gate opened."""
    return next(i for i, p in enumerate(s.to_list()) if p is not None)


class TestSettledFrac:
    def test_is_the_decay_time_seen_before_the_row_as_a_fraction_of_steady_state(self):
        out = po.ModelBank([spec(halflife=10.0)]).fit_predict(frame(60))
        got = field(out, "settled_frac").to_list()
        # A row-count clock: one clock unit per row, and a group's first row
        # decays by nothing, so the decay time seen before row `i` is `i - 1`
        # (docs/WARMUP-AND-CONVERGENCE.md §8).
        want = [1.0 - 2.0 ** (-max(i - 1, 0) / 10.0) for i in range(60)]
        assert got == pytest.approx(want, abs=1e-12)
        assert got[0] == 0.0 and got[1] == 0.0
        assert got[11] == pytest.approx(0.5)
        assert got[21] == pytest.approx(0.75)
        assert got[31] == pytest.approx(0.875)

    def test_is_rate_independent_on_a_clock(self):
        """Ten, fifty or a hundred rows per halflife: the fraction is the
        same at one, two and three halflives, because it is the *clock* the
        decay has covered, not a count of anything (§5.2)."""
        for rows_per_halflife in (10, 50, 100):
            n = 4 * rows_per_halflife
            df = frame(n).with_columns((pl.col("t") * (50.0 / rows_per_halflife)).alias("t"))
            s = spec(clock="t", halflife=50.0, max_dclock=1e9)
            got = field(po.ModelBank([s]).fit_predict(df), "settled_frac")
            for halflives in (1, 2, 3):
                # One row on: the first row's delta is zero.
                assert got[halflives * rows_per_halflife + 1] == pytest.approx(
                    1.0 - 2.0**-halflives
                ), rows_per_halflife

    def test_is_null_without_decay(self):
        out = po.ModelBank([spec(halflife=math.inf)]).fit_predict(frame(30))
        assert field(out, "settled_frac").null_count() == 30

    def test_counts_a_capped_gap_as_the_cap(self):
        """A weekend capped to ``max_dclock`` warms the model by
        ``max_dclock``, since that is what it decayed by."""
        df = frame(40).with_columns(
            pl.when(pl.col("t") >= 20).then(pl.col("t") + 1e6).otherwise(pl.col("t")).alias("t")
        )
        s = spec(clock="t", halflife=10.0, max_dclock=5.0, min_backwards_jump=0.0)
        got = field(po.ModelBank([s]).fit_predict(df), "settled_frac")
        # Rows 1..19 are one unit apart (19 units before row 20); the jump
        # at row 20 counts 5, not 1e6, and is seen from row 21 on.
        assert got[20] == pytest.approx(1.0 - 2.0 ** (-19 / 10.0))
        assert got[21] == pytest.approx(1.0 - 2.0 ** (-(19 + 5) / 10.0))
        assert got[22] == pytest.approx(1.0 - 2.0 ** (-(19 + 5 + 1) / 10.0))

    def test_restarts_with_the_state(self):
        df = frame(40).with_columns(
            pl.when(pl.col("t") >= 20).then(pl.col("t") - 100.0).otherwise(pl.col("t")).alias("t")
        )
        s = spec(clock="t", halflife=10.0, max_dclock=5.0, on_clock_reset="reset_state")
        got = field(po.ModelBank([s]).fit_predict(df), "settled_frac")
        # The reset row is a first row again: it decays by nothing.
        assert got[20] == 0.0 and got[21] == 0.0
        assert got[22] == pytest.approx(1.0 - 2.0 ** (-1 / 10.0))


class TestMinSettledFrac:
    def test_withholds_until_the_fraction_and_says_why(self):
        s = spec(halflife=10.0, min_settled_frac=0.5)
        out = po.ModelBank([s]).fit_predict(frame(40))
        pred = field(out, "pred_y")
        why = reasons(out)
        # T = row index - 1 on a row-count clock, so 0.5 is reached at row 11.
        assert pred[:11].null_count() == 11
        assert pred[11:].null_count() == 0
        assert why[:11] == ["below_min_settled_frac"] * 11
        assert why[11:] == [None] * 29

    def test_is_off_by_default(self):
        out = po.ModelBank([spec()]).fit_predict(frame(40))
        assert "below_min_settled_frac" not in set(reasons(out))
        assert field(out, "pred_y")[4:].null_count() == 0

    def test_is_a_fraction_below_one(self):
        for bad in (1.0, 1.5, -0.1, math.nan, math.inf):
            with pytest.raises(ValueError, match="min_settled_frac"):
                spec(min_settled_frac=bad)

    def test_needs_a_decay_to_settle_toward(self):
        with pytest.raises(ValueError, match="min_settled_frac"):
            spec(halflife=math.inf, min_settled_frac=0.5)

    def test_predict_withholds_the_same_way(self):
        s = spec(halflife=10.0, min_settled_frac=0.5)
        bank = po.ModelBank([s])
        bank.fit_predict(frame(5))
        out = bank.predict(frame(3, seed=1))
        assert field(out, "pred_y").null_count() == 3
        assert reasons(out) == ["below_min_settled_frac"] * 3

    def test_a_new_group_settles_on_its_own(self):
        df = pl.concat(
            [
                frame(40).with_columns(pl.lit("a").alias("g")),
                frame(20, seed=1).with_columns(pl.lit("b").alias("g")),
            ]
        )
        s = spec(halflife=10.0, min_settled_frac=0.5, group="g")
        out = po.ModelBank([s]).fit_predict(df)
        why = reasons(out)
        assert why[40:51] == ["below_min_settled_frac"] * 11
        assert why[51:] == [None] * 9


class TestMaxErrorInflation:
    def test_the_default_is_the_old_floor_on_unit_weights(self):
        """``sqrt(2)`` opens where ``k + 1`` did on an undecayed stream: the
        first prediction is on row ``k + 1`` (rows count from zero), as it
        always was. Under a halflife Kish's ``n`` after ``k + 1`` rows is a
        hair under ``k + 1`` -- the gate sits exactly on its boundary -- so
        it opens one row later."""
        out = po.ModelBank([spec(halflife=math.inf)]).fit_predict(frame(30))
        pred = field(out, "pred_y")
        assert pred[:3].null_count() == 3
        assert pred[3:].null_count() == 0
        assert reasons(out)[:3] == ["above_max_error_inflation"] * 3
        decayed = field(po.ModelBank([spec()]).fit_predict(frame(30)), "pred_y")
        assert decayed[:4].null_count() == 4
        assert decayed[4:].null_count() == 0

    def test_uneven_weights_withhold_for_longer(self):
        """One row carrying a hundred times the weight of the others is not a
        hundred observations: Kish's ``n`` says it is barely one, and the gate
        reads that. ``min_periods`` counted the weight and let it through."""
        df = frame(300).with_columns(
            pl.when(pl.col("t") == 0).then(100.0).otherwise(1.0).alias("w")
        )
        out = po.ModelBank([spec(weight="w", halflife=math.inf)]).fit_predict(df)
        pred = field(out, "pred_y")
        why = reasons(out)
        assert pred[10] is None and pred[50] is None
        assert why[50] == "above_max_error_inflation"
        assert pred[250] is not None
        # n_kish = (99 + n)^2 / (9999 + n) passes edf = 3 near row 75.
        first = next(i for i, p in enumerate(pred.to_list()) if p is not None)
        assert 60 < first < 90, first

    def test_is_a_ratio_above_one_or_off_at_inf(self):
        for bad in (1.0, 0.5, -1.0, math.nan):
            with pytest.raises(ValueError, match="max_error_inflation"):
                spec(max_error_inflation=bad)
        out = po.ModelBank([spec(max_error_inflation=math.inf)]).fit_predict(frame(30))
        assert "above_max_error_inflation" not in set(reasons(out))

    def test_a_tighter_ratio_waits_longer(self):
        loose = field(po.ModelBank([spec()]).fit_predict(frame(200)), "pred_y")
        tight = field(
            po.ModelBank([spec(max_error_inflation=1.05)]).fit_predict(frame(200)), "pred_y"
        )
        # sqrt(1 + 3 / n_kish) < 1.05 needs n_kish > 29.3 (steady state 57.6).
        assert first_present(loose) == 4
        assert 25 <= first_present(tight) <= 40, first_present(tight)

    def test_min_periods_still_floors_when_set(self):
        """The explicit floor is the user's, so it is named ahead of the
        noise gate, whose ratio is infinite until the model has solved."""
        s = spec(min_periods=20.0, halflife=math.inf)
        out = po.ModelBank([s]).fit_predict(frame(60))
        pred = field(out, "pred_y")
        assert pred[:20].null_count() == 20
        assert pred[20:].null_count() == 0
        assert reasons(out)[10] == "below_min_periods"

    def test_settled_takes_precedence_in_the_reason(self):
        s = spec(halflife=10.0, min_settled_frac=0.5)
        out = po.ModelBank([s]).fit_predict(frame(20))
        assert reasons(out)[1] == "below_min_settled_frac"


class TestErrorInflationField:
    def test_is_opt_in_and_per_slot(self):
        default = po.ModelBank([spec()]).fit_predict(frame(30))
        assert "error_inflation_y" not in [f.name for f in default.schema["m"].fields]
        out = po.ModelBank([spec(emit_error_inflation=True)]).fit_predict(frame(300))
        got = field(out, "error_inflation_y")
        assert got.null_count() < 5
        tail = got[100:].to_numpy()
        assert (tail >= 1.0).all()
        # Steady state: sqrt(1 + h) with h around edf / n_kish = 3 / 57.6.
        assert 1.0 < float(np.median(tail)) < 1.1

    def test_needs_a_model_that_has_one(self):
        with pytest.raises(ValueError, match="emit_error_inflation"):
            po.spec.sgd(
                "m",
                targets=["y"],
                features=["x0"],
                halflife=20.0,
                learning_rate=0.01,
                emit_error_inflation=True,
            )


class TestSupportCoef:
    def test_a_duplicated_column_reads_half_and_a_clean_one_reads_one(self):
        df = frame(200, k=2).with_columns(pl.col("x0").alias("x2"))
        s = spec(features=["x0", "x1", "x2"], ridge=1e-8, coef_every=1, halflife=math.inf)
        out = po.ModelBank([s]).fit_predict(df)
        support = field(out, "support_coef")[-1]
        assert support.to_list() == pytest.approx([None, 0.5, 1.0, 0.5], abs=1e-3)

    def test_rides_on_the_coef_schedule(self):
        out = po.ModelBank([spec(coef_every=0)]).fit_predict(frame(50))
        assert field(out, "support_coef").null_count() == 49
        assert field(out, "coef")[-1] is not None
        assert field(out, "support_coef")[-1] is not None

    def test_a_heavy_ridge_reads_low_everywhere(self):
        s = spec(ridge=5.0, coef_every=1, halflife=math.inf)
        out = po.ModelBank([s]).fit_predict(frame(200))
        support = field(out, "support_coef")[-1].to_list()
        assert all(v < 0.3 for v in support[1:]), support


class TestSummaryAndWarnings:
    def test_summary_carries_the_readiness_columns(self):
        bank = po.ModelBank([spec()])
        bank.fit_predict(frame(100))
        s = bank.summary("m")
        for col in (
            "settled_frac",
            "error_inflation",
            "min_support_coef",
            "min_support_coef_feature",
            "n_coef",
        ):
            assert col in s.columns, col
        assert s["settled_frac"][0] == pytest.approx(1.0 - 2.0 ** (-99 / 20.0))
        assert 1.0 < s["error_inflation"][0] < 1.1
        assert s["min_support_coef"][0] == pytest.approx(1.0, abs=1e-3)
        assert s["n_coef"][0] == 3

    def test_a_coefficient_more_ridge_than_data_is_named_once(self):
        df = frame(200, k=2).with_columns(pl.col("x0").alias("x2"))
        s = spec(features=["x0", "x1", "x2"], ridge=1e-8, halflife=math.inf)
        bank = po.ModelBank([s])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit_predict(df[:100])
            bank.fit_predict(df[100:])
        got = [w for w in caught if issubclass(w.category, po.ReadinessWarning)]
        assert len(got) == 1, [str(w.message) for w in caught]
        assert "x2" in str(got[0].message) or "x0" in str(got[0].message)
        assert "support_coef" in str(got[0].message)

    def test_an_unreachable_gate_is_named_with_the_fix(self):
        """Halflife 2 rows tops Kish's ``n`` out near 5.8; with ten features
        ``sqrt(1 + 11 / 5.8)`` never reaches ``sqrt(2)``."""
        k = 10
        df = frame(400, k=k)
        s = spec(features=[f"x{j}" for j in range(k)], halflife=2.0)
        bank = po.ModelBank([s])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = bank.fit_predict(df)
        assert field(out, "pred_y").null_count() == 400
        got = [w for w in caught if issubclass(w.category, po.ReadinessWarning)]
        assert len(got) == 1, [str(w.message) for w in caught]
        msg = str(got[0].message)
        assert "max_error_inflation" in msg and "halflife" in msg

    def test_a_reachable_gate_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            po.ModelBank([spec()]).fit_predict(frame(300))


class TestInvariants:
    def test_the_new_fields_are_chunk_invariant(self):
        df = frame(210, k=2).with_columns(pl.col("x0").alias("x2"))
        s = spec(
            features=["x0", "x1", "x2"],
            coef_every=1,
            emit_error_inflation=True,
            min_settled_frac=0.3,
            halflife=15.0,
        )
        one = po.ModelBank([s]).fit_predict(df)
        bank = po.ModelBank([s])
        many = pl.concat([bank.fit_predict(df[i : i + 30]) for i in range(0, 210, 30)])
        for name in ("settled_frac", "error_inflation_y", "support_coef", "pred_y"):
            assert field(one, name).equals(field(many, name)), name
        assert reasons(one) == reasons(many)

    def test_the_new_fields_survive_the_state_file(self):
        df = frame(300)
        s = spec(coef_every=1, emit_error_inflation=True, min_settled_frac=0.3)
        one = po.ModelBank([s]).fit_predict(df)
        bank = po.ModelBank([s])
        head = bank.fit_predict(df[:100])
        bank = po.ModelBank.load_bytes(bank.save_bytes())
        tail = bank.fit_predict(df[100:])
        two = pl.concat([head, tail])
        for name in ("settled_frac", "error_inflation_y", "support_coef", "pred_y"):
            assert field(one, name).equals(field(two, name)), name
        assert reasons(one) == reasons(two)

    def test_withheld_reason_is_categorical_not_string(self):
        out = po.ModelBank([spec()]).fit_predict(frame(30))
        dtype = (
            out.schema["m"]
            .fields[[f.name for f in out.schema["m"].fields].index("withheld_reason")]
            .dtype
        )
        assert dtype in (pl.Categorical, pl.Enum) or isinstance(dtype, (pl.Categorical, pl.Enum)), (
            dtype
        )
        assert set(reasons(out)) <= REASONS | {None}

    def test_field_order_and_names(self):
        s = spec(coef_every=1, emit_error_inflation=True)
        assert po.spec.output_fields(s) == [
            "pred_y",
            "resid_y",
            "error_inflation_y",
            "n_eff",
            "settled_frac",
            "withheld_reason",
            "coef",
            "support_coef",
        ]

    def test_every_per_row_model_says_how_settled_it_is(self):
        """The two per-row readiness fields ride on every model that writes a
        row; only the state-only models (`marginal`, `rcov`) have no row."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from test_model_registry import MINIMAL, _build

        for name in sorted(MINIMAL):
            fields = po.spec.output_fields(_build(name))
            if name in ("marginal", "rcov"):
                assert "settled_frac" not in fields
            else:
                assert "settled_frac" in fields, name
                assert "withheld_reason" in fields, name
