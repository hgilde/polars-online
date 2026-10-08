"""Not using a model before it is ready (docs/WARMUP-AND-CONVERGENCE.md).

Two gates, both stated as intent rather than as a number that needs a formula
in the user's head:

- ``min_settled_frac`` -- how far the decay window has filled toward steady
  state, ``settled_frac = 1 - 2 ** (-T / half_life)`` with ``T`` the decay time
  the models have actually seen. Off by default: under a stationary process a
  mean-form fit is unbiased from its first row, so what this guards is a
  history that does not represent the process, which only the user can judge.
- ``max_error_inflation`` -- how much estimation error is expected to inflate
  a prediction's error over the noise floor, ``sqrt(1 + edf / n_kish)``.
  Default ``sqrt(2)``: withhold while the estimation variance exceeds the
  noise being fitted. It replaces the ``k + 1`` floor ``min_weight`` gave
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
from test_every_kind import frame_for
from test_model_registry import MINIMAL

REASONS = {"below_min_settled_frac", "above_max_error_inflation", "below_min_weight"}


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
    base = dict(targets=["y"], features=["x0", "x1"], half_life=20.0, max_rows_between_solves=1)
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
        out = po.ModelBank([spec(half_life=10.0)]).fit_predict(frame(60))
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
        """Ten, fifty or a hundred rows per half-life: the fraction is the
        same at one, two and three half-lives, because it is the *clock* the
        decay has covered, not a count of anything (§5.2)."""
        for rows_per_halflife in (10, 50, 100):
            n = 4 * rows_per_halflife
            df = frame(n).with_columns((pl.col("t") * (50.0 / rows_per_halflife)).alias("t"))
            s = spec(clock="t", half_life=50.0, gap_cap=1e9)
            got = field(po.ModelBank([s]).fit_predict(df), "settled_frac")
            for half_lives in (1, 2, 3):
                # One row on: the first row's delta is zero.
                assert got[half_lives * rows_per_halflife + 1] == pytest.approx(
                    1.0 - 2.0**-half_lives
                ), rows_per_halflife

    def test_is_null_without_decay(self):
        out = po.ModelBank([spec(half_life=math.inf)]).fit_predict(frame(30))
        assert field(out, "settled_frac").null_count() == 30

    def test_counts_a_capped_gap_as_the_cap(self):
        """A weekend capped to ``gap_cap`` warms the model by
        ``gap_cap``, since that is what it decayed by."""
        df = frame(40).with_columns(
            pl.when(pl.col("t") >= 20).then(pl.col("t") + 1e6).otherwise(pl.col("t")).alias("t")
        )
        s = spec(clock="t", half_life=10.0, gap_cap=5.0)
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
        s = spec(
            clock="t",
            half_life=10.0,
            gap_cap=5.0,
            restart_after_step_back=0.0,
        )
        got = field(po.ModelBank([s]).fit_predict(df), "settled_frac")
        # The reset row is a first row again: it decays by nothing.
        assert got[20] == 0.0 and got[21] == 0.0
        assert got[22] == pytest.approx(1.0 - 2.0 ** (-1 / 10.0))


class TestMinSettledFrac:
    def test_withholds_until_the_fraction_and_says_why(self):
        s = spec(half_life=10.0, min_settled_frac=0.5)
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
            spec(half_life=math.inf, min_settled_frac=0.5)

    def test_predict_withholds_the_same_way(self):
        s = spec(half_life=10.0, min_settled_frac=0.5)
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
        s = spec(half_life=10.0, min_settled_frac=0.5, group="g")
        out = po.ModelBank([s]).fit_predict(df)
        why = reasons(out)
        assert why[40:51] == ["below_min_settled_frac"] * 11
        assert why[51:] == [None] * 9


class TestMaxErrorInflation:
    def test_the_default_is_the_old_floor_on_unit_weights(self):
        """``sqrt(2)`` opens where ``k + 1`` did on an undecayed stream: the
        first prediction is on row ``k + 1`` (rows count from zero), as it
        always was. Under a half-life Kish's ``n`` after ``k + 1`` rows is a
        hair under ``k + 1`` -- the gate sits exactly on its boundary -- so
        it opens one row later."""
        out = po.ModelBank([spec(half_life=math.inf)]).fit_predict(frame(30))
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
        reads that. ``min_weight`` counted the weight and let it through."""
        df = frame(300).with_columns(
            pl.when(pl.col("t") == 0).then(100.0).otherwise(1.0).alias("w")
        )
        out = po.ModelBank([spec(weight="w", half_life=math.inf)]).fit_predict(df)
        pred = field(out, "pred_y")
        why = reasons(out)
        assert pred[10] is None and pred[50] is None
        assert why[50] == "above_max_error_inflation"
        assert pred[250] is not None
        # n_kish = (99 + n)^2 / (9999 + n) passes edf = 3 near row 75.
        first = next(i for i, p in enumerate(pred.to_list()) if p is not None)
        assert 60 < first < 90, first

    def test_is_refused_by_name_where_no_ridge_system_reads_it(self):
        """A model without a noise statistic to read the gate from took the
        setting and dropped it (docs/PLAN.md task 109). Since task 116 the
        gate reads `rls`, `kalman` and `lasso` too
        (`tests/test_readiness_models.py`)."""
        for build, kw in [
            (po.spec.sgd, dict(learning_rate=0.01)),
            (po.spec.huber, {}),
        ]:
            with pytest.raises(ValueError, match="max_error_inflation needs a model with"):
                build(
                    "m",
                    targets=["y"],
                    features=["x0"],
                    half_life=10.0,
                    max_error_inflation=2.0,
                    **kw,
                )

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
        s = spec(min_weight=20.0, half_life=math.inf)
        out = po.ModelBank([s]).fit_predict(frame(60))
        pred = field(out, "pred_y")
        assert pred[:20].null_count() == 20
        assert pred[20:].null_count() == 0
        assert reasons(out)[10] == "below_min_weight"

    def test_settled_takes_precedence_in_the_reason(self):
        s = spec(half_life=10.0, min_settled_frac=0.5)
        out = po.ModelBank([s]).fit_predict(frame(20))
        assert reasons(out)[1] == "below_min_settled_frac"

    def test_min_weight_is_named_ahead_of_the_noise_gate_where_both_bind(self):
        """The precedence `docs/OUTPUTS.md` states, settled > `min_weight` >
        inflation, where the last two bind together. With no floor set, rows
        0 to 2 are withheld by the noise gate alone: its ratio is infinite
        until the model has solved. With `min_weight = 20` the same rows are
        withheld by both, and the floor is named; the floor alone holds the
        rows after them (review 2026-10-06, TB8: only row 10, where the
        noise gate had opened, was asserted)."""
        df = frame(60)
        alone = po.ModelBank([spec(half_life=math.inf)]).fit_predict(df)
        assert reasons(alone)[:4] == ["above_max_error_inflation"] * 3 + [None]
        both = po.ModelBank([spec(half_life=math.inf, min_weight=20.0)]).fit_predict(df)
        assert reasons(both)[:21] == ["below_min_weight"] * 20 + [None]


#: The kinds that write a row: `marginal` and `rcov` report through the
#: state alone.
ROW_KINDS = sorted(set(MINIMAL) - {"marginal", "rcov"})


@pytest.mark.parametrize("name", ROW_KINDS)
def test_every_kind_settles_by_the_formula_and_withholds_until_it(name):
    """`settled_frac` is ``1 - 2 ** (-T / half_life)`` on every kind that
    writes a row, ``T = i - 1`` on a row-count clock (a group's first row
    decays by nothing), and ``min_settled_frac = 0.5`` withholds rows 0 to
    10, naming that gate, with the kind's first field null there, and no row
    after them for that reason. `seqtest` and `bocpd` take no half-life and
    are refused by name: an e-process does not forget, and a run-length
    posterior is what forgets. It was value-tested on `ewridge` alone, the
    rest for the fields' presence (review 2026-10-06, TB7)."""
    kw: dict[str, object] = dict(
        targets=["y"], features=["x0", "x1"], half_life=10.0, min_settled_frac=0.5
    )
    kw |= {k: v for k, v in MINIMAL[name].items() if k != "half_life"}
    if name == "corrchange":
        kw["scalar"] = True  # its half-life parametrises the scalar form's standardiser

    def build() -> dict:
        return getattr(po.spec, name)("m", **{k: v for k, v in kw.items() if v is not None})

    if name in ("seqtest", "bocpd"):
        with pytest.raises(ValueError, match=f"half_life/lam do not apply to {name}"):
            build()
        return
    out = po.ModelBank([build()]).fit_predict(frame_for(name))["m"].struct.unnest()
    got = out["settled_frac"].to_list()
    assert got == pytest.approx(
        [1.0 - 2.0 ** (-max(i - 1, 0) / 10.0) for i in range(len(got))], abs=1e-12
    )
    why = out["withheld_reason"].cast(pl.String).to_list()
    assert why[:11] == ["below_min_settled_frac"] * 11
    assert "below_min_settled_frac" not in why[11:]
    skip = ("weight_sum", "settled_frac", "withheld_reason", "coef", "support_coef")
    first = next(c for c in out.columns if c not in skip)
    assert out[first][:11].null_count() == 11, first


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
                half_life=20.0,
                learning_rate=0.01,
                emit_error_inflation=True,
            )


class TestSupportCoef:
    def test_a_duplicated_column_reads_half_and_a_clean_one_reads_one(self):
        df = frame(200, k=2).with_columns(pl.col("x0").alias("x2"))
        s = spec(features=["x0", "x1", "x2"], ridge=1e-8, coef_every=1, half_life=math.inf)
        out = po.ModelBank([s]).fit_predict(df)
        support = field(out, "support_coef")[-1]
        assert support.to_list() == pytest.approx([None, 0.5, 1.0, 0.5], abs=1e-3)

    def test_rides_on_the_coef_schedule(self):
        # Unset, `coef` is on the chunk's last row alone.
        out = po.ModelBank([spec()]).fit_predict(frame(50))
        assert field(out, "support_coef").null_count() == 49
        assert field(out, "coef")[-1] is not None
        assert field(out, "support_coef")[-1] is not None

    def test_a_heavy_ridge_reads_low_everywhere(self):
        s = spec(ridge=5.0, coef_every=1, half_life=math.inf)
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
        s = spec(features=["x0", "x1", "x2"], ridge=1e-8, half_life=math.inf)
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
        """Half-life 2 rows tops Kish's ``n`` out near 5.8; with ten features
        ``sqrt(1 + 11 / 5.8)`` never reaches ``sqrt(2)``."""
        k = 10
        df = frame(400, k=k)
        s = spec(features=[f"x{j}" for j in range(k)], half_life=2.0)
        bank = po.ModelBank([s])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = bank.fit_predict(df)
        assert field(out, "pred_y").null_count() == 400
        got = [w for w in caught if issubclass(w.category, po.ReadinessWarning)]
        assert len(got) == 1, [str(w.message) for w in caught]
        msg = str(got[0].message)
        assert "max_error_inflation" in msg and "half_life" in msg

    def test_an_ordinary_spec_does_not_warn_while_the_gates_withhold(self):
        """The first solve of any spec is under-determined by construction --
        one row against `k` slopes -- and the warning cannot be retracted, so
        it is raised only where the gates let the row's prediction through.
        Found verifying the shipped 0.9.0 wheel: an ordinary two-feature fit
        warned at `weight_sum = 1.00` ("0.27 data and 0.73 ridge") and read
        `support_coef = 1.00` from the next row to the end of the stream."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            po.ModelBank([spec(coef_every=1)]).fit_predict(frame(60))
            # `min_weight = 0` puts the first solve at `weight_sum = 0`, where the
            # fit is entirely the ridge -- what `tests/test_window_budget.py`
            # sets, and what made that file warn in CI.
            po.ModelBank([spec(coef_every=1, min_weight=0.0)]).fit_predict(frame(60))

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
            half_life=15.0,
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

    @pytest.mark.parametrize("limit", [1.05, math.sqrt(2), 3.0])
    def test_the_gate_decides_alike_whether_the_shares_are_read_or_not(self, limit):
        """A solve's data shares and ``edf`` are taken only when something
        reads them (docs/PLAN.md task 140): a ``coef`` row, or the noise gate
        where the bound ``edf <= 1 + k`` cannot decide. Read on every row or
        only at each chunk's end, every field is the same, on both sides of
        the gate and with a cadence that leaves solves unread across rows."""
        df = frame(400, k=6)
        kw = dict(
            features=[f"x{j}" for j in range(6)],
            half_life=60.0,
            max_rows_between_solves=4,
            max_error_inflation=limit,
            emit_error_inflation=True,
        )

        def run(**cadence: int) -> pl.DataFrame:
            bank = po.ModelBank([spec(**cadence, **kw)])
            return pl.concat([bank.fit_predict(df[i : i + 100]) for i in range(0, 400, 100)])

        # Every row (`coef_every = 0`), and unset: each chunk's last row.
        read, unread = run(coef_every=0), run()
        for name in ("pred_y", "resid_y", "error_inflation_y", "weight_sum", "settled_frac"):
            assert field(read, name).equals(field(unread, name)), name
        assert reasons(read) == reasons(unread)
        # Each chunk's last row reads the shares either way.
        ends = [99, 199, 299, 399]
        for name in ("coef", "support_coef"):
            assert field(read, name).gather(ends).equals(field(unread, name).gather(ends)), name
        why = reasons(read)
        assert "above_max_error_inflation" in why and None in why, set(why)

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
            "weight_sum",
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


def _regular(n: int, d: float, weight: float | None, seed: int = 3) -> pl.DataFrame:
    """Rows ``d`` clock units apart, each of one weight where there is a weight
    column: the regular stream ``weight_sum_settled`` is exact on."""
    df = frame(n, k=2, seed=seed).with_columns(pl.col("t") * d)
    return df.with_columns(w=pl.lit(weight)) if weight is not None else df


def _last_reason(out: pl.DataFrame) -> str | None:
    return reasons(out)[-1]


class TestSettledWeight:
    """What task 116 builds that moves no default (docs/PLAN.md task 198, D8):
    ``weight_sum_settled`` in the summary, the warning where a ``min_weight`` can
    never be met, and the half-life figure in the noise gate's notice. The
    oracle is the definition: the ceiling ``w / (1 - 2 ** (-d / h))`` of a
    regular stream, docs/WARMUP-AND-CONVERGENCE.md §5.1."""

    @pytest.mark.parametrize(
        ("d", "h", "weight"),
        [(1.0, 20.0, None), (0.5, 7.0, None), (1.0, 5.0, 2.5), (3.0, 40.0, 0.25)],
    )
    def test_weight_sum_settled_is_the_ceiling_from_the_second_row(self, d, h, weight):
        df = _regular(60, d, weight)
        kw = {"weight": "w"} if weight is not None else {}
        s = spec(clock="t", gap_cap=10.0 * d, half_life=h, **kw)
        ceiling = (weight or 1.0) / (1.0 - 2.0 ** (-d / h))
        for rows in (1, 2, 3, 17, 60):
            bank = po.ModelBank([s])
            bank.fit_predict(df[:rows])
            got = bank.summary("m")["weight_sum_settled"][0]
            if rows == 1:
                assert got is None, "one row has brought no clock"
            else:
                assert got == pytest.approx(ceiling, rel=1e-12), rows

    def test_weight_sum_settled_is_null_without_a_decay_or_under_a_window(self):
        df = frame(50)
        for s in (spec(half_life=math.inf), spec(window_size=20.0)):
            bank = po.ModelBank([s])
            bank.fit_predict(df)
            assert bank.summary("m")["weight_sum_settled"][0] is None

    def test_an_unreachable_min_weight_is_named_with_its_ceiling(self):
        """Half-life 5 rows: every target's weight tops out at
        ``1 / (1 - 2 ** (-1 / 5)) = 7.73``, so a ``min_weight`` of 20 withholds
        every prediction for good, and says so once the stream is 95% settled
        -- once, however many chunks follow."""
        ceiling = 1.0 / (1.0 - 2.0 ** (-1.0 / 5.0))
        df = frame(200)
        s = po.spec.rls("m", targets=["y"], features=["x0", "x1"], half_life=5.0, min_weight=20.0)
        bank = po.ModelBank([s])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = bank.fit_predict(df[:100])
            bank.fit_predict(df[100:])
        assert field(out, "pred_y").null_count() == 100
        got = [w for w in caught if issubclass(w.category, po.ReadinessWarning)]
        assert len(got) == 1, [str(w.message) for w in caught]
        msg = str(got[0].message)
        assert "min_weight = 20" in msg and '"y"' in msg and "half_life" in msg, msg
        assert f"{ceiling:.4}" in msg, (ceiling, msg)
        assert bank.summary("m")["weight_sum_settled"][0] == pytest.approx(ceiling, rel=1e-12)

    def test_a_min_weight_between_the_settled_weight_and_the_ceiling_does_not_warn(self):
        """At 95% settled the weight is 95% of the ceiling, so a floor between
        the two is met later: no warning, and predictions once it is."""
        ceiling = 1.0 / (1.0 - 2.0 ** (-1.0 / 5.0))
        s = po.spec.rls(
            "m", targets=["y"], features=["x0", "x1"], half_life=5.0, min_weight=0.99 * ceiling
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            out = po.ModelBank([s]).fit_predict(frame(200))
        assert field(out, "pred_y")[-1] is not None

    @pytest.mark.parametrize("kind", ["ewridge", "sgd"])
    def test_under_an_embargo_a_reachable_min_weight_does_not_warn(self, kind):
        """Review round 5 (C1): the row's ``settled_frac`` counts the clock
        the held rows have covered (§8), and the notice paired it with a
        weight that had neither decayed by nor accumulated those rows, so
        the ceiling read negative before anything was learned, and a floor
        of 3 was said to be unreachable at row 9 -- then 374 of 400 rows
        were predicted. The notices read the learned rows' clock, the one
        the weight has settled on, as ``weight_sum_settled`` does: at a
        half-life of 2 the ceiling is 3.4142, above the floor, so no notice,
        and the predictions arrive once the held rows are released. The
        row's field keeps counting the held rows."""
        n, h, embargo = 400, 2.0, 20.0
        ceiling = 1.0 / (1.0 - 2.0 ** (-1.0 / h))
        kw: dict = dict(
            targets=["y"],
            features=["x0"],
            clock="t",
            gap_cap=1e9,
            half_life=h,
            min_weight=3.0,
            embargo=embargo,
        )
        if kind == "sgd":
            kw["learning_rate"] = 0.01
        bank = po.ModelBank([getattr(po.spec, kind)("m", **kw)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            out = bank.fit_predict(frame(n))
        preds = field(out, "pred_y")
        # Row i releases rows 0..i-20, and the floor needs seven of them:
        # (1 - lam^7)/(1 - lam) is 3.11 at lam = 2^(-1/2), where six give 2.99.
        assert first_present(preds) == 26, first_present(preds)
        assert preds.tail(n - 30).null_count() == 0
        assert bank.summary("m")["weight_sum_settled"][0] == pytest.approx(ceiling, rel=1e-12)
        # The field counts the held rows' clock too (§8): before row 10 the
        # stream has covered 9 clock units, none of them learned yet.
        assert field(out, "settled_frac")[10] == pytest.approx(1.0 - 2.0 ** (-9.0 / h))

    @pytest.mark.parametrize(
        ("k", "h", "limit"),
        [
            # Ten features at a half-life of 2 rows: the weight tops out near
            # 3.4, below the 11 the model needs before it solves, so the ratio
            # is infinite and the figure is the weight's.
            (10, 2.0, None),
            # Two features at 3 rows solve, and read about 1.16 against a
            # limit of 1.1: the figure is Kish's size's.
            (2, 3.0, 1.1),
        ],
    )
    def test_the_unreachable_noise_gate_names_the_half_life_that_opens_it(self, k, h, limit):
        """The figure is the steady state's: a half-life a little above it
        opens the gate, and one a little below it does not."""
        df = frame(1200, k=k)
        features = [f"x{j}" for j in range(k)]
        kw = {} if limit is None else {"max_error_inflation": limit}

        def run(h: float) -> tuple[pl.DataFrame, list[str]]:
            bank = po.ModelBank([spec(features=features, half_life=h, **kw)])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = bank.fit_predict(df)
            msgs = [str(w.message) for w in caught if issubclass(w.category, po.ReadinessWarning)]
            return out, msgs

        out, msgs = run(h)
        assert _last_reason(out) == "above_max_error_inflation"
        assert len(msgs) == 1, msgs
        figure = float(msgs[0].split("aise the half_life above ")[1].split()[0])
        assert figure > h, msgs[0]
        opened, quiet = run(1.05 * figure)
        assert _last_reason(opened) is None, figure
        assert not quiet, quiet
        shut, _ = run(0.95 * figure)
        assert _last_reason(shut) == "above_max_error_inflation", figure
