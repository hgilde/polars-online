"""T-D2: property-based tests over adversarial streams.

Section C of docs/TESTING.md lists edge cases someone thought of. This file
generates them instead: mixed nulls, duplicate and backwards clocks, constant
and collinear features, weight extremes, tiny groups, unusual chunkings — and
asserts the invariants that must hold for *every* model on *every* stream.

Hypothesis shrinks a failure to a minimal reproducing frame, which is the point:
these tests are meant to produce a small counterexample, not just a red mark.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import polars_online as po

MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"ridge": 1.0}),
    ("kalman", {"coef_half_life": 50.0}),
    ("lasso", {"lasso_path": [0.1, 0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
    ("sgd", {"learning_rate": 0.01, "clip_gradient": 1e3}),
    ("pa", {}),
    ("holt", {"features": []}),
]
IDS = [m[0] for m in MODELS]

SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# Values that have historically caused trouble, plus ordinary floats.
_values = st.one_of(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, -0.0, 1e-12, 1e8, -1e8]),
    st.none(),
)
_weights = st.one_of(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.none(),
)


@st.composite
def streams(draw, min_rows=1, max_rows=40, max_groups=3):
    """An adversarial but *valid* stream: the clock is non-decreasing within
    each group (mis-ordered input is its own test, T-E4)."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_groups = draw(st.integers(min_value=1, max_value=max_groups))
    groups = draw(st.lists(st.integers(0, n_groups - 1), min_size=n, max_size=n))

    # per-group non-decreasing clock, with duplicates and long gaps allowed
    clocks, last = [], dict.fromkeys(range(n_groups), 0.0)
    for g in groups:
        step = draw(st.sampled_from([0.0, 0.5, 1.0, 7.0, 1e4]))
        last[g] += step
        clocks.append(last[g])

    cols = {
        "g": [f"g{g}" for g in groups],
        "t": clocks,
        "x0": draw(st.lists(_values, min_size=n, max_size=n)),
        "x1": draw(st.lists(_values, min_size=n, max_size=n)),
        "y0": draw(st.lists(_values, min_size=n, max_size=n)),
        "w": draw(st.lists(_weights, min_size=n, max_size=n)),
    }
    floats = ["t", "x0", "x1", "y0", "w"]
    return pl.DataFrame(cols, schema_overrides={c: pl.Float64 for c in floats})


#: The late-row minimum the stepping-back streams are run under: a step back
#: larger than it starts the model over, one no larger is refused.
MINIMUM = 5.0


@st.composite
def stepping_back_streams(draw, min_rows=2, max_rows=40, max_groups=3):
    """`streams`, with each group's clock stepping back now and then -- by more
    than `MINIMUM`, so that under `"reset_state"` every step back starts the
    group over and none is refused (task 120)."""
    df = draw(streams(min_rows=min_rows, max_rows=max_rows, max_groups=max_groups))
    clocks, last = [], {}
    for g in df["g"].to_list():
        step = draw(st.sampled_from([0.0, 0.5, 1.0, 7.0, 1e4, -(MINIMUM + 2.0), -1e4]))
        last[g] = last.get(g, 0.0) + step if g in last else 0.0
        clocks.append(last[g])
    return df.with_columns(t=pl.Series(clocks, dtype=pl.Float64))


def build(model, extra, **kw):
    opts = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        half_life=20.0,
        weight="w",
        group="g",
        min_weight=2.0,
    )
    opts.update(extra)
    opts.update(kw)
    return getattr(po.spec, model)("m", **opts)


def binarize(df, model):
    """ftrl needs a 0/1 target; keep nulls so the null policy is still exercised."""
    if model != "ftrl":
        return df
    return df.with_columns(
        y0=pl.when(pl.col("y0").is_null()).then(None).otherwise((pl.col("y0") > 0).cast(pl.Float64))
    )


def unnested(out):
    keep = [
        c for c in out.select("m").unnest("m").columns if not c.startswith(("coef", "support_coef"))
    ]
    return out.select("m").unnest("m").select(keep)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestUniversalProperties:
    @SETTINGS
    @given(df=streams(), chunk=st.integers(min_value=1, max_value=13))
    def test_chunking_never_changes_the_output(self, model, extra, df, chunk):
        df = binarize(df, model)
        spec = build(model, extra)
        one = unnested(po.ModelBank([spec]).fit_predict(df))
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
        many = unnested(pl.concat(parts))
        assert one.equals(many, null_equal=True)

    @SETTINGS
    @given(df=stepping_back_streams(), chunk=st.integers(min_value=1, max_value=13))
    def test_chunking_never_changes_the_output_when_the_clock_steps_back(
        self, model, extra, df, chunk
    ):
        """Hard rule 3 with the clock stepping back: under `"reset_state"` each
        step back past the minimum starts its group over, and where the chunks
        fall cannot move a number."""
        df = binarize(df, model)
        spec = build(model, extra, restart_after_step_back=MINIMUM)
        one = unnested(po.ModelBank([spec]).fit_predict(df))
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
        assert one.equals(unnested(pl.concat(parts)), null_equal=True)

    @SETTINGS
    @given(df=streams(min_rows=2), chunk=st.integers(min_value=1, max_value=13), data=st.data())
    def test_a_late_row_is_refused_at_its_input_row_whatever_the_chunking(
        self, model, extra, df, chunk, data
    ):
        """A row stepped back by no more than the minimum is a late row: the
        stream is refused there, naming the row's place in the input, whether
        the row before it in its group is in the same chunk or in the state."""
        groups = df["g"].to_list()
        later = [i for i in range(1, df.height) if groups[i] in groups[:i]]
        assume(later)
        i = data.draw(st.sampled_from(later), label="late row")
        prev = max(j for j in range(i) if groups[j] == groups[i])
        t = df["t"].to_list()
        t[i] = t[prev] - 2.0
        # Every later row of its group stays after it. The late row and the
        # one before it are ordinary rows, so the null policy skips neither
        # and the step back is measured between the two.
        ordinary = pl.int_range(pl.len()).is_in([prev, i])
        df = df.with_columns(
            t=pl.Series(t, dtype=pl.Float64),
            x0=pl.when(ordinary).then(1.0).otherwise("x0"),
            x1=pl.when(ordinary).then(1.0).otherwise("x1"),
            w=pl.when(ordinary).then(1.0).otherwise("w"),
        )
        df = binarize(df, model)
        spec = build(model, extra, restart_after_step_back=MINIMUM)
        says = f"goes backwards by 2 at row {i}, no more than restart_after_step_back = 5"
        with pytest.raises(ValueError, match=says):
            po.ModelBank([spec]).fit_predict(df)
        chunks = [df.slice(k, chunk) for k in range(0, df.height, chunk)]
        with pytest.raises(ValueError, match=says):
            list(po.ModelBank([spec]).fit_predict_batches(iter(chunks)))

    @SETTINGS
    @given(df=streams(min_rows=2), split=st.integers(min_value=1, max_value=39))
    def test_save_load_is_transparent(self, model, extra, df, split):
        assume(split < df.height)
        df = binarize(df, model)
        spec = build(model, extra)
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, split))
        b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
        rest = df.slice(split, df.height - split)
        assert unnested(a.fit_predict(rest)).equals(unnested(b.fit_predict(rest)), null_equal=True)

    @SETTINGS
    @given(df=streams())
    def test_outputs_are_finite_or_null(self, model, extra, df):
        df = binarize(df, model)
        out = po.ModelBank([build(model, extra)]).fit_predict(df)
        for f in out.schema["m"].fields:
            # `coef`/`support_coef` are lists; `withheld_reason` is an enum.
            if f.name.startswith(("coef", "support_coef")) or not f.dtype.is_float():
                continue
            vals = np.array(
                [v for v in out["m"].struct.field(f.name).to_list() if v is not None],
                dtype=float,
            )
            assert np.isfinite(vals).all(), f"{f.name} produced a non-finite value"

    @SETTINGS
    @given(df=streams())
    def test_feature_or_weight_null_means_all_outputs_null(self, model, extra, df):
        """A row with a null feature or weight is skipped, and every field of
        its record is null, not `weight_sum` alone (review 2026-10-05, TA10)."""
        df = binarize(df, model)
        spec = build(model, extra)
        out = po.ModelBank([spec]).fit_predict(df)
        # Only the columns this spec actually declares can skip a row -- holt
        # reads no features, so an unused null column must not disturb it.
        cond = pl.col("w").is_null()
        for f in spec["features"]:
            cond = cond | pl.col(f).is_null()
        skipped = df.select(cond).to_series().to_list()
        fields = out.select("m").unnest("m")
        for i, skip in enumerate(skipped):
            if skip:
                present = [c for c in fields.columns if fields[c][i] is not None]
                assert not present, f"row {i} was skipped but reported {present}"

    @SETTINGS
    @given(df=streams(max_groups=3))
    def test_groups_are_independent(self, model, extra, df):
        df = binarize(df, model)
        keys = df["g"].unique().to_list()
        assume(len(keys) > 1)
        spec = build(model, extra)
        both = po.ModelBank([spec]).fit_predict(df)
        for key in keys:
            solo = po.ModelBank([spec]).fit_predict(df.filter(pl.col("g") == key))
            a = unnested(both.filter(pl.col("g") == key))
            b = unnested(solo)
            assert a.equals(b, null_equal=True), f"group {key} was affected by the others"

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_prediction_never_depends_on_the_current_target(self, model, extra):
        """Out-of-sample by construction (docs/PLAN.md hard rule 2): changing a
        row's target must not change that row's own prediction.

        The row is drawn from those the model scored and whose target is
        there. The first such target is no use: every model withholds its
        first row, so a test that perturbed it compared two nulls in every
        stream (review 2026-10-05, TA1). The count at the end says how many
        streams compared a prediction that was there.

        `ewridge`'s error-inflation gate is switched off. On these short,
        mostly-null streams it withholds all but 13 in 100 of them, so
        Hypothesis would discard most streams. The gate only withholds, and
        the comparison still fails if the perturbed run withholds a row the
        base run scored."""
        compared = []
        kw = {"max_error_inflation": float("inf")} if model == "ewridge" else {}

        @SETTINGS
        @given(df=streams(), data=st.data())
        def check(df, data):
            df = binarize(df, model)
            spec = build(model, extra, **kw)
            base = po.ModelBank([spec]).fit_predict(df)
            field = next(f.name for f in base.schema["m"].fields if f.name.startswith("pred_"))
            preds = base["m"].struct.field(field).to_list()
            y = df["y0"].to_list()
            scored = [i for i in range(df.height) if preds[i] is not None and y[i] is not None]
            assume(scored)
            idx = data.draw(st.sampled_from(scored), label="scored row")
            y[idx] = (0.0 if y[idx] else 1.0) if model == "ftrl" else y[idx] + 12345.0
            perturbed = po.ModelBank([spec]).fit_predict(
                df.with_columns(y0=pl.Series(y, dtype=pl.Float64))
            )
            a = preds[idx]
            b = perturbed["m"].struct.field(field).to_list()[idx]
            assert a == b, f"row {idx}: changing its own target changed its prediction ({a} -> {b})"
            compared.append(idx)

        check()
        assert len(compared) >= 10, (
            f"only {len(compared)} streams compared a prediction that was there"
        )
