"""T-D2: property-based tests over adversarial streams.

Section C of docs/TESTING.md lists edge cases someone thought of. This file
generates them instead: mixed nulls, duplicate and backwards clocks, constant
and collinear features, weight extremes, tiny groups, unusual chunkings — and
asserts the invariants that must hold for *every* model on *every* stream.
The values reach the input bound (``1e100``, as weights too, beside
``1e-100``) and past it (NaN and the infinities, which the bank reads as
missing); every regression runs under each switch it takes; and the kinds
that are not regressions run on a stream of their own (review 2026-10-06,
TA3: none of that was drawn).

Hypothesis shrinks a failure to a minimal reproducing frame, which is the point:
these tests are meant to produce a small counterexample, not just a red mark.
"""

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import polars_online as po
from test_model_registry import MINIMAL, REGRESSIONS, _build

#: The bank's bound on a value it learns from (`online_core::INPUT_BOUND`):
#: a magnitude past it, NaN and the infinities are missing, as a null is.
INPUT_BOUND = 1e100

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

# Values that have historically caused trouble, plus ordinary floats: the
# input bound on both sides, and what lies past it.
_values = st.one_of(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, -0.0, 1e-12, 1e8, -1e8]),
    st.sampled_from([INPUT_BOUND, -INPUT_BOUND, float("nan"), float("inf"), float("-inf")]),
    st.none(),
)
_weights = st.one_of(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([1e-100, INPUT_BOUND, float("nan"), float("inf")]),
    st.none(),
)


def _missing(col: str) -> pl.Expr:
    """Where the bank reads ``col`` as missing: null, NaN, infinite or past
    the bound (`stream.rs::usable`)."""
    c = pl.col(col)
    return c.is_null() | ~c.is_finite() | (c.abs() > INPUT_BOUND)


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
        # A session that changes now and then, for the switch that reads it.
        "s": [f"s{k}" for k in np.cumsum(draw(st.lists(st.booleans(), min_size=n, max_size=n)))],
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
    """ftrl needs a 0/1 target; keep the missing ones missing, so the null
    policy is still exercised."""
    if model != "ftrl":
        return df
    return df.with_columns(
        y0=pl.when(_missing("y0")).then(None).otherwise((pl.col("y0") > 0).cast(pl.Float64))
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
        # reads no features, so an unused null column must not disturb it. A
        # value the bank reads as missing skips a row as a null does.
        cond = _missing("w")
        for f in spec["features"]:
            cond = cond | _missing(f)
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


#: Each switch a regression may take, with what it needs beside it. A
#: blocked Gram is `test_semantics_all_models`'s variant: a block of 5 that
#: does not divide the solve cadence, so a solve merges a partial block.
SWITCHES: dict[str, dict] = {
    "plain": {},
    "window": {"window_size": 15.0},
    "blocked gram": {"solve_every": 4.0, "max_rows_between_solves": 8, "gram_block_rows": 5},
    "pairwise gaps": {"target_gaps": "pairwise"},
    "no intercept": {"fit_intercept": False},
    "standardize": {"standardize": True},
    "embargo": {"embargo": 3.0},
    "session": {"session": "s", "session_gap": 2.0},
}


def _takes(model: str, extra: dict, switch: dict) -> bool:
    """Whether the builder takes ``switch`` for ``model``: probed, so each
    model draws only the switches it accepts and no case is a refusal."""
    try:
        build(model, extra, **switch)
    except (TypeError, ValueError):
        return False
    return True


#: Per model, the switches it takes, probed from the builders.
SWITCHES_FOR = {
    model: [name for name, sw in SWITCHES.items() if _takes(model, extra, sw)]
    for model, extra in MODELS
}


def test_every_switch_reaches_a_model_and_every_model_a_switch():
    """The probe is no false skip: each switch is taken by some model, and
    each model takes one besides the plain spec (task 189)."""
    for name in SWITCHES:
        assert any(name in got for got in SWITCHES_FOR.values()), name
    for model, got in SWITCHES_FOR.items():
        assert len(got) > 1, (model, got)
    assert {m for m, _ in MODELS} == set(REGRESSIONS)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestEverySwitch:
    """The universal properties under each switch a model takes: one chunk
    or many give the same numbers, a save and load at any row is
    transparent, every number is finite or null, and a skipped row reports
    nothing (review 2026-10-06, TA3)."""

    @SETTINGS
    @given(df=streams(min_rows=2), chunk=st.integers(min_value=1, max_value=13), data=st.data())
    def test_the_properties_hold_under_each_switch(self, model, extra, df, chunk, data):
        switch = data.draw(st.sampled_from(SWITCHES_FOR[model]), label="switch")
        split = data.draw(st.integers(1, df.height - 1), label="split")
        df = binarize(df, model)
        spec = build(model, extra, **SWITCHES[switch])
        one = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
        assert unnested(one).equals(unnested(many), null_equal=True), "chunking moved a number"
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, split))
        b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
        rest = df.slice(split)
        assert unnested(a.fit_predict(rest)).equals(unnested(b.fit_predict(rest)), null_equal=True)
        fields = one.select("m").unnest("m")
        _finite_or_null(fields)
        cond = _missing("w")
        for f in spec["features"]:
            cond = cond | _missing(f)
        for i, skip in enumerate(df.select(cond).to_series().to_list()):
            if skip:
                present = [c for c in fields.columns if fields[c][i] is not None]
                assert not present, f"row {i} was skipped but reported {present}"


def _finite_or_null(fields: pl.DataFrame) -> None:
    for name, dtype in fields.schema.items():
        if name.startswith(("coef", "support_coef")) or not dtype.is_float():
            continue
        vals = np.array([v for v in fields[name].to_list() if v is not None], dtype=float)
        assert np.isfinite(vals).all(), f"{name} produced a non-finite value"


#: The kinds that are not regressions, from the registry: a new one is drawn
#: the day it is registered.
OTHER_KINDS = sorted(set(MINIMAL) - set(REGRESSIONS))

#: What a kind needs beside `MINIMAL`'s arguments on a stream of at most 60
#: rows: a short warm-up, so `kmeans` and `hmm` seed and report, and a
#: short span, so `corrchange` closes some. Under `MINIMAL`'s settings they
#: reported a number in 0, 1 and 6 of 60 such streams.
KIND_ARGS: dict[str, dict] = {
    "kmeans": {"warm_rows": 4},
    "hmm": {"warm_rows": 4},
    "corrchange": {"span_rows": 8},
}


def _kind_spec(kind: str) -> dict:
    return {**_build(kind), "model": {**_build(kind)["model"], **KIND_ARGS.get(kind, {})}}


def _state(bank: po.ModelBank, kind: str) -> pl.DataFrame | None:
    """What a kind that reports through its state reports: `marginal`'s
    pairs and `rcov`'s closed blocks."""
    if kind == "marginal":
        return bank.marginal("m")
    if kind == "rcov":
        return bank.closed_groups("m", drop=False)
    return None


@st.composite
def kind_streams(draw, kind: str, min_rows=2, max_rows=60):
    """A stream any kind runs on, adversarial as `streams` is: features
    ``x0`` and ``x1`` and a target ``y`` reaching the bound and past it, with
    nulls; a group key ``g`` that only grows, for `rcov`'s monotone close;
    and, for `ew_class`, ``y`` as its label, ``a`` or ``b``, null where the
    number was missing."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    df = pl.DataFrame(
        {
            "x0": draw(st.lists(_values, min_size=n, max_size=n)),
            "x1": draw(st.lists(_values, min_size=n, max_size=n)),
            "y": draw(st.lists(_values, min_size=n, max_size=n)),
            "g": np.cumsum(draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))),
        },
        schema_overrides={c: pl.Float64 for c in ("x0", "x1", "y")},
    )
    if kind == "ew_class":
        df = df.with_columns(
            y=pl.when(_missing("y"))
            .then(None)
            .otherwise(pl.when(pl.col("y") > 0).then(pl.lit("a")).otherwise(pl.lit("b")))
        )
    return df


@pytest.mark.parametrize("kind", OTHER_KINDS)
class TestEveryOtherKind:
    """The universal properties on the eleven kinds the sweeps above leave
    out, each as `MINIMAL` builds it, which had a fixed-stream chunking test
    or none (review 2026-10-06, TA3)."""

    @SETTINGS
    @given(data=st.data(), chunk=st.integers(min_value=1, max_value=13))
    def test_the_properties_hold(self, kind, data, chunk):
        df = data.draw(kind_streams(kind), label="stream")
        split = data.draw(st.integers(1, df.height - 1), label="split")
        spec = _kind_spec(kind)
        whole = po.ModelBank([spec])
        one = whole.fit_predict(df)
        bank = po.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
        assert unnested(one).equals(unnested(many), null_equal=True), "chunking moved a number"
        if (state := _state(whole, kind)) is not None:
            assert state.equals(_state(bank, kind), null_equal=True), "chunking moved the state"
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, split))
        b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
        rest = df.slice(split)
        assert unnested(a.fit_predict(rest)).equals(unnested(b.fit_predict(rest)), null_equal=True)
        if (state := _state(a, kind)) is not None:
            assert state.equals(_state(b, kind), null_equal=True), "a load moved the state"
        fields = one.select("m").unnest("m")
        _finite_or_null(fields)
        cond = pl.lit(False)
        for f in spec["features"]:
            cond = cond | _missing(f)
        for i, skip in enumerate(df.select(cond).to_series().to_list()):
            if skip:
                present = [c for c in fields.columns if fields[c][i] is not None]
                assert not present, f"row {i} was skipped but reported {present}"
