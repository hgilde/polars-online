"""T-D2: property-based tests over adversarial streams.

Section C of docs/TESTING.md lists edge cases someone thought of. This file
generates them instead: mixed nulls, duplicate and backwards clocks, constant
and collinear features, weight extremes, tiny groups, unusual chunkings — and
asserts the invariants that must hold for *every* model on *every* stream.
The values reach the input bound (``1e100``, as weights too, beside
``1e-100``) and past it (NaN and the infinities, which the bank reads as
missing); every regression runs under each switch it takes; and the kinds
that are not regressions run on a stream of their own (review 2026-10-06,
TA3: none of that was drawn). Each property runs on streams of mixed signs
and on one-sided ones, all positive and all negative, at a level (docs/PLAN.md
task 209 (a), `SIGNS`).

Each parameter with no chunking or save test of its own is a switch here
(task 278, batch 13): the solve cadences, the grids and sets, the priors, a
session's shrink, a window's edge, cadence and budget, and each model's own
keywords; `test_each_switch_keeps_the_properties` runs each switch on each model, one case
a switch in the essentials and the rest in the extended tier. The seven
kinds no sweep ran under a clock, a weight, a session or an embargo run
under each, with the clock restarting, on drawn streams and on a long one
cut into 1, 7 and 600 chunks.

Hypothesis shrinks a failure to a minimal reproducing frame, which is the point:
these tests are meant to produce a small counterexample, not just a red mark.
"""

import math
import warnings
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

import polars_online as po
from test_model_registry import MINIMAL, REGRESSIONS, _build
from test_properties_temporal import CLOCK_MODELS
from tiers import examples

TIER = "mixed"

#: The bank's bound on a value it learns from (`online_core::INPUT_BOUND`):
#: a magnitude past it, NaN and the infinities are missing, as a null is.
INPUT_BOUND = 1e100

MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"delta": 1.0}),
    ("kalman", {"coef_half_life": 50.0}),
    ("lasso", {"penalties": [0.1, 0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
    ("sgd", {"learning_rate": 0.01, "clip_gradient": 1e3}),
    ("pa", {}),
    ("holt", {"features": None}),  # no features keyword
]
IDS = [m[0] for m in MODELS]

#: 30 streams a property in the extended tier, the tier's count in the
#: essentials (``tiers.ESSENTIAL_EXAMPLES``).
SETTINGS = settings(
    max_examples=examples(30),
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
#: A weight of ``-0.0`` is a weight of 0, "advance the clock, learn nothing"
#: (hard rule 9): it is not below zero, so the bank takes it. Hypothesis never
#: draws it from ``floats(min_value=0.0)``, so it is named (task 209 (e)).
_weights = st.one_of(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([-0.0, 1e-100, INPUT_BOUND, float("nan"), float("inf"), 1e101, -1e101]),
    st.none(),
)


def _missing(col: str) -> pl.Expr:
    """Where the bank reads ``col`` as missing: null, NaN, infinite or past
    the bound (`stream.rs::usable`)."""
    c = pl.col(col)
    return c.is_null() | ~c.is_finite() | (c.abs() > INPUT_BOUND)


#: The side of zero a stream's values sit on, as the generators' ``sign``
#: and ``offset`` (docs/PLAN.md task 209 (a)). Mixed, the values are drawn as
#: they are; one-sided, each value ``v`` is read as ``sign * |v| + offset``:
#: all positive at a level of 1e3, or all negative at -1e3. A null and NaN
#: stay what they are, an infinity takes the sign, and the bound stays the
#: bound, since ``1e100 + 1e3`` is ``1e100``. The generators could draw a
#: one-sided stream and were never made to: an all-negative feature reached 9
#: of the 21 kinds, an all-negative target 7 of the 12 with one. This catches
#: a NaN, an overflow or a divergence at a level, not a wrong fit, which
#: `crates/online-core/tests/held_values.rs` holds.
SIGNS: dict[str, dict[str, float]] = {
    "mixed": {},
    "positive": {"sign": 1.0, "offset": 1e3},
    "negative": {"sign": -1.0, "offset": -1e3},
}

#: How many streams of each sign a property draws for a model. Mixed keeps
#: the 30 it drew before one-sided streams were added; each one-sided sign
#: draws 10, so a property runs 50 streams a model where tripling would run
#: 90. Every stream draws its clocks, groups, nulls, weights and chunks
#: alike, whatever its sign, so a one-sided stream adds one dimension, the
#: level, and a third of mixed's draws covers it. In the essentials (task
#: 210, which keeps this file among the tests run on every commit) mixed
#: draws the tier's 10 and each one-sided sign 4, a third of that rounded
#: up: 18 streams a property and model where the full tier runs 50.
EXAMPLES = {
    "mixed": examples(30),
    "positive": examples(10, essential=4),
    "negative": examples(10, essential=4),
}

#: Streams a random draw found a failure on, run first wherever a property
#: names them (`_given`'s ``pinned``), as each is mixed in sign. Four rows on
#: which `huber`'s MAD scale panicked during 1.3.0's development: the
#: residuals' median sat at 0 while one arrived more than the scale's band
#: below every other, and the sketch's fold read a negative place in its
#: buckets (task 272 (p); 2 of 40 random seeds, the derandomized CI profile
#: none).
FOUND = (
    pl.DataFrame(
        {
            "g": ["g0"] * 4,
            "t": [50003.5, 60004.5, 70013.0, 80027.0],
            "x0": [-1e100, -1e100, -1e8, -0.0],
            "x1": [-0.0, 1e-12, -0.0, 1e100],
            "y0": [1e-12, 1e8, 1e-12, 0.0],
            "w": [1e100, 7.7322, 1e100, 1e100],
            "s": ["s0"] * 4,
        }
    ),
)

#: The chunk sizes a property cuts a stream into.
CHUNKS = st.integers(min_value=1, max_value=13)


def _signed(v: float | None, sign: float | None, offset: float) -> float | None:
    """``v`` read as ``sign * |v| + offset``, or as it is without a sign or
    where it is null or NaN (`SIGNS`)."""
    if sign is None or v is None or math.isnan(v):
        return v
    return sign * abs(v) + offset


def _given(
    sign: str,
    prop: Callable[..., None],
    pinned: tuple[dict, ...] = (),
    **strategies: st.SearchStrategy,
) -> None:
    """Run ``prop`` as a Hypothesis test over ``strategies``, at the number
    of streams ``sign`` draws (`EXAMPLES`), and on each of ``pinned``'s
    arguments first, as an ``@example``: run in every profile, the
    derandomized one included."""
    test = given(**strategies)(prop)
    for kw in pinned:
        test = example(**kw)(test)
    settings(SETTINGS, max_examples=EXAMPLES[sign])(test)()


def _signed_columns(model: str) -> tuple[str, ...]:
    """The columns a one-sided stream signs for ``model``: ``ftrl``'s target
    is a label, ``y0 > 0``, so it signs its features only, and ``holt`` reads
    its target alone."""
    if model == "ftrl":
        return ("x0", "x1")
    if model == "holt":
        return ("y0",)
    return ("x0", "x1", "y0")


@st.composite
def streams(
    draw,
    min_rows=1,
    max_rows=40,
    max_groups=3,
    sign: float | None = None,
    offset: float = 0.0,
    columns: tuple[str, ...] = ("x0", "x1", "y0"),
    min_groups=1,
):
    """An adversarial but *valid* stream: the clock is non-decreasing within
    each group (mis-ordered input is its own test, T-E4). With a ``sign``,
    ``columns`` sit on its side of zero at ``offset`` (`SIGNS`). Its first
    ``min_groups`` rows are each of a group of their own, so that many
    groups always appear: generated, not filtered after (review round 9,
    J-6: `assume(len(keys) > 1)` dropped a third of the streams)."""
    n = draw(st.integers(min_value=max(min_rows, min_groups), max_value=max_rows))
    n_groups = draw(st.integers(min_value=min_groups, max_value=max(max_groups, min_groups)))
    groups = draw(st.lists(st.integers(0, n_groups - 1), min_size=n, max_size=n))
    groups[:min_groups] = range(min_groups)

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
    for c in columns:
        cols[c] = [_signed(v, sign, offset) for v in cols[c]]
    floats = ["t", "x0", "x1", "y0", "w"]
    return pl.DataFrame(cols, schema_overrides={c: pl.Float64 for c in floats})


def _streams(model: str, sign: str, **kw) -> st.SearchStrategy:
    """`streams` for ``model`` on ``sign``'s side of zero."""
    return streams(**kw, **SIGNS[sign], columns=_signed_columns(model))


#: The late-row minimum the stepping-back streams are run under: a step back
#: larger than it starts the model over, one no larger is refused.
MINIMUM = 5.0


@st.composite
def stepping_back_streams(draw, min_rows=2, max_rows=40, max_groups=3, **signs):
    """`streams`, with each group's clock stepping back now and then -- by more
    than `MINIMUM`, so that under `"reset_state"` every step back starts the
    group over and none is refused (task 120). ``signs`` are `streams`'s
    ``sign``, ``offset`` and ``columns``."""
    df = draw(streams(min_rows=min_rows, max_rows=max_rows, max_groups=max_groups, **signs))
    clocks, last = [], {}
    for g in df["g"].to_list():
        step = draw(st.sampled_from([0.0, 0.5, 1.0, 7.0, 1e4, -(MINIMUM + 2.0), -1e4]))
        last[g] = last.get(g, 0.0) + step if g in last else 0.0
        clocks.append(last[g])
    return df.with_columns(t=pl.Series(clocks, dtype=pl.Float64))


#: Ordinary rows at the head of `warmed_streams`. At weight 1, one clock unit
#: apart under `build`'s `half_life` of 20, the weight before the fourth is
#: `1 + lam + lam^2` = 2.90 at `lam = 2^(-1/20)`, past its `min_weight` of 2,
#: so every model scores that row (review round 5, E4: 2.80 was the
#: once-decayed value).
WARM_ROWS = 4


@st.composite
def warmed_streams(
    draw,
    sign: float | None = None,
    offset: float = 0.0,
    columns: tuple[str, ...] = ("x0", "x1", "y0"),
):
    """`streams`, behind `WARM_ROWS` rows of group `g0` with finite values at
    weight 1, so every stream has a row the model scored. Drawn at random, too
    few streams had one, and kalman failed Hypothesis's `filter_too_much`
    health check once in a gate and once in 460 runs (2026-10-07). With a
    ``sign``, the warm rows' ``columns`` sit on its side of zero too. Their
    targets are at least 1e-3 from 0, and their features 0 or at least 1e-3
    from it: kalman sizes a target's prior from three squared innovations
    above 0 and predicts nothing while it is unsized (task 270, review round
    9 B-1), so warm targets of 0, or of 1e-172, whose square underflows to
    0, left it no row to score, and so did a feature at 7.9e-57 beside one
    at 11, after which its prior was unsized again at the fourth row."""
    df = draw(streams(sign=sign, offset=offset, columns=columns))
    finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    rows = st.lists(finite, min_size=WARM_ROWS, max_size=WARM_ROWS)
    warm = {c: draw(rows) for c in ("x0", "x1", "y0")}
    warm["y0"] = [v if abs(v) >= 1e-3 else 1.0 for v in warm["y0"]]
    for c in ("x0", "x1"):
        warm[c] = [v if abs(v) >= 1e-3 else 0.0 for v in warm[c]]
    for c in columns:
        warm[c] = [_signed(v, sign, offset) for v in warm[c]]
    head = pl.DataFrame(
        {
            "g": ["g0"] * WARM_ROWS,
            "t": [float(i) for i in range(WARM_ROWS)],
            **warm,
            "w": [1.0] * WARM_ROWS,
            "s": ["s0"] * WARM_ROWS,
        },
        schema=df.schema,
    )
    # The drawn stream's `g0` rows follow the warm ones on its clock.
    rest = df.with_columns(
        t=pl.when(pl.col("g") == "g0").then(pl.col("t") + WARM_ROWS).otherwise("t")
    )
    return pl.concat([head, rest])


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
    if opts.get("features", ()) is None:
        del opts["features"]  # a model with no features keyword (holt)
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


@pytest.mark.parametrize("sign", list(SIGNS))
@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestUniversalProperties:
    """The invariants every regression keeps, each on streams of `sign`: one
    Hypothesis run of `EXAMPLES[sign]` streams a property, model and sign."""

    def test_chunking_never_changes_the_output(self, model, extra, sign):
        def prop(df, chunk):
            df = binarize(df, model)
            spec = build(model, extra)
            one = unnested(po.ModelBank([spec]).fit_predict(df))
            bank = po.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
            many = unnested(pl.concat(parts))
            assert one.equals(many, null_equal=True)

        _given(sign, prop, df=_streams(model, sign), chunk=CHUNKS)

    def test_chunking_never_changes_the_output_when_the_clock_steps_back(self, model, extra, sign):
        """Hard rule 3 with the clock stepping back: under `"reset_state"` each
        step back past the minimum starts its group over, and where the chunks
        fall cannot move a number."""

        def prop(df, chunk):
            df = binarize(df, model)
            spec = build(model, extra, restart_after_step_back=MINIMUM)
            one = unnested(po.ModelBank([spec]).fit_predict(df))
            bank = po.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
            assert one.equals(unnested(pl.concat(parts)), null_equal=True)

        signs = {**SIGNS[sign], "columns": _signed_columns(model)}
        _given(sign, prop, df=stepping_back_streams(**signs), chunk=CHUNKS)

    def test_a_late_row_is_refused_at_its_input_row_whatever_the_chunking(self, model, extra, sign):
        """A row stepped back by no more than the minimum is a late row: the
        stream is refused there, naming the row's place in the input, whether
        the row before it in its group is in the same chunk or in the state."""

        def prop(df, chunk, data):
            groups = df["g"].to_list()
            later = [i for i in range(1, df.height) if groups[i] in groups[:i]]
            assume(later)
            i = data.draw(st.sampled_from(later), label="late row")
            prev = max(j for j in range(i) if groups[j] == groups[i])
            t = df["t"].to_list()
            t[i] = t[prev] - 2.0
            # Every later row of its group stays after it. The late row and
            # the one before it are ordinary rows, so the null policy skips
            # neither and the step back is measured between the two. Their
            # features sit on the stream's side of zero.
            ordinary = pl.int_range(pl.len()).is_in([prev, i])
            one = _signed(1.0, SIGNS[sign].get("sign"), SIGNS[sign].get("offset", 0.0))
            df = df.with_columns(
                t=pl.Series(t, dtype=pl.Float64),
                x0=pl.when(ordinary).then(one).otherwise("x0"),
                x1=pl.when(ordinary).then(one).otherwise("x1"),
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

        _given(sign, prop, df=_streams(model, sign, min_rows=2), chunk=CHUNKS, data=st.data())

    def test_save_load_is_transparent(self, model, extra, sign):
        def prop(df, split):
            assume(split < df.height)
            df = binarize(df, model)
            spec = build(model, extra)
            a = po.ModelBank([spec])
            a.fit_predict(df.slice(0, split))
            b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
            rest = df.slice(split, df.height - split)
            got = unnested(b.fit_predict(rest))
            assert unnested(a.fit_predict(rest)).equals(got, null_equal=True)

        splits = st.integers(min_value=1, max_value=39)
        _given(sign, prop, df=_streams(model, sign, min_rows=2), split=splits)

    def test_outputs_are_finite_or_null(self, model, extra, sign):
        def prop(df):
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

        found = tuple({"df": df} for df in FOUND) if sign == "mixed" else ()
        _given(sign, prop, found, df=_streams(model, sign))

    def test_feature_or_weight_null_means_all_outputs_null(self, model, extra, sign):
        """A row with a null feature or weight is skipped, and every field of
        its record is null, not `weight_sum` alone (review 2026-10-05, TA10)."""

        def prop(df):
            df = binarize(df, model)
            spec = build(model, extra)
            out = po.ModelBank([spec]).fit_predict(df)
            # Only the columns this spec actually declares can skip a row --
            # holt reads no features, so an unused null column must not
            # disturb it. A value the bank reads as missing skips a row as a
            # null does.
            cond = _missing("w")
            for f in spec["features"]:
                cond = cond | _missing(f)
            skipped = df.select(cond).to_series().to_list()
            fields = out.select("m").unnest("m")
            for i, skip in enumerate(skipped):
                if skip:
                    present = [c for c in fields.columns if fields[c][i] is not None]
                    assert not present, f"row {i} was skipped but reported {present}"

        _given(sign, prop, df=_streams(model, sign))

    def test_groups_are_independent(self, model, extra, sign):
        def prop(df):
            df = binarize(df, model)
            keys = df["g"].unique().to_list()
            assert len(keys) > 1, keys
            spec = build(model, extra)
            both = po.ModelBank([spec]).fit_predict(df)
            for key in keys:
                solo = po.ModelBank([spec]).fit_predict(df.filter(pl.col("g") == key))
                a = unnested(both.filter(pl.col("g") == key))
                b = unnested(solo)
                assert a.equals(b, null_equal=True), f"group {key} was affected by the others"

        _given(sign, prop, df=_streams(model, sign, min_groups=2, max_groups=3))

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_prediction_never_depends_on_the_current_target(self, model, extra, sign):
        """Out-of-sample by construction (docs/PLAN.md hard rule 2): changing a
        row's target must not change that row's own prediction.

        The row is drawn from those the model scored and whose target is
        there. The first such target is no use: every model withholds its
        first row, so a test that perturbed it compared two nulls in every
        stream (review 2026-10-05, TA1). `warmed_streams` gives every stream
        such a row. The count at the end says how many streams compared a
        prediction that was there, held to a third of the streams run.

        `ewridge`'s error-inflation gate is switched off. On these short,
        mostly-null streams it withholds all but 13 in 100 of them, and the
        last warm row in about 4 draws in 5. The gate only withholds, and
        the comparison still fails if the perturbed run withholds a row the
        base run scored."""
        compared = []
        kw = {"max_error_inflation": float("inf")} if model == "ewridge" else {}

        def check(df, data):
            df = binarize(df, model)
            spec = build(model, extra, **kw)
            base = po.ModelBank([spec]).fit_predict(df)
            field = next(f.name for f in base.schema["m"].fields if f.name.startswith("pred_"))
            preds = base["m"].struct.field(field).to_list()
            y = df["y0"].to_list()
            # A row the model scored and whose target it reads: a NaN,
            # infinite or beyond-bound target is skipped by the bank, and
            # `1e100 + 12345` is `1e100`, so perturbing one compared the
            # same stream with itself (review round 5, E3).
            usable = df.select((~_missing("y0")).alias("ok"))["ok"].to_list()
            scored = [i for i in range(df.height) if preds[i] is not None and usable[i]]
            assert scored, f"no row scored, not even warm row {WARM_ROWS - 1}"
            idx = data.draw(st.sampled_from(scored), label="scored row")
            was = y[idx]
            if model == "ftrl":
                y[idx] = 0.0 if was else 1.0
            elif abs(was) < 1e15:
                y[idx] = was + 12345.0
            else:
                # Past 1e15 an addition of 12345 can round away; halving
                # moves every finite value and stays inside the bound.
                y[idx] = was / 2.0
            assert y[idx] != was
            perturbed = po.ModelBank([spec]).fit_predict(
                df.with_columns(y0=pl.Series(y, dtype=pl.Float64))
            )
            a = preds[idx]
            b = perturbed["m"].struct.field(field).to_list()[idx]
            assert a == b, f"row {idx}: changing its own target changed its prediction ({a} -> {b})"
            compared.append(idx)

        streams = warmed_streams(**SIGNS[sign], columns=_signed_columns(model))
        _given(sign, check, df=streams, data=st.data())
        assert len(compared) >= EXAMPLES[sign] // 3, (
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
    # `kalman`, `sgd` and `pa` standardize by default, so the switch above
    # was their plain spec (`_takes`); this is theirs.
    "no standardize": {"standardize": False},
    "embargo": {"embargo": 3.0},
    "session": {"session": "s", "session_gap": 2.0},
    # Every parameter of a regression's own below had no chunking or save
    # test of its own (task 278, batch 13; C6): the switches are the
    # invariance test of each.
    #
    # The solve cadences. `MODELS` solves every row (`max_rows_between_solves
    # = 1`), so no schedule was ever cut by a chunk: the default schedule by
    # weight, and one on the clock. At `build`'s half-life of 20 the default
    # schedule solves every 0.4 of clock in steady state, so at nearly every
    # row of a stream stepping by 1; at 1,000 it solves every 20.
    "default cadence": {"max_rows_between_solves": None, "half_life": 1000.0},
    "clock cadence": {"solve_every": 3.0, "max_rows_between_solves": None},
    "ridge grid": {"ridge": [1e-6, 0.5]},
    "feature sets": {"feature_sets": {"a": ["x0"], "b": ["x0", "x1"]}},
    "sum scale": {"ridge_scale": "sum"},
    # One target and two features with an intercept is every model's shape
    # here: `[intercept, x0, x1]`, the intercept read only under "sum".
    "coef prior": {"coef_prior": [[0.5, 2.0, -1.0]]},
    "session shrink": {
        "session": "s",
        "session_gap": 2.0,
        "session_shrink": 0.5,
        "long_half_life": 200.0,
    },
    "closed both": {"window_size": 15.0, "closed": "both"},
    "snapshot cadence": {
        "window_size": 15.0,
        "snapshot_every": 4.0,
        "max_rows_between_snapshots": 3,
    },
    # MiB: about half a kilobyte, a few snapshots, so the ring thins on
    # these streams (`test_every_switch_moves_an_output`).
    "thin budget": {"window_size": 15.0, "window_budget": {"thin": 0.0005}},
    # One covariance for both targets: with one target it is the per-target
    # filter, so the switch adds a second (`SECOND_TARGET`).
    "share_p": {"targets": ["y0", "y1"], "share_p": True},
    "revert": {"revert_half_life": 30.0},
    "quantile_eps": {"quantile_eps": 0.5},
    "l1_ratio": {"l1_ratio": 0.5},
    "select_half_life": {"select_half_life": 5.0},
    "trend off": {"trend": False},
}

#: Switches that wait for task 277, which rebuilds huber's warm-up and its
#: scale. Each is ready: registering it is one line,
#: ``SWITCHES.update(AFTER_TASK_277)``, and `test_the_waiting_switches_build`
#: holds that each builds today.
AFTER_TASK_277: dict[str, dict] = {
    "huber rms": {"huber_scale": "rms"},
}

#: The second target a switch may name (`share_p`): the first target's
#: values in reverse, so its nulls, zeros and values past the bound fall on
#: other rows.
SECOND_TARGET = pl.col("y0").reverse().alias("y1")


def _takes(model: str, extra: dict, switch: dict) -> bool:
    """Whether the builder takes ``switch`` for ``model``: probed, so each
    model draws only the switches it accepts and no case is a refusal. A
    switch that builds the plain spec again is not taken: ``standardize=True``
    on a model that standardizes by default ran the plain spec under another
    name (task 278, batch 13)."""
    try:
        spec = build(model, extra, **switch)
    except (TypeError, ValueError):
        return False
    return not switch or spec != build(model, extra)


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


def test_the_waiting_switches_build():
    """Each switch waiting for task 277 builds for some model today, so
    registering it is the one line `AFTER_TASK_277` names."""
    for name, sw in AFTER_TASK_277.items():
        assert name not in SWITCHES, name
        assert any(_takes(model, extra, sw) for model, extra in MODELS), name


def _switched(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    """``df`` with the second target where ``spec`` names it."""
    return df.with_columns(SECOND_TARGET) if "y1" in spec["targets"] else df


def _holds_under(model: str, extra: dict, switch: str, df, chunk: int, split: int) -> None:
    """The universal properties under ``switch``: one chunk or many give the
    same numbers, a save and load at row ``split`` is transparent, every
    number is finite or null, and a skipped row reports nothing."""
    df = binarize(df, model)
    spec = build(model, extra, **SWITCHES[switch])
    df = _switched(df, spec)
    one = po.ModelBank([spec]).fit_predict(df)
    bank = po.ModelBank([spec])
    many = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
    assert unnested(one).equals(unnested(many), null_equal=True), "chunking moved a number"
    a = po.ModelBank([spec])
    a.fit_predict(df.slice(0, split))
    b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
    rest = df.slice(split)
    got = unnested(b.fit_predict(rest))
    assert unnested(a.fit_predict(rest)).equals(got, null_equal=True)
    fields = one.select("m").unnest("m")
    _finite_or_null(fields)
    cond = _missing("w")
    for f in spec["features"]:
        cond = cond | _missing(f)
    for i, skip in enumerate(df.select(cond).to_series().to_list()):
        if skip:
            present = [c for c in fields.columns if fields[c][i] is not None]
            assert not present, f"row {i} was skipped but reported {present}"


@pytest.mark.parametrize("sign", list(SIGNS))
@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestEverySwitch:
    """The universal properties under each switch a model takes: one chunk
    or many give the same numbers, a save and load at any row is
    transparent, every number is finite or null, and a skipped row reports
    nothing (review 2026-10-06, TA3); on streams of each sign. Each stream
    draws its switch, so every model meets a mix of them in the essentials;
    `test_each_switch_keeps_the_properties` runs each switch on each model."""

    def test_the_properties_hold_under_each_switch(self, model, extra, sign):
        def prop(df, chunk, data):
            switch = data.draw(st.sampled_from(SWITCHES_FOR[model]), label="switch")
            split = data.draw(st.integers(1, df.height - 1), label="split")
            _holds_under(model, extra, switch, df, chunk, split)

        _given(sign, prop, df=_streams(model, sign, min_rows=2), chunk=CHUNKS, data=st.data())


def _representatives() -> dict[str, str]:
    """The model each switch runs on in the essentials: of the models that
    take it, the one given the fewest switches so far, so the essentials
    spread over the models rather than run every `ewridge` switch on it."""
    given: dict[str, int] = {model: 0 for model, _ in MODELS}
    out = {}
    for name in SWITCHES:
        takers = [model for model, _ in MODELS if name in SWITCHES_FOR[model]]
        out[name] = min(takers, key=lambda m: given[m])
        given[out[name]] += 1
    return out


REPRESENTATIVE = _representatives()


def _switch_cases() -> list:
    """Every (model, switch, sign) a model takes: in the essentials, each
    switch on its `REPRESENTATIVE` with streams of mixed signs; the rest of
    the grid in the extended tier, which CI runs on each of its legs and
    `scripts/gate.sh --extended` before each push."""
    cases = []
    for (model, extra), mid in zip(MODELS, IDS, strict=True):
        for name in SWITCHES_FOR[model]:
            if name == "plain":
                continue  # `TestUniversalProperties` is the plain spec
            for sign in SIGNS:
                essential = sign == "mixed" and REPRESENTATIVE[name] == model
                marks = (
                    ()
                    if essential
                    else pytest.mark.extended(reason="a grid: every switch on every model")
                )
                cases.append(
                    pytest.param(model, extra, name, sign, id=f"{mid}-{name}-{sign}", marks=marks)
                )
    return cases


@pytest.mark.parametrize(("model", "extra", "switch", "sign"), _switch_cases())
def test_each_switch_keeps_the_properties(model, extra, switch, sign):
    """`TestEverySwitch`'s properties with the switch fixed, so each switch
    on each model draws its own streams (task 278, batch 13): a stream that
    draws its switch among the fifteen `ewridge` takes ran each about twice
    in the essentials."""

    def prop(df, chunk, data):
        split = data.draw(st.integers(1, df.height - 1), label="split")
        _holds_under(model, extra, switch, df, chunk, split)

    _given(sign, prop, df=_streams(model, sign, min_rows=2), chunk=CHUNKS, data=st.data())


def _ordinary(n: int = 120) -> pl.DataFrame:
    """An ordinary stream for the activation checks: finite features, a
    target missing now and then, unit weights but a few rows of weight 0, a
    clock in unit steps with a gap of 40, and a session that changes every
    30 rows."""
    r = np.random.default_rng(278)
    x0, x1 = r.normal(size=n), r.normal(size=n)
    y0 = 1.0 + 2.0 * x0 - x1 + 0.5 * r.standard_t(3, size=n)
    t = np.arange(n, dtype=float) + np.where(np.arange(n) >= 70, 40.0, 0.0)
    w = np.where(np.arange(n) % 17 == 5, 0.0, 1.0)
    return pl.DataFrame(
        {
            "g": ["g0"] * n,
            "t": t,
            "x0": x0,
            "x1": x1,
            "y0": np.where(np.arange(n) % 23 == 7, np.nan, y0),
            "w": w,
            "s": [f"s{i // 30}" for i in range(n)],
        }
    )


def _moved(a: pl.DataFrame, b: pl.DataFrame) -> list[str]:
    """The fields of two runs of one stream that differ, or that one has and
    the other does not."""
    return sorted(
        c
        for c in set(a.columns) | set(b.columns)
        if c not in a.columns or c not in b.columns or not a[c].equals(b[c], null_equal=True)
    )


#: A switch a model takes that moves nothing by design: `holt` has no
#: intercept to drop, and `test_spec_validation.py` pins `fit_intercept` as
#: inert on it.
INERT = frozenset({("holt", "no intercept")})


def _new_value(c: str, plain: pl.DataFrame, got: pl.DataFrame) -> bool:
    """Whether field ``c`` of ``got`` says something ``plain`` does not: it
    is a field of both that moved, or a slot of a grid or a set (``pred_y0__a``)
    that differs from the plain spec's field (``pred_y0``), or a field of a
    target ``plain`` has none of."""
    if c in plain.columns:
        return True
    base = c.split("__")[0]
    return base not in plain.columns or not got[c].equals(plain[base], null_equal=True)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
def test_every_switch_moves_an_output(model, extra):
    """No switch is dead in the sweep: on an ordinary stream, each moves a
    value the plain spec reports, so its invariance is not that of a
    parameter that did nothing; a grid's or a set's slot differs from the
    plain spec's field it renames (`_new_value`). `weight_sum` is where a window's ring
    shows: `closed="both"` keeps a row exactly a window old (the clock
    steps by 1), and a cadence or a thinned ring holds fewer rows."""
    df = _ordinary()
    if model == "ftrl":
        df = df.with_columns(y0=(pl.col("y0") > 1).cast(pl.Float64))
    shared = unnested(po.ModelBank([build(model, extra)]).fit_predict(df))
    for name in SWITCHES_FOR[model]:
        if name == "plain":
            continue
        spec = build(model, extra, **SWITCHES[name])
        got = unnested(po.ModelBank([spec]).fit_predict(_switched(df, spec)))
        moved = _moved(shared, got)
        values = [c for c in moved if c in got.columns and _new_value(c, shared, got)]
        print(f"{model} {name}: moved {moved}")
        if (model, name) in INERT:
            assert not moved, f"{name} is pinned inert on {model} and moved {moved}"
            continue
        assert values, f"{name} moved no value {model} reports on its own"
        if "window_size" in SWITCHES[name] and name != "window":
            # Against the window alone: the switch itself moves the ring.
            window = build(model, extra, **SWITCHES["window"])
            alone = unnested(po.ModelBank([window]).fit_predict(df))
            assert "weight_sum" in _moved(alone, got), f"{name} left {model}'s ring as it was"


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
    "corrchange": {"window_rows": 8},
}


def _kind_spec(kind: str) -> dict:
    return {**_build(kind), "model": {**_build(kind)["model"], **KIND_ARGS.get(kind, {})}}


def _state(bank: po.ModelBank, kind: str) -> pl.DataFrame | None:
    """What a kind that reports through its state reports: `marginal`'s
    pairs, `rcov`'s closed blocks and `audit`'s counts."""
    if kind == "marginal":
        return bank.marginal("m")
    if kind == "audit":
        return pl.concat(
            [bank.audit("m"), bank.audit("m", pooled=True).with_columns(group=pl.lit("*"))],
            how="diagonal_relaxed",
        )
    if kind == "rcov":
        return bank.closed_groups("m", drop=False)
    return None


@st.composite
def kind_streams(
    draw, kind: str, min_rows=2, max_rows=60, sign: float | None = None, offset: float = 0.0
):
    """A stream any kind runs on, adversarial as `streams` is: features
    ``x0`` and ``x1`` and a target ``y`` reaching the bound and past it, with
    nulls; a group key ``g`` that only grows, for `rcov`'s monotone close;
    and, for `ew_class`, ``y`` as its label, ``a`` or ``b``, null where the
    number was missing. With a ``sign``, the numbers sit on its side of zero
    at ``offset`` (`SIGNS`), but for `ew_class`'s label, which reads the side
    of zero a number drawn of mixed signs fell on."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cols = {c: draw(st.lists(_values, min_size=n, max_size=n)) for c in ("x0", "x1", "y")}
    for c in ("x0", "x1") if kind == "ew_class" else ("x0", "x1", "y"):
        cols[c] = [_signed(v, sign, offset) for v in cols[c]]
    df = pl.DataFrame(
        {**cols, "g": np.cumsum(draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)))},
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
    or none (review 2026-10-06, TA3); on streams of each sign."""

    @pytest.mark.parametrize("sign", list(SIGNS))
    def test_the_properties_hold(self, kind, sign):
        def prop(data, chunk):
            df = data.draw(kind_streams(kind, **SIGNS[sign]), label="stream")
            split = data.draw(st.integers(1, df.height - 1), label="split")
            spec = _kind_spec(kind)
            whole = po.ModelBank([spec])
            one = whole.fit_predict(df)
            bank = po.ModelBank([spec])
            many = pl.concat(
                [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
            )
            assert unnested(one).equals(unnested(many), null_equal=True), "chunking moved a number"
            if (state := _state(whole, kind)) is not None:
                assert state.equals(_state(bank, kind), null_equal=True), "chunking moved the state"
            a = po.ModelBank([spec])
            a.fit_predict(df.slice(0, split))
            b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
            rest = df.slice(split)
            got = unnested(b.fit_predict(rest))
            assert unnested(a.fit_predict(rest)).equals(got, null_equal=True)
            if (state := _state(a, kind)) is not None:
                assert state.equals(_state(b, kind), null_equal=True), "a load moved the state"
            fields = one.select("m").unnest("m")
            _finite_or_null(fields)
            cond = pl.lit(False)
            # An audit reads every row: what the others skip is what it counts.
            for f in [] if kind == "audit" else spec["features"]:
                cond = cond | _missing(f)
            for i, skip in enumerate(df.select(cond).to_series().to_list()):
                if skip:
                    present = [c for c in fields.columns if fields[c][i] is not None]
                    assert not present, f"row {i} was skipped but reported {present}"

        _given(sign, prop, data=st.data(), chunk=CHUNKS)

    def test_a_negative_weight_is_refused_naming_the_row(self, kind):
        """One check refuses a finite negative weight for every kind
        (`bank.rs`, `extract`), and `test_semantics_all_models` held the ten
        regressions to it; these eleven were held to it nowhere (task 209
        (e)). A weight of ``-0.0`` is not below zero, and is taken.
        `seqtest` takes no weight column at all: every learned row is one
        trial, and the bank refuses the spec; nor does `audit`, which counts
        every row whatever it weighs."""
        n = 12
        rng = np.random.default_rng(9)
        df = pl.DataFrame(
            {
                "x0": rng.normal(size=n),
                "x1": rng.normal(size=n),
                "y": rng.normal(size=n),
                "g": np.arange(n) // 4,
                "w": [1.0] * 6 + [-0.0] + [1.0] * 5,
            }
        )
        if kind == "ew_class":
            df = df.with_columns(
                y=pl.when(pl.col("y") > 0).then(pl.lit("a")).otherwise(pl.lit("b"))
            )
        spec = {**_kind_spec(kind), "weight": "w"}
        if kind in ("seqtest", "audit"):
            with pytest.raises(ValueError, match=f"weight does not apply to {kind}"):
                po.ModelBank([spec])
            return
        po.ModelBank([spec]).fit_predict(df)
        w = df["w"].to_list()
        w[8] = -0.5
        bad = df.with_columns(w=pl.Series(w, dtype=pl.Float64))
        says = r'weight column "w" has a negative value \(-0\.5\) at row 8'
        with pytest.raises((pl.exceptions.ComputeError, ValueError), match=says):
            po.ModelBank([spec]).fit_predict(bad)


# --- the stream parameters on the kinds no sweep cut (task 278, batch 13) ---

#: The kinds neither property sweep ran under a clock, a weight, a session or
#: an embargo: `TestEveryOtherKind` runs each as `MINIMAL` builds it, and
#: `test_properties_temporal.py`'s `CLOCK_MODELS` leaves them out (A6).
#: `test_clock_events_every_kind.py` holds a restart and a session reset on
#: them to a fresh run, in one chunk; here the same events are cut by chunks
#: and by a save.
UNCLOCKED_KINDS = sorted(set(OTHER_KINDS) - set(CLOCK_MODELS))

#: The stream parameters a kind may run under. The clock steps back by more
#: than `MINIMUM` now and then (`cut_streams`), so a clocked stream restarts.
STREAM_PARAMETERS: dict[str, dict] = {
    "clock": {"clock": "t", "gap_cap": 100.0, "restart_after_step_back": MINIMUM},
    "weight": {"weight": "w"},
    "session": {"session": "s", "session_gap": 2.0},
    "session reset": {"session": "s", "session_gap": "reset"},
    "embargo": {"embargo": 3.0},
}

#: What a kind refuses of `STREAM_PARAMETERS`, by name: `audit` counts every
#: row whatever it weighs and has no label to wait for, `seqtest` reads each
#: learned row as one trial, and `rcov`'s block closes with its group, which
#: cannot release the rows an embargo holds.
REFUSED: dict[str, set[str]] = {
    "audit": {"weight", "embargo"},
    "seqtest": {"weight"},
    "rcov": {"embargo"},
}


def _cut_spec(kind: str, names: tuple[str, ...]) -> dict:
    """``kind`` as `_kind_spec` builds it, under the stream parameters
    ``names``, and with a coefficient row on every row where it reports
    coefficients, so `coef` is compared on each row too."""
    kw: dict = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
    kw.update(MINIMAL[kind])
    kw.update(KIND_ARGS.get(kind, {}))
    for name in names:
        kw.update(STREAM_PARAMETERS[name])
    kw = {k: v for k, v in kw.items() if v is not None}
    build = getattr(po.spec, kind)
    try:
        return build("m", coef_every=0, **kw)
    except ValueError as e:
        if "coef_every" not in str(e):
            raise
        return build("m", **kw)


def _cuts(kind: str, names: tuple[str, ...]) -> bool:
    """Whether ``kind`` runs under ``names``: the builder and the bank both
    take it."""
    try:
        po.ModelBank([_cut_spec(kind, names)])
    except (TypeError, ValueError):
        return False
    return True


def _combinations(kind: str) -> list[tuple[str, ...]]:
    """Every combination of the stream parameters ``kind`` runs under, each
    given or not, and the session as a gap or a reset or neither: probed, so
    a stream draws only one it takes."""
    out = []
    for clock in ((), ("clock",)):
        for weight in ((), ("weight",)):
            for session in ((), ("session",), ("session reset",)):
                for embargo in ((), ("embargo",)):
                    names = clock + weight + session + embargo
                    if names and _cuts(kind, names):
                        out.append(names)
    return out


COMBINATIONS = {kind: _combinations(kind) for kind in UNCLOCKED_KINDS}


def test_each_kind_takes_every_stream_parameter_but_those_it_refuses():
    """The probe is no false skip: the seven kinds are the ones no sweep
    cut, and each runs under every stream parameter but the ones `REFUSED`
    names, all together too."""
    assert UNCLOCKED_KINDS == [
        "audit",
        "corrchange",
        "deco",
        "hmm",
        "kmeans",
        "rcov",
        "seqtest",
    ]
    for kind in UNCLOCKED_KINDS:
        taken = {name for names in COMBINATIONS[kind] for name in names}
        assert taken == set(STREAM_PARAMETERS) - REFUSED.get(kind, set()), kind
        for name in REFUSED.get(kind, ()):
            assert not _cuts(kind, (name,)), (kind, name)
        fullest = tuple(n for n in STREAM_PARAMETERS if n in taken and n != "session reset")
        assert fullest in COMBINATIONS[kind], (kind, fullest)


#: `rcov`'s weights: a row is a return or it is not, so it takes 0 or 1 and
#: refuses any other usable weight; NaN, an infinity, a magnitude past the
#: bound and a null are missing (`test_a_weight_past_the_bound_is_missing`).
_RCOV_WEIGHTS = st.sampled_from(
    [0.0, -0.0, 1.0, float("nan"), float("inf"), float("-inf"), 1e101, -1e101, None]
)


@st.composite
def cut_streams(draw, kind: str, sign: float | None = None, offset: float = 0.0):
    """`kind_streams`, with a clock ``t`` that steps forward by 0 to 1e4 and
    back by more than `MINIMUM` now and then, so a clocked run restarts; a
    weight ``w`` drawn as `streams` draws it; and a session ``s`` that
    changes now and then. The group key ``g`` only grows, and `rcov` closes
    each group as the next opens; a step back across a close is the next
    group's first row, which has no clock to step back from."""
    df = draw(kind_streams(kind, sign=sign, offset=offset))
    n = df.height
    steps = draw(
        st.lists(
            st.sampled_from([0.0, 0.5, 1.0, 7.0, 1e4, -(MINIMUM + 2.0), -1e4]),
            min_size=n,
            max_size=n,
        )
    )
    flips = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    weights = _RCOV_WEIGHTS if kind == "rcov" else _weights
    return df.with_columns(
        t=pl.Series(np.cumsum([0.0, *steps[1:]]), dtype=pl.Float64),
        w=pl.Series(draw(st.lists(weights, min_size=n, max_size=n)), dtype=pl.Float64),
        s=pl.Series([f"s{k}" for k in np.cumsum(flips)]),
    )


def _fields(out: pl.DataFrame) -> pl.DataFrame:
    """Every field of the spec's output, `coef` included: `_cut_spec` writes
    one on every row, so a chunk's end writes no extra one."""
    return out.select("m").unnest("m")


def _skipped(df: pl.DataFrame, kind: str, spec: dict) -> list[bool]:
    """The rows the bank skips: a feature missing, or the weight where the
    spec reads one. An audit reads every row: what the others skip is what
    it counts."""
    cond = pl.lit(False)
    if kind != "audit":
        for f in spec["features"]:
            cond = cond | _missing(f)
        if spec.get("weight"):
            cond = cond | _missing(spec["weight"])
    return df.select(cond).to_series().to_list()


def _cut_holds(kind: str, names: tuple[str, ...], df: pl.DataFrame, chunks, splits) -> None:
    """Under ``names``: the stream cut at ``chunks`` (chunk lengths) gives
    the numbers and the state one chunk does, a save and load at each of
    ``splits`` is transparent, every number is finite or null, and a skipped
    row reports nothing."""
    spec = _cut_spec(kind, names)
    whole = po.ModelBank([spec])
    one = _fields(whole.fit_predict(df))
    for chunk in chunks:
        bank = po.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)])
        assert one.equals(_fields(many), null_equal=True), f"{chunk}-row chunks moved a number"
        if (state := _state(whole, kind)) is not None:
            assert state.equals(_state(bank, kind), null_equal=True), "chunking moved the state"
    for split in splits:
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, split))
        b = po.ModelBank.load_bytes(a.save_bytes(), specs=[spec])
        rest = df.slice(split)
        got = _fields(b.fit_predict(rest))
        assert _fields(a.fit_predict(rest)).equals(got, null_equal=True), f"a load at {split}"
        assert one.slice(split).equals(got, null_equal=True), f"a load at {split}"
        if (state := _state(a, kind)) is not None:
            assert state.equals(_state(b, kind), null_equal=True), f"a load at {split}"
    _finite_or_null(one)
    for i, skip in enumerate(_skipped(df, kind, spec)):
        if skip:
            present = [c for c in one.columns if one[c][i] is not None]
            assert not present, f"row {i} was skipped but reported {present}"


@pytest.mark.parametrize("kind", UNCLOCKED_KINDS)
@pytest.mark.parametrize("sign", list(SIGNS))
def test_the_properties_hold_under_the_stream_parameters(kind, sign):
    """`TestEveryOtherKind`'s properties under a combination of a clock that
    steps back past the restart bound, a weight, a session (a gap or a
    reset) and an embargo, each drawn from those the kind takes
    (`COMBINATIONS`), on adversarial streams of each sign."""

    def prop(data, chunk):
        names = data.draw(st.sampled_from(COMBINATIONS[kind]), label="parameters")
        df = data.draw(cut_streams(kind, **SIGNS[sign]), label="stream")
        split = data.draw(st.integers(1, df.height - 1), label="split")
        _cut_holds(kind, names, df, [chunk], [split])

    _given(sign, prop, data=st.data(), chunk=CHUNKS)


#: The long stream's events: the clock steps back past the restart bound at
#: `RESTARTS`, inside `rcov`'s groups of 100 rows; the session changes at
#: `CHANGES`; the clock leaps past `gap_cap` at `LEAPS`.
RESTARTS = (150, 420)
CHANGES = (200, 350, 500)
LEAPS = (90, 260)


def _long_stream(kind: str, n: int = 600) -> pl.DataFrame:
    """600 ordinary rows with every event: two restarts, three session
    changes, two gaps past the cap, weights of 0 and missing ones, a missing
    feature now and then, and a group key that grows every 100 rows."""
    r = np.random.default_rng(278)
    i = np.arange(n)
    steps = np.where(np.isin(i, LEAPS), 400.0, r.choice([0.0, 1.0, 2.5], size=n))
    steps[list(RESTARTS)] = -(MINIMUM + 2.0)
    steps[0] = 0.0
    x0 = np.where(i % 37 == 11, np.nan, r.normal(size=n))
    w = np.where(i % 23 == 4, 0.0, r.uniform(0.5, 2.0, size=n))
    if kind == "rcov":
        w = np.where(i % 23 == 4, 0.0, 1.0)  # a return or not (`_RCOV_WEIGHTS`)
    df = pl.DataFrame(
        {
            "x0": x0,
            "x1": r.normal(size=n) + np.where(i >= 300, 2.0, 0.0),
            "y": r.normal(size=n),
            "g": i // 100,
            "t": np.cumsum(steps),
            "w": pl.Series(w).scatter([31, 333], None),
            "s": np.searchsorted(CHANGES, i, side="right"),
        }
    )
    if kind == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 0).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


def _fullest(kind: str, session: str) -> tuple[str, ...]:
    """Every stream parameter ``kind`` takes, with the session as ``session``."""
    taken = {name for names in COMBINATIONS[kind] for name in names}
    keep = {*taken} - ({"session", "session reset"} - {session})
    return tuple(n for n in STREAM_PARAMETERS if n in keep)


@pytest.mark.parametrize("session", ["session", "session reset"])
@pytest.mark.parametrize("kind", UNCLOCKED_KINDS)
def test_one_seven_and_six_hundred_chunks_and_a_save_at_each_event(kind, session):
    """Hard rule 3 on a long stream with every event, under every stream
    parameter the kind takes: 1, 7 and 600 chunks give the same numbers and
    the same state, and a save and load is transparent at a restart's row,
    the rows beside it, a session change and a gap past the cap. The run
    under them differs from the plain spec's, and the summary counts each
    restart and each reset, so the events took effect."""
    df = _long_stream(kind)
    names = _fullest(kind, session)
    n = df.height
    chunks = [n, -(-n // 7), 1]
    splits = [1, RESTARTS[0] - 1, RESTARTS[0], RESTARTS[0] + 1, CHANGES[0], LEAPS[1], n - 1]
    with warnings.catch_warnings():
        # A restart's and a reset's readiness notices are expected here.
        warnings.simplefilter("ignore")
        _cut_holds(kind, names, df, chunks, splits)
        bank = po.ModelBank([_cut_spec(kind, names)])
        cut = _fields(bank.fit_predict(df))
        plain_bank = po.ModelBank([_kind_spec(kind)])
        plain = _fields(plain_bank.fit_predict(df))
    moved = _moved(plain, cut)
    print(f"{kind} under {names}: {moved} moved against the plain spec")
    if kind == "audit":
        # Its rows count what it read, whatever the parameters; a restart
        # starts its runs of equal values over, in its tables.
        assert not _state(bank, kind).equals(_state(plain_bank, kind))
    else:
        assert [c for c in moved if c in plain.columns], f"{names} moved nothing on {kind}"
    if kind == "rcov":
        # Its rows hold only `weight_sum`; the estimate is each closed row.
        assert not _state(bank, kind).equals(_state(plain_bank, kind))
    if kind != "rcov":  # its summary keeps only the group still open
        resets = bank.summary()["resets"].sum()
        want = len(RESTARTS) + (len(CHANGES) if session == "session reset" else 0)
        print(f"{kind}: {resets} resets, {want} events")
        assert resets == want


# A weight past the bound is missing, as the README says of a feature, a
# target or a weight ("NaN, ±inf and any magnitude above 1e100 count as
# null"). `bank.rs` checked a weight's sign, and `rcov`'s 0 or 1, on every
# *finite* value, so -1e101 was refused as a negative weight on every kind,
# and 1e101 as neither 0 nor 1 on `rcov` (task 278, batch 13 found it; D-WB
# fixed it: the checks read the usable values only).
def _weight_cases() -> list:
    cases = []
    for kind in sorted(MINIMAL):
        if kind in ("seqtest", "audit"):
            continue  # neither takes a weight column
        for value in (float("inf"), float("-inf"), 1e101, -1e101):
            cases.append(pytest.param(kind, value, id=f"{kind}-{value:g}"))
    return cases


@pytest.mark.parametrize(("kind", "value"), _weight_cases())
def test_a_weight_past_the_bound_is_missing(kind, value):
    """A weight of ±inf or of a magnitude past `INPUT_BOUND` gives the
    numbers a null weight on that row gives."""
    n = 12
    rng = np.random.default_rng(9)
    df = pl.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "y": rng.normal(size=n),
            "g": np.arange(n) // 4,
            "w": [1.0] * n,
        }
    )
    if kind == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 0).then(pl.lit("a")).otherwise(pl.lit("b")))
    spec = {**_kind_spec(kind), "weight": "w"}
    w = df["w"].to_list()
    w[8] = None
    null = po.ModelBank([spec]).fit_predict(df.with_columns(w=pl.Series(w, dtype=pl.Float64)))
    w[8] = value
    got = po.ModelBank([spec]).fit_predict(df.with_columns(w=pl.Series(w, dtype=pl.Float64)))
    assert got["m"].equals(null["m"], null_equal=True)
