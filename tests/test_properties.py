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
    ("kalman", {"coef_halflife": 50.0}),
    ("lasso", {"lasso_path": [0.1, 0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
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


def build(model, extra, **kw):
    opts = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=100.0,
        halflife=20.0,
        weight="w",
        group="g",
        min_periods=2.0,
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
    keep = [c for c in out.select("m").unnest("m").columns if not c.startswith("coef")]
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
            if f.name.startswith("coef"):
                continue
            vals = np.array(
                [v for v in out["m"].struct.field(f.name).to_list() if v is not None],
                dtype=float,
            )
            assert np.isfinite(vals).all(), f"{f.name} produced a non-finite value"

    @SETTINGS
    @given(df=streams())
    def test_feature_or_weight_null_means_all_outputs_null(self, model, extra, df):
        df = binarize(df, model)
        out = po.ModelBank([build(model, extra)]).fit_predict(df)
        skipped = (
            df.select(pl.col("x0").is_null() | pl.col("x1").is_null() | pl.col("w").is_null())
            .to_series()
            .to_list()
        )
        neff = out["m"].struct.field("n_eff").to_list()
        for i, skip in enumerate(skipped):
            if skip:
                assert neff[i] is None, f"row {i} was skipped but reported n_eff"

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

    @SETTINGS
    @given(df=streams())
    def test_prediction_never_depends_on_the_current_target(self, model, extra, df):
        """Out-of-sample by construction (docs/PLAN.md hard rule 2): changing a
        row's target must not change that row's own prediction."""
        df = binarize(df, model)
        assume(df["y0"].is_not_null().any())
        spec = build(model, extra)
        base = po.ModelBank([spec]).fit_predict(df)
        idx = next(i for i, v in enumerate(df["y0"].to_list()) if v is not None)
        y = df["y0"].to_list()
        y[idx] = (0.0 if y[idx] else 1.0) if model == "ftrl" else y[idx] + 12345.0
        perturbed = po.ModelBank([spec]).fit_predict(
            df.with_columns(y0=pl.Series(y, dtype=pl.Float64))
        )
        field = next(f.name for f in base.schema["m"].fields if f.name.startswith("pred_"))
        a = base["m"].struct.field(field).to_list()[idx]
        b = perturbed["m"].struct.field(field).to_list()[idx]
        assert a == b or (a is None and b is None), (
            f"row {idx}: changing its own target changed its prediction ({a} -> {b})"
        )
