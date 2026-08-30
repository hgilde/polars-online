"""Task 8: the expression plugin, and that expression == bank (docs/PLAN.md
section 9 class 6)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic

COMMON = dict(
    clock="t",
    halflife=300.0,
    max_dclock=50.0,
    session="session",
    session_gap=25.0,
    weight="w",
    min_periods=5.0,
    max_rows_between_solves=1,
    ridge=1e-6,
)


def _bank_out(df, features, targets=("y0",), **kw):
    spec = po.spec.ewridge(
        "m", targets=list(targets), features=features, group="group", **{**COMMON, **kw}
    )
    return po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")


def _expr_out(df, features, target="y0", **kw):
    return df.select(
        pl.col(target).online.ewridge(features=features, **{**COMMON, **kw}).over("group")
    ).unnest(target)


def _assert_same(a: pl.DataFrame, b: pl.DataFrame):
    # coef is emitted on each chunk's last row, so it is chunk-dependent by
    # design (docs/PLAN.md section 3); compare everything else.
    a = a.select(c for c in a.columns if not c.startswith("coef"))
    b = b.select(c for c in b.columns if not c.startswith("coef"))
    assert a.columns == b.columns
    for c in a.columns:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        both_nan = np.isnan(x) & np.isnan(y)
        assert (both_nan | (x == y)).all(), f"column {c} differs"


class TestExpressionEqualsBank:
    def test_single_target_over_group(self):
        df, _ = synthetic(seed=11, n_groups=3, n_rows=200, k=3)
        feats = ["x0", "x1", "x2"]
        _assert_same(_expr_out(df, feats), _bank_out(df, feats))

    def test_with_standardize_and_ridge_grid(self):
        df, _ = synthetic(seed=12, n_groups=2, n_rows=150, k=2)
        feats = ["x0", "x1"]
        kw = dict(standardize=True, ridge=[1e-6, 1.0])
        _assert_same(_expr_out(df, feats, **kw), _bank_out(df, feats, **kw))

    def test_halflife_grid(self):
        df, _ = synthetic(seed=13, n_groups=2, n_rows=150, k=2)
        feats = ["x0", "x1"]
        kw = dict(halflife=[50.0, 500.0])
        _assert_same(_expr_out(df, feats, **kw), _bank_out(df, feats, **kw))

    def test_single_group_no_over(self):
        df, _ = synthetic(seed=14, n_groups=1, n_rows=150, k=2)
        feats = ["x0", "x1"]
        e = df.select(pl.col("y0").online.ewridge(features=feats, **COMMON)).unnest("y0")
        _assert_same(e, _bank_out(df, feats))


class TestExpressionBehaviour:
    def test_output_schema_matches_fields(self):
        df, _ = synthetic(seed=15, n_groups=1, n_rows=20, k=2)
        spec = po.spec.ewridge("m", targets=["y0"], features=["x0", "x1"], **COMMON)
        out = df.select(pl.col("y0").online.ewridge(features=["x0", "x1"], **COMMON))
        assert out.schema["y0"].fields is not None
        got = [f.name for f in out.schema["y0"].fields]
        assert got == po.spec.output_fields(spec)

    def test_lazy_and_alias(self):
        df, _ = synthetic(seed=16, n_groups=2, n_rows=60, k=2)
        out = (
            df.lazy()
            .with_columns(
                pl.col("y0")
                .online.ewridge(features=["x0", "x1"], **COMMON)
                .over("group")
                .alias("fit")
            )
            .collect()
        )
        assert "fit" in out.columns
        assert out["fit"].struct.field("pred_y0").null_count() < out.height

    def test_groups_do_not_leak(self):
        # A per-group stream must not see other groups' rows: running one group
        # alone gives identical output.
        df, _ = synthetic(seed=17, n_groups=3, n_rows=120, k=2)
        allg = df.with_columns(
            pl.col("y0").online.ewridge(features=["x0", "x1"], **COMMON).over("group").alias("f")
        ).filter(pl.col("group") == "g1")
        solo_df = df.filter(pl.col("group") == "g1")
        solo = solo_df.with_columns(
            pl.col("y0").online.ewridge(features=["x0", "x1"], **COMMON).alias("f")
        )
        _assert_same(allg.select("f").unnest("f"), solo.select("f").unnest("f"))

    def test_bad_spec_raises(self):
        df, _ = synthetic(seed=18, n_groups=1, n_rows=10, k=1)
        with pytest.raises(Exception, match="mutually exclusive"):
            df.select(pl.col("y0").online.ewridge(features=["x0"], halflife=10.0, lam=0.9))
