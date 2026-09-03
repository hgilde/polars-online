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


class TestMultiTargetExpressions:
    """E9: the expression namespace accepts extra targets.

    Multi-target specs share one `X'X`, so fitting several horizons in one call
    is much cheaper than one expression per target.
    """

    def test_matches_a_multi_target_bank(self):
        df, _ = synthetic(seed=91, n_groups=2, n_rows=200, k=2, n_targets=2, null_frac=0.0)
        kw = dict(
            clock="t",
            halflife=300.0,
            max_dclock=50.0,
            weight="w",
            min_periods=5.0,
            max_rows_between_solves=1,
        )
        expr = df.select(
            pl.col("y0")
            .online.ewridge(features=["x0", "x1"], extra_targets=["y1"], **kw)
            .over("group")
        ).unnest("y0")
        bank = (
            po.ModelBank(
                [
                    po.spec.ewridge(
                        "m",
                        targets=["y0", "y1"],
                        features=["x0", "x1"],
                        group="group",
                        **kw,
                    )
                ]
            )
            .fit_predict(df)
            .select("m")
            .unnest("m")
        )
        _assert_same(bank, expr)

    def test_works_for_every_model(self):
        df, _ = synthetic(seed=92, n_groups=1, n_rows=150, k=2, n_targets=2, null_frac=0.0)
        cases = [
            ("ewridge", {"max_rows_between_solves": 1}),
            ("rls", {"ridge": 1.0}),
            ("kalman", {"coef_halflife": 100.0}),
            ("huber", {"max_rows_between_solves": 1}),
            ("sgd", {"learning_rate": 0.01}),
            ("pa", {}),
        ]
        for model, extra in cases:
            out = df.select(
                getattr(pl.col("y0").online, model)(
                    features=["x0", "x1"],
                    extra_targets=["y1"],
                    halflife=300.0,
                    min_periods=5.0,
                    **extra,
                )
            ).unnest("y0")
            assert "pred_y0" in out.columns and "pred_y1" in out.columns, model

    def test_duplicate_target_is_rejected(self):
        df, _ = synthetic(seed=93, n_groups=1, n_rows=20, k=1, n_targets=2, null_frac=0.0)
        with pytest.raises(Exception, match="already the target"):
            df.select(
                pl.col("y0").online.ewridge(features=["x0"], extra_targets=["y0"], halflife=100.0)
            )
        with pytest.raises(Exception, match="duplicates"):
            df.select(
                pl.col("y0").online.ewridge(
                    features=["x0"], extra_targets=["y1", "y1"], halflife=100.0
                )
            )


class TestExpressionFeatures:
    """Features may be expressions, not only column names (docs/IMPROVEMENTS.md
    U1). Under ``.over(group)`` they are evaluated per group, so a lag of the
    target stays inside its group -- the natural way to write an AR term."""

    def test_lagged_expression_equals_a_materialized_lag(self):
        df, _ = synthetic(seed=21, n_groups=3, n_rows=200, k=2)
        lagged = df.with_columns(pl.col("y0").shift(1).over("group").alias("y0_lag"))
        bank = _bank_out(lagged, ["x0", "y0_lag"])
        expr = df.select(
            pl.col("y0")
            .online.ewridge(features=["x0", pl.col("y0").shift(1).alias("y0_lag")], **COMMON)
            .over("group")
        ).unnest("y0")
        _assert_same(expr, bank)
        assert "coef" in expr.columns

    def test_expression_without_a_name_is_rejected(self):
        df, _ = synthetic(seed=22, n_groups=1, n_rows=20, k=1)
        with pytest.raises(ValueError, match="determinable output name"):
            pl.col("y0").online.ewridge(features=[pl.col("^x.*$")], halflife=100.0)
        # `.alias` settles it -- and the feature name is the alias.
        out = df.select(
            pl.col("y0").online.ewridge(
                features=[(pl.col("x0") * pl.col("x0")).alias("x0_sq")],
                halflife=100.0,
                min_periods=2.0,
            )
        ).unnest("y0")
        assert "pred_y0" in out.columns

    def test_non_feature_type_is_rejected(self):
        with pytest.raises(TypeError, match="column names or expressions"):
            pl.col("y0").online.ewridge(features=[3], halflife=100.0)

    def test_column_used_twice_does_not_collide(self):
        # The packed struct names its fields positionally, so the same column
        # can serve two roles (here feature and weight) without a duplicate
        # field name.
        df, _ = synthetic(seed=23, n_groups=2, n_rows=100, k=2)
        df = df.with_columns(pl.col("x1").abs().alias("x1"))
        kw = {**COMMON, "weight": "x1"}
        expr = df.select(
            pl.col("y0").online.ewridge(features=["x0", "x1"], **kw).over("group")
        ).unnest("y0")
        _assert_same(expr, _bank_out(df, ["x0", "x1"], **kw))

    def test_ew_cov_accepts_expressions(self):
        df, _ = synthetic(seed=24, n_groups=2, n_rows=100, k=2)
        out = df.select(
            pl.col("x0")
            .online.ew_cov([pl.col("x1").shift(1).alias("x1_lag")], halflife=50.0, min_periods=3.0)
            .over("group")
        ).unnest("x0")
        assert any(c.startswith("cov") or c.startswith("corr") for c in out.columns), out.columns


class TestEveryEmitFlagThroughThePlugin:
    """docs/IMPROVEMENTS.md C1: the plugin declares its output struct up
    front and polars checks the declaration against what the bank realizes,
    so every field dtype has to be right -- `drift_*` is bool and
    `selected_*` is str, which a name-prefix rule once declared as f64. Each
    flag runs through `.over()` and must match the bank."""

    CASES = [
        {"emit_sigma": True},
        {"emit_resid_z": True},
        {"emit_selected": True, "ridge": [1e-6, 1.0]},
        {"emit_averaged": True, "ridge": [1e-6, 1.0]},
        {"emit_metrics": True},
        {"resid_quantiles": [0.5, 0.9]},
        {"emit_autocorr": True},
        {"emit_drift": True},
        {"emit_drift": True, "drift_action": "reset"},
        {"emit_selected": True, "emit_drift": True, "halflife": [100.0, 300.0]},
    ]

    @pytest.mark.parametrize("kw", CASES, ids=[",".join(c) for c in CASES])
    def test_matches_the_bank(self, kw):
        df, _ = synthetic(seed=31, n_groups=2, n_rows=150, k=2)
        feats = ["x0", "x1"]
        expr = _expr_out(df, feats, **kw)
        bank = _bank_out(df, feats, **kw)
        assert expr.schema == bank.schema
        # Non-float fields are compared as-is; `_assert_same` is for floats.
        for c in expr.columns:
            if expr.schema[c] in (pl.Boolean, pl.String):
                assert expr[c].equals(bank[c]), c
        floats = [c for c in expr.columns if expr.schema[c] == pl.Float64]
        _assert_same(expr.select(floats), bank.select(floats))

    def test_declared_dtypes_are_the_index(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            halflife=100.0,
            ridge=[1e-6, 1.0],
            emit_selected=True,
            emit_drift=True,
        )
        idx = po.spec.output_index(spec)
        assert set(idx["dtype"]) == {"f64", "bool", "str", "list[f64]"}
        df, _ = synthetic(seed=32, n_groups=1, n_rows=60, k=1)
        out = df.select(
            pl.col("y0").online.ewridge(
                features=["x0"],
                halflife=100.0,
                ridge=[1e-6, 1.0],
                emit_selected=True,
                emit_drift=True,
            )
        ).unnest("y0")
        as_polars = {
            "f64": pl.Float64,
            "bool": pl.Boolean,
            "str": pl.String,
            "list[f64]": pl.List(pl.Float64),
        }
        declared = {f: as_polars[d] for f, d in zip(idx["field"], idx["dtype"], strict=True)}
        assert dict(out.schema) == declared


class TestTheExpressionWarnsThatItRunsInMemory:
    """The expression form is O(data) -- polars hands it the whole column in
    either engine -- and it says so on every call, so the difference from
    `lf.online.fit_predict` is learned at the call site, not from a memory
    profile (docs/PLAN.md section 6)."""

    FRAME = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x0": [1.0, 3.0, 2.0, 5.0]})

    @staticmethod
    def _calls():
        ns = pl.col("y").online
        common = dict(features=["x0"], halflife=2.0)
        # Keyed by spec `type`, valued by the namespace method's name and call.
        return {
            "ew_ridge": ("ewridge", lambda: ns.ewridge(**common)),
            "rls": ("rls", lambda: ns.rls(**common)),
            "lasso": ("lasso", lambda: ns.lasso(lasso_path=[1.0, 0.1], **common)),
            "kalman": ("kalman", lambda: ns.kalman(coef_halflife=100.0, **common)),
            "huber": ("huber", lambda: ns.huber(**common)),
            "quantile": ("quantile", lambda: ns.quantile(quantile=0.5, **common)),
            "ftrl": ("ftrl", lambda: ns.ftrl(**common)),
            "sgd": ("sgd", lambda: ns.sgd(learning_rate=0.01, **common)),
            "pa": ("pa", lambda: ns.pa(**common)),
            "holt": ("holt", lambda: ns.holt(halflife=2.0)),
            "ew_cov": ("ew_cov", lambda: ns.ew_cov(others=["x0"], halflife=2.0)),
        }

    def test_every_method_warns_and_names_the_spelling_that_streams(self):
        calls = self._calls()
        assert set(calls) == set(po._polars_online.model_kinds())
        for kind, (method, call) in calls.items():
            with pytest.warns(po.InMemoryExpressionWarning) as rec:
                call()
            assert len(rec) == 1, kind
            text = str(rec[0].message)
            assert f"pl.col('y').online.{method}(...) runs on the whole column" in text
            assert "lf.online.fit_predict([spec]) is O(chunk)" in text
            assert "category=polars_online.InMemoryExpressionWarning" in text
            # stacklevel: the warning is attributed to the caller, not to _expr.py.
            assert rec[0].filename == __file__, kind

    def test_the_expression_still_works_and_matches_the_bank(self):
        # A warning, not an error: the result is the bank's, for a frame in memory.
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=2.0)
        with pytest.warns(po.InMemoryExpressionWarning):
            out = self.FRAME.select(pl.col("y").online.ewridge(features=["x0"], halflife=2.0))
        assert out["y"].equals(po.fit_predict(self.FRAME, [spec])["m"])

    def test_silenced_by_category(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
            pl.col("y").online.ewridge(features=["x0"], halflife=2.0)

    def test_a_usage_error_is_raised_before_the_warning(self):
        import warnings

        # An invalid call gets its error and nothing else.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="group is not an expression parameter"):
                pl.col("y").online.ewridge(features=["x0"], halflife=2.0, group="g")

    def test_shown_by_default_from_a_module(self, tmp_path):
        # A DeprecationWarning is hidden by default unless raised in __main__,
        # which is exactly the wrong way round for a pipeline module: that is
        # where the streaming spelling matters. The category is a UserWarning
        # so it is shown from anywhere; a DeprecationWarning emitted at the
        # same place is the control.
        import subprocess
        import sys
        import textwrap

        assert issubclass(po.InMemoryExpressionWarning, UserWarning)
        assert not issubclass(po.InMemoryExpressionWarning, DeprecationWarning)
        (tmp_path / "pipeline.py").write_text(
            textwrap.dedent("""
                import warnings
                import polars as pl
                import polars_online  # noqa: F401

                def build():
                    warnings.warn("control: a DeprecationWarning here", DeprecationWarning)
                    return pl.col("y").online.ewridge(features=["x0"], halflife=2.0)
                """)
        )
        r = subprocess.run(
            [sys.executable, "-c", "import pipeline; pipeline.build()"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(tmp_path),
            check=False,
        )
        assert r.returncode == 0, r.stderr
        assert "InMemoryExpressionWarning" in r.stderr, r.stderr
        assert "pipeline.py:8" in r.stderr, r.stderr  # attributed to the caller's line
        assert "control" not in r.stderr, r.stderr
