"""`output_index` / `coef_index`: reaching outputs without constructing names.

The output field names are a string grammar (`pred_y__r0.5@h500`), and before
these existed the only way to a grid slot was to rebuild that string by hand —
including the float rendering, which is an implementation detail. The index is
produced by the *same Rust code that renders the names*, so these tests are
what guarantees the metadata and the strings cannot drift apart.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def grid_spec(**kw):
    d = dict(
        targets=["y", "z"],
        features=["x0", "x1"],
        feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
        ridge=[1e-6, 0.5],
        halflife=[100.0, 500.0],
        min_periods=3.0,
        emit_sigma=True,
        resid_quantiles=[0.05, 0.95],
        conformal=0.9,
        emit_selected=True,
        emit_averaged=True,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


class TestOutputIndex:
    def test_field_column_is_output_fields_exactly(self):
        """The index IS the schema: same names, same order, nothing extra."""
        for spec in [
            grid_spec(),
            po.spec.holt("m", targets=["y"], halflife=50.0, min_periods=2.0),
            po.spec.lasso(
                "m",
                targets=["y"],
                features=["x0"],
                lasso_path=[0.1, 0.0],
                halflife=50.0,
                min_periods=2.0,
            ),
            po.spec.ew_cov(
                "m",
                features=["x0", "x1", "x2"],
                stats=["mean", "var", "cov", "corr"],
                halflife=50.0,
                min_periods=2.0,
            ),
        ]:
            idx = po.spec.output_index(spec)
            assert idx["field"].to_list() == po.spec.output_fields(spec)

    def test_filtering_replaces_string_construction(self):
        """The advertised workflow: filter metadata, get the exact name, use it
        on a real output — no string built anywhere."""
        spec = grid_spec()
        idx = po.spec.output_index(spec)
        name = idx.filter(
            (pl.col("kind") == "pred")
            & (pl.col("target") == "z")
            & (pl.col("ridge") == 0.5)
            & (pl.col("halflife") == 500.0)
            & (pl.col("feature_set") == "b")
        )["field"].item()

        rng = np.random.default_rng(0)
        n = 500
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
            }
        ).with_columns(y=pl.col("x0"), z=2 * pl.col("x1"))
        out = po.ModelBank([spec]).fit_predict(df)
        vals = out["m"].struct.field(name)
        assert vals.drop_nulls().len() > 0

    def test_every_pred_row_knows_its_target_and_instance(self):
        idx = po.spec.output_index(grid_spec())
        preds = idx.filter(pl.col("kind") == "pred")
        # Grid slots carry everything; selection outputs carry target only.
        grid = preds.filter(pl.col("halflife").is_not_null())
        assert grid["target"].null_count() == 0
        assert grid["ridge"].null_count() == 0
        assert set(grid["halflife"].to_list()) == {100.0, 500.0}
        sel = idx.filter(pl.col("kind").is_in(["pred_selected", "pred_averaged", "selected"]))
        assert sel.height == 6  # 3 kinds x 2 targets
        assert sel["target"].null_count() == 0

    def test_quantile_levels_are_machine_readable(self):
        idx = po.spec.output_index(grid_spec())
        q = idx.filter(pl.col("kind") == "absresid_q")
        assert set(q["quantile"].to_list()) == {0.05, 0.95}

    def test_lam_specs_report_lam_not_halflife(self):
        spec = grid_spec(halflife=None, lam=0.97, ridge=[1e-6])
        idx = po.spec.output_index(spec)
        assert idx["halflife"].null_count() == idx.height
        assert set(idx.filter(pl.col("kind") == "pred")["lam"].to_list()) == {0.97}

    def test_ew_cov_columns(self):
        spec = po.spec.ew_cov(
            "m", features=["a", "b"], stats=["std", "corr"], halflife=10.0, min_periods=2.0
        )
        idx = po.spec.output_index(spec)
        rows = {r["field"]: r["columns"] for r in idx.iter_rows(named=True)}
        assert rows["std_a"] == ["a"]
        assert rows["corr_a_b"] == ["a", "b"]
        assert rows["n_eff"] is None

    def test_compact_float_rendering_is_reachable_without_knowing_it(self):
        """The whole point: a user never needs to know that 1e-300 renders as
        `1e-300` and 0.5 as `0.5`."""
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            ridge=[1e-300, 0.5],
            halflife=[100.0, 1e9],
            min_periods=2.0,
        )
        idx = po.spec.output_index(spec)
        name = idx.filter(
            (pl.col("kind") == "pred") & (pl.col("ridge") == 1e-300) & (pl.col("halflife") == 1e9)
        )["field"].item()
        assert name == "pred_y__r1e-300@h1e9"


class TestCoefIndex:
    def test_positions_recover_the_generating_coefficients(self):
        rng = np.random.default_rng(1)
        n = 3000
        df = pl.DataFrame(
            {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)}
        ).with_columns(y=2 * pl.col("x0") - pl.col("x1") + 3)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=[1e-9, 0.5],
            halflife=1e9,
            min_periods=3.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        coef = out["m"].struct.field("coef").to_list()[-1]
        ci = po.spec.coef_index(spec)
        assert ci.height == len(coef)
        for term, want in [("intercept", 3.0), ("x0", 2.0), ("x1", -1.0)]:
            pos = ci.filter((pl.col("ridge") == 1e-9) & (pl.col("term") == term))["position"].item()
            assert coef[pos] == pytest.approx(want, abs=1e-3), term

    def test_holt_terms(self):
        spec = po.spec.holt("m", targets=["a", "b"], halflife=50.0, min_periods=2.0)
        ci = po.spec.coef_index(spec)
        assert ci["term"].to_list() == ["level", "trend", "level", "trend"]
        assert ci["target"].to_list() == ["a", "a", "b", "b"]

    def test_no_intercept(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            halflife=50.0,
            min_periods=2.0,
            add_intercept=False,
        )
        assert po.spec.coef_index(spec)["term"].to_list() == ["x0"]

    def test_ew_cov_is_refused(self):
        spec = po.spec.ew_cov("m", features=["a"], stats=["mean"], halflife=10.0, min_periods=2.0)
        with pytest.raises(ValueError, match="statistics, not coefficients"):
            po.spec.coef_index(spec)


class TestUnpackWithSpec:
    def test_spec_resolves_prefix_targets_the_heuristic_cannot(self):
        """Targets `y` and `y2` where `y2`'s fields could confuse a prefix
        matcher: with the spec, resolution is exact."""
        from polars_online import eval as ev

        rng = np.random.default_rng(2)
        n = 400
        df = pl.DataFrame({"x0": rng.standard_normal(n)}).with_columns(
            y=pl.col("x0"), y2=2 * pl.col("x0")
        )
        spec = po.spec.ewridge(
            "m",
            targets=["y", "y2"],
            features=["x0"],
            halflife=50.0,
            min_periods=3.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        long = ev.unpack(out, "m", spec=spec)
        got = dict(long.group_by("slot").agg(pl.col("target").first()).iter_rows())
        assert got == {"pred_y": "y", "pred_y2": "y2"}
