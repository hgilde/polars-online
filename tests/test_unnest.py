"""`online.unnest` and `spec.coef_fields`: a bank's output flat, with the
coefficients named.

The semantic pin is the last test: with a solve on every row, the next row's
`pred` is the current row's `coef` applied to the next row's features, so
`pred[t+1] == coef_..._intercept[t] + sum_j coef_..._xj[t] * xj[t+1]` for
every (target, combo, instance) -- true only if each named column is the
coefficient its name says, in the slot the models write it to.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

N = 300


def _frame(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x0, x1, x2 = (rng.standard_normal(N) for _ in range(3))
    return pl.DataFrame(
        {
            "t": np.arange(N),
            "g": ["a", "b"] * (N // 2),
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "y": 2.0 + 1.5 * x0 - 0.5 * x1 + 0.1 * rng.standard_normal(N),
            "z": -1.0 + 0.3 * x2 + 0.1 * rng.standard_normal(N),
        }
    )


OLS = po.spec.ewridge(
    "ols", targets=["y"], features=["x0", "x1"], halflife=50.0, min_periods=5.0, group="g"
)
GRID = po.spec.ewridge(
    "grid",
    targets=["y", "z"],
    features=["x0", "x1", "x2"],
    halflife=[50.0, 200.0],
    ridge=[0.0, 0.5],
    feature_sets={"a": ["x0"], "b": ["x0", "x1", "x2"]},
    min_periods=5.0,
)
COV = po.spec.ew_cov(
    "cov", features=["x0", "x1"], stats=["mean", "corr"], halflife=50.0, min_periods=2.0
)


class TestCoefFields:
    def test_columns_and_layout(self):
        cf = po.spec.coef_fields(GRID)
        assert cf.columns == [
            "field",
            "position",
            "name",
            "target",
            "halflife",
            "lam",
            "ridge",
            "feature_set",
            "lambda",
            "term",
        ]
        # Per instance: targets x combos slots, each the full term vector.
        combos = 4  # 2 feature sets x 2 ridges
        terms = 4  # intercept + 3 features
        assert cf.height == 2 * 2 * combos * terms
        one = cf.filter(pl.col("field") == "coef@h50")
        assert one["position"].to_list() == list(range(2 * combos * terms))
        assert one["term"].to_list()[:terms] == ["intercept", "x0", "x1", "x2"]
        assert one["target"].to_list() == ["y"] * (combos * terms) + ["z"] * (combos * terms)
        assert set(cf["field"]) == {"coef@h50", "coef@h200"}
        assert cf["halflife"].unique().sort().to_list() == [50.0, 200.0]
        assert cf["lam"].null_count() == cf.height

    def test_names_follow_the_field_grammar(self):
        """`coef_{target}_{term}` takes the combo and instance suffix of the
        `pred` field it belongs to, so the two sort together."""
        cf = po.spec.coef_fields(GRID)
        idx = po.spec.output_index(GRID).filter(pl.col("kind") == "pred")
        for row in cf.iter_rows(named=True):
            pred = idx.filter(
                (pl.col("target") == row["target"])
                & pl.col("ridge").eq_missing(row["ridge"])
                & pl.col("feature_set").eq_missing(row["feature_set"])
                & pl.col("halflife").eq_missing(row["halflife"])
            )["field"].item()
            suffix = pred.removeprefix(f"pred_{row['target']}")
            assert row["name"] == f"coef_{row['target']}_{row['term']}{suffix}"

    def test_single_instance_and_no_combo(self):
        assert po.spec.coef_fields(OLS)["name"].to_list() == [
            "coef_y_intercept",
            "coef_y_x0",
            "coef_y_x1",
        ]

    def test_holt_lasso_and_no_intercept(self):
        holt = po.spec.holt("h", targets=["y", "z"], halflife=50.0, min_periods=2.0)
        assert po.spec.coef_fields(holt)["name"].to_list() == [
            "coef_y_level",
            "coef_y_trend",
            "coef_z_level",
            "coef_z_trend",
        ]
        lasso = po.spec.lasso(
            "l",
            targets=["y"],
            features=["x0"],
            lasso_path=[1.0, 0.1],
            halflife=50.0,
            min_periods=2.0,
        )
        assert po.spec.coef_fields(lasso)["name"].to_list() == [
            "coef_y_intercept__l1",
            "coef_y_x0__l1",
            "coef_y_intercept__l0.1",
            "coef_y_x0__l0.1",
        ]
        bare = po.spec.ewridge(
            "m", targets=["y"], features=["x0"], halflife=50.0, min_periods=2.0, add_intercept=False
        )
        assert po.spec.coef_fields(bare)["name"].to_list() == ["coef_y_x0"]

    def test_ew_cov_has_none(self):
        assert po.spec.coef_fields(COV).is_empty()

    def test_coef_index_is_one_instance_of_it(self):
        cf = po.spec.coef_fields(GRID)
        ci = po.spec.coef_index(GRID)
        want = cf.filter(pl.col("field") == "coef@h50").select(
            pl.col("position").cast(pl.Int64), "target", "ridge", "feature_set", "lambda", "term"
        )
        assert ci.equals(want)

    def test_invalid_spec_is_refused(self):
        with pytest.raises(ValueError, match="halflife"):
            po.spec.coef_fields({**OLS, "halflife": -1.0})


class TestUnnest:
    def test_lazy_takes_the_struct_s_place(self):
        df = _frame()
        nested = df.online.fit_predict([OLS, GRID])
        flat = nested.lazy().online.unnest([OLS]).collect()
        # `ols` becomes its fields, in place; `grid` is untouched.
        assert flat.columns == [
            *df.columns,
            "pred_y",
            "resid_y",
            "n_eff",
            "coef_y_intercept",
            "coef_y_x0",
            "coef_y_x1",
            "grid",
        ]
        assert flat["grid"].dtype == nested["grid"].dtype
        for field in ("pred_y", "resid_y", "n_eff"):
            assert flat[field].equals(nested["ols"].struct.field(field))
        coef = nested["ols"].struct.field("coef")
        for pos, term in enumerate(["intercept", "x0", "x1"]):
            assert flat[f"coef_y_{term}"].equals(coef.list.get(pos).alias(f"coef_y_{term}"))

    def test_eager_and_function_forms_agree(self):
        nested = _frame().online.fit_predict([OLS, GRID])
        lazy = nested.lazy().online.unnest([OLS, GRID]).collect()
        assert nested.online.unnest([OLS, GRID]).equals(lazy)
        assert po.unnest(nested, [OLS, GRID]).equals(lazy)
        assert po.unnest(nested.lazy(), [OLS, GRID]).collect().equals(lazy)
        with pytest.raises(TypeError, match="DataFrame or LazyFrame"):
            po.unnest(nested.to_dict(), [OLS])  # type: ignore[call-overload]

    def test_grid_columns(self):
        flat = _frame().online.fit_predict([GRID]).online.unnest([GRID])
        names = po.spec.coef_fields(GRID)["name"].to_list()
        assert all(n in flat.columns for n in names)
        assert "coef_y_x1__b_r0.5@h200" in names
        # A feature outside its set is reported as 0 (the models scatter each
        # combo's solution into the full term vector).
        tail = flat.tail(1)
        assert tail["coef_y_x1__a_r0@h50"].item() == 0.0
        assert tail["coef_y_x0__a_r0@h50"].item() != 0.0

    def test_bank_and_state_path_carry_the_specs(self, tmp_path):
        df = _frame()
        bank = po.ModelBank([OLS])
        nested = bank.fit_predict(df)
        bank.save(tmp_path / "state.bin")
        want = nested.online.unnest([OLS])
        assert nested.online.unnest(bank).equals(want)
        assert nested.online.unnest(tmp_path / "state.bin").equals(want)
        assert nested.lazy().online.unnest(str(tmp_path / "state.bin")).collect().equals(want)

    def test_parquet_written_nested_comes_back_flat(self, tmp_path):
        """The CLI's output is the nested struct; `scan_parquet` + `unnest` is
        how it is read back with the coefficients named."""
        df = _frame()
        df.online.fit_predict([OLS]).write_parquet(tmp_path / "out.parquet")
        flat = (
            pl.scan_parquet(tmp_path / "out.parquet")
            .online.unnest([OLS])
            .select("t", "g", "^coef_.*$")
            .collect()
        )
        assert flat.columns == ["t", "g", "coef_y_intercept", "coef_y_x0", "coef_y_x1"]
        assert flat.height == N

    def test_ew_cov_unnests_to_its_statistics(self):
        flat = _frame().online.fit_predict([COV]).online.unnest([COV])
        assert [c for c in flat.columns if c not in _frame().columns] == po.spec.output_fields(COV)

    def test_holt(self):
        holt = po.spec.holt("h", targets=["y"], halflife=50.0, min_periods=2.0)
        flat = _frame().online.fit_predict([holt]).online.unnest([holt])
        assert {"coef_y_level", "coef_y_trend"} <= set(flat.columns)

    def test_errors_while_the_plan_is_built(self, tmp_path):
        df = _frame()
        nested = df.online.fit_predict([OLS])
        with pytest.raises(ValueError, match="no column.*'grid'"):
            nested.lazy().online.unnest([GRID])
        with pytest.raises(ValueError, match="'t' is Int64, not spec 't'"):
            nested.lazy().online.unnest([{**OLS, "name": "t"}])
        with pytest.raises(ValueError, match="lacks the field.*coef@h50"):
            nested.lazy().online.unnest([{**OLS, "halflife": [50.0, 200.0]}])
        with pytest.raises(ValueError, match="given twice"):
            nested.lazy().online.unnest([OLS, OLS])
        with pytest.raises(ValueError, match="halflife"):
            nested.lazy().online.unnest([{**OLS, "halflife": -1.0}])
        with pytest.raises(TypeError, match="specs, a ModelBank or the path"):
            nested.lazy().online.unnest([OLS, "ols"])  # type: ignore[list-item]
        with pytest.raises(FileNotFoundError):
            nested.lazy().online.unnest(tmp_path / "missing.bin")

    def test_two_specs_with_the_same_fields_are_polars_duplicate_error(self):
        other = {**OLS, "name": "ols2"}
        nested = _frame().online.fit_predict([OLS, other])
        with pytest.raises(pl.exceptions.DuplicateError):
            nested.online.unnest([OLS, other])
        with pytest.raises(pl.exceptions.DuplicateError):
            nested.lazy().online.unnest([OLS, other]).collect()
        # The documented way out: prefix the fields of one of them first.
        renamed = nested.with_columns(pl.col("ols2").name.prefix_fields("ols2_"))
        assert "ols2_pred_y" in renamed["ols2"].struct.fields
        assert "pred_y" in renamed.online.unnest([OLS]).columns


def test_named_coefficients_predict_the_next_row():
    """`pred[t+1] == coef[t] . x[t+1]` per named column, over the whole grid:
    targets, feature sets, ridges and halflives at once. A wrong slot order,
    or a name on the wrong list position, breaks this for some column."""
    spec = po.spec.ewridge(
        "grid",
        targets=["y", "z"],
        features=["x0", "x1", "x2"],
        halflife=[50.0, 200.0],
        ridge=[0.0, 0.5],
        feature_sets={"a": ["x0"], "b": ["x0", "x1", "x2"]},
        min_periods=5.0,
        coef_every=1,
        max_rows_between_solves=1,
    )
    df = _frame()
    flat = df.online.fit_predict([spec]).online.unnest([spec])
    cf = po.spec.coef_fields(spec)
    idx = po.spec.output_index(spec).filter(pl.col("kind") == "pred")
    checked = 0
    for pred in idx.iter_rows(named=True):
        rows = cf.filter(
            (pl.col("target") == pred["target"])
            & pl.col("ridge").eq_missing(pred["ridge"])
            & pl.col("feature_set").eq_missing(pred["feature_set"])
            & pl.col("halflife").eq_missing(pred["halflife"])
        )
        assert rows["term"].to_list() == ["intercept", "x0", "x1", "x2"]
        by_term = dict(zip(rows["term"], rows["name"], strict=True))
        want = flat[by_term["intercept"]][:-1].to_numpy() + sum(
            flat[by_term[f]][:-1].to_numpy() * df[f][1:].to_numpy() for f in ("x0", "x1", "x2")
        )
        got = flat[pred["field"]][1:].to_numpy()
        ok = ~np.isnan(got)
        assert ok.sum() > N // 2, pred["field"]
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9, atol=1e-9, err_msg=pred["field"])
        checked += 1
    assert checked == 2 * 4 * 2
