"""`rheon.unnest` and `spec.coef_fields`: a bank's output flat, with the
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

from polars_version import needs_polars
from rheon import polars as rp

TIER = "essential"

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


OLS = rp.spec.ewridge(
    "ols", targets=["y"], features=["x0", "x1"], half_life=50.0, min_weight=5.0, group="g"
)
GRID = rp.spec.ewridge(
    "grid",
    targets=["y", "z"],
    features=["x0", "x1", "x2"],
    half_life=[50.0, 200.0],
    ridge=[0.0, 0.5],
    feature_sets={"a": ["x0"], "b": ["x0", "x1", "x2"]},
    min_weight=5.0,
)
COV = rp.spec.ew_cov(
    "cov", features=["x0", "x1"], stats=["mean", "corr"], half_life=50.0, min_weight=2.0
)


class TestCoefFields:
    def test_columns_and_layout(self):
        cf = rp.spec.coef_fields(GRID)
        assert cf.columns == [
            "field",
            "position",
            "name",
            "target",
            "instance",
            "half_life",
            "lam",
            "ridge",
            "feature_set",
            "penalty",
            "cluster",
            "state",
            "class",
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
        assert cf["half_life"].unique().sort().to_list() == [50.0, 200.0]
        assert cf["lam"].null_count() == cf.height

    def test_names_follow_the_field_grammar(self):
        """`coef_{target}_{term}` takes the combo and instance suffix of the
        `pred` field it belongs to, so the two sort together."""
        cf = rp.spec.coef_fields(GRID)
        idx = rp.spec.output_index(GRID).filter(pl.col("kind") == "pred")
        for row in cf.iter_rows(named=True):
            pred = idx.filter(
                (pl.col("target") == row["target"])
                & pl.col("ridge").eq_missing(row["ridge"])
                & pl.col("feature_set").eq_missing(row["feature_set"])
                & pl.col("half_life").eq_missing(row["half_life"])
            )["field"].item()
            suffix = pred.removeprefix(f"pred_{row['target']}")
            assert row["name"] == f"coef_{row['target']}_{row['term']}{suffix}"

    def test_single_instance_and_no_combo(self):
        assert rp.spec.coef_fields(OLS)["name"].to_list() == [
            "coef_y_intercept",
            "coef_y_x0",
            "coef_y_x1",
        ]

    def test_holt_lasso_and_no_intercept(self):
        holt = rp.spec.holt("h", targets=["y", "z"], half_life=50.0, min_weight=2.0)
        assert rp.spec.coef_fields(holt)["name"].to_list() == [
            "coef_y_level",
            "coef_y_trend",
            "coef_z_level",
            "coef_z_trend",
        ]
        lasso = rp.spec.lasso(
            "l",
            targets=["y"],
            features=["x0"],
            penalties=[1.0, 0.1],
            half_life=50.0,
            min_weight=2.0,
        )
        assert rp.spec.coef_fields(lasso)["name"].to_list() == [
            "coef_y_intercept__p1",
            "coef_y_x0__p1",
            "coef_y_intercept__p0.1",
            "coef_y_x0__p0.1",
        ]
        bare = rp.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=50.0, min_weight=2.0, fit_intercept=False
        )
        assert rp.spec.coef_fields(bare)["name"].to_list() == ["coef_y_x0"]

    def test_ew_cov_has_none(self):
        assert rp.spec.coef_fields(COV).is_empty()

    def test_coef_index_is_one_instance_of_it(self):
        cf = rp.spec.coef_fields(GRID)
        ci = rp.spec.coef_index(GRID)
        want = cf.filter(pl.col("field") == "coef@h50").select(
            "position",
            "target",
            "ridge",
            "feature_set",
            "penalty",
            "cluster",
            "state",
            "class",
            "term",
        )
        assert ci.equals(want)

    def test_invalid_spec_is_refused(self):
        with pytest.raises(ValueError, match="half_life"):
            rp.spec.coef_fields({**OLS, "half_life": -1.0})


class TestUnnest:
    def test_lazy_takes_the_struct_s_place(self):
        df = _frame()
        nested = df.rheon.fit_predict([OLS, GRID])
        flat = nested.lazy().rheon.unnest([OLS]).collect()
        # `ols` becomes its fields, in place; `grid` is untouched.
        assert flat.columns == [
            *df.columns,
            "pred_y",
            "resid_y",
            "weight_sum",
            "settled_frac",
            "withheld_reason",
            "coef_y_intercept",
            "coef_y_x0",
            "coef_y_x1",
            # One share per coefficient, unnested the same way.
            "support_coef_y_intercept",
            "support_coef_y_x0",
            "support_coef_y_x1",
            "grid",
        ]
        assert flat["grid"].dtype == nested["grid"].dtype
        for field in ("pred_y", "resid_y", "weight_sum"):
            assert flat[field].equals(nested["ols"].struct.field(field))
        coef = nested["ols"].struct.field("coef")
        for pos, term in enumerate(["intercept", "x0", "x1"]):
            assert flat[f"coef_y_{term}"].equals(coef.list.get(pos).alias(f"coef_y_{term}"))

    def test_eager_and_function_forms_agree(self):
        nested = _frame().rheon.fit_predict([OLS, GRID])
        lazy = nested.lazy().rheon.unnest([OLS, GRID]).collect()
        assert nested.rheon.unnest([OLS, GRID]).equals(lazy)
        assert rp.unnest(nested, [OLS, GRID]).equals(lazy)
        assert rp.unnest(nested.lazy(), [OLS, GRID]).collect().equals(lazy)
        with pytest.raises(TypeError, match="DataFrame or LazyFrame"):
            rp.unnest(nested.to_dict(), [OLS])  # type: ignore[call-overload]

    def test_grid_columns(self):
        flat = _frame().rheon.fit_predict([GRID]).rheon.unnest([GRID])
        names = rp.spec.coef_fields(GRID)["name"].to_list()
        assert all(n in flat.columns for n in names)
        assert "coef_y_x1__b_r0.5@h200" in names
        # A feature outside its set is reported as 0 (the models scatter each
        # combo's solution into the full term vector).
        tail = flat.tail(1)
        assert tail["coef_y_x1__a_r0@h50"].item() == 0.0
        assert tail["coef_y_x0__a_r0@h50"].item() != 0.0

    def test_bank_and_state_path_carry_the_specs(self, tmp_path):
        df = _frame()
        bank = rp.ModelBank([OLS])
        nested = bank.fit_predict(df)
        bank.save(tmp_path / "state.bin")
        want = nested.rheon.unnest([OLS])
        assert nested.rheon.unnest(bank).equals(want)
        assert nested.rheon.unnest(tmp_path / "state.bin").equals(want)
        assert nested.lazy().rheon.unnest(str(tmp_path / "state.bin")).collect().equals(want)

    def test_parquet_written_nested_comes_back_flat(self, tmp_path):
        """The CLI's output is the nested struct; `scan_parquet` + `unnest` is
        how it is read back with the coefficients named."""
        df = _frame()
        df.rheon.fit_predict([OLS]).write_parquet(tmp_path / "out.parquet")
        flat = (
            pl.scan_parquet(tmp_path / "out.parquet")
            .rheon.unnest([OLS])
            .select("t", "g", "^coef_y_.*$")
            .collect()
        )
        assert flat.columns == ["t", "g", "coef_y_intercept", "coef_y_x0", "coef_y_x1"]
        assert flat.height == N

    def test_ew_cov_unnests_to_its_statistics(self):
        flat = _frame().rheon.fit_predict([COV]).rheon.unnest([COV])
        assert [c for c in flat.columns if c not in _frame().columns] == rp.spec.output_fields(COV)

    def test_holt(self):
        holt = rp.spec.holt("h", targets=["y"], half_life=50.0, min_weight=2.0)
        flat = _frame().rheon.fit_predict([holt]).rheon.unnest([holt])
        assert {"coef_y_level", "coef_y_trend"} <= set(flat.columns)

    def test_errors_while_the_plan_is_built(self, tmp_path):
        df = _frame()
        nested = df.rheon.fit_predict([OLS])
        with pytest.raises(ValueError, match='no output column of spec "grid"'):
            nested.lazy().rheon.unnest([GRID])
        with pytest.raises(ValueError, match="'t' is Int64, not the output column of spec \"t\""):
            nested.lazy().rheon.unnest([{**OLS, "name": "t"}])
        with pytest.raises(ValueError, match="lacks the fields.*coef@h50"):
            nested.lazy().rheon.unnest([{**OLS, "half_life": [50.0, 200.0]}])
        with pytest.raises(ValueError, match="given twice"):
            nested.lazy().rheon.unnest([OLS, OLS])
        with pytest.raises(ValueError, match="half_life"):
            nested.lazy().rheon.unnest([{**OLS, "half_life": -1.0}])
        with pytest.raises(TypeError, match="specs, a ModelBank or the path"):
            nested.lazy().rheon.unnest([OLS, "ols"])  # type: ignore[list-item]
        with pytest.raises(FileNotFoundError):
            nested.lazy().rheon.unnest(tmp_path / "missing.bin")

    def test_two_specs_with_the_same_fields_are_refused_by_name(self):
        """Two specs of one kind and one target share every field, eleven
        here. Polars' own refusal named one column and neither spec, after
        the plan was built; this one names the shared fields (the first five
        and a count), the specs, and the two ways out, while the plan is
        built, and is Polars' ``DuplicateError`` still, so a handler for
        Polars' own catches it."""
        other = {**OLS, "name": "ols2"}
        nested = _frame().rheon.fit_predict([OLS, other])
        want = (
            "rheon.unnest: pred_y, resid_y, weight_sum, settled_frac, withheld_reason and 6 more "
            'are fields of specs "ols" and "ols2", so the columns unnested would share a name. '
            'Pass separator="." to name each column after its spec, or take the fields you '
            'need with pl.col("ols").struct.field("pred_y").'
        )
        for unnest in (nested.rheon.unnest, nested.lazy().rheon.unnest):
            with pytest.raises(pl.exceptions.DuplicateError) as err:
                unnest([OLS, other])  # the lazy form refuses with no collect
            assert str(err.value) == want
        # Both ways out run: a separator, on either form ...
        flat = nested.rheon.unnest([OLS, other], separator=".")
        assert {"ols.pred_y", "ols2.pred_y", "ols2.weight_sum"} <= set(flat.columns)
        assert nested.lazy().rheon.unnest([OLS, other], separator=".").collect().equals(flat)
        # ... or the fields taken by name, beside the other spec unnested.
        picked = nested.with_columns(
            pl.col("ols2").struct.field("pred_y").alias("pred_y_ols2")
        ).drop("ols2")
        out = picked.rheon.unnest([OLS])
        assert out["pred_y_ols2"].equals(flat["ols2.pred_y"].alias("pred_y_ols2"))
        # The prefix the earlier way out added still works.
        renamed = nested.with_columns(pl.col("ols2").name.prefix_fields("ols2_"))
        assert "ols2_pred_y" in renamed["ols2"].struct.fields
        assert "pred_y" in renamed.rheon.unnest([OLS]).columns

    def test_a_field_named_like_a_column_of_the_frame_is_refused_by_name(self):
        """A column of the frame that a spec's field would repeat is named as
        the frame's; a separator already given is named as the one to change."""
        nested = _frame().rheon.fit_predict([OLS]).with_columns(weight_sum=pl.lit(1.0))
        with pytest.raises(pl.exceptions.DuplicateError) as err:
            nested.lazy().rheon.unnest([OLS])
        assert str(err.value).startswith(
            'rheon.unnest: weight_sum is a field of spec "ols" and a column of the input, '
            "so the columns unnested would share a name. "
            'Pass separator="." to name each column after its spec'
        )
        assert "ols.weight_sum" in nested.rheon.unnest([OLS], separator=".").columns
        clash = nested.with_columns(pl.lit(1.0).alias("ols.weight_sum"))
        with pytest.raises(pl.exceptions.DuplicateError) as err:
            clash.rheon.unnest([OLS], separator=".")
        assert str(err.value) == (
            'rheon.unnest: ols.weight_sum is a field of spec "ols" and a column of the input, '
            "so the columns unnested would share a name. Pass a separator other than '.'."
        )


def test_named_coefficients_predict_the_next_row():
    """`pred[t+1] == coef[t] . x[t+1]` per named column, over the whole grid:
    targets, feature sets, ridges and half-lives at once. A wrong slot order,
    or a name on the wrong list position, breaks this for some column."""
    spec = rp.spec.ewridge(
        "grid",
        targets=["y", "z"],
        features=["x0", "x1", "x2"],
        half_life=[50.0, 200.0],
        ridge=[0.0, 0.5],
        feature_sets={"a": ["x0"], "b": ["x0", "x1", "x2"]},
        min_weight=5.0,
        coef_every=1,
        max_rows_between_solves=1,
    )
    df = _frame()
    flat = df.rheon.fit_predict([spec]).rheon.unnest([spec])
    cf = rp.spec.coef_fields(spec)
    idx = rp.spec.output_index(spec).filter(pl.col("kind") == "pred")
    checked = 0
    for pred in idx.iter_rows(named=True):
        rows = cf.filter(
            (pl.col("target") == pred["target"])
            & pl.col("ridge").eq_missing(pred["ridge"])
            & pl.col("feature_set").eq_missing(pred["feature_set"])
            & pl.col("half_life").eq_missing(pred["half_life"])
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


class TestSeparator:
    """``separator=``, as Polars 2.0's ``DataFrame.unnest(separator=)``: each
    column is named ``{spec}{separator}{name}``, so two specs that write a
    field of one name unnest side by side (task 244, M-18 / C-10 / O-13)."""

    A = rp.spec.ewridge("a", targets=["y"], features=["x0"], half_life=50.0)
    B = rp.spec.ewridge("b", targets=["y"], features=["x1"], half_life=50.0)

    def _out(self) -> pl.DataFrame:
        return rp.ModelBank([self.A, self.B]).fit_predict(_frame())

    def test_two_specs_of_one_target_unnest_side_by_side(self):
        out = self._out()
        with pytest.raises(pl.exceptions.DuplicateError):
            out.rheon.unnest([self.A, self.B])
        flat = out.rheon.unnest([self.A, self.B], separator=".")
        assert flat.columns[:7] == list(_frame().columns)
        for name in ("a.pred_y", "a.coef_y_x0", "b.pred_y", "b.coef_y_x1", "b.support_coef_y_x1"):
            assert name in flat.columns, name
        lazy = out.lazy().rheon.unnest([self.A, self.B], separator=".").collect()
        assert lazy.equals(flat)
        assert rp.unnest(out, [self.A, self.B], separator=".").equals(flat)
        assert rp.unnest(out.lazy(), [self.A, self.B], separator=".").collect().equals(flat)
        # The values are the fields' own.
        assert flat["a.pred_y"].equals(out["a"].struct.field("pred_y").alias("a.pred_y"))
        with pytest.raises(TypeError, match="separator"):
            out.rheon.unnest([self.A], separator=1)

    @needs_polars("2.0.0", "DataFrame.unnest(separator=) is Polars 2.0's")
    def test_the_names_are_polars_own(self):
        """Every scalar field under the name Polars gives it; a ``coef``
        list, which Polars keeps whole, as one column per coefficient under
        the same prefix."""
        out = self._out()
        ours = out.rheon.unnest([self.A], separator="::")
        theirs = out.unnest("a", separator="::")
        scalar = [c for c in theirs.columns if c not in ("a::coef", "a::support_coef")]
        assert ours.select(scalar).equals(theirs.select(scalar))
        cov = rp.spec.ew_cov("c", features=["x0", "x1"], half_life=50.0)
        moments = rp.ModelBank([cov]).fit_predict(_frame())
        assert moments.rheon.unnest([cov], separator="_").equals(moments.unnest("c", separator="_"))
