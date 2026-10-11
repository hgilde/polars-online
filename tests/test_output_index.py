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
from test_model_registry import MINIMAL, fullest

TIER = "essential"


def grid_spec(**kw):
    d = dict(
        targets=["y", "z"],
        features=["x0", "x1"],
        feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
        ridge=[1e-6, 0.5],
        half_life=[100.0, 500.0],
        min_weight=3.0,
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
            po.spec.holt("m", targets=["y"], half_life=50.0, min_weight=2.0),
            po.spec.lasso(
                "m",
                targets=["y"],
                features=["x0"],
                penalties=[0.1, 0.0],
                half_life=50.0,
                min_weight=2.0,
            ),
            po.spec.ew_cov(
                "m",
                features=["x0", "x1", "x2"],
                stats=["mean", "var", "cov", "corr"],
                half_life=50.0,
                min_weight=2.0,
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
            & (pl.col("half_life") == 500.0)
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
        grid = preds.filter(pl.col("half_life").is_not_null())
        assert grid["target"].null_count() == 0
        assert grid["ridge"].null_count() == 0
        assert set(grid["half_life"].to_list()) == {100.0, 500.0}
        sel = idx.filter(pl.col("kind").is_in(["pred_selected", "pred_averaged", "selected"]))
        assert sel.height == 6  # 3 kinds x 2 targets
        assert sel["target"].null_count() == 0

    def test_a_single_combo_carries_its_machine_values(self):
        """With one ridge and one feature set the combo's metadata was
        empty, while a grid's carried both; and a single named set was
        dropped from a ridge grid's (review 2026-09-12, S5)."""
        one = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0, ridge=0.5)
        preds = po.spec.output_index(one).filter(pl.col("kind") == "pred")
        assert preds["ridge"].to_list() == [0.5]
        named = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=10.0,
            ridge=[0.5, 1.0],
            feature_sets={"a": ["x0"]},
        )
        preds = po.spec.output_index(named).filter(pl.col("kind") == "pred")
        assert preds["feature_set"].to_list() == ["a", "a"]

    def test_an_empty_list_of_feature_sets_is_refused(self):
        """``feature_sets = []`` beside a ridge grid rendered no slots for a
        model that emits one per ridge (V23)."""
        spec = po.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=10.0, ridge=[1e-6, 1e-3]
        )
        spec["model"]["feature_sets"] = []
        with pytest.raises(ValueError, match="feature_sets"):
            po.spec.output_index(spec)
        with pytest.raises(ValueError, match="feature_sets"):
            po.ModelBank([spec])

    def test_quantile_levels_are_machine_readable(self):
        idx = po.spec.output_index(grid_spec())
        q = idx.filter(pl.col("kind") == "abs_resid_q")
        assert set(q["quantile"].to_list()) == {0.05, 0.95}

    def test_lam_specs_report_lam_not_halflife(self):
        spec = grid_spec(half_life=None, lam=0.97, ridge=[1e-6])
        idx = po.spec.output_index(spec)
        assert idx["half_life"].null_count() == idx.height
        assert set(idx.filter(pl.col("kind") == "pred")["lam"].to_list()) == {0.97}

    def test_ew_cov_columns(self):
        spec = po.spec.ew_cov(
            "m", features=["a", "b"], stats=["std", "corr"], half_life=10.0, min_weight=2.0
        )
        idx = po.spec.output_index(spec)
        rows = {r["field"]: r["columns"] for r in idx.iter_rows(named=True)}
        assert rows["std_a"] == ["a"]
        assert rows["corr_a_b"] == ["a", "b"]
        assert rows["weight_sum"] is None

    def test_ew_cov_scores_are_over_every_column_and_carry_their_level(self):
        # E37/E38: `mahal` and the components are over all the columns, a
        # loading is over its own column, and a `mahal_q` row carries its
        # level like `abs_resid_q` does.
        spec = po.spec.ew_cov(
            "m",
            features=["a", "b"],
            stats=["mahal"],
            precision_prior=1e-6,
            mahal_quantiles=[0.95],
            pca=1,
            half_life=[10.0, 100.0],
            min_weight=2.0,
        )
        idx = po.spec.output_index(spec)
        rows = {r["field"]: r for r in idx.iter_rows(named=True)}
        assert rows["mahal@h10"]["columns"] == ["a", "b"]
        assert rows["mahal@h10"]["kind"] == "mahal"
        assert rows["mahal_q0.95@h100"]["quantile"] == 0.95
        assert rows["mahal_q0.95@h100"]["kind"] == "mahal_q"
        assert rows["mahal_q0.95@h100"]["half_life"] == 100.0
        assert rows["pc0_loading_a@h10"]["columns"] == ["a"]
        assert rows["pc0_loading_a@h10"]["kind"] == "pc_loading"
        assert rows["pc0_score@h10"]["columns"] == ["a", "b"]
        assert {rows[f"pc0_{k}@h10"]["kind"] for k in ("var", "share", "score")} == {
            "pc_var",
            "pc_share",
            "pc_score",
        }
        assert idx.filter(pl.col("kind") == "mahal_q")["quantile"].null_count() == 0
        assert idx.filter(pl.col("kind") != "mahal_q")["quantile"].null_count() == idx.height - 2

    def test_compact_float_rendering_is_reachable_without_knowing_it(self):
        """The whole point: a user never needs to know that 1e-300 renders as
        `1e-300` and 0.5 as `0.5`."""
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            ridge=[1e-300, 0.5],
            half_life=[100.0, 1e9],
            min_weight=2.0,
        )
        idx = po.spec.output_index(spec)
        name = idx.filter(
            (pl.col("kind") == "pred") & (pl.col("ridge") == 1e-300) & (pl.col("half_life") == 1e9)
        )["field"].item()
        assert name == "pred_y__r1e-300@h1e9"


def _every_kind():
    """Each kind's spec as the registry builds it, and the kinds whose
    fields once differed only in their names: ``ew_cov``'s ``lag_corr`` at
    two lags, ``hmm`` at three states, ``ew_class`` at three classes and
    ``bocpd`` over three columns."""
    specs = {}
    for name, extra in MINIMAL.items():
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": 50.0}
        kw.update(extra)
        specs[name] = getattr(po.spec, name)("m", **{k: v for k, v in kw.items() if v is not None})
    specs["ew_cov lag_corr"] = po.spec.ew_cov(
        "m", features=["a", "b"], stats=["lag_corr", "corr"], lags=[1, 3], half_life=10.0
    )
    specs["hmm k=3"] = po.spec.hmm(
        "m", features=["a", "b"], n_components=3, precision_prior=1.0, half_life=10.0
    )
    specs["ew_class"] = po.spec.ew_class(
        "m", label="y", features=["a"], classes=["u", "v", "w"], precision_prior=1.0, half_life=10.0
    )
    specs["bocpd"] = po.spec.bocpd("m", features=["a", "b", "c"])
    specs["ew_cov pca=2"] = po.spec.ew_cov("m", features=["a", "b"], pca=2, half_life=10.0)
    for name in MINIMAL:
        specs[f"{name} at its fullest"] = fullest(name)
    return specs


@pytest.mark.parametrize(("label", "spec"), list(_every_kind().items()))
def test_no_two_rows_match_apart_from_the_field(label, spec):
    """The index reaches every field by its values alone: no two rows
    match apart from ``field``, which ``output_index``'s docstring promises.
    ``lag`` tells ``lag_corr`` at its lags apart, ``component`` a principal
    component's fields, ``state`` an ``hmm`` posterior, ``class`` an
    ``ew_class`` one, ``columns`` a ``bocpd`` prediction, which is about its
    own column, and ``instance`` an instance of a grid. Each kind is held at
    its fullest too, every switch it takes turned on: the PCA's fields once
    matched apart from their names, which only ``pca=2`` showed."""
    idx = po.spec.output_index(spec)
    rest = [c for c in idx.columns if c != "field"]
    same = idx.filter(pl.struct(rest).is_duplicated())["field"].to_list()
    assert not same, f"{label}: {same}"


def test_a_principal_component_is_told_apart_by_its_component():
    idx = po.spec.output_index(po.spec.ew_cov("m", features=["a", "b"], pca=2, half_life=10.0))
    assert idx.schema["component"] == pl.Int64
    pc = idx.filter(pl.col("kind").str.starts_with("pc_"))
    assert pc.select("field", "component").rows() == [
        ("pc0_var", 0),
        ("pc0_share", 0),
        ("pc0_loading_a", 0),
        ("pc0_loading_b", 0),
        ("pc0_score", 0),
        ("pc1_var", 1),
        ("pc1_share", 1),
        ("pc1_loading_a", 1),
        ("pc1_loading_b", 1),
        ("pc1_score", 1),
    ]
    assert idx.filter(~pl.col("kind").str.starts_with("pc_"))["component"].null_count() == (
        idx.height - pc.height
    )


def test_bocpd_kinds_keep_their_full_stems():
    """The kind was the name cut at its first ``_``, so ``p_change`` read
    ``p`` and ``run_mode`` and ``run_mean`` both read ``run``."""
    idx = po.spec.output_index(po.spec.bocpd("m", features=["a", "b"]))
    assert idx.select("field", "kind").rows()[:6] == [
        ("p_change", "p_change"),
        ("run_mode", "run_mode"),
        ("run_mean", "run_mean"),
        ("pred_a", "pred"),
        ("pred_b", "pred"),
        ("loglik", "loglik"),
    ]


def test_every_field_names_its_instance():
    """The instance's suffix as ``coef()``, ``gram()`` and ``closed_groups()``
    report it, beside the decay it encodes: ``""`` for a single instance,
    null on a field that chooses across the instances or is the spec's own."""
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        half_life=["100s", "5m"],
        clock="t",
        gap_cap="1h",
        emit_selected=True,
        emit_clocks=True,
    )
    idx = po.spec.output_index(spec)
    assert idx.columns[:5] == ["field", "kind", "target", "instance", "half_life"]
    rows = dict(idx.select("field", "instance").iter_rows())
    assert rows["pred_y@h100s"] == "@h100s"
    assert rows["coef@h5m"] == "@h5m"
    assert rows["withheld_reason@h5m"] == "@h5m"
    assert rows["selected_y"] is None
    assert rows["scored_clock"] is None
    single = po.spec.output_index(
        po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0)
    )
    assert set(single["instance"].to_list()) == {""}
    seq = po.spec.output_index(po.spec.seqtest("m", targets=["y"]))
    assert set(seq["instance"].to_list()) == {""}
    cf = po.spec.coef_fields(spec)
    assert cf.columns[:6] == ["field", "position", "name", "target", "instance", "half_life"]
    assert set(cf["instance"].to_list()) == {"@h100s", "@h5m"}


def test_an_infinite_half_life_is_written_as_inf():
    """The tables travelled as JSON, which has no infinity, so an infinite
    half-life read null, as ``lam`` does, and a filter for it found nothing."""
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=[float("inf"), 100.0])
    for table in (po.spec.output_index(spec), po.spec.coef_fields(spec)):
        hinf = table.filter(pl.col("half_life") == float("inf"))
        assert hinf.height > 0
        assert all(f.endswith("@hinf") for f in hinf["field"].to_list())
        assert table.filter(pl.col("half_life").is_null()).height == 0


def test_a_one_point_lasso_path_drops_its_label():
    """A single ridge or half-life writes no suffix; a single path point
    wrote ``__l0.1``. The index still carries the penalty."""
    spec = po.spec.lasso("m", targets=["y"], features=["x0"], half_life=10.0, penalties=[0.1])
    assert po.spec.output_fields(spec)[:2] == ["pred_y", "resid_y"]
    idx = po.spec.output_index(spec)
    assert idx.filter(pl.col("kind") == "pred")["penalty"].to_list() == [0.1]
    assert po.spec.coef_fields(spec)["name"].to_list() == ["coef_y_intercept", "coef_y_x0"]


def test_one_rule_places_every_qualifier():
    """A qualifier -- a letter glued to its value -- follows the kind and
    comes before the columns: ``lag_corr_l1_x0_x1`` (it was
    ``lag_corr_x0_x1_l1``) as ``abs_resid_q0.05_y``; an ``hmm`` state is
    numbered where it is named, as a component is (``pc0_var``) and a
    centre (``coef_cluster0_x1``): ``filtered_state0``; and a lasso
    penalty is ``__p``, since ``l`` is a lag's."""
    cov = po.spec.ew_cov(
        "m", features=["x0", "x1"], stats=["lag_corr"], lags=[1, 5], half_life=10.0
    )
    lag = po.spec.output_index(cov).filter(pl.col("kind") == "lag_corr")
    assert lag["field"].to_list()[:4] == [
        "lag_corr_l1_x0_x0",
        "lag_corr_l1_x0_x1",
        "lag_corr_l1_x1_x0",
        "lag_corr_l1_x1_x1",
    ]
    assert lag.filter(pl.col("field") == "lag_corr_l5_x1_x0")["columns"].to_list() == [["x1", "x0"]]
    hmm = po.spec.output_fields(
        po.spec.hmm("m", features=["x0"], n_components=2, precision_prior=1.0, half_life=10.0)
    )
    assert hmm[:4] == ["filtered_state0", "filtered_state1", "predicted_state0", "predicted_state1"]
    lasso = po.spec.lasso("m", targets=["y"], features=["x0"], half_life=10.0, penalties=[0.1, 0.0])
    assert po.spec.output_fields(lasso)[:4] == [
        "pred_y__p0.1",
        "resid_y__p0.1",
        "pred_y__p0",
        "resid_y__p0",
    ]


def test_a_centre_a_state_and_a_class_are_columns_of_their_own():
    """``coef_index`` gave a ``kmeans`` centre's slot as the target
    ``"cluster0"``, an ``hmm`` state's as ``"state0"`` and an ``ew_class``
    class's as the class: each is now its own column, typed as
    ``output_index`` types ``state`` and ``class``, and ``target`` is the
    target or null."""
    km = po.spec.coef_index(
        po.spec.kmeans("m", features=["x0", "x1"], n_clusters=2, half_life=10.0)
    )
    assert km.schema["cluster"] == pl.Int64 and km.schema["state"] == pl.Int64
    assert km.select("cluster", "term").rows() == [(0, "x0"), (0, "x1"), (1, "x0"), (1, "x1")]
    assert km["target"].null_count() == km.height
    hmm = po.spec.coef_index(
        po.spec.hmm("m", features=["x0"], n_components=2, precision_prior=1.0, half_life=10.0)
    )
    assert hmm.select("state", "term").rows() == [(0, "x0"), (1, "x0")]
    cls = po.spec.coef_index(
        po.spec.ew_class(
            "m",
            label="lab",
            features=["x0"],
            classes=["a", "b"],
            precision_prior=1.0,
            half_life=10.0,
        )
    )
    assert cls.select("target", "class", "term").rows() == [("lab", "a", "x0"), ("lab", "b", "x0")]
    ridge = po.spec.coef_index(po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0))
    assert ridge.select("cluster", "state", "class").null_count().row(0) == (2, 2, 2)
    # The unnested columns keep their names.
    names = po.spec.coef_fields(
        po.spec.kmeans("m", features=["x0", "x1"], n_clusters=2, half_life=10.0)
    )["name"].to_list()
    assert names[:2] == ["coef_cluster0_x0", "coef_cluster0_x1"]


def test_the_values_that_tell_the_fields_apart():
    specs = _every_kind()
    lag = po.spec.output_index(specs["ew_cov lag_corr"]).filter(pl.col("kind") == "lag_corr")
    assert sorted(set(lag["lag"].to_list())) == [1, 3]
    assert lag.filter(pl.col("field") == "lag_corr_l3_b_a")["columns"].to_list() == [["b", "a"]]
    hmm = po.spec.output_index(specs["hmm k=3"])
    filtered = hmm.filter(pl.col("kind") == "filtered")
    assert filtered["state"].to_list() == [0, 1, 2]
    assert hmm.filter(pl.col("field") == "state")["state"].to_list() == [None]
    cls = po.spec.output_index(specs["ew_class"]).filter(pl.col("kind") == "p")
    assert cls.select("field", "class").rows() == [("p_u", "u"), ("p_v", "v"), ("p_w", "w")]
    boc = po.spec.output_index(specs["bocpd"])
    pred = boc.filter(pl.col("kind") == "pred")
    assert pred["columns"].to_list() == [["a"], ["b"], ["c"]]
    assert boc.filter(pl.col("field") == "p_change")["columns"].to_list() == [["a", "b", "c"]]
    # Typed for a filter: integers and a string, null on every other field.
    assert boc.schema["lag"] == pl.Int64 and boc.schema["state"] == pl.Int64
    assert boc.schema["class"] == pl.String
    assert boc.select("lag", "state", "class").null_count().row(0) == (boc.height,) * 3


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
            half_life=1e9,
            min_weight=3.0,
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
        spec = po.spec.holt("m", targets=["a", "b"], half_life=50.0, min_weight=2.0)
        ci = po.spec.coef_index(spec)
        assert ci["term"].to_list() == ["level", "trend", "level", "trend"]
        assert ci["target"].to_list() == ["a", "a", "b", "b"]

    def test_no_intercept(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=50.0,
            min_weight=2.0,
            fit_intercept=False,
        )
        assert po.spec.coef_index(spec)["term"].to_list() == ["x0"]

    def test_a_spec_without_coefficients_has_an_empty_layout(self):
        """``coef_index`` raised for a kind with no coefficients where
        ``coef_fields`` gave an empty frame: both give the empty frame now,
        with their columns, as ``ModelBank.coef()``'s sweep does."""
        spec = po.spec.ew_cov("m", features=["a"], stats=["mean"], half_life=10.0, min_weight=2.0)
        index, fields = po.spec.coef_index(spec), po.spec.coef_fields(spec)
        assert index.is_empty() and fields.is_empty()
        ridge = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0)
        assert index.schema == po.spec.coef_index(ridge).schema
        assert fields.schema == po.spec.coef_fields(ridge).schema

    def test_position_is_int64_in_both_tables(self):
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=10.0)
        assert po.spec.coef_fields(spec).schema["position"] == pl.Int64
        assert po.spec.coef_index(spec).schema["position"] == pl.Int64


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
            half_life=50.0,
            min_weight=3.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        long = ev.unpack(out, spec)
        got = dict(long.group_by("slot").agg(pl.col("target").first()).iter_rows())
        assert got == {"pred_y": "y", "pred_y2": "y2"}
