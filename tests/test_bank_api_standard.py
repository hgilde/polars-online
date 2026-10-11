"""The bank's surface held to one convention before 1.0 (PLAN task 244, the
``task244-bank`` part; review api1, slice M).

Each test names the item it pins. A renamed parameter or function is read
under its old name with a :class:`polars_online.PolarsOnlineDeprecationWarning`
naming the new one, so code written against 0.13 keeps working; a renamed
output column cannot be forwarded, and is pinned under its new name.
"""

from __future__ import annotations

import inspect
import io
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

BASE = dict(targets=["y"], features=["x0"], half_life=10.0)


def _df(n: int = 120, groups: tuple[str, ...] = ("a", "b")) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": x,
            "y": 0.5 * x + rng.normal(scale=0.3, size=n),
            "g": [groups[i % len(groups)] for i in range(n)],
        }
    )


def _bank() -> po.ModelBank:
    return po.ModelBank([po.spec.ewridge("m", **BASE)])


# --- M-1: a missing column ---------------------------------------------------


def test_m1_a_missing_column_is_polars_column_not_found_and_a_value_error():
    bank = _bank()
    with pytest.raises(po.ColumnNotFoundError) as caught:
        bank.fit_predict(_df().drop("x0"))
    err = caught.value
    assert isinstance(err, pl.exceptions.ColumnNotFoundError)
    assert isinstance(err, ValueError)
    assert issubclass(po.ColumnNotFoundError, pl.exceptions.ColumnNotFoundError)
    assert issubclass(po.ColumnNotFoundError, ValueError)
    assert "x0" in str(err)
    assert not str(err).startswith("not found: "), str(err)
    assert "ColumnNotFoundError" in po.__all__


def test_m1_the_target_predict_and_plan_forms_raise_the_same_class():
    with pytest.raises(po.ColumnNotFoundError, match="y"):
        _bank().fit_predict(_df().drop("y"))
    with pytest.raises(po.ColumnNotFoundError, match="x0"):
        _bank().predict(_df().drop("x0"))
    with pytest.raises(po.ColumnNotFoundError, match="x0"):
        po.fit_predict(_df().drop("x0"), [po.spec.ewridge("m", **BASE)])
    lf = _df().drop("x0").lazy()
    with pytest.raises(po.ColumnNotFoundError, match="x0"):
        lf.online.fit_predict([po.spec.ewridge("m", **BASE)])


# --- M-2 and M-21: one base for every warning ---------------------------------


@pytest.mark.parametrize(
    ("name", "second"),
    [
        ("ReadinessWarning", UserWarning),
        ("OrderNotGuaranteedWarning", UserWarning),
        ("ConsumedSourceWarning", UserWarning),
        ("UnstableWarning", UserWarning),
        ("PolarsOnlineDeprecationWarning", DeprecationWarning),
    ],
)
def test_m2_every_warning_shares_one_exported_base(name, second):
    cls = getattr(po, name)
    assert issubclass(cls, po.PolarsOnlineWarning)
    assert issubclass(cls, second)
    assert "PolarsOnlineWarning" in po.__all__
    assert issubclass(po.PolarsOnlineWarning, Warning)


def test_m21_unstable_warning_is_also_polars_own():
    assert issubclass(po.UnstableWarning, pl.exceptions.UnstableWarning)


# --- M-3: the batch methods' options by keyword --------------------------------


@pytest.mark.parametrize("method", ["fit", "fit_predict_batches"])
def test_m3_options_are_keyword_only(method):
    params = inspect.signature(getattr(po.ModelBank, method)).parameters
    assert params["closed_groups"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["chunk_size"].kind is inspect.Parameter.KEYWORD_ONLY


def test_m3_a_positional_option_is_read_with_the_deprecation_warning():
    df = _df()
    want = _bank()
    want.fit(df, chunk_size=10)
    got = _bank()
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="chunk_size.*by keyword"):
        got.fit(df, None, 10)
    assert got.save_bytes() == want.save_bytes()
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="closed_groups"):
        outs = list(_bank().fit_predict_batches(df, None))
    assert len(outs) == 1


def test_m9_fit_predict_batches_returns_an_iterator():
    sig = inspect.signature(po.ModelBank.fit_predict_batches)
    assert "Iterator" in str(sig.return_annotation)


# --- M-10: chunk_size on every eager form --------------------------------------


def test_m10_chunk_size_on_the_frame_forms():
    df = _df()
    specs = [po.spec.ewridge("m", **BASE)]
    whole = df.online.fit_predict(specs)
    chunked = df.online.fit_predict(specs, chunk_size=7)
    pred = pl.col("m").struct.field("pred_y")
    assert whole.select(pred).equals(chunked.select(pred))
    # Chunked, coef is reported at each chunk's last row, as a plan reports it.
    coef = pl.col("m").struct.field("coef")
    assert chunked.select(coef.is_not_null().sum()).item() == len(range(0, 120, 7))
    assert po.fit_predict(df, specs, chunk_size=7).equals(chunked)
    bank = _bank()
    bank.fit_predict(df)
    scored = df.online.predict(bank, chunk_size=7)
    assert scored.select(pred).equals(df.online.predict(bank).select(pred))
    assert po.predict(df, bank, chunk_size=7).equals(scored)
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        df.online.fit_predict(specs, chunk_size=0)


# --- M-25: load_state takes a bank, as predict does ----------------------------


def test_m25_load_state_takes_a_bank_snapshotted_when_built(tmp_path):
    df = _df()
    first, rest = df.head(60), df.tail(60)
    bank = _bank()
    bank.fit_predict(first)
    before = bank.save_bytes()
    path = tmp_path / "b.state"
    bank.save(path)
    want = rest.online.fit_predict(load_state=path)
    assert rest.online.fit_predict(load_state=bank).equals(want)
    assert bank.save_bytes() == before, "the caller's bank does not move"
    plan = rest.lazy().online.fit_predict(load_state=bank)
    bank.fit_predict(rest)  # moving the bank after the plan is built changes nothing
    assert plan.collect().equals(want)
    assert po.fit_predict(rest, load_state=_reload(before)).equals(want)


def _reload(data: bytes) -> po.ModelBank:
    return po.ModelBank.load_bytes(data)


# --- M-14: file objects, and write_json ----------------------------------------


def test_m14_save_and_load_take_file_objects_and_bytes(tmp_path):
    bank = _bank()
    bank.fit_predict(_df())
    buf = io.BytesIO()
    bank.save(buf)
    assert buf.getvalue() == bank.save_bytes()
    buf.seek(0)
    assert po.ModelBank.load(buf).save_bytes() == bank.save_bytes()
    assert po.ModelBank.load(bank.save_bytes()).save_bytes() == bank.save_bytes()
    path = tmp_path / "b.state"
    with open(path, "wb") as f:
        bank.save(f)
    with open(path, "rb") as f:
        assert po.ModelBank.load(f).save_bytes() == bank.save_bytes()
    assert po.ModelBank.load(str(path)).save_bytes() == bank.save_bytes()


def test_m14_write_json_returns_the_text_without_a_file(tmp_path):
    bank = _bank()
    bank.fit_predict(_df())
    text = bank.write_json()
    assert isinstance(text, str)
    assert json.loads(text)
    path = tmp_path / "b.json"
    assert bank.write_json(path) is None
    assert path.read_text(encoding="utf-8") == text
    sio = io.StringIO()
    bank.write_json(sio, pretty=False)
    assert sio.getvalue() == bank.write_json(pretty=False)
    bio = io.BytesIO()
    bank.write_json(bio)
    assert bio.getvalue().decode("utf-8") == text


def test_m14_the_old_json_names_are_forwarded(tmp_path):
    bank = _bank()
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="to_json.*write_json"):
        assert bank.to_json(pretty=False) == bank.write_json(pretty=False)
    path = tmp_path / "b.json"
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="save_json.*write_json"):
        bank.save_json(path)
    assert path.read_text(encoding="utf-8") == bank.write_json()


def test_m14_save_and_load_keyword_path_is_forwarded(tmp_path):
    bank = _bank()
    path = tmp_path / "b.state"
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="path.*file"):
        bank.save(path=path)
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="path.*source"):
        po.ModelBank.load(path=path)


# --- M-15: properties -----------------------------------------------------------


def test_m15_rows_fed_and_output_fields_are_properties():
    assert isinstance(inspect.getattr_static(po.ModelBank, "rows_fed"), property)
    assert isinstance(inspect.getattr_static(po.ModelBank, "output_fields"), property)
    bank = _bank()
    bank.fit_predict(_df())
    assert bank.rows_fed == 120
    assert bank.output_fields["m"][0] == "pred_y"


def test_m15_the_call_form_is_forwarded():
    bank = _bank()
    bank.fit_predict(_df())
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="rows_fed"):
        assert bank.rows_fed() == 120
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="output_fields"):
        assert bank.output_fields() == bank.output_fields


# --- M-5, M-6, M-8, M-19, M-20, M-23: po.eval -----------------------------------


def _scored(n: int = 120):
    spec = po.spec.ewridge("m", group="g", **BASE)
    return spec, po.ModelBank([spec]).fit_predict(_df(n))


def test_m5_group_none_is_the_default_and_accepted():
    _, out = _scored()
    for fn in (po.eval.metrics, po.eval.sums):
        assert inspect.signature(fn).parameters["group"].default is None
        assert fn(out, "m", group=None).equals(fn(out, "m"))
    assert po.eval.diebold_mariano(out, a="m", b="m", group=None).height == 1


def test_m6_a_thin_key_keeps_its_row_with_null_metrics():
    _, out = _scored()
    full = po.eval.metrics(out, "m", group="g", min_samples=1)
    n = full["n"].to_list()
    assert len(n) == 2 and full["r2"].null_count() == 0
    got = po.eval.metrics(out, "m", group="g", min_samples=max(n) + 1)
    assert got.height == 2
    assert got["n"].to_list() == n
    for col in ("r2", "ic", "hit_rate", "mse", "rmse"):
        assert got[col].null_count() == 2, col
    s = po.eval.sums(out, "m", group="g")
    thin = po.eval.from_sums(s, min_samples=max(n) + 1)
    assert thin.height == 2 and thin["r2"].null_count() == 2 and thin["n"].to_list() == n
    dm = po.eval.diebold_mariano(out, a="m", b="m", group="g", min_samples=61)
    assert dm.height == 2 and dm["dm"].null_count() == 2 and dm["mean"].null_count() == 2
    assert dm["p_value"].null_count() == 2


def test_m8_rmse_in_every_metric_frame():
    _, out = _scored()
    m = po.eval.metrics(out, "m")
    assert m["rmse"].item() == pytest.approx(m["mse"].item() ** 0.5)
    assert "rmse" in po.eval.window_metrics(out, "m", clock="t", every=60.0).columns
    assert "rmse" in po.eval.compare_specs(out, ["m"]).columns
    assert m.columns[-2:] == ["mse", "rmse"]
    assert po.eval.from_sums(po.eval.sums(out, "m")).columns[-2:] == ["mse", "rmse"]


def _logistic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    df = pl.DataFrame({"x0": x, "y": (x + rng.normal(scale=0.5, size=200) > 0).astype(float)})
    spec = po.spec.sgd("c", targets=["y"], features=["x0"], loss="logistic", half_life=50.0)
    return spec, po.ModelBank([spec]).fit_predict(df)


def test_m19_binary_is_inferred_from_the_spec_and_its_columns_are_named():
    spec, out = _logistic()
    got = po.eval.metrics(out, spec, min_samples=1)
    assert got.columns == [
        "slot",
        "target",
        "n",
        "brier_skill",
        "ic",
        "accuracy",
        "log_loss",
        "mse",
        "rmse",
    ]
    assert inspect.signature(po.eval.metrics).parameters["binary"].default is None
    explicit = po.eval.metrics(out, "c", binary=True, min_samples=1)
    assert explicit.equals(got)
    plain = po.eval.metrics(out, spec, binary=False, min_samples=1)
    assert "hit_rate" in plain.columns and "r2" in plain.columns
    s = po.eval.sums(out, spec)
    assert po.eval.from_sums(s, binary=True, min_samples=1)["accuracy"].item() == pytest.approx(
        got["accuracy"].item()
    )


def test_m20_spec_takes_a_name_or_the_dict():
    spec, out = _scored()
    by_name = po.eval.metrics(out, "m")
    assert po.eval.metrics(out, spec).equals(by_name)
    assert "spec_name" not in inspect.signature(po.eval.metrics).parameters
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="spec_name"):
        assert po.eval.metrics(out, spec_name="m").equals(by_name)
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="spec"):
        assert po.eval.metrics(out, "m", spec=spec).equals(by_name)
    assert po.eval.compare_specs(out, [spec]).equals(po.eval.compare_specs(out, ["m"]))
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="spec_names"):
        po.eval.compare_specs(out, spec_names=["m"])


def test_m23_a_lazyframe_gets_one_clear_type_error():
    _, out = _scored()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="takes a DataFrame, not a LazyFrame: collect it"):
            po.eval.metrics(out.lazy(), "m")
        with pytest.raises(TypeError, match="not a LazyFrame"):
            po.eval.diebold_mariano(out.lazy(), a="m", b="m")
        with pytest.raises(TypeError, match="not a LazyFrame"):
            po.eval.unpack(out.lazy(), "m")


# --- M-9 and M-17: the operators -------------------------------------------------


def test_m9_operator_parameters_are_typed():
    params = inspect.signature(po.ewm_mean).parameters
    assert "Duration" in str(params["half_life"].annotation)
    assert "Duration" in str(params["window_size"].annotation)
    assert "Literal" in str(params["closed"].annotation)
    assert "Literal" in str(params["partial"].annotation)
    assert "Literal" in str(inspect.signature(po.ModelBank.audit).parameters["table"].annotation)
    assert "Literal" in str(inspect.signature(po.ModelBank.gram).parameters["layout"].annotation)


@pytest.mark.parametrize("kind", ["mean", "sum", "rate"])
def test_m17_reverse_is_the_window_looking_ahead(kind):
    new = getattr(po, f"ewm_{kind}")("x0", half_life=2.0, window_size=5.0, reverse=True)
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match=f"rewm_{kind}.*reverse=True"):
        old = getattr(po, f"rewm_{kind}")("x0", half_life=2.0, window_size=5.0)
    assert new.meta.eq(old)
    assert f"rewm_{kind}" not in po.__all__
    assert f"rewm_{kind}" not in po.ops.__all__
    with pytest.raises(TypeError, match="window_size is required"):
        getattr(po, f"ewm_{kind}")("x0", half_life=2.0, reverse=True)


def test_m17_reverse_runs_through_with_windows_and_targets():
    df = _df()
    fwd = po.ewm_mean("y", half_life=2.0, window_size=5.0, reverse=True)
    out = po.stream.with_windows(df, fwd_y=fwd, clock="t", gap_cap=10.0)
    assert out["fwd_y"].null_count() < df.height
    spec = po.spec.ewridge(
        "m",
        targets=[fwd.alias("fwd")],
        features=["x0"],
        half_life=10.0,
        clock="t",
        gap_cap=10.0,
        embargo=5.0,
    )
    assert spec["targets"][0]["name"] == "fwd"


# --- M-11: pl.col targets --------------------------------------------------------


def test_m11_a_bare_col_target_is_told_to_write_the_string():
    with pytest.raises(ValueError, match="write the string 'p'"):
        po.spec.ewridge("m", targets=[pl.col("p")], features=["x0"], half_life=10.0)


def test_m11_an_aliased_col_target_is_po_target():
    got = po.spec.ewridge("m", targets=[pl.col("p").alias("price")], features=["x0"], half_life=1.0)
    want = po.spec.ewridge(
        "m", targets=[po.target("p", name="price")], features=["x0"], half_life=1.0
    )
    assert got == want


# --- M-13: renamed and deprecated names stay out of the snapshot and reference -----


def test_m13_renamed_and_deprecated_members_are_marked():
    from polars_online._renamed import is_renamed

    for name in ("rows_seen", "to_json", "save_json"):
        assert is_renamed(getattr(po.ModelBank, name)), name
    for name in ("fit", "save", "write_json"):
        assert not is_renamed(getattr(po.ModelBank, name)), name
    snapshot = (Path(__file__).parent / "api_surface.txt").read_text(encoding="utf-8")
    section = snapshot.split("[ModelBank]")[1].split("\n[")[0]
    for name in ("rows_seen", "to_json", "save_json"):
        assert f"\n  {name}" not in section, name


# --- M-24: po.gram --------------------------------------------------------------


def _gram():
    spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=1e9)
    df = _df().with_columns(x1=pl.col("x0") ** 2)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")[0]


def test_m24_subset_takes_columns_and_forwards_cols():
    g = _gram()
    assert list(inspect.signature(po.gram.subset).parameters)[1] == "columns"
    want = po.gram.subset(g, columns=["intercept", "x0"])
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="cols.*columns"):
        old = po.gram.subset(g, cols=["intercept", "x0"])
    assert old["columns"] == want["columns"]


def test_m24_lars_takes_max_iter_and_forwards_max_steps():
    g = _gram()
    assert "max_iter" in inspect.signature(po.gram.lars_path).parameters
    assert "max_iter" in inspect.signature(po.gram.lars_paths).parameters
    want = po.gram.lars_path(g, max_iter=1)
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="max_steps.*max_iter"):
        old = po.gram.lars_path(g, max_steps=1)
    assert np.array_equal(old["coef"], want["coef"])
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="max_steps.*max_iter"):
        po.gram.solve_subsets([g], [[0]], path={"max_steps": 1})


def test_m24_solve_and_solve_subsets_take_the_same_ridge():
    g = _gram()
    ann = {
        str(inspect.signature(f).parameters["ridge"].annotation)
        for f in (po.gram.solve, po.gram.solve_subsets)
    }
    assert len(ann) == 1, ann
    grid = po.gram.solve_subsets([g], [[0]], ridge=[0.0, 0.5])
    one = po.gram.solve_subsets([g], [[0]], ridge=0.5)
    assert np.allclose(grid[0][0]["coef"][1], one[0][0]["coef"])
    assert np.allclose(grid[0][0]["coef"], po.gram.solve(g, ridge=[0.0, 0.5]), rtol=1e-9)


def test_m24_a_gram_typed_dict():
    assert "Gram" in po.gram.__all__
    assert po.gram.Gram.__annotations__["columns"]
    ret = str(inspect.signature(po.ModelBank.gram).return_annotation)
    assert "Gram" in ret


def test_one_correlation_of_a_gram():
    g = _gram()
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="po.gram.correlation"):
        old = po.corr.matrix(g)
    assert np.array_equal(old, po.gram.correlation(g), equal_nan=True)
    assert "matrix" not in po.corr.__all__


# --- M-16: fit's docstring ------------------------------------------------------


def test_m16_fit_says_it_continues():
    assert "partial_fit" in (po.ModelBank.fit.__doc__ or "")
