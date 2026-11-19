"""The bank's surface held to one convention before 1.0 (PLAN task 244, the
``task244-bank`` part; review api1, slice M).

Each test names the item it pins. A renamed parameter or function is refused
under its old name, naming the new one; a renamed output column is pinned
under its new name.
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

from rheon import polars as rp

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


def _bank() -> rp.ModelBank:
    return rp.ModelBank([rp.spec.ewridge("m", **BASE)])


# --- M-1: a missing column ---------------------------------------------------


def test_m1_a_missing_column_is_polars_column_not_found_and_a_value_error():
    bank = _bank()
    with pytest.raises(rp.ColumnNotFoundError) as caught:
        bank.fit_predict(_df().drop("x0"))
    err = caught.value
    assert isinstance(err, pl.exceptions.ColumnNotFoundError)
    assert isinstance(err, ValueError)
    assert issubclass(rp.ColumnNotFoundError, pl.exceptions.ColumnNotFoundError)
    assert issubclass(rp.ColumnNotFoundError, ValueError)
    assert "x0" in str(err)
    assert not str(err).startswith("not found: "), str(err)
    assert "ColumnNotFoundError" in rp.__all__


def test_m1_the_target_predict_and_plan_forms_raise_the_same_class():
    with pytest.raises(rp.ColumnNotFoundError, match="y"):
        _bank().fit_predict(_df().drop("y"))
    with pytest.raises(rp.ColumnNotFoundError, match="x0"):
        _bank().predict(_df().drop("x0"))
    with pytest.raises(rp.ColumnNotFoundError, match="x0"):
        rp.fit_predict(_df().drop("x0"), [rp.spec.ewridge("m", **BASE)])
    lf = _df().drop("x0").lazy()
    with pytest.raises(rp.ColumnNotFoundError, match="x0"):
        lf.rheon.fit_predict([rp.spec.ewridge("m", **BASE)])


# --- M-2 and M-21: one base for every warning ---------------------------------


@pytest.mark.parametrize(
    ("name", "second"),
    [
        ("ReadinessWarning", UserWarning),
        ("OrderNotGuaranteedWarning", UserWarning),
        ("ConsumedSourceWarning", UserWarning),
        ("UnstableWarning", UserWarning),
        ("RheonDeprecationWarning", DeprecationWarning),
    ],
)
def test_m2_every_warning_shares_one_exported_base(name, second):
    cls = getattr(rp, name)
    assert issubclass(cls, rp.RheonWarning)
    assert issubclass(cls, second)
    assert "RheonWarning" in rp.__all__
    assert issubclass(rp.RheonWarning, Warning)


def test_m21_unstable_warning_is_also_polars_own():
    assert issubclass(rp.UnstableWarning, pl.exceptions.UnstableWarning)


# --- M-3: the batch methods' options by keyword --------------------------------


@pytest.mark.parametrize("method", ["fit", "fit_predict_batches"])
def test_m3_options_are_keyword_only(method):
    params = inspect.signature(getattr(rp.ModelBank, method)).parameters
    assert params["closed_groups"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["chunk_size"].kind is inspect.Parameter.KEYWORD_ONLY


def test_m3_a_positional_option_is_refused():
    df = _df()
    with pytest.raises(TypeError, match="positional"):
        _bank().fit(df, None, 10)
    with pytest.raises(TypeError, match="positional"):
        list(_bank().fit_predict_batches(df, None))


def test_m9_fit_predict_batches_returns_an_iterator():
    sig = inspect.signature(rp.ModelBank.fit_predict_batches)
    assert "Iterator" in str(sig.return_annotation)


# --- M-10: chunk_size on every eager form --------------------------------------


def test_m10_chunk_size_on_the_frame_forms():
    df = _df()
    specs = [rp.spec.ewridge("m", **BASE)]
    whole = df.rheon.fit_predict(specs)
    chunked = df.rheon.fit_predict(specs, chunk_size=7)
    pred = pl.col("m").struct.field("pred_y")
    assert whole.select(pred).equals(chunked.select(pred))
    # Chunked, coef is reported at each chunk's last row, as a plan reports it.
    coef = pl.col("m").struct.field("coef")
    assert chunked.select(coef.is_not_null().sum()).item() == len(range(0, 120, 7))
    assert rp.fit_predict(df, specs, chunk_size=7).equals(chunked)
    bank = _bank()
    bank.fit_predict(df)
    scored = df.rheon.predict(bank, chunk_size=7)
    assert scored.select(pred).equals(df.rheon.predict(bank).select(pred))
    assert rp.predict(df, bank, chunk_size=7).equals(scored)
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        df.rheon.fit_predict(specs, chunk_size=0)


# --- M-25: load_state takes a bank, as predict does ----------------------------


def test_m25_load_state_takes_a_bank_snapshotted_when_built(tmp_path):
    df = _df()
    first, rest = df.head(60), df.tail(60)
    bank = _bank()
    bank.fit_predict(first)
    before = bank.save_bytes()
    path = tmp_path / "b.state"
    bank.save(path)
    want = rest.rheon.fit_predict(load_state=path)
    assert rest.rheon.fit_predict(load_state=bank).equals(want)
    assert bank.save_bytes() == before, "the caller's bank does not move"
    plan = rest.lazy().rheon.fit_predict(load_state=bank)
    bank.fit_predict(rest)  # moving the bank after the plan is built changes nothing
    assert plan.collect().equals(want)
    assert rp.fit_predict(rest, load_state=_reload(before)).equals(want)


def _reload(data: bytes) -> rp.ModelBank:
    return rp.ModelBank.load_bytes(data)


# --- M-14: file objects, and write_json ----------------------------------------


def test_m14_save_and_load_take_file_objects_and_bytes(tmp_path):
    bank = _bank()
    bank.fit_predict(_df())
    buf = io.BytesIO()
    bank.save(buf)
    assert buf.getvalue() == bank.save_bytes()
    buf.seek(0)
    assert rp.ModelBank.load(buf).save_bytes() == bank.save_bytes()
    assert rp.ModelBank.load(bank.save_bytes()).save_bytes() == bank.save_bytes()
    path = tmp_path / "b.state"
    with open(path, "wb") as f:
        bank.save(f)
    with open(path, "rb") as f:
        assert rp.ModelBank.load(f).save_bytes() == bank.save_bytes()
    assert rp.ModelBank.load(str(path)).save_bytes() == bank.save_bytes()


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


def test_m14_the_old_json_names_are_refused(tmp_path):
    bank = _bank()
    with pytest.raises(AttributeError, match="ModelBank.to_json.*renamed write_json"):
        bank.to_json(pretty=False)
    with pytest.raises(AttributeError, match="ModelBank.save_json.*renamed write_json"):
        bank.save_json(tmp_path / "b.json")


def test_m14_save_and_load_keyword_path_is_refused(tmp_path):
    bank = _bank()
    path = tmp_path / "b.state"
    with pytest.raises(TypeError, match="ModelBank.save: path was renamed file"):
        bank.save(path=path)
    with pytest.raises(TypeError, match="ModelBank.load: path was renamed source"):
        rp.ModelBank.load(path=path)


# --- M-15: properties -----------------------------------------------------------


def test_m15_rows_fed_and_output_fields_are_properties():
    assert isinstance(inspect.getattr_static(rp.ModelBank, "rows_fed"), property)
    assert isinstance(inspect.getattr_static(rp.ModelBank, "output_fields"), property)
    bank = _bank()
    bank.fit_predict(_df())
    assert bank.rows_fed == 120
    assert bank.output_fields["m"][0] == "pred_y"


def test_m15_the_call_form_is_refused():
    bank = _bank()
    bank.fit_predict(_df())
    with pytest.raises(TypeError, match="ModelBank.rows_fed: it is a property"):
        bank.rows_fed()
    with pytest.raises(TypeError, match="ModelBank.output_fields: it is a property"):
        bank.output_fields()


# --- M-5, M-6, M-8, M-19, M-20, M-23: rp.eval -----------------------------------


def _scored(n: int = 120):
    spec = rp.spec.ewridge("m", group="g", **BASE)
    return spec, rp.ModelBank([spec]).fit_predict(_df(n))


def test_m5_group_none_is_the_default_and_accepted():
    _, out = _scored()
    for fn in (rp.eval.metrics, rp.eval.sums):
        assert inspect.signature(fn).parameters["group"].default is None
        assert fn(out, "m", group=None).equals(fn(out, "m"))
    assert rp.eval.diebold_mariano(out, a="m", b="m", group=None).height == 1


def test_m6_a_thin_key_keeps_its_row_with_null_metrics():
    _, out = _scored()
    full = rp.eval.metrics(out, "m", group="g", min_samples=1)
    n = full["n"].to_list()
    assert len(n) == 2 and full["r2"].null_count() == 0
    got = rp.eval.metrics(out, "m", group="g", min_samples=max(n) + 1)
    assert got.height == 2
    assert got["n"].to_list() == n
    for col in ("r2", "ic", "hit_rate", "mse", "rmse"):
        assert got[col].null_count() == 2, col
    s = rp.eval.sums(out, "m", group="g")
    thin = rp.eval.from_sums(s, min_samples=max(n) + 1)
    assert thin.height == 2 and thin["r2"].null_count() == 2 and thin["n"].to_list() == n
    dm = rp.eval.diebold_mariano(out, a="m", b="m", group="g", min_samples=61)
    assert dm.height == 2 and dm["dm"].null_count() == 2 and dm["mean"].null_count() == 2
    assert dm["p_value"].null_count() == 2


def test_m8_rmse_in_every_metric_frame():
    _, out = _scored()
    m = rp.eval.metrics(out, "m")
    assert m["rmse"].item() == pytest.approx(m["mse"].item() ** 0.5)
    assert "rmse" in rp.eval.window_metrics(out, "m", clock="t", every=60.0).columns
    assert "rmse" in rp.eval.compare_specs(out, ["m"]).columns
    assert m.columns[-2:] == ["mse", "rmse"]
    assert rp.eval.from_sums(rp.eval.sums(out, "m")).columns[-2:] == ["mse", "rmse"]


def _logistic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    df = pl.DataFrame({"x0": x, "y": (x + rng.normal(scale=0.5, size=200) > 0).astype(float)})
    spec = rp.spec.sgd("c", targets=["y"], features=["x0"], loss="logistic", half_life=50.0)
    return spec, rp.ModelBank([spec]).fit_predict(df)


def test_m19_binary_is_inferred_from_the_spec_and_its_columns_are_named():
    spec, out = _logistic()
    got = rp.eval.metrics(out, spec, min_samples=1)
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
    assert inspect.signature(rp.eval.metrics).parameters["binary"].default is None
    explicit = rp.eval.metrics(out, "c", binary=True, min_samples=1)
    assert explicit.equals(got)
    plain = rp.eval.metrics(out, spec, binary=False, min_samples=1)
    assert "hit_rate" in plain.columns and "r2" in plain.columns
    s = rp.eval.sums(out, spec)
    assert rp.eval.from_sums(s, binary=True, min_samples=1)["accuracy"].item() == pytest.approx(
        got["accuracy"].item()
    )


def test_m20_spec_takes_a_name_or_the_dict():
    spec, out = _scored()
    by_name = rp.eval.metrics(out, "m")
    assert rp.eval.metrics(out, spec).equals(by_name)
    assert "spec_name" not in inspect.signature(rp.eval.metrics).parameters
    with pytest.raises(TypeError, match="spec_name was merged into spec"):
        rp.eval.metrics(out, spec_name="m")
    with pytest.raises(TypeError, match="beside spec= was merged into spec"):
        rp.eval.metrics(out, "m", spec=spec)
    assert rp.eval.compare_specs(out, [spec]).equals(rp.eval.compare_specs(out, ["m"]))
    with pytest.raises(TypeError, match="spec_names was merged into specs"):
        rp.eval.compare_specs(out, spec_names=["m"])


def test_m23_a_lazyframe_is_read_where_a_call_reads_a_query_and_refused_by_name_elsewhere():
    """M-23 refused a ``LazyFrame`` with one clear ``TypeError`` where Polars'
    own calls warned three times first. Task 380: every call that reads a
    bank's output reads the query; ``merge_sums`` and ``from_sums``, which
    read the sums a call returned, still refuse it, by name, without a
    warning."""
    _, out = _scored()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # To the bit but `ic`, which rounds apart (tests/test_eval_query.py).
        got, want = rp.eval.metrics(out.lazy(), "m"), rp.eval.metrics(out, "m")
        assert got.drop("ic").equals(want.drop("ic"))
        assert got["ic"][0] == pytest.approx(want["ic"][0], rel=1e-14)
        assert rp.eval.unpack(out.lazy(), "m").collect().equals(rp.eval.unpack(out, "m"))
        with pytest.raises(
            TypeError, match="must be a Polars DataFrame, but it is a LazyFrame. Pass the query"
        ):
            rp.eval.from_sums(rp.eval.sums(out, "m").lazy())


# --- M-9 and M-17: the operators -------------------------------------------------


def test_m9_operator_parameters_are_typed():
    params = inspect.signature(rp.ewm_mean).parameters
    assert "Duration" in str(params["half_life"].annotation)
    assert "Duration" in str(params["window_size"].annotation)
    assert "Literal" in str(params["closed"].annotation)
    assert "Literal" in str(params["partial"].annotation)
    assert "Literal" in str(inspect.signature(rp.ModelBank.audit).parameters["table"].annotation)
    assert "Literal" in str(inspect.signature(rp.ModelBank.gram).parameters["layout"].annotation)


@pytest.mark.parametrize("kind", ["mean", "sum", "rate"])
def test_m17_reverse_is_the_window_looking_ahead(kind):
    new = getattr(rp, f"ewm_{kind}")("x0", half_life=2.0, window_size=5.0, reverse=True)
    with pytest.raises(TypeError, match=f"rewm_{kind}.*renamed.*reverse=True"):
        getattr(rp, f"rewm_{kind}")("x0", half_life=2.0, window_size=5.0)
    assert new.meta.eq(new)
    assert f"rewm_{kind}" not in rp.__all__
    assert f"rewm_{kind}" not in rp.ops.__all__
    with pytest.raises(ValueError, match="reverse=True needs window_size"):
        getattr(rp, f"ewm_{kind}")("x0", half_life=2.0, reverse=True)


def test_m17_reverse_runs_through_with_windows_and_targets():
    df = _df()
    fwd = rp.ewm_mean("y", half_life=2.0, window_size=5.0, reverse=True)
    out = rp.stream.with_windows(df, fwd_y=fwd, clock="t", gap_cap=10.0)
    assert out["fwd_y"].null_count() < df.height
    spec = rp.spec.ewridge(
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
    with pytest.raises(ValueError, match="Write the string 'p' in targets"):
        rp.spec.ewridge("m", targets=[pl.col("p")], features=["x0"], half_life=10.0)


def test_m11_an_aliased_col_target_is_po_target():
    got = rp.spec.ewridge("m", targets=[pl.col("p").alias("price")], features=["x0"], half_life=1.0)
    want = rp.spec.ewridge(
        "m", targets=[rp.target("p", name="price")], features=["x0"], half_life=1.0
    )
    assert got == want


# --- M-13: renamed and deprecated names stay out of the snapshot and reference -----


def test_m13_renamed_and_deprecated_members_are_marked():
    from rheon._renamed import is_renamed

    for name in ("rows_seen", "to_json", "save_json"):
        assert is_renamed(getattr(rp.ModelBank, name)), name
    for name in ("fit", "save", "write_json"):
        assert not is_renamed(getattr(rp.ModelBank, name)), name
    snapshot = (Path(__file__).parent / "api_surface.txt").read_text(encoding="utf-8")
    section = snapshot.split("[ModelBank]")[1].split("\n[")[0]
    for name in ("rows_seen", "to_json", "save_json"):
        assert f"\n  {name}" not in section, name


# --- M-24: rheon.gram --------------------------------------------------------------


def _gram():
    spec = rp.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=1e9)
    df = _df().with_columns(x1=pl.col("x0") ** 2)
    bank = rp.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")[0]


def test_m24_subset_takes_columns_and_refuses_cols():
    g = _gram()
    assert list(inspect.signature(rp.gram.subset).parameters)[1] == "columns"
    want = rp.gram.subset(g, columns=["intercept", "x0"])
    assert want["columns"] == ["intercept", "x0"]
    with pytest.raises(TypeError, match="cols was renamed columns"):
        rp.gram.subset(g, cols=["intercept", "x0"])  # type: ignore[call-arg]


def test_m24_lars_takes_max_iter_and_refuses_max_steps():
    g = _gram()
    assert "max_iter" in inspect.signature(rp.gram.lars_path).parameters
    assert "max_iter" in inspect.signature(rp.gram.lars_paths).parameters
    rp.gram.lars_path(g, max_iter=1)
    with pytest.raises(TypeError, match="max_steps was renamed max_iter"):
        rp.gram.lars_path(g, max_steps=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="max_steps was renamed max_iter"):
        rp.gram.lars_paths([g], max_steps=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="path has max_steps, which was renamed max_iter"):
        rp.gram.solve_subsets([g], [[0]], path={"max_steps": 1})


def test_m24_solve_and_solve_subsets_take_the_same_ridge():
    g = _gram()
    ann = {
        str(inspect.signature(f).parameters["ridge"].annotation)
        for f in (rp.gram.solve, rp.gram.solve_subsets)
    }
    assert len(ann) == 1, ann
    grid = rp.gram.solve_subsets([g], [[0]], ridge=[0.0, 0.5])
    one = rp.gram.solve_subsets([g], [[0]], ridge=0.5)
    assert np.allclose(grid[0][0]["coef"][1], one[0][0]["coef"])
    assert np.allclose(grid[0][0]["coef"], rp.gram.solve(g, ridge=[0.0, 0.5]), rtol=1e-9)


def test_m24_a_gram_typed_dict():
    assert "Gram" in rp.gram.__all__
    assert rp.gram.Gram.__annotations__["columns"]
    ret = str(inspect.signature(rp.ModelBank.gram).return_annotation)
    assert "Gram" in ret


def test_one_correlation_of_a_gram():
    g = _gram()
    with pytest.raises(TypeError, match="rheon.corr.matrix.*renamed rheon.gram.correlation"):
        rp.corr.matrix(g)
    assert rp.gram.correlation(g).shape[0] == len(g["columns"])
    assert "matrix" not in rp.corr.__all__


# --- M-16: fit's docstring ------------------------------------------------------


def test_m16_fit_says_it_continues():
    assert "partial_fit" in (rp.ModelBank.fit.__doc__ or "")
