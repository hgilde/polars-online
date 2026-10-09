"""The ``**common`` of the spec builders is typed with PEP 692
``Unpack[TypedDict]`` (docs/IMPROVEMENTS.md U4), so an editor completes it and
a type checker catches a typo. Each TypedDict is a copy of a builder's
signature, and a copy drifts, so every one is pinned here to the builder it
mirrors: same keys, same annotations, same required set. Change a builder and
this says which class to update.
"""

from __future__ import annotations

import typing

import polars as pl
import pytest

import polars_online as po
from polars_online import _kwargs, _spec

TIER = "mixed"

# Every model with a spec builder: what the parametrised test below covers.
BUILDERS = [
    "audit",
    "bocpd",
    "corrchange",
    "deco",
    "hmm",
    "rcov",
    "ewridge",
    "rls",
    "lasso",
    "kalman",
    "huber",
    "quantile",
    "ftrl",
    "ew_cov",
    "sgd",
    "pa",
    "holt",
    "kmeans",
    "micro",
    "ew_class",
    "seqtest",
    "marginal",
]


def _unwrapped(builder):
    return getattr(builder, "__wrapped__", builder)


def _hints(fn) -> dict[str, object]:
    return {k: v for k, v in typing.get_type_hints(fn).items() if k != "return"}


def _common_hints() -> dict[str, object]:
    return {k: v for k, v in _hints(_spec._common).items() if k not in ("name", "model")}


def test_common_kwargs_mirror_the_shared_parameters():
    shared = {k: v for k, v in _common_hints().items() if k not in ("targets", "features")}
    assert typing.get_type_hints(_kwargs.CommonKwargs) == shared
    # `ExprKwargs` is the shared base: everything but the group and its
    # close policy, which only the builders take.
    assert typing.get_type_hints(_kwargs.ExprKwargs) == {
        k: v for k, v in shared.items() if k not in ("group", "group_close")
    }
    assert _kwargs.CommonKwargs.__required_keys__ == frozenset()


@pytest.mark.parametrize("name", BUILDERS)
def test_each_builder_takes_common_as_the_typed_dict(name):
    builder = _unwrapped(getattr(_spec, name))
    assert typing.get_type_hints(builder)["common"] == typing.Unpack[_kwargs.CommonKwargs]


def test_a_typo_is_still_named_at_runtime():
    with pytest.raises(
        TypeError, match="ewridge\\(\\) got an unexpected keyword argument 'halflif'"
    ):
        po.spec.ewridge("m", targets=["y"], features=["x0"], halflif=10.0)


def test_the_typed_dicts_change_nothing_at_runtime():
    # A TypedDict is a plain dict at runtime: the same kwargs reach the same
    # builder, and a required key missing is still the builder's error.
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x0": [1.0, 3.0, 2.0, 5.0]})
    spec = po.spec.ewridge("f", targets=["y"], features=["x0"], half_life=2.0, min_weight=1.0)
    out = po.ModelBank([spec]).fit_predict(df)
    assert isinstance(out.schema["f"], pl.Struct)
    assert out["f"].struct.field("pred_y").null_count() < 4
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'lasso_path'"):
        po.spec.lasso("m", targets=["y"], features=["x0"], half_life=2.0)


#: Every form of ``targets`` the docs show, as a caller writes it. A builder's
#: ``targets`` was a union of two ``list`` types, and ``list`` is invariant, so
#: mypy refused a list of tables, a list of expressions, a mix and a dict
#: target (review 2026-10-05, YA4).
_TARGET_FORMS = """
import polars as pl

import polars_online as po

t = po.target("y", name="m")
cols: list[str] = ["y"]
mixed: list[str | po.Target] = ["y", t]
forward = (po.rewm_mean("y", half_life="1m", window_size="1m") - pl.col("y")).alias("f")
po.spec.ewridge("m", targets=["y"], features=["x"])
po.spec.ewridge("m", targets=cols, features=cols)
po.spec.ewridge("m", targets=[t], features=["x"])
po.spec.ewridge("m", targets=[forward], features=["x"], embargo="1m", clock="t", gap_cap="1m")
po.spec.ewridge("m", targets=["y", t], features=["x"])
po.spec.ewridge("m", targets=mixed, features=["x"])
po.spec.ewridge("m", targets=["ret", {"column": "p", "name": "price"}], features=["x"])
po.spec.rls("m", targets=("y", t), features=["x"])
"""


@pytest.mark.extended(reason="a second or more: a mypy run with a cold cache (2.0 s)")
def test_every_documented_target_form_type_checks(tmp_path, monkeypatch):
    from pathlib import Path

    from mypy import api

    snippet = tmp_path / "forms.py"
    snippet.write_text(_TARGET_FORMS)
    # The package's own source and stub, as `uv run mypy` reads them, so
    # nothing built is needed. An installed wheel is found where it is
    # (it ships `py.typed`), and mypy refuses a site-packages in MYPYPATH.
    root = Path(_spec.__file__).parents[1]
    if root.name not in ("site-packages", "dist-packages"):
        monkeypatch.setenv("MYPYPATH", str(root))
    out, err, status = api.run(
        [str(snippet), "--cache-dir", str(tmp_path / "cache"), "--python-version", "3.12"]
    )
    assert status == 0, out + err


def test_the_native_stub_names_the_built_module():
    # `_polars_online.pyi` is what a type checker sees of the pyo3 module; it
    # went stale once (no gram, no spec_output_index), so it is checked
    # against what the built module actually exports.
    import ast
    from pathlib import Path

    from polars_online import _polars_online as native

    stub = Path(native.__file__).with_name("_polars_online.pyi").read_text()
    tree = ast.parse(stub)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    exported = {n for n in dir(native) if not n.startswith("_")}
    assert functions == exported - set(classes)
    assert set(classes) <= exported, "the stub declares a class the module has not got"
    for name, cls in classes.items():
        declared = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        # The public names must match exactly, both ways: a method the stub
        # lacks and a method the class lacks are each a stale stub.
        public = {n for n in declared if not n.startswith("_")}
        want = {n for n in dir(getattr(native, name)) if not n.startswith("_")}
        assert public == want, name
        # A dunder the stub declares -- a protocol method like
        # `__arrow_c_array__` -- must exist on the class, or a type checker is
        # being told about a method that is not there. The class may have
        # dunders the stub does not declare; pyo3 adds many.
        dunders = {n for n in declared if n.startswith("__") and n != "__init__"}
        missing = dunders - set(dir(getattr(native, name)))
        assert not missing, (
            f"{name}: the stub declares {sorted(missing)}, which the class has not got"
        )


def test_each_native_item_carries_its_own_docstring():
    """Review 2026-10-05 (YA9): ``format_of_path``'s doc comment sat above the
    ``RefreshTime`` class in ``lib.rs``, so the class's docstring opened with
    the function's text and the function had none."""
    from polars_online import _polars_online as native

    undocumented = [
        n for n in dir(native) if not n.startswith("_") and not getattr(native, n).__doc__
    ]
    assert not undocumented, undocumented
    assert "extension" in native.format_of_path.__doc__
    assert native.RefreshTime.__doc__.startswith("Refresh-time sampling"), (
        native.RefreshTime.__doc__[:80]
    )
