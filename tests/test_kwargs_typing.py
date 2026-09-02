"""The ``**kwargs`` of the expression namespace and the ``**common`` of the
builders are typed with PEP 692 ``Unpack[TypedDict]`` (docs/IMPROVEMENTS.md
U4), so an editor completes them and a type checker catches a typo. Each
TypedDict is a copy of a builder's signature, and a copy drifts, so every one
is pinned here to the builder it mirrors: same keys, same annotations, same
required set. Change a builder and this says which class to update.
"""

from __future__ import annotations

import inspect
import typing

import polars as pl
import pytest

import polars_online as po
from polars_online import _expr, _kwargs, _spec

# What the expression supplies itself, and so does not take as a keyword.
EXPR_SUPPLIES = {"name", "targets", "features", "group"}
NAMESPACE_METHODS = [
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
]


def _unwrapped(builder):
    return getattr(builder, "__wrapped__", builder)


def _hints(fn) -> dict[str, object]:
    return {k: v for k, v in typing.get_type_hints(fn).items() if k != "return"}


def _common_hints() -> dict[str, object]:
    return {k: v for k, v in _hints(_spec._common).items() if k not in ("name", "model")}


def _typed_dict_behind(method) -> type:
    kw = typing.get_type_hints(method)["kwargs"]
    assert typing.get_origin(kw) is typing.Unpack, kw
    (td,) = typing.get_args(kw)
    assert typing.is_typeddict(td), td
    return td


def test_the_namespace_methods_are_the_builders():
    public = [n for n in dir(_expr.OnlineNamespace) if not n.startswith("_")]
    assert sorted(public) == sorted(NAMESPACE_METHODS)


def test_common_kwargs_mirror_the_shared_parameters():
    shared = {k: v for k, v in _common_hints().items() if k not in ("targets", "features")}
    assert typing.get_type_hints(_kwargs.CommonKwargs) == shared
    # The expression form is the same minus the group, which is .over()'s job.
    assert typing.get_type_hints(_kwargs.ExprKwargs) == {
        k: v for k, v in shared.items() if k != "group"
    }
    assert _kwargs.CommonKwargs.__required_keys__ == frozenset()


@pytest.mark.parametrize("name", NAMESPACE_METHODS)
def test_each_builder_takes_common_as_the_typed_dict(name):
    builder = _unwrapped(getattr(_spec, name))
    assert typing.get_type_hints(builder)["common"] == typing.Unpack[_kwargs.CommonKwargs]


@pytest.mark.parametrize("name", NAMESPACE_METHODS)
def test_each_namespace_typed_dict_mirrors_its_builder(name):
    builder = _unwrapped(getattr(_spec, name))
    td = _typed_dict_behind(getattr(_expr.OnlineNamespace, name))

    own = {k: v for k, v in _hints(builder).items() if k not in EXPR_SUPPLIES | {"common"}}
    shared = {k: v for k, v in _common_hints().items() if k not in EXPR_SUPPLIES}
    assert typing.get_type_hints(td) == {**shared, **own}, name

    required = {
        p.name
        for p in inspect.signature(builder).parameters.values()
        if p.default is inspect.Parameter.empty and p.name not in EXPR_SUPPLIES | {"common"}
    }
    assert set(td.__required_keys__) == required, name


def test_po_online_is_the_registered_namespace():
    # pl.col("y").online is invisible to a type checker ("Expr" has no
    # attribute "online"); po.online(expr) is the same thing, visibly typed.
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x0": [1.0, 3.0, 2.0, 5.0]})
    typed = po.online(pl.col("y")).ewridge(features=["x0"], halflife=2.0)
    registered = pl.col("y").online.ewridge(features=["x0"], halflife=2.0)
    assert typed.meta.eq(registered)
    assert df.with_columns(typed).equals(df.with_columns(registered))


def test_the_expression_refuses_a_group_keyword():
    # The Rust side sets group = None: it would have been silently ignored.
    with pytest.raises(TypeError, match=r"group is not an expression parameter.*\.over\('g'\)"):
        pl.col("y").online.ewridge(features=["x0"], halflife=10.0, group="g")


def test_a_typo_is_still_named_at_runtime():
    with pytest.raises(
        TypeError, match="ewridge\\(\\) got an unexpected keyword argument 'halflif'"
    ):
        pl.col("y").online.ewridge(features=["x0"], halflif=10.0)


def test_the_typed_dicts_change_nothing_at_runtime():
    # A TypedDict is a plain dict at runtime: the same kwargs reach the same
    # builder, and a required key missing is still the builder's error.
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x0": [1.0, 3.0, 2.0, 5.0]})
    out = df.with_columns(pl.col("y").online.ewridge(features=["x0"], halflife=2.0).alias("f"))
    assert isinstance(out.schema["f"], pl.Struct)
    assert out["f"].struct.field("pred_y").null_count() < 4
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'lasso_path'"):
        pl.col("y").online.lasso(features=["x0"], halflife=2.0)


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
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelBank"]
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} - {"__init__"}
    assert functions == {n for n in dir(native) if not n.startswith("_")} - {"ModelBank"}
    assert methods == {n for n in dir(native.ModelBank) if not n.startswith("_")}
