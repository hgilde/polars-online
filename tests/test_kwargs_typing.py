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

# Every model with a spec builder: what the parametrised test below covers.
BUILDERS = [
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
    spec = po.spec.ewridge("f", targets=["y"], features=["x0"], halflife=2.0, min_periods=1.0)
    out = po.ModelBank([spec]).fit_predict(df)
    assert isinstance(out.schema["f"], pl.Struct)
    assert out["f"].struct.field("pred_y").null_count() < 4
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'lasso_path'"):
        po.spec.lasso("m", targets=["y"], features=["x0"], halflife=2.0)


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
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} - {"__init__"}
        want = {n for n in dir(getattr(native, name)) if not n.startswith("_")}
        assert methods == want, name
