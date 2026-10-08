"""The surfaces 1.0 does not promise say so (docs/PLAN.md task 198; review
round 4, D5, N23, AP12, DA4 and YB11): a docstring admonition, and an
``UnstableWarning`` raised only under ``POLARS_ONLINE_WARN_UNSTABLE=1``, as
Polars raises its own under ``POLARS_WARN_UNSTABLE``.

The surfaces: the ``with_windows`` state file's format, the formula tree's
written form (a formula target in a TOML file -- the command line's, in
``crates/online-cli/tests/run.rs`` -- and in a saved state),
``fit_predict_arrow``, ``predict_arrow`` and ``ArrowStruct``, and the modules
``po.sim`` and ``po.corr``. Each emits under the variable and is silent
without it.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online._warnings import UNSTABLE_VAR

TIER = "essential"

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def frame(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((n, 2))
    return pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": x @ [1.0, -0.5] + 0.1 * rng.standard_normal(n),
            "mid": 100.0 + np.cumsum(0.1 * rng.standard_normal(n)),
        }
    )


def ridge() -> dict:
    return po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=20.0)


def formula_spec() -> dict:
    target = (po.rewm_mean("mid", half_life=5.0, window_size=10.0) - pl.col("mid")).alias("fwd")
    return po.spec.ewridge(
        "edge",
        targets=[target],
        features=["x0"],
        clock="t",
        gap_cap=50.0,
        half_life=40.0,
        embargo=12.0,
    )


def unstable_calls(tmp_path) -> dict[str, Callable[[], object]]:
    """One call of each labelled surface."""
    df = frame()
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    held = po.ModelBank([formula_spec()])
    held.fit_predict(df)
    saved = tmp_path / "held.state"

    def windows_state() -> object:
        out = po.stream.with_windows(
            df,
            level=po.ewm_mean("mid", half_life=5.0, window_size=10.0),
            clock="t",
            gap_cap=50.0,
            save_state=tmp_path / "windows.state",
        )
        return out

    def save_formula() -> object:
        held.save(saved)
        return saved

    def load_formula() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.UnstableWarning)
            held.save(saved)
        return po.ModelBank.load(saved)

    return {
        "with_windows state": windows_state,
        "save a formula target": save_formula,
        "save_bytes a formula target": held.save_bytes,
        "to_json a formula target": held.to_json,
        "load a formula target": load_formula,
        "fit_predict_arrow": lambda: po.ModelBank([ridge()]).fit_predict_arrow(df),
        "predict_arrow": lambda: po.ModelBank([ridge()]).predict_arrow(df),
        "po.sim.regimes": lambda: po.sim.regimes(
            2, states=[np.eye(2)], transition=[[1.0]], n_blocks=2, rows_per_block=5, seed=1
        ),
        "po.corr.nearest": lambda: po.corr.nearest(corr),
        "po.corr.shrink": lambda: po.corr.shrink(corr, alpha=0.3),
    }


@pytest.fixture
def calls(tmp_path, monkeypatch):
    monkeypatch.delenv(UNSTABLE_VAR, raising=False)
    return unstable_calls(tmp_path)


SURFACES = [
    "with_windows state",
    "save a formula target",
    "save_bytes a formula target",
    "to_json a formula target",
    "load a formula target",
    "fit_predict_arrow",
    "predict_arrow",
    "po.sim.regimes",
    "po.corr.nearest",
    "po.corr.shrink",
]


def test_the_warning_is_exported_and_named_as_polars_names_it():
    assert issubclass(po.UnstableWarning, Warning)
    assert "UnstableWarning" in po.__all__
    assert UNSTABLE_VAR == "POLARS_ONLINE_WARN_UNSTABLE"


@pytest.mark.parametrize("surface", SURFACES)
def test_each_surface_warns_under_the_variable(surface, calls, monkeypatch):
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    with pytest.warns(po.UnstableWarning, match="considered unstable") as caught:
        calls[surface]()
    assert caught, surface


@pytest.mark.parametrize("value", [None, "0", "true"])
@pytest.mark.parametrize("surface", SURFACES)
def test_each_surface_is_silent_without_it(surface, value, calls, monkeypatch):
    if value is not None:
        monkeypatch.setenv(UNSTABLE_VAR, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.UnstableWarning)
        calls[surface]()


def test_a_promised_surface_never_warns(monkeypatch, tmp_path):
    """The bank, its file without a formula target, the window operators
    without a state, and the eval and gram modules are promised."""
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    df = frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.UnstableWarning)
        bank = po.ModelBank([ridge()])
        out = bank.fit_predict(df)
        bank.save(tmp_path / "plain.state")
        po.ModelBank.load(tmp_path / "plain.state")
        bank.to_json()
        po.stream.with_windows(
            df, level=po.ewm_mean("mid", half_life=5.0, window_size=10.0), clock="t", gap_cap=50.0
        )
        po.eval.metrics(out, "m")


def test_a_corr_call_that_runs_others_warns_once(monkeypatch):
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        po.corr.shrink(corr, alpha=0.3)
    got = [w for w in caught if issubclass(w.category, po.UnstableWarning)]
    assert len(got) == 1, [str(w.message) for w in got]
    assert "polars_online.corr.shrink" in str(got[0].message)


def test_every_corr_function_is_labelled():
    for name in po.corr.__all__:
        fn = getattr(po.corr, name)
        if callable(fn):
            assert hasattr(fn, "__wrapped__"), name


@pytest.mark.parametrize(
    "obj",
    [
        po.sim,
        po.corr,
        po.ArrowStruct,
        po.ModelBank.fit_predict_arrow,
        po.ModelBank.predict_arrow,
        po.ModelBank.save,
        po.stream.with_windows,
    ],
    ids=lambda o: getattr(o, "__qualname__", getattr(o, "__name__", str(o))),
)
def test_each_surface_says_so_in_its_docstring(obj):
    doc = inspect.getdoc(obj) or ""
    assert ".. warning::" in doc and "**unstable**" in doc, doc[-600:]
