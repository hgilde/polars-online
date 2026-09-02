"""Every model is wired through every layer, or a test here names the layer.

docs/EXTENDING.md lists the places a new model touches. Most of them have a
check that fails when one is skipped -- the compiler for the Rust match arms,
the API snapshot for the Python surface -- but the per-model sweeps in this
directory are plain lists, and a model left out of a list is simply never
swept. The registry is ``ModelKind::KINDS`` on the Rust side (held to the
enum by a unit test); these tests hold the builders, the sweeps and the README
to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars_online as po
import test_api_surface
import test_edge_cases
import test_golden_pipeline
import test_kwargs_typing
import test_portability
import test_properties
import test_semantics_all_models
from polars_online import _polars_online as _native

README = Path(__file__).resolve().parent.parent / "README.md"

#: Builder -> the least it needs beyond targets/features/halflife to be
#: constructible; ``None`` drops that argument. A new builder goes here first,
#: and then wherever the tests below say.
MINIMAL: dict[str, dict[str, object]] = {
    "ewridge": {},
    "rls": {},
    "lasso": {"lasso_path": [0.1, 0.0]},
    "kalman": {"coef_halflife": 50.0},
    "huber": {},
    "quantile": {"quantile": 0.5},
    "ftrl": {},
    "ew_cov": {"targets": None, "features": ["x0", "x1"]},
    "sgd": {"learning_rate": 0.01},
    "pa": {},
    "holt": {"features": None},
}

#: The sweeps fit a target, so ``ew_cov`` -- moments, no target -- sits them out.
REGRESSIONS = frozenset(MINIMAL) - {"ew_cov"}


def _build(name: str) -> dict:
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "halflife": 50.0}
    kw.update(MINIMAL[name])
    return getattr(po.spec, name)("m", **{k: v for k, v in kw.items() if v is not None})


#: What ``polars_online.spec`` exports besides builders.
HELPERS = {"output_fields", "coef_index", "output_index"}


def _builders() -> set[str]:
    return set(po.spec.__all__) - HELPERS


def test_minimal_names_every_builder():
    assert set(MINIMAL) == _builders()


def test_every_rust_kind_has_exactly_one_builder():
    """A `ModelKind` variant nobody can construct from Python is dead code;
    two builders for one kind would be a `huber`/`quantile` style split that
    the README and the sweeps then have to know about."""
    kinds = _native.model_kinds()
    assert len(kinds) == len(set(kinds))
    built = {name: _build(name)["model"]["type"] for name in MINIMAL}
    assert sorted(built.values()) == sorted(kinds), built


def test_every_builder_has_a_namespace_method():
    """`test_kwargs_typing` holds the namespace's public methods to its own
    list; this holds that list to the builders, so a model reachable from
    `po.spec` is reachable from `pl.col(...).online` too."""
    assert set(test_kwargs_typing.NAMESPACE_METHODS) == _builders()


def test_the_api_snapshot_pins_every_models_output_fields():
    """Output field names are API (README, "Output field names are part of
    the API"), and the snapshot is where they are pinned -- one
    `<model> minimal:` block each."""
    surface = test_api_surface.describe_api()
    pinned = set(re.findall(r"^  ([a-z_]+) minimal:$", surface, flags=re.MULTILINE))
    assert pinned == set(MINIMAL), "tests/test_api_surface.py pins no output fields for a model"


def test_the_sweeps_cover_every_regression_model():
    sweeps = {
        "test_semantics_all_models.MODELS": test_semantics_all_models.MODELS,
        "test_properties.MODELS": test_properties.MODELS,
        "test_edge_cases.MODELS": test_edge_cases.MODELS,
        "test_portability.TestOutputSchemaStability._ALL_MODELS": (
            test_portability.TestOutputSchemaStability._ALL_MODELS
        ),
    }
    for where, models in sweeps.items():
        names = [m for m, _ in models]
        assert len(names) == len(set(names)), f"{where} lists a model twice"
        assert set(names) == REGRESSIONS, f"{where} is missing a model"


def test_the_golden_pipeline_pins_every_model():
    """The cross-platform golden numbers are only a guarantee for the models
    they include. This check found `ftrl` missing from them on its first run:
    nine models were pinned on three operating systems and the tenth was
    not, and nothing said so."""
    pinned = [spec["model"]["type"] for spec in test_golden_pipeline.specs()]
    assert len(pinned) == len(set(pinned)), "a model is pinned twice; one bank per kind"
    assert set(pinned) == set(_native.model_kinds()), "the golden bank is missing a model"


def test_the_readme_documents_every_model():
    """Every builder gets a `### \\`name\\`` heading under "## Models"; the
    heading text is what the section index links to."""
    text = README.read_text(encoding="utf-8")
    models = text.split("\n## Models\n", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^### `([a-z_]+)`", models, flags=re.MULTILINE))
    # `huber` and `quantile` share a heading: "### `huber` / `quantile` -- ...".
    documented |= set(re.findall(r"^### `[a-z_]+` / `([a-z_]+)`", models, flags=re.MULTILINE))
    assert documented == set(MINIMAL)
