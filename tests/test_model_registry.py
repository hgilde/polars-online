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
import typing
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
from polars_online import _spec

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CORE_GOLDEN = ROOT / "crates" / "online-core" / "tests" / "golden.rs"

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
    "kmeans": {"targets": None, "features": ["x0", "x1"], "k": 2},
    "micro": {"targets": None, "features": ["x0", "x1"], "eps": 0.3},
    "ew_class": {"targets": None, "label": "y", "classes": ["a", "b"], "precision_prior": 1.0},
    "seqtest": {"features": None, "halflife": None},
    "marginal": {},
    "deco": {"targets": None, "features": ["x0", "x1"]},
    "corrchange": {
        "targets": None,
        "features": ["x0", "x1"],
        "halflife": None,
        "span_rows": 20,
    },
    "bocpd": {"targets": None, "features": ["x0", "x1"], "halflife": None},
    "hmm": {
        "targets": None,
        "features": ["x0", "x1"],
        "k": 2,
        "precision_prior": 0.1,
    },
    "rcov": {
        "targets": None,
        "features": ["x0", "x1"],
        "halflife": None,
        "group": "g",
        "group_close": "monotone",
        "block_rows": 100,
    },
}

#: The sweeps fit a numeric target, so the models that predict none --
#: ``ew_cov`` (moments), ``kmeans`` and ``micro`` (assignments, no target),
#: ``ew_class`` (a label), ``seqtest`` (evidence), ``marginal`` (pairwise
#: moments, read from the state), ``deco`` (an equicorrelation), ``rcov``
#: (a block's realised covariance, read at the group's close), ``hmm`` (a
#: hidden state), ``corrchange`` (a test statistic), ``bocpd`` (a posterior
#: over run lengths) -- sit them out.
REGRESSIONS = frozenset(MINIMAL) - {
    "ew_cov",
    "kmeans",
    "micro",
    "ew_class",
    "seqtest",
    "marginal",
    "deco",
    "rcov",
    "hmm",
    "corrchange",
    "bocpd",
}


def _build(name: str) -> dict:
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "halflife": 50.0}
    kw.update(MINIMAL[name])
    return getattr(po.spec, name)("m", **{k: v for k, v in kw.items() if v is not None})


#: What ``polars_online.spec`` exports besides builders.
HELPERS = {"output_fields", "coef_fields", "coef_index", "output_index"}


def _builders() -> set[str]:
    return set(po.spec.__all__) - HELPERS


def test_minimal_names_every_builder():
    assert set(MINIMAL) == _builders()


def test_coef_index_is_a_layout_or_a_reason_for_every_kind():
    """``coef_index`` refused three of the kinds with no ``coef`` by name and
    the others with whatever polars raises on an empty series
    (review 2026-09-12, S26)."""
    for name in sorted(MINIMAL):
        spec = _build(name)
        kind = spec["model"]["type"]
        try:
            index = po.spec.coef_index(spec)
        except ValueError as e:
            assert kind in str(e), (name, str(e))
            continue
        assert index["position"].to_list() == list(range(index.height)), name


def test_coef_every_is_taken_exactly_where_there_is_a_coef():
    """``coef_every`` schedules the ``coef`` field, so a kind whose output has
    none has nothing for it to do, and refuses it; the rule is held to the
    renderer of the fields rather than to a list (review 2026-09-12, S22)."""
    for name in sorted(MINIMAL):
        spec = _build(name)
        has_coef = any(f.startswith("coef") for f in po.spec.output_fields(spec))
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "halflife": 50.0}
        kw.update(MINIMAL[name])
        kw = {k: v for k, v in kw.items() if v is not None}
        builder = getattr(po.spec, name)
        if has_coef:
            builder("m", coef_every=5, **kw)
        else:
            try:
                builder("m", coef_every=5, **kw)
            except ValueError as e:
                assert "coef_every" in str(e), (name, str(e))
            else:
                raise AssertionError(f"{name} has no coef and took coef_every")


def _float_leaves(hint: object) -> set[object]:
    leaves = {hint, *typing.get_args(hint)}
    for _ in range(2):
        leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
    return leaves


def test_every_float_parameter_survives_a_state_file():
    """A float parameter comes back from a state file as the float it was --
    ``inf`` included -- only if ``_NUMERIC_KEYS`` names it, and that listed
    sixteen builders of twenty-one (review 2026-09-12, D8)."""
    for name in sorted(_builders()):
        fn = getattr(po.spec, name)
        hints = typing.get_type_hints(getattr(fn, "__wrapped__", fn))
        floats = {k for k, h in hints.items() if float in _float_leaves(h)}
        missing = floats - _spec._NUMERIC_KEYS
        assert not missing, (name, sorted(missing))


def test_every_rust_kind_has_exactly_one_builder():
    """A `ModelKind` variant nobody can construct from Python is dead code;
    two builders for one kind would be a `huber`/`quantile` style split that
    the README and the sweeps then have to know about."""
    kinds = _native.model_kinds()
    assert len(kinds) == len(set(kinds))
    built = {name: _build(name)["model"]["type"] for name in MINIMAL}
    assert sorted(built.values()) == sorted(kinds), built


def test_the_builder_list_covers_every_builder():
    """`test_kwargs_typing` parametrises its typed-dict checks on its own list
    of builders; this holds that list to what `po.spec` exports, so a model
    added to one cannot be missed by the other."""
    assert set(test_kwargs_typing.BUILDERS) == _builders()


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
    specs = test_golden_pipeline.specs()
    # One bank per kind, plus variants named `<kind>_<variant>` that pin a
    # feature of that kind (`sgd_simplex`, `pa_box`: the constrained path
    # of task 28, which the base banks never take).
    base = [s["model"]["type"] for s in specs if not _is_variant(s)]
    assert len(base) == len(set(base)), "a model is pinned twice; one bank per kind"
    assert set(base) == set(_native.model_kinds()), "the golden bank is missing a model"
    for s in specs:
        if _is_variant(s):
            kind = s["model"]["type"]
            assert kind in base, f"variant bank {s['name']!r} has no base bank for {kind!r}"


def _is_variant(spec) -> bool:
    return spec["name"].startswith(spec["model"]["type"] + "_")


#: Builders whose per-model file is named for the Rust kind, not the builder.
PER_MODEL_FILE = {"huber": "robust", "quantile": "robust"}


def test_every_builder_has_a_per_model_test_file():
    """`tests/test_<model>.py` is where a model's *arithmetic* is held to an
    oracle (docs/EXTENDING.md step 11); the sweeps hold every model to the
    shared invariants and cannot see a wrong coefficient. The file has to
    build the model itself -- a file that only imports it proves nothing.
    Before this check, `ewridge` and `rls` had theirs inside `test_bank.py`,
    where nothing said which model a test belonged to."""
    for builder in MINIMAL:
        path = ROOT / "tests" / f"test_{PER_MODEL_FILE.get(builder, builder)}.py"
        assert path.exists(), f"{builder} has no per-model test file {path.name}"
        assert f"po.spec.{builder}(" in path.read_text(encoding="utf-8"), (
            f"{path.name} never builds po.spec.{builder}(...)"
        )


def test_the_core_golden_file_pins_every_model():
    """`crates/online-core/tests/golden.rs` pins the core arithmetic of a
    model with a `fn <kind>_golden()`, so that a divergence the pipeline
    check above reports can be placed in the core or above it. It went four
    models without one (`sgd`, `pa`, `holt`, `ew_cov`) before this check."""
    text = CORE_GOLDEN.read_text(encoding="utf-8")
    pinned = set(re.findall(r"^fn ([a-z_]+)_golden\(\)", text, flags=re.MULTILINE))
    missing = set(_native.model_kinds()) - pinned
    assert not missing, f"{CORE_GOLDEN.name} has no fn <kind>_golden() for {sorted(missing)}"


def test_the_readme_documents_every_model():
    """Every builder gets a `#### \\`name\\`` heading under its family in
    "## Models"; the heading text is what the model table links to."""
    text = README.read_text(encoding="utf-8")
    models = text.split("\n## Models\n", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^#### `([a-z_]+)`", models, flags=re.MULTILINE))
    # `huber` and `quantile` share a heading: "#### `huber` / `quantile` -- ...".
    documented |= set(re.findall(r"^#### `[a-z_]+` / `([a-z_]+)`", models, flags=re.MULTILINE))
    assert documented == set(MINIMAL)
