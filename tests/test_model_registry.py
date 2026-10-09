"""Every model is wired through every layer, or a test here names the layer.

docs/EXTENDING.md lists the places a new model touches. Most of them have a
check that fails when one is skipped -- the compiler for the Rust match arms,
the API snapshot for the Python surface -- but the per-model sweeps in this
directory are plain lists, and a model left out of a list is simply never
swept. The registry is ``ModelKind::KINDS`` on the Rust side (held to the
enum by a unit test); these tests hold the builders, the sweeps, the README,
the release comparison's workload and ``_spec.UNSUPERVISED`` to it.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import typing
from pathlib import Path

import polars_online as po
import test_api_surface
import test_edge_cases
import test_golden_pipeline
import test_kwargs_typing
import test_portability
import test_semantics_all_models
from polars_online import _polars_online as _native
from polars_online import _spec

TIER = "essential"

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CORE_GOLDEN = ROOT / "crates" / "online-core" / "tests" / "golden.rs"

#: Builder -> the least it needs beyond targets/features/half-life to be
#: constructible; ``None`` drops that argument. A new builder goes here first,
#: and then wherever the tests below say.
MINIMAL: dict[str, dict[str, object]] = {
    "ewridge": {},
    "rls": {},
    "lasso": {"lasso_path": [0.1, 0.0]},
    "kalman": {"coef_half_life": 50.0},
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
    "seqtest": {"features": None, "half_life": None},
    "marginal": {},
    "deco": {"targets": None, "features": ["x0", "x1"]},
    "corrchange": {
        "targets": None,
        "features": ["x0", "x1"],
        "half_life": None,
        "span_rows": 20,
    },
    "bocpd": {"targets": None, "features": ["x0", "x1"], "half_life": None},
    "hmm": {
        "targets": None,
        "features": ["x0", "x1"],
        "k": 2,
        "precision_prior": 0.1,
    },
    "rcov": {
        "targets": None,
        "features": ["x0", "x1"],
        "half_life": None,
        "group": "g",
        "group_close": "monotone",
        "block_rows": 100,
    },
    "audit": {"targets": None, "features": None, "half_life": None, "columns": ["x0", "x1"]},
}

#: The sweeps fit a numeric target, so the models that predict none --
#: ``ew_cov`` (moments), ``kmeans`` and ``micro`` (assignments, no target),
#: ``ew_class`` (a label), ``seqtest`` (evidence), ``marginal`` (pairwise
#: moments, read from the state), ``deco`` (an equicorrelation), ``rcov``
#: (a block's realised covariance, read at the group's close), ``hmm`` (a
#: hidden state), ``corrchange`` (a test statistic), ``bocpd`` (a posterior
#: over run lengths), ``audit`` (counts, read from the state) -- sit them out.
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
    "audit",
}


def _build(name: str) -> dict:
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
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
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
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


#: ``marginal`` specs that give every key its builder writes only when it is
#: given, the ones the Rust spec skips when absent; three, as ``bin_edges``
#: is refused beside the learned bins, ``window_lags`` without a window, and
#: bins beside a window.
MARGINAL_GIVEN = [
    dict(
        lags=[1, 2],
        cross_lags=[2],
        serial_rule="bartlett",
        bins=4,
        bin_rule="fixed",
        bin_warm_rows=50,
        bin_budget=float("inf"),
        shards="auto",
        feature_moments="shared",
    ),
    dict(lags=[1], window_size=100.0, window_lags=True, shards=2),
    dict(bin_edges=[[-0.5, 0.5]], bin_budget=64.0),
]


def test_every_kind_reports_back_the_dict_its_builder_made():
    """A bank's ``specs`` are the dicts that built it, for every kind. The
    ``marginal`` builder wrote eleven optional keys as ``None``, which the
    Rust spec skips when absent, on purpose: a key written as null moves
    every spec's bytes (``crates/online-polars/src/spec.rs``, above
    ``lags``). So ``ModelBank([s]).specs[0] != s``, and a state saved and
    loaded compared unequal to its own spec (review 2026-10-05, the TB5
    leg in ``tests/test_properties_temporal.py``)."""
    specs = {name: _build(name) for name in MINIMAL}
    for i, given in enumerate(MARGINAL_GIVEN):
        specs[f"marginal, given {i}"] = po.spec.marginal(
            "m", targets=["y"], features=["x0"], half_life=50.0, **given
        )
    differ = {name: s for name, s in specs.items() if po.ModelBank([s]).specs[0] != s}
    assert not differ, sorted(differ)
    for given in MARGINAL_GIVEN:
        model = po.spec.marginal("m", targets=["y"], features=["x0"], half_life=50.0, **given)[
            "model"
        ]
        assert set(given) <= set(model), "each key given is written"


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
    """Output field names are API -- the README's *Output field names*: "a
    test holds every name, default and signature to a checked-in snapshot"
    -- and the snapshot is where they are pinned, one `<model> minimal:`
    block each. The quote is checked too: the one here before named a
    sentence the README no longer had (README-ITERATIONS, E15)."""
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "a test holds every name, default and signature to a checked-in snapshot" in readme
    surface = test_api_surface.describe_api()
    pinned = set(re.findall(r"^  ([a-z_]+) minimal:$", surface, flags=re.MULTILINE))
    assert pinned == set(MINIMAL), "tests/test_api_surface.py pins no output fields for a model"


def test_the_sweeps_cover_every_regression_model():
    # Here and not at the top: `test_properties` builds its lists from this
    # module's `MINIMAL` and `REGRESSIONS`, so importing it at the top made a
    # cycle that failed whenever this module was imported first.
    import test_properties

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


def test_the_cross_os_hand_off_holds_every_kind():
    """`crates/online-polars/tests/state_portability.rs` writes the state
    `release.yml` hands from macOS to Windows and Linux, and hard rule 5
    promises every kind's state loads on both. Its bank held one `ewridge`
    spec (review 2026-10-06, TA2); it holds each builder's kind by name
    now, each spec named after its builder, and this keeps it so."""
    text = (ROOT / "crates" / "online-polars" / "tests" / "state_portability.rs").read_text(
        encoding="utf-8"
    )
    table = text.split("const SPECS", 1)[1].split("];", 1)[0]
    named = set(re.findall(r'\(\s*"([a-z_]+)",', table))
    missing = set(MINIMAL) - named
    assert not missing, f"the hand-off bank has no spec for {sorted(missing)}"


def test_the_readme_documents_every_model():
    """Every builder gets a `#### \\`name\\`` heading under its family in
    "## Models"; the heading text is what the model table links to."""
    text = README.read_text(encoding="utf-8")
    models = text.split("\n## Models\n", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^#### `([a-z_]+)`", models, flags=re.MULTILINE))
    # `huber` and `quantile` share a heading: "#### `huber` / `quantile` -- ...".
    documented |= set(re.findall(r"^#### `[a-z_]+` / `([a-z_]+)`", models, flags=re.MULTILINE))
    assert documented == set(MINIMAL)


def test_the_release_workload_builds_every_model():
    """`scripts/release_probe.py`'s `WORKLOAD` is what `compare_release.py`
    holds a build to, bit for bit, and what the released-state and
    weight-scale checks run. A builder missing from it is compared by none
    of them, and until task 154 nothing said so."""
    spec = importlib.util.spec_from_file_location(
        "release_probe", ROOT / "scripts" / "release_probe.py"
    )
    assert spec and spec.loader
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert {builder for _, builder, _, _ in probe.WORKLOAD} == _builders()


def test_unsupervised_is_the_models_the_bank_fills_a_target_for():
    """`_spec.UNSUPERVISED` names the models with no target column. The
    Rust side's list is `ModelKind::is_unsupervised`, which Python cannot
    call; what it does can be seen, though: a spec of such a model given no
    `targets` is filled from `features[0]` (E53), and any other is refused.
    The two lists were kept in step by a comment alone until task 154."""
    filled = set()
    for name in MINIMAL:
        spec = dict(_build(name), targets=[])
        try:
            po.ModelBank([spec])
        except ValueError:
            continue
        filled.add(name)
    assert filled == _spec.UNSUPERVISED


def test_output_index_names_every_dtype_it_declares():
    """Task 159 (P3): the ``dtype`` column's vocabulary over every model's
    minimal spec, and one with the clock fields on, is the one
    `output_index`'s docstring lists; it listed four of the eight."""
    seen: set[str] = set()
    for name in MINIMAL:
        seen |= set(po.spec.output_index(_build(name))["dtype"].to_list())
    clocks = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        half_life=50.0,
        clock="t",
        gap_cap=1.0,
        emit_clocks=True,
    )
    seen |= set(po.spec.output_index(clocks)["dtype"].to_list())
    documented = {"f64", "bool", "str", "list[f64]", "enum", "i32", "i64", "clock"}
    assert seen == documented, seen ^ documented
    doc = po.spec.output_index.__doc__ or ""
    assert all(f"``{d}``" in doc for d in documented)


def _literal_templates(text: str) -> list[str]:
    """Each ``literal`` of ``text`` as a pattern a field name matches: a
    ``<placeholder>`` or a ``{placeholder}`` stands for any run of characters,
    and so does a ``*``. A literal counts only if it begins with text of its
    own: ``<stat>_<column>`` matched every field with an underscore in it,
    which made the test below vacuous for ``ew_cov`` (review 2026-10-06,
    YA10)."""
    patterns = []
    for lit in re.findall(r"``([^`]+)``", text):
        parts = [p for p in re.split(r"(<[^>]*>|\{[^}]*\}|\*)", lit) if p]
        if not parts or parts[0][0] in "<{*":
            continue
        patterns.append("".join(".+" if p[0] in "<{*" else re.escape(p) for p in parts))
    return patterns


def _rubric(doc: str, title: str) -> str:
    # `cleandoc` first: CPython 3.13+ dedents a docstring at compile time
    # and 3.12 does not, and the rubric's directive is matched at a line's
    # start either way.
    text = inspect.cleandoc(doc)
    found = re.search(rf"^\.\. rubric:: {title}\n(.*?)(?=^\.\. rubric::|\Z)", text, re.S | re.M)
    assert found, f"no {title} rubric"
    return found.group(1)


def test_every_builders_output_rubric_names_the_fields_its_plainest_spec_writes():
    """Review 2026-10-05 (YA10): every model writes ``settled_frac`` and
    ``withheld_reason``, and ``ewridge`` ``support_coef`` beside ``coef``, and
    no builder's Output rubric said so. Each field of the plainest spec is
    named in the rubric, literally or by a template such as ``pred_<t>``."""
    gaps = {}
    for name in MINIMAL:
        rubric = _rubric(getattr(po.spec, name).__doc__ or "", "Output")
        patterns = _literal_templates(rubric)
        fields = po.spec.output_fields(_build(name))
        missing = [f for f in fields if not any(re.fullmatch(p, f) for p in patterns)]
        if missing:
            gaps[name] = missing
    assert not gaps, gaps


def test_the_field_grammar_names_every_field_a_grid_writes():
    """Review 2026-10-05 (YA10): the grammar block of
    :mod:`polars_online.spec` left out the per-instance ``settled_frac``,
    ``withheld_reason`` and ``support_coef`` a half-life grid writes. And
    review 2026-10-06 (YA7): it left out what a lasso path and a selection
    write, ``penalty_selected_<t>`` and ``selected_<t>``, so the grid below
    holds a lasso path with both switches on, beside the ridge grid."""
    from polars_online import spec as spec_module

    block = re.search(
        r"^A grid writes one set of fields per instance.*?code-block:: text\n\n(.*?)\n\n",
        inspect.cleandoc(spec_module.__doc__ or ""),
        re.S | re.M,
    )
    assert block, "no grammar block"
    names = [line.split()[0] for line in block.group(1).splitlines() if line.strip()]
    patterns = [
        "".join(".*" if p.startswith("{") else re.escape(p) for p in re.split(r"(\{[^}]*\})", n))
        for n in names
        if not n.startswith("|")
    ]
    grid = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        ridge=[1e-6, 0.5],
        feature_sets={"a": ["x0"], "b": ["x0", "x1"]},
        half_life=[10.0, 100.0],
        emit_sigma=True,
    )
    path = po.spec.lasso(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        lasso_path=[0.1, 0.0],
        half_life=[10.0, 100.0],
        emit_selected=True,
        emit_averaged=True,
    )
    fields = po.spec.output_fields(grid) + po.spec.output_fields(path)
    assert "penalty_selected_y@h10" in fields and "selected_y" in fields, fields
    missing = [f for f in fields if not any(re.fullmatch(p, f) for p in patterns)]
    assert not missing, (missing, names)
