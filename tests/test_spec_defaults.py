"""The README's warm-up defaults, held to the ones the bank resolves.

The README's *Warm-up* section states each readiness gate's default, and in
a table the default `min_weight` of every model as a rule. The bank chooses
them in Rust (`Spec::default_min_periods`, the `*_or_default` methods), and
every builder passes `None` for them, so until `_polars_online.resolved_defaults`
no Python surface showed them and a change to any row would have shipped
silently (review 2026-10-06, TB1). This reads both tables from the README
and checks them against what the bank resolves, for every kind, so the
README cannot drift from the code. `tests/api_surface.txt` pins the
resolved values themselves (`[resolved defaults]`).
"""

import json
import re
from pathlib import Path

import polars_online as po
from polars_online import _polars_online as native
from polars_online import _spec
from test_model_registry import MINIMAL

TIER = "essential"

README = Path(__file__).resolve().parent.parent / "README.md"

#: Each row of the README's `min_weight` table, by its first cell, as the
#: rule it states: the threshold for `k` features, with or without an
#: intercept. A row reworded in the README fails below until it is here.
RULES = {
    "one per unknown: the features, and the intercept when there is one": (
        lambda k, intercept: k + intercept
    ),
    "the feature count plus one (1 for `holt`)": lambda k, intercept: k + 1,
    "3": lambda k, intercept: 3,
    "1": lambda k, intercept: 1,
    "0, each having a gate of its own or nothing to gate": lambda k, intercept: 0,
}


def table_after(marker: str) -> list[list[str]]:
    """The cells of the first Markdown table after `marker`, header and rule
    rows dropped."""
    text = README.read_text(encoding="utf-8")
    at = text.index(marker)
    rows: list[list[str]] = []
    for line in text[at:].splitlines()[1:]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[2:]


def resolved(spec: dict) -> dict:
    return json.loads(native.resolved_defaults(_spec._json(spec)))


def build(name: str, k: int, intercept: bool) -> dict:
    """`name`'s minimal spec (test_model_registry) with `k` features, for a
    model that takes features, and the intercept on or off."""
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
    kw.update(MINIMAL[name])
    if kw.get("features") is not None:
        kw["features"] = [f"x{i}" for i in range(k)]
    if not intercept:
        kw["fit_intercept"] = False
    return getattr(po.spec, name)("m", **{key: v for key, v in kw.items() if v is not None})


def test_the_readme_min_weight_table_is_what_the_bank_resolves():
    rows = table_after("The default depends on the model:")
    assert [row[0] for row in rows] == list(RULES), "the README's table changed its rows"
    rule_of = {model: RULES[row[0]] for row in rows for model in re.findall(r"`([a-z_]+)`", row[1])}
    named = [model for row in rows for model in re.findall(r"`([a-z_]+)`", row[1])]
    assert sorted(named) == sorted(MINIMAL), "the table names every model exactly once"
    checked = 0
    for name in sorted(MINIMAL):
        takes_features = MINIMAL[name].get("features", ["x0"]) is not None
        # The intercept moves the first rule alone; the others are checked
        # with it on, as a spec is written.
        counts = (2, 3) if takes_features else (0,)
        intercepts = (True, False) if rule_of[name] is RULES[rows[0][0]] else (True,)
        for k in counts:
            for intercept in intercepts:
                spec = build(name, k, intercept)
                want = float(rule_of[name](k, intercept))
                got = resolved(spec)["stream"]["min_weight"]
                assert got == [want] * len(got) and got, (name, k, intercept, got, want)
                checked += 1
    # 8 models by the first rule at 2 counts x 2 intercepts, 11 others at
    # 2 counts, and `holt`, `seqtest` and `audit`, which take no features
    # (an audit's are its `columns`), at one.
    assert checked == 8 * 4 + 11 * 2 + 3


def test_the_readme_readiness_gate_defaults_are_what_the_bank_resolves():
    """The two readiness gates, in the README's table above `min_weight`'s:
    `min_settled_frac` off, `max_error_inflation` sqrt(2) on `ewridge` and
    off on the other models that have it (docs/PLAN.md task 116)."""
    defaults = {row[0]: row[1] for row in table_after("### Warm-up")}
    assert defaults == {
        "`min_weight`": "by model, below",
        "`min_settled_frac`": "`0`, off",
        "`max_error_inflation`": "`sqrt(2)` on `ewridge`; off on `rls`, `kalman` and `lasso`",
    }
    kw = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
    stream = resolved(po.spec.ewridge("m", **kw))["stream"]
    assert stream["min_settled_frac"] == 0.0
    assert stream["max_error_inflation"] == 2**0.5
    for spec in (
        po.spec.rls("m", **kw),
        po.spec.kalman("m", coef_half_life=50.0, **kw),
        po.spec.lasso("m", penalties=[0.1], **kw),
    ):
        assert resolved(spec)["stream"]["max_error_inflation"] == "inf", spec["model"]
    # The two readiness row fields are opt-in on every model that has them
    # (task 116, F): a builder writes them off, and the bank adds nothing.
    for build, extra in (
        (po.spec.ewridge, {}),
        (po.spec.rls, {}),
        (po.spec.kalman, {"coef_half_life": 50.0}),
    ):
        spec = build("m", **kw, **extra)
        assert (spec["emit_error_inflation"], spec["emit_se_coef"]) == (False, False), spec["model"]


def test_the_defaults_task_195_decided_are_what_the_bank_resolves():
    """docs/PLAN.md task 195 (U1, U2, U3, U5, N11): the Huber constant is
    1.345 under one name in both models that take it, `eps` is 0.01 of the
    target's own spread in both (task 202 moved the unit, task 203 the
    value, task 207 measured it against a residual-scaled band and kept
    it), `pa`'s `c` is 1 in the target's units, `sgd` and `pa`
    standardize, `bocpd`'s `nu`
    is the smallest integer giving each predictive's variance a mean, and
    `rls`'s prior strength is `delta`."""
    kw = {"targets": ["y"], "features": ["x0", "x1", "x2"], "half_life": 50.0}
    assert resolved(po.spec.huber("m", **kw))["model"]["loss"] == {"huber": {"delta": 1.345}}
    sgd = resolved(po.spec.sgd("m", **kw, loss="huber"))["model"]
    assert sgd["loss"] == {"huber": {"delta": 1.345}}
    assert sgd["standardize"] is True
    eps = resolved(po.spec.sgd("m", **kw, loss="epsilon_insensitive"))["model"]["loss"]
    assert eps == {"epsilon_insensitive": {"eps": 0.01}}
    pa = resolved(po.spec.pa("m", **kw))["model"]
    assert (pa["eps"], pa["c"], pa["standardize"]) == (0.01, 1.0, True)
    assert resolved(po.spec.pa("m", **kw, c=0.1))["model"]["c"] == 0.1
    given = resolved(po.spec.sgd("m", **kw, loss="epsilon_insensitive", eps=0.1))["model"]
    assert given["loss"] == {"epsilon_insensitive": {"eps": 0.1}}
    assert resolved(po.spec.pa("m", **kw, eps=0.1))["model"]["eps"] == 0.1
    assert resolved(po.spec.rls("m", **kw))["model"]["delta"] == 1.0
    feats = {"features": kw["features"]}
    for emission, nu in (
        ({"covariance_type": "diag"}, 3.0),
        ({"robust_beta": 0.1}, 3.0),
        ({"covariance_type": "full"}, 5.0),
    ):
        derived = resolved(po.spec.bocpd("m", **feats, **emission))["derived"]
        assert derived["degrees_of_freedom_prior"] == nu, emission
        assert derived["warm_rows"] == 5, emission


def _spec_module_doc() -> str:
    import inspect

    from polars_online import spec as spec_module

    # `cleandoc` first: CPython 3.13+ dedents a docstring at compile time.
    return inspect.cleandoc(spec_module.__doc__ or "")


def _list_table_after(doc: str, marker: str) -> list[list[str]]:
    """The cells of the first ``list-table`` after `marker` in `doc`, the
    header row dropped and each cell's wrapped lines joined."""
    lines = doc[doc.index(marker) :].splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == ".. list-table::")
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        s = line.strip()
        if s.startswith(":") or not s:
            if rows and not s and rows[-1]:
                continue
            continue
        if s.startswith("* - "):
            rows.append([s[4:]])
        elif s.startswith("- ") and rows:
            rows[-1].append(s[2:])
        elif rows and not line.startswith((" " * 4, "\t")):
            break
        elif rows:
            rows[-1][-1] += " " + s
    return [[cell.replace("``", "`") for cell in row] for row in rows[1:]]


def test_the_spec_module_min_weight_table_is_what_the_bank_resolves():
    """Task 239: the `min_weight` default table in :mod:`polars_online.spec`'s
    own docstring, held to the bank as the README's is above. It left out
    `audit`, which resolves 0 and refuses a `min_weight`."""
    rows = _list_table_after(_spec_module_doc(), "The default depends on")
    assert [row[0] for row in rows] == list(RULES), "the docstring's table changed its rows"
    named = [model for row in rows for model in re.findall(r"`([a-z_]+)`", row[1])]
    assert sorted(named) == sorted(MINIMAL), "the table names every model exactly once"
    for row in rows:
        for name in re.findall(r"`([a-z_]+)`", row[1]):
            takes_features = MINIMAL[name].get("features", ["x0"]) is not None
            k = 2 if takes_features else 0
            got = resolved(build(name, k, True))["stream"]["min_weight"]
            assert got and got == [float(RULES[row[0]](k, True))] * len(got), (name, got)


def test_the_spec_module_states_each_standardize_default():
    """Task 239: the module docstring's sentence on ``standardize`` named
    seven models and gave ``sgd`` the wrong default, and left out ``pa``.
    Every builder that takes it is named once, under the default the bank
    resolves."""
    doc = " ".join(_spec_module_doc().split())
    found = re.search(
        r"``standardize``, which (\w+) models take, defaults to ``False`` in (.*?), and to "
        r"``True`` in (.*?)\. ",
        doc,
    )
    assert found, "no standardize sentence"
    off = set(re.findall(r"``([a-z_]+)``", found.group(2)))
    on = set(re.findall(r"``([a-z_]+)``", found.group(3)))
    takes = {
        name
        for name in MINIMAL
        if "standardize" in getattr(po.spec, name).__wrapped__.__code__.co_varnames
    }
    assert off | on == takes and not off & on, (off, on, takes)
    words = {7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    assert found.group(1) == words[len(takes)]
    for name in takes:
        model = resolved(build(name, 2, True))["model"]
        assert model["standardize"] is (name in on), name


def test_the_spec_module_lists_every_clock_parameter():
    """Task 239: *Clock units* lists the parameters measured in clock units.
    Each one a builder's annotation lets take a duration is there, and
    nothing else, but for ``bocpd``'s ``hazard``, which the next paragraph
    gives its own two forms, and the diagnostics' memories, which the list
    names as ``*_half_life``."""
    doc = _spec_module_doc()
    para = doc[doc.index("The clock parameters are") :].split("\n\n", 1)[0]
    named = set(re.findall(r"``([a-z_*]+)``", para)) - set(MINIMAL)
    assert "*_half_life" in named
    diagnostics = {k for k in _spec._CLOCK_KEYS if k.endswith("_half_life")} & set(
        _spec._common.__code__.co_varnames
    ) - {"half_life"}
    assert named - {"*_half_life"} == _spec._CLOCK_KEYS - diagnostics - {"hazard"}
