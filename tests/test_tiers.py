"""The two tiers hold together (docs/TESTING.md, "Two tiers"; PLAN task 210).

The essentials gate each commit while a task is in progress; everything runs
before every push, in CI and before a release. These checks keep the split
honest without timing anything (timing is the machine's; the gate prints
it):

- every test module says which tier it is in, and says it truly;
- every ``extended`` mark, and every Rust ``#[ignore]``, gives its reason;
- every hard rule names tests that run in the essentials;
- a plain ``pytest`` is the full run, and every place that runs the full
  tier asks for all of it, the Rust tests the essentials leave out included.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from hypothesis import settings

import tiers

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
TIERS = {"essential", "extended", "mixed", "soak"}
WORKFLOWS = REPO / ".github" / "workflows"
#: What the essentials gate passes pytest; no workflow passes it.
ESSENTIALS_MARKERS = '-m "not extended and not soak"'


def _modules() -> list[Path]:
    return sorted((REPO / "tests").glob("test_*.py"))


def _is_mark(node: ast.AST, name: str) -> bool:
    """``pytest.mark.<name>``, called or not."""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _module_marks(tree: ast.Module) -> list[ast.AST]:
    """The marks of a module's ``pytestmark``, one or a list."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            value = node.value
            return list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return []


def _tier(tree: ast.Module) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TIER" for t in node.targets
        ):
            return node.value.value if isinstance(node.value, ast.Constant) else node.value
    return None


def _uses(tree: ast.Module, name: str) -> list[ast.AST]:
    """Every ``pytest.mark.<name>`` in a module: each call, and each use
    that is not called."""
    nodes = list(ast.walk(tree))
    calls = [n for n in nodes if isinstance(n, ast.Call) and _is_mark(n.func, name)]
    called = {id(c.func) for c in calls}
    bare = [
        n
        for n in nodes
        if isinstance(n, ast.Attribute) and _is_mark(n, name) and id(n) not in called
    ]
    return calls + bare


@pytest.fixture(scope="module")
def trees() -> dict[str, ast.Module]:
    return {p.name: ast.parse(p.read_text(encoding="utf-8")) for p in _modules()}


def test_every_test_module_declares_its_tier(trees):
    """A module-level ``TIER``, so a new module cannot land in no tier: one
    without it would run in the essentials however slow it is."""
    missing = sorted(name for name, tree in trees.items() if _tier(tree) not in TIERS)
    assert not missing, f"no TIER, or not one of {sorted(TIERS)}: {missing}"


def test_each_modules_tier_is_what_its_marks_make_it(trees):
    wrong = {}
    for name, tree in trees.items():
        module = _module_marks(tree)
        if any(_is_mark(m, "soak") for m in module):
            want = "soak"
        elif any(_is_mark(m, "extended") for m in module):
            want = "extended"
        elif _uses(tree, "extended"):
            want = "mixed"
        else:
            want = "essential"
        if _tier(tree) != want:
            wrong[name] = (_tier(tree), want)
    assert not wrong, f"TIER against what the marks make it: {wrong}"


def test_every_extended_mark_says_why(trees):
    """``pytest.mark.extended(reason="...")``, the reason a sentence and
    nothing positional, so the tier can be read off the source."""
    bad = []
    for name, tree in trees.items():
        for use in _uses(tree, "extended"):
            reason = None
            if isinstance(use, ast.Call) and not use.args:
                reason = next((k.value for k in use.keywords if k.arg == "reason"), None)
            ok = isinstance(reason, ast.Constant) and isinstance(reason.value, str)
            if not ok or len(reason.value.split()) < 2:
                bad.append(f"{name}:{use.lineno}")
    assert not bad, f"an extended mark without a reason: {bad}"


def _rust_sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "crates"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [REPO / p for p in out.stdout.split("\0") if p.endswith(".rs")]


IGNORE = re.compile(r"#\[ignore\b[^\]]*\]")
EXTENDED_IGNORE = re.compile(r'#\[ignore = "extended: ([^"]+)"\]')


def test_every_ignored_rust_test_is_extended_and_says_why():
    """``#[ignore = "extended: <reason>"]`` and nothing else: an ignored test
    that is not the extended tier's would run nowhere, since every place
    that runs the full tier passes ``--include-ignored``."""
    seen, bad = 0, []
    for path in _rust_sources():
        text = path.read_text(encoding="utf-8")
        for m in IGNORE.finditer(text):
            seen += 1
            ok = EXTENDED_IGNORE.fullmatch(m.group(0))
            if not ok or len(ok.group(1).split()) < 2:
                line = text.count("\n", 0, m.start()) + 1
                bad.append(f"{path.relative_to(REPO)}:{line}: {m.group(0)}")
    assert not bad, "\n".join(bad)
    assert seen, "the extended tier has Rust tests; the pattern found none"


# --- every hard rule names a test the essentials run -------------------------

#: The hard rules a test can hold (CLAUDE.md): 1, no data files; 2, out of
#: sample; 3, chunk invariance; 5, the frozen fixtures; 8, `n_eff`; 9, a zero
#: weight.
HELD_RULES = {1, 2, 3, 5, 8, 9}


def _hard_rule_table() -> dict[int, list[str]]:
    """The rows of the table under "Two tiers" whose first cell is a rule,
    each with the tests its second cell names in backticks."""
    doc = (REPO / "docs" / "TESTING.md").read_text(encoding="utf-8")
    section = doc.split("\n### Two tiers\n", 1)[1].split("\n### ", 1)[0]
    rows = re.findall(r"^\| (\d+), [^|]+\| (.+) \|$", section, flags=re.M)
    table = {}
    for rule, cell in rows:
        tests = re.findall(r"`((?:tests|crates)/[^`]+::[^`]+)`", cell)
        table.setdefault(int(rule), []).extend(tests)
    return table


def _python_test_is_essential(spec: str) -> str | None:
    """None if ``spec`` names a test the essentials run, else why not."""
    path, *names = spec.split("::")
    file = REPO / path
    if not file.exists():
        return "no such file"
    tree = ast.parse(file.read_text(encoding="utf-8"))
    if _tier(tree) == "extended" or any(_is_mark(m, "extended") for m in _module_marks(tree)):
        return "its module is extended"
    body = tree.body
    for name in names:
        node = next(
            (
                n
                for n in body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name.split("[")[0]
            ),
            None,
        )
        if node is None:
            return f"no {name}"
        if any(_is_mark(d, "extended") for d in node.decorator_list):
            return f"{name} is extended"
        body = node.body if isinstance(node, ast.ClassDef) else []
    return None


def _rust_test_is_essential(spec: str) -> str | None:
    path, name = spec.split("::", 1)
    file = REPO / path
    if not file.exists():
        return "no such file"
    lines = file.read_text(encoding="utf-8").splitlines()
    at = [i for i, ln in enumerate(lines) if re.match(rf"\s*fn {re.escape(name)}\s*[(<]", ln)]
    if not at:
        return f"no fn {name}"
    for i in at:
        j = i - 1
        while j >= 0 and lines[j].strip().startswith(("#[", "//")):
            if lines[j].strip().startswith("#[ignore"):
                return f"fn {name} is ignored"
            j -= 1
    return None


def test_every_hard_rule_names_a_test_the_essentials_run():
    """docs/TESTING.md's table, rule by rule: each names tests that exist and
    that the essentials run, so a commit gated by the essentials alone
    still checked every hard rule a test can hold."""
    table = _hard_rule_table()
    assert set(table) >= HELD_RULES, f"rules with no row: {sorted(HELD_RULES - set(table))}"
    bad = {}
    for rule, specs in table.items():
        assert specs, f"hard rule {rule} names no test"
        for spec in specs:
            check = (
                _python_test_is_essential if spec.startswith("tests/") else _rust_test_is_essential
            )
            why = check(spec)
            if why:
                bad[spec] = why
    assert not bad, f"named for a hard rule, but not an essentials test: {bad}"


# --- a plain run is the full one, and the full tier is asked for in full -----


def test_a_plain_pytest_is_the_full_run(monkeypatch):
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    ini = pyproject["tool"]["pytest"]["ini_options"]
    assert "extended" not in ini["addopts"], ini["addopts"]
    assert any(m.startswith("extended:") for m in ini["markers"])
    monkeypatch.delenv(tiers.PROFILE_VARIABLE, raising=False)
    assert tiers.profile() == "extended"
    assert tiers.examples(150, essential=15) == 150
    monkeypatch.setenv(tiers.PROFILE_VARIABLE, "essential")
    assert tiers.examples(150, essential=15) == 15
    assert tiers.examples(30) == tiers.ESSENTIAL_EXAMPLES
    monkeypatch.setenv(tiers.PROFILE_VARIABLE, "everything")
    with pytest.raises(ValueError, match="essential, extended"):
        tiers.profile()


def test_this_session_runs_the_profile_the_environment_names():
    assert settings.get_current_profile_name() == os.environ.get(tiers.PROFILE_VARIABLE, "extended")


def _gate() -> str:
    return (REPO / "scripts" / "gate.sh").read_text(encoding="utf-8")


def test_the_gate_runs_the_essentials_by_default_and_everything_when_asked():
    gate = _gate()
    assert "cargo_tier=(-- --include-ignored)" in gate
    assert f"pytest_tier=({ESSENTIALS_MARKERS})" in gate
    assert re.search(r"export PROPTEST_CASES=\d+\n", gate)
    assert "export HYPOTHESIS_PROFILE=essential" in gate
    assert "export HYPOTHESIS_PROFILE=extended" in gate
    assert "--extended) tier=extended" in gate
    # The verdict names the tier, and the essentials' says it is not the
    # pre-push check (the user, 2026-10-08).
    assert "not the pre-push check: run scripts/gate.sh --extended before every push" in gate


def _steps(workflow: str):
    wf = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    for job_name, job in wf["jobs"].items():
        for step in job.get("steps", []):
            yield job_name, step


def test_every_workflow_runs_the_full_tier_and_none_the_essentials():
    """CI on every push and pull request, the release and the canary run
    every Rust test (``--include-ignored``) and never leave the Python
    extended tier out. No workflow runs the essentials, not even to time
    them: the gate does that locally, where they run, and CI's minutes go
    to the full tier alone (the user, 2026-10-08: "Do the work before the
    push just to save ci time")."""
    workflows = sorted(WORKFLOWS.glob("*.yml"))
    assert {w.name for w in workflows} >= {"ci.yml", "release.yml", "polars-canary.yml"}
    tier_settings = ("HYPOTHESIS_PROFILE", "PROPTEST_CASES")
    for workflow in workflows:
        wf = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        envs = [wf.get("env") or {}]
        for job in wf["jobs"].values():
            envs.append(job.get("env") or {})
            envs.extend(step.get("env") or {} for step in job.get("steps", []))
        for env in envs:
            assert not set(tier_settings) & set(env), (workflow.name, env)
        for job, step in _steps(workflow.name):
            run = str(step.get("run", ""))
            assert not any(f"{name}=" in run for name in tier_settings), (workflow.name, job)
            if re.search(r"\bcargo test\b", run):
                assert "-- --include-ignored" in run, (workflow.name, job, run)
            if "pytest" in run:
                assert "extended" not in run, (workflow.name, job, run)


def test_mutation_testing_runs_the_extended_rust_tests_too():
    """cargo-mutants runs ``cargo test``, which skips an ignored test: a
    mutant only an extended test catches would count as a survivor."""
    runs = [
        str(step.get("run", ""))
        for _, step in _steps("mutants.yml")
        if "cargo mutants" in str(step.get("run", ""))
    ]
    assert runs
    for run in runs:
        calls = [
            line
            for line in run.replace("\\\n", " ").splitlines()
            if "cargo mutants" in line and "--list" not in line
        ]
        assert calls, run
        assert all(line.rstrip().endswith("-- -- --include-ignored") for line in calls), calls
    script = (REPO / "scripts" / "mutants.sh").read_text(encoding="utf-8")
    assert 'cargo mutants "${args[@]}" -- -- --include-ignored' in script
