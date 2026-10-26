"""The tests for what a branch changed, and nothing else: the loop inside a task.

    scripts/quick.sh                  # changes since the merge-base with main
    scripts/quick.sh --base HEAD~1    # changes since another commit
    scripts/quick.sh --dry-run        # print the plan, run nothing
    scripts/quick.sh --extended       # the mapped files' extended tests too

Reads the changed paths -- `git diff --name-only` against the base, the
working tree's edits and untracked files included -- and maps each to the
tests that cover it (`plan`). A Rust source change runs that crate's `cargo
test` and rebuilds the extension before pytest, since pytest would otherwise
test the old binary. Python test files run at the essentials tier unless
`--extended` is given.

It never replaces the gate. `scripts/gate.sh` (the essentials) still runs
before each commit and `scripts/gate.sh --extended` before each push: a
change can break a test this map does not reach, and only those find it.

How a path maps, first rule that applies:

| a changed path | runs |
|---|---|
| `tests/test_*.py` | that file |
| `tests/conftest.py`, `tests/tiers.py`, `tests/_site/*` | every test file |
| another `tests/*.py` helper | the test files that import it |
| `crates/<crate>/src/<module>...` | the crate's `cargo test`, a rebuild, the files named after it |
| `crates/<crate>/tests/...` | the crate's `cargo test` |
| `crates/online-cli/...` | also the test files that run the CLI |
| `python/polars_online/<module>.py` | the test files named after the module |
| any path | also the test files that name it, by path or by a distinctive file name |

A model's source (a Rust crate the extension links, `python/polars_online`)
and a test file that registers oracle groups also run the oracle ledger's
hash guard (docs/TESTING.md, "The oracle ledger") for the groups they can
move: those whose id names the module, or one of its `ALIASES`, as a word
(`kmeans.rs` the `kmeans_*` groups, `spec_regimes.py` the `hmm_*`,
`kmeans_*` and `micro_*` ones); a source module that names none, such as
`solve.rs`, every group; a test file its own groups. Seconds, and a moved
hash names the oracle runs the change needs.

A test file is named after a module when its name, underscores dropped,
starts with the module's (`ewcov.rs` runs `test_ew_cov.py` and
`test_ew_cov_bias.py`); `ALIASES` holds the modules whose Python name is
another word.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

#: Helpers every test runs under: a change to one can move any test.
EVERYWHERE = {"tests/conftest.py", "tests/tiers.py"}

#: The crates with Rust tests; `online-py` has none (its test binaries
#: would link libpython), so a change there rebuilds the extension only.
RUST_TESTED = {"online-core", "online-polars", "online-cli", "online-bench"}

#: The crates the extension links, so a source change there rebuilds it.
IN_THE_EXTENSION = {"online-core", "online-polars", "online-py"}

#: A Rust or Python module whose test files carry another word.
ALIASES = {
    "margbins": ["marginal_bins"],
    "marglag": ["marginal_lags"],
    "ewclass": ["ew_class"],
    "gramfit": ["gram"],
    "windows_frame": ["windows"],
    "resid_window": ["window"],
    "spec_audit": ["audit"],
    "spec_changepoint": ["bocpd", "corrchange", "seqtest"],
    "spec_regimes": ["hmm", "kmeans", "micro"],
    "spec_words": ["spec_words"],
    "frame": ["frame", "unnest"],
    "formula": ["formula_targets", "window"],
    "renamed": ["renames", "deprecation"],
    "polars_online": ["api_surface", "kwargs_typing"],
    "init": ["api_surface"],
    "bank": ["bank"],
    "cluster": ["kmeans", "micro"],
    "stats": ["eval"],
}

#: The extension's build in the loop: release's numbers, built in a
#: fraction of its time (Cargo.toml, `[profile.dev-wheel]`).
PROFILE = ["--profile", "dev-wheel"]

#: Past this many test files, pytest runs them in several processes.
PARALLEL_FROM = 8

#: The oracle ledger's hash guard, one test a group (`[<group id>]`).
GUARD = "tests/test_oracle_ledger.py::TestTheHashGuard::test_our_outputs_on_the_cases_inputs_hold"

#: File names too common to say which file a test means.
GENERIC = {"lib.rs", "mod.rs", "main.rs", "tests.rs", "__init__.py", "Cargo.toml", "README.md"}


@dataclass
class Plan:
    """What to run for a set of changed paths."""

    crates: set[str] = field(default_factory=set)
    rebuild: bool = False
    tests: set[str] = field(default_factory=set)
    guards: set[str] = field(default_factory=set)
    unmapped: list[str] = field(default_factory=list)

    def pytest_args(self) -> list[str]:
        """The test files, and the guard's groups unless its file runs whole."""
        guards = [] if GUARD.split("::")[0] in self.tests else sorted(self.guards)
        return sorted(self.tests) + [f"{GUARD}[{g}]" for g in guards]


def _test_files() -> list[str]:
    return sorted(f"tests/{p.name}" for p in TESTS.glob("test_*.py"))


def _text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _flat(name: str) -> str:
    return name.replace("_", "").lower()


def named_after(module: str) -> set[str]:
    """The test files named after `module` (see the module's docstring)."""
    words = [module.lstrip("_"), *ALIASES.get(module.strip("_"), [])]
    found = set()
    for test in _test_files():
        stem = _flat(Path(test).stem.removeprefix("test_"))
        if any(stem.startswith(_flat(w)) for w in words if w):
            found.add(test)
    return found


def importers(helper: str) -> set[str]:
    """The test files that import the helper module `helper`."""
    pattern = re.compile(rf"^\s*(import {helper}\b|from {helper} import)", re.M)
    return {t for t in _test_files() if pattern.search(_text(t))}


def naming(path: str) -> set[str]:
    """The test files that name `path`: its repo path, or its file name when
    that is distinctive."""
    name = Path(path).name
    needles = [path]
    if name not in GENERIC and len(Path(name).stem) >= 4:
        needles.append(name)
    return {t for t in _test_files() if t != path and any(n in _text(t) for n in needles)}


_GUARD_CALL = re.compile(r"""^ledger\.guard\(\s*["']([a-z0-9_]+)["']""", re.M)


def ledger_groups() -> dict[str, str]:
    """Every oracle group's id and the test file that registers it, read
    from the source as `tests/ledger.py`'s `registered_modules` reads it."""
    out = {}
    for test in _test_files():
        short = Path(test).stem.removeprefix("test_")
        for name in _GUARD_CALL.findall(_text(test)):
            out[f"{short}.{name}"] = test
    return out


def guards_for(module: str) -> set[str]:
    """The oracle groups a change to `module` can move: those whose id has
    the module, or one of its aliases, as a word; every group when none
    does."""
    groups = ledger_groups()
    words = {_flat(w) for w in [module.strip("_"), *ALIASES.get(module.strip("_"), [])] if w}
    named = {g for g in groups if words & {_flat(t) for t in re.split(r"[._]", g)}}
    return named or set(groups)


def _module(parts: tuple[str, ...]) -> str | None:
    """The module a source path belongs to: its first part under `src/`,
    or the file's own stem; `None` for a crate's root."""
    if "src" not in parts:
        return None
    rest = parts[parts.index("src") + 1 :]
    if not rest:
        return None
    first = rest[0].removesuffix(".rs")
    return None if first in {"lib", "main", "mod"} else first


def _one(path: str) -> Plan:
    """What covers one changed path."""
    out = Plan()
    parts = Path(path).parts
    if path in EVERYWHERE or path.startswith("tests/_site/"):
        out.tests.update(_test_files())
    elif re.fullmatch(r"tests/test_\w+\.py", path):
        if (REPO / path).exists():
            out.tests.add(path)
            out.guards.update(g for g, t in ledger_groups().items() if t == path)
    elif re.fullmatch(r"tests/\w+\.py", path):
        out.tests.update(importers(Path(path).stem))
    elif parts[:1] == ("crates",) and len(parts) > 2:
        crate = parts[1]
        if crate in RUST_TESTED:
            out.crates.add(crate)
        in_source = "src" in parts or parts[-1] in {"Cargo.toml", "build.rs"}
        if in_source and crate in IN_THE_EXTENSION:
            out.rebuild = True
        if in_source and (module := _module(parts)):
            out.tests.update(named_after(module))
        if in_source and crate in IN_THE_EXTENSION:
            out.guards.update(guards_for(_module(parts) or crate))
        if crate == "online-cli":
            out.tests.update(t for t in _test_files() if "online_cli" in _text(t))
    elif parts[:2] == ("python", "polars_online"):
        stem = Path(path).stem
        module = "polars_online" if stem in {"__init__", "_polars_online"} else stem
        out.tests.update(named_after(module))
        out.guards.update(guards_for(module))
    if path in {"Cargo.toml", "Cargo.lock"}:
        out.crates.update(RUST_TESTED)
        out.rebuild = True
    out.tests.update(naming(path))
    return out


def plan(paths: list[str]) -> Plan:
    """Map changed repo-relative paths to what covers them. A path that
    maps to nothing is listed in `unmapped`, whatever the others map to."""
    out = Plan()
    for path in paths:
        one = _one(path)
        out.crates |= one.crates
        out.rebuild |= one.rebuild
        out.tests |= one.tests
        out.guards |= one.guards
        if not (one.crates or one.rebuild or one.tests):
            out.unmapped.append(path)
    return out


def changed(base: str) -> list[str]:
    """The paths changed since `base`: committed, staged, edited and untracked."""

    def git(*args: str) -> list[str]:
        res = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True
        )
        return [line for line in res.stdout.splitlines() if line]

    return sorted(
        set(git("diff", "--name-only", base))
        | set(git("ls-files", "--others", "--exclude-standard"))
    )


def default_base() -> str:
    res = subprocess.run(
        ["git", "merge-base", "HEAD", "main"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return res.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", help="the commit to diff against; default the merge-base with main")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    ap.add_argument(
        "--extended", action="store_true", help="the extended tests of the mapped files too"
    )
    ap.add_argument("paths", nargs="*", help="paths to map instead of the branch's changes")
    args = ap.parse_args()
    paths = args.paths or changed(args.base or default_base())
    p = plan(paths)
    print(f"quick: {len(paths)} changed paths")
    for crate in sorted(p.crates):
        print(f"quick: cargo test -p {crate}")
    if p.rebuild:
        print("quick: rebuild the extension")
    print(
        f"quick: {len(p.tests)} test files" + (": " + " ".join(sorted(p.tests)) if p.tests else "")
    )
    if p.guards:
        print(
            f"quick: the oracle hash guard for {len(p.guards)} groups: {' '.join(sorted(p.guards))}"
        )
    for path in p.unmapped:
        print(f"quick: no test maps to {path}; the essentials before the commit cover it")
    if args.dry_run:
        return 0

    env = dict(os.environ)
    if not args.extended:
        env.setdefault("HYPOTHESIS_PROFILE", "essential")
        env.setdefault("PROPTEST_CASES", "32")
    steps: list[list[str]] = []
    if p.crates:
        crates = [a for c in sorted(p.crates) for a in ("-p", c)]
        tier = ["--", "--include-ignored"] if args.extended else []
        steps.append(["cargo", "test", *crates, *tier])
    if p.rebuild and (p.tests or p.guards):
        steps.append(
            ["uv", "run", "maturin", "develop", *PROFILE, "-m", "crates/online-py/Cargo.toml"]
        )
    if p.tests or p.guards:
        marks = (
            "not soak and not oracle_grid"
            if args.extended
            else "not extended and not soak and not oracle_grid"
        )
        many = []
        if len(p.tests) > PARALLEL_FROM:
            workers = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "workers.py")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            ).stdout.strip()
            many = ["-n", workers, "--dist", "worksteal"]
        steps.append(["uv", "run", "pytest", "-q", *many, "-m", marks, *p.pytest_args()])
    for step in steps:
        print("quick: $ " + " ".join(step), flush=True)
        if subprocess.run(step, cwd=REPO, env=env, check=False).returncode:
            print("quick: FAIL")
            return 1
    print("quick: PASS (the mapped tests only; the essentials gate runs on the merged wave)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
