"""The spellings the library had before it was renamed `rheon` appear only where
they are meant to: in the history, in the environment fallback that still reads
them, and in the files that test that they are still read.

A rule is a spelling, the trees it is held over, and the files that may keep it.
`git ls-files` and the untracked files that are not ignored are scanned, so a
new file is held to the rule before it is committed.

The rules start with the names the code carries (the warnings, the frame
namespace, the environment variables, the `po.` that prefixed a message). The
reader-facing prose (the README, the guides, `llms.txt`), the command line's
own name and the package's URLs follow in their own commits, and each adds its
rule here as it lands: `DEFERRED` names what is not yet scanned, so that the
list of what remains is in one place.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: Paths no rule scans: the history, the maintainers' coordination files, and
#: what is not text.
HISTORY = (
    "docs/records/*",
    ".claude/*",
    "uv.lock",
    "Cargo.lock",
    "*.parquet",
    "*.arrow",
    "*.so",
    "*.pyc",
)

#: Not yet scanned by any rule: the prose that explains the library to a reader
#: (rewritten in its own commit), and the changelog's past sections.
DEFERRED = (
    "README.md",
    "llms.txt",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "CITATION.cff",
    "SECURITY.md",
    "docs/*.md",
    "docs/how-to/*",
    "docs/tutorials/*",
    "docs/explanation/*",
    "docs/contributing/*",
)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    #: Globs of the files held to the rule (everything else is not scanned).
    trees: tuple[str, ...]
    #: Globs of the files that may keep the spelling, and why.
    allowed: tuple[str, ...] = ()


#: The files that read the old environment variables by their old names, and
#: the tests of that.
ENV_FALLBACK = (
    "python/rheon/_env.py",
    "python/rheon/_warnings.py",  # RheonDeprecationWarning's docstring names them
    "crates/rheon-polars/src/env.rs",
    "crates/rheon-polars/src/pool.rs",  # THREADS_VAR's doc names the old one
    "crates/rheon-polars/src/timing.rs",  # so does TIMING_VAR's
    "crates/rheon-cli/tests/surface.rs",
    "tests/test_env_rename.py",
    "tests/test_api_surface.py",  # reads the pairs out of the two tables
    "tests/api_surface.txt",  # and lists them
    "tests/test_old_names.py",
)

CODE = (
    "python/*",
    "crates/*",
    "tests/*",
    "scripts/*",
    "examples/*",
    ".github/*",
    "docs/reference/*",
    "pyproject.toml",
    "Cargo.toml",
    "budgets.toml",
    "writing_ratchet.toml",
)

RULES = (
    Rule(
        "the warning classes were renamed RheonWarning and RheonDeprecationWarning",
        re.compile(r"PolarsOnline(?:Deprecation)?Warning|__polars_online_renamed__"),
        CODE,
        allowed=("tests/test_old_names.py",),
    ),
    Rule(
        "the frame namespaces are LazyFrameRheonNamespace and DataFrameRheonNamespace",
        re.compile(r"(?:Lazy|Data)FrameOnlineNamespace"),
        CODE,
        allowed=("tests/test_old_names.py",),
    ),
    Rule(
        "the frame namespace is `.rheon`, registered by `rheon.polars`",
        # An accessor: `.online` after a name, a call or a subscript, as
        # `lf.online.fit_predict`, `df.online[...]`, `scan(...).online.`.
        re.compile(r"""(?<=[A-Za-z0-9_)\]"'])\\?\.online(?![A-Za-z0-9_-])"""),
        CODE,
        allowed=(
            "tests/test_old_names.py",
            # Reads docs/FEATURES.md, whose index still names `lf.online`.
            "tests/test_feature_index.py",
        ),
    ),
    Rule(
        "the environment variables are RHEON_*; only the fallback reads the old names",
        re.compile(r"POLARS_ONLINE_"),
        CODE,
        allowed=ENV_FALLBACK,
    ),
    Rule(
        "a message names the path a user calls, `rp.` or `rheon.`, not `po.`",
        # `po.` at the start of a word: not `repo.`, `expo.` or a variable
        # called `po` in a Rust test (`po.link`).
        re.compile(r"(?<![A-Za-z0-9_])po\\?\.[A-Za-z_]"),
        ("python/*", "crates/*", "tests/test_*.py"),
        allowed=(
            "crates/rheon-core/src/sgd.rs",  # a local variable called `po`
            "tests/test_old_names.py",
            # Child interpreters that import the package as `po` (code in a
            # string), and the tests of documents that still name the alias.
            "tests/test_budgets_memory.py",
            "tests/test_eval_query.py",
            "tests/test_ffi_memory.py",
            "tests/test_hardening.py",
            "tests/test_kwargs_typing.py",
            "tests/test_marginal_shards.py",
            "tests/test_portability.py",
            "tests/test_production_hardening.py",
            "tests/test_pyarrow_interop.py",
            "tests/test_solve_pool_size.py",
            "tests/test_threads.py",
            "tests/test_released_state.py",
            "tests/test_api_links.py",
            "tests/test_llms_txt.py",
            "tests/test_calibration_page.py",
            "tests/test_diagnostics_page.py",
            "tests/test_doc_review.py",
            "tests/test_doc_structure.py",
            "tests/test_feature_index.py",
            "tests/test_readiness.py",
            "tests/test_gram_page.py",
            "tests/test_gbdt_page.py",
        ),
    ),
)


def tracked_and_new() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        check=True,
    ).stdout
    return [p for p in out.splitlines() if (REPO / p).is_file()]


def matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch(path, g) for g in globs)


def hits(rule: Rule) -> list[str]:
    found = []
    for path in tracked_and_new():
        if matches(path, HISTORY) or matches(path, DEFERRED):
            continue
        if not matches(path, rule.trees) or matches(path, rule.allowed):
            continue
        try:
            text = (REPO / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rule.pattern.search(line):
                found.append(f"{path}:{n}: {line.strip()[:110]}")
    return found


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name.split(" ")[1])
def test_an_old_spelling_appears_only_where_it_is_allowed(rule):
    found = hits(rule)
    assert not found, f"{rule.name}:\n" + "\n".join(found[:40])


def test_the_scan_sees_what_it_is_meant_to():
    """A scan that finds nothing because it scans nothing proves nothing: each
    rule's pattern matches a spelling it is meant to catch, and each allowed
    file exists, so a renamed file does not leave an exemption behind."""
    samples = (
        "PolarsOnlineWarning",
        "__polars_online_renamed__",
        "LazyFrameOnlineNamespace",
        "lf.online.fit_predict",
        "df\\.online\\.predict",
        "POLARS_ONLINE_TIMING",
        "po.spec.ewridge",
        r"po\.stream\.embargo",
    )
    caught = [s for s in samples if any(r.pattern.search(s) for r in RULES)]
    assert caught == list(samples)
    for s in ("repo.spec", "rheon.polars", "self.online_mode", "x.online-prefetch1"):
        assert not any(r.pattern.search(s) for r in RULES), s
    files = set(tracked_and_new())
    for rule in RULES:
        for allowed in rule.allowed:
            assert allowed in files, f"{allowed} is allowed by a rule and is not a file"
    assert len(files) > 500
