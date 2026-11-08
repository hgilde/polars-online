"""`docs/FEATURES.md` lists every feature, so a feature cannot ship unlisted.

The user's goal (2026-11-06): "one place to discover the full list of
features". The index is that place, and the README's map links it first.
It replaced the rule that every guide has a README section with a recipe
(PLAN task 338, stage 8), which could not hold once the README became a
landing page over some seventy how-tos.

The public names come from the contract the API snapshot already holds,
``tests/api_surface.txt`` (regenerated from the code by
``tests/test_api_surface.py``), and from the live registries beside it: the
spec builders of ``po.spec.__all__`` and the native model kinds. A name the
snapshot gains, a builder ``po.spec`` exports, a parameter every spec
takes, a flag or a run-config key of the command line, or an environment
variable the library reads fails here until the index has a line for it.
Each name must appear in backticks, in a row's first cell (the cell that
names its features), in the form a reader types it:
``po.<name>``, ``po.<module>.<name>``, ``ModelBank.<name>``,
``lf.online.<name>``, a spec parameter or a run-config key as itself in its
own section, and a flag as ``--flag``.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars_online as po
from polars_online import _polars_online as _native

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "docs" / "FEATURES.md"
SNAPSHOT = REPO / "tests" / "api_surface.txt"


def _sections(text: str) -> dict[str, list[str]]:
    """The snapshot's ``[section]`` blocks, each as its lines."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        head = re.match(r"^\[([^\]]+)\]", line)
        if head:
            current = head.group(1)
            out[current] = []
        elif current is not None and line.strip():
            out[current].append(line)
    return out


def _name(line: str) -> str:
    """A snapshot line's name: what precedes its signature, `=` or comment."""
    return re.split(r"[(\s=#]", line.strip(), maxsplit=1)[0]


def _section_of(text: str, heading: str) -> str:
    """The index's text under one ``##`` heading."""
    part = text.split(f"\n## {heading}\n", 1)
    assert len(part) == 2, f"docs/FEATURES.md has no section {heading!r}"
    return part[1].split("\n## ", 1)[0]


def _wanted(snapshot: str) -> list[tuple[str, str]]:
    """Every public name, as (the section of the index that must hold it, or
    "" for anywhere, the backticked form it must take there)."""
    sec = _sections(snapshot)
    want: list[tuple[str, str]] = []
    modules = {"corr", "eval", "gbdt", "gram", "ops", "sim", "spec", "stream"}
    for line in sec["package"]:
        name = _name(line)
        # A module is listed by its functions; `po.<module>.` stands for it.
        want.append(("", f"po.{name}." if name in modules else f"po.{name}"))
    for line in sec["ModelBank"]:
        name = _name(line)
        if not name.startswith("__"):
            want.append(("", f"ModelBank.{name}"))
    for line in sec["frame namespaces"]:
        want.append(("", f"lf.online.{_name(line).rsplit('.', 1)[1]}"))
    module = None
    for line in sec["helper modules"]:
        if not line.startswith("    "):
            module = line.strip().split(":", 1)[0]
            continue
        want.append(("", f"po.{module}.{_name(line)}"))
    for name in po.spec.__all__:
        want.append(("", f"po.spec.{name}"))
    for kind in _native.model_kinds():
        want.append(("Models", f"po.spec.{kind}"))
    for line in sec["common parameters"]:
        want.append(("Parameters every spec takes", _name(line)))
    flags = [
        flag
        for line in sec["cli flags"]
        for flag in re.findall(r"--[a-z][a-z-]*", line)
        if not line.strip().startswith("Usage:")
    ]
    want += [("The command line", f) for f in flags]
    run_config = sec["toml keys"]
    start = run_config.index("  run config:")
    for line in run_config[start + 1 :]:
        if not line.startswith("    "):
            break
        want.append(("The command line", line.strip()))
    for line in sec["env vars"]:
        if "renamed" not in line:  # an old name, read only to say it was renamed
            want.append(("Versions and the environment", line.strip().split()[0]))
    return want


def _name_cells(index: str) -> dict[str, str]:
    """Each ``##`` section's name cells, joined: a row's first cell, where
    its features are named, or under *Parameters every spec takes* its second,
    after the family. A name mentioned only in prose, or in another row's
    description, has no line of its own, so it is not counted."""
    out: dict[str, str] = {}
    for heading in re.findall(r"^## (.+)$", index, re.M):
        cells = []
        for line in _section_of(index, heading).splitlines():
            if not line.startswith("| ") or line.startswith("|---"):
                continue
            row = [c.strip() for c in line.strip().strip("|").split(" | ")]
            cells.append(row[1] if heading == "Parameters every spec takes" else row[0])
        out[heading] = "\n".join(cells[1:])  # the header row is no feature
    return out


def _missing(index: str, snapshot: str) -> list[str]:
    cells = _name_cells(index)
    everywhere = "\n".join(cells.values())
    missing = []
    for section, name in _wanted(snapshot):
        where = cells.get(section, "") if section else everywhere
        # `po.<module>.` must open a backticked name; any other name is one.
        pattern = rf"`{re.escape(name)}[\w.]*`" if name.endswith(".") else rf"`{re.escape(name)}`"
        if not re.search(pattern, where):
            missing.append(f"{name}" + (f" (under ## {section})" if section else ""))
    return missing


def test_every_public_name_has_a_line_in_the_index():
    index = INDEX.read_text(encoding="utf-8")
    snapshot = SNAPSHOT.read_text(encoding="utf-8")
    assert len(_wanted(snapshot)) > 250, "the snapshot's sections were not found"
    missing = _missing(index, snapshot)
    assert not missing, (
        "docs/FEATURES.md has no line for these public names; add one, with a "
        "link to the page that shows each:\n" + "\n".join(missing)
    )


def test_a_name_removed_from_the_index_is_found():
    """The activation: the index with one model's line, one method, one
    parameter and one flag taken out is caught for each."""
    index = INDEX.read_text(encoding="utf-8")
    snapshot = SNAPSHOT.read_text(encoding="utf-8")
    cut = "\n".join(
        line for line in index.splitlines() if not line.startswith("| [`po.spec.holt`]")
    )
    cut = cut.replace("[`ModelBank.last_row`]", "[last row]")
    cut = cut.replace("`gap_cap` |", "|").replace("`--skip-learned`, ", "")
    missing = _missing(cut, snapshot)
    for name in ("po.spec.holt", "ModelBank.last_row", "gap_cap", "--skip-learned"):
        assert any(m.startswith(name) for m in missing), (name, missing)
    # A row cut whose name another row's description still mentions, and
    # one whose name the section's prose mentions, are caught too.
    cut = "\n".join(
        line
        for line in index.splitlines()
        if not line.startswith(("| [`ModelBank.gram`]", "| [`lf.online.with_windows`]"))
    )
    missing = _missing(cut, snapshot)
    for name in ("ModelBank.gram", "lf.online.with_windows"):
        assert name in missing, (name, missing)


def test_every_line_of_the_index_links_a_page():
    """A line with no link leaves the reader with a name and nowhere to go."""
    index = INDEX.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", index, re.M)
    assert len(headings) >= 12, headings
    for heading in headings:
        rows = [
            line
            for line in _section_of(index, heading).splitlines()
            if line.startswith("| ") and not line.startswith("|---")
        ][1:]
        assert rows, heading
        unlinked = [r for r in rows if "](" not in r]
        assert not unlinked, (heading, unlinked)


def test_the_readme_map_links_the_index_first():
    """The README's map is where a casual reader looks, so its first link
    is the index."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    contents = readme.split("\n## ", 1)[0]
    assert "](docs/FEATURES.md)" in contents
    where = readme.split("\n## Where to go next\n", 1)[1]
    first = re.search(r"\]\(([^)]+)\)", where)
    assert first and first.group(1) == "docs/FEATURES.md", first
