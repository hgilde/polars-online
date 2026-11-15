"""Every citation of the plan resolves, though the plan's finished work moved.

``docs/PLAN.md`` keeps the design, the open tasks, one line per finished task
and every section's heading; the text of the finished tasks and the closed
sections lives in ``docs/records/PLAN-ARCHIVE-N.md``. Citations all over the
repository name a task by number ("task 143") or a section ("PLAN §18 D1"), so
this module holds the two ways the move could orphan one: a cited task with no
line in the plan, and a cited section with no heading. It also holds the links
between the plan and its archives: each pointer line reaches an anchor that
exists, and no archived entry is left without a pointer.

A task cited as ``task N`` resolves to the plan's line for task ``N`` (or for
``Na``, a lettered step of it); a range ``task N-M`` and ``tasks N and M``
resolve each end. Comma lists are not read: a number after a comma is as often
a date or a count as a task.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "docs" / "PLAN.md"
RECORDS = REPO / "docs" / "records"

#: The most an archive file may hold, under the 250,000-byte source cap.
ARCHIVE_CAP = 240_000

TEXT = re.compile(r"\.(rs|py|md|toml|sh|yml|yaml|txt)$")
TASK_LINE = re.compile(r"^- \[[ x]\] (\d+)[a-z]?[. ]", re.M)
POINTER = re.compile(
    r"^- \[x\] (?P<id>\S+(?: follows \S+)?)\. \*\*.+\*\* -- "
    r"\[part \d+\]\((?P<link>records/PLAN-ARCHIVE-\d+\.md)#(?P<anchor>task-[0-9a-z-]+)\)$",
    re.M,
)
CITED_TASK = re.compile(r"\b[Tt]asks? (\d+[a-z]?(?:(?:,? and |\s*[–-]\s*)\d+[a-z]?)*)")
CITED_SECTION = re.compile(r"PLAN(?:\.md`?)? §(\d+[a-z]?)")
HEADING = re.compile(r"^## (\d+[a-z]?)\.", re.M)


def archives() -> list[Path]:
    return sorted(RECORDS.glob("PLAN-ARCHIVE-*.md"))


def cited_files() -> list[tuple[str, str]]:
    """Every tracked text file that may cite the plan, but the plan and its archives."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.splitlines()
    files = []
    for name in tracked:
        if not TEXT.search(name) or name == "docs/PLAN.md" or "PLAN-ARCHIVE-" in name:
            continue
        path = REPO / name
        if path.is_file():
            files.append((name, path.read_text(encoding="utf-8", errors="replace")))
    return files


def test_every_cited_task_has_a_line_in_the_plan():
    tasks = set(TASK_LINE.findall(PLAN.read_text(encoding="utf-8")))
    assert len(tasks) > 300, "the plan lost its task lines"
    missing = {}
    for name, text in cited_files():
        for m in CITED_TASK.finditer(text):
            for n in re.findall(r"\d+", m.group(1)):
                if n not in tasks:
                    missing.setdefault(n, name)
    assert not missing, f"tasks cited but absent from docs/PLAN.md (number: first file): {missing}"


def test_every_cited_plan_section_has_a_heading():
    headings = set(HEADING.findall(PLAN.read_text(encoding="utf-8")))
    assert {"1", "11", "11a", "18", "26"} <= headings
    missing = {}
    for name, text in cited_files():
        for m in CITED_SECTION.finditer(text):
            if m.group(1) not in headings:
                missing.setdefault(m.group(1), name)
    assert not missing, (
        f"sections cited but absent from docs/PLAN.md (section: first file): {missing}"
    )


def test_the_archives_fit_under_the_cap_and_open_as_records():
    parts = archives()
    assert parts, "no plan archive"
    for part in parts:
        assert part.stat().st_size < ARCHIVE_CAP, part.name
        assert part.read_text(encoding="utf-8").startswith("<!-- kind: record -->\n"), part.name


def test_every_pointer_reaches_its_anchor_and_every_archived_task_has_a_pointer():
    plan = PLAN.read_text(encoding="utf-8")
    anchored: dict[str, set[str]] = {}
    for part in archives():
        anchored[f"records/{part.name}"] = set(
            re.findall(r'^<a id="([^"]+)"></a>$', part.read_text(encoding="utf-8"), re.M)
        )
    pointed = set()
    for m in POINTER.finditer(plan):
        assert m["anchor"] in anchored.get(m["link"], ()), f"no anchor behind: {m[0]}"
        pointed.add((m["link"], m["anchor"]))
    archived = {(f, a) for f, anchors in anchored.items() for a in anchors if a.startswith("task-")}
    assert archived == pointed, sorted(archived ^ pointed)[:5]
    # A section pointer's anchor exists too, and each archived section's heading stays in the plan.
    for link, section in re.findall(r"\]\((records/PLAN-ARCHIVE-\d+\.md)#s([0-9a-z]+)\)", plan):
        assert f"s{section}" in anchored[link], (link, section)
    headings = set(HEADING.findall(plan))
    for anchors in anchored.values():
        assert {a[1:] for a in anchors if a.startswith("s")} <= headings
