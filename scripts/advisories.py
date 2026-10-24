"""Published advisories against the two lock files, from OSV.

Every registry package in ``Cargo.lock`` (crates.io) and ``uv.lock`` (PyPI)
goes to OSV's batch API (``https://api.osv.dev/v1/querybatch``), and each
advisory that comes back is sorted three ways:

- **failing**: a vulnerability against a crate in a shipped graph -- the
  wheel's (``online-py``) or the command line's (``online-cli``), its normal
  and build edges on every target, as ``cargo tree`` lists them;
- **notices**: an informational advisory against a shipped crate (RustSec's
  ``unmaintained``, ``unsound``, ``notice``: what ``cargo audit`` warns of
  and does not fail on), and one in ``ACCEPTED``, with its reason;
- **reported**: the rest -- a dev-dependency's, the bench crate's, the
  Python dev and docs groups' -- which reach neither file that ships.

It prints all three, to the job summary too when ``GITHUB_STEP_SUMMARY`` is
set, and exits 1 when anything fails. OSV that cannot be reached is an
error, not a pass. Standard library only, so the weekly job needs nothing
installed but cargo (task 270, decision D3)::

    python3 scripts/advisories.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OSV = "https://api.osv.dev/v1"
#: OSV's batch takes at most 1,000 queries.
BATCH = 1000
#: The crates whose graphs ship: the wheel's extension and the command line.
SHIPPED_ROOTS = ("online-py", "online-cli")

#: Advisories against a shipped crate that are known, cannot be fixed here,
#: and are reported as notices rather than failing the job: id -> reason.
#: An entry lapses by itself when the crate moves past the advisory.
ACCEPTED: dict[str, str] = {
    "RUSTSEC-2026-0194": (
        "quick-xml 0.39 (quadratic time on hostile XML): fixed in 0.41, a semver-major "
        "object_store 0.13 must move to first. It parses an object store's listing, reached "
        "only through Polars' cloud reader, which the command line refuses (it reads local "
        "files) and the wheel's Polars is never handed a URL."
    ),
    "RUSTSEC-2026-0195": (
        "quick-xml 0.39 (unbounded allocation on hostile XML): as RUSTSEC-2026-0194."
    ),
}


@dataclass(frozen=True)
class Hit:
    """One advisory against one locked package."""

    ecosystem: str
    name: str
    version: str
    id: str
    summary: str = ""


@dataclass
class Report:
    failing: list[Hit] = field(default_factory=list)
    notices: list[tuple[Hit, str]] = field(default_factory=list)
    reported: list[Hit] = field(default_factory=list)


def cargo_lock_packages(text: str) -> list[tuple[str, str]]:
    """The crates.io packages of a ``Cargo.lock``: the workspace's own
    path crates have no ``source`` and are left out."""
    lock = tomllib.loads(text)
    return sorted(
        {
            (p["name"], p["version"])
            for p in lock.get("package", [])
            if str(p.get("source", "")).startswith("registry+")
        }
    )


def uv_lock_packages(text: str) -> list[tuple[str, str]]:
    """The registry packages of a ``uv.lock``: the project itself, editable
    and path sources are left out."""
    lock = tomllib.loads(text)
    return sorted(
        {
            (p["name"], p["version"])
            for p in lock.get("package", [])
            if "registry" in (p.get("source") or {}) and "version" in p
        }
    )


def parse_cargo_tree(text: str) -> set[tuple[str, str]]:
    """``(name, version)`` from ``cargo tree --prefix none --format {p}``
    lines: ``tokio v1.54.1``, ``paste v1.0.15 (proc-macro)``,
    ``rustls v0.23.46 (*)``."""
    out = set()
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 2 and words[1].startswith("v"):
            out.add((words[0], words[1][1:]))
    return out


def shipped_crates(roots: tuple[str, ...] = SHIPPED_ROOTS) -> set[tuple[str, str]]:
    """Every crate the shipped files link or build with, on any target."""
    crates: set[tuple[str, str]] = set()
    for root in roots:
        tree = subprocess.run(
            [
                "cargo",
                "tree",
                "--locked",
                "--edges",
                "normal,build",
                "--target",
                "all",
                "--package",
                root,
                "--prefix",
                "none",
                "--format",
                "{p}",
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout
        found = parse_cargo_tree(tree)
        if not found:
            raise SystemExit(f"cargo tree listed nothing for {root}")
        crates |= found
    return crates


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def query_osv(packages: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    """``(ecosystem, name, version, advisory id)`` for each advisory OSV
    lists against each of ``packages``, ``(ecosystem, name, version)``."""
    hits: list[tuple[str, str, str, str]] = []
    for start in range(0, len(packages), BATCH):
        chunk = packages[start : start + BATCH]
        queries = [
            {"package": {"ecosystem": eco, "name": name}, "version": version}
            for eco, name, version in chunk
        ]
        results = _post(f"{OSV}/querybatch", {"queries": queries})["results"]
        if len(results) != len(chunk):
            raise SystemExit(f"OSV answered {len(results)} of {len(chunk)} queries")
        for (eco, name, version), result in zip(chunk, results, strict=True):
            if result.get("next_page_token"):
                raise SystemExit(f"OSV paged the advisories of {eco} {name} {version}")
            hits.extend((eco, name, version, v["id"]) for v in result.get("vulns", []))
    return hits


def is_informational(vuln: dict) -> bool:
    """A RustSec informational advisory (``unmaintained``, ``unsound``,
    ``notice``), which names no vulnerability."""
    return any(
        (a.get("database_specific") or {}).get("informational") for a in vuln.get("affected", [])
    )


def classify(
    hits: list[tuple[str, str, str, str]],
    details: dict[str, dict],
    shipped: set[tuple[str, str]],
    accepted: dict[str, str] = ACCEPTED,
) -> Report:
    """Sort ``hits`` (``query_osv``'s) by ``details`` (each advisory's OSV
    record by id) against ``shipped`` (``shipped_crates``'s). A withdrawn
    advisory is dropped."""
    report = Report()
    for eco, name, version, vid in sorted(set(hits)):
        vuln = details.get(vid, {})
        if vuln.get("withdrawn"):
            continue
        hit = Hit(eco, name, version, vid, vuln.get("summary") or vuln.get("details", "")[:120])
        if eco != "crates.io" or (name, version) not in shipped:
            report.reported.append(hit)
        elif vid in accepted:
            report.notices.append((hit, f"accepted: {accepted[vid]}"))
        elif is_informational(vuln):
            informational = next(
                (a.get("database_specific") or {}).get("informational")
                for a in vuln.get("affected", [])
                if (a.get("database_specific") or {}).get("informational")
            )
            report.notices.append((hit, f"informational: {informational}"))
        else:
            report.failing.append(hit)
    return report


def render(report: Report, counts: dict[str, int]) -> str:
    """The report as Markdown, for the log and the job summary."""

    def line(h: Hit) -> str:
        return f"| {h.ecosystem} | {h.name} | {h.version} | {h.id} | {h.summary} |"

    head = "| ecosystem | package | version | advisory | summary |\n|---|---|---|---|---|"
    queried = ", ".join(f"{n} {k}" for k, n in counts.items())
    parts = [f"# Advisories\n\nQueried OSV for {queried} packages."]
    parts.append(
        f"\n## Failing: in a shipped graph ({len(report.failing)})\n\n"
        + ("\n".join([head, *map(line, report.failing)]) if report.failing else "None.")
    )
    parts.append(
        f"\n## Notices: shipped, informational or accepted ({len(report.notices)})\n\n"
        + (
            "\n".join(f"- {h.name} {h.version} {h.id}: {why}" for h, why in report.notices)
            if report.notices
            else "None."
        )
    )
    parts.append(
        f"\n## Reported: outside the shipped graphs ({len(report.reported)})\n\n"
        + ("\n".join([head, *map(line, report.reported)]) if report.reported else "None.")
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    crates = cargo_lock_packages((REPO / "Cargo.lock").read_text(encoding="utf-8"))
    pypi = uv_lock_packages((REPO / "uv.lock").read_text(encoding="utf-8"))
    packages = [("crates.io", n, v) for n, v in crates] + [("PyPI", n, v) for n, v in pypi]
    shipped = shipped_crates()
    hits = query_osv(packages)
    details = {vid: _get(f"{OSV}/vulns/{vid}") for vid in sorted({h[3] for h in hits})}
    report = classify(hits, details, shipped)
    text = render(report, {"crates.io": len(crates), "PyPI": len(pypi)})
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if report.failing else 0


if __name__ == "__main__":
    sys.exit(main())
