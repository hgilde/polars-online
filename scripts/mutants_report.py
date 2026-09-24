"""What a cargo-mutants run left uncaught, less the mutants no test can catch.

    python3 scripts/mutants_report.py mutants.out [more mutants.out ...] [--fail-on-missed]

Reads each run's `outcomes.json` (several, for a sharded pass), drops the
missed mutants `scripts/mutants_equivalent.toml` names, and prints a Markdown
report: the counts, then every remaining survivor by file with the line it
mutates. The report is also appended to `$GITHUB_STEP_SUMMARY` when that is
set. `--fail-on-missed` exits 1 if a survivor remains; that is for the job on
a change's own lines. The weekly pass reports and exits 0.

An equivalent matches by file, function, the mutation as cargo-mutants names
it, and code the mutated line must contain. So an entry follows its line when
code above it moves, and lapses when the line itself changes.

Standard library only, so a CI job runs it with the runner's own Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EQUIVALENTS = REPO / "scripts" / "mutants_equivalent.toml"


def load_equivalents(path: Path = EQUIVALENTS) -> list[dict]:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("equivalent", [])


def mutation_of(name: str) -> str:
    """`crates/x.rs:70:27: replace < with <= in F` -> `replace < with <=`."""
    text = name.split(": ", 1)[-1]
    return text.rsplit(" in ", 1)[0]


def source_line(root: Path, file: str, line: int) -> str:
    try:
        lines = (root / file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return lines[line - 1] if 0 < line <= len(lines) else ""


def matching_equivalent(mutant: dict, equivalents: list[dict], root: Path) -> dict | None:
    line = source_line(root, mutant["file"], mutant["span"]["start"]["line"])
    for e in equivalents:
        if (
            e["file"] == mutant["file"]
            and e["function"] == mutant["function"]["function_name"]
            and e["mutation"] == mutation_of(mutant["name"])
            and e["line_has"] in line
        ):
            return e
    return None


def outcomes(runs: list[Path]) -> list[dict]:
    """Every mutant's outcome across the runs; the unmutated baseline is not one."""
    found = []
    for run in runs:
        path = run / "outcomes.json"
        if path.exists():  # a change that touched no mutable code writes none
            data = json.loads(path.read_text(encoding="utf-8"))
            found += [o for o in data["outcomes"] if "Mutant" in o["scenario"]]
    return found


def report(
    runs: list[Path], root: Path = REPO, equivalents: list[dict] | None = None
) -> tuple[str, int]:
    """The Markdown report, and how many survivors are not equivalent."""
    equivalents = load_equivalents() if equivalents is None else equivalents
    results = outcomes(runs)
    counts = Counter(o["summary"] for o in results)
    survivors: dict[str, list[str]] = defaultdict(list)
    excused: list[str] = []
    for o in results:
        if o["summary"] != "MissedMutant":
            continue
        m = o["scenario"]["Mutant"]
        where = f"`{m['name'].split(': ', 1)[0]}`"
        what = m["name"].split(": ", 1)[-1]
        e = matching_equivalent(m, equivalents, root)
        if e is None:
            survivors[m["file"]].append(f"- {where} {what}")
        else:
            excused.append(f"- {where} {what}: {' '.join(e['why'].split())}")
    missed = sum(len(v) for v in survivors.values())
    lines = [
        "## Mutation testing",
        "",
        f"{len(results)} mutants: {counts['CaughtMutant']} caught, {missed} survived, "
        f"{len(excused)} equivalent, {counts['Timeout']} timed out, "
        f"{counts['Unviable']} unviable.",
    ]
    if not results:
        lines[-1] = "No mutants: the code examined has nothing cargo-mutants can change."
    if survivors:
        lines += ["", "### Survivors: a change here would pass every test", ""]
        for file in sorted(survivors):
            lines += [f"**{file}** ({len(survivors[file])})", *sorted(survivors[file]), ""]
    if excused:
        lines += ["", "### Equivalent, not counted (scripts/mutants_equivalent.toml)", "", *excused]
    return "\n".join(lines).rstrip() + "\n", missed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="mutants.out directories")
    ap.add_argument("--fail-on-missed", action="store_true", help="exit 1 if a survivor remains")
    args = ap.parse_args()
    text, missed = report(args.runs)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 1 if args.fail_on_missed and missed else 0


if __name__ == "__main__":
    sys.exit(main())
