"""What a cargo-mutants run left uncaught, less the mutants no test can catch.

    python3 scripts/mutants_report.py mutants.out [more mutants.out ...]
        [--fail-on-missed] [--expect-runs N] [--fail-on-incomplete]

Reads each run's `outcomes.json` (several, for a sharded pass), drops the
missed mutants `scripts/mutants_equivalent.toml` names, sets apart those
`scripts/mutants_tolerated.toml` names, and prints a Markdown report: the
counts, then every remaining survivor by file with the line it mutates, then
the tolerated ones by kind. The report is also appended to
`$GITHUB_STEP_SUMMARY` when that is set. `--fail-on-missed` exits 1 if a
survivor remains, never for a tolerated one; that is for the job on a
change's own lines. The weekly pass reports its survivors and exits 0.

It also says when a run did not finish (task 155). A run that writes
`listed.txt` beside its outcomes, the mutants it was given one per line, is
incomplete when it tested fewer: cargo-mutants keeps the outcomes it
finished when it is stopped. `--expect-runs N` counts a run that sent no
directory at all. `--fail-on-incomplete` exits 1 on either, so a pass cut
short at its time limit cannot read as a smaller clean one.

An entry of either list matches by file, function, the mutation as
cargo-mutants names it, and code the mutated line must contain, and a
tolerated entry by its `columns` where it gives them. A `line_has` with a
newline in it is matched against the mutated line and the lines just before
it, for a line whose twin elsewhere in the function is caught. So an entry follows its
line when code above it moves, and lapses when the line itself changes.

Standard library only, so a CI job runs it with the runner's own Python.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EQUIVALENTS = REPO / "scripts" / "mutants_equivalent.toml"
TOLERATED = REPO / "scripts" / "mutants_tolerated.toml"


def load_equivalents(path: Path = EQUIVALENTS) -> list[dict]:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("equivalent", [])


def load_tolerated(path: Path = TOLERATED) -> list[dict]:
    return tomllib.loads(path.read_text(encoding="utf-8")).get("tolerated", [])


def mutation_of(name: str) -> str:
    """`crates/x.rs:70:27: replace < with <= in F` -> `replace < with <=`."""
    text = name.split(": ", 1)[-1]
    return text.rsplit(" in ", 1)[0]


@functools.cache
def _lines(path: Path) -> tuple[str, ...]:
    """A source file's lines, read once a run: every entry reads them."""
    try:
        return tuple(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return ()


def source_line(root: Path, file: str, line: int, before: int = 0) -> str:
    """Line `line` of the file, with the `before` lines above it, joined."""
    lines = _lines(root / file)
    if not 0 < line <= len(lines):
        return ""
    return "\n".join(lines[max(line - 1 - before, 0) : line])


def matching_entry(mutant: dict, entries: list[dict], root: Path) -> dict | None:
    """The first entry of an equivalent or tolerated list that names the mutant."""
    start = mutant["span"]["start"]
    for e in entries:
        before = e["line_has"].count("\n")
        if (
            e["file"] == mutant["file"]
            and e["function"] == mutant["function"]["function_name"]
            and e["mutation"] == mutation_of(mutant["name"])
            and e["line_has"] in source_line(root, mutant["file"], start["line"], before)
            and start["column"] in e.get("columns", [start["column"]])
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


def listed(run: Path) -> int | None:
    """How many mutants the run was given, if it wrote its list."""
    path = run / "listed.txt"
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def incomplete(runs: list[Path], expect_runs: int | None = None) -> list[str]:
    """Each way the runs fell short of what they were given, in words."""
    present = [run for run in runs if run.is_dir()]
    problems = []
    for run in present:
        given = listed(run)
        tested = len(outcomes([run]))
        if given is not None and tested < given:
            problems.append(f"{run.name}: tested {tested} of the {given} mutants it was given")
    if expect_runs is not None and len(present) < expect_runs:
        problems.append(f"{expect_runs - len(present)} of the {expect_runs} runs reported nothing")
    return problems


def report(
    runs: list[Path],
    root: Path = REPO,
    equivalents: list[dict] | None = None,
    expect_runs: int | None = None,
    tolerated: list[dict] | None = None,
) -> tuple[str, int]:
    """The Markdown report, and how many survivors are neither equivalent nor
    tolerated."""
    equivalents = load_equivalents() if equivalents is None else equivalents
    tolerated = load_tolerated() if tolerated is None else tolerated
    results = outcomes(runs)
    short = incomplete(runs, expect_runs)
    counts = Counter(o["summary"] for o in results)
    survivors: dict[str, list[str]] = defaultdict(list)
    excused: list[str] = []
    kept: dict[str, list[str]] = defaultdict(list)
    for o in results:
        if o["summary"] != "MissedMutant":
            continue
        m = o["scenario"]["Mutant"]
        where = f"`{m['name'].split(': ', 1)[0]}`"
        what = m["name"].split(": ", 1)[-1]
        e = matching_entry(m, equivalents, root)
        t = None if e is not None else matching_entry(m, tolerated, root)
        if e is not None:
            excused.append(f"- {where} {what}: {' '.join(e['why'].split())}")
        elif t is not None:
            kept[t["kind"]].append(f"- {where} {what}: {' '.join(t['why'].split())}")
        else:
            survivors[m["file"]].append(f"- {where} {what}")
    missed = sum(len(v) for v in survivors.values())
    n_kept = sum(len(v) for v in kept.values())
    lines = [
        "## Mutation testing",
        "",
        f"{len(results)} mutants: {counts['CaughtMutant']} caught, {missed} survived, "
        f"{len(excused)} equivalent, {n_kept} tolerated, {counts['Timeout']} timed out, "
        f"{counts['Unviable']} unviable.",
    ]
    if not results and not short:
        lines[-1] = "No mutants: the code examined has nothing cargo-mutants can change."
    if short:
        lines += [
            "",
            "### Incomplete: these counts leave mutants out",
            "",
            *(f"- {s}" for s in short),
        ]
    if survivors:
        lines += ["", "### Survivors: a change here would pass every test", ""]
        for file in sorted(survivors):
            lines += [f"**{file}** ({len(survivors[file])})", *sorted(survivors[file]), ""]
    if kept:
        lines += ["", "### Tolerated, not counted (scripts/mutants_tolerated.toml)", ""]
        for kind in sorted(kept):
            lines += [f"**{kind}** ({len(kept[kind])})", *sorted(kept[kind]), ""]
    if excused:
        lines += ["", "### Equivalent, not counted (scripts/mutants_equivalent.toml)", "", *excused]
    return "\n".join(lines).rstrip() + "\n", missed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="mutants.out directories")
    ap.add_argument("--fail-on-missed", action="store_true", help="exit 1 if a survivor remains")
    ap.add_argument("--expect-runs", type=int, help="how many runs should have reported")
    ap.add_argument(
        "--fail-on-incomplete", action="store_true", help="exit 1 if a run tested fewer than given"
    )
    args = ap.parse_args()
    text, missed = report(args.runs, expect_runs=args.expect_runs)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    if args.fail_on_incomplete and incomplete(args.runs, args.expect_runs):
        return 1
    return 1 if args.fail_on_missed and missed else 0


if __name__ == "__main__":
    sys.exit(main())
