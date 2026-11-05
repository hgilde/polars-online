"""Performance and memory budgets that only tighten: `budgets.toml`.

Every budget is a fixed number a measurement is held to, never the previous
run, so small regressions add up against one number rather than each
passing against the last. Three kinds:

- ``instructions.<case>``: the instructions one block of rows of a hot path
  executes under Callgrind (``crates/online-bench``), from CI's Linux run;
- ``memory.<spec>.<field>``: the bytes per row of one output field of a
  standard workload, from ``estimated_size()`` (``tests/test_budgets_memory.py``);
- ``rss.<case>.<platform>``: the peak resident memory a standard run adds
  over its process's baseline, in MiB (the same file, one per platform).

A measurement past its budget by more than the kind's tolerance fails. One
*below* it by more than the tolerance fails too -- by more than a quarter of
it for resident memory, whose wide tolerance would otherwise let a budget
sit far above what it holds -- with the number to tighten the budget to, so
a gain is locked in by the change that made it, unless the run is a
measuring run. A budget may only be raised with a ``[[raised]]``
entry in the file naming the old and new number, the date, the reason and
the CHANGELOG *Performance* entry it points to; ``raises`` holds every
increase in the file's git history to that.

    python scripts/budgets.py check SUMMARY.json [SECOND.json] [--measure]
        [--baseline SOURCE] [--require-baseline] [--release-check VERSION]
    python scripts/budgets.py raises
    python scripts/budgets.py apply MEASURED.json...
    python scripts/budgets.py release-check MEASURED.json... --version VERSION
    python scripts/budgets.py baseline MEASURED.json... --version V --sha SHA --out FILE
    python scripts/budgets.py runner-version | toolchain

The bench also counts each cell of the scaling table at two sizes of one
dimension (the ``scale_*`` cases). Those have no budget: ``check`` holds the
ratio of a cell's two counts *within the run* to the growth ``[scaling]``
declares for it, ``count(large) / count(small) <= g(large) / g(small)`` times
``1 + scaling.tolerance``, so a cost that grows the wrong way fails on its
first run, with no number from an earlier one.

``check`` reads Gungraun's JSON output (``--output-format=json``), one
summary per case; given a second run of the same binary, it also reports
how far the two agree, and fails when they differ by more than a quarter of
the tolerance, since the tolerance would then be noise. It writes a
Markdown report to stdout and to ``$GITHUB_STEP_SUMMARY`` when set, and the
counts to ``--measured`` for ``apply``. Standard library only: the CI job
that runs it has no project environment.

**The drift since the last release is read from that release, not from
this file.** A release's workflow merges the run's own measurements -- the
instruction counts and each platform's memory -- into one file,
``budgets-baseline.json`` (``baseline``), and attaches it to the GitHub
release as an asset. Every later run reads it from the newest release
(``BASELINE_URL``; the repository is public, so no token is needed) and
reports each case's drift since; a release fails on a case past
``meta.drift_signoff`` with no ``[[signoff]]`` for its version. So nothing
is committed after a release to record its numbers. ``--baseline`` (or
``POLARS_ONLINE_BUDGETS_BASELINE``) reads a local file, a directory holding
one, or another URL instead. A run that cannot read it says so and goes on,
unless it is told ``--require-baseline``, as CI and a release are.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGETS = REPO / "budgets.toml"
CHANGELOG = REPO / "CHANGELOG.md"
BENCH = REPO / "crates/online-bench/benches/instructions.rs"
#: The bench cases of the scaling table: ``scale_<surface>_by_<dimension>_small``
#: and ``..._large``, a cell's two sizes.
SCALE_PREFIX = "scale_"
SCALE_SIZES = ("small", "large")
KINDS = ("instructions", "memory", "rss")
PLATFORMS = ("linux", "darwin", "windows")
#: How far two runs of one binary may differ, as a share of the tolerance.
REPEAT_SHARE = 0.25
#: How far under its budget a run may read before the budget is tightened,
#: as a share of the tolerance, where it is not the whole of it. A budget
#: tightened only past the whole tolerance ``t`` settles anywhere up to
#: ``m/(1-t)`` of the run that set it, and a later run then passes up to
#: ``m(1+t)/(1-t)``: +86% at the resident memory's 30%. Past a quarter of
#: it, the ceiling is ``m(1+t)/(1-t/4)``, +41%.
TIGHTEN_SHARE = {"rss": 0.25}
MEASURE_VARIABLE = "POLARS_ONLINE_BUDGETS"
#: The line that opens the file's last section, the raises and sign-offs.
RAISES_MARK = "# --- raises and sign-offs"
#: The release asset that holds a release's own measurements, the baseline
#: every later run's drift is read from.
BASELINE_ASSET = "budgets-baseline.json"
#: The newest release's copy: GitHub redirects ``latest`` to the newest
#: release that is not a draft or a prerelease.
BASELINE_URL = f"https://github.com/hgilde/polars-online/releases/latest/download/{BASELINE_ASSET}"
BASELINE_VARIABLE = "POLARS_ONLINE_BUDGETS_BASELINE"
#: The platforms whose resident memory a baseline must hold.
BASELINE_PLATFORMS = PLATFORMS


def load(path: Path | None = None) -> dict:
    return tomllib.loads((path or BUDGETS).read_text(encoding="utf-8"))


def loads(text: str) -> dict:
    return tomllib.loads(text)


def measuring() -> bool:
    """Whether the environment asks for a measuring run: report, never fail."""
    return os.environ.get(MEASURE_VARIABLE, "") == "measure"


# --- the file ----------------------------------------------------------------


def entries(doc: dict) -> dict[str, dict]:
    """Every budgeted quantity by its flat key, each a table that may hold
    ``budget``: ``instructions.<case>``,
    ``memory.<spec>.<field>`` and ``rss.<case>.<platform>``."""
    out: dict[str, dict] = {}
    for case, table in doc.get("instructions", {}).items():
        out[f"instructions.{case}"] = table
    for spec, fields in doc.get("memory", {}).items():
        for field, table in fields.items():
            out[f"memory.{spec}.{field}"] = table
    for case, table in doc.get("rss", {}).items():
        for platform in PLATFORMS:
            if platform in table:
                out[f"rss.{case}.{platform}"] = table[platform]
    return out


def tolerances(doc: dict) -> dict[str, float]:
    return {f"tolerance.{k}": float(v) for k, v in doc["meta"]["tolerance"].items()}


def tolerance(doc: dict, key: str) -> float:
    return float(doc["meta"]["tolerance"][key.split(".", 1)[0]])


def tighten_at(key: str, tol: float) -> float:
    """How far under its budget ``key`` may read before it is tightened."""
    return tol * TIGHTEN_SHARE.get(key.split(".", 1)[0], 1.0)


def bench_cases(text: str | None = None) -> list[str]:
    """The budgeted case ids `benches/instructions.rs` names, in its order:
    every case but the scaling table's."""
    return [c for c in _bench_ids(text) if not c.startswith(SCALE_PREFIX)]


def _bench_ids(text: str | None = None) -> list[str]:
    text = BENCH.read_text(encoding="utf-8") if text is None else text
    return re.findall(r"^\s*#\[bench::(\w+)\(", text, re.M)


def scale_case(cell: str, size: str) -> str:
    """The bench id of one size of a scaling cell: ``bank.groups_seen``'s
    larger size is ``scale_bank_by_groups_seen_large``."""
    return f"{SCALE_PREFIX}{cell.replace('.', '_by_', 1)}_{size}"


def scale_cells(text: str | None = None) -> list[str]:
    """The scaling cells the bench counts at both sizes, as ``<surface>.<dim>``."""
    ids = set(_bench_ids(text))
    cells = []
    for c in sorted(ids):
        if c.startswith(SCALE_PREFIX) and c.endswith("_small"):
            cell = c[len(SCALE_PREFIX) : -len("_small")].replace("_by_", ".", 1)
            if scale_case(cell, "large") in ids:
                cells.append(cell)
    return cells


def growth_bound(growth: str, small: float, large: float) -> float:
    """How much a cost of declared ``growth`` may grow from size ``small``
    to ``large``: ``"1"`` not at all, ``"log n"`` as the logarithm, ``"n"``
    and ``"n^e"`` as that power."""
    if growth == "1":
        return 1.0
    if growth == "log n":
        return math.log(large) / math.log(small)
    m = re.fullmatch(r"n(?:\^(\d+))?", growth)
    if m is None:
        raise ValueError(f"growth {growth!r}: one of 1, log n, n, n^e")
    return (large / small) ** int(m.group(1) or 1)


@dataclass
class Scaled:
    cell: str
    small: float
    large: float
    bound: float
    tol: float

    @property
    def ratio(self) -> float:
        return self.large / self.small

    @property
    def failed(self) -> bool:
        return self.ratio > self.bound * (1.0 + self.tol)

    @property
    def message(self) -> str:
        allowed = self.bound * (1.0 + self.tol)
        verdict = "past" if self.failed else "within"
        return (
            f"scaling.{self.cell}: {self.ratio:.3f}x from the small size to the large, "
            f"{verdict} the {allowed:.3f}x its declared growth allows"
        )


def judge_scaling(doc: dict, counts: dict[str, float]) -> tuple[list[Scaled], list[str]]:
    """Each cell of ``[scaling.cells]`` held to its declared growth by its
    two counts in ``counts`` (the bench ids); and the cells a run missed."""
    table = doc.get("scaling", {})
    tol = float(table.get("tolerance", 0.0))
    out, missing = [], []
    for cell, t in sorted(table.get("cells", {}).items()):
        s, lg = (counts.get(scale_case(cell, size)) for size in SCALE_SIZES)
        if s is None or lg is None:
            missing.append(f"scaling.{cell}: the run did not count both sizes")
            continue
        out.append(Scaled(cell, s, lg, growth_bound(t["growth"], t["small"], t["large"]), tol))
    return out, missing


def scaling_report(scaled: list[Scaled]) -> str:
    lines = [
        "### Scaling",
        "",
        "Each cell's instructions at its larger size over its smaller, "
        "against its declared growth.",
        "",
        "| cell | small | large | ratio | allowed | |",
        "|---|---|---|---|---|---|",
    ]
    for v in scaled:
        mark = "**past**" if v.failed else ""
        lines.append(
            f"| {v.cell} | {_fmt(v.small)} | {_fmt(v.large)} | {v.ratio:.3f}x "
            f"| {v.bound * (1 + v.tol):.3f}x | {mark} |"
        )
    return "\n".join(lines) + "\n"


def performance_text(changelog: str) -> str:
    """Every *Performance* section of the CHANGELOG, whitespace folded, the
    text a raise's ``changelog`` must quote."""
    out, inside = [], False
    for line in changelog.splitlines():
        if line.startswith("## "):
            inside = False
        elif line.startswith("### "):
            inside = line.strip() == "### Performance"
        elif inside:
            out.append(line)
    return " ".join(" ".join(out).split())


def check_file(
    doc: dict, changelog: str, cases: list[str], scaled: list[str] | None = None
) -> list[str]:
    """What is wrong with the file on its own: its shape, the cases against
    the bench, the scaling cells against the bench's (``scaled``, when
    given), and every ``[[raised]]`` entry against the CHANGELOG."""
    errors = []
    meta = doc.get("meta", {})
    for key in ("toolchain", "runner", "gungraun", "rows", "tolerance", "drift_signoff"):
        if key not in meta:
            errors.append(f"[meta] has no {key}")
    if "release" in meta:
        errors.append(
            "[meta] names a release: the baseline is read from the newest release's "
            f"{BASELINE_ASSET} asset, not written here"
        )
    tol = meta.get("tolerance", {})
    if set(tol) != set(KINDS):
        errors.append(f"[meta.tolerance] names {sorted(tol)}, not {list(KINDS)}")
    for kind, value in tol.items():
        if not (isinstance(value, float) and 0.0 < value < 0.5):
            errors.append(f"tolerance.{kind} = {value!r}: a share above 0 and below 0.5")
    listed = set(doc.get("instructions", {}))
    if listed != set(cases):
        only_file, only_bench = sorted(listed - set(cases)), sorted(set(cases) - listed)
        errors.append(
            f"[instructions] and the bench disagree: in the file only {only_file}, "
            f"in the bench only {only_bench}"
        )
    errors += check_scaling_file(doc, scaled)
    for case, table in doc.get("instructions", {}).items():
        if not table.get("what"):
            errors.append(f"instructions.{case} says nothing in `what`")
    for key, table in entries(doc).items():
        v = table.get("budget")
        if v is not None and not (isinstance(v, int | float) and v > 0 and math.isfinite(v)):
            errors.append(f"{key}.budget = {v!r}: a positive number")
        if "release" in table:
            errors.append(
                f"{key} has a `release` number: the baseline is read from the newest "
                f"release's {BASELINE_ASSET} asset, not written here"
            )
        if key.startswith("instructions.") and not isinstance(table.get("budget", 0), int):
            errors.append(f"{key}.budget is a count of instructions, a whole number")
    known = set(entries(doc)) | set(tolerances(doc))
    quoted = performance_text(changelog).casefold()
    for i, r in enumerate(doc.get("raised", [])):
        where = f"[[raised]] #{i + 1} ({r.get('key', '?')})"
        missing = {"key", "from", "to", "date", "reason", "changelog"} - set(r)
        if missing:
            errors.append(f"{where} has no {sorted(missing)}")
            continue
        if r["key"] not in known:
            errors.append(f"{where}: no such budget")
        if not r["to"] > r["from"]:
            errors.append(f"{where}: `to` {r['to']} is not above `from` {r['from']}")
        if not isinstance(r["date"], dt.date):
            errors.append(f"{where}: `date` is not a TOML date")
        if len(str(r["reason"]).split()) < 4:
            errors.append(f"{where}: the reason says too little: {r['reason']!r}")
        if " ".join(str(r["changelog"]).split()).casefold() not in quoted:
            errors.append(
                f"{where}: the CHANGELOG has no *Performance* entry containing {r['changelog']!r}"
            )
    for i, s in enumerate(doc.get("signoff", [])):
        missing = {"key", "version", "reason"} - set(s)
        if missing:
            errors.append(f"[[signoff]] #{i + 1} has no {sorted(missing)}")
        elif s["key"] not in entries(doc):
            errors.append(f"[[signoff]] #{i + 1} ({s['key']}): no such budget")
    return errors


def check_scaling_file(doc: dict, cells: list[str] | None) -> list[str]:
    """``[scaling]`` on its own, and against the cells the bench counts."""
    errors = []
    table = doc.get("scaling")
    if table is None:
        return [] if not cells else ["no [scaling] table, and the bench counts scaling cells"]
    tol = table.get("tolerance")
    if not (isinstance(tol, float) and 0.0 <= tol < 0.5):
        errors.append(f"scaling.tolerance = {tol!r}: a share from 0 to below 0.5")
    for cell, t in table.get("cells", {}).items():
        where = f"scaling.cells.{cell}"
        if set(t) != {"growth", "small", "large", "what"}:
            errors.append(f"{where} holds {sorted(t)}, not growth, small, large and what")
            continue
        try:
            growth_bound(t["growth"], 2, 4)
        except ValueError as e:
            errors.append(f"{where}: {e}")
        if not (isinstance(t["small"], int) and isinstance(t["large"], int)):
            errors.append(f"{where}: the sizes are whole numbers")
        elif not 1 <= t["small"] < t["large"] or (t["growth"] == "log n" and t["small"] < 2):
            errors.append(
                f"{where}: sizes {t['small']} and {t['large']}, not 1 <= small < large "
                "(2 <= small under log n)"
            )
    if cells is not None and set(cells) != set(table.get("cells", {})):
        listed = set(table.get("cells", {}))
        only_file, only_bench = sorted(listed - set(cells)), sorted(set(cells) - listed)
        errors.append(
            f"[scaling.cells] and the bench disagree: in the file only {only_file}, "
            f"in the bench only {only_bench}"
        )
    return errors


def budget_numbers(doc: dict) -> dict[str, float]:
    """Every number a raise must answer for: each budget, and each tolerance."""
    out = {k: t["budget"] for k, t in entries(doc).items() if "budget" in t}
    out.update(tolerances(doc))
    return out


def unexplained_raises(old: dict, new: dict, ledger: dict | None = None) -> list[str]:
    """The numbers that went up from ``old`` to ``new`` with no ``[[raised]]``
    entry naming that key, that old number and that new one in ``ledger``,
    the file as it stands (``new`` when not given): entries accumulate, so
    a raise is answered by an entry in any later version, and deleting an
    entry brings its raise back."""
    ledger = new if ledger is None else ledger
    return _unexplained(budget_numbers(old), budget_numbers(new), ledger)


def _unexplained(before: dict[str, float], after: dict[str, float], ledger: dict) -> list[str]:
    raised = {(r.get("key"), r.get("from"), r.get("to")) for r in ledger.get("raised", [])}
    return [
        f"{key}: {before[key]} -> {after[key]} with no [[raised]] entry "
        f"(key = {key!r}, from = {before[key]}, to = {after[key]})"
        for key in sorted(set(before) & set(after))
        if after[key] > before[key] and (key, before[key], after[key]) not in raised
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def history(path: str = "budgets.toml", repo: Path = REPO) -> tuple[list[tuple[str, str]], bool]:
    """Every committed version of the file, oldest first, as (sha, text),
    and whether the clone is shallow -- a shallow clone holds only the
    commits it fetched, so its history may start part way."""
    shallow = _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true"
    # Main's own versions, in its order: a branch's commits, which a merge
    # brings in together, are read as the merge's one version, so a branch
    # cut before main tightened a budget cannot read as raising it again
    # (as `scripts/oracles.py` reads its ledger).
    shas = _git(repo, "log", "--first-parent", "--reverse", "--format=%H", "--", path).split()
    return [(sha, _git(repo, "show", f"{sha}:{path}")) for sha in shas], shallow


def check_history(versions: list[tuple[str, str]], working: str | None) -> list[str]:
    """Every raise in the file's versions, the working tree's last, answered
    by a ``[[raised]]`` entry of the file as it stands. Each number is held
    to the last one the history gave its key, not only to the version
    before: a budget deleted in one version and written back higher in a
    later one is a raise like any other."""
    texts = [(sha[:10], text) for sha, text in versions]
    if working is not None and (not texts or texts[-1][1] != working):
        texts.append(("working tree", working))
    if not texts:
        return []
    ledger = loads(texts[-1][1])
    errors: list[str] = []
    last: dict[str, float] = {}
    for i, (label, text) in enumerate(texts):
        numbers = budget_numbers(loads(text))
        if i:
            errors += [f"{label}: {e}" for e in _unexplained(last, numbers, ledger)]
        last.update(numbers)
    return errors


# --- measurements --------------------------------------------------------------


def parse_gungraun(text: str) -> dict[str, int]:
    """The Callgrind instruction count (``Ir``) of each case in Gungraun's
    JSON output: one summary object per case, as lines or concatenated."""
    out: dict[str, int] = {}
    decoder, i = json.JSONDecoder(), 0
    while True:
        while i < len(text) and text[i] in " \t\r\n,[]":
            i += 1
        if i >= len(text):
            break
        obj, i = decoder.raw_decode(text, i)
        case = obj.get("id") or obj["function_name"]
        counts = [
            p["data"]["total"]["metrics"]["Ir"]["values"]["new"]
            for p in obj.get("profiles", [])
            if p.get("tool") == "Callgrind"
        ]
        if len(counts) != 1:
            raise ValueError(f"{case}: {len(counts)} Callgrind profiles in its summary")
        if case in out:
            raise ValueError(f"{case}: two summaries")
        out[case] = int(counts[0])
    return out


@dataclass
class Verdict:
    key: str
    measured: float
    budget: float | None
    baseline: float | None
    status: str  # ok | over | under | unmeasured
    message: str

    @property
    def failed(self) -> bool:
        return self.status in ("over", "under")

    @property
    def drift(self) -> float | None:
        """The change since the last release's own measurement."""
        return None if not self.baseline else self.measured / self.baseline - 1.0


def _fmt(v: float) -> str:
    """A count with its thousands, bytes and MiB to the four places a budget
    is written to."""
    if float(v).is_integer() or abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def judge(
    key: str,
    measured: float,
    table: dict | None,
    tol: float,
    tighten: float | None = None,
    baseline: float | None = None,
) -> Verdict:
    """One measurement against its budget: past it by more than ``tol``
    fails as a regression; below it by more than ``tighten`` (``tol`` when
    not given) fails as a budget to tighten, to the measurement itself.
    ``baseline`` is the last release's measurement, for the drift."""
    tighten = tol if tighten is None else tighten
    table = table or {}
    budget = table.get("budget")
    if budget is None:
        msg = f"{key}: no budget yet; measured {_fmt(measured)}: "
        msg += f"set budget = {_round(key, measured)}"
        return Verdict(key, measured, None, baseline, "unmeasured", msg)
    # To the twelfth place: a measurement exactly the tolerance away is
    # within it, whatever the last bit of the division says.
    ratio = round(measured / budget - 1.0, 12)
    against = f"{key}: {_fmt(measured)} is {ratio:+.2%} "
    if ratio > tol:
        msg = against + (
            f"over its budget of {_fmt(budget)} (tolerance {tol:.0%}). Find what made it "
            "slower or larger; if that is the price of a change worth having, raise the budget "
            "with a [[raised]] entry naming the reason and a CHANGELOG *Performance* entry."
        )
        return Verdict(key, measured, budget, baseline, "over", msg)
    if ratio < -tighten:
        msg = against + (
            f"under its budget of {_fmt(budget)}: tighten the budget to "
            f"{_round(key, measured)}, so the gain is kept."
        )
        return Verdict(key, measured, budget, baseline, "under", msg)
    return Verdict(key, measured, budget, baseline, "ok", f"{key}: {ratio:+.2%} of its budget")


def _round(key: str, measured: float) -> int | float:
    """A measurement as its budget is written: a count whole, bytes and MiB
    to four places, rounded up so the budget is never below what it holds."""
    if key.startswith("instructions."):
        return int(measured)
    # `round` first: 198.205 * 10,000 is 1982050.0000000002 in doubles.
    return math.ceil(round(measured * 10_000, 6)) / 10_000


def judge_all(
    doc: dict,
    measured: dict[str, float],
    measure: bool = False,
    baseline: Baseline | None = None,
) -> list[Verdict]:
    """Every measurement against the file, and against ``baseline``'s for
    the drift; in a measuring run nothing fails."""
    table = entries(doc)
    then = baseline.measured if baseline is not None else {}
    out = []
    for key, value in measured.items():
        tol = tolerance(doc, key)
        v = judge(key, value, table.get(key), tol, tighten_at(key, tol), then.get(key))
        if measure and v.failed:
            v = Verdict(v.key, v.measured, v.budget, v.baseline, "ok", f"measuring: {v.message}")
        out.append(v)
    return out


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text))


def signoff_failures(
    doc: dict, verdicts: list[Verdict], version: str | None = None, released: str = ""
) -> list[str]:
    """The cases that moved past ``meta.drift_signoff`` since the last
    release, ``released`` (the baseline's version), either way, with no
    ``[[signoff]]`` for ``version`` -- or, on a push (no ``version``), none
    for a release above ``released``: a past release's sign-off speaks for
    nothing to come."""
    limit = float(doc["meta"]["drift_signoff"])
    since = released
    released_key = _version(released)
    signed = {
        s["key"]
        for s in doc.get("signoff", [])
        if (
            s["version"] == version
            if version is not None
            else _version(s["version"]) > released_key
        )
    }
    return [
        f"{v.key}: {v.drift:+.1%} since {since} -- past "
        f"{limit:.0%}, a release needs a [[signoff]] entry for it"
        for v in verdicts
        if v.drift is not None and abs(v.drift) > limit and v.key not in signed
    ]


def repeat_failures(doc: dict, a: dict[str, float], b: dict[str, float]) -> list[str]:
    """Where two runs of one binary differ by more than a quarter of the
    tolerance: the tolerance cannot be trusted there."""
    out = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            out.append(f"{key}: in one run only")
            continue
        spread = abs(a[key] - b[key]) / min(a[key], b[key])
        if spread > REPEAT_SHARE * tolerance(doc, key):
            out.append(f"{key}: two runs of one binary differ by {spread:.3%}")
    return out


def report(
    doc: dict,
    verdicts: list[Verdict],
    second: dict[str, float] | None = None,
    title: str = "",
    baseline: Baseline | str | None = None,
) -> str:
    """The Markdown table a run writes to its job summary. ``baseline`` is
    the last release's measurements, or why they could not be read."""
    meta = doc["meta"]
    lines = [f"### {title or 'Budgets'}", ""]
    if isinstance(baseline, Baseline):
        since = f"drift since release {baseline.version} (its {BASELINE_ASSET})"
    elif baseline:
        since = f"no drift: {baseline}"
    else:
        since = "no drift: no baseline read"
    tighter = ", ".join(
        f"{k} past {float(meta['tolerance'][k]) * share:.1%} under"
        for k, share in TIGHTEN_SHARE.items()
        if k in meta["tolerance"]
    )
    lines.append(
        f"Tolerance {', '.join(f'{k} {v:.0%}' for k, v in meta['tolerance'].items())}"
        f"{f' (tightened {tighter})' if tighter else ''}; "
        f"{since}, sign-off past {float(meta['drift_signoff']):.0%}."
    )
    lines.append("")
    columns = ["case", "budget", "measured", "vs budget", "since release"]
    columns += ["second run", ""] if second is not None else [""]
    head = "| " + " | ".join(columns) + " |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for v in sorted(verdicts, key=lambda v: v.key):
        vs = "" if v.budget is None else f"{v.measured / v.budget - 1:+.2%}"
        drift = "" if v.drift is None else f"{v.drift:+.2%}"
        mark = {"ok": "", "over": "**over**", "under": "**tighten**", "unmeasured": "new"}[v.status]
        row = f"| {v.key} | {'' if v.budget is None else _fmt(v.budget)} | {_fmt(v.measured)} "
        row += f"| {vs} | {drift} |"
        if second is not None:
            other = second.get(v.key)
            row += f" {'' if other is None else _fmt(other)} |"
        lines.append(f"{row} {mark} |")
    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


# --- editing the file ------------------------------------------------------------


def _set_number(text: str, key: str, field: str, value: float) -> str:
    """``text`` with ``field`` of ``key``'s entry set to ``value``, written
    where the entry is: a ``[instructions.<case>]`` table, a field's inline
    table under ``[memory.<spec>]``, or a platform's under ``[rss.<case>]``."""
    kind, rest = key.split(".", 1)
    if kind == "instructions":
        header, line_key = f"[instructions.{rest}]", None
    elif kind == "memory":
        spec, line_key = rest.split(".", 1)
        header = f"[memory.{spec}]"
    else:
        case, line_key = rest.rsplit(".", 1)
        header = f"[rss.{case}]"
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration:
        # A new table goes before the raises, which close the file.
        at = next((i for i, ln in enumerate(lines) if ln.startswith(RAISES_MARK)), len(lines))
        lines[at:at] = [header, ""]
        start = at
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith(("[", "# ---"))),
        len(lines),
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    shown = repr(value) if isinstance(value, float) else str(value)
    if line_key is None:
        for i in range(start + 1, end):
            if re.match(rf"{field}\s*=", lines[i]):
                lines[i] = f"{field} = {shown}"
                break
        else:
            lines.insert(end, f"{field} = {shown}")
    else:
        # A field's name may hold a dot (`abs_resid_q0.1_y`), which a bare
        # TOML key may not.
        written = line_key if re.fullmatch(r"[A-Za-z0-9_-]+", line_key) else f'"{line_key}"'
        pattern = re.compile(rf"^{re.escape(written)}\s*=\s*\{{(.*)\}}\s*$")
        for i in range(start + 1, end):
            m = pattern.match(lines[i])
            if m:
                parts = dict(p.split("=", 1) for p in m.group(1).split(",") if "=" in p)
                parts = {k.strip(): v.strip() for k, v in parts.items()}
                parts[field] = shown
                body = ", ".join(f"{k} = {v}" for k, v in parts.items())
                lines[i] = f"{written} = {{ {body} }}"
                break
        else:
            lines.insert(end, f"{written} = {{ {field} = {shown} }}")
    return "\n".join(lines) + "\n"


def apply(text: str, verdicts: list[Verdict]) -> tuple[str, list[str]]:
    """Write what a measurement settles into the file's text: a budget for
    each case that has none, and a tightened budget for each case under its
    own by more than the tolerance. A budget is never raised here: that
    takes a reason. A release's own numbers are not written here at all:
    they are its ``budgets-baseline.json`` asset."""
    done = []
    for v in verdicts:
        if v.status in ("unmeasured", "under"):
            text = _set_number(text, v.key, "budget", _round(v.key, v.measured))
            done.append(f"{v.key}: budget {_round(v.key, v.measured)}")
    return text, done


def read_measured(path: Path) -> dict[str, float]:
    """A measurement file: flat ``{key: value}``, written by ``check
    --measured`` or by the memory tests."""
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


# --- the baseline: the last release's own measurements -----------------------------


@dataclass(frozen=True)
class Baseline:
    """A release's own measurements, every kind, as its asset holds them."""

    version: str
    sha: str
    measured: dict[str, float]


class NoBaseline(RuntimeError):
    """The baseline could not be read: no network, no such asset yet, no
    such file. A file that *is* read and is malformed is a ``ValueError``."""


def parse_baseline(text: str, where: str = "the baseline") -> Baseline:
    """A baseline file's text, checked: a release version, the sha it was
    measured at, and a positive finite number for each key."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where} is not JSON: {e}") from e
    if not isinstance(doc, dict) or set(doc) != {"version", "sha", "measured"}:
        raise ValueError(f"{where} holds {sorted(doc) if isinstance(doc, dict) else doc!r}")
    version, sha, measured = doc["version"], doc["sha"], doc["measured"]
    if not (isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version)):
        raise ValueError(f"{where}: version {version!r} is not a release's")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        raise ValueError(f"{where}: sha {sha!r} is not a commit's")
    if not isinstance(measured, dict) or not measured:
        raise ValueError(f"{where} measures nothing")
    out = {}
    for key, v in measured.items():
        if key.split(".", 1)[0] not in KINDS:
            raise ValueError(f"{where}: {key!r} is no kind of budget")
        if (
            isinstance(v, bool)
            or not isinstance(v, int | float)
            or not (math.isfinite(v) and v > 0)
        ):
            raise ValueError(f"{where}: {key} = {v!r}, not a positive number")
        out[key] = float(v)
    return Baseline(version, sha, out)


def _fetch(url: str, attempts: int = 3, timeout: float = 30.0) -> str:
    """``url``'s body, retried on what a retry could fix; ``NoBaseline``
    once every attempt has failed, or at once on an answer such as a 404."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise NoBaseline(f"{url} answered {e.code}") from e
            last: Exception = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt + 1 < attempts:
            time.sleep(2.0 * (attempt + 1))
    raise NoBaseline(f"{url} could not be read: {last}")


def read_baseline(source: str | Path | None = None) -> Baseline:
    """The last release's measurements, from ``source``: a URL, a file, or a
    directory holding ``budgets-baseline.json``. Unset, from
    ``POLARS_ONLINE_BUDGETS_BASELINE``, else from the newest GitHub release
    (``BASELINE_URL``)."""
    source = str(source or os.environ.get(BASELINE_VARIABLE) or BASELINE_URL)
    if source.startswith(("https://", "http://")):
        return parse_baseline(_fetch(source), source)
    path = Path(source)
    if path.is_dir():
        path = path / BASELINE_ASSET
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise NoBaseline(f"{path} could not be read: {e}") from e
    return parse_baseline(text, str(path))


def baseline_or_reason(source: str | Path | None = None) -> Baseline | str:
    """The baseline, or why it could not be read, for a report that goes on
    without it."""
    try:
        return read_baseline(source)
    except NoBaseline as e:
        return f"could not read the last release's baseline ({e})"


def make_baseline(
    version: str, sha: str, measured: list[dict[str, float]], doc: dict | None = None
) -> dict:
    """One release's measurements merged into the asset's form, refused if
    two files disagree on a key, or if any budgeted quantity of the file --
    every case, every memory field, the resident memory on each platform --
    is missing: a hole in the baseline would hide its case's drift for a
    whole release."""
    doc = load() if doc is None else doc
    merged: dict[str, float] = {}
    for part in measured:
        for key, v in part.items():
            if key in merged and merged[key] != v:
                raise ValueError(f"{key}: two measurements disagree, {merged[key]} and {v}")
            merged[key] = v
    missing = sorted(k for k, t in entries(doc).items() if "budget" in t and k not in merged)
    if missing:
        raise ValueError(f"the release's measurements miss {len(missing)} budgets: {missing}")
    out = {"version": version, "sha": sha, "measured": dict(sorted(merged.items()))}
    parse_baseline(json.dumps(out), "the merged baseline")  # its own reader takes it
    return out


# --- the command line ------------------------------------------------------------


def _split(counts: dict[str, int]) -> tuple[dict, dict]:
    """A run's counts: the budgeted cases by key, and the scaling cases by
    bench id."""
    budgeted = {f"instructions.{k}": v for k, v in counts.items() if not k.startswith(SCALE_PREFIX)}
    scaled = {k: v for k, v in counts.items() if k.startswith(SCALE_PREFIX)}
    return budgeted, scaled


def _cmd_check(args: argparse.Namespace) -> int:
    doc = load()
    first, first_scaled = _split(parse_gungraun(args.summary.read_text(encoding="utf-8")))
    second = second_scaled = None
    if args.second is not None:
        second, second_scaled = _split(parse_gungraun(args.second.read_text(encoding="utf-8")))
    measure = args.measure or measuring()
    found = baseline_or_reason(args.baseline)
    baseline = found if isinstance(found, Baseline) else None
    verdicts = judge_all(doc, first, measure, baseline)
    missing = sorted({f"instructions.{c}" for c in doc["instructions"]} - set(first))
    text = report(
        doc,
        verdicts,
        second,
        title="Instruction counts" + (" (measuring run: nothing fails)" if measure else ""),
        baseline=found,
    )
    problems = [v.message for v in verdicts if v.failed]
    problems += [f"{k}: budgeted, and the run did not measure it" for k in missing]
    repeat = repeat_failures(doc, first, second) if second is not None else []
    if second_scaled is not None:
        repeat += repeat_failures(
            doc,
            *(
                {f"instructions.{k}": v for k, v in run.items()}
                for run in (first_scaled, second_scaled)
            ),
        )
    scaled, scale_missing = judge_scaling(doc, first_scaled)
    if scaled:
        text += "\n" + scaling_report(scaled)
    scale_failed = [v.message for v in scaled if v.failed] + scale_missing
    released = baseline.version if baseline is not None else ""
    drift = signoff_failures(doc, verdicts, args.release_check, released)
    unread = [] if baseline is not None else [str(found)]
    notes = [v.message for v in verdicts if v.status == "unmeasured"]
    for heading, items in (
        ("Fails", problems),
        ("Two runs of one binary disagree", repeat),
        ("Grown past the declared growth", scale_failed),
        ("Past the sign-off since the last release", drift),
        ("No baseline", unread),
        ("New cases", notes),
    ):
        if items:
            text += f"\n**{heading}:**\n\n" + "".join(f"- {m}\n" for m in items)
    print(text)
    write_summary(text)
    if args.measured is not None:
        args.measured.write_text(
            json.dumps(first, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    for m in notes:
        print(f"::warning::{m}")
    # A ratio within one run needs no budget, so a measuring run holds it
    # too: a wrong growth is a defect, not a number to re-measure.
    failed = list(repeat) + scale_failed
    if not measure:
        failed += problems
    if args.release_check is not None:
        failed += drift
    else:
        for m in drift:
            print(f"::warning::{m}")
    # A release cannot check its drift without the baseline, and CI must
    # not let a lost asset quietly turn the drift report off.
    if args.require_baseline or args.release_check is not None:
        failed += unread
    else:
        for m in unread:
            print(f"::warning::{m}")
    for m in failed:
        print(f"::error::{m}")
    return 1 if failed else 0


def _cmd_raises(_: argparse.Namespace) -> int:
    doc = load()
    errors = check_file(doc, CHANGELOG.read_text(encoding="utf-8"), bench_cases(), scale_cells())
    versions, shallow = history()
    errors += check_history(versions, BUDGETS.read_text(encoding="utf-8"))
    print(f"budgets.toml: {len(versions)} committed versions read{' (shallow clone)' * shallow}")
    for e in errors:
        print(f"::error::{e}")
    return 1 if errors else 0


def _cmd_apply(args: argparse.Namespace) -> int:
    doc = load()
    measured: dict[str, float] = {}
    for path in args.measured:
        measured.update(read_measured(path))
    verdicts = judge_all(doc, measured)
    text, done = apply(BUDGETS.read_text(encoding="utf-8"), verdicts)
    loads(text)  # still TOML
    BUDGETS.write_text(text, encoding="utf-8")
    print("\n".join(done) or "nothing to change")
    return 0


def _cmd_release_check(args: argparse.Namespace) -> int:
    doc = load()
    measured: dict[str, float] = {}
    for path in args.measured:
        measured.update(read_measured(path))
    try:
        baseline = read_baseline(args.baseline)
    except NoBaseline as e:
        print(f"::error::could not read the last release's baseline ({e})")
        return 1
    verdicts = judge_all(doc, measured, measure=True, baseline=baseline)
    failures = signoff_failures(doc, verdicts, args.version, baseline.version)
    print(
        "\n".join(failures)
        or f"every case within the sign-off since the last release, {baseline.version}"
    )
    return 1 if failures else 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    parts = [read_measured(p) for p in args.measured]
    out = make_baseline(args.version, args.sha, parts)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    kinds = {k: sum(key.startswith(f"{k}.") for key in out["measured"]) for k in KINDS}
    print(
        f"{args.out}: {args.version} at {args.sha[:10]}, "
        + ", ".join(f"{n} {k}" for k, n in kinds.items())
    )
    return 0


def runner_version() -> str:
    """The Gungraun the lock resolves, which its runner binary must match."""
    lock = tomllib.loads((REPO / "Cargo.lock").read_text(encoding="utf-8"))
    (version,) = {p["version"] for p in lock["package"] if p["name"] == "gungraun"}
    return version


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="hold Gungraun's counts to the budgets")
    c.add_argument("summary", type=Path)
    c.add_argument("second", type=Path, nargs="?", help="a second run of the same binary")
    c.add_argument("--measure", action="store_true", help="a measuring run: report, never fail")
    c.add_argument("--measured", type=Path, help="write the counts here, for `apply`")
    c.add_argument(
        "--release-check",
        metavar="VERSION",
        help="fail on a case past the sign-off since the last release with no signoff for VERSION",
    )
    c.add_argument(
        "--baseline",
        metavar="SOURCE",
        help=f"the last release's measurements: a URL, a file, or a directory holding "
        f"{BASELINE_ASSET} (default: ${BASELINE_VARIABLE}, else the newest release's asset)",
    )
    c.add_argument(
        "--require-baseline",
        action="store_true",
        help="fail when the baseline cannot be read (a release check always does)",
    )
    c.set_defaults(fn=_cmd_check)
    r = sub.add_parser("raises", help="every raise in the file's history has its reason")
    r.set_defaults(fn=_cmd_raises)
    a = sub.add_parser("apply", help="write missing and tightened budgets from measurements")
    a.add_argument("measured", type=Path, nargs="+")
    a.set_defaults(fn=_cmd_apply)
    rc = sub.add_parser("release-check", help="the sign-off rule, on measurement files")
    rc.add_argument("measured", type=Path, nargs="+")
    rc.add_argument("--version", required=True)
    rc.add_argument("--baseline", metavar="SOURCE", help="as `check --baseline`")
    rc.set_defaults(fn=_cmd_release_check)
    b = sub.add_parser("baseline", help="merge a release's measurements into its baseline asset")
    b.add_argument("measured", type=Path, nargs="+")
    b.add_argument("--version", required=True)
    b.add_argument("--sha", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(fn=_cmd_baseline)
    sub.add_parser("runner-version").set_defaults(fn=lambda _: print(runner_version()) or 0)
    sub.add_parser("toolchain").set_defaults(fn=lambda _: print(load()["meta"]["toolchain"]) or 0)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
