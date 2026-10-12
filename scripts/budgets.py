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
*below* it by more than the tolerance fails too, with the number to tighten
the budget to, so a gain is locked in by the change that made it -- unless
the run is a measuring run. A budget may only be raised with a ``[[raised]]``
entry in the file naming the old and new number, the date, the reason and
the CHANGELOG *Performance* entry it points to; ``raises`` holds every
increase in the file's git history to that.

    python scripts/budgets.py check SUMMARY.json [SECOND.json] [--measure]
    python scripts/budgets.py raises
    python scripts/budgets.py apply MEASURED.json [--release VERSION]
    python scripts/budgets.py release-check MEASURED.json
    python scripts/budgets.py runner-version | toolchain

``check`` reads Gungraun's JSON output (``--output-format=json``), one
summary per case; given a second run of the same binary, it also reports
how far the two agree, and fails when they differ by more than a quarter of
the tolerance, since the tolerance would then be noise. It writes a
Markdown report to stdout and to ``$GITHUB_STEP_SUMMARY`` when set, and the
counts to ``--measured`` for ``apply``. Standard library only: the CI job
that runs it has no project environment.
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
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGETS = REPO / "budgets.toml"
CHANGELOG = REPO / "CHANGELOG.md"
BENCH = REPO / "crates/online-bench/benches/instructions.rs"
KINDS = ("instructions", "memory", "rss")
PLATFORMS = ("linux", "darwin", "windows")
#: How far two runs of one binary may differ, as a share of the tolerance.
REPEAT_SHARE = 0.25
MEASURE_VARIABLE = "POLARS_ONLINE_BUDGETS"
#: The line that opens the file's last section, the raises and sign-offs.
RAISES_MARK = "# --- raises and sign-offs"


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
    ``budget`` and ``release``: ``instructions.<case>``,
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


def bench_cases(text: str | None = None) -> list[str]:
    """The case ids `benches/instructions.rs` names, in its order."""
    text = BENCH.read_text(encoding="utf-8") if text is None else text
    return re.findall(r"^\s*#\[bench::(\w+)\(", text, re.M)


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


def check_file(doc: dict, changelog: str, cases: list[str]) -> list[str]:
    """What is wrong with the file on its own: its shape, the cases against
    the bench, and every ``[[raised]]`` entry against the CHANGELOG."""
    errors = []
    meta = doc.get("meta", {})
    for key in ("toolchain", "runner", "gungraun", "rows", "tolerance", "drift_signoff"):
        if key not in meta:
            errors.append(f"[meta] has no {key}")
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
    for case, table in doc.get("instructions", {}).items():
        if not table.get("what"):
            errors.append(f"instructions.{case} says nothing in `what`")
    for key, table in entries(doc).items():
        for field in ("budget", "release"):
            v = table.get(field)
            if v is not None and not (isinstance(v, int | float) and v > 0 and math.isfinite(v)):
                errors.append(f"{key}.{field} = {v!r}: a positive number")
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
    before, after = budget_numbers(old), budget_numbers(new)
    ledger = new if ledger is None else ledger
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
    shas = _git(repo, "log", "--reverse", "--format=%H", "--", path).split()
    return [(sha, _git(repo, "show", f"{sha}:{path}")) for sha in shas], shallow


def check_history(versions: list[tuple[str, str]], working: str | None) -> list[str]:
    """Every raise from one version of the file to the next, the working
    tree's last, answered by a ``[[raised]]`` entry of the file as it
    stands."""
    texts = [(sha[:10], text) for sha, text in versions]
    if working is not None and (not texts or texts[-1][1] != working):
        texts.append(("working tree", working))
    if not texts:
        return []
    ledger = loads(texts[-1][1])
    errors = []
    for (_, a), (label, b) in zip(texts, texts[1:], strict=False):
        errors += [f"{label}: {e}" for e in unexplained_raises(loads(a), loads(b), ledger)]
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
    release: float | None
    status: str  # ok | over | under | unmeasured
    message: str

    @property
    def failed(self) -> bool:
        return self.status in ("over", "under")

    @property
    def drift(self) -> float | None:
        """The change since the last release's own measurement."""
        return None if not self.release else self.measured / self.release - 1.0


def _fmt(v: float) -> str:
    """A count with its thousands, bytes and MiB to the four places a budget
    is written to."""
    if float(v).is_integer() or abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.4f}".rstrip("0").rstrip(".")


def judge(key: str, measured: float, table: dict | None, tol: float) -> Verdict:
    """One measurement against its budget: past it by more than ``tol``
    fails as a regression; below it by more than ``tol`` fails as a budget
    to tighten, to the measurement itself."""
    table = table or {}
    budget, release = table.get("budget"), table.get("release")
    if budget is None:
        msg = f"{key}: no budget yet; measured {_fmt(measured)}: "
        msg += f"set budget = {_round(key, measured)}"
        return Verdict(key, measured, None, release, "unmeasured", msg)
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
        return Verdict(key, measured, budget, release, "over", msg)
    if ratio < -tol:
        msg = against + (
            f"under its budget of {_fmt(budget)}: tighten the budget to "
            f"{_round(key, measured)}, so the gain is kept."
        )
        return Verdict(key, measured, budget, release, "under", msg)
    return Verdict(key, measured, budget, release, "ok", f"{key}: {ratio:+.2%} of its budget")


def _round(key: str, measured: float) -> int | float:
    """A measurement as its budget is written: a count whole, bytes and MiB
    to four places, rounded up so the budget is never below what it holds."""
    if key.startswith("instructions."):
        return int(measured)
    # `round` first: 198.205 * 10,000 is 1982050.0000000002 in doubles.
    return math.ceil(round(measured * 10_000, 6)) / 10_000


def judge_all(doc: dict, measured: dict[str, float], measure: bool = False) -> list[Verdict]:
    """Every measurement against the file; in a measuring run nothing fails."""
    table = entries(doc)
    out = []
    for key, value in measured.items():
        v = judge(key, value, table.get(key), tolerance(doc, key))
        if measure and v.failed:
            v = Verdict(v.key, v.measured, v.budget, v.release, "ok", f"measuring: {v.message}")
        out.append(v)
    return out


def signoff_failures(doc: dict, verdicts: list[Verdict], version: str | None = None) -> list[str]:
    """The cases that moved past ``meta.drift_signoff`` since the last
    release, either way, with no ``[[signoff]]`` for ``version``."""
    limit = float(doc["meta"]["drift_signoff"])
    signed = {
        s["key"] for s in doc.get("signoff", []) if version is None or s["version"] == version
    }
    return [
        f"{v.key}: {v.drift:+.1%} since {doc['meta'].get('release')} -- past "
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
    doc: dict, verdicts: list[Verdict], second: dict[str, float] | None = None, title: str = ""
) -> str:
    """The Markdown table a run writes to its job summary."""
    meta = doc["meta"]
    lines = [f"### {title or 'Budgets'}", ""]
    release = meta.get("release")
    lines.append(
        f"Tolerance {', '.join(f'{k} {v:.0%}' for k, v in meta['tolerance'].items())}; "
        f"drift since {'release ' + release if release else 'the last release (none recorded)'}"
        f", sign-off past {float(meta['drift_signoff']):.0%}."
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


def apply(text: str, verdicts: list[Verdict], release: str | None = None) -> tuple[str, list[str]]:
    """Write what a measurement settles into the file's text: a budget for
    each case that has none, a tightened budget for each case under its own
    by more than the tolerance, and -- with ``release`` -- every case's
    release count. A budget is never raised here: that takes a reason."""
    done = []
    for v in verdicts:
        if v.status in ("unmeasured", "under"):
            text = _set_number(text, v.key, "budget", _round(v.key, v.measured))
            done.append(f"{v.key}: budget {_round(v.key, v.measured)}")
        if release is not None:
            text = _set_number(text, v.key, "release", _round(v.key, v.measured))
    if release is not None:
        text = re.sub(r'(?m)^release = ".*"$', f'release = "{release}"', text, count=1)
        done.append(f"release counts recorded as {release}")
    return text, done


def read_measured(path: Path) -> dict[str, float]:
    """A measurement file: flat ``{key: value}``, written by ``check
    --measured`` or by the memory tests."""
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


# --- the command line ------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    doc = load()
    first = {f"instructions.{k}": v for k, v in parse_gungraun(args.summary.read_text()).items()}
    second = None
    if args.second is not None:
        second = {
            f"instructions.{k}": v for k, v in parse_gungraun(args.second.read_text()).items()
        }
    measure = args.measure or measuring()
    verdicts = judge_all(doc, first, measure)
    missing = sorted({f"instructions.{c}" for c in doc["instructions"]} - set(first))
    text = report(
        doc,
        verdicts,
        second,
        title="Instruction counts" + (" (measuring run: nothing fails)" if measure else ""),
    )
    problems = [v.message for v in verdicts if v.failed]
    problems += [f"{k}: budgeted, and the run did not measure it" for k in missing]
    repeat = repeat_failures(doc, first, second) if second is not None else []
    drift = signoff_failures(doc, verdicts, args.release_check)
    notes = [v.message for v in verdicts if v.status == "unmeasured"]
    for heading, items in (
        ("Fails", problems),
        ("Two runs of one binary disagree", repeat),
        ("Past the sign-off since the last release", drift),
        ("New cases", notes),
    ):
        if items:
            text += f"\n**{heading}:**\n\n" + "".join(f"- {m}\n" for m in items)
    print(text)
    write_summary(text)
    if args.measured is not None:
        args.measured.write_text(json.dumps(first, indent=1, sort_keys=True) + "\n")
    for m in notes:
        print(f"::warning::{m}")
    failed = list(repeat)
    if not measure:
        failed += problems
    if args.release_check is not None:
        failed += drift
    else:
        for m in drift:
            print(f"::warning::{m}")
    for m in failed:
        print(f"::error::{m}")
    return 1 if failed else 0


def _cmd_raises(_: argparse.Namespace) -> int:
    doc = load()
    errors = check_file(doc, CHANGELOG.read_text(encoding="utf-8"), bench_cases())
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
    text, done = apply(BUDGETS.read_text(encoding="utf-8"), verdicts, args.release)
    loads(text)  # still TOML
    BUDGETS.write_text(text, encoding="utf-8")
    print("\n".join(done) or "nothing to change")
    return 0


def _cmd_release_check(args: argparse.Namespace) -> int:
    doc = load()
    measured: dict[str, float] = {}
    for path in args.measured:
        measured.update(read_measured(path))
    failures = signoff_failures(doc, judge_all(doc, measured, measure=True), args.version)
    print("\n".join(failures) or "every case within the sign-off since the last release")
    return 1 if failures else 0


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
    c.set_defaults(fn=_cmd_check)
    r = sub.add_parser("raises", help="every raise in the file's history has its reason")
    r.set_defaults(fn=_cmd_raises)
    a = sub.add_parser("apply", help="write missing and tightened budgets from measurements")
    a.add_argument("measured", type=Path, nargs="+")
    a.add_argument("--release", metavar="VERSION", help="also record these as VERSION's counts")
    a.set_defaults(fn=_cmd_apply)
    rc = sub.add_parser("release-check", help="the sign-off rule, on measurement files")
    rc.add_argument("measured", type=Path, nargs="+")
    rc.add_argument("--version", required=True)
    rc.set_defaults(fn=_cmd_release_check)
    sub.add_parser("runner-version").set_defaults(fn=lambda _: print(runner_version()) or 0)
    sub.add_parser("toolchain").set_defaults(fn=lambda _: print(load()["meta"]["toolchain"]) or 0)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
