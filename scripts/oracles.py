"""The oracle ledger that only tightens: ``oracles.toml``.

The heavy grids against a third-party oracle (``pytest -m oracle_grid``) run
on CI, not in the gate. The ledger holds each case's error against its
oracle -- the worst seed's value and the 95th percentile's, in the test's
own units -- with the oracle library's version, and every run is held to
those fixed numbers, never to the previous run, so small drifts add up
against one number instead of each passing against the last
(``budgets.toml``'s rule, for accuracy instead of speed).

A case is judged on its gap from its ideal, ``|value - ideal|`` (0 for an
error, 1 for a score such as an ARI). A gap past the ledger's by more than
the tolerance fails; it may only be raised with a ``[[raised]]`` entry
naming the old and new number, the date, the reason, and a ``changelog``
text the CHANGELOG contains, which ``raises`` holds every increase in the
file's git history to. A gap below the ledger's by more than the tolerance
fails too, naming the number to tighten to, so the change that made the
gain keeps it (``apply`` writes it). An oracle library whose version moved
fails until ``apply`` records it. A case with no numbers yet is reported
and fails nothing. A count (``ledger.record(..., exact=True)``) carries a
``tolerance`` of 0: one more is past the ledger, one fewer asks to tighten.

The numbers are one platform's, ``[meta] platform``: CI's, where ``check``
judges every push. A run on another platform (``rerun`` on a developer's
Mac) is held to ``tolerance_elsewhere`` instead, since the two platforms'
arithmetic differs in the last bits, and writes none of its numbers: a
gain it sees is reported, and a case new to the ledger is written without
numbers, for CI's ``measured.json`` to fill. A version of the file that
moves the ledger to another platform may move each number by up to the
tolerance with a ``[[rebased]]`` entry; further needs a ``[[raised]]`` one.

    python scripts/oracles.py check RECORDS... [--expect-shards N] [--measured OUT.json]
    python scripts/oracles.py apply RECORDS...
    python scripts/oracles.py raises
    uv run python scripts/oracles.py rerun [GROUP...]
    uv run python scripts/oracles.py fetch [--rounds N]

``RECORDS`` are directories of the JSON lines ``tests/ledger.py`` writes
under ``POLARS_ONLINE_ORACLES_OUT``, or a merged file ``check --measured``
wrote. ``rerun`` is the local command the gate's hash guard names: it runs
the oracle grids of the groups whose hashes or oracle versions moved (or
those named), fails if a test fails or a case is past the ledger, and
otherwise applies the run and writes those groups' hashes to
``oracle_hashes.toml`` -- so a hash moves only with a passing run of its
oracles. ``fetch`` downloads every dataset the ledger's groups name, once,
and fails naming any it could not get: CI runs it before the shards, so no
shard skips a grid as offline. Everything but ``rerun`` and ``fetch`` uses
the standard library alone: the CI job that checks has no project
environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "oracles.toml"
HASHES = REPO / "oracle_hashes.toml"
CHANGELOG = REPO / "CHANGELOG.md"
#: The line that opens the ledger's last section, the raises.
RAISES_MARK = "# --- raises"
#: The guard's own test, which writes the hashes it computes here.
GUARD_TEST = "tests/test_oracle_ledger.py::TestTheHashGuard"
HASHES_OUT = "POLARS_ONLINE_ORACLE_HASHES_OUT"
OUT_VARIABLE = "POLARS_ONLINE_ORACLES_OUT"


def load(path: Path | None = None) -> dict:
    return tomllib.loads((path or LEDGER).read_text(encoding="utf-8"))


def system(tag: str) -> str:
    """A platform tag without Python's version: ``linux-x86_64-py3.12`` is
    ``linux-x86_64``. The numbers are the arithmetic's, not Python's."""
    return tag.split("-py", 1)[0]


# --- measurements ------------------------------------------------------------


def read_records(paths: list[Path]) -> tuple[dict[str, dict], dict[str, dict], int]:
    """Every case and group the runs measured, from record directories (the
    ``.jsonl`` files in them) or merged files, and how many shards said they
    finished (a ``done`` file in a directory)."""
    cases: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    done = 0

    def add(line: dict) -> None:
        case, gid = line["case"], line["group"]
        # One case recorded twice (a re-run in one directory): the later
        # wins only when it agrees, so a mixed run cannot hide a drift.
        if case in cases and (cases[case]["worst"], cases[case]["p95"]) != (
            line["worst"],
            line["p95"],
        ):
            raise ValueError(f"{case}: two records disagree: {cases[case]} and {line}")
        cases[case] = line
        g = groups.setdefault(gid, {"oracles": {}, "dataset": "", "tests": set()})
        g["oracles"].update(line.get("oracles", {}))
        g["dataset"] = line.get("dataset", "")
        if line.get("test"):
            g["tests"].add(line["test"])

    for path in paths:
        if path.is_dir():
            done += sum(1 for _ in path.rglob("done"))
            for f in sorted(path.rglob("*.jsonl")):
                for text in f.read_text(encoding="utf-8").splitlines():
                    if text.strip():
                        add(json.loads(text))
        else:
            merged = json.loads(path.read_text(encoding="utf-8"))
            for line in merged["cases"].values():
                add(line)
            done += merged.get("shards", 0)
    for g in groups.values():
        g["tests"] = sorted(g["tests"])
    return cases, groups, done


# --- judging -------------------------------------------------------------------


@dataclass
class Verdict:
    key: str  # <case>.worst, <case>.p95, or <group>.oracles
    status: str  # ok | over | under | elsewhere | new | version | missing
    message: str
    measured: float | None = None

    @property
    def failed(self) -> bool:
        return self.status in ("over", "under", "version", "missing")


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def written(v: float, ideal: float = 0.0) -> float:
    """A measurement as the ledger writes it: its gap from ``ideal`` to six
    significant figures, far inside the tolerance, in the fewest digits
    that keep it (an ARI of 0.9999871 is 0.999987, its gap 1.29e-5 kept to
    1e-6 of itself, where six figures of the value would move the gap by
    4%)."""
    gap = float(f"{v - ideal:.6g}")
    for digits in range(6, 18):
        out = float(f"{ideal + gap:.{digits}g}")
        if abs((out - ideal) - gap) <= 1e-6 * abs(gap):
            return out
    return ideal + gap


def judge_case(
    case: str,
    measured: dict,
    entry: dict | None,
    tol: float,
    home: str = "",
    elsewhere: float | None = None,
) -> list[Verdict]:
    """``home`` is the ledger's platform; a case measured on another is held
    to ``elsewhere`` past the ledger and never asked to tighten."""
    platform = system(measured.get("platform", "")) or home
    here = not home or platform == home
    if entry is None or "worst" not in entry:
        how = "`apply` writes it" if here else f"CI's run on {home} writes it"
        return [
            Verdict(
                case,
                "new",
                f"{case}: not in the ledger yet; measured worst {_fmt(measured['worst'])}, "
                f"p95 {_fmt(measured['p95'])} ({measured['n']} values): {how}",
            )
        ]
    tol = float(entry.get("tolerance", tol))
    if "tolerance" in measured:
        # A count (`ledger.record(..., exact=True)`) is held exactly even
        # before `apply` has written its tolerance into the ledger.
        tol = min(tol, float(measured["tolerance"]))
    under_tol = tol
    if not here and elsewhere is not None and tol > 0:
        # Another platform's last bits: a count is held exactly everywhere.
        tol = max(tol, elsewhere)
    ideal = float(entry.get("ideal", 0.0))
    if abs(float(measured.get("ideal", 0.0)) - ideal) > 0:
        return [
            Verdict(
                case,
                "over",
                f"{case}: the test's ideal is {measured.get('ideal')}, the ledger's {ideal}",
            )
        ]
    floor = max(float(entry.get("floor", 0.0)), float(measured.get("floor", 0.0)))
    out = []
    for field in ("worst", "p95"):
        key = f"{case}.{field}"
        have, want = float(measured[field]), float(entry[field])
        g, g0 = max(abs(have - ideal), floor), max(abs(want - ideal), floor)
        # To the twelfth place: exactly the tolerance away is within it.
        if round(g - g0 * (1 + tol), 12) > 0:
            where = "" if here else f" (measured on {platform}, the ledger on {home})"
            out.append(
                Verdict(
                    key,
                    "over",
                    f"{key}: {_fmt(have)} is {_fmt(g)} from its ideal {_fmt(ideal)}, past the "
                    f"ledger's {_fmt(want)} ({_fmt(g0)}) by more than {tol:.0%}{where}. Find "
                    "what moved it; if that is the price of a change worth having, raise it "
                    "with a [[raised]] entry naming the reason and a CHANGELOG line.",
                    have,
                )
            )
        elif round(g - g0 * (1 - under_tol), 12) < 0 and not here:
            out.append(
                Verdict(
                    key,
                    "elsewhere",
                    f"{key}: {_fmt(have)} on {platform}, under the ledger's {_fmt(want)} by "
                    f"more than {under_tol:.0%}; the ledger holds {home}'s numbers, so CI's "
                    "run there tightens it if the gain is real",
                    have,
                )
            )
        elif round(g - g0 * (1 - tol), 12) < 0:
            out.append(
                Verdict(
                    key,
                    "under",
                    f"{key}: {_fmt(have)} is {_fmt(g)} from its ideal, under the ledger's "
                    f"{_fmt(want)} ({_fmt(g0)}) by more than {tol:.0%}: tighten it to "
                    f"{written(have, ideal)!r} (`apply`), so the gain is kept.",
                    have,
                )
            )
        else:
            out.append(Verdict(key, "ok", f"{key}: {_fmt(have)}, ledger {_fmt(want)}", have))
    return out


def judge_group(gid: str, measured: dict, entry: dict | None) -> list[Verdict]:
    if entry is None:
        return []
    out = []
    want = entry.get("oracles", {})
    for lib, version in sorted(measured["oracles"].items()):
        if want.get(lib) != version:
            out.append(
                Verdict(
                    f"{gid}.oracles",
                    "version",
                    f"{gid}: {lib} {want.get(lib, '(none)')} in the ledger, {version} measured: "
                    "the numbers held, so `apply` records the version they were measured on",
                )
            )
    return out


def judge_all(doc: dict, cases: dict[str, dict], groups: dict[str, dict]) -> list[Verdict]:
    meta = doc["meta"]
    tol = float(meta["tolerance"])
    home = system(meta.get("platform", ""))
    elsewhere = meta.get("tolerance_elsewhere")
    elsewhere = None if elsewhere is None else float(elsewhere)
    out: list[Verdict] = []
    ledger_cases = doc.get("case", {})
    for case in sorted(cases):
        out += judge_case(case, cases[case], ledger_cases.get(case), tol, home, elsewhere)
    for gid in sorted(groups):
        out += judge_group(gid, groups[gid], doc.get("group", {}).get(gid))
    return out


def missing(doc: dict, cases: dict[str, dict], only: set[str] | None = None) -> list[Verdict]:
    """The ledger's cases a complete run did not measure (``only``: of these
    groups)."""
    out = []
    for case, entry in sorted(doc.get("case", {}).items()):
        gid = case.rsplit(".", 1)[0]
        if case not in cases and "worst" in entry and (only is None or gid in only):
            out.append(
                Verdict(
                    case,
                    "missing",
                    f"{case}: in the ledger, and the run did not measure it "
                    "(skipped, offline, or its record call is gone)",
                )
            )
    return out


# --- the file --------------------------------------------------------------------


def numbers(doc: dict) -> dict[str, float]:
    """Every number a raise must answer for, as a gap: each case's worst and
    p95, its floor and tolerance where it has them, and the tolerance. A gap
    under its case's floor is the floor, as ``check`` judges it: round-off
    moving below the floor raises nothing, and the floor's own key answers
    for a wider one."""
    out = {"tolerance": float(doc["meta"]["tolerance"])}
    for case, e in doc.get("case", {}).items():
        ideal = float(e.get("ideal", 0.0))
        floor = float(e.get("floor", 0.0))
        for f in ("worst", "p95"):
            if f in e:
                out[f"{case}.{f}"] = max(abs(float(e[f]) - ideal), floor)
        if "worst" in e:
            # No floor is a floor of 0: adding one is a raise too.
            out[f"{case}.floor"] = floor
        if "tolerance" in e:
            out[f"{case}.tolerance"] = float(e["tolerance"])
    return out


def raised_value(doc: dict, key: str, v: float) -> float:
    """A ``[[raised]]`` entry's ``from``/``to`` as the gap ``numbers`` keys."""
    case, f = key.rsplit(".", 1) if "." in key else ("", key)
    if f in ("worst", "p95"):
        e = doc.get("case", {}).get(case, {})
        return max(abs(v - float(e.get("ideal", 0.0))), float(e.get("floor", 0.0)))
    return v


def _rebase_allows(before_doc: dict, after_doc: dict, ledger: dict) -> dict[str, float]:
    """The most each number may reach without a ``[[raised]]`` entry when
    ``after_doc`` moves the ledger to another platform than ``before_doc``'s
    and a ``[[rebased]]`` entry names the move: its gap and the tolerance
    (a case's own, else the file's) -- the share the ledger already gives
    another platform's last bits. A floor or a tolerance gets nothing."""
    old = before_doc.get("meta", {}).get("platform")
    new = after_doc.get("meta", {}).get("platform")
    if old == new or not any(
        (r.get("from"), r.get("to")) == (old, new) for r in ledger.get("rebased", [])
    ):
        return {}
    tol = float(before_doc.get("meta", {}).get("tolerance", 0.0))
    out = {}
    for case, e in before_doc.get("case", {}).items():
        ideal = float(e.get("ideal", 0.0))
        t = float(e.get("tolerance", tol))
        floor = float(e.get("floor", 0.0))
        for f in ("worst", "p95"):
            if f in e:
                out[f"{case}.{f}"] = max(abs(float(e[f]) - ideal), floor) * (1 + t)
    return out


def _unexplained(
    before: dict, after: dict, ledger: dict, allowed: dict[str, float] | None = None
) -> list[str]:
    allowed = allowed or {}
    raised = set()
    for r in ledger.get("raised", []):
        if {"key", "from", "to"} <= set(r):
            k = r["key"]
            raised.add(
                (
                    k,
                    written(raised_value(ledger, k, r["from"])),
                    written(raised_value(ledger, k, r["to"])),
                )
            )

    def at_floor(k: str) -> bool:
        # A worst or p95 that rose only to its case's floor: the floor's
        # own key answers for it.
        case, f = k.rsplit(".", 1)
        return f in ("worst", "p95") and after[k] == after.get(f"{case}.floor")

    return [
        f"{k}: {before[k]:.6g} -> {after[k]:.6g} with no [[raised]] entry for it"
        for k in sorted(set(before) & set(after))
        if after[k] > before[k]
        and not at_floor(k)
        and (k, written(before[k]), written(after[k])) not in raised
        # Within the rebase's share, to the six figures the file writes.
        and not (k in allowed and after[k] <= allowed[k] * (1 + 1e-6))
    ]


def check_file(doc: dict, changelog: str) -> list[str]:
    """What is wrong with the ledger on its own."""
    errors = []
    meta = doc.get("meta", {})
    for key in ("tolerance", "tolerance_elsewhere", "platform"):
        if key not in meta:
            errors.append(f"[meta] has no {key}")
    tol = meta.get("tolerance")
    if not (isinstance(tol, float) and 0.0 < tol < 0.5):
        errors.append(f"tolerance = {tol!r}: a share above 0 and below 0.5")
    elsewhere = meta.get("tolerance_elsewhere")
    if isinstance(tol, float) and not (isinstance(elsewhere, float) and tol <= elsewhere < 0.5):
        errors.append(f"tolerance_elsewhere = {elsewhere!r}: from the tolerance to below 0.5")
    if "platform" in meta and system(str(meta["platform"])) != meta["platform"]:
        errors.append(f"platform = {meta['platform']!r}: the system and processor alone")
    groups = doc.get("group", {})
    for case, e in doc.get("case", {}).items():
        gid = case.rsplit(".", 1)[0]
        if gid not in groups:
            errors.append(f"{case}: no [group.{gid!r}]")
        for f in ("ideal", "worst", "p95", "floor", "tolerance"):
            v = e.get(f)
            if v is not None and not (isinstance(v, int | float) and math.isfinite(v)):
                errors.append(f"{case}.{f} = {v!r}: a finite number")
        if ("worst" in e) != ("p95" in e):
            errors.append(f"{case}: worst and p95 come together")
        if "worst" in e and "p95" in e:
            ideal = float(e.get("ideal", 0.0))
            if abs(e["p95"] - ideal) > abs(e["worst"] - ideal):
                errors.append(f"{case}: p95 {e['p95']} is further from {ideal} than worst")
    for gid, g in groups.items():
        if not isinstance(g.get("oracles", {}), dict):
            errors.append(f"{gid}.oracles: a table of library = version")
    folded = " ".join(changelog.split()).casefold()
    for i, r in enumerate(doc.get("raised", [])):
        where = f"[[raised]] #{i + 1} ({r.get('key', '?')})"
        lack = {"key", "from", "to", "date", "reason", "changelog"} - set(r)
        if lack:
            errors.append(f"{where} has no {sorted(lack)}")
            continue
        if r["key"] not in numbers(doc):
            errors.append(f"{where}: no such number")
        elif not raised_value(doc, r["key"], r["to"]) > raised_value(doc, r["key"], r["from"]):
            errors.append(f"{where}: `to` {r['to']} is no further from the ideal than {r['from']}")
        if not isinstance(r["date"], dt.date):
            errors.append(f"{where}: `date` is not a TOML date")
        if len(str(r["reason"]).split()) < 4:
            errors.append(f"{where}: the reason says too little: {r['reason']!r}")
        if " ".join(str(r["changelog"]).split()).casefold() not in folded:
            errors.append(f"{where}: the CHANGELOG has no line containing {r['changelog']!r}")
    for i, r in enumerate(doc.get("rebased", [])):
        where = f"[[rebased]] #{i + 1} ({r.get('from', '?')} -> {r.get('to', '?')})"
        lack = {"from", "to", "date", "reason", "changelog"} - set(r)
        if lack:
            errors.append(f"{where} has no {sorted(lack)}")
            continue
        if r["from"] == r["to"]:
            errors.append(f"{where}: a move needs two platforms")
        if not isinstance(r["date"], dt.date):
            errors.append(f"{where}: `date` is not a TOML date")
        if len(str(r["reason"]).split()) < 4:
            errors.append(f"{where}: the reason says too little: {r['reason']!r}")
        if " ".join(str(r["changelog"]).split()).casefold() not in folded:
            errors.append(f"{where}: the CHANGELOG has no line containing {r['changelog']!r}")
    return errors


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def history() -> tuple[list[tuple[str, str]], bool]:
    shallow = _git("rev-parse", "--is-shallow-repository").strip() == "true"
    shas = _git("log", "--reverse", "--format=%H", "--", "oracles.toml").split()
    return [(s, _git("show", f"{s}:oracles.toml")) for s in shas], shallow


def check_history(versions: list[tuple[str, str]], working: str | None) -> list[str]:
    """Every raise across the file's versions, the working tree's last,
    answered by a ``[[raised]]`` entry of the file as it stands; each number
    held to the last one the history gave its key."""
    texts = [(sha[:10], text) for sha, text in versions]
    if working is not None and (not texts or texts[-1][1] != working):
        texts.append(("working tree", working))
    if not texts:
        return []
    ledger = tomllib.loads(texts[-1][1])
    errors: list[str] = []
    last: dict[str, float] = {}
    previous: dict = {}
    for i, (label, text) in enumerate(texts):
        doc = tomllib.loads(text)
        now = numbers(doc)
        if i:
            allowed = _rebase_allows(previous, doc, ledger)
            errors += [f"{label}: {e}" for e in _unexplained(last, now, ledger, allowed)]
        last.update(now)
        previous = doc
    return errors


# --- writing the file --------------------------------------------------------------


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        return "[\n" + "".join(f"    {_toml_value(x)},\n" for x in v) + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        return "{ " + ", ".join(f"{json.dumps(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _set_table(text: str, header: str, values: dict) -> str:
    """``text`` with the table under ``header`` holding ``values`` (each key
    set, others kept), the table written before the raises when new."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration:
        at = next((i for i, ln in enumerate(lines) if ln.startswith(RAISES_MARK)), len(lines))
        lines[at:at] = [header, ""]
        start = at
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith(("[", "# ---"))),
        len(lines),
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    body = "\n".join(lines[start + 1 : end])
    current = tomllib.loads(body) if body.strip() else {}
    current.update(values)
    new = [f"{k} = {_toml_value(v)}" for k, v in current.items()]
    lines[start + 1 : end] = "\n".join(new).splitlines()
    return "\n".join(lines) + "\n"


def _quoted(key: str) -> str:
    return json.dumps(key)


def apply(
    text: str, doc: dict, cases: dict, groups: dict, verdicts: list[Verdict]
) -> tuple[str, list[str]]:
    """Write what a run settles: each new case, each tightened number, each
    measured group's tests, oracles and dataset. Never a raise."""
    done = []
    by_key = {v.key: v for v in verdicts}
    home = system(doc.get("meta", {}).get("platform", ""))
    for gid, g in sorted(groups.items()):
        old = doc.get("group", {}).get(gid, {})
        # The tests that recorded it: a run measures a group whole, so a test
        # renamed or removed leaves the list (a stale name fails the guard).
        tests = sorted(g["tests"]) or sorted(old.get("tests", []))
        values = {
            "tests": tests,
            "oracles": dict(sorted(g["oracles"].items())),
            "dataset": g["dataset"],
        }
        if {k: old.get(k) for k in values} != values:
            text = _set_table(text, f"[group.{_quoted(gid)}]", values)
            done.append(f"{gid}: tests, oracles and dataset recorded")
    for case, m in sorted(cases.items()):
        entry = doc.get("case", {}).get(case)
        header = f"[case.{_quoted(case)}]"
        # Another platform's numbers never enter the ledger: CI's do.
        here = not home or (system(m.get("platform", "")) or home) == home
        if entry is None or "worst" not in entry:
            values = {"ideal": m["ideal"]}
            if here:
                values["worst"] = written(m["worst"], m["ideal"])
                values["p95"] = written(m["p95"], m["ideal"])
            values["n"] = m["n"]
            if m.get("floor"):
                values["floor"] = m["floor"]
            if "tolerance" in m:
                values["tolerance"] = m["tolerance"]
            if entry is not None and not here:
                continue
            text = _set_table(text, header, values)
            how = "" if here else f", its numbers left to CI's run on {home}"
            done.append(f"{case}: worst {_fmt(m['worst'])}, p95 {_fmt(m['p95'])} (new{how})")
            continue
        values = {}
        for f in ("worst", "p95"):
            v = by_key.get(f"{case}.{f}")
            if v is not None and v.status == "under":
                values[f] = written(m[f], float(entry.get("ideal", 0.0)))
                done.append(f"{case}.{f}: tightened to {_fmt(m[f])}")
        if entry.get("n") != m["n"]:
            values["n"] = m["n"]
        if "tolerance" in m and float(entry.get("tolerance", 1.0)) > m["tolerance"]:
            # A case the test now says is a count: a narrower tolerance, so
            # a tightening, which `apply` may write.
            values["tolerance"] = m["tolerance"]
            done.append(f"{case}: held exactly, a count")
        if values:
            text = _set_table(text, header, values)
    tomllib.loads(text)
    return text, done


# --- the hash file -------------------------------------------------------------------


def write_hashes(text: str, hashes: dict[str, str], platform: str) -> str:
    doc = tomllib.loads(text) if text.strip() else {}
    current = dict(doc.get("hashes", {}))
    current.update(hashes)
    head = text.split("[meta]", 1)[0] if "[meta]" in text else ""
    meta = dict(doc.get("meta", {}))
    meta["platform"] = platform
    out = head + "[meta]\n" + "".join(f"{k} = {_toml_value(v)}\n" for k, v in meta.items())
    out += "\n[hashes]\n" + "".join(
        f"{_quoted(k)} = {json.dumps(v)}\n" for k, v in sorted(current.items())
    )
    tomllib.loads(out)
    return out


# --- reporting -----------------------------------------------------------------------


def report(verdicts: list[Verdict], extra: dict[str, list[str]]) -> str:
    lines = ["### Oracle ledger", ""]
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    lines.append(", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "nothing measured")
    for heading, status in (
        ("Past the ledger", "over"),
        ("Tighten", "under"),
        ("Under the ledger on another platform (not applied)", "elsewhere"),
        ("Oracle versions moved", "version"),
        ("Not measured", "missing"),
        ("New cases", "new"),
    ):
        items = [v.message for v in verdicts if v.status == status]
        if items:
            lines += ["", f"**{heading}:**", ""] + [f"- {m}" for m in items]
    for heading, items in extra.items():
        if items:
            lines += ["", f"**{heading}:**", ""] + [f"- {m}" for m in items]
    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


# --- the command line ------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    doc = load(args.ledger)
    cases, groups, done = read_records(args.records)
    verdicts = judge_all(doc, cases, groups)
    extra: dict[str, list[str]] = {}
    complete = args.expect_shards is None or done >= args.expect_shards
    if complete:
        verdicts += missing(doc, cases)
    else:
        extra["Incomplete run"] = [
            f"{done} of {args.expect_shards} shards finished: cases not measured are not judged"
        ]
    text = report(verdicts, extra)
    print(text)
    write_summary(text)
    if args.measured is not None:
        merged = {"cases": cases, "shards": done}
        args.measured.write_text(
            json.dumps(merged, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    failed = [v.message for v in verdicts if v.failed] + extra.get("Incomplete run", [])
    for m in failed:
        print(f"::error::{m}")
    for v in verdicts:
        if v.status == "new":
            print(f"::warning::{v.message}")
    return 1 if failed else 0


def _cmd_apply(args: argparse.Namespace) -> int:
    doc = load()
    cases, groups, _ = read_records(args.records)
    verdicts = judge_all(doc, cases, groups)
    over = [v.message for v in verdicts if v.status == "over"]
    text, done = apply(LEDGER.read_text(encoding="utf-8"), doc, cases, groups, verdicts)
    LEDGER.write_text(text, encoding="utf-8")
    print("\n".join(done) or "nothing to change")
    for m in over:
        print(f"not applied, past the ledger: {m}")
    return 0


def _cmd_raises(_: argparse.Namespace) -> int:
    doc = load()
    errors = check_file(doc, CHANGELOG.read_text(encoding="utf-8"))
    versions, shallow = history()
    errors += check_history(versions, LEDGER.read_text(encoding="utf-8"))
    print(f"oracles.toml: {len(versions)} committed versions read{' (shallow clone)' * shallow}")
    for e in errors:
        print(f"::error::{e}")
    return 1 if errors else 0


def _guard_run(groups: list[str], out: Path) -> dict[str, dict]:
    """Run the hash guard, writing each group's computed hash and verdict."""
    env = dict(os.environ, **{HASHES_OUT: str(out)})
    cmd = ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", GUARD_TEST]
    if groups:
        cmd += ["-k", " or ".join(groups)]
    subprocess.run(cmd, cwd=REPO, env=env, check=False, stdout=subprocess.DEVNULL)
    found = {}
    if out.exists():
        for text in out.read_text(encoding="utf-8").splitlines():
            line = json.loads(text)
            found[line["group"]] = line
    return found


def _exists(nodeid: str) -> bool:
    """Whether the test a node id names is still defined in its file."""
    path, *names = nodeid.split("::")
    file = REPO / path
    if not file.exists():
        return False
    name = names[-1].split("[", 1)[0] if names else ""
    return f"def {name}(" in file.read_text(encoding="utf-8")


def _cmd_rerun(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        hashes = _guard_run([], Path(tmp) / "hashes.jsonl")
        if not hashes:
            print("rerun: the hash guard computed nothing; run it to see why:")
            print(f"  uv run pytest {GUARD_TEST}")
            return 1
        stale = sorted(g for g, h in hashes.items() if not h["holds"])
        chosen = sorted(set(args.groups)) if args.groups else stale
        unknown = sorted(set(chosen) - set(hashes))
        if unknown:
            print(f"rerun: no such groups: {unknown}")
            return 2
        if not chosen:
            print("rerun: every group's hash and oracle versions hold; nothing to run")
            return 0
        doc = load()
        targets: list[str] = []
        for g in chosen:
            tests = doc.get("group", {}).get(g, {}).get("tests", [])
            # A test renamed or removed since the ledger named it: the whole
            # module, whose grids record the group's tests as they are now.
            if not tests or not all(_exists(t) for t in tests):
                tests = [f"tests/{hashes[g]['module']}.py"]
            targets += tests
        targets = sorted(set(targets))
        print(f"rerun: {len(chosen)} groups: {' '.join(chosen)}")
        records = Path(tmp) / "records"
        env = dict(os.environ, **{OUT_VARIABLE: str(records)})
        workers = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "workers.py")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        cmd = ["uv", "run", "pytest", "-q", "-m", "oracle_grid", "-n", workers]
        cmd += ["--dist", "worksteal", *targets]
        print("rerun: $ " + " ".join(cmd), flush=True)
        if subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode:
            print("rerun: FAIL: an oracle test failed; the hashes are not updated")
            return 1
        cases, groups, _ = read_records([records])
        cases = {c: m for c, m in cases.items() if m["group"] in chosen}
        groups = {g: m for g, m in groups.items() if g in chosen}
        verdicts = judge_all(doc, cases, groups) + missing(doc, cases, set(chosen))
        unmeasured = [g for g in chosen if g not in groups]
        bad = [v.message for v in verdicts if v.status in ("over", "missing")]
        bad += [f"{g}: its oracle tests recorded nothing" for g in unmeasured]
        print(report(verdicts, {}))
        if bad:
            print("rerun: FAIL: the hashes are not updated:")
            print("\n".join(f"  {m}" for m in bad))
            return 1
        text, done = apply(LEDGER.read_text(encoding="utf-8"), doc, cases, groups, verdicts)
        LEDGER.write_text(text, encoding="utf-8")
        # The tests the ledger now names are part of each hash: compute again.
        again = _guard_run(chosen, Path(tmp) / "again.jsonl")
        new = {g: again[g]["hash"] for g in chosen}
        platform = next(iter(again.values()))["platform"]
        old = HASHES.read_text(encoding="utf-8") if HASHES.exists() else ""
        HASHES.write_text(write_hashes(old, new, platform), encoding="utf-8")
        print("\n".join(done))
        print(f"rerun: PASS: {len(new)} hashes written to oracle_hashes.toml")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Every dataset the ledger's groups name, each tried ``rounds`` times
    (the loaders retry three times within a round), into ``.cache``."""
    import time

    sys.path.insert(0, str(REPO / "tests"))
    import data
    import public_benchmarks

    wanted = sorted({g.get("dataset", "") for g in load().get("group", {}).values()} - {""})
    failed = []
    for dataset in wanted:
        for attempt in range(args.rounds):
            try:
                got = public_benchmarks.fetch(dataset)
            except data.Offline as e:
                if attempt + 1 == args.rounds:
                    failed.append(f"{dataset}: {e.__cause__!r}")
                else:
                    time.sleep(30.0 * (attempt + 1))
                continue
            print(f"fetch: {', '.join(got)}", flush=True)
            break
    for f in failed:
        print(f"::error::fetch: could not download {f}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="hold a run's records to the ledger")
    c.add_argument("records", type=Path, nargs="+")
    c.add_argument(
        "--expect-shards", type=int, help="judge missing cases only if N shards finished"
    )
    c.add_argument("--measured", type=Path, help="write the merged records here, for `apply`")
    c.add_argument("--ledger", type=Path, help="another ledger than oracles.toml (for tests)")
    c.set_defaults(fn=_cmd_check)
    a = sub.add_parser("apply", help="write new cases, tightenings and versions from records")
    a.add_argument("records", type=Path, nargs="+")
    a.set_defaults(fn=_cmd_apply)
    r = sub.add_parser("raises", help="every raise in the file's history has its reason")
    r.set_defaults(fn=_cmd_raises)
    rr = sub.add_parser("rerun", help="run the oracle grids whose hashes moved, then record")
    rr.add_argument("groups", nargs="*")
    rr.set_defaults(fn=_cmd_rerun)
    f = sub.add_parser("fetch", help="download every dataset the ledger's groups name")
    f.add_argument("--rounds", type=int, default=3)
    f.set_defaults(fn=_cmd_fetch)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
