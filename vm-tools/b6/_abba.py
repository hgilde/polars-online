"""ABBA driver shared by the b6 scripts (release vs main, with a control).

Each script defines ``make_cases()`` returning {name: (callable() -> seconds, rows_or_None)}.
Run a script with ``--venv rel=PY main=PY main2=PY`` (main2 is a second copy
of the main build: the control) and ``--rounds N``. Each round runs the
processes in the order rel, main, main2, main2, main, rel; each process
times every case (best of ``--reps`` inside) and prints one JSON line.
A round's figure for a venv is the faster of its two processes.
Ratio is rel_seconds / main_seconds: above 1, main is faster.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from typing import Callable

ORDER = ["rel", "main", "main2", "main2", "main", "rel"]


def load() -> float:
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def worker(cases: dict[str, tuple[Callable[[], float], int | None]], reps: int) -> None:
    out = {}
    for name, (fn, rows) in cases.items():
        out[name] = {"secs": min(fn() for _ in range(reps)), "rows": rows}
    import polars_online

    out["_file"] = polars_online.__file__
    print("RESULT " + json.dumps(out), flush=True)


def driver(script: str, venvs: dict[str, str], rounds: int, reps: int, extra: list[str]) -> None:
    runs: dict[str, dict[str, list[float]]] = {}
    files: dict[str, str] = {}
    infos: dict[str, set] = {}
    l0 = load()
    for r in range(rounds):
        for v in ORDER:
            p = subprocess.run(
                [venvs[v], script, "--worker", "--reps", str(reps), *extra],
                capture_output=True, text=True, check=True,
            )
            res = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("RESULT "))[7:])
            infos.setdefault(v, set()).update(x for x in p.stdout.splitlines() if x.startswith('INFO '))
            files[v] = res.pop("_file")
            for case, d in res.items():
                runs.setdefault(case, {}).setdefault(v, []).append((r, d["secs"], d["rows"]))
    l1 = load()
    print(f"files: {files}")
    for v, i in infos.items():
        print(v, sorted(i))
    print(f"load before {l0:.2f} after {l1:.2f}; rounds {rounds}, reps {reps}")
    print("| case | rel | main | main/rel | control main2/main (min..max) | flag |")
    print("|---|---|---|---|---|---|")
    for case, per in runs.items():
        best = {v: {} for v in venvs}
        for v, lst in per.items():
            for r, s, _ in lst:
                best[v][r] = min(best[v].get(r, 1e99), s)
        rows = per["rel"][0][2]
        fmt = (lambda s: f"{rows / s:,.0f} rows/s") if rows else (lambda s: f"{s * 1e3:.3f} ms")
        rel = statistics.median(best["rel"].values())
        main = statistics.median(best["main"].values())
        ratios = [best["rel"][r] / best["main"][r] for r in range(rounds)]
        ctl = [best["main"][r] / best["main2"][r] for r in range(rounds)]
        tol = max(abs(c - 1) for c in ctl)
        med = statistics.median(ratios)
        flag = "SLOWER" if med < 1 - tol else ""
        print(
            f"| {case} | {fmt(rel)} | {fmt(main)} | {med:.3f} "
            f"({min(ratios):.3f}..{max(ratios):.3f}) | {min(ctl):.3f}..{max(ctl):.3f} | {flag} |"
        )


def main(script: str, make_cases: Callable[[], dict]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--venv", nargs="*", default=[], help="NAME=PYTHON for rel, main, main2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.worker:
        worker(make_cases(), a.reps)
    else:
        venvs = dict(x.split("=", 1) for x in a.venv)
        driver(script, venvs, a.rounds, a.reps, [])
