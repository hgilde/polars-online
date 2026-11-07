"""Each step rule on each stream: coverage, mean half-width and the worst
coverage over blocks of 2,000 scored rows (docs/records/CONFORMAL-STEP.md).

Run: .venv/bin/python -I scripts/conformal_step/measure.py [processes]
Writes .cache/conformal-step/measure.json and prints the summary tables."""

from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold  # noqa: E402
import streams  # noqa: E402

FIXED = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
AG = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
LEVELS = (0.9, 0.5, 0.99)


def rules(coverage):
    for r in FIXED:
        yield f"fixed {r}", lambda sc, s, r=r: fold.track(sc, s["resid"], s["w"], coverage, rate=r)
    for c, tag in ((1 / math.e, "1/e"), (0.5, "1/2"), (1.0, "1")):
        yield (
            f"lag, mean held, c={tag}",
            lambda sc, s, c=c: fold.track(
                sc, s["resid"], s["w"], coverage, kind="lag_held_ew", c=c
            ),
        )
    yield (
        "lag, held now, c=1/e",
        lambda sc, s: fold.track(sc, s["resid"], s["w"], coverage, kind="lag_held", c=1 / math.e),
    )
    yield (
        "lag, own lag, c=1/e",
        lambda sc, s: fold.track(sc, s["resid"], s["w"], coverage, kind="lag_own", c=1 / math.e),
    )
    yield "agreeing radii", lambda sc, s: fold.track(sc, s["resid"], s["w"], coverage, kind="agree")
    yield (
        "current radius",
        lambda sc, s: fold.track(sc, s["resid"], s["w"], coverage, kind="current"),
    )
    yield "AgACI", lambda sc, s: fold.agaci(sc, s["resid"], coverage, AG)
    yield "SAOCP", lambda sc, s: fold.saocp(sc, s["resid"], coverage)


def one(case):
    kind, arg = case
    s = streams.flights(arg) if kind == "flights" else streams.generated(*arg)
    sc = fold.schedule(s["t"], s["avail"], s["resid"], s["w"], s["half_life"], s["gap_cap"])
    held = float(sc["held_before"][s["mask"]].mean())
    out = []
    levels = LEVELS if kind == "flights" or arg[0] == 0 else (0.9,)
    for cov in levels:
        for name, f in rules(cov):
            m = fold.score(f(sc, s), s["resid"], s["mask"])
            out.append(
                {
                    "stream": kind,
                    "case": list(arg) if kind != "flights" else [arg],
                    "level": cov,
                    "rule": name,
                    "held": held,
                    **m,
                }
            )
    print(kind, arg, flush=True)
    return out


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cases = [("flights", d) for d in streams.FLIGHT_DELAYS] + [
        ("generated", (seed, nz, rt, d))
        for seed in (0, 1, 2)
        for nz in streams.NOISES
        for rt in streams.RATES
        for d in streams.DELAYS
    ]
    with Pool(procs) as p:
        rows = [r for part in p.imap(one, cases) for r in part]
    (streams.CACHE / "measure.json").write_text(json.dumps(rows), encoding="utf-8")
