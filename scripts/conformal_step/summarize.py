"""The tables of docs/records/CONFORMAL-STEP.md from measure.json."""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streams  # noqa: E402

rows = json.loads((streams.CACHE / "measure.json").read_text(encoding="utf-8"))
FIXED = [r for r in dict.fromkeys(x["rule"] for x in rows) if r.startswith("fixed")]
RULES = list(dict.fromkeys(x["rule"] for x in rows))


def key(x):
    return (x["stream"], tuple(x["case"]), x["level"])


by = defaultdict(dict)
for x in rows:
    by[key(x)][x["rule"]] = x


TOL = {0.9: 0.005, 0.5: 0.01, 0.99: 0.002}


def best(cell, level, pre=""):
    """The narrowest fixed rate whose coverage is within tolerance: the rate
    a user would have chosen with hindsight (NaN when none is)."""
    tol = TOL[level]
    ok = [
        cell[r][pre + "half_width"] for r in FIXED if abs(cell[r][pre + "coverage"] - level) <= tol
    ]
    return min(ok) if ok else float("nan"), tol


def verdicts(level, pre="", width_slack=1.10):
    res = {r: [0, 0, 0] for r in RULES}  # holds, coverage off, too wide
    worst = {r: 0.0 for r in RULES}
    widest = {r: 0.0 for r in RULES}
    for k, cell in by.items():
        if k[2] != level:
            continue
        b, tol = best(cell, level, pre)
        for r in RULES:
            c, h = cell[r][pre + "coverage"], cell[r][pre + "half_width"]
            worst[r] = max(worst[r], abs(c - level))
            if b == b:
                widest[r] = max(widest[r], h / b)
            if abs(c - level) > tol:
                res[r][1] += 1
            elif b == b and h > width_slack * b:
                res[r][2] += 1
            else:
                res[r][0] += 1
    return res, worst, widest


for level in (0.9, 0.5, 0.99):
    for pre, what in (
        ("", "rows from 10% (flights: October to December)"),
        ("late_", "the second half of those rows"),
    ):
        res, worst, widest = verdicts(level, pre)
        n = sum(res[RULES[0]])
        print(
            f"\n### level {level}, {what}: {n} streams; holds = coverage within "
            f"{TOL[level]} and half-width within 10% of the best fixed rate in hindsight"
        )
        print("| rule | holds | coverage off | too wide | worst coverage error | widest / best |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in RULES:
            print(
                f"| {r} | {res[r][0]} | {res[r][1]} | {res[r][2]} "
                f"| {worst[r]:.4f} | {widest[r]:.2f} |"
            )

print("\n### flights, 90%: coverage / half-width (minutes)")
show = RULES
print("| rule | " + " | ".join(streams.FLIGHT_DELAYS) + " |")
print("|---|" + "---:|" * len(streams.FLIGHT_DELAYS))
for r in show:
    cells = []
    for d in streams.FLIGHT_DELAYS:
        x = by[("flights", (d,), 0.9)][r]
        cells.append(f"{x['coverage']:.4f} / {x['half_width']:.1f}")
    print(f"| {r} | " + " | ".join(cells) + " |")
print(
    "| mean rows held | "
    + " | ".join(
        f"{by[('flights', (d,), 0.9)][RULES[0]]['held']:.0f}" for d in streams.FLIGHT_DELAYS
    )
    + " |"
)

print("\n### generated, 90%: coverage / half-width, mean over 3 seeds and the 4 noises")
conds = [(rt, d) for rt in streams.RATES for d in streams.DELAYS]
print("| rule | " + " | ".join(f"{rt} {d}" for rt, d in conds) + " |")
print("|---|" + "---:|" * len(conds))
for r in show:
    cells = []
    for rt, d in conds:
        xs = [
            by[k][r]
            for k in by
            if k[0] == "generated" and k[1][2] == rt and k[1][3] == d and k[2] == 0.9
        ]
        c = sum(x["coverage"] for x in xs) / len(xs)
        h = sum(x["half_width"] for x in xs) / len(xs)
        cells.append(f"{c:.4f} / {h:.2f}")
    print(f"| {r} | " + " | ".join(cells) + " |")
heldrow = []
for rt, d in conds:
    xs = [
        by[k][RULES[0]]["held"]
        for k in by
        if k[0] == "generated" and k[1][2] == rt and k[1][3] == d and k[2] == 0.9
    ]
    heldrow.append(f"{sum(xs) / len(xs):.0f}")
print("| mean rows held | " + " | ".join(heldrow) + " |")

print(
    "\n### generated, 90%, by noise: coverage / half-width, "
    "mean over 3 seeds, 2 row rates, the 3 delays"
)
print("| rule | " + " | ".join(streams.NOISES) + " |")
print("|---|" + "---:|" * len(streams.NOISES))
for r in show:
    cells = []
    for nz in streams.NOISES:
        xs = [
            by[k][r]
            for k in by
            if k[0] == "generated" and k[1][1] == nz and k[1][3] != "none" and k[2] == 0.9
        ]
        cells.append(
            f"{sum(x['coverage'] for x in xs) / len(xs):.4f} / "
            f"{sum(x['half_width'] for x in xs) / len(xs):.2f}"
        )
    print(f"| {r} | " + " | ".join(cells) + " |")
