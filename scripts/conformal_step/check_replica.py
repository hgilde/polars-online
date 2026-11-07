"""The fold's `fixed` rule at the default rate is the bank's tracker: the
radius every row is shown, read back from the bank's `lo`, to the bit, on
each stream and delay the measurement uses (docs/records/CONFORMAL-STEP.md)."""

import sys

import fold
import numpy as np
import streams

cases = [("flights", d) for d in streams.FLIGHT_DELAYS] + [
    ("generated", (0, nz, rt, d))
    for nz in ("gauss", "shift")
    for rt in streams.RATES
    for d in streams.DELAYS
]
bad = 0
for kind, arg in cases:
    s = streams.flights(arg) if kind == "flights" else streams.generated(*arg)
    sch = fold.schedule(s["t"], s["avail"], s["resid"], s["w"], s["half_life"], s["gap_cap"])
    shown = fold.track(sch, s["resid"], s["w"], 0.9)
    lo = s["pred"] - shown
    both = np.isfinite(lo) & np.isfinite(s["lo"])
    same_nan = np.array_equal(np.isfinite(lo), np.isfinite(s["lo"]))
    exact = np.array_equal(lo[both], s["lo"][both])
    held = sch["held_before"][s["mask"]].mean()
    print(
        f"{kind} {arg}: rows {both.sum()}, nulls agree {same_nan}, to the bit {exact}, "
        f"mean held {held:.1f}",
        flush=True,
    )
    bad += not (same_nan and exact)
sys.exit(1 if bad else 0)
