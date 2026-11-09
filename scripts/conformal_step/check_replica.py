"""The bank's tracker is `fold.track_m`: the radius every row is shown, read
back from the bank's `lo`, to the bit, at the default rate and under
`conformal_rate="auto"`, on each stream and delay the measurement uses
(docs/records/CONFORMAL-STEP.md §8). The rules of §3-§5 replay the tracker on
the radius the bank kept before task 368, which no longer runs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold  # noqa: E402
import streams  # noqa: E402

import polars_online as po  # noqa: E402

cases = [("flights", d) for d in streams.FLIGHT_DELAYS] + [
    ("generated", (0, nz, rt, d))
    for nz in ("gauss", "shift")
    for rt in streams.RATES
    for d in streams.DELAYS
]
banked = {}
plain_run = streams.run


def both_rates(frame, spec, half_life, gap_cap, mask):
    """`streams.run`, and the same spec under "auto": its `lo`."""
    s = plain_run(frame, spec, half_life, gap_cap, mask)
    auto = po.ModelBank([{**spec, "conformal_rate": "auto"}]).fit_predict(frame)["m"].struct
    banked["auto"] = auto.field("lo_y").to_numpy().astype(float)
    return s


streams.run = both_rates
bad = 0
for kind, arg in cases:
    s = streams.flights(arg) if kind == "flights" else streams.generated(*arg)
    sch = fold.schedule(s["t"], s["avail"], s["resid"], s["w"], s["half_life"], s["gap_cap"])
    for rate, auto, want in (("0.05", False, s["lo"]), ("auto", True, banked["auto"])):
        shown = fold.track_m(sch, s["resid"], s["w"], s["sigma"], 0.9, auto=auto)
        lo = s["pred"] - shown
        same_nan = np.array_equal(np.isfinite(lo), np.isfinite(want))
        both = np.isfinite(lo) & np.isfinite(want)
        exact = np.array_equal(lo[both], want[both])
        held = sch["held_before"][s["mask"]].mean()
        print(
            f"{kind} {arg} {rate}: rows {both.sum()}, nulls agree {same_nan}, to the bit {exact}, "
            f"mean held {held:.1f}",
            flush=True,
        )
        bad += not (same_nan and exact)
sys.exit(1 if bad else 0)
