"""The reader's rule of thumb, one fixed rate per stream from its mean rows
waiting n: 0.05 / (1 + 0.1 * g(alpha) * n), g(alpha) the Gaussian loop gain
(docs/records/CONFORMAL-STEP.md, §5), against today's default."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold  # noqa: E402
import streams  # noqa: E402

print("| flights | level | n | divisor | rate | coverage / half-width | default 0.05 |")
print("|---|---:|---:|---:|---:|---:|---:|")
for d in streams.FLIGHT_DELAYS:
    s = streams.flights(d)
    sc = fold.schedule(s["t"], s["avail"], s["resid"], s["w"], s["half_life"], s["gap_cap"])
    n = float(sc["held_before"][s["mask"]].mean()) - 1.0
    for lvl in (0.5, 0.9, 0.99):
        div = 1.0 / (0.1 * fold.gain(1 - lvl))
        rate = 0.05 / (1 + n / div)
        m = fold.score(fold.track(sc, s["resid"], s["w"], lvl, rate=rate), s["resid"], s["mask"])
        m0 = fold.score(fold.track(sc, s["resid"], s["w"], lvl), s["resid"], s["mask"])
        print(
            f"| {d} | {lvl} | {n:.0f} | {div:.0f} | {rate:.4f} "
            f"| {m['coverage']:.4f} / {m['half_width']:.1f} "
            f"| {m0['coverage']:.4f} / {m0['half_width']:.1f} |"
        )
