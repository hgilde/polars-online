"""Cases for emit_breaks through faer from order 24 (task 463).

k 4/20/200, 1 and 3 targets, 1/100/10,000 groups, chunks of 1 and 1,000 rows,
a 1M-row stream, nulls, a save and a load in the middle. A grid row is
(tag, rows, k, targets, groups, chunk, extra case fields).
"""

GRID = [
    ("k4-t1-g1-c1000", 200_000, 4, 1, 1, 1000, {}),
    ("k20-t1-g1-c1000", 200_000, 20, 1, 1, 1000, {}),
    ("k32-t1-g1-c1000", 100_000, 32, 1, 1, 1000, {}),
    ("k64-t1-g1-c1000", 40_000, 64, 1, 1, 1000, {}),
    ("k200-t1-g1-c1000", 8_000, 200, 1, 1, 1000, {}),
]

BREAKS = dict(emit_breaks=True, emit_drift=True, drift_threshold=20.0, emit_zscore=True)


def _decisions(targets, k):
    QS = {1: [4.605170185988092, 9.21034037197618, 13.815510557964274, 18.420680743952584], 2: [6.251388631170322, 11.34486673014437, 16.26623619623813, 21.10751346616044], 3: [7.779440339734858, 13.276704135987622, 18.46682695290317, 23.512742444991076], 4: [9.236356899781121, 15.086272469388987, 20.515005652432876, 25.74483195905612], 8: [14.683656573259839, 21.66599433346192, 27.87716487125657, 33.719948438964906], 12: [19.811929307127564, 27.68824961045705, 34.52817897487089, 40.8706550138363], 20: [29.615089436182735, 38.932172683516065, 46.79703804156131, 53.96200011641195], 22: [32.0068996817043, 41.638398118858476, 49.72823246643149, 57.074643138555636], 23: [33.19624428862818, 42.979820139351624, 51.17859777737738, 58.61296974830205], 24: [34.38158701755296, 44.314104896219135, 52.61965577617282, 60.1402919116224], 25: [35.563171271923466, 45.64168266628313, 54.05196238857663, 61.65726127708595], 26: [36.74121674779765, 46.96294212475146, 55.476020205745186, 63.16446741833317], 28: [39.087469770693964, 49.58788447289881, 58.30117348979492, 66.15168462853987], 30: [41.42173582978521, 52.19139483319192, 61.098306081058126, 69.10569228986755], 31: [42.58474508298083, 53.48577183623537, 62.48721905708849, 70.57124757534287], 32: [43.74517955943419, 54.775539760110334, 63.87009852234495, 72.02964379125795], 64: [79.97300265487546, 94.42207900788506, 105.98814308961288, 116.1599074780067], 100: [119.5886672425632, 136.97100384679405, 150.66705566784535, 162.58027012169546], 200: [227.0851408464757, 250.56135893915402, 268.69477444676386, 284.2456950861589]}

    out = []
    for t in range(targets):
        out += [
            {"name": f"|zscore| > 3 (y{t})", "kind": "threshold", "column": f"m.zscore_y{t}",
             "value": 3.0, "op": ">", "abs": True},
            {"name": f"drift > 20 (y{t})", "kind": "threshold", "column": f"m.drift_y{t}",
             "value": 20.0, "op": ">"},
            *[
                {"name": f"break_wald > chi2({k + 1}).ppf({q}) (y{t})", "kind": "threshold",
                 "column": f"m.break_wald_y{t}", "value": v, "op": ">"}
                for q, v in zip((0.9, 0.99, 0.999, 0.9999), QS[k])
            ],
            {"name": f"|cusum| > 3 (y{t})", "kind": "threshold", "column": f"m.cusum_y{t}",
             "value": 3.0, "op": ">", "abs": True},
            {"name": f"|cusum_sq| > 3 (y{t})", "kind": "threshold", "column": f"m.cusum_sq_y{t}",
             "value": 3.0, "op": ">", "abs": True},
        ]
    return out


def _specs(extra):
    def build(po, ctx):
        return [
            po.spec.ewridge(
                "m", targets=ctx["targets"], features=ctx["features"], group=ctx["group"],
                clock="t", gap_cap=ctx["gap_cap"], half_life=ctx["half_life"],
                weight=ctx["weight"], min_weight=float(ctx["k"]), **extra,
            )
        ]

    return build


def cases():
    out = []
    for tag, rows, k, m, g, chunk, more in GRID:
        case = {
            "name": f"c/{tag}", "item": "c", "family": "time469",
            "rows": rows, "k": k, "targets": m,
            "groups": g, "chunk": chunk, "specs": _specs(BREAKS), "decisions": _decisions(m, k), **more,
        }
        out.append(case)
    return out
