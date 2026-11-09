"""The conformal tracker's start under a label delay, and the candidates
measured for it (docs/records/CONFORMAL-STEP.md §8).

    .venv/bin/python -I scripts/conformal_step/startup.py cache     # the banks, once
    .venv/bin/python -I scripts/conformal_step/startup.py measure   # every rule, a minute
    .venv/bin/python -I scripts/conformal_step/startup.py tables    # §8's tables

`cache` runs every stream of §4's grid through the bank once, with
`emit_sigma`, and the streams §8 adds: the generated stream with each label
late by a delay drawn uniformly from 50 to 300 rows ("spread"), and the
regimes stream of the highlights page's monitor task at three delays, two
half-lives and two seeds, beside its 8,000-row form. The tracker does not
reach the prediction, so one run fixes every residual; each rule is then a
fold over them in the bank's release order (`fold.schedule`).

The rules: the tracker on the radius the bank kept before (`fold._track`'s
fixed and "auto" rules) and its candidates -- the start waiting for a count
of residuals (`wait`), and the gain read at the current radius (`gs`) --
and the multiplier the bank keeps now (`fold.track_m`), at six fixed rates
and under "auto", with and without the wait. `measure` checks first that the
multiplier's rule at 0.05 is the bank's `lo` to the bit on every stream.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold  # noqa: E402
import streams  # noqa: E402

import polars_online as po  # noqa: E402

OUT = streams.CACHE / "startup"
OUT.mkdir(parents=True, exist_ok=True)
FIXED = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
TOL = {0.9: 0.005, 0.5: 0.01, 0.99: 0.002}

# ------------------------------------------------------------------ the streams


def spread(seed, noise, rate):
    """`streams.generated` with each label late by U(50, 300) rows."""
    rng_cls = np.random.default_rng

    def patched(s=None):
        g = rng_cls(s)
        if s == 2000 + seed:  # the delays' generator: its exponential, uniform

            class Uniform:
                def exponential(self, _scale, n):
                    return g.uniform(50 * streams.GAP, 300 * streams.GAP, n)

            return Uniform()
        return g

    old = streams.DELAYS
    try:
        np.random.default_rng = patched
        streams.DELAYS = {**old, "spread": -1.0}
        return streams.generated(seed, noise, rate, "spread")
    finally:
        np.random.default_rng = rng_cls
        streams.DELAYS = old


def regimes(seed, lo, hi, kind, half_life, rows_per_block=5_000):
    """The highlights page's monitor stream: two features whose correlation
    takes turns, a slope that flips half-way, Student's t noise, and each
    label late by U(lo, hi) rows, or an exponential delay of mean `lo`."""
    sim = po.sim.regimes(
        2,
        states=[0.0, 0.7],
        transition=[[0.0, 1.0], [1.0, 0.0]],
        n_blocks=4,
        rows_per_block=rows_per_block,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    n = sim["rows"].height
    slope = pl.when(pl.col("t") < 2 * rows_per_block).then(0.5).otherwise(-0.5)
    noise = rng.standard_t(3, n) / np.sqrt(3.0)
    late = rng.uniform(lo, hi, n) if kind == "uniform" else rng.exponential(lo, n)
    frame = (
        sim["rows"]
        .with_columns(pl.col("x_1", "x_2").diff().fill_null(0.0))
        .with_columns(
            y=pl.col("x_1") + slope * pl.col("x_2") + pl.lit(noise),
            avail=pl.col("t").cast(pl.Float64) + pl.lit(late),
            t=pl.col("t").cast(pl.Float64),
        )
        .select("t", "avail", "x_1", "x_2", "y")
    )
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x_1", "x_2"],
        clock="t",
        gap_cap=10.0,
        half_life=half_life,
        label_clock="avail",
        conformal=0.9,
        emit_sigma=True,
    )
    return streams.run(frame, spec, half_life, 10.0, np.arange(n) >= n // 10)


def cases():
    for d in streams.FLIGHT_DELAYS:
        yield f"flights|{d}", lambda d=d: streams.flights(d)
    for seed in (0, 1, 2):
        for nz in streams.NOISES:
            for rt in streams.RATES:
                for d in streams.DELAYS:
                    yield (
                        f"generated|{seed}|{nz}|{rt}|{d}",
                        lambda a=(seed, nz, rt, d): streams.generated(*a),
                    )
                yield (
                    f"generated|{seed}|{nz}|{rt}|spread",
                    lambda a=(seed, nz, rt): spread(*a),
                )
    for seed in (4, 5):
        for lo, hi, kind in (
            (50.0, 300.0, "uniform"),
            (250.0, 350.0, "uniform"),
            (175.0, 0, "exp"),
        ):
            for hl in (500.0, 2000.0):
                yield (
                    f"regimes|{seed}|{kind}{lo:.0f}-{hi:.0f}|{hl:.0f}",
                    lambda a=(seed, lo, hi, kind, hl): regimes(*a),
                )
    for hl in (500.0, 2000.0):
        yield (
            f"regimes8k|4|uniform50-300|{hl:.0f}",
            lambda hl=hl: regimes(4, 50.0, 300.0, "uniform", hl, 2_000),
        )


def fname(name):
    return OUT / (name.replace("|", "__").replace(" ", "") + ".npz")


def cache():
    for name, make in cases():
        s = make()
        keep = ("t", "avail", "resid", "pred", "w", "lo", "sigma", "mask")
        np.savez(
            fname(name),
            name=name,
            half_life=s["half_life"],
            gap_cap=s["gap_cap"],
            **{k: s[k] for k in keep},
        )
        print(name, flush=True)


# ------------------------------------------------- the radius tracker's candidates


@nb.njit(cache=True)
def _radius(
    n,
    rows,
    at,
    lam,
    sigma,
    released_before,
    held_before,
    resid,
    w,
    alpha,
    rate,
    auto,
    c,
    g,
    zq,
    wait,
    m_rows,
    gs,
):
    """The tracker on the radius (the bank's before task 368), with the
    candidates of §8. `wait`: 0 none; 1 the start waits for `m_rows`
    residuals, decayed (`w_rows`), under every rate; 2 the same once rows
    wait; 3 for `1 + nbar` decayed; 4 for `1 + 2 nbar`; 5 for `m_rows +
    nbar`; 7 for `1 + nbar` counted without decay. `gs`: 1 the gain read at
    the larger of `z_alpha` and the current radius in sigma units, 2 at the
    radius shown."""
    shown = np.full(n, np.nan)
    m = len(rows)
    q = np.nan
    w_mean = 0.0
    w_rows = 0.0
    dbar = 0.0
    given = 0.0
    k = 0
    for i in range(n):
        while k < m and at[k] == i:
            j = rows[k]
            r = resid[j]
            lm = lam[k]
            s = sigma[k]
            dk = held_before[i] - (k - released_before[i]) - 1
            if not math.isfinite(r):
                w_rows *= lm
                k += 1
                continue
            w_rows = lm * w_rows + 1.0
            w_mean += (w[j] - w_mean) / w_rows
            if auto and (dk > 0 or dbar != 0.0):
                dbar += (dk - dbar) / w_rows
            usable = math.isfinite(s) and s > 0.0
            if not math.isfinite(q):
                given += 1.0
                ready = True
                if wait == 1:
                    ready = w_rows >= m_rows
                elif wait == 2:
                    ready = dbar == 0.0 or w_rows >= m_rows
                elif wait == 3:
                    ready = w_rows >= 1.0 + dbar
                elif wait == 4:
                    ready = w_rows >= 1.0 + 2.0 * dbar
                elif wait == 5:
                    ready = dbar == 0.0 or w_rows >= m_rows + dbar
                elif wait == 7:
                    ready = given >= 1.0 + dbar
                if usable and ready:
                    q = s * zq
                k += 1
                continue
            against = shown[j]
            if not math.isfinite(against):
                k += 1
                continue
            miss = 1.0 if abs(r) > against else 0.0
            eta = rate * s if usable else 0.0
            if dbar > 0.0:
                gg = g
                if gs > 0 and usable:
                    x = (q if gs == 1 else against) / s
                    if x > zq:
                        gg = 0.7978845608028654 * math.exp(-0.5 * x * x)
                eta = eta / (1.0 + rate * gg * dbar / c)
            rel = w[j] / w_mean if w_mean > 0.0 else 1.0
            q = max(q + eta * rel * (miss - alpha), 0.0)
            k += 1
        shown[i] = q
    return shown


def radius(sc, resid, w, coverage, auto=False, wait=0, m_rows=0.0, gs=0, rate=0.05, c=0.5):
    alpha = 1.0 - coverage
    return _radius(
        len(resid), sc["row"], sc["at"], sc["lam"], sc["sigma"], sc["released_before"],
        sc["held_before"], resid, w, alpha, rate, auto, c, fold.gain_bits(alpha),
        fold.norm_ppf(1.0 - 0.5 * alpha, bank=True), wait, float(m_rows), gs,
    )  # fmt: skip


def m_rule(coverage, c=0.5):
    """A wait from the radius's sampling error: `z / sqrt(2 m)` sigma from
    `m` Gaussian residuals, held under `alpha c / g`, the error the loop
    removes within its time constant."""
    a = 1.0 - coverage
    z, g = fold.norm_ppf(1.0 - 0.5 * a, bank=True), fold.gain_bits(a)
    return 0.5 * (z * g / (a * c)) ** 2


def rules(coverage):
    """Each rule as `(name, the radius it shows each row)`, given the stream."""
    m_a = m_rule(coverage)
    for r in FIXED:
        yield f"fixed {r}", lambda sc, s, r=r: radius(sc, s["resid"], s["w"], coverage, rate=r)
    for name, kw in (
        ("auto", dict(auto=True)),
        ("auto+gs", dict(auto=True, gs=1)),
        ("auto+gs shown", dict(auto=True, gs=2)),
        ("auto+wait30", dict(auto=True, wait=2, m_rows=30)),
        ("auto+wait m(a)", dict(auto=True, wait=2, m_rows=m_a)),
        ("auto+wait nbar decayed", dict(auto=True, wait=3)),
        ("auto+wait 2nbar", dict(auto=True, wait=4)),
        ("auto+wait nbar", dict(auto=True, wait=7)),
        ("fixed 0.05+wait30", dict(wait=1, m_rows=30)),
    ):
        yield name, lambda sc, s, kw=kw: radius(sc, s["resid"], s["w"], coverage, **kw)
    for r in FIXED:
        yield (
            f"M fixed {r}",
            lambda sc, s, r=r: fold.track_m(sc, s["resid"], s["w"], s["sigma"], coverage, rate=r),
        )
    yield (
        "M auto",
        lambda sc, s: fold.track_m(sc, s["resid"], s["w"], s["sigma"], coverage, auto=True),
    )


def first_in_band(shown, resid, coverage, tol, window):
    """Rows from the first interval to the end of the first trailing window
    of `window` scored rows whose coverage is within `tol` of the target."""
    idx = np.nonzero(np.isfinite(shown) & np.isfinite(resid))[0]
    if len(idx) < window:
        return math.nan
    cs = np.concatenate([[0.0], np.cumsum(np.abs(resid[idx]) <= shown[idx])])
    good = np.nonzero(np.abs((cs[window:] - cs[:-window]) / window - coverage) <= tol)[0]
    return float(idx[good[0] + window - 1] - idx[0]) if len(good) else float(len(shown))


def measure():
    out, bad = [], []
    for f in sorted(OUT.glob("*.npz")):
        s = dict(np.load(f))
        name = str(s["name"])
        sc = fold.schedule(s["t"], s["avail"], s["resid"], s["w"], float(s["half_life"]),
                           float(s["gap_cap"]))  # fmt: skip
        if not np.array_equal(s["pred"] - fold.track_m(sc, s["resid"], s["w"], s["sigma"], 0.9),
                              s["lo"], equal_nan=True):  # fmt: skip
            bad.append(name)
        kind = name.split("|")[0]
        levels = (0.9, 0.5, 0.99) if kind != "generated" or name.split("|")[1] == "0" else (0.9,)
        held = float(sc["held_before"][s["mask"]].mean())
        for cov in levels:
            tol, win = {0.9: (0.01, 1000), 0.5: (0.01, 1000), 0.99: (0.002, 5000)}[cov]
            for rule, f_ in rules(cov):
                shown = f_(sc, s)
                m = fold.score(shown, s["resid"], s["mask"])
                ok = ~s["mask"] & np.isfinite(shown) & np.isfinite(s["resid"])
                hits = np.abs(s["resid"][ok]) <= shown[ok]
                m["early_coverage"] = float(hits.mean()) if ok.any() else math.nan
                m["early_half_width"] = float(shown[ok].mean()) if ok.any() else math.nan
                m["startup_rows"] = first_in_band(shown, s["resid"], cov, tol, win)
                out.append({"stream": name, "level": cov, "rule": rule, "held": held, **m})
        print(name, flush=True)
    (OUT / "results.json").write_text(json.dumps(out), encoding="utf-8")
    print("the multiplier at 0.05 is not the bank's lo on:", bad)
    sys.exit(1 if bad else 0)


# ------------------------------------------------------------------ the tables


def tables():
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    by = defaultdict(dict)
    for x in rows:
        by[(x["stream"], x["level"])][x["rule"]] = x
    names = list(dict.fromkeys(x["rule"] for x in rows))
    reference = [f"fixed {r}" for r in FIXED]

    def verdict(cell, level, r, pre=""):
        ok = [cell[f][pre + "half_width"] for f in reference
              if abs(cell[f][pre + "coverage"] - level) <= TOL[level]]  # fmt: skip
        best = min(ok) if ok else math.nan
        c, h = cell[r][pre + "coverage"], cell[r][pre + "half_width"]
        if abs(c - level) > TOL[level]:
            return "cov"
        return "wide" if best == best and h > 1.10 * best else "holds"

    def grid(s):
        p = s.split("|")
        return p[0] == "flights" or (p[0] == "generated" and p[4] != "spread")

    def spread_delays(s):
        return s.endswith("spread") or s.startswith("regimes")

    for select, title in ((grid, "§4's grid"), (spread_delays, "spread delays")):
        for pre in ("", "late_"):
            print(f"\n### holds, {title}{', second half' if pre else ''}: 90% / 50% / 99%")
            print("| rule | holds | lost against today's auto |")
            print("|---|---:|---:|")
            for r in names:
                cells, lost = [], []
                for lv in (0.9, 0.5, 0.99):
                    ks = [k for k in by if k[1] == lv and select(k[0])]
                    v = [verdict(by[k], lv, r, pre) for k in ks]
                    cells.append(f"{v.count('holds')}/{len(ks)}")
                    lost.append(sum(verdict(by[k], lv, "auto", pre) == "holds" and x != "holds"
                                    for k, x in zip(ks, v, strict=True)))  # fmt: skip
                print(f"| {r} | {' / '.join(cells)} | {'/'.join(map(str, lost))} |")
    for d in ("none", "short", "long", "variable", "spread"):
        print(f"\n### start-up, generated, {d}, 90%: rows to the band (median / max), "
              "first 10%: coverage / half-width")  # fmt: skip
        for r in names:
            xs = [by[k][r] for k in by if k[1] == 0.9 and k[0].startswith("generated")
                  and k[0].split("|")[4] == d]  # fmt: skip
            su = sorted(x["startup_rows"] for x in xs)
            ec = sum(x["early_coverage"] for x in xs) / len(xs)
            eh = sum(x["early_half_width"] for x in xs) / len(xs)
            print(f"| {r} | {su[len(su) // 2]:.0f} / {su[-1]:.0f} | {ec:.4f} / {eh:.3f} |")
    print("\n### second-half coverage off the target by more than the holds tolerance: streams")
    for r in names:
        cells = []
        for lv in (0.9, 0.5, 0.99):
            ks = [k for k in by if k[1] == lv]
            cells.append(str(sum(abs(by[k][r]["late_coverage"] - lv) > TOL[lv] for k in ks)))
        print(f"| {r} | {' / '.join(cells)} |")


if __name__ == "__main__":
    {"cache": cache, "measure": measure, "tables": tables}[sys.argv[1]]()
