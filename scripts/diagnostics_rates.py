"""The false-alarm rates and detection counts ``docs/DIAGNOSTICS.md`` and
``docs/DATA-QUALITY.md`` cite for the break and tails diagnostics, measured
on seeded streams whose truth is written down.

Two pages once cited the same quantity from different runs (0.22% against
0.31% for ``|cusum_sq| > 3`` with no break, a mean kurtosis of 36 against
32, 2.5% against 3.5% for the run-once CUSUM); each figure now comes from
one experiment here, and the pages say which.

    uv run python scripts/diagnostics_rates.py all
    uv run python scripts/diagnostics_rates.py windowed

Experiments:

``windowed``
    200 streams of 25,000 rows, ``y = 0.5 + x0 - x1 + noise``, ``ewridge`` at
    a half-life of 200 with ``emit_breaks``, ``emit_tails`` and ``emit_drift``
    at its defaults, for Gaussian noise and Student's t at 10, 6, 5, 4 and 3
    degrees of freedom. Per noise: the mean and median of ``kurtosis`` over
    its rows, ``drift`` flags per 100,000 rows at thresholds of 20, 30, 50
    and 80, and the share of rows where ``|cusum| > 3``, ``|cusum_sq| > 3``
    and ``break_wald > 21.1`` (``chi2(3)``'s 0.01% value). A share is of the
    rows where the field is not null. ``over 2`` is ``|cusum_sq| > 3`` with
    a Gaussian's 2 in place of the measured ``E[z^4] - 1``, rebuilt from
    ``studentized`` by the definition, which is checked against the field.
``run_once``
    1,000 streams of 3,000 rows, ``ewridge`` run once (no decay, ridge 1e-12):
    no break, an intercept that rises half a noise sd at row 1,500, and
    ``x0``'s slope rising by half at row 1,500, all three from the same
    draws. Per case: the share of streams whose CUSUM path crosses Brown,
    Durbin and Evans' 5% boundary ``0.948 (sqrt(T) + 2r / sqrt(T))``, and
    the median rows from the break to the first crossing after it. Statsmodels'
    ``recursive_olsresiduals`` against its own boundary is the second
    opinion.
``drift_t3``
    10 streams of 200,000 rows of the ``docs/DIAGNOSTICS.md`` recipe's
    stable stretch (``y = x0 + 0.5 x1 + t(3)``, half-life 200): ``drift``
    flags per stream at the default threshold of 20.

``memories``
    The CUSUM's spread beside ten fits at breaks memories of 50, 200 and
    800 on 30 clean streams of 6,000 rows, from each stream's readiness on,
    and the share of 60 streams of 3,000 rows whose intercept rose half a
    noise sd at row 1,500 that ``|cusum| > 3`` and ``break_wald > 21.1``
    found (``holt``: 12,000 and 9,000 rows, the break at 6,000).
``breaks``
    200 streams of 3,000 rows beside ``ewridge`` at a half-life of 200, each
    broken at row 1,500 in one of the page's three ways (an intercept half
    a noise sd up, the noise doubled, ``x1``'s slope flipped): the share of
    streams each of ``|cusum| > 3``, ``|cusum_sq| > 3`` and ``break_wald >
    21.1`` fired on after the break, and the median delay.

``all`` runs them in that order. It took two minutes on an M4 Pro at a
load of 2.5 to 5.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_header import header  # noqa: E402

#: The noises of the windowed experiment: a name and a draw of n.
NOISES: dict[str, object] = {
    "Gaussian": lambda rng, n: rng.standard_normal(n),
    "t(10)": lambda rng, n: rng.standard_t(10, n),
    "t(6)": lambda rng, n: rng.standard_t(6, n),
    "t(5)": lambda rng, n: rng.standard_t(5, n),
    "t(4)": lambda rng, n: rng.standard_t(4, n),
    "t(3)": lambda rng, n: rng.standard_t(3, n),
}
THRESHOLDS = (20.0, 30.0, 50.0, 80.0)


def _windowed_specs() -> list[dict]:
    common = dict(targets=["y"], features=["x0", "x1"], half_life=200.0)
    specs = [
        po.spec.ewridge(
            "m20",
            emit_breaks=True,
            emit_tails=True,
            emit_drift=True,
            drift_threshold=THRESHOLDS[0],
            **common,
        )
    ]
    specs += [
        po.spec.ewridge(f"m{t:.0f}", emit_drift=True, drift_threshold=t, **common)
        for t in THRESHOLDS[1:]
    ]
    return specs


def _young_sq_share(s1, s2, w4, v1):
    """The CUSUM of squares' null share as far as its sums have filled, as
    the bank reads it: the sums' age ``x = kappa T`` from ``S1 / sum w_s =
    (1 + p)(1 + p^2) / 4``, ``p = e^(-x/4)``, filled below ``p = 1.5 / K``
    (``K`` the sums' Kish size), and the scale's lead ``D = q (1 - c)`` from
    its weights ``v1``: ``1/2 + (q - D) / (1 + q) + D^2 ln(N / D) / (1 -
    q^2)``, ``N = D + 1 - q``; 1 at the first row, 1/2 filled."""
    with np.errstate(invalid="ignore", divide="ignore"):
        target = 4.0 * s1 / w4
        p = np.ones_like(target)
        for _ in range(60):
            f = ((p + 1.0) * p + 1.0) * p + 1.0 - target
            p = np.minimum(p, p - f / ((3.0 * p + 2.0) * p + 1.0))
        x = -4.0 * np.log(p)
        x = np.where((target <= 1.0) | (p <= 1.5 * s2 / (s1 * s1)), np.inf, x)
        x = np.where(target >= 4.0, 0.0, x)
        q, m = np.exp(-x), -np.expm1(-x)
        d = np.clip(m * v1 / s1 - m, 0.0, q)
        tail = np.where(
            d > 0.0, d * d * np.log1p(m / np.where(d > 0, d, 1.0)) / (m * (1.0 + q)), 0.0
        )
        r2 = 0.5 + (q - d) / (1.0 + q) + tail
        r2 = np.where(x == 0.0, 1.0, np.where(np.isinf(x), 0.5, r2))
    return r2


def _over_two(z: np.ndarray, half_life: float) -> tuple[np.ndarray, np.ndarray]:
    """``cusum_sq`` rebuilt from the studentized residuals ``z`` by its
    definition, over the measured ``m4 - 1`` and over a Gaussian's 2: the
    sums of the rows before each row, decayed at ``half_life`` (the
    fourth moment at four times it), and the windowed null's share as far as
    the sums have filled (``_young_sq_share``). The scale the residuals are
    studentized by folded the rows before the first ``z`` whose Kish size
    first reached 10, which unit weights make the same count on every
    stream. The first is checked against the bank's field; the second is
    the column the bank does not write."""
    from scipy.signal import lfilter

    lam = 0.5 ** (1.0 / half_life)
    slow = lam**0.25
    ok = np.isfinite(z)
    zz = np.where(ok, z, 0.0)
    w = ok.astype(float)

    def ew(x: np.ndarray, f: float) -> np.ndarray:
        # The sum over the rows before each row: shifted by one.
        return np.concatenate([[0.0], lfilter([1.0], [1.0, -f], x)[:-1]])

    # The rows the scale read before the first z: the fewest unit-weight
    # rows decayed at `lam` whose Kish size reaches 10.
    ahead = 1
    while (sum(lam**j for j in range(ahead)) ** 2) < 10 * sum(lam ** (2 * j) for j in range(ahead)):
        ahead += 1
    folded = w.copy()
    if ok.any():
        first = int(np.argmax(ok))
        folded[max(first - ahead, 0) : first] = 1.0
    s1, s2, z2 = ew(w, lam), ew(w, lam * lam), ew(w * zz * zz, lam)
    w4, v1 = ew(w, slow), ew(folded, lam)
    share = _young_sq_share(s1, s2, w4, v1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m4 = ew(w * zz**4, slow) / w4
        measured = (z2 - s1) / np.sqrt(share * (m4 - 1.0) * s2)
        gaussian = (z2 - s1) / np.sqrt(share * 2.0 * s2)
    return measured, gaussian


def windowed(streams: int = 200, n: int = 25_000) -> None:
    print(f"\n## windowed: {streams} streams of {n:,} rows, half-life 200", flush=True)
    specs = _windowed_specs()
    print(
        "| noise | mean kurtosis | median kurtosis | drift per 100,000 at 20 | 30 | 50 | 80 "
        "| |cusum| > 3 | |cusum_sq| > 3 | over 2 | break_wald > 21.1 "
        "| per-stream |cusum_sq| > 3 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    rebuilt_gap = 0.0
    for name, draw in NOISES.items():
        kurt, flags = [], np.zeros(len(THRESHOLDS))
        hits = {"cusum": [0, 0], "cusum_sq": [0, 0], "break_wald": [0, 0], "over_two": [0, 0]}
        per_stream_sq = []
        for s in range(streams):
            rng = np.random.default_rng(1000 + s)
            x = rng.standard_normal((n, 2))
            y = 0.5 + x[:, 0] - x[:, 1] + draw(rng, n)  # type: ignore[operator]
            df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
            out = po.ModelBank(specs).fit_predict(df)
            m = out["m20"].struct.unnest()
            k = m["kurtosis_y"].drop_nulls().drop_nans()
            kurt.append(k.to_numpy())
            for i, t in enumerate(THRESHOLDS):
                flags[i] += out[f"m{t:.0f}"].struct.field("drift_y").sum()
            for field, cut in (("cusum", 3.0), ("cusum_sq", 3.0), ("break_wald", 21.1)):
                v = m[f"{field}_y"].drop_nulls().drop_nans()
                over = int((v.abs() > cut).sum())
                hits[field][0] += over
                hits[field][1] += v.len()
                if field == "cusum_sq":
                    per_stream_sq.append(over / max(v.len(), 1))
            measured, gaussian = _over_two(m["studentized_y"].to_numpy().astype(float), 200.0)
            field_sq = m["cusum_sq_y"].to_numpy().astype(float)
            both = np.isfinite(field_sq) & np.isfinite(measured)
            rebuilt_gap = max(rebuilt_gap, float(np.max(np.abs(field_sq[both] - measured[both]))))
            g = gaussian[np.isfinite(field_sq)]
            hits["over_two"][0] += int((np.abs(g) > 3.0).sum())
            hits["over_two"][1] += g.size
        allk = np.concatenate(kurt)
        per100k = flags / (streams * n) * 1e5
        share = {f: 100.0 * h[0] / h[1] for f, h in hits.items()}
        print(
            f"| {name} | {np.mean([k.mean() for k in kurt]):.1f} | {np.median(allk):.1f} | "
            + " | ".join(f"{v:.1f}" for v in per100k)
            + f" | {share['cusum']:.2f}% | {share['cusum_sq']:.2f}% | {share['over_two']:.2f}% "
            f"| {share['break_wald']:.3f}% "
            f"| {100 * min(per_stream_sq):.2f}%-{100 * max(per_stream_sq):.2f}% |",
            flush=True,
        )
    print(f"the rebuilt cusum_sq against the bank's field: largest difference {rebuilt_gap:.1e}")


def _crossing(z: np.ndarray) -> np.ndarray:
    """The rows where the CUSUM path of ``z`` is past Brown, Durbin and
    Evans' 5% boundary, over a run of as many rows as ``z`` has values."""
    ok = np.isfinite(z)
    path, r = np.cumsum(np.where(ok, z, 0.0)), np.cumsum(ok)
    t = r[-1]
    return ok & (np.abs(path) > 0.948 * (np.sqrt(t) + 2 * r / np.sqrt(t)))


def run_once(streams: int = 1_000, n: int = 3_000, at: int = 1_500) -> None:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import recursive_olsresiduals

    print(f"\n## run_once: {streams} streams of {n:,} rows, the break at row {at:,}", flush=True)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        ridge=1e-12,
        emit_breaks=True,
    )
    after = np.arange(n) >= at
    cases = {
        "no break": lambda x: 0.0,
        "intercept +0.5 sd": lambda x: 0.5 * after,
        "x0's slope +0.5": lambda x: 0.5 * x[:, 0] * after,
    }
    print("| case | crossed, ours | median first crossing after the break | crossed, statsmodels |")
    print("|---|---:|---:|---:|")
    for name, move in cases.items():
        ours, theirs, delays, their_delays = 0, 0, [], []
        for s in range(streams):
            rng = np.random.default_rng(5000 + s)
            x = rng.standard_normal((n, 2))
            y = 0.5 + x[:, 0] - x[:, 1] + rng.standard_normal(n) + move(x)
            df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
            z = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("studentized_y")
            crossed = _crossing(z.to_numpy().astype(float))
            ours += bool(crossed.any())
            if name != "no break" and crossed[at:].any():
                delays.append(int(np.argmax(crossed[at:])))
            X = sm.add_constant(x)
            rr = recursive_olsresiduals(sm.OLS(y, X).fit(), skip=3)
            # rcusum's entry i + 1 is the path through row 3 + i.
            past = np.abs(rr[5][1:]) > rr[6][1]
            theirs += bool(past.any())
            if name != "no break" and past[at - 3 :].any():
                their_delays.append(int(np.argmax(past[at - 3 :])))
        delay = f"{np.median(delays):.0f} rows" if delays else "-"
        if their_delays:
            delay += f" (statsmodels {np.median(their_delays):.0f})"
        print(
            f"| {name} | {100 * ours / streams:.1f}% | {delay} | {100 * theirs / streams:.1f}% |",
            flush=True,
        )


def _grouped(
    groups: int, n: int, seed: int, at: int | None = None, kind: str = "none"
) -> pl.DataFrame:
    """``groups`` streams of ``n`` rows, ``y = 0.5 + x0 - x1 + noise``, with
    one break at row ``at``: ``"mean"`` (the intercept rises half a noise
    sd), ``"variance"`` (the noise doubles) or ``"slope"`` (``y = x0 +
    0.5 x1 + noise``, ``x1``'s slope flipping sign), as the page's second
    recipe has them."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((groups * n, 2))
    e = rng.standard_normal(groups * n)
    after = (np.tile(np.arange(n), groups) >= at) if at is not None else np.zeros(groups * n, bool)
    if kind == "slope":
        y = x[:, 0] + np.where(after, -0.5, 0.5) * x[:, 1] + e
    else:
        y = 0.5 + 0.5 * (kind == "mean") * after + x[:, 0] - x[:, 1]
        y = y + e * np.where(after & (kind == "variance"), 2.0, 1.0)
    return pl.DataFrame(
        {"g": np.repeat(np.arange(groups), n), "x0": x[:, 0], "x1": x[:, 1], "y": y}
    )


#: The fits of the ``memories`` experiment and what each needs beyond the
#: target and the features.
MEMORY_FITS: dict[str, dict] = {
    "ewridge": {},
    "rls": {},
    "lasso": {"penalties": [0.001]},
    "kalman": {"coef_half_life": 200.0},
    "quantile": {"quantile": 0.5},
    "huber": {},
    "holt": {},
    "ftrl": {"loss": "squared"},
    "sgd": {},
    "pa": {},
}


def memories(clean: int = 30, broken: int = 60) -> None:
    """The CUSUM's spread beside each fit at breaks memories of 50, 200 and
    800, on ``clean`` streams of 6,000 rows from each stream's readiness
    on, and the share of ``broken`` streams of 3,000 rows, whose intercept
    rose half a noise sd at row 1,500, on which ``|cusum| > 3`` and
    ``break_wald > 21.1`` passed after it; ``holt``'s streams are 12,000
    and 9,000 rows, the break at 6,000, past its trend's readiness. And
    the most rows of the clean streams each flag held on beside any fit."""
    print(f"\n## memories: {clean} clean streams, {broken} broken, half-life 200", flush=True)
    print(
        "| model | spread of cusum, memory 50 / 200 / 800 | |cusum| > 3 found | break_wald found "
        "| spread of cusum_sq |"
    )
    print("|---|---|---|---|---|")
    held = {"cusum": 0.0, "break_wald": 0.0}
    for model, kw in MEMORY_FITS.items():
        n_clean, n_broken, at = (12_000, 9_000, 6_000) if model == "holt" else (6_000, 3_000, 1_500)
        quiet = _grouped(clean, n_clean, 4_100)
        moved = _grouped(broken, n_broken, 4_200, at, "mean")
        features = {} if model == "holt" else {"features": ["x0", "x1"]}
        spreads, sq_spreads, found, wald = [], [], [], []
        for memory in (50.0, 200.0, 800.0):
            spec = getattr(po.spec, model)(
                "m",
                targets=["y"],
                half_life=200.0,
                group="g",
                emit_breaks=True,
                breaks_half_life=memory,
                **features,
                **kw,
            )
            q = po.ModelBank([spec]).fit_predict(quiet)["m"].struct.unnest()
            c = q["cusum_y"].drop_nulls().drop_nans()
            sq = q["cusum_sq_y"].drop_nulls().drop_nans()
            spreads.append(f"{c.std():.2f}")
            sq_spreads.append(f"{sq.std():.2f}")
            held["cusum"] = max(held["cusum"], float((c.abs() > 3).mean()))
            bw = q["break_wald_y"].drop_nulls().drop_nans()
            if bw.len():
                held["break_wald"] = max(held["break_wald"], float((bw > 21.1).mean()))
            b = po.ModelBank([spec]).fit_predict(moved)["m"].struct.unnest()
            for field, cut, into in (("cusum_y", 3.0, found), ("break_wald_y", 21.1, wald)):
                v = np.nan_to_num(b[field].to_numpy().astype(float)).reshape(broken, n_broken)
                hit = (np.abs(v[:, at:]) > cut) if field == "cusum_y" else (v[:, at:] > cut)
                into.append(f"{100 * hit.any(axis=1).mean():.0f}%")
        print(
            f"| `{model}` | {' / '.join(spreads)} | {' / '.join(found)} | {' / '.join(wald)} "
            f"| {' / '.join(sq_spreads)} |",
            flush=True,
        )
    print(
        f"most clean rows past |cusum| > 3: {100 * held['cusum']:.2f}%, past break_wald > 21.1: "
        f"{100 * held['break_wald']:.2f}%"
    )


def breaks(streams: int = 200, n: int = 3_000, at: int = 1_500) -> None:
    """Beside ``ewridge`` at a half-life of 200, at its default memory, on
    ``streams`` streams of ``n`` rows that break at row ``at`` in each of
    the three ways (``_grouped``): the share of streams each flag passed on
    after the break, and the median rows from the break to its first."""
    print(
        f"\n## breaks: {streams} streams of {n:,} rows, a break at row {at:,}, half-life 200",
        flush=True,
    )
    print("| break | flag | streams it fired on | median delay |")
    print("|---|---|---:|---:|")
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], half_life=200.0, group="g", emit_breaks=True
    )
    for kind in ("mean", "variance", "slope"):
        m = (
            po.ModelBank([spec])
            .fit_predict(_grouped(streams, n, 4_300, at, kind))["m"]
            .struct.unnest()
        )
        for field, cut in (("cusum_y", 3.0), ("cusum_sq_y", 3.0), ("break_wald_y", 21.1)):
            v = np.nan_to_num(m[field].to_numpy().astype(float)).reshape(streams, n)[:, at:]
            hit = (np.abs(v) > cut) if field != "break_wald_y" else (v > cut)
            fired = hit.any(axis=1)
            delay = f"{np.median(hit.argmax(axis=1)[fired]):.0f} rows" if fired.any() else "-"
            print(f"| {kind} | {field} > {cut} | {100 * fired.mean():.1f}% | {delay} |", flush=True)


def drift_t3(streams: int = 10, n: int = 200_000) -> None:
    print(f"\n## drift_t3: {streams} streams of {n:,} rows, t(3) noise, threshold 20", flush=True)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        emit_drift=True,
        drift_threshold=20.0,
    )
    counts = []
    for s in range(streams):
        rng = np.random.default_rng(9000 + s)
        x0, x1 = rng.standard_normal((2, n))
        df = pl.DataFrame({"x0": x0, "x1": x1, "y": x0 + 0.5 * x1 + rng.standard_t(3, n)})
        counts.append(int(po.ModelBank([spec]).fit_predict(df)["m"].struct.field("drift_y").sum()))
    print(f"flags per stream: {sorted(counts)}; mean {np.mean(counts):.1f}", flush=True)


EXPERIMENTS = {
    "windowed": windowed,
    "run_once": run_once,
    "drift_t3": drift_t3,
    "memories": memories,
    "breaks": breaks,
}


def main(argv: list[str]) -> None:
    which = argv[1:] or ["all"]
    names = list(EXPERIMENTS) if which == ["all"] else which
    print(header(), flush=True)
    for name in names:
        start = time.perf_counter()
        EXPERIMENTS[name]()
        print(f"({name}: {time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv)
