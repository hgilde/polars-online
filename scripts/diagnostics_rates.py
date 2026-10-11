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


def _over_two(z: np.ndarray, half_life: float) -> tuple[np.ndarray, np.ndarray]:
    """``cusum_sq`` rebuilt from the studentized residuals ``z`` by its
    definition, over the measured ``m4 - 1`` and over a Gaussian's 2: the
    sums of the rows before each row, decayed at ``half_life`` (the
    fourth moment at four times it), and the windowed null's share 1/2.
    The first is checked against the bank's field; the second is the
    column the bank does not write."""
    from scipy.signal import lfilter

    lam = 0.5 ** (1.0 / half_life)
    slow = lam**0.25
    ok = np.isfinite(z)
    zz = np.where(ok, z, 0.0)
    w = ok.astype(float)

    def ew(x: np.ndarray, f: float) -> np.ndarray:
        # The sum over the rows before each row: shifted by one.
        return np.concatenate([[0.0], lfilter([1.0], [1.0, -f], x)[:-1]])

    s1, s2, z2 = ew(w, lam), ew(w, lam * lam), ew(w * zz * zz, lam)
    with np.errstate(invalid="ignore", divide="ignore"):
        m4 = ew(w * zz**4, slow) / ew(w, slow)
        measured = (z2 - s1) / np.sqrt(0.5 * (m4 - 1.0) * s2)
        gaussian = (z2 - s1) / np.sqrt(0.5 * 2.0 * s2)
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


EXPERIMENTS = {"windowed": windowed, "run_once": run_once, "drift_t3": drift_t3}


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
