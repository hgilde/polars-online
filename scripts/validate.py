"""Run the docs/PLAN.md [validate] experiments and print a markdown report.

Usage: uv run python scripts/validate.py [--rows N]

Uses the public intraday dataset (cached under .cache/) when available and the
seeded synthetic generator otherwise, and writes nothing: pipe stdout into
docs/VALIDATION.md.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from functools import reduce
from operator import add
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import polars_online as po  # noqa: E402
from data import VALIDATION_DATES as DATES  # noqa: E402
from data import public_intraday, synthetic  # noqa: E402

#: Every model's memory, in rows of the row clock: `half_life` for each, and
#: `coef_half_life` for kalman's coefficients too (task 212).
HL = 500.0

#: Each target's horizon: the sum of the log returns of the next `k` rows.
#: Built by `ahead`, so the target and its embargo come from one number.
HORIZONS = {"y0": 1, "y1": 5}


def ahead(k: int) -> pl.Expr:
    """The sum of the next ``k`` rows' log returns. It reads rows ``t + 1``
    to ``t + k``, so it is known ``k`` rows after its own: its embargo."""
    return reduce(add, [pl.col("lr").shift(-i) for i in range(1, k + 1)])


def zscores(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Each feature in units of its current spread, ``(x - ewm_mean(x)) /
    ewm_std(x)`` at the models' half-life, as columns ``z_<name>``: the
    per-standard-deviation recipe (README, *Features in units of their
    spread*). Streaming window operators, so nothing looks ahead."""
    z = {
        f"z_{f}": (pl.col(f) - po.ewm_mean(f, half_life=HL)) / po.ewm_std(f, half_life=HL)
        for f in feats
    }
    return po.stream.with_windows(df, **z)


def load(rows: int) -> tuple[pl.DataFrame, str, list[str], list[str], dict[str, float | None]]:
    """Returns (df, source, features, targets, embargo per target)."""
    try:
        raw = public_intraday(DATES)
        source = (
            f"public intraday (BTCUSDT 1m, Binance public dump, "
            f"{DATES[0]}..{DATES[-1]}, {len(DATES)} days)"
        )
    except RuntimeError:
        df, _ = synthetic(
            seed=1234, n_groups=1, n_rows=rows or 5000, k=4, n_targets=2, null_frac=0.0
        )
        feats = ["x0", "x1", "x2", "x3"]
        df = zscores(df, feats).drop_nulls([f"z_{f}" for f in feats])
        # The synthetic targets are of their own row: nothing to wait for.
        return df, "synthetic (offline fallback)", feats, ["y0", "y1"], {"y0": None, "y1": None}

    # Simple, honest intraday features: past returns and volume/activity
    # signals. Targets are future returns, each learned under the embargo
    # its horizon needs, so none is learned before it is known.
    feats = ["x0", "x1", "x2", "x3"]
    df = (
        (raw.head(rows) if rows > 0 else raw)
        .with_columns(
            lr=(pl.col("close") / pl.col("close").shift(1)).log(),
            rng=(pl.col("high") - pl.col("low")) / pl.col("close"),
            vol_z=(pl.col("volume") - pl.col("volume").rolling_mean(60))
            / pl.col("volume").rolling_std(60),
            trades_z=(pl.col("n_trades") - pl.col("n_trades").rolling_mean(60))
            / pl.col("n_trades").rolling_std(60),
        )
        .with_columns(
            x0=pl.col("lr"),
            x1=pl.col("lr").rolling_sum(5),
            x2=pl.col("vol_z"),
            x3=pl.col("trades_z"),
            **{t: ahead(k) for t, k in HORIZONS.items()},
        )
        .drop_nulls([*feats, *HORIZONS])
        .with_columns(group=pl.lit("BTCUSDT"))
    )
    # The z-scores' first row has one value and no spread: null, dropped,
    # so every section reads the same rows.
    df = zscores(df, feats).drop_nulls([f"z_{f}" for f in feats])
    embargo: dict[str, float | None] = {t: float(k) for t, k in HORIZONS.items()}
    return df, source, feats, list(HORIZONS), embargo


def run(df: pl.DataFrame, specs: list[dict], targets: list[str]) -> pl.DataFrame:
    bank = po.ModelBank(specs)
    out = bank.fit_predict(df)
    return po.eval.compare_specs(out, [s["name"] for s in specs], targets=targets)


def section(title: str, body: str) -> None:
    print(f"\n## {title}\n")
    print(body)


def table(df: pl.DataFrame, cols: list[str]) -> str:
    df = df.select(cols)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for r in df.iter_rows():
        cells = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def matched(target: str, feats: list[str], embargo: float | None, suffix: str = "") -> list[dict]:
    """The models of sections 5 and 6 on one target at matched memory: each
    at half-life `HL`, kalman's coefficients too, and `ewridge` both at its
    default solve cadence and solved every row."""
    one = dict(targets=[target], features=feats, half_life=HL, min_weight=50.0, embargo=embargo)
    return [
        po.spec.ewridge(f"ewridge{suffix}", ridge=1e-4, standardize=True, **one),
        po.spec.ewridge(
            f"ewridge_every_row{suffix}", ridge=1e-4, standardize=True, solve_every=1.0, **one
        ),
        po.spec.rls(f"rls{suffix}", delta=1e-4, **one),
        po.spec.kalman(f"kalman{suffix}", coef_half_life=HL, **one),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=0, help="cap the row count (0 = use everything)")
    args = ap.parse_args()

    df, source, feats, targets, embargo = load(args.rows)
    target = targets[0]
    common = dict(
        targets=[target],
        features=feats,
        half_life=HL,
        min_weight=50.0,
        standardize=True,
        embargo=embargo[target],
    )

    data = (
        "a public minute-bar dataset"
        if source.startswith("public")
        else "a seeded synthetic dataset"
    )
    print("<!-- kind: reference -->")
    print("# Validation results")
    print()
    print(
        f"This reference page shows the experiments behind several defaults, run on {data}, "
        "for anyone asking why a default is what it is."
    )
    print()
    print("To run these experiments again:")
    print()
    print("```sh")
    print("uv run python scripts/validate.py > docs/VALIDATION.md")
    print("```")
    print()
    print(f"- Data: {source}")
    print(f"- Rows: {df.height}, features: {feats}, targets: {targets}")
    print(f"- Polars {pl.__version__}, polars-online {po.__version__}")
    section(
        "How the runs are set",
        "The clock is the row count: a row is a minute, and every number in clock units "
        "on this page is a count of rows.\n\n"
        "| rule | setting |\n"
        "|---|---|\n"
        "| **no target is learned before it is known** | `y0` is the next row's log return "
        "and `y1` the sum of the next five, so each is known 1 and 5 rows after its own row; "
        "each spec takes that as its `embargo`, and a spec with both targets (section 4) the "
        "longer, 5 |\n"
        f"| **the models are compared at matched memory** | every model at `half_life` = {HL:g} "
        f"rows, and `kalman`'s `coef_half_life` = {HL:g} too; `ewridge` and `lasso` solve at "
        "their default cadence, once the weight learned since the last solve reaches "
        "`ln 2 / 50` of the fit's (every 10 rows at this half-life), so their coefficients "
        "lag a row-by-row fit; section 5 runs `ewridge` solved every row beside it |\n"
        "| **`ewridge` is standardized in sections 1, 5 and 6** | each `ewridge` there runs "
        "with `standardize=True`, so section 1's default schedule is section 2's `std` row |\n"
        "| **the features** | `x0` the row's log return, `x1` the last five's sum, `x2` and `x3` "
        "volume and trade count against their last 60 rows' mean and spread; section 6 also "
        "runs each as a z-score, `(x - po.ewm_mean(x)) / po.ewm_std(x)` at the same half-life |\n"
        "| **the scores** | from `po.eval.compare_specs`: `n` the rows scored, `r2` the "
        "out-of-sample R² against the target's mean over those rows, `ic` the correlation of "
        "the prediction with the target, `hit_rate` the share of rows whose sign it got "
        "right, `mse` the mean squared residual |\n\n"
        "Every prediction is out-of-sample: each row is predicted before the model learns its "
        "target. A negative `r2` on financial data is normal. It means the model does worse "
        "than the target's mean over the rows scored, a mean no model could know in advance, "
        "not that anything is broken.",
    )

    # ---- 1. solve schedule: is half-life/50 the right cadence? ----
    # The default is by weight (docs/PLAN.md task 115 (b)): a solve once the
    # weight learned since the last reaches ln 2 / 50 of the weight the fit
    # holds, which on evenly spaced rows is every half-life/50 of clock. The
    # divisors measure the clock cadences; the default is measured beside.
    divisors = [1, 5, 10, 50, 200, 1000]
    specs = [po.spec.ewridge("default", ridge=1e-4, **common)]
    for d in divisors:
        specs.append(po.spec.ewridge(f"s{d}", ridge=1e-4, solve_every=HL / d, **common))
    t0 = time.perf_counter()
    res = run(df, specs, [target])
    elapsed = time.perf_counter() - t0
    default = res.filter(pl.col("spec") == "default").row(0, named=True)
    res = (
        res.filter(pl.col("spec") != "default")
        .with_columns(divisor=pl.col("spec").str.strip_prefix("s").cast(pl.Int64))
        .sort("divisor")
    )
    best = res.sort("mse").row(0, named=True)
    # Every sentence of the reading is computed from the numbers it states,
    # so it says nothing the table does not show.
    gap = 100.0 * (default["mse"] - best["mse"]) / default["mse"]
    rel = {
        d: 100.0 * (m - default["mse"]) / default["mse"]
        for d, m in zip(res["divisor"], res["mse"], strict=True)
    }
    near = [d for d, r in rel.items() if abs(r) <= 1.0]
    reading = ""
    if default["mse"] > best["mse"]:
        reading += f" Divisor {best['divisor']}'s MSE is {gap:.1f}% below the default's."
    if near:
        names = ", ".join(str(d) for d in near[:-1]) + (" and " if len(near) > 1 else "")
        names += str(near[-1])
        # Rounded up, so the bound holds of every divisor it names.
        spread = max(0.1, math.ceil(10.0 * max(abs(rel[d]) for d in near)) / 10.0)
        reading += (
            f" {'Divisors' if len(near) > 1 else 'Divisor'} {names} "
            f"{'score' if len(near) > 1 else 'scores'} within {spread:.1f}% of the "
            "default's MSE."
        )
    worse = [d for d, r in rel.items() if r > 1.0]
    if worse:
        names = ", ".join(str(d) for d in worse[:-1]) + (" and " if len(worse) > 1 else "")
        names += str(worse[-1])
        pcts = ", ".join(f"{rel[d]:.1f}%" for d in worse[:-1]) + (" and " if len(worse) > 1 else "")
        pcts += f"{rel[worse[-1]]:.1f}%"
        reading += (
            f" {'Divisors' if len(worse) > 1 else 'Divisor'} {names} "
            f"{'score' if len(worse) > 1 else 'scores'} {pcts} worse than the default."
        )
    if (res["r2"] < 0).all() and default["r2"] < 0:
        reading += (
            f" Every schedule's r2 is below 0 here, so none predicts `{target}` better than "
            "its mean over the rows scored."
        )
    n_schedules = len(divisors) + 1
    section(
        "1. Solve schedule (`solve_every` default = by weight, half_life/50 in steady state)",
        f"Solving every `half_life/d` clock units, half_life = {HL}, on `{target}`. "
        f"The {n_schedules} schedules run as {n_schedules} specs in one pass over the rows "
        f"({elapsed:.2f}s).\n\n"
        + table(res, ["divisor", "n", "r2", "ic", "hit_rate", "mse"])
        + f"\n\n**Result:** lowest MSE at divisor {best['divisor']} "
        f"(mse {best['mse']:.6g}). The default, a solve once the weight learned "
        f"since the last reaches `ln 2 / 50` of the weight the fit holds: "
        f"mse {default['mse']:.6g}, r2 {default['r2']:.6g}, ic {default['ic']:.6g}"
        + (", below every clock schedule's" if default["mse"] < best["mse"] else "")
        + ". Until a clock schedule's first period has passed, `half_life/d` "
        "from the stream's start, the model solves on the default schedule, so even "
        "the longest cadence does not hold the fit of its first few rows for that "
        "whole period. After it, each schedule solves on its own clock." + reading,
    )

    # ---- 2. standardize default ----
    specs = [
        po.spec.ewridge("plain", ridge=1e-4, **{**common, "standardize": False}),
        po.spec.ewridge("std", ridge=1e-4, **{**common, "standardize": True}),
    ]
    res = run(df, specs, [target])
    plain = res.filter(pl.col("spec") == "plain")["mse"][0]
    std = res.filter(pl.col("spec") == "std")["mse"][0]
    section(
        "2. `ewridge`'s `standardize` default (`False`)",
        table(res, ["spec", "n", "r2", "ic", "mse"])
        + f"\n\n**Result:** the default, `standardize=False`, scores an MSE of {plain:.6g} "
        f"against {std:.6g} standardized, so this table "
        + ("supports" if plain <= std else "does not support")
        + " the default.",
    )

    # ---- 3. elastic net l1_ratio [validate] ----
    # l1_ratio only bites where the penalty is nonzero, so the comparison is
    # made at the penalized path points, not at lambda = 0.
    path = [1e-2, 1e-3, 1e-4, 0.0]
    specs = [
        po.spec.lasso(
            f"l1_{int(r * 100)}",
            penalties=path,
            l1_ratio=r,
            targets=[target],
            features=feats,
            half_life=HL,
            min_weight=50.0,
            embargo=embargo[target],
        )
        for r in (1.0, 0.5, 0.1)
    ]
    res = run(df, specs, [target]).filter(~pl.col("slot").str.ends_with("__p0"))
    # A tie is a tie in the six digits the table shows.
    shown = res.with_columns(mse=pl.Series([float(f"{v:.6g}") for v in res["mse"]]))
    lead = shown.filter(pl.col("spec") == "l1_100").join(
        shown.group_by("slot").agg(best=pl.col("mse").min()), on="slot"
    )
    wins = lead.filter(pl.col("mse") <= pl.col("best")).height
    section(
        "3. Elastic net `l1_ratio`",
        "Lasso path "
        + str(path)
        + ". `l1_ratio` = 1 is pure lasso; below 1 the penalty is part ridge. "
        "The lambda = 0 slot is excluded because `l1_ratio` cannot affect it.\n\n"
        + table(res, ["spec", "slot", "n", "r2", "ic", "mse"])
        + f"\n\n**Result:** `l1_ratio=1`, the default, has the lowest MSE, or one equal to "
        f"it in six digits, at {wins} of {lead.height} penalties"
        + (
            ", so the ridge part of an elastic net does not help here."
            if wins == lead.height
            else "."
        ),
    )

    # ---- 4. share_p [validate] ----
    # One spec holds both targets, so it takes the longer horizon's embargo.
    both = [e for e in (embargo[t] for t in targets) if e is not None]
    kal_multi = dict(
        targets=targets,
        features=feats,
        coef_half_life=HL,
        half_life=HL,
        min_weight=50.0,
        embargo=max(both) if both else None,
    )
    specs = [
        po.spec.kalman("per_target_p", share_p=False, **kal_multi),
        po.spec.kalman("shared_p", share_p=True, **kal_multi),
    ]
    t0 = time.perf_counter()
    res = run(df, specs, targets)
    share_elapsed = time.perf_counter() - t0

    def scores(name: str) -> list[str]:
        return [f"{v:.6g}" for v in res.filter(pl.col("spec") == name).sort("slot")["mse"]]

    same = scores("per_target_p") == scores("shared_p")
    section(
        "4. Kalman `share_p` approximation",
        f"Two targets (`{targets[0]}` and `{targets[1]}`) with very different noise "
        f"levels. One spec holds both, so both wait the longer embargo. "
        f"`P` is held in units of the noise, so with both targets on every row the "
        f"shared `P` is each target's own and the two specs score the same: sharing "
        f"saves work and changes no prediction. Both specs run in {share_elapsed:.2f}s total.\n\n"
        + table(res, ["spec", "slot", "target", "n", "r2", "ic", "mse"])
        + (
            "\n\n**Result:** the two specs score the same in every digit shown, so the "
            "default, `share_p=False`, costs no accuracy, only the work of a second `P`."
            if same
            else "\n\n**Result:** the two specs score differently on at least one target."
        ),
    )

    # ---- 5. model comparison at matched settings ----
    frames = []
    for t in targets:
        specs = matched(t, feats, embargo[t])
        specs.append(
            po.spec.lasso(
                "lasso",
                penalties=[1e-3, 1e-4, 0.0],
                targets=[t],
                features=feats,
                half_life=HL,
                min_weight=50.0,
                embargo=embargo[t],
            )
        )
        frames.append(run(df, specs, [t]))
    res = pl.concat(frames)
    leaders = [res.filter(pl.col("target") == t).sort("mse").row(0, named=True) for t in targets]
    led = ", and ".join(f"on `{r['target']}` `{r['spec']}`'s `{r['slot']}`" for r in leaders)

    def mse_of(spec: str, t: str) -> float:
        return res.filter((pl.col("spec") == spec) & (pl.col("target") == t))["mse"][0]

    cadence = sum(mse_of("ewridge", t) < mse_of("ewridge_every_row", t) for t in targets)
    reading = (
        f"**Result:** the lowest MSE is {led}. `ewridge` at its default cadence scores a "
        f"lower MSE than solved every row on {cadence} of {len(targets)} targets"
        + (", so the default cadence costs no accuracy here." if cadence == len(targets) else ".")
    )
    if (res["r2"] < 0).all():
        reading += (
            " Every r2 is below 0: no model predicts either target better than its mean "
            "over the rows scored."
        )
    section(
        "5. Models at matched settings",
        "Each target under its own embargo, every model at the same memory "
        "([How the runs are set](#how-the-runs-are-set)).\n\n"
        + table(res, ["spec", "slot", "target", "n", "r2", "ic", "hit_rate", "mse"])
        + "\n\n"
        + reading,
    )

    # ---- 6. the per-standard-deviation recipe ----
    zfeats = [f"z_{f}" for f in feats]
    frames = []
    for t in targets:
        specs = matched(t, feats, embargo[t]) + matched(t, zfeats, embargo[t], "_z")
        frames.append(
            run(df, specs, [t]).with_columns(
                features=pl.when(pl.col("spec").str.ends_with("_z"))
                .then(pl.lit("z-scores"))
                .otherwise(pl.lit("raw")),
                spec=pl.col("spec").str.strip_suffix("_z"),
            )
        )
    res = pl.concat(frames).sort("target", "spec", "features", descending=[False, False, True])
    paired = (
        res.filter(pl.col("features") == "z-scores")
        .select("spec", "target", z=pl.col("mse"))
        .join(
            res.filter(pl.col("features") == "raw").select("spec", "target", raw=pl.col("mse")),
            on=["spec", "target"],
        )
    )
    better = paired.filter(pl.col("z") < pl.col("raw")).height
    section(
        "6. Features as z-scores (the per-standard-deviation recipe)",
        "The models of section 5 on the raw features and on each feature in units of its "
        "current spread, as [Scale a feature by its spread]"
        "(how-to/scale-a-feature-by-its-spread.md) makes it,"
        f" `(x - po.ewm_mean(x, half_life={HL:g})) / "
        f"po.ewm_std(x, half_life={HL:g})`, computed as columns by "
        "`po.stream.with_windows`. A z-scored feature's coefficient is a response per "
        "standard deviation of the feature as it is now, where a raw one is per unit. "
        "A z-scored feature fits better where the target moves with the feature relative "
        "to its volatility. The same rows, embargoes and memory as section 5.\n\n"
        + table(res, ["spec", "target", "features", "n", "r2", "ic", "mse"])
        + f"\n\n**Result:** the z-scored features score a lower MSE than the raw ones in "
        f"{better} of {paired.height} pairs of a model and a target. No default rides on "
        "this: the recipe is one you apply to the features yourself.",
    )


if __name__ == "__main__":
    main()
