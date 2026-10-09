"""``ModelBank.check``: findings about the data a bank was fed, read from what
every bank already keeps (docs/PLAN.md task 223 (a)).

Nothing here runs during :meth:`~polars_online.ModelBank.fit_predict`: each
check reads a table the bank keeps anyway (``summary``, ``describe``,
``solve_failures``, ``last_row``, ``gram``, ``marginal``), so it costs the run
nothing and a bank loaded from a file gives the same findings as the bank that
saved it. Every threshold below was measured before it shipped; the docstring
of :meth:`~polars_online.ModelBank.check` gives each one with its rates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polars_online._bank import ModelBank

#: The page that explains each finding, with a recipe for it: a message ends
#: with this address and ``#<code>``, the anchor of that code's section
#: (docs/PLAN.md task 224).
DATA_ISSUES = "https://github.com/hgilde/polars-online/blob/main/docs/DATA-ISSUES.md"

#: Every code a finding can carry. ``tests/test_data_issues.py`` holds it to
#: the docstring's tables and to the anchors of ``docs/DATA-ISSUES.md``.
CODES = frozenset(
    [
        # read from what every bank keeps (task 223 (a))
        "nothing_learned",
        "missing",
        "few_learned",
        "constant",
        "level_over_spread",
        "scales_apart",
        "ridge_shrinks",
        "collinear",
        "leakage",
        "step_back",
        "resets",
        "few_rows",
        "group_sizes",
        "never_settled",
        "below_min_weight",
        "withheld",
        "low_support",
        "solve_failures",
        "not_checked",
        # read from an audit (task 223 (b))
        "sentinel",
        "frozen",
        "few_values",
        "random_walk",
        "heavy_tails",
        "duplicate",
        "duplicate_stamps",
        "gaps",
        "irregular_clock",
    ]
)

#: The severities, worst first: the dtype of the ``severity`` column.
SEVERITY = pl.Enum(["error", "warning", "info"])

#: The columns of the findings frame, in order.
SCHEMA = pl.Schema(
    [
        ("severity", SEVERITY),
        ("code", pl.String()),
        ("spec", pl.String()),
        ("group", pl.String()),
        ("column", pl.String()),
        ("value", pl.Float64()),
        ("threshold", pl.Float64()),
        ("message", pl.String()),
    ]
)

#: A feature or weight column null on this share of rows or more (``missing``).
MISSING_SHARE = 0.1
#: A target null on this share of rows or more (``missing``, info).
MISSING_TARGET_SHARE = 0.5
#: Fewer than this share of the rows fed processed: the rest skipped for a
#: missing feature or weight (``few_learned``).
PROCESSED_SHARE = 0.5
#: ``lasso``'s coordinate descent runs out of sweeps on up to 7 of a stream's
#: first rows on clean data; past this count it is told (``solve_failures``).
LASSO_SWEEPS = 10
#: ``|mean| / std`` past which a model that centres its features is told its
#: input keeps few digits of its variation (``level_over_spread``, info).
LEVEL_INFO = 1e6
#: A level or a spread ratio counts only when it is this many standard errors
#: from what a column with no level, or of equal spread, would show by chance.
SIGMAS = 5.0
#: The largest variance inflation factor, and Belsley's condition index, past
#: which a design is collinear (``collinear``).
COLLINEAR = 30.0
#: ``|corr(feature, target)|`` at or past which a feature is the target, or
#: nearly (``leakage``).
LEAKAGE = 0.9999
#: The Gram's checks wait for this many Kish rows per coefficient.
KISH_PER_COEF = 10.0
#: ``ridge / var`` at or past which the ridge takes a fifth of a coefficient
#: (``ridge_shrinks``).
RIDGE_OVER_VAR = 0.25
#: ``settled_frac`` below one half-life of clock (``never_settled``).
SETTLED = 0.5
#: The smallest coefficient's data share below which the ridge, not the data,
#: set it (``low_support``): the command line's readiness line uses the same.
SUPPORT = 0.5
#: The largest group's learned rows over the smallest's (``group_sizes``).
GROUP_SIZES = 100.0

# -- the thresholds of the checks that read an ``audit`` (task 223 (b)) -------

#: The most repeated value's share of a column's usable values at or past
#: which it may be a sentinel (``sentinel``) ...
SENTINEL_SHARE = 0.01
#: ... when its count is at least this many times the next value's upper
#: bound, the next count plus the counters' error.
SENTINEL_RATIO = 5.0
#: The share of rows equal to the row before, beyond what independent rows
#: of the column's frequencies give, at or past which a column is frozen
#: (``frozen``) ...
FROZEN_EXCESS = 0.05
#: ... or a run of one value this many times the longest run independent
#: rows would show, and at least ``FROZEN_RUN_MIN`` rows.
FROZEN_RUN_FACTOR = 4.0
FROZEN_RUN_MIN = 10
#: At most this many distinct values over at least ``FEW_VALUES_ROWS`` usable
#: rows: a category or a flag stored as numbers (``few_values``).
FEW_VALUES = 10
FEW_VALUES_ROWS = 100
#: The Dickey-Fuller statistic above which a column cannot be told from a
#: random walk (``random_walk``), over at least ``RANDOM_WALK_ROWS`` pairs of
#: consecutive rows: past the 1% critical value with a constant, -3.43.
RANDOM_WALK_TAU = -3.5
RANDOM_WALK_ROWS = 100
#: The excess kurtosis, or the largest robust z, at or past which a column's
#: tails are heavy (``heavy_tails``).
HEAVY_KURTOSIS = 5.0
HEAVY_ROBUST_Z = 10.0
#: ``|corr|`` of two columns at or past which they are one (``duplicate``).
DUPLICATE = 0.999
#: The regular steps' coefficient of variation at or past which the clock
#: is irregular (``irregular_clock``).
IRREGULAR_CV = 0.5

#: The regressions: the models that fit coefficients to targets. A new one
#: joins here (``tests/test_check.py`` holds the list to the registry's), and
#: is measured for ``_UNCENTRED_LIMITS`` if it does not centre its features.
_REGRESSIONS = frozenset(
    ["ewridge", "lasso", "kalman", "huber", "quantile", "ftrl", "sgd", "pa", "rls"]
)
#: The models whose fit reads a ridge in the features' squared units unless
#: ``standardize``.
_RIDGED = frozenset(["ewridge", "huber", "quantile"])
#: The models that keep a co-moment matrix ``gram()`` reports.
_GRAMMED = frozenset(["ewridge", "lasso", "ew_cov"])

#: Per model that does not centre its features: the ``|mean| / std`` and the
#: ratio of feature spreads at which its out-of-sample R^2 had lost no more
#: than 0.05 at half-lives 20, 200 and infinite (docs/PLAN.md task 223's
#: measurement). ``kalman``, ``sgd`` and ``pa`` centre unless
#: ``standardize = False``; ``rls`` and ``ftrl`` never do.
_UNCENTRED_LIMITS: dict[str, tuple[float, float]] = {
    "ftrl": (0.5, 1.5),
    "kalman": (2.0, 2.0),
    "sgd": (3.0, 3.0),
    "pa": (3.0, 2.0),
    "rls": (10.0, 30.0),
}


@dataclass(frozen=True)
class _Finding:
    severity: str
    code: str
    spec: str
    group: str | None
    column: str | None
    value: float | None
    threshold: float | None
    message: str
    #: The group's place in :meth:`ModelBank.groups`' order.
    rank: int


def _uncentred(spec: dict[str, Any]) -> tuple[float, float] | None:
    """The level and spread-ratio limits of a spec whose model does not centre
    its features, or ``None`` for one that does (or is not a regression)."""
    model = spec["model"]
    kind = model["type"]
    if kind not in _UNCENTRED_LIMITS:
        return None
    if kind in ("kalman", "sgd", "pa") and model.get("standardize", True) is not False:
        return None
    return _UNCENTRED_LIMITS[kind]


def _fmt(v: float) -> str:
    if v == 0 or 1e-3 <= abs(v) < 1e5:
        return f"{v:.4g}"
    return f"{v:.3g}"


def _who(group: str | None) -> str:
    if group is None:
        return "the null group"
    if group == "":
        return "the stream"
    return f"group {group!r}"


def check(
    bank: ModelBank, spec: str | int | None, group: str | Iterable[str | None] | None
) -> pl.DataFrame:
    """The findings frame :meth:`ModelBank.check` returns."""
    from polars_online._bank import _group_keys

    names = bank._native.spec_names()
    picked = range(len(names)) if spec is None else [bank._spec_index(spec)]
    keys = _group_keys(group)
    np: Any
    try:
        import numpy

        np = numpy
    except ModuleNotFoundError:
        np = None
    failures = bank.solve_failures()
    found: list[_Finding] = []
    for i in picked:
        found.extend(_spec_findings(bank, i, keys, failures, np))
    order = {"error": 0, "warning": 1, "info": 2}
    rank = {n: j for j, n in enumerate(names)}
    found.sort(
        key=lambda f: (
            order[f.severity],
            rank[f.spec],
            f.rank,
            f.code,
            f.column or "",
        )
    )
    return pl.DataFrame(
        [
            (f.severity, f.code, f.spec, f.group, f.column, f.value, f.threshold, f.message)
            for f in found
        ],
        schema=SCHEMA,
        orient="row",
    )


def _spec_findings(
    bank: ModelBank,
    i: int,
    keys: list[str | None] | None,
    failures: dict[str, dict[str | None, int]],
    np: Any,
) -> list[_Finding]:
    spec = bank._specs[i]
    name = spec["name"]
    kind = spec["model"]["type"]
    out: list[_Finding] = []

    def add(
        severity: str,
        code: str,
        group: str | None,
        column: str | None,
        value: float | None,
        threshold: float | None,
        message: str,
    ) -> None:
        out.append(
            _Finding(
                severity,
                code,
                name,
                group,
                column,
                value,
                threshold,
                f"{message}. See {DATA_ISSUES}#{code}",
                place.get(group, -1),
            )
        )

    summary = bank._native.summary([i], keys)
    describe = bank._native.describe(i, keys)
    groups = summary["group"].to_list()
    wanted = set(groups)
    place = {g: j for j, g in enumerate(groups)}
    flagged: set[tuple[str | None, str]] = set()
    if kind == "audit":
        _audit_findings(bank, i, spec, keys, add)
        # The summary's clock counts, as a model's spec reads them (review
        # 6, F-6: an audit-only bank said nothing of a step back).
        for row in summary.iter_rows(named=True):
            _clock_findings(spec, row, add)
        return out

    # (1), (3), (6), (7): per input column, from describe.
    cols_by_group: dict[str | None, list[dict[str, Any]]] = {}
    for row in describe.iter_rows(named=True):
        cols_by_group.setdefault(row["group"], []).append(row)
    uncentred = _uncentred(spec)
    resolved = _resolved(bank, i)
    for g, rows in cols_by_group.items():
        feats = []
        for r in rows:
            fed = r["count"] + r["null_count"]
            role, col = r["role"], r["column"]
            if fed == 0:
                continue
            share = r["null_count"] / fed
            if role in ("feature", "weight") and share >= MISSING_SHARE:
                sev = "error" if share == 1.0 else "warning"
                add(
                    sev,
                    "missing",
                    g,
                    col,
                    share,
                    MISSING_SHARE,
                    f"{role} {col!r} is null, NaN, infinite or past 1e100 on "
                    f"{share:.1%} of the rows fed to {_who(g)}, and each such row "
                    "is skipped: fill it upstream, or drop the column from the spec",
                )
            elif role == "target" and share >= MISSING_TARGET_SHARE:
                add(
                    "info",
                    "missing",
                    g,
                    col,
                    share,
                    MISSING_TARGET_SHARE,
                    f"target {col!r} is missing on {share:.1%} of the rows fed to "
                    f"{_who(g)}; those rows are predicted but not learned from, "
                    "which is right for sparse labels and a sign of a join gone "
                    "wrong otherwise",
                )
            std, mean = r["std"], r["mean"]
            if std is None or mean is None or role == "weight":
                continue
            if std == 0.0:
                if role == "target":
                    add(
                        "error",
                        "constant",
                        g,
                        col,
                        0.0,
                        0.0,
                        f"target {col!r} took one value ({_fmt(mean)}) on every "
                        f"row of {_who(g)}: there is nothing to learn; check the "
                        "column the target is built from",
                    )
                else:
                    flagged.add((g, col))
                    add(
                        "warning",
                        "constant",
                        g,
                        col,
                        0.0,
                        0.0,
                        f"feature {col!r} took one value ({_fmt(mean)}) on every "
                        f"row of {_who(g)}, "
                        + (
                            "so it is collinear with the intercept and its "
                            "coefficient is set by the prior, not the data"
                            if kind in _REGRESSIONS
                            else "so it has no spread to correlate or cluster on"
                        )
                        + ": drop it, or check the feed it comes from",
                    )
                continue
            if role != "feature":
                continue
            feats.append(r)
            ratio = abs(mean) / std
            limit = uncentred[0] if uncentred else LEVEL_INFO
            if ratio > limit and ratio * math.sqrt(r["count"]) >= SIGMAS:
                if uncentred:
                    add(
                        "warning",
                        "level_over_spread",
                        g,
                        col,
                        ratio,
                        limit,
                        f"feature {col!r} sits {_fmt(ratio)} standard deviations "
                        f"from zero in {_who(g)}, and {kind} does not centre it, "
                        "which costs it accuracy past "
                        f"{_fmt(limit)}: subtract its mean or difference it "
                        "upstream"
                        + (", or set standardize=True" if kind in ("kalman", "sgd", "pa") else ""),
                    )
                else:
                    add(
                        "info",
                        "level_over_spread",
                        g,
                        col,
                        ratio,
                        limit,
                        f"feature {col!r} sits {_fmt(ratio)} standard deviations "
                        f"from zero in {_who(g)}, so a double keeps about "
                        f"{max(0, 16 - round(math.log10(ratio)))} digits of its "
                        f"variation; {kind} centres it and loses nothing, but "
                        "arithmetic on its raw values elsewhere may: subtract an "
                        "origin upstream",
                    )
        # (7) scales apart: only where the fit is not scale-free.
        if uncentred and len(feats) >= 2:
            big = max(feats, key=lambda r: r["std"])
            small = min(feats, key=lambda r: r["std"])
            ratio = big["std"] / small["std"]
            n = min(big["count"], small["count"])
            limit = uncentred[1]
            if ratio > limit and n > 1 and math.log(ratio) * math.sqrt(n - 1) >= SIGMAS:
                add(
                    "warning",
                    "scales_apart",
                    g,
                    small["column"],
                    ratio,
                    limit,
                    f"feature {big['column']!r} spreads {_fmt(ratio)} times as "
                    f"wide as {small['column']!r} in {_who(g)}, and {kind} takes "
                    "one step for every column, which costs it accuracy past "
                    f"{_fmt(limit)}: divide each feature by its spread upstream"
                    + (", or set standardize=True" if kind in ("kalman", "sgd", "pa") else ""),
                )
        # (7) a shared ridge as large as a feature's variance.
        model = resolved["model"]
        # `ridge_scale` resolves to whether the ridge sits on the decaying sum
        # scale ("sum", where it fades), so `false` is "mean".
        if kind in _RIDGED and not model.get("standardize") and not model.get("ridge_scale"):
            ridges = model.get("ridge")
            ridge = min(ridges) if isinstance(ridges, list) else float(ridges)
            for r in feats:
                var = r["std"] ** 2
                if ridge > 0 and ridge / var >= RIDGE_OVER_VAR:
                    flagged.add((g, r["column"]))
                    add(
                        "warning",
                        "ridge_shrinks",
                        g,
                        r["column"],
                        ridge / var,
                        RIDGE_OVER_VAR,
                        f"the ridge {_fmt(ridge)} is {_fmt(ridge / var)} times the "
                        f"variance of feature {r['column']!r} in {_who(g)}, so it "
                        f"takes {ridge / (ridge + var):.0%} of that coefficient: "
                        "set standardize=True, or rescale the feature upstream",
                    )

    # (10), (11) from the Gram, and (11) from a marginal.
    if kind in _GRAMMED:
        if np is None:
            add(
                "info",
                "not_checked",
                None,
                None,
                None,
                None,
                "numpy is not installed, so collinearity and leakage were not "
                "checked: pip install polars-online[numpy]",
            )
        else:
            _gram_findings(bank, i, spec, keys, flagged, np, add)
    if kind == "marginal":
        pairs = bank._native.marginal(i, keys)
        worst: dict[tuple[str | None, str, str], float] = {}
        for pg, feat, tgt, corr in pairs.select("group", "feature", "target", "corr").iter_rows():
            if corr is not None and abs(corr) >= LEAKAGE:
                worst[(pg, feat, tgt)] = max(worst.get((pg, feat, tgt), 0.0), abs(corr))
        for (pg, feat, tgt), top in worst.items():
            add(*_leak(pg, feat, tgt, top))

    # (1) the rows learned, (12) the clock, (13) groups, (14) the fit: per group.
    learned_by_group = {}
    need = float(max(resolved["stream"].get("min_weight") or [0.0]))
    worst_missing = _worst_missing(cols_by_group, ("feature", "weight", "target"))
    worst_skipping = _worst_missing(cols_by_group, ("feature", "weight"))
    for row in summary.iter_rows(named=True):
        g = row["group"]
        fed, learned = row["rows_fed"], row["rows_learned"]
        learned_by_group[g] = learned
        if fed > 0 and learned == 0:
            add(
                "error",
                "nothing_learned",
                g,
                worst_missing.get(g),
                0.0,
                0.0,
                f"{_who(g).capitalize()} was fed {fed} rows and learned from none"
                + (
                    f"; {worst_missing[g]!r} is the column null most often"
                    if worst_missing.get(g)
                    else ""
                )
                + ": check that a target and every feature are present on the "
                "same rows, and that the weights are not all zero",
            )
        elif fed > 0 and row["rows_processed"] / fed < PROCESSED_SHARE:
            share = row["rows_processed"] / fed
            skipper = worst_skipping.get(g)
            add(
                "warning",
                "few_learned",
                g,
                skipper,
                share,
                PROCESSED_SHARE,
                f"{_who(g).capitalize()} skipped {1 - share:.1%} of the {fed} rows it "
                "was fed for a missing feature or weight"
                + (f", {skipper!r} most often" if skipper else "")
                + ": fill the gaps upstream, or drop the column that causes most",
            )
        _clock_findings(spec, row, add)
        n_coef = row["n_coef"]
        if kind in _REGRESSIONS and n_coef and 0 < learned < n_coef:
            add(
                "warning",
                "few_rows",
                g,
                None,
                float(learned),
                float(n_coef),
                f"{_who(g).capitalize()} learned from {learned} rows for {n_coef} "
                "coefficients a target, too few to determine them: merge small "
                "groups, or use fewer features",
            )
        settled = row["settled_frac"]
        if settled is not None and learned > 0 and settled < SETTLED:
            add(
                "warning",
                "never_settled",
                g,
                None,
                settled,
                SETTLED,
                f"{_who(g).capitalize()} has seen less than one half-life of clock "
                f"(settled_frac {settled:.2f}), so its fit rests on a window that "
                "has not filled: feed more history, or shorten the half-life",
            )
        settled_w = row["weight_sum_settled"]
        if settled_w is not None and need > 0 and settled_w < need:
            add(
                "warning",
                "below_min_weight",
                g,
                None,
                settled_w,
                need,
                f"{_who(g).capitalize()} settles at a weight of {_fmt(settled_w)}, "
                f"below min_weight {_fmt(need)}, so its predictions will stay "
                "null: lengthen the half-life, or lower min_weight",
            )
        support = row["min_support_coef"]
        feature = row["min_support_coef_feature"]
        if support is not None and support < SUPPORT and (g, feature) not in flagged:
            add(
                "warning",
                "low_support",
                g,
                feature,
                support,
                SUPPORT,
                f"the coefficient of {feature!r} in {_who(g)} is {support:.0%} data "
                "and the rest ridge: a duplicated or constant column, or a ridge "
                "as large as the feature's variance",
            )
        n_fail = failures.get(name, {}).get(g, 0)
        if kind == "lasso" and n_fail > LASSO_SWEEPS:
            add(
                "info",
                "solve_failures",
                g,
                None,
                float(n_fail),
                float(LASSO_SWEEPS),
                f"{_who(g).capitalize()}'s coordinate descent ran out of max_iter "
                f"sweeps {n_fail} times, common on correlated features: raise "
                "max_iter or loosen tol if the path's coefficients matter",
            )
        elif kind != "lasso" and n_fail:
            add(
                "warning",
                "solve_failures",
                g,
                None,
                float(n_fail),
                0.0,
                f"{_who(g).capitalize()} needed jitter or kept the previous fit "
                f"{n_fail} times: the features are constant, collinear or too few "
                "rows for their number",
            )

    # (13) group sizes far apart.
    sizes = [v for v in learned_by_group.values() if v > 0]
    if len(sizes) >= 2 and max(sizes) / min(sizes) >= GROUP_SIZES:
        smallest = min(
            (g for g in learned_by_group if learned_by_group[g] > 0),
            key=lambda g: learned_by_group[g],
        )
        add(
            "info",
            "group_sizes",
            smallest,
            None,
            max(sizes) / min(sizes),
            GROUP_SIZES,
            f"the largest group learned from {max(sizes) / min(sizes):.0f} times "
            f"as many rows as the smallest, {_who(smallest)}: a small group's fit "
            "is noisy beside the others'",
        )

    # (14) a prediction withheld on the last row.
    last = bank.last_row(i, keys)
    reason_cols = [c for c in last.columns if c.startswith("withheld_reason")]
    for row in last.select("group", *reason_cols).iter_rows(named=True):
        reasons = sorted({str(row[c]) for c in reason_cols if row[c] is not None})
        if reasons and row["group"] in wanted:
            add(
                "warning",
                "withheld",
                row["group"],
                None,
                None,
                None,
                f"the last row of {_who(row['group'])} had its predictions "
                f"withheld ({', '.join(reasons)}): the stream is not ready for "
                "use yet",
            )

    return out


def _leak(
    g: str | None, feat: str, tgt: str, corr: float
) -> tuple[str, str, str | None, str, float, float, str]:
    return (
        "error",
        "leakage",
        g,
        feat,
        corr,
        LEAKAGE,
        f"feature {feat!r} correlates with target {tgt!r} at {corr:.6f} in "
        f"{_who(g)}: it contains the target, or both are the same random walk; "
        "build the feature from rows before the target's, or difference both",
    )


def _gram_findings(
    bank: ModelBank,
    i: int,
    spec: dict[str, Any],
    keys: list[str | None] | None,
    flagged: set[tuple[str | None, str]],
    np: Any,
    add: Any,
) -> None:
    from polars_online import gram as pg

    kind = spec["model"]["type"]
    features = list(spec["features"])
    worst_vif: dict[str | None, tuple[float, str, str]] = {}
    worst_leak: dict[tuple[str | None, str, str], float] = {}
    for g in bank.gram(i, keys):
        group = g["group"]
        live = [f for f in features if (group, f) not in flagged and _variance(g, f) > 0.0]
        n_kish = g["n_kish"]
        if n_kish is None or n_kish < KISH_PER_COEF * (len(features) + 1):
            continue
        if len(live) >= 2:
            vif = np.asarray(pg.vif(g, features=live), dtype=float)
            kappa = pg.condition(g, features=live)["kappa"]
            j = int(np.argmax(vif))
            stat = max(float(vif[j]), float(kappa))
            if stat > COLLINEAR:
                corr = pg.correlation(pg.subset(g, live))
                np.fill_diagonal(corr, 0.0)
                a, b = np.unravel_index(int(np.nanargmax(np.abs(corr))), corr.shape)
                pair = (live[a], live[b])
                if stat > worst_vif.get(group, (0.0, "", ""))[0]:
                    worst_vif[group] = (stat, pair[0], pair[1])
        cols = list(g["columns"])
        for t, target in enumerate(g["targets"]):
            vy = float(g["target_vars"][t])
            if not vy > 0.0:
                continue
            for f in live:
                j = cols.index(f)
                vx = float(g["comoments"][j][j])
                corr = float(g["cross_centred"][t][j]) / math.sqrt(vx * vy)
                if abs(corr) >= LEAKAGE:
                    key = (group, f, target)
                    worst_leak[key] = max(worst_leak.get(key, 0.0), min(abs(corr), 1.0))
    sev = "info" if kind == "ew_cov" else "warning"
    for group, (stat, a, b) in worst_vif.items():
        flagged.update([(group, a), (group, b)])
        add(
            sev,
            "collinear",
            group,
            a,
            stat,
            COLLINEAR,
            f"features {a!r} and {b!r} are nearly linear combinations of the others "
            f"in {_who(group)} (largest VIF or condition index {_fmt(stat)}), so "
            + (
                "the covariance matrix is near singular: drop one before inverting it"
                if kind == "ew_cov"
                else "their coefficients split one effect arbitrarily: drop one, or raise the ridge"
            ),
        )
    for (group, f, target), corr in worst_leak.items():
        add(*_leak(group, f, target, corr))


def _variance(g: dict[str, Any], col: str) -> float:
    cols = list(g["columns"])
    j = cols.index(col)
    return float(g["comoments"][j][j])


def _worst_missing(
    cols_by_group: dict[str | None, list[dict[str, Any]]], roles: tuple[str, ...]
) -> dict[str | None, str]:
    """Per group, the input column of these roles null on the most rows, when
    any is."""
    out = {}
    for g, rows in cols_by_group.items():
        mine = [r for r in rows if r["role"] in roles]
        best = max(mine, key=lambda r: r["null_count"], default=None)
        if best is not None and best["null_count"] > 0:
            out[g] = best["column"]
    return out


def _resolved(bank: ModelBank, i: int) -> dict[str, Any]:
    """A spec's parameters as the bank resolved them, its defaults included."""
    import json

    from polars_online import _polars_online as native
    from polars_online._spec import _json

    out: dict[str, Any] = json.loads(native.resolved_defaults(_json(bank._specs[i])))
    return out


def _clock_findings(spec: dict[str, Any], row: dict[str, Any], add: Any) -> None:
    """``step_back`` and ``resets`` from one summary row, for any spec."""
    g = row["group"]
    back = row["clock_backwards"]
    if back:
        add(
            "warning",
            "step_back",
            g,
            spec.get("clock"),
            float(back),
            0.0,
            f"the clock stepped back {back} times in {_who(g)}, each a restart "
            "or a late row under restart_after_step_back: sort the input by "
            "its clock within each group if that was not meant",
        )
    resets = row["resets"]
    if resets:
        forgets = (
            "each start begins its runs and its clock steps afresh, and an audit keeps its counts"
            if spec["model"]["type"] == "audit"
            else "each start forgets what came before"
        )
        add(
            "info",
            "resets",
            g,
            spec.get("clock"),
            float(resets),
            0.0,
            f"{_who(g).capitalize()} started over {resets} times, at a "
            f"session_gap or a step back: {forgets}",
        )


def _audit_findings(
    bank: ModelBank,
    i: int,
    spec: dict[str, Any],
    keys: list[str | None] | None,
    add: Any,
) -> None:
    """An ``audit`` spec's findings: what its columns and its clock hold, for
    any model that would read them (task 223 (b)). An audit knows no roles,
    so a column's missing share is information unless it is every row."""
    for r in bank._native.audit(i, keys, "columns", False).iter_rows(named=True):
        _audit_column(r, add)
    if spec["model"].get("pairs"):
        for r in bank._native.audit(i, keys, "pairs", False).iter_rows(named=True):
            corr = r["corr"]
            if corr is None or abs(corr) < DUPLICATE:
                continue
            same = r["equal"] / r["count"] if r["count"] else 0.0
            add(
                "warning",
                "duplicate",
                r["group"],
                r["column_b"],
                abs(corr),
                DUPLICATE,
                f"columns {r['column_a']!r} and {r['column_b']!r} correlate at "
                f"{corr:.6f} in {_who(r['group'])}"
                + (f" and are equal on {same:.1%} of the rows" if same else "")
                + ": one is the other, rescaled; a model reading both splits one "
                "effect between them, so drop one",
            )
    clock = spec.get("clock")
    for r in bank._native.audit(i, keys, "clock", False).iter_rows(named=True):
        g, steps = r["group"], r["steps"]
        if not steps:
            continue
        if r["duplicates"]:
            add(
                "info",
                "duplicate_stamps",
                g,
                clock,
                r["duplicates"] / steps,
                0.0,
                f"{r['duplicates']} rows of {_who(g)} share the clock of the row "
                "before: no decay passes between them, and a window holds them "
                "together; if a row was fed twice, deduplicate it upstream",
            )
        if r["gaps"]:
            add(
                "info",
                "gaps",
                g,
                clock,
                float(r["gaps"]),
                0.0,
                f"{r['gaps']} steps of {_who(g)} reach gap_cap: each is a break, "
                "across which a model forgets only gap_cap of clock and drops its "
                "lags",
            )
        cv = r["step_cv"]
        if cv is not None and cv >= IRREGULAR_CV:
            add(
                "info",
                "irregular_clock",
                g,
                clock,
                cv,
                IRREGULAR_CV,
                f"the clock's steps in {_who(g)} vary by {cv:.2f} of their mean: a "
                "half-life in clock units weighs rows by time, as it should, but a "
                "lag or a window counted in rows means a different time on each row",
            )


def _audit_column(r: dict[str, Any], add: Any) -> None:
    """One column's findings, from its row of ``ModelBank.audit``."""
    g, col, rows, n = r["group"], r["column"], r["rows"], r["count"]
    if rows == 0:
        return
    who = _who(g)
    share = (rows - n) / rows
    if share >= MISSING_SHARE:
        add(
            "error" if share == 1.0 else "info",
            "missing",
            g,
            col,
            share,
            MISSING_SHARE,
            f"column {col!r} is null, NaN, infinite or past 1e100 on {share:.1%} of "
            f"the rows of {who}: a model reading it as a feature skips each such "
            "row, and as a target predicts it without learning",
        )
    odd = r["pos_inf"] + r["neg_inf"] + r["beyond_bound"]
    if odd:
        add(
            "warning",
            "sentinel",
            g,
            col,
            float(odd),
            0.0,
            f"column {col!r} holds +inf on {r['pos_inf']}, -inf on {r['neg_inf']} "
            f"and a value past 1e100 on {r['beyond_bound']} rows of {who}, which a "
            "model skips as it skips a null: a division by zero, or a sentinel "
            "such as 1e308, upstream; make them null, or fix the arithmetic",
        )
    if r["nan"]:
        add(
            "info",
            "sentinel",
            g,
            col,
            float(r["nan"]),
            0.0,
            f"column {col!r} is NaN, not null, on {r['nan']} rows of {who}: the "
            "models read it as missing, and Polars' fill_null, drop_nulls and "
            "null_count do not; fill_nan(None) makes them null",
        )
    if n < 2:
        return
    if r["std"] == 0.0:
        add(
            "warning",
            "constant",
            g,
            col,
            0.0,
            0.0,
            f"column {col!r} took one value ({_fmt(r['mean'])}) on every usable row "
            f"of {who}: as a feature it is collinear with the intercept, and as a "
            "target there is nothing to learn",
        )
        return
    distinct = r["distinct"]
    few = distinct is not None and distinct <= FEW_VALUES
    adjacent = r["adjacent"]
    repeats = r["equal_prev"] / adjacent if adjacent else 0.0
    chance = r["equal_by_chance"] or 0.0
    top = r["top_count"] / n
    expected_run = math.log(n) / -math.log(top) if 0.0 < top < 1.0 else 1.0
    run_limit = max(FROZEN_RUN_MIN, FROZEN_RUN_FACTOR * expected_run)
    frozen = repeats - chance >= FROZEN_EXCESS or r["longest_run"] >= run_limit
    if frozen:
        add(
            "warning",
            "frozen",
            g,
            col,
            repeats,
            chance + FROZEN_EXCESS,
            f"column {col!r} repeats the row before on {repeats:.1%} of the rows of "
            f"{who}, where its values' frequencies give {chance:.1%}, and holds one "
            f"value for {r['longest_run']} rows at most: a feed that stopped, or a "
            "forward fill; null the stale rows, or join on the time each value was "
            "observed",
        )
    bound = max(r["second_count"] + r["count_error"], 1)
    if not (few or frozen) and top >= SENTINEL_SHARE and r["top_count"] >= SENTINEL_RATIO * bound:
        add(
            "warning",
            "sentinel",
            g,
            col,
            top,
            SENTINEL_SHARE,
            f"column {col!r} is {_fmt(r['top_value'])} on {top:.1%} of the usable "
            f"rows of {who}, at least {SENTINEL_RATIO:g} times as often as any "
            "other value: if it stands for missing, make it null; if it is real, a "
            "model may want an indicator for it",
        )
    if few and n >= FEW_VALUES_ROWS:
        add(
            "info",
            "few_values",
            g,
            col,
            float(distinct),
            float(FEW_VALUES),
            f"column {col!r} takes {distinct} distinct values over {n} rows of {who}: "
            "a category or a flag stored as numbers, which a linear model reads as "
            "a quantity; one-hot encode a category",
        )
    tau = r["unit_root_t"]
    if tau is not None and adjacent >= RANDOM_WALK_ROWS and tau > RANDOM_WALK_TAU:
        add(
            "warning",
            "random_walk",
            g,
            col,
            tau,
            RANDOM_WALK_TAU,
            f"column {col!r} cannot be told from a random walk in {who} (lag-1 "
            f"autocorrelation {_fmt(r['autocorr'])}, Dickey-Fuller {_fmt(tau)}): a "
            "regression of one level on another is spurious; difference it, or "
            "take its deviation from a moving mean",
        )
    kurt, rz = r["kurtosis"], r["robust_z"]
    far = rz is not None and rz >= HEAVY_ROBUST_Z
    if far or (kurt is not None and kurt >= HEAVY_KURTOSIS):
        # The statistic that fired, beside its own threshold: the robust z
        # where it passed, the kurtosis otherwise (review 6, G-11: the robust
        # z was reported whichever fired).
        add(
            "info",
            "heavy_tails",
            g,
            col,
            rz if far else kurt,
            HEAVY_ROBUST_Z if far else HEAVY_KURTOSIS,
            f"column {col!r} has heavy tails in {who}: excess kurtosis "
            f"{'undefined' if kurt is None else _fmt(kurt)}, and a value "
            f"{'undefined' if rz is None else _fmt(rz)} robust standard deviations "
            "from the median; a squared loss follows the largest rows, so huber or "
            "quantile resist them, or winsorize upstream",
        )
