"""Evaluation harness (docs/PLAN.md section 8).

Pure Polars over the output structs: per (spec, group, target) out-of-sample
R^2, IC (correlation of prediction with realized target) and hit rate, either
overall or in rolling windows measured in clock units. One ``group_by``
compares specs.

Everything here consumes the frame a :class:`~polars_online.ModelBank` returns:
the original columns plus one struct column per spec. It needs the whole
frame -- these are Polars aggregations over collected output -- where a
spec's ``emit_metrics`` keeps the same numbers beside the model, O(state),
for a stream too long to hold.

What each function raises is what :func:`unpack` raises, since each starts
there: ``KeyError`` for a ``spec_name`` the frame has not got, ``TypeError``
for a column that is not a model's prediction struct, ``ValueError`` for a
slot whose target column cannot be found. A ``by`` or ``clock`` column the
frame has not got is polars' ``ColumnNotFoundError``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

__all__ = [
    "metrics",
    "rolling_metrics",
    "compare_specs",
    "unpack",
    "seqtest",
    "sums",
    "merge_sums",
    "from_sums",
]

#: The columns :func:`sums` produces beside the keys, in order. Ten doubles
#: per (key, slot) is the whole memory cost of evaluating a stream.
SUM_FIELDS = (
    "n",
    "w",
    "mean_y",
    "mean_pred",
    "m2_y",
    "m2_pred",
    "cov",
    "sse",
    "hits",
    "signed",
)


def _pred_fields(df: pl.DataFrame, spec_name: str) -> list[str]:
    dtype = df.schema[spec_name]
    if not isinstance(dtype, pl.Struct):
        msg = f"column {spec_name!r} is not a model-output struct"
        raise TypeError(msg)
    fields = [f.name for f in dtype.fields if f.name.startswith("pred_")]
    if not fields:
        msg = (
            f"column {spec_name!r} has no prediction fields (pred_*) to unpack; "
            "an ew_cov struct holds statistics, a kmeans or micro struct assignments, "
            "an ew_class struct a class and its posteriors and a seqtest struct "
            "evidence, not predictions"
        )
        raise TypeError(msg)
    return fields


#: `online_core::INPUT_BOUND`: a magnitude beyond it is a missing value to
#: the bank (docs/PLAN.md section 3), so it is one to :func:`seqtest` too.
_INPUT_BOUND = 1e100

#: Column names :func:`unpack` produces. Input columns with these names are
#: dropped rather than duplicated (the target's values come back as ``y``).
RESERVED = ("slot", "target", "pred", "y")


def unpack(
    df: pl.DataFrame,
    spec_name: str,
    *,
    spec: dict | None = None,
    targets: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Long form: one row per (row, prediction slot).

    Returns ``slot`` (the struct field name), ``target`` (the target column it
    predicts), ``pred``, ``y`` and every other non-struct column of ``df``.
    Input columns named like the output ones (see :data:`RESERVED`) are dropped
    -- a target column called ``y`` would otherwise collide with the ``y``
    output.

    Pass ``spec`` to resolve each slot's target exactly (via
    :func:`polars_online.spec.output_index`); without it a name-based heuristic
    is used, which can misattribute a target whose name embeds another's.

    Raises ``KeyError`` for a ``spec_name`` the frame has not got;
    ``TypeError`` for a column that is not a struct, or a struct with no
    ``pred_*`` fields (an ``ew_cov``, ``kmeans``, ``micro`` or ``ew_class``
    output); ``ValueError`` when a slot's
    target column cannot be found -- the frame no longer has it, or
    ``targets`` does not name it -- and, with ``spec``, whatever
    :func:`polars_online.spec.output_index` raises for it.
    """
    fields = _pred_fields(df, spec_name)
    keep = [c for c, d in df.schema.items() if not isinstance(d, pl.Struct) and c not in RESERVED]
    # With the spec in hand, the slot -> target mapping comes from
    # `output_index` -- the same Rust code that named the fields -- instead of
    # the heuristic name-parsing below, which exists only for callers who have
    # a frame but no spec.
    exact: dict[str, str] = {}
    if spec is not None:
        from polars_online import spec as _spec_mod

        idx = _spec_mod.output_index(spec)
        exact = {
            r["field"]: r["target"]
            for r in idx.filter(pl.col("kind") == "pred").iter_rows(named=True)
            if r["target"] is not None
        }
    frames = []
    for slot in fields:
        target = exact.get(slot) or _target_of(slot, df, targets)
        frames.append(
            df.select(
                *keep,
                pl.lit(slot).alias("slot"),
                pl.lit(target).alias("target"),
                df[spec_name].struct.field(slot).alias("pred"),
                pl.col(target).alias("y"),
            )
        )
    return pl.concat(frames)


def _target_of(slot: str, df: pl.DataFrame, targets: Sequence[str] | None) -> str:
    """``pred_<target>[__combo][@hN]`` -> ``<target>``. Longest match wins, so
    targets whose names are prefixes of each other still resolve."""
    body = slot[len("pred_") :]
    candidates = targets if targets is not None else list(df.columns)
    matches = [
        t for t in candidates if body == t or body.startswith(t + "__") or body.startswith(t + "@")
    ]
    if not matches:
        msg = f"cannot infer the target column for slot {slot!r}"
        raise ValueError(msg)
    return max(matches, key=len)


def _metric_exprs(min_obs: int) -> list[pl.Expr]:
    resid = pl.col("y") - pl.col("pred")
    ybar = pl.col("y").mean()
    return [
        pl.len().alias("n"),
        # Out-of-sample R^2 against the realized mean of y in the window.
        (1.0 - (resid.pow(2).sum() / (pl.col("y") - ybar).pow(2).sum())).alias("r2"),
        pl.corr("pred", "y").alias("ic"),
        # Hit rate: fraction of rows where the sign of pred matches the sign of
        # y (rows with y == 0 excluded).
        (
            ((pl.col("pred").sign() == pl.col("y").sign()) & (pl.col("y") != 0))
            .sum()
            .truediv((pl.col("y") != 0).sum())
        ).alias("hit_rate"),
        resid.pow(2).mean().alias("mse"),
    ] + [pl.when(pl.len() >= min_obs).then(True).otherwise(False).alias("enough")]


def metrics(
    df: pl.DataFrame,
    spec_name: str,
    *,
    by: Iterable[str] = (),
    targets: Sequence[str] | None = None,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Overall out-of-sample metrics per ``(slot, *by)``.

    Rows where the prediction or the target is null are dropped, so warmup and
    skipped rows never enter the numbers; a group with fewer than ``min_obs``
    rows left is dropped from the result rather than reported on too little.
    ``n`` is the rows counted, ``r2`` out-of-sample R^2 against the realized
    mean, ``ic`` the correlation of prediction with target, ``hit_rate`` the
    fraction of sign agreements (rows with ``y == 0`` excluded), ``mse`` the
    mean squared residual. Raises as :func:`unpack` does; a ``by`` column the
    frame has not got is polars' ``ColumnNotFoundError``.
    """
    long = unpack(df, spec_name, targets=targets).drop_nulls(["pred", "y"])
    keys = ["slot", "target", *by]
    out = long.group_by(keys).agg(_metric_exprs(min_obs)).sort(keys)
    return out.filter(pl.col("enough")).drop("enough")


def rolling_metrics(
    df: pl.DataFrame,
    spec_name: str,
    *,
    clock: str,
    window: float,
    by: Iterable[str] = (),
    targets: Sequence[str] | None = None,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Metrics in non-overlapping windows of ``window`` clock units.

    ``window_start`` is the left edge of each bucket (``floor(clock/window)*window``);
    the columns are :func:`metrics`'s, per window. Raises as :func:`unpack`
    does, ``ValueError`` for a ``window`` that is not above 0, ``TypeError``
    for a ``clock`` column that is not numeric, and polars'
    ``ColumnNotFoundError`` for a ``clock`` or ``by`` column the frame has
    not got.
    """
    if not window > 0:
        msg = f"window must be > 0, got {window}"
        raise ValueError(msg)
    if clock in df.schema and not df.schema[clock].is_numeric():
        msg = f"clock column {clock!r} must be numeric, got {df.schema[clock]}"
        raise TypeError(msg)
    long = unpack(df, spec_name, targets=targets).drop_nulls(["pred", "y"])
    long = long.with_columns(((pl.col(clock) / window).floor() * window).alias("window_start"))
    keys = ["slot", "target", *by, "window_start"]
    out = long.group_by(keys).agg(_metric_exprs(min_obs)).sort(keys)
    return out.filter(pl.col("enough")).drop("enough")


def compare_specs(
    df: pl.DataFrame,
    spec_names: Iterable[str],
    *,
    by: Iterable[str] = (),
    targets: Sequence[str] | None = None,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Stack :func:`metrics` for several specs, adding a ``spec`` column.

    Raises as :func:`metrics` does for each; no specs give an empty frame."""
    frames = [
        metrics(df, name, by=by, targets=targets, min_obs=min_obs).with_columns(
            pl.lit(name).alias("spec")
        )
        for name in spec_names
    ]
    cols = ["spec", *frames[0].columns[:-1]] if frames else []
    return pl.concat(frames).select(cols) if frames else pl.DataFrame()


def _resid_fields(df: pl.DataFrame, spec_name: str) -> list[str]:
    dtype = df.schema.get(spec_name)
    if dtype is None:
        raise KeyError(spec_name)
    if not isinstance(dtype, pl.Struct):
        msg = f"column {spec_name!r} is not a model-output struct"
        raise TypeError(msg)
    return [f.name for f in dtype.fields if f.name.startswith("resid_")]


def seqtest(
    df: pl.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    by: Iterable[str] = (),
    min_periods: float = 0.0,
    name: str = "seqtest",
) -> pl.DataFrame:
    """:func:`polars_online.spec.seqtest` in polars expressions, over a frame
    in memory: the same e-processes, the same fields, row for row.

    Column mode (no ``a``/``b``): ``targets`` name the columns whose sign is
    tested. Compare mode: ``a`` and ``b`` name two output structs of ``df``
    (two specs the bank ran), ``targets`` the residuals both carry -- ``t``
    for ``resid_<t><a_suffix>`` of ``a`` against ``resid_<t><b_suffix>`` of
    ``b``; every ``t`` they share when ``None`` -- and the sign tested is
    that of ``|resid_b| - |resid_a|``, positive when ``a`` was closer.

    Returns ``df`` with a struct column ``name`` holding, per target ``t``
    and read *before* the row, as the bank emits them: ``log_e_pos_<t>``,
    ``log_e_neg_<t>``, ``n_pos_<t>``, ``n_neg_<t>`` in column mode,
    ``log_e_a_<t>``, ``log_e_b_<t>``, ``wins_a_<t>``, ``wins_b_<t>`` in
    compare mode, then ``n_eff`` -- the rows before this one in its ``by``
    group -- with every other field null until ``n_eff`` reaches
    ``min_periods``.
    ``by`` runs one process per group, in row order (``.over(by)``). A null,
    zero or NaN value bets nothing and counts nothing, as in the bank; what
    the bank adds is the clock (``session``, ``on_clock_reset``), which a
    frame in memory has not got.

    Per target, with ``s`` the sign and the counts *before* the row::

        lam_pos = max(0, (n_pos - n_neg) / (n_pos + n_neg + 1))
        log_e_pos += log1p(lam_pos * s)         (lam_neg, log_e_neg likewise)

    ``tests/test_seqtest.py`` holds the bank's struct to this one to the
    last bit; the difference is that the bank is O(state) over a stream and
    this is O(rows) over a frame.

    Raises ``ValueError`` for ``a`` without ``b`` (or the reverse), for
    column mode without ``targets``, and for a target neither side has a
    residual for (naming the fields it does have); ``KeyError`` for a spec
    the frame has not got, ``TypeError`` for one that is not a struct.
    """
    keys = list(by)
    if (a is None) != (b is None):
        msg = "seqtest: a and b go together; name both specs to compare them, or neither"
        raise ValueError(msg)
    if a is not None and b is not None:
        have = {a: _resid_fields(df, a), b: _resid_fields(df, b)}
        if targets is None:
            # `resid_<t><a_suffix>` of a, for every t with `resid_<t><b_suffix>` in b.
            bodies = [f.removeprefix("resid_") for f in have[a] if f.endswith(a_suffix)]
            stems = [x[: len(x) - len(a_suffix)] for x in bodies]
            targets = [t for t in stems if f"resid_{t}{b_suffix}" in have[b]]
            if not targets:
                msg = (
                    f"seqtest: {a!r} and {b!r} share no residual field (with a_suffix "
                    f"{a_suffix!r}, b_suffix {b_suffix!r}); {a!r} has {have[a]}, "
                    f"{b!r} has {have[b]}"
                )
                raise ValueError(msg)
        for t in targets:
            for side, suffix in ((a, a_suffix), (b, b_suffix)):
                want = f"resid_{t}{suffix}"
                if want not in have[side]:
                    msg = (
                        f"seqtest: target {t!r} names no residual of {side!r}: it has no field "
                        f"{want!r} (its residual fields are {have[side]})"
                    )
                    raise ValueError(msg)
        signs = {
            t: pl.col(b).struct.field(f"resid_{t}{b_suffix}").abs()
            - pl.col(a).struct.field(f"resid_{t}{a_suffix}").abs()
            for t in targets
        }
        names = ("log_e_a", "log_e_b", "wins_a", "wins_b")
    else:
        if targets is None:
            msg = "seqtest: targets name the columns whose sign is tested (or give a and b)"
            raise ValueError(msg)
        signs = {t: pl.col(t) for t in targets}
        names = ("log_e_pos", "log_e_neg", "n_pos", "n_neg")

    def over(e: pl.Expr) -> pl.Expr:
        return e.over(keys) if keys else e

    def before(e: pl.Expr) -> pl.Expr:
        """The running sum of ``e`` over the rows before this one."""
        return over(e.cum_sum().shift(1, fill_value=0))

    n_eff = over(pl.int_range(pl.len())).cast(pl.Float64)
    ready = n_eff >= min_periods
    fields: list[pl.Expr] = []
    for t, d in signs.items():
        # What the bank does not learn from -- null, NaN, an infinity, a
        # magnitude beyond its input bound (docs/PLAN.md section 3) -- is no
        # sign here either. Polars orders NaN above every float, so without
        # this `NaN > 0` would be a positive sign.
        d = d.cast(pl.Float64)
        d = pl.when(d.is_finite() & (d.abs() <= _INPUT_BOUND)).then(d)
        s = pl.when(d > 0).then(1.0).when(d < 0).then(-1.0).otherwise(0.0)
        n_pos = before((d > 0).cast(pl.Int64).fill_null(0))
        n_neg = before((d < 0).cast(pl.Int64).fill_null(0))
        n1 = (n_pos + n_neg + 1).cast(pl.Float64)
        lam_pos = pl.max_horizontal((n_pos - n_neg).cast(pl.Float64) / n1, 0.0)
        lam_neg = pl.max_horizontal((n_neg - n_pos).cast(pl.Float64) / n1, 0.0)
        log_e_pos = before((lam_pos * s).log1p())
        log_e_neg = before((-lam_neg * s).log1p())
        for label, e in zip(names, (log_e_pos, log_e_neg, n_pos, n_neg), strict=True):
            fields.append(pl.when(ready).then(e).alias(f"{label}_{t}"))
    fields.append(n_eff.alias("n_eff"))
    return df.with_columns(pl.struct(fields).alias(name))


def sums(
    df: pl.DataFrame,
    spec_name: str,
    *,
    by: Iterable[str] = (),
    targets: Sequence[str] | None = None,
    spec: dict | None = None,
    weight: str | None = None,
) -> pl.DataFrame:
    """Reduce a chunk of output to the sufficient statistics of its metrics
    (docs/ENHANCEMENTS.md E49).

    :func:`metrics` needs the whole frame. This needs one chunk at a time:
    ten doubles per ``(slot, target, *by)``, which :func:`merge_sums` adds
    together and :func:`from_sums` turns back into the same numbers. A run
    that compares fifty slots over a billion rows then keeps ten doubles per
    key instead of writing the rows out to evaluate them later.

    The columns beside the keys are :data:`SUM_FIELDS`: ``n`` rows and ``w``
    weight behind them, the weighted means ``mean_y`` and ``mean_pred``, the
    **centred** sums ``m2_y``, ``m2_pred`` and ``cov``, the residual sum of
    squares ``sse``, and ``hits`` / ``signed`` for the hit rate.

    Centred, not raw. The obvious form -- keep ``sum(y)`` and ``sum(y**2)``
    and subtract -- is one addition simpler and loses the variance entirely
    when the mean is large relative to the spread: a unit-variance target
    around 1e8 has ``var / E[y**2]`` of about 1e-16, and the subtraction has
    nothing left. :func:`merge_sums` pays for the centring with a
    parallel-axis term, which is a multiply, and keeps every digit. It is the
    same choice `EwCov` makes for the same reason (E11b).

    Rows where the prediction or the target is null are dropped, as
    :func:`metrics` drops them, so warmup and skipped rows never enter the
    numbers. ``weight`` names a column to weight rows by; without it every row
    counts 1 and ``w`` equals ``n``. ``spec``, ``targets`` and the errors are
    :func:`unpack`'s.
    """
    long = unpack(df, spec_name, spec=spec, targets=targets).drop_nulls(["pred", "y"])
    wexpr = pl.col(weight).cast(pl.Float64) if weight is not None else pl.lit(1.0)
    long = long.with_columns(wexpr.alias("__w"))
    w, y, p = pl.col("__w"), pl.col("y"), pl.col("pred")
    tw = w.sum()
    my, mp = (w * y).sum() / tw, (w * p).sum() / tw
    keys = ["slot", "target", *by]
    return (
        long.group_by(keys)
        .agg(
            pl.len().alias("n"),
            tw.alias("w"),
            my.alias("mean_y"),
            mp.alias("mean_pred"),
            (w * (y - my) ** 2).sum().alias("m2_y"),
            (w * (p - mp) ** 2).sum().alias("m2_pred"),
            (w * (y - my) * (p - mp)).sum().alias("cov"),
            (w * (y - p) ** 2).sum().alias("sse"),
            (w * ((y.sign() == p.sign()) & (y != 0))).sum().alias("hits"),
            (w * (y != 0)).sum().alias("signed"),
        )
        .sort(keys)
    )


def merge_sums(first: pl.DataFrame, *rest: pl.DataFrame) -> pl.DataFrame:
    """Add the sufficient statistics of disjoint row sets.

    Exact, whatever the split: the means are pooled by weight and the centred
    sums pick up the parallel-axis term for the distance between each part's
    mean and the pooled one::

        w    = sum(w_g)
        mean = sum(w_g * mean_g) / w
        m2   = sum(m2_g + w_g * (mean_g - mean)**2)
        cov  = sum(cov_g + w_g * (mean_y_g - mean_y) * (mean_p_g - mean_p))

    That is the n-way form of Chan, Golub and LeVeque's merge -- every part
    enters as a sum, never as a difference of running totals -- so merging a
    thousand chunks loses no more than merging two.

    Keys present in one part and not another are carried through as they are.
    Merging one frame returns it unchanged.
    """
    frames = [first, *rest]
    keys = [c for c in frames[0].columns if c not in SUM_FIELDS]
    for f in frames[1:]:
        other = [c for c in f.columns if c not in SUM_FIELDS]
        if other != keys:
            msg = f"merge_sums needs the same keys in every part: {keys} vs {other}"
            raise ValueError(msg)
    w = pl.col("w").sum()
    mean_y = (pl.col("w") * pl.col("mean_y")).sum() / w
    mean_p = (pl.col("w") * pl.col("mean_pred")).sum() / w
    return (
        pl.concat(frames)
        .group_by(keys)
        .agg(
            pl.col("n").sum(),
            w.alias("w"),
            mean_y.alias("mean_y"),
            mean_p.alias("mean_pred"),
            (pl.col("m2_y") + pl.col("w") * (pl.col("mean_y") - mean_y) ** 2).sum().alias("m2_y"),
            (pl.col("m2_pred") + pl.col("w") * (pl.col("mean_pred") - mean_p) ** 2)
            .sum()
            .alias("m2_pred"),
            (
                pl.col("cov")
                + pl.col("w") * (pl.col("mean_y") - mean_y) * (pl.col("mean_pred") - mean_p)
            )
            .sum()
            .alias("cov"),
            pl.col("sse").sum(),
            pl.col("hits").sum(),
            pl.col("signed").sum(),
        )
        .select(*keys, *SUM_FIELDS)
        .sort(keys)
    )


def from_sums(s: pl.DataFrame, *, min_obs: int = 30) -> pl.DataFrame:
    """The metrics :func:`metrics` reports, from :func:`sums` instead of rows.

    Same columns and same numbers: ``n``, ``r2`` (out-of-sample against the
    realized mean), ``ic`` (correlation of prediction with target),
    ``hit_rate``, ``mse``, and ``rmse`` beside it. A key with fewer than
    ``min_obs`` rows is dropped, as :func:`metrics` drops it.

    ``r2`` and ``ic`` are ``null`` where they are undefined -- a key whose
    target or prediction never varied has no correlation to report, and
    dividing by its zero variance would give an infinity that reads as a
    number.
    """
    keys = [c for c in s.columns if c not in SUM_FIELDS]
    mse = pl.col("sse") / pl.col("w")
    denom = (pl.col("m2_y") * pl.col("m2_pred")).sqrt()
    return (
        s.filter(pl.col("n") >= min_obs)
        .select(
            *keys,
            pl.col("n"),
            pl.when(pl.col("m2_y") > 0)
            .then(1.0 - pl.col("sse") / pl.col("m2_y"))
            .otherwise(None)
            .alias("r2"),
            pl.when(denom > 0).then(pl.col("cov") / denom).otherwise(None).alias("ic"),
            pl.when(pl.col("signed") > 0)
            .then(pl.col("hits") / pl.col("signed"))
            .otherwise(None)
            .alias("hit_rate"),
            mse.alias("mse"),
            mse.sqrt().alias("rmse"),
        )
        .sort(keys)
    )
