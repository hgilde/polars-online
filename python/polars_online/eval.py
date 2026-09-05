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

__all__ = ["metrics", "rolling_metrics", "compare_specs", "unpack"]


def _pred_fields(df: pl.DataFrame, spec_name: str) -> list[str]:
    dtype = df.schema[spec_name]
    if not isinstance(dtype, pl.Struct):
        msg = f"column {spec_name!r} is not a model-output struct"
        raise TypeError(msg)
    fields = [f.name for f in dtype.fields if f.name.startswith("pred_")]
    if not fields:
        msg = (
            f"column {spec_name!r} has no prediction fields (pred_*) to unpack; "
            "an ew_cov struct holds statistics and a kmeans or micro struct assignments, "
            "not predictions"
        )
        raise TypeError(msg)
    return fields


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
    ``pred_*`` fields (an ``ew_cov``, ``kmeans`` or ``micro`` output); ``ValueError``
    when a slot's
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
