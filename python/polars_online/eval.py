"""Evaluation harness (docs/PLAN.md section 8).

Pure Polars over the output structs: per (spec, group, target) out-of-sample
R^2, IC (correlation of prediction with realized target) and hit rate, either
overall or in rolling windows measured in clock units. One ``group_by``
compares specs.

Everything here consumes the frame a :class:`~polars_online.ModelBank` returns:
the original columns plus one struct column per spec.
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
    return [f.name for f in dtype.fields if f.name.startswith("pred_")]


#: Column names :func:`unpack` produces. Input columns with these names are
#: dropped rather than duplicated (the target's values come back as ``y``).
RESERVED = ("slot", "target", "pred", "y")


def unpack(
    df: pl.DataFrame,
    spec_name: str,
    *,
    targets: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Long form: one row per (row, prediction slot).

    Returns ``slot`` (the struct field name), ``target`` (the target column it
    predicts), ``pred``, ``y`` and every other non-struct column of ``df``.
    Input columns named like the output ones (see :data:`RESERVED`) are dropped
    -- a target column called ``y`` would otherwise collide with the ``y``
    output.
    """
    fields = _pred_fields(df, spec_name)
    keep = [c for c, d in df.schema.items() if not isinstance(d, pl.Struct) and c not in RESERVED]
    frames = []
    for slot in fields:
        target = _target_of(slot, df, targets)
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
    """Overall out-of-sample metrics per (slot, *by).

    Rows where the prediction or the target is null are dropped, so warmup and
    skipped rows never enter the numbers.
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

    ``window_start`` is the left edge of each bucket (``floor(clock/window)*window``).
    """
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
    """Stack :func:`metrics` for several specs, adding a ``spec`` column."""
    frames = [
        metrics(df, name, by=by, targets=targets, min_obs=min_obs).with_columns(
            pl.lit(name).alias("spec")
        )
        for name in spec_names
    ]
    cols = ["spec", *frames[0].columns[:-1]] if frames else []
    return pl.concat(frames).select(cols) if frames else pl.DataFrame()
