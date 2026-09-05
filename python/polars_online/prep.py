"""Frame preparation for streams whose labels arrive late
(docs/ENHANCEMENTS.md E47).

One function so far. :func:`embargo` turns a frame into the doubled stream
that a forward-looking target needs: every row appears twice, once as a
prediction at its own clock with zero weight, and once as a lesson at
``clock + delay``, the two merged back into clock order. It is the recipe a
spec's ``label_delay`` runs natively, written out in Polars -- useful for
seeing what the delay does, for a model that has no ``label_delay``, and as
the oracle the native path is tested against.

Everything here is lazy and streaming: ``merge_sorted`` on two sorted halves
of the same frame, so a stream too long to hold is still too long to hold and
this does not change that.
"""

from __future__ import annotations

import polars as pl

__all__ = ["embargo"]

#: Column :func:`embargo` adds to say which copy of a row this is.
ROLE = "_online_role"


def embargo(
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    clock: str,
    delay: float,
    weight: str | None = None,
    role: str = ROLE,
) -> pl.LazyFrame:
    """The doubled stream for a target that is only known ``delay`` later.

    Every row comes back twice, in clock order:

    - a **predict** row at ``clock``, with its weight forced to 0, so the
      model scores it and learns nothing from it;
    - a **learn** row at ``clock + delay``, carrying the same features and
      target at full weight.

    A ``role`` column says which is which (``"predict"`` / ``"learn"``), so
    the output is filtered back down with
    ``out.filter(pl.col(role) == "predict")``.

    Why bother: a target that is a forward quantity over ``delay`` clock
    units is not known at the row it sits on. A stream that learns it there
    has seen ``delay`` of the future before predicting the rows in between,
    and every "out-of-sample" number after that is contaminated -- with an
    autocorrelated feature, even a pure noise column will show a correlation
    with its target. Zero-weight rows are legal and mean "advance the clock,
    learn nothing", so the doubled stream says exactly what is wanted:
    predict here, learn later.

    ``weight`` names an existing weight column; without it the function adds
    one (named ``role + "_weight"``) that is 1 on learn rows and 0 on predict
    rows -- pass that name to the spec's ``weight=``.

    The frame must already be in ``clock`` order, as a stream must be. The
    result is sorted by ``clock`` with **learn rows before predict rows** at
    the same clock value: a label whose ``delay`` has just run out is known
    at that instant, so a prediction made then may use it. A spec's
    ``label_delay`` releases in the same order, which is what lets the two
    be compared row for row.

    ``delay`` must be finite and positive; ``0`` would be the undoubled
    stream, and negative is a label from the past, which is not what this is
    for.

    A spec's ``label_delay=`` does the same thing in the stream with no
    doubling and no filtering, which is cheaper and does not need the frame
    rewritten. Reach for this when a delay has to be visible in the data --
    an oracle, a demonstration, or an engine that is not this one.
    """
    if not (delay > 0.0) or delay == float("inf"):
        msg = f"embargo: delay must be finite and > 0, got {delay!r}"
        raise ValueError(msg)
    lazy = lf.lazy()
    schema = lazy.collect_schema()
    if clock not in schema:
        msg = f"embargo: no clock column {clock!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    if weight is not None and weight not in schema:
        msg = f"embargo: no weight column {weight!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    if role in schema:
        msg = f"embargo: the frame already has a column named {role!r}; pass another `role=`"
        raise ValueError(msg)

    wcol = weight if weight is not None else f"{role}_weight"
    # `merge_sorted` needs both halves sorted on the key it merges by. Each
    # half is the input in its own order, so a single key sorts both: the
    # clock, with the learn copy first at a tie.
    predict = lazy.with_columns(
        pl.lit("predict").alias(role),
        (pl.col(weight) * 0.0 if weight is not None else pl.lit(0.0)).alias(wcol),
        pl.lit(1, pl.UInt8).alias("__embargo_order"),
    )
    learn = lazy.with_columns(
        pl.lit("learn").alias(role),
        (pl.col(weight) if weight is not None else pl.lit(1.0)).alias(wcol),
        (pl.col(clock) + delay).alias(clock),
        pl.lit(0, pl.UInt8).alias("__embargo_order"),
    )
    # One sort key, so the merge is by (clock, order): at a tie the lesson
    # lands before the prediction that may use it.
    key = "__embargo_key"
    both = [
        f.with_columns(
            pl.struct(pl.col(clock), pl.col("__embargo_order")).alias(key),
        )
        for f in (predict, learn)
    ]
    return (
        both[0]
        .merge_sorted(both[1], key=key)
        .drop(key, "__embargo_order")
        .select(*schema.names(), *([] if weight is not None else [wcol]), role)
    )
