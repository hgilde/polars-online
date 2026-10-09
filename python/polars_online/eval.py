"""Scoring a bank's output: out-of-sample R², IC and hit rate per (spec, group,
target), overall or in windows, and the sums that give them one chunk at a
time.

Everything here consumes the frame a :class:`~polars_online.ModelBank`
returns: the original columns plus one struct column per spec. The functions
are polars aggregations over collected output, so they need the whole frame; a
spec's ``emit_metrics`` keeps the same numbers beside the model, O(state), for
a stream too long to hold, and :func:`sums`, :func:`merge_sums` and
:func:`from_sums` reduce each chunk to ten numbers per key and add them up
exactly.

A target that is not a column under its own name -- a column renamed with
:func:`polars_online.target`, or a window expression looking ahead -- is
read through the spec that wrote the output: pass it as ``spec`` (see
:func:`unpack`).

What each function raises is what :func:`unpack` raises, since each starts
there: ``KeyError`` for a ``spec_name`` the frame has not got, ``TypeError``
for a column that is not a model's prediction struct, ``ValueError`` for a
slot whose target column cannot be found. A ``group`` or ``clock`` column
the frame has not got is polars' ``ColumnNotFoundError``.

``group`` names one key column, as a spec's ``group`` does, or a list of
them. ``min_samples`` is the fewest rows a key is reported on, Polars' name
for that count. Before 1.0 they were ``by`` and ``min_obs``, and
:func:`window_metrics` was ``rolling_metrics``. An old name is refused, and
the error names the new one.

`docs/DIAGNOSTICS.md <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md>`_
says which question each call and each diagnostic switch answers, with its
threshold, its false-alarm rate as measured, and a recipe that runs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl

from polars_online._duration import Duration, clock_nanoseconds
from polars_online._polars_online import format_duration
from polars_online._renamed import renamed_function, renamed_keywords

__all__ = [
    "metrics",
    "window_metrics",
    "compare_specs",
    "unpack",
    "seqtest",
    "diebold_mariano",
    "clark_west",
    "sums",
    "merge_sums",
    "from_sums",
    "SUM_FIELDS",
    "RESERVED",
]

# The keywords renamed before 1.0 (docs/PLAN.md task 197): `min_obs` is
# Polars' `min_samples` for the same count, and `by` is `group`, the specs'
# word since 0.12.0 (review round 4, AP6); `window_metrics`' `every` was
# `rolling_metrics`' `window_size` (YB5). Literal tables, each whole: Sphinx's
# source parser fails on a `{**a, **b}` in a decorator, and then drops every
# `#:` comment of the module, so `SUM_FIELDS` and `RESERVED` went unrendered.
_GROUP = {"by": "group"}
_MIN_SAMPLES = {"min_obs": "min_samples"}
_BOTH = {"by": "group", "min_obs": "min_samples"}
_WINDOW = {"by": "group", "min_obs": "min_samples", "window_size": "every"}

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
    """A spec's output in long form: one row per (row, prediction slot).

    Returns the non-struct columns of ``df`` plus:

    ``slot``
        The struct field name (``pred_y__r0.5@h500``).
    ``target``
        The target it predicts: the name its output fields carry, which for a
        plain target is its column.
    ``pred``, ``y``
        The prediction and the target's value.

    Input columns named like these four (:data:`RESERVED`) are dropped rather than
    duplicated: a target column called ``y`` would otherwise collide with the
    ``y`` output.

    Pass ``spec`` to read each slot's target as the spec writes it, through
    :func:`polars_online.spec.output_index` and the spec's ``targets``. A
    plain target is its column, and so is a column renamed with ``name=``. A
    formula target is read from a column of its own name, which
    :func:`polars_online.stream.with_windows` makes; the bank resolves it
    inside its stream and does not write it out.

    Without ``spec`` a name-based rule is used: a slot is read against the
    column named after it. That can misattribute a target whose name embeds
    another's, and a renamed target (``po.target("p", name="price")`` writes
    ``pred_price``) names no column, so only ``spec`` can say what it read.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        long = po.eval.unpack(out, "ridge")       # slot, target, pred, y, and the input columns

    Raises:

    - ``KeyError`` for a ``spec_name`` the frame has not got;
    - ``TypeError`` for a column that is not a struct, or a struct with no
      ``pred_*`` fields (an ``ew_cov``, ``kmeans``, ``micro`` or ``ew_class``
      output);
    - ``ValueError`` when a slot's target column cannot be found (the frame
      has not got it, or ``targets`` does not name it, or without ``spec`` no
      column is named after the slot) and, with ``spec``, whatever
      :func:`polars_online.spec.output_index` raises for it.
    """
    fields = _pred_fields(df, spec_name)
    keep = [c for c, d in df.schema.items() if not isinstance(d, pl.Struct) and c not in RESERVED]
    # With the spec in hand, the slot -> target mapping comes from
    # `output_index` -- the same Rust code that named the fields -- and each
    # target's value from the spec's own table of it, instead of the
    # heuristic name-parsing below, which exists only for callers who have a
    # frame but no spec.
    exact = {} if spec is None else _spec_targets(spec, df)
    frames = []
    for slot in fields:
        if slot in exact:
            target, y = exact[slot]
        else:
            target = _target_of(slot, df, targets)
            y = pl.col(target)
        frames.append(
            df.select(
                *keep,
                pl.lit(slot).alias("slot"),
                pl.lit(target).alias("target"),
                df[spec_name].struct.field(slot).alias("pred"),
                y.alias("y"),
            )
        )
    return pl.concat(frames)


def _spec_targets(spec: dict, df: pl.DataFrame) -> dict[str, tuple[str, pl.Expr]]:
    """Each ``pred`` field of ``spec`` -> its target's name and the
    expression of the target's value on the frame."""
    from polars_online import spec as _spec_mod
    from polars_online._spec import target_name

    tables = {target_name(t): t for t in spec.get("targets") or []}
    idx = _spec_mod.output_index(spec).filter(pl.col("kind") == "pred")
    return {
        r["field"]: (
            r["target"],
            _target_value(r["field"], tables.get(r["target"]), r["target"], df),
        )
        for r in idx.iter_rows(named=True)
        if r["target"] is not None
    }


def _target_value(slot: str, table: Any, name: str, df: pl.DataFrame) -> pl.Expr:
    """A target's value on the frame as the bank reads it: its column, or
    for a formula target the column of its own name that
    :func:`polars_online.stream.with_windows` writes. A value the bank
    would not learn from is dropped by :func:`_scored`."""

    def column(c: str) -> pl.Expr:
        if c not in df.columns:
            msg = (
                f"cannot find column {c!r}, which slot {slot!r} reads for target {name!r}; the "
                f"frame has {df.columns}"
            )
            raise ValueError(msg)
        return pl.col(c)

    if table is None or isinstance(table, str):
        return column(name if table is None else table)
    if "formula" in table:
        if name not in df.columns:
            msg = (
                f"slot {slot!r}: target {name!r} is a formula of the row's future, which the bank "
                f"resolves inside its stream and does not write out, and the frame has no column "
                f"{name!r} to score it against; add one with po.stream.with_windows"
            )
            raise ValueError(msg)
        return pl.col(name)
    return column(table["column"])  # a column renamed


def _target_of(slot: str, df: pl.DataFrame, targets: Sequence[str] | None) -> str:
    """``pred_<target>[__combo][@hN]`` -> ``<target>``. Longest match wins, so
    targets whose names are prefixes of each other still resolve."""
    body = slot[len("pred_") :]
    candidates = targets if targets is not None else list(df.columns)
    matches = [
        t for t in candidates if body == t or body.startswith(t + "__") or body.startswith(t + "@")
    ]
    if not matches:
        msg = (
            f"cannot infer the target column for slot {slot!r}: no column of the frame is named "
            "after it. A target with a name of its own (po.target with name=) is read through "
            "the spec that wrote it: pass spec="
        )
        raise ValueError(msg)
    return max(matches, key=len)


def _group_keys(group: str | Iterable[str]) -> list[str]:
    """``group`` as the list of its key columns: a bare string is one key,
    as in Polars' ``group_by`` and ``over`` and a spec's ``group``. Iterated
    as it was, ``by="group"`` was the keys ``g``, ``r``, ``o``, ``u``, ``p``
    (review round 4, YB16)."""
    return [group] if isinstance(group, str) else list(group)


def _usable(value: pl.Expr) -> pl.Expr:
    """Whether a float is one the bank learns from: finite, and within its
    input bound (docs/PLAN.md section 3); null where ``value`` is."""
    return value.is_finite() & (value.abs() <= _INPUT_BOUND)


def _scored(long: pl.DataFrame) -> pl.DataFrame:
    """The rows of :func:`unpack`'s long form whose prediction and target are
    both values the bank would learn from: not null, not NaN, not infinite and
    not past its input bound (docs/PLAN.md section 3). The bank reads such a
    target as missing and still predicts its row, so dropping nulls alone kept
    the row, and ``r2``, ``ic`` and ``mse`` came out NaN while ``hit_rate``
    was quietly lowered (review 2026-10-05, YB4)."""
    return long.filter(
        _usable(pl.col("pred").cast(pl.Float64)) & _usable(pl.col("y").cast(pl.Float64))
    )


def _no_hit(spec: dict | None) -> bool:
    """Whether ``spec`` names a fit with no sign to hit, whose ``hit_rate``
    the bank's ``emit_metrics`` nulls: a Poisson ``sgd`` (``loss =
    "poisson"``), where a rate is positive and a count never negative, so
    the sign test about zero agrees on every row and reads 1.0 whatever the
    fit (docs/PLAN.md task 195, S5; review round 5, E1). Without ``spec``
    the loss cannot be known, and the sign test is read."""
    model = (spec or {}).get("model") or {}
    return model.get("type") == "sgd" and model.get("loss") == "poisson"


def _metric_exprs(min_samples: int, *, binary: bool = False, hit: bool = True) -> list[pl.Expr]:
    resid = pl.col("y") - pl.col("pred")
    # The centred sums, as :func:`sums` keeps them: a metric is null where
    # one it divides by is 0, as :func:`from_sums` gives it, where this gave
    # -inf and NaN (review 2026-10-05, YB9).
    m2_y = (pl.col("y") - pl.col("y").mean()).pow(2).sum()
    m2_pred = (pl.col("pred") - pl.col("pred").mean()).pow(2).sum()
    exprs = [
        pl.len().alias("n"),
        # Out-of-sample R^2 against the realized mean of y in the window --
        # the Brier skill score when `binary` (`pred` a probability, `y` a
        # 0/1 label): same formula, different name (docs/PLAN.md task 76).
        pl.when(m2_y > 0).then(1.0 - resid.pow(2).sum() / m2_y).alias("r2"),
        # Correlation of prediction with target -- the point-biserial
        # correlation when `binary`, again the same formula.
        pl.when(m2_y * m2_pred > 0).then(pl.corr("pred", "y")).alias("ic"),
    ]
    if binary:
        # Accuracy at a 0.5 threshold. Every row scores: 0 is one of the two
        # classes here, not the sign test's excluded case, and `pred` and `y`
        # are both positive by construction, so the sign test below always
        # agrees and is not a hit rate at all on this kind of fit (task 76
        # found it reading exactly 1.0 on a fit that had learned nothing).
        exprs.append(
            (((pl.col("pred") > 0.5) == (pl.col("y") > 0.5)).sum() / pl.len()).alias("hit_rate")
        )
        # Log loss: -(y*ln(p) + (1-y)*ln(1-p)), mean over the window. Lives
        # here, over a collected frame, rather than in the streaming
        # `emit_metrics` -- an EW accumulator would put a `ln` result into
        # model state, which `docs/PLAN.md` §11a's B4 rule forbids (glibc's
        # last bit is not Apple's, and that difference broke a frozen state
        # fixture on Linux alone). Clipped so a confident, correct prediction
        # is a strong negative number rather than -inf.
        p = pl.col("pred").clip(1e-15, 1.0 - 1e-15)
        exprs.append(
            (-(pl.col("y") * p.log() + (1.0 - pl.col("y")) * (1.0 - p).log()))
            .mean()
            .alias("log_loss")
        )
    elif not hit:
        # A fit with no sign to hit (`_no_hit`): null, as the bank nulls it.
        exprs.append(pl.lit(None, dtype=pl.Float64).alias("hit_rate"))
    else:
        # Fraction of rows where the sign of pred matches the sign of y
        # (rows with y == 0 or pred == 0 excluded: neither up nor down, not a
        # class), and null where no row has a sign. A prediction of exactly
        # zero is left out as the bank leaves it out (docs/PLAN.md task 195,
        # S6): polars' `sign(0)` is 0, a miss here, where the bank's
        # `signum(+0.0)` was 1, a hit.
        y, pred = pl.col("y"), pl.col("pred")
        scored = (y != 0) & (pred != 0)
        signed = scored.sum()
        hits = ((pred.sign() == y.sign()) & scored).sum()
        exprs.append(pl.when(signed > 0).then(hits.truediv(signed)).alias("hit_rate"))
    exprs.append(resid.pow(2).mean().alias("mse"))
    exprs.append(pl.when(pl.len() >= min_samples).then(True).otherwise(False).alias("enough"))
    return exprs


@renamed_keywords("po.eval.metrics", _BOTH)
def metrics(
    df: pl.DataFrame,
    spec_name: str,
    *,
    group: str | Iterable[str] = (),
    targets: Sequence[str] | None = None,
    spec: dict | None = None,
    min_samples: int = 30,
    binary: bool = False,
) -> pl.DataFrame:
    """Out-of-sample metrics per ``(slot, target, *group)``, over the whole frame.

    Rows where the prediction or the target is missing are dropped, so warm-up
    and skipped rows never enter the numbers. Missing is what the bank reads as
    missing: null, NaN, infinite, or past its input bound of 1e100 in
    magnitude, a target the bank still predicts the row of. ``group`` is one
    key column, or a list of them. A key with fewer than ``min_samples`` rows
    left is dropped from the result rather than reported on too little.
    ``spec`` and ``targets`` are :func:`unpack`'s: pass the spec for a target
    that is not a plain column. The columns:

    ``slot``, ``target``, and the ``group`` columns
        The key.
    ``n``
        The rows counted.
    ``r2``
        Out-of-sample R² against the realized mean of the target; null where
        the target never varied.
    ``ic``
        The correlation of prediction with target; null where either never
        varied.
    ``hit_rate``
        The share of rows whose sign the prediction got right, rows with ``y ==
        0`` or ``pred == 0`` excluded, since either says neither up nor down,
        as the bank's ``emit_metrics`` takes them; null where every row is
        excluded. A target that sits about 1, such as a plain ratio of two
        prices, reads 1.0 whatever the fit, so a return is better a
        difference or a log ratio. A Poisson ``sgd`` fit (``loss =
        "poisson"``) has no sign to hit -- a rate is positive and a count
        never negative -- and the bank nulls its ``hit_rate``: with ``spec``
        naming one, null here too. Without ``spec`` the loss cannot be
        known, and the sign test reads its 1.0.
    ``mse``
        The mean squared residual.

    A metric that is undefined is null, as :func:`from_sums` gives it, rather
    than an infinity or a NaN that reads as a number.

    ``binary=True`` reads ``pred`` as a probability and ``y`` as a 0/1 label, the
    output of a ``sgd`` or ``ftrl`` fit with ``loss = "logistic"``. ``hit_rate``
    becomes accuracy at a 0.5 threshold and every row scores. A ``log_loss``
    column is added, ``-(y*ln(p) + (1-y)*ln(1-p))`` averaged. ``r2`` and ``ic``
    keep their formulas under names that mean something different on a 0/1
    target, the Brier skill score and the point-biserial correlation. Nothing here
    reads the target's values to decide which reading applies; name it explicitly,
    the way you chose the loss.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        scores = po.eval.metrics(out, "ridge", group="stock_id")   # n, r2, ic, hit_rate, mse

    Raises as :func:`unpack` does; a ``group`` column the frame has not got is
    polars' ``ColumnNotFoundError``.
    """
    long = _scored(unpack(df, spec_name, spec=spec, targets=targets))
    keys = ["slot", "target", *_group_keys(group)]
    exprs = _metric_exprs(min_samples, binary=binary, hit=not _no_hit(spec))
    out = long.group_by(keys).agg(exprs).sort(keys)
    return out.filter(pl.col("enough")).drop("enough")


@renamed_keywords("po.eval.window_metrics", _WINDOW)
def window_metrics(
    df: pl.DataFrame,
    spec_name: str,
    *,
    clock: str,
    every: float | Duration,
    group: str | Iterable[str] = (),
    targets: Sequence[str] | None = None,
    spec: dict | None = None,
    min_samples: int = 30,
    binary: bool = False,
) -> pl.DataFrame:
    """:func:`metrics` in tumbling windows of ``every`` clock units.

    The windows do not overlap: each row falls in one, as in Polars'
    ``group_by_dynamic(every=)``. Polars' ``rolling`` windows overlap, which
    is why this is not ``rolling_metrics``, its name before 1.0.

    The columns are :func:`metrics`'s, per window, plus ``window_start``, the
    left edge of each bucket (``floor(clock / every) * every``). ``binary``,
    ``group``, ``min_samples``, ``spec`` and ``targets`` are :func:`metrics`'s.
    ``every`` is measured the way a spec's clock parameters are: a number for
    a numeric clock, and a duration for a ``Datetime``, ``Date`` or
    ``Duration`` one. ``window_start`` is of the clock's own dtype, an
    integer clock's included, whose ``every`` must then be a whole number of
    its units: a bucket's edges are values of the column.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        by_hour = po.eval.window_metrics(out, "ridge", clock="t", every=100.0)
        # on a Datetime clock: every=pl.duration(hours=1), timedelta(hours=1) or "1h"

    Raises as :func:`unpack` does, and:

    - ``ValueError`` for an ``every`` that is not finite and above 0, of the
      wrong kind for the clock, or not a whole number on an integer clock;
    - ``TypeError`` for a ``clock`` column that is neither numeric nor
      temporal;
    - polars' ``ColumnNotFoundError`` for a ``clock`` or ``group`` column the
      frame has not got.
    """
    dtype = df.schema.get(clock)
    if dtype is None:
        # Named before the window is checked against it (review R2, P9).
        raise pl.exceptions.ColumnNotFoundError(clock)
    ns = clock_nanoseconds(every, dtype, "window_metrics", "every", clock)
    if ns is None and not every > 0:  # type: ignore[operator]
        msg = f"window_metrics: every must be > 0, got {every}"
        raise ValueError(msg)
    if ns is None and math.isinf(every):  # type: ignore[arg-type]
        # One bucket, whose `window_start` was NaN (review 2026-10-05, YB11).
        msg = (
            f"window_metrics: every must be finite, got {every}; one window over the whole "
            "frame is po.eval.metrics"
        )
        raise ValueError(msg)
    if dtype is not None and not (dtype.is_numeric() or dtype.is_temporal()):
        msg = f"clock column {clock!r} must be numeric or temporal, got {dtype}"
        raise TypeError(msg)
    if ns is None and dtype.is_integer():
        # The bucket's left edge in the clock's own dtype, as a temporal
        # clock's is, in integers, so no edge is rounded through a float; a
        # window that is not whole would put edges between the column's
        # values (review round 4, YB14).
        if not float(every).is_integer():  # type: ignore[arg-type]
            msg = (
                f"window_metrics: every {every!r} is not a whole number, and clock column "
                f"{clock!r} is {dtype}, which holds whole numbers; give a whole every, or cast "
                "the clock to a float"
            )
            raise ValueError(msg)
        whole = int(every)  # type: ignore[arg-type]
        start = (pl.col(clock) // whole * whole).cast(dtype)
    elif ns is None:
        start = (pl.col(clock) / every).floor() * every
    elif isinstance(dtype, pl.Duration):
        start = (
            (pl.col(clock).dt.total_nanoseconds() // ns * ns).cast(pl.Duration("ns")).cast(dtype)
        )
    else:
        start = pl.col(clock).dt.truncate(format_duration(ns))
    long = _scored(unpack(df, spec_name, spec=spec, targets=targets)).with_columns(
        start.alias("window_start")
    )
    keys = ["slot", "target", *_group_keys(group), "window_start"]
    exprs = _metric_exprs(min_samples, binary=binary, hit=not _no_hit(spec))
    out = long.group_by(keys).agg(exprs).sort(keys)
    return out.filter(pl.col("enough")).drop("enough")


#: Renamed :func:`window_metrics`, and its ``window_size`` ``every``, before
#: 1.0 (docs/PLAN.md task 197; review round 4, YB5).
rolling_metrics = renamed_function(
    "po.eval.rolling_metrics", "po.eval.window_metrics", ", and window_size was renamed every"
)


@renamed_keywords("po.eval.compare_specs", _BOTH)
def compare_specs(
    df: pl.DataFrame,
    spec_names: Iterable[str],
    *,
    group: str | Iterable[str] = (),
    targets: Sequence[str] | None = None,
    specs: Sequence[dict] | None = None,
    min_samples: int = 30,
    binary: bool = False,
) -> pl.DataFrame:
    """:func:`metrics` for several specs in one table, with a ``spec`` column first.

    ``binary``, ``group`` and ``min_samples`` are :func:`metrics`'s, applied to
    every spec alike: compare specs that share a loss, not a logistic fit
    against a regression one. ``specs`` are the specs that wrote the columns,
    each read as :func:`metrics` reads its ``spec`` for the column of its
    name; a name without one is read by the name-based rule.

    .. code-block:: python

        ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        kalman = po.spec.kalman(
            "kalman", targets=["y"], features=["x0", "x1"], half_life=100.0, coef_half_life=50.0
        )
        out = po.ModelBank([ridge, kalman]).fit_predict(df)
        table = po.eval.compare_specs(out, ["ridge", "kalman"])   # which had the lower error

    Raises as :func:`metrics` does for each; no specs give an empty frame.
    """
    written = {s.get("name"): s for s in specs or ()}
    frames = [
        metrics(
            df,
            name,
            group=group,
            targets=targets,
            spec=written.get(name),
            min_samples=min_samples,
            binary=binary,
        ).with_columns(pl.lit(name).alias("spec"))
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


@renamed_keywords("po.eval.seqtest", _GROUP)
def seqtest(
    df: pl.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    group: str | Iterable[str] = (),
    min_weight: float = 0.0,
    name: str = "seqtest",
) -> pl.DataFrame:
    """:func:`polars_online.spec.seqtest` in polars expressions, over a frame in
    memory: the same e-processes, the same fields, row for row.

    Column mode (no ``a`` and ``b``): ``targets`` name the columns whose sign
    is tested. Compare mode: ``a`` and ``b`` name two output structs of
    ``df`` (two specs the bank ran), and ``targets`` the residuals both
    carry. ``t`` means ``resid_<t><a_suffix>`` of ``a`` against
    ``resid_<t><b_suffix>`` of ``b``, every ``t`` they share when ``None``.
    The sign tested is that of ``|resid_b| - |resid_a|``, positive when
    ``a`` was closer.

    Per target, with ``s`` the sign and the counts before the row:

    .. code-block:: text

        lam_pos = max(0, (n_pos - n_neg) / (n_pos + n_neg + 1))
        log_e_pos += log1p(lam_pos * s)         (lam_neg, log_e_neg likewise)

    Returns ``df`` with a struct column ``name`` holding, per target ``t``
    and read before the row, as the bank writes them:

    ``log_e_pos_<t>``, ``log_e_neg_<t>``, ``n_pos_<t>``, ``n_neg_<t>``
        In column mode: the two gamblers' log wealth and the signs counted.
    ``log_e_a_<t>``, ``log_e_b_<t>``, ``wins_a_<t>``, ``wins_b_<t>``
        In compare mode: the same, for "``a`` was closer" and "``b`` was
        closer".
    ``weight_sum``
        The rows before this one in its group; every other field is null
        until it reaches ``min_weight``.

    ``group`` runs one process per group, in row order (``.over(group)``):
    one key column, as a spec's ``group`` is, or a list of them. A null,
    zero or NaN value bets nothing and counts nothing, as in the bank. What
    the bank adds is the clock (``session``, ``restart_after_step_back``),
    which a frame in memory has not got. The bank's struct is held to this
    one to the last bit; the difference is that the bank is O(state) over a
    stream and this is O(rows) over a frame.

    .. code-block:: python

        ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        kalman = po.spec.kalman(
            "kalman", targets=["y"], features=["x0", "x1"], half_life=100.0, coef_half_life=50.0
        )
        out = po.ModelBank([ridge, kalman]).fit_predict(df)
        evidence = po.eval.seqtest(out, a="kalman", b="ridge", group="stock_id")

    Raises:

    - ``ValueError`` for ``a`` without ``b`` (or the reverse), for column
      mode without ``targets``, and for a target neither side has a residual
      for (naming the fields it does have);
    - ``KeyError`` for a spec the frame has not got;
    - ``TypeError`` for one that is not a struct.
    """
    keys = _group_keys(group)
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

    # Two passes, so that no window sits inside another: the log wealth is a
    # running sum over the rows before, of a bet sized by counts that are
    # themselves running sums. Polars 1.34.0 refuses a window inside a
    # window ("window expression not allowed in aggregation"; 1.44.2 takes
    # it), which failed every `group` at the declared floor (task 109's floor run,
    # 2026-09-27). The counts go into columns first, then the wealth reads
    # them; the arithmetic is the same.
    taken = set(df.columns)

    def temp(label: str) -> str:
        column = f"__po_seqtest_{label}"
        while column in taken:
            column = "_" + column
        taken.add(column)
        return column

    n_eff_col = temp("weight_sum")
    first: list[pl.Expr] = [over(pl.int_range(pl.len())).cast(pl.Float64).alias(n_eff_col)]
    staged: dict[str, tuple[str, str, str]] = {}
    for i, (t, d) in enumerate(signs.items()):
        # What the bank does not learn from -- null, NaN, an infinity, a
        # magnitude beyond its input bound (docs/PLAN.md section 3) -- is no
        # sign here either. Polars orders NaN above every float, so without
        # this `NaN > 0` would be a positive sign.
        d = d.cast(pl.Float64)
        d = pl.when(d.is_finite() & (d.abs() <= _INPUT_BOUND)).then(d)
        cols = (temp(f"s_{i}"), temp(f"n_pos_{i}"), temp(f"n_neg_{i}"))
        first += [
            pl.when(d > 0).then(1.0).when(d < 0).then(-1.0).otherwise(0.0).alias(cols[0]),
            before((d > 0).cast(pl.Int64).fill_null(0)).alias(cols[1]),
            before((d < 0).cast(pl.Int64).fill_null(0)).alias(cols[2]),
        ]
        staged[t] = cols
    ready = pl.col(n_eff_col) >= min_weight
    fields: list[pl.Expr] = []
    for t, (s_col, pos_col, neg_col) in staged.items():
        s, n_pos, n_neg = pl.col(s_col), pl.col(pos_col), pl.col(neg_col)
        n1 = (n_pos + n_neg + 1).cast(pl.Float64)
        lam_pos = pl.max_horizontal((n_pos - n_neg).cast(pl.Float64) / n1, 0.0)
        lam_neg = pl.max_horizontal((n_neg - n_pos).cast(pl.Float64) / n1, 0.0)
        log_e_pos = before((lam_pos * s).log1p())
        log_e_neg = before((-lam_neg * s).log1p())
        for label, e in zip(names, (log_e_pos, log_e_neg, n_pos, n_neg), strict=True):
            fields.append(pl.when(ready).then(e).alias(f"{label}_{t}"))
    fields.append(pl.col(n_eff_col).alias("weight_sum"))
    temps = [n_eff_col, *(c for cols in staged.values() for c in cols)]
    return df.with_columns(first).with_columns(pl.struct(fields).alias(name)).drop(temps)


@renamed_keywords("po.eval.sums", _GROUP)
def sums(
    df: pl.DataFrame,
    spec_name: str,
    *,
    group: str | Iterable[str] = (),
    targets: Sequence[str] | None = None,
    spec: dict | None = None,
    weight: str | None = None,
    binary: bool = False,
) -> pl.DataFrame:
    """Reduce a chunk of output to the sufficient statistics of its metrics.

    :func:`metrics` needs the whole frame. This needs one chunk at a time: ten
    doubles per ``(slot, target, *group)``, which :func:`merge_sums` adds together
    and :func:`from_sums` turns back into the same numbers. A run that compares
    fifty slots over a billion rows then keeps ten doubles per key instead of
    writing the rows out to evaluate them later. The columns beside the keys are
    :data:`SUM_FIELDS`:

    ``n``, ``w``
        The rows, and the weight behind them.
    ``mean_y``, ``mean_pred``
        The weighted means.
    ``m2_y``, ``m2_pred``, ``cov``
        The centred second moments, and the centred cross-moment.
    ``sse``
        The residual sum of squares.
    ``hits``, ``signed``
        For the hit rate: ``hits`` counts sign agreements and ``signed`` the rows
        with ``y != 0`` and ``pred != 0``, as :func:`metrics` takes them. Under
        ``binary=True`` (:func:`metrics`'s reading), ``hits`` counts agreement
        at a 0.5 threshold and ``signed`` every row, since every row scores.
        With ``spec`` naming a Poisson ``sgd`` fit, which has no sign to hit,
        both are 0, and :func:`from_sums` gives the null the bank gives.

    Centred, not raw. The obvious form (keep ``sum(y)`` and ``sum(y**2)`` and
    subtract) is one addition simpler and loses the variance entirely when the
    mean is large relative to the spread. A unit-variance target around 1e8 has
    ``var / E[y**2]`` of about 1e-16, and the subtraction has nothing left. The
    centring takes :func:`merge_sums` a parallel-axis term, which is a multiply,
    and keeps every digit. Chunks reduced with different ``binary``
    settings must not be merged: :func:`merge_sums` sums whatever is in ``hits``
    and ``signed`` without knowing which reading produced it.

    Rows where the prediction or the target is missing are dropped, as
    :func:`metrics` drops them: null, NaN, infinite or past the bank's input
    bound. ``weight`` names a column to weight rows by; a row whose weight
    the bank would not learn from -- null, NaN, infinite, past the input
    bound, or negative -- is dropped too, and a zero weight is kept, counted
    in ``n`` and not in ``w``. Without ``weight`` every row counts 1 and ``w``
    equals ``n``. ``group`` is :func:`metrics`'s; ``spec``, ``targets`` and the
    errors are :func:`unpack`'s.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        part = po.eval.sums(out.head(200), "ridge", group="stock_id")   # ten numbers per key
        rest = po.eval.sums(out.tail(200), "ridge", group="stock_id")
        running = po.eval.merge_sums(part, rest)                     # exact, whatever the split
        scores = po.eval.from_sums(running, min_samples=10)          # r2, ic, hit_rate, mse, rmse

    """
    long = _scored(unpack(df, spec_name, spec=spec, targets=targets))
    if weight is not None:
        # What the bank does with the same weight: it skips a row whose
        # weight it cannot use and refuses a negative one (review round 4,
        # YB21: a null weight counted in `n` and not in `w`).
        wexpr = pl.col(weight).cast(pl.Float64)
        long = long.filter(_usable(wexpr) & (wexpr >= 0.0))
    else:
        wexpr = pl.lit(1.0)
    long = long.with_columns(wexpr.alias("__w"))
    w, y, p = pl.col("__w"), pl.col("y"), pl.col("pred")
    tw = w.sum()
    my, mp = (w * y).sum() / tw, (w * p).sum() / tw
    if binary:
        hits, signed = w * ((p > 0.5) == (y > 0.5)), w
    else:
        # A target or a prediction of zero has no sign, as in :func:`metrics`
        # and the bank (docs/PLAN.md task 195, S6); a fit with no sign to
        # hit signs no row at all, so :func:`from_sums` gives null (E1).
        scored = pl.lit(False) if _no_hit(spec) else (y != 0) & (p != 0)
        hits, signed = w * ((y.sign() == p.sign()) & scored), w * scored
    keys = ["slot", "target", *_group_keys(group)]
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
            hits.sum().alias("hits"),
            signed.sum().alias("signed"),
        )
        .sort(keys)
    )


def merge_sums(first: pl.DataFrame, *rest: pl.DataFrame) -> pl.DataFrame:
    """Add the sufficient statistics of disjoint row sets.

    Exact, whatever the split: the means are pooled by weight and the centred sums
    pick up the parallel-axis term for the distance between each part's mean and
    the pooled one:

    .. code-block:: text

        w    = sum(w_g)
        mean = sum(w_g * mean_g) / w
        m2   = sum(m2_g + w_g * (mean_g - mean)**2)
        cov  = sum(cov_g + w_g * (mean_y_g - mean_y) * (mean_p_g - mean_p))

    That is the n-way form of Chan, Golub and LeVeque's merge: every part enters
    as a sum, never as a difference of running totals, so merging a thousand
    chunks loses no more than merging two. Returns a frame with the same columns.
    Keys present in one part and not another are carried through as they are;
    merging one frame returns it unchanged. ``ValueError`` for parts whose key
    columns differ.
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


@renamed_keywords("po.eval.from_sums", _MIN_SAMPLES)
def from_sums(s: pl.DataFrame, *, min_samples: int = 30) -> pl.DataFrame:
    """The metrics :func:`metrics` reports, from :func:`sums` instead of rows.

    Same columns and same numbers: ``n``, ``r2`` (out-of-sample against the
    realized mean), ``ic`` (correlation of prediction with target), ``hit_rate``,
    ``mse``, and ``rmse`` beside it. Which reading ``hit_rate`` is (sign
    agreement, or accuracy at 0.5) was fixed when the sums were built
    (:func:`sums`'s ``binary``); this divides ``hits`` by ``signed``, so there is
    nothing to choose here. There is no ``log_loss`` column (:func:`metrics`'s
    ``binary=True`` has it): :data:`SUM_FIELDS` would need a mean and a weight for
    it, not added since nothing has asked for the chunked form yet. A key with
    fewer than ``min_samples`` rows is dropped, as :func:`metrics` drops it.

    ``r2`` and ``ic`` are null where they are undefined: a key whose target or
    prediction never varied has no correlation to report, and dividing by its zero
    variance would give an infinity that reads as a number.
    """
    keys = [c for c in s.columns if c not in SUM_FIELDS]
    mse = pl.col("sse") / pl.col("w")
    denom = (pl.col("m2_y") * pl.col("m2_pred")).sqrt()
    return (
        s.filter(pl.col("n") >= min_samples)
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


def _paired_residuals(
    df: pl.DataFrame,
    a: str,
    b: str,
    targets: Sequence[str] | None,
    a_suffix: str,
    b_suffix: str,
    who: str,
) -> dict[str, tuple[pl.Expr, pl.Expr]]:
    """Per target, ``a``'s and ``b``'s residual fields, resolved as
    :func:`seqtest` resolves them, NaN, infinities and values past the
    bank's input bound read as null."""
    have = {a: _resid_fields(df, a), b: _resid_fields(df, b)}
    if targets is None:
        bodies = [f.removeprefix("resid_") for f in have[a] if f.endswith(a_suffix)]
        stems = [x[: len(x) - len(a_suffix)] for x in bodies]
        targets = [t for t in stems if f"resid_{t}{b_suffix}" in have[b]]
        if not targets:
            msg = (
                f"{who}: {a!r} and {b!r} share no residual field (with a_suffix "
                f"{a_suffix!r}, b_suffix {b_suffix!r}); {a!r} has {have[a]}, {b!r} has {have[b]}"
            )
            raise ValueError(msg)
    out: dict[str, tuple[pl.Expr, pl.Expr]] = {}
    for t in targets:
        pair = []
        for side, suffix in ((a, a_suffix), (b, b_suffix)):
            want = f"resid_{t}{suffix}"
            if want not in have[side]:
                msg = (
                    f"{who}: target {t!r} names no residual of {side!r}: it has no field "
                    f"{want!r} (its residual fields are {have[side]})"
                )
                raise ValueError(msg)
            r = pl.col(side).struct.field(want).cast(pl.Float64)
            pair.append(pl.when(r.is_finite() & (r.abs() <= _INPUT_BOUND)).then(r))
        out[t] = (pair[0], pair[1])
    return out


def _normal_sf(z: float | None) -> float | None:
    """``P(Z > z)`` for a standard normal ``Z``."""
    return None if z is None or math.isnan(z) else 0.5 * math.erfc(z / math.sqrt(2.0))


def _hac_test(
    df: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    stat: str,
    min_samples: int,
) -> pl.DataFrame:
    """Per key and target, the mean of each value column and its t
    statistic against Newey and West's variance: with ``d`` the value and
    ``d̄`` its mean over the ``n`` rows that have one,

    ``S = Σ (d − d̄)² + 2 Σ_{l ≤ L} (1 − l/(L+1)) Σ_t (d_t − d̄)(d_{t−l} − d̄)``,
    ``stat = d̄ · n / sqrt(S)``.
    """
    frames = []
    for t, d in values.items():
        col = "__po_d"
        while col in df.columns:
            col = "_" + col
        rows = df.select(*keys, d.alias(col)).filter(pl.col(col).is_not_null())
        dd = pl.col(col)
        dev = dd - dd.mean()
        s = (dev * dev).sum()
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1.0)
            s = s + 2.0 * weight * (dev * (dd.shift(lag) - dd.mean())).sum()
        exprs = [
            pl.len().alias("n"),
            dd.mean().alias("mean"),
            pl.when(s > 0).then(dd.mean() * pl.len() / s.sqrt()).alias(stat),
        ]
        g = rows.group_by(keys, maintain_order=True).agg(exprs) if keys else rows.select(exprs)
        frames.append(g.with_columns(pl.lit(t).alias("target")))
    out = pl.concat(frames).filter(pl.col("n") >= min_samples)
    return out.select(*keys, "target", pl.col("n").cast(pl.Int64), "mean", stat)


def _ew_test(
    df: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
) -> pl.DataFrame:
    """Per row, the same t statistic with each row before it weighed by
    ``λ^age``, ``λ = 0.5 ** (1 / half_life)``, one step per row that has a
    value, the row itself included:

    ``m = Σ ω d / Σ ω``, ``S = Σ ω_t² (d_t − m)² + 2 Σ_l (1 − l/(L+1)) Σ_t ω_t ω_{t−l}
    (d_t − m)(d_{t−l} − m)``, ``stat = m Σ ω / sqrt(S)``.

    Each double sum is an exponentially weighted sum at ``λ²`` -- of the
    products, the two sides and the pairs -- read as ``ewm_mean`` times its
    weights, so the statistic is a few window expressions, not a loop.
    """
    lam = 0.5 ** (1.0 / half_life)
    lam2 = lam * lam
    idx = "__po_row"
    while idx in df.columns:
        idx = "_" + idx
    base = df.with_row_index(idx)
    fields = []

    def over(e: pl.Expr) -> pl.Expr:
        return e.over(keys) if keys else e

    i = over(pl.int_range(pl.len())).cast(pl.Float64)
    w1 = (1.0 - pl.lit(lam).pow(i + 1.0)) / (1.0 - lam)
    w2 = (1.0 - pl.lit(lam2).pow(i + 1.0)) / (1.0 - lam2)

    def ew_sum2(e: pl.Expr) -> pl.Expr:
        """``Σ λ^{2(T−t)} e_t``, the missing terms as zeros."""
        return over(e.fill_null(0.0).ewm_mean(alpha=1.0 - lam2, adjust=True)) * w2

    for t, d in values.items():
        col = "__po_d"
        while col in base.columns:
            col = "_" + col
        rows = base.select(idx, *keys, d.alias(col)).filter(pl.col(col).is_not_null())
        dd = pl.col(col)
        m = over(dd.ewm_mean(alpha=1.0 - lam, adjust=True))
        s = ew_sum2(dd * dd) - 2.0 * m * ew_sum2(dd) + m * m * w2
        for lag in range(1, lags + 1):
            back = over(dd.shift(lag))
            has = back.is_not_null().cast(pl.Float64)
            a_l = ew_sum2(dd * back)
            b_l = ew_sum2(dd * has)
            c_l = ew_sum2(back)
            d_l = ew_sum2(has)
            weight = 1.0 - lag / (lags + 1.0)
            s = s + 2.0 * weight * lam**lag * (a_l - m * (b_l + c_l) + m * m * d_l)
        value = pl.when(s > 0).then(m * w1 / s.sqrt()).alias(f"{stat}_{t}")
        got = rows.select(idx, value)
        base = base.join(got, on=idx, how="left")
        fields.append(f"{stat}_{t}")
    return base.with_columns(pl.struct(fields).alias(name)).drop(idx, *fields)


@renamed_keywords("po.eval.diebold_mariano", _GROUP)
def diebold_mariano(
    df: pl.DataFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] = (),
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "diebold_mariano",
) -> pl.DataFrame:
    """Diebold and Mariano's (1995) test of equal squared error between two
    specs' out-of-sample predictions, with Newey and West's variance.

    Per target ``t``, the loss differential ``d = resid_b**2 - resid_a**2``,
    positive when ``a`` was closer, over the rows where both have a
    residual; its mean ``d̄`` over Newey and West's (1987) standard error,
    Bartlett weights ``1 - l / (lags + 1)`` to ``lags``:

    .. code-block:: text

        S  = Σ (d - d̄)² + 2 Σ_{l ≤ lags} (1 - l/(lags+1)) Σ_t (d_t - d̄)(d_{t-l} - d̄)
        dm = d̄ · n / sqrt(S)                  ~ N(0, 1) with equal accuracy

    ``lags`` is 0 for one-step predictions, whose differentials are not
    correlated under the null; for a target looking ``h`` rows ahead give
    at least ``h - 1``. Between nested models -- ``b`` a restriction of
    ``a`` -- the test leans toward the smaller one, since the larger pays
    for estimating a coefficient that is zero: use :func:`clark_west`.

    With ``half_life`` (in rows), the statistic on every row instead, each
    row before it at ``0.5 ** (age / half_life)``, one step per row that has
    a differential, the row itself included: ``df`` with a struct column
    ``name`` of ``dm_<t>``, null where a side has no residual. It is the
    exponentially weighted form of the same test, read at every row as
    :func:`seqtest` is.

    Without ``half_life``: one row per (``group``, target) with ``n`` rows
    and at least ``min_samples``, the ``mean`` differential, ``dm`` and its
    two-sided ``p_value``. ``a``, ``b``, ``targets``, the suffixes and
    ``group`` are :func:`seqtest`'s, and raise as it raises.

    .. code-block:: python

        small = po.spec.ewridge("small", targets=["y"], features=["x0"], half_life=500.0)
        big = po.spec.ewridge("big", targets=["y"], features=["x0", "x1"], half_life=500.0)
        out = po.ModelBank([small, big]).fit_predict(df)
        po.eval.diebold_mariano(out, a="big", b="small")   # dm > 1.96: big predicts better
    """
    keys = _group_keys(group)
    pairs = _paired_residuals(df, a, b, targets, a_suffix, b_suffix, "diebold_mariano")
    values = {t: rb * rb - ra * ra for t, (ra, rb) in pairs.items()}
    if lags < 0:
        msg = f"diebold_mariano: lags must be >= 0, got {lags}"
        raise ValueError(msg)
    if half_life is not None:
        if not half_life > 0:
            msg = f"diebold_mariano: half_life must be > 0, got {half_life}"
            raise ValueError(msg)
        return _ew_test(df, values, keys, lags, half_life, "dm", name)
    out = _hac_test(df, values, keys, lags, "dm", min_samples)
    p = [
        None if z is None or math.isnan(z) else math.erfc(abs(z) / math.sqrt(2.0))
        for z in out["dm"].to_list()
    ]
    return out.with_columns(pl.Series("p_value", p, dtype=pl.Float64))


@renamed_keywords("po.eval.clark_west", _GROUP)
def clark_west(
    df: pl.DataFrame,
    *,
    big: str,
    small: str,
    targets: Sequence[str] | None = None,
    big_suffix: str = "",
    small_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] = (),
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "clark_west",
) -> pl.DataFrame:
    """Clark and West's (2007) test that a larger model nesting a smaller
    one predicts better, with Newey and West's variance.

    Diebold and Mariano's differential leans toward the smaller of two
    nested models: under the null the larger one's extra coefficients are
    zero but estimated, and their noise adds ``(p_small - p_big)²`` to its
    squared error on average. Clark and West add it back:

    .. code-block:: text

        f  = resid_small² - (resid_big² - (pred_small - pred_big)²)
           = resid_small² - resid_big² + (resid_big - resid_small)²
        cw = f̄ · n / sqrt(S)          S as in diebold_mariano, of f

    ``cw`` is about ``N(0, 1)`` under the null that the smaller model is the
    truth, and the test is one-sided: ``cw > 1.645`` rejects it at 5%.
    ``pred_small - pred_big`` is ``resid_big - resid_small``, so the two
    residual fields are all it reads. ``half_life`` gives the per-row
    exponentially weighted form, a struct ``name`` of ``cw_<t>``, as
    :func:`diebold_mariano`'s does; without it, one row per (``group``,
    target) with ``n``, the ``mean`` of ``f``, ``cw`` and its one-sided
    ``p_value``.
    """
    keys = _group_keys(group)
    pairs = _paired_residuals(df, big, small, targets, big_suffix, small_suffix, "clark_west")
    values = {t: rs * rs - rb * rb + (rb - rs) * (rb - rs) for t, (rb, rs) in pairs.items()}
    if lags < 0:
        msg = f"clark_west: lags must be >= 0, got {lags}"
        raise ValueError(msg)
    if half_life is not None:
        if not half_life > 0:
            msg = f"clark_west: half_life must be > 0, got {half_life}"
            raise ValueError(msg)
        return _ew_test(df, values, keys, lags, half_life, "cw", name)
    out = _hac_test(df, values, keys, lags, "cw", min_samples)
    return out.with_columns(
        pl.Series("p_value", [_normal_sf(z) for z in out["cw"].to_list()], dtype=pl.Float64)
    )
