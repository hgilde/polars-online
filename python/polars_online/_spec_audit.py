"""The data audit's builder, :func:`audit`: a part of :mod:`polars_online._spec`
in a file of its own, keeping that one under ``tests/test_repo_hygiene.py``'s
250 KB cap for a source file. :mod:`polars_online.spec` documents it with the
rest.
"""

from __future__ import annotations

import json
from typing import Any, Unpack

from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target


@_checked
def audit(
    name: str,
    *,
    columns: list[str],
    pairs: bool = False,
    distinct_cap: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """What a stream's columns hold, read in one pass: a spec that learns nothing.

    A model cannot tell you that a column is ``-999`` on a twentieth of its
    rows, that a feed stopped and its last value was carried forward, or
    that a feature is a price level where a change was meant. An audit
    counts those things as the rows go by, so that
    :meth:`polars_online.ModelBank.check` can say so. Run it alone before
    choosing a model, or beside the models in the same pass
    (``po.ModelBank([audit, model])``). It predicts nothing, so it writes
    only ``weight_sum`` per row; what it counted is read back with
    :meth:`polars_online.ModelBank.audit`.

    .. code-block:: python

        audit = po.spec.audit(
            "audit", columns=["x0", "x1", "x2", "y"], clock="t", gap_cap=300.0, pairs=True
        )
        bank = po.ModelBank([audit])
        bank.fit_predict(df)
        columns = bank.audit()                    # one row per column
        problems = bank.check()                   # what the counts say is wrong

    .. rubric:: What it measures

    Per column, over every row the stream holds, in the order it holds
    them. A *usable* value is a number a model can learn from: not null,
    not NaN, not infinite, and no larger than ``1e100`` in magnitude. Every
    statistic past the first row of the table is over the usable values.

    .. list-table::
       :header-rows: 1
       :widths: 22 78

       * - measurement
         - definition
       * - what is not usable
         - nulls, NaNs, ``+inf``, ``-inf`` and finite values past ``1e100``,
           each counted apart (:meth:`~polars_online.ModelBank.describe`'s
           ``null_count`` lumps them together, as the models do)
       * - moments
         - the mean, the standard deviation (``ddof = 1``), the skew
           ``sqrt(n) M3 / M2^1.5`` and the excess kurtosis ``n M4 / M2^2 - 3``
           (scipy's biased estimators), the smallest and the largest value.
           One pass, by Terriberry's update of the central sums ``M2``,
           ``M3``, ``M4``, with the mean kept to twice a double's precision
       * - repeats
         - each distinct value's count, exact up to ``distinct_cap`` distinct
           values, so the distinct count is exact up to the cap. Past it a
           Misra-Gries summary: a value not held takes one from every
           counter, and no count is more than ``count_error`` short of the
           truth, ``count_error`` being at most ``n / (distinct_cap + 1)``.
           The most repeated value and the next count are read from it
       * - runs
         - the longest run of one value on consecutive rows, and how many
           rows equal the row before (out of those whose row before was
           usable), beside ``sum p_v^2`` over the values' shares, which is
           what independent rows would give. A row that is not usable ends
           a run
       * - persistence
         - the lag-1 autocorrelation, the correlation of each usable row with
           the usable row before it, and the Dickey-Fuller statistic of the
           same regression with a constant, ``(rho - 1) / se(rho)``, as
           statsmodels' ``adfuller`` gives it with no lags. A random walk
           reads near 0; independent rows read about ``-sqrt(n)``
       * - location and spread, robustly
         - the median and the median absolute deviation, exact while the
           counts are, and otherwise from a t-digest whose centroids hold at
           most ``2n / 200`` rows each, so a rank is known to about 1% of the
           rows. The largest robust z is ``max(max - median, median - min)
           / (1.4826 MAD)``
       * - pairs, with ``pairs=True``
         - for every pair of columns, over the rows where both are usable,
           the correlation and the rows on which the two are equal
       * - the clock, with a ``clock`` column
         - each step between consecutive rows is a duplicate stamp (0), a
           gap (at or past ``gap_cap``, which caps the step a model sees) or
           a regular step, whose mean, population spread, coefficient of
           variation and largest value are kept

    **Cost.** Each row costs a binary search among ``distinct_cap``
    counters a column, and a digest compression every 256 rows, amortized
    to a few comparisons a row; with ``pairs`` it costs ``k(k - 1)/2`` pair
    updates. Ten columns ran at 1.3 million rows a second, and 1.2 million
    with their pairs, where an ``ewridge`` on the same columns ran at 4.1
    million (500,000 rows on an Apple M4 Pro, under a load average of 10).
    Memory does not grow with the rows: about 10 KiB a column at the
    defaults (``distinct_cap`` counters of 16 bytes, a digest of at most
    about 200 centroids and 256 values waiting), and 64 bytes a pair, in
    every group; a saved state held 4.8 KB a column.

    **Every row is read.** Another model skips a row whose feature is not
    usable; an audit counts it, which is the point. A restart, at a session
    gap set to ``"reset"`` or a step back past ``restart_after_step_back``,
    starts its runs and its clock steps over and keeps its counts: an audit
    is a record of what the stream held, not a fit. :meth:`~polars_online.ModelBank.predict`
    counts nothing, as it teaches nothing. The counts are the same whether
    the stream comes in one chunk or a thousand (hard rule 3), and are kept
    in the state, so an audit saved after one file goes on over the next.
    Hard rule 2, out of sample by construction, has nothing to apply to.

    **Groups merge.** ``ModelBank.audit(pooled=True)`` reads one audit of
    every group, as if they were one stream. The counts, the moments
    (Pébay's pairwise formulas), the longest run, the repeats of the row
    before, the lag pairs, the pairs and the clock merge exactly, up to
    rounding. Two do not: past the cap the counters merge as Misra-Gries
    summaries do (Agarwal et al. 2012, the errors added), and the digests by
    compressing their centroids together. A run or a lag pair across two
    groups is not one, so none is counted.

    ``columns``
        The columns to audit, any number. A spec's ``features`` in the
        bank's tables (:meth:`~polars_online.ModelBank.describe` lists them
        with role ``"feature"``).
    ``pairs``
        Keep each pair of columns' correlation and equal rows, ``O(k^2)``:
        what ``check`` reads for ``duplicate``. Default ``False``.
    ``distinct_cap``
        Counters a column: the distinct count is exact up to this many
        values, and the most repeated value's count within
        ``n / (distinct_cap + 1)`` past it. ``None`` (the default) is 256,
        where a value on 3% of a continuous column's rows was found on ten
        seeds of ten as a sentinel; at 1,024 one on 2% was. At least 1 and at
        most 65,536.

    The stream parameters are in :mod:`polars_online.spec`: ``clock``,
    ``gap_cap`` (required with a ``clock``, as for every spec, and the line
    a gap is counted at), ``group``, ``session`` and the rest.
    ``half_life`` and ``lam`` are refused, since nothing decays. So is
    ``weight``, since a value is a value whatever its row weighs: to audit a
    weight column, list it in ``columns``. (The Rust model takes a weight
    as every model does, into ``weight_sum`` alone, and counts a row of
    weight 0 as any other.) ``embargo`` is refused too, as there is no label
    to wait for. ``weight_sum`` is the count of rows before this one.

    .. rubric:: Output

    One struct column named after the spec, holding ``weight_sum`` alone
    (`docs/OUTPUTS.md#audit
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#audit>`_);
    the counts are read with :meth:`polars_online.ModelBank.audit`.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets`` and for ``features``, which an audit calls ``columns``;
    ``ValueError`` for an empty ``columns``, for ``half_life``/``lam``,
    ``weight`` or ``embargo``, and for a ``distinct_cap`` outside 1 to
    65,536.
    """
    if "features" in common:
        msg = f"spec {json.dumps(name)}: audit() takes columns=, not features="
        raise TypeError(msg)
    if not columns:
        raise ValueError(f"spec {json.dumps(name)}: columns must be non-empty")
    # Each key is written only when given, as the Rust spec skips it when
    # absent: a spec from a TOML file that leaves it out saves the same bytes,
    # and the bank reports back the dict made here (`_SKIPPED_WHEN_ABSENT`).
    model: dict[str, Any] = {"type": "audit"}
    if pairs:
        model["pairs"] = True
    if distinct_cap is not None:
        model["distinct_cap"] = distinct_cap
    targets = _mirror_target(name, "audit", columns, common, "it counts what is in")
    return _common(name, model, targets=targets, features=columns, **common)
