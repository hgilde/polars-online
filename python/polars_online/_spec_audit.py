"""The data audit's builder, :func:`audit`: a part of :mod:`polars_online._spec`
in a file of its own, keeping that one under ``tests/test_repo_hygiene.py``'s
250 KB cap for a source file. :mod:`polars_online.spec` documents it with the
rest.
"""

from __future__ import annotations

from typing import Any, Unpack

from polars_online._kwargs import CommonKwargs
from polars_online._spec import _checked, _common, _mirror_target
from polars_online._spec_words import _listed, quoted, refusal


@_checked
def audit(
    name: str,
    *,
    columns: list[str],
    pairs: bool = False,
    distinct_cap: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A count of what a stream's columns hold, read in one pass by a spec that fits
    nothing and predicts nothing.

    Use it to find what a model cannot tell you: that a column is ``-999``
    on a twentieth of its rows, that a feed stopped and its last value was
    carried forward, or that a feature is a price level where a change was
    meant. :meth:`polars_online.ModelBank.check` reads its counts and says
    so. Run it alone before choosing a model, or beside the models in the
    same pass (``po.ModelBank([audit, model])``).

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``
    and ``session``, are in :mod:`polars_online.spec`. ``gap_cap`` is
    required with a ``clock``, as for every spec, and is the line a gap is
    counted at. An audit takes ``columns`` in place of ``targets`` and
    ``features``. ``fit_intercept`` has nothing to act on, as there is no
    regression. An audit counts and does not fit, so it refuses these:

    .. list-table::
       :header-rows: 1
       :widths: 30 70

       * - refused
         - why
       * - ``half_life``, ``lam``
         - nothing decays
       * - ``weight``
         - a value is a value whatever its row weighs. To audit a weight
           column, list it in ``columns``
       * - ``embargo``, ``label_clock``
         - there is no target to wait for
       * - ``min_weight``
         - there is no prediction to withhold
       * - ``group_close``
         - a closed group's counts would leave the bank with it, where
           :meth:`~polars_online.ModelBank.audit` and
           :meth:`~polars_online.ModelBank.check` cannot read them

    This model's own:

    ``columns``
        The columns to audit, any number, each once. They are a spec's
        ``features`` in the bank's tables
        (:meth:`~polars_online.ModelBank.describe` lists them with role
        ``"feature"``). Required, and not empty.
    ``pairs``
        Keep each pair of columns' correlation and the rows on which the two
        are equal, which ``check()`` reads for its ``duplicate`` finding. Memory and work
        grow as ``k^2`` for ``k`` columns. Default ``False``.
    ``distinct_cap``
        The number of counters each column keeps. The distinct count is
        exact up to this many values, and the most repeated value's count within
        ``n / (distinct_cap + 1)`` past it. ``None`` (the default) is 256,
        where a value on 3% of a continuous column's rows was found on ten
        seeds of ten as a sentinel; at 1,024 one on 2% was. At least 1 and at
        most 65,536.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record holding ``weight_sum`` alone, the count of rows before this one.
    `docs/OUTPUTS.md#audit
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#audit>`_
    gives its type. An audit predicts nothing, so what it counted is read
    back with :meth:`polars_online.ModelBank.audit`, one row per column (and
    with ``table="pairs"`` and ``table="clock"``), and judged by
    :meth:`polars_online.ModelBank.check`. *What it counts* in Notes defines
    each measurement.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), ``TypeError`` for
    ``targets`` and for ``features``, which an audit calls ``columns``, and
    ``ValueError`` naming the problem for:

    - an empty ``columns``, or one that names a column twice;
    - ``half_life``, ``lam``, ``weight``, ``embargo``, ``label_clock``,
      ``min_weight`` or ``group_close``;
    - a ``distinct_cap`` outside 1 to 65,536.

    .. rubric:: The update

    Per column, each row's value ``x`` is first classified: a null, a NaN,
    ``+inf``, ``-inf`` and a finite value past ``1e100`` in magnitude each
    add one to its own count, and anything else is *usable*. A usable value
    is added to the moments by Terriberry's update of the central sums
    ``M2``, ``M3`` and ``M4``, with ``n`` the usable values before it:

    .. code-block:: text

        n' = n + 1      d = x - mean      e = d / n'      t = d * e * n
        mean' = mean + e
        M4' = M4 + t * e^2 * (n'^2 - 3 n' + 3) + 6 e^2 * M2 - 4 e * M3
        M3' = M3 + t * e * (n' - 2) - 3 e * M2
        M2' = M2 + t

    and to its counter: a held value's count adds one, a new value takes a
    counter while fewer than ``distinct_cap`` are held, and otherwise takes
    one from every counter and drops those at 0 (a Misra-Gries summary). A
    usable row whose row before was usable adds the pair to the lag-1
    regression and counts whether the two are equal. Nothing decays.

    .. rubric:: Example

    This code audits a column with ``-999`` planted on about 5% of its rows
    beside a random walk, and prints what :meth:`~polars_online.ModelBank.check`
    finds in each:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(4)
        x = rng.standard_normal(10_000)
        x[rng.random(10_000) < 0.05] = -999.0          # a sentinel on a twentieth of the rows
        level = np.cumsum(rng.standard_normal(10_000))  # a level, where a change was meant
        pl.DataFrame({"x": x, "level": level}).write_parquet("feed.parquet")

        audit = po.spec.audit("audit", columns=["x", "level"])
        bank = po.ModelBank([audit])
        bank.fit(pl.scan_parquet("feed.parquet"))      # count every row, keep no output
        for r in bank.check().iter_rows(named=True):
            print(f"{r['column']}: {r['code']} ({r['value']:.3g})")

    .. code-block:: text

        level: random_walk (-1.99)
        x: sentinel (0.0437)
        x: heavy_tails (953)

    .. rubric:: See also

    - :meth:`polars_online.ModelBank.audit`, the counts, one row per column.
    - :meth:`polars_online.ModelBank.check`, what the counts say is wrong.
    - :meth:`polars_online.ModelBank.describe`, a few statistics of every
      column each spec has been fed.

    .. rubric:: Notes

    **What it counts.** Per column, over every row the stream holds, in the
    order it holds them. Every statistic past the first row of the table is
    over the usable values.

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
       * - the smallest change
         - the smallest nonzero change between two consecutive usable rows,
           and how many changes sit at it, to the rounding two changes of
           one step of a grid differ by: whether the column's repeats are
           one of its steps or an atom a stale feed adds
       * - persistence
         - the lag-1 autocorrelation, the correlation of each usable row with
           the row before it where that row is usable too, and the
           Dickey-Fuller statistic of the
           same regression with a constant, ``(rho - 1) / se(rho)``, as
           statsmodels' ``adfuller`` gives it with no lags. A random walk
           reads near 0; independent rows read about ``-sqrt(n)``; a fit
           with no residual reads ``+inf`` at a slope of 1 or more (a row
           counter, a clock in epoch seconds) and ``-inf`` below
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
         - each step between consecutive rows, the time that passed,
           uncapped, is a duplicate stamp (0), a gap (at or past
           ``gap_cap``, which caps the step a model ages by) or a regular
           step. The regular steps' mean, population spread and coefficient
           of variation are kept, with the largest step of all. On a
           temporal clock the mean, the spread and the largest step are
           ``Duration`` values

    **Every row is read.** Every other model skips a row whose feature is not
    usable, and an audit counts it, since that is what it is for. A reset,
    at a session gap set to ``"reset"`` or a step back past
    ``restart_after_step_back``, starts its runs and its clock steps over
    and keeps its counts: an audit is a record of what the stream held, not
    a fit. An audit records the clock as fed, so a numeric ``session_gap``
    changes none of its rows or tables. A step back of the clock is refused,
    as for every spec: to count the steps back instead, set
    ``restart_after_step_back`` (``0`` resets at each), and
    :meth:`~polars_online.ModelBank.summary`'s ``clock_backwards`` and
    ``check()``'s ``step_back`` count them.
    :meth:`~polars_online.ModelBank.predict` counts nothing, as it learns
    nothing. The counts are the same whether the stream comes in one chunk
    or a thousand, and are kept in the state, so an audit saved after one
    file goes on over the next. An audit predicts nothing, so it has no
    prediction to keep out-of-sample.

    **Groups merge.** ``ModelBank.audit(pooled=True)`` reads one audit of
    every group, as if they were one stream. The counts, the moments
    (Pébay's pairwise formulas), the longest run, the repeats of the row
    before, the lag pairs, the pairs and the clock merge exactly, up to
    rounding. Two do not: past the cap the counters merge as Misra-Gries
    summaries do (Agarwal et al. 2012, the errors added), and the digests by
    compressing their centroids together. A run or a lag pair across two
    groups is not one, so none is counted.

    **Cost.** A value costs one lookup in a hash table of a column's
    counters, held or not. Past the cap, a new value takes one from every
    counter, which empties the table at most once in ``distinct_cap + 1``
    new values, so a continuous column pays a few operations a row at any
    cap. The digest compresses every 256 rows, a few comparisons a row
    amortized, and ``pairs`` costs ``k(k - 1)/2`` pair updates a row for
    ``k`` columns. Memory does not grow with the rows. At the defaults it is
    about 14 KiB a column, and 72 bytes a pair, in every group. A column's
    share is a table of ``distinct_cap`` counters, 16 bytes each and as much
    again of the table's slack, and a digest of at most about 200 centroids
    and 256 values waiting.

    Measured through :meth:`~polars_online.ModelBank.fit_predict` on frames
    in memory, of columns drawn from ``N(0, 1)``, the best of seven runs on an
    Apple M4 Pro under a load average of 22 to 24 (polars 2.0.0):

    .. list-table::
       :header-rows: 1
       :widths: 64 36

       * - what ran
         - million rows a second
       * - one column, 100,000 rows, at 256 / 8,192 / 65,536 counters
         - 9.3 / 9.4 / 9.1
       * - ten columns, 500,000 rows
         - 1.71
       * - the same, with ``pairs=True``
         - 1.51
       * - an ``ewridge`` on the same columns (nine features, one target)
         - 4.4

    The saved state of those ten columns held 4.8 KB a column
    (:meth:`~polars_online.ModelBank.save_bytes`, 48,625 bytes, of which an
    unfed bank is 996), and 4.7 KB a column at fifty.
    """
    # Every problem at once, where several hold.
    who = f"spec {quoted(name)}"
    problems: list[Exception] = []
    if "features" in common:
        problems.append(
            TypeError(
                f"{who}: audit() takes no features, since columns names the columns it "
                "audits. Pass them as columns, and leave out features."
            )
        )
    if not columns:
        problems.append(
            ValueError(
                f"{who}: columns must name at least one column, but it is empty. List the "
                "columns to audit in columns."
            )
        )
    # Named here, in the builder's word: the shared check below says
    # `features`, which an audit does not take.
    twice = list(dict.fromkeys(c for i, c in enumerate(columns) if c in columns[:i]))
    if twice:
        listed = _listed([quoted(c) for c in twice])
        problems.append(
            ValueError(
                f"{who}: columns must name each column once, but it names {listed} more than "
                "once. Remove the repeats from columns."
            )
        )
    if problems:
        raise refusal(who, problems)
    # Each key is written only when given, as the Rust spec skips it when
    # absent: a spec from a TOML file that leaves it out saves the same bytes,
    # and the bank reports back the dict made here (`_SKIPPED_WHEN_ABSENT`).
    model: dict[str, Any] = {"type": "audit"}
    if pairs:
        model["pairs"] = True
    if distinct_cap is not None:
        model["distinct_cap"] = distinct_cap
    targets = _mirror_target(name, "audit", columns, common, "it counts what is in", "columns")
    return _common(name, model, targets=targets, features=columns, **common)
