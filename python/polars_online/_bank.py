"""The chunk-fed model bank: memory O(state), not O(data)."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NoReturn, overload

import polars as pl

from polars_online import _polars_online as _native
from polars_online._polars_online import ArrowStruct
from polars_online._spec import (
    _coef_index_schema,
    _from_json,
    _json,
    _renamed_keywords,
    coef_index,
    target_name,
)

#: What `gram()` calls the constant column a spec's `fit_intercept` puts in
#: front of the features -- the `term` name `coef_index` gives it.
_INTERCEPT = "intercept"

#: Models whose Gram is over the features alone: they learn from no target,
#: so there is no cross-moment and no intercept column in the accumulator.
_NO_TARGET_GRAM = ("ew_cov",)

__all__ = ["ModelBank"]


def _group_keys(group: str | Iterable[str | None] | None) -> list[str | None] | None:
    """``group=`` on the readers, as the native side takes it: one key, or
    several, each as :meth:`ModelBank.groups` reports it -- a string, or
    ``None`` for the null group, which a lone ``None``, every group, cannot
    name (review round 4, N21)."""
    if group is None:
        return None
    if isinstance(group, str):
        return [group]
    if not isinstance(group, Iterable):
        msg = (
            "group: a key is a str or None, as groups() reports it, or a list of them; "
            f"got {type(group).__name__} {group!r}"
        )
        raise TypeError(msg)
    keys = list(group)
    for k in keys:
        if k is not None and not isinstance(k, str):
            msg = (
                "group: a key is a str or None, as groups() reports it; "
                f"got {type(k).__name__} {k!r}"
            )
            raise TypeError(msg)
    return keys


class ModelBank:
    """A set of models fitted together over the same rows, fed one chunk at a time.

    A bank holds one state per (spec, group). Feed it chunks in stream order and
    it hands each chunk back with one struct column per spec, every prediction
    made before the row it is for was learned from. Memory is the models' state
    plus one chunk, however long the stream, and the numbers do not depend on how
    the stream is chunked. What the bank has learned can be read back without the
    rows (:meth:`coef`, :meth:`gram`, :meth:`marginal`, :meth:`last_row`,
    :meth:`summary`, :meth:`describe`), saved to a file (:meth:`save`) and loaded
    on any OS (:meth:`load`), and scored against without moving (:meth:`predict`).

    The same bank runs as a polars plan, ``lf.online.fit_predict(specs)``
    (:mod:`polars_online._frame`), and as a file-to-file job, the ``online``
    command line; this class is the in-process form, for a plan you hand it to
    chunk (:meth:`fit_predict_batches`, :meth:`fit`) or frames you already have.

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        for chunk in df.iter_slices(100):          # chunks in stream order
            out = bank.fit_predict(chunk)          # the chunk plus a struct column per spec
        betas = bank.coef()                        # what it has learned, without the rows
        bank.save("bank.state")                    # resume later, on any OS

    ``specs`` are the dicts the :mod:`polars_online.spec` builders make.
    ``ValueError`` when there are none, when two share a name, or when a dict is
    not a spec (the message names the field).

    A bank is one ordered stream, so it is not for two threads at once: a method
    that finds the bank in use on another thread raises ``RuntimeError`` rather
    than interleave with it (:meth:`fit_predict` releases the GIL while it works).
    :meth:`predict` learns nothing, so any number of ``predict`` calls may
    overlap; only a ``fit_predict`` in flight refuses them, and they refuse it.
    """

    def __init__(self, specs: Iterable[dict[str, Any]]) -> None:
        # A hand-written dict may put a window expression in `targets`, as
        # the builders take one; `_json` writes it as the builders would
        # (review R1 D8, R2 P5).
        self._native = _native.ModelBank(_json(list(specs)))
        # The specs as the bank runs them -- filled, as a state file carries
        # them -- so they read the same before a round trip as after it. The
        # caller's own dicts were kept, and a dict the builders did not write
        # was one spec here and another after a save (review 2026-09-12, S21).
        self._specs = _from_json(self._native.specs_json())

    @property
    def specs(self) -> list[dict[str, Any]]:
        """The specs this bank runs, filled in.

        The dicts the builders made, with what a hand-written dict may leave out
        filled in (``targets`` for a model with no target, ``drift_action``). A state
        file carries them, so a bank loaded from one reports its specs without being
        told what they are, every field included; that is what makes a state file
        self-describing, and with :meth:`groups`, :meth:`output_fields` and the
        diagnostic tables a file can be walked by a caller who knows nothing about it
        in advance.

        A copy, and read-only. The bank's behaviour comes from the Rust state built at
        construction, not from this list, so a mutation here could only disagree with
        it: a stale copy would mislabel coefficients. Assigning to ``bank.specs``
        raises ``AttributeError``, and mutating what it returns changes nothing.
        """
        return copy.deepcopy(self._specs)

    def __repr__(self) -> str:
        names = ", ".join(repr(s["name"]) for s in self._specs)
        n_groups = max(self._native.group_counts(), default=0)
        return f"ModelBank([{names}], groups={n_groups}, rows_fed={self.rows_fed()})"

    def rows_fed(self) -> int:
        """Rows fed so far, over every chunk and group.

        Skipped rows and dropped groups are counted too, so this is not the sum of
        :meth:`summary`'s ``rows_fed``, the same count per group.
        """
        return self._native.rows_fed()

    def rows_seen(self) -> NoReturn:
        """Renamed :meth:`rows_fed`, the frames' word for the same count."""
        msg = "ModelBank.rows_seen was renamed rows_fed"
        raise AttributeError(msg)

    def groups(self, spec: str | int | None = None) -> pl.DataFrame:
        """The groups the bank holds state for: one row per (spec, group).

        ``spec``, ``group``
            The spec's name, and the group key as a string: ``""`` for a spec without
            a ``group`` column, null for rows whose key was null. A zoned ``Datetime``
            key is its instant, written as the UTC wall time.
        ``rows_processed``
            The group's rows that the null policy did not skip.
        ``last_clock``
            Its last clock value, in the clock column's own dtype, exactly: a
            ``Datetime`` in its unit and zone, a ``Date``, a ``Duration``, and a
            ``Float64`` for a number clock, which the bank reads as a double (the
            dtype ``emit_clocks`` gives ``scored_clock``). Null before the first row,
            or on a row-count clock, where the column is a ``Float64`` of nulls.

        Groups are sorted by key: integer keys as numbers, text keys as text, the
        null group first.

        State lives until :meth:`drop_groups` removes it, so this is how a
        long-running bank finds the groups that have gone quiet:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0"], group="stock_id",
                clock="ts", gap_cap="5m", half_life="1h",
            )
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            now = df["ts"].max()                   # last_clock is a Datetime, as ts is
            month = pl.lit(now) - pl.duration(days=30)
            stale = bank.groups().filter(pl.col("last_clock") < month)
            bank.drop_groups(stale["group"])     # the groups quiet for 30 days

        ``spec``, a name or a position, narrows the table to one spec: ``KeyError``
        for a name the bank has not got (the message lists the names), ``IndexError``
        for a position it has not got. One column holds one dtype, so a bank whose
        specs read clocks of different dtypes is read one spec at a time:
        ``ValueError`` names them otherwise.
        """
        n = len(self._native.spec_names())
        picked = range(n) if spec is None else [self._spec_index(spec)]
        return self._native.groups(list(picked))

    def drop_groups(self, keys: Iterable[str | None], spec: str | int | None = None) -> int:
        """Forget the state of these groups, in every spec or in one, and return how many
        streams were dropped.

        Keys are as :meth:`groups` reports them; a key the bank does not hold is not
        an error, it drops nothing. A dropped group starts cold if it appears again,
        exactly as a never-seen one would. Nothing else in the bank changes, and
        :meth:`rows_fed` still counts the rows it was fed. ``spec`` is as for
        :meth:`groups` (``KeyError`` / ``IndexError`` for one the bank has not got).
        """
        index = None if spec is None else self._spec_index(spec)
        return self._native.drop_groups(list(keys), index)

    @overload
    def skip_learned(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def skip_learned(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def skip_learned(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """The rows of ``frame`` the bank has not learned, for resuming a saved bank on
        input that overlaps it.

        A loaded bank resumes at the next row. Input that starts before the
        save, such as a rerun of the day or a file that overlaps the last one,
        steps each group's clock back to rows the state has already learned.
        The bank refuses that chunk, unless ``restart_after_step_back`` reads
        it as a new start. This method keeps a row whose clock is after its
        group's last clock in every spec that reads a clock, and every row of
        a group the bank has not seen, so that each row is learned once:

        .. code-block:: python

            spec = po.spec.ewridge(
                "m", targets=["y"], features=["x0"], clock="t", half_life=50.0, gap_cap=10.0
            )
            saved = po.ModelBank([spec])
            saved.fit_predict(df.head(300))   # the stream as far as the save
            saved.save("bank.state")

            bank = po.ModelBank.load("bank.state")
            rerun = df                        # the whole day again: its first 300 rows overlap
            out = bank.fit_predict(bank.skip_learned(rerun))   # rows 300 to 399, once each

        A row at a group's last clock counts as learned, so a stream that
        repeats a clock value, saved between two rows with that value, loses
        the later ones. A row with a null clock is kept, for
        :meth:`fit_predict` to judge. Row order is kept, and a ``LazyFrame``
        stays lazy. The comparison is exact: a temporal clock is compared in
        the integer nanoseconds the bank keeps, whatever its unit or time
        zone.

        ``ValueError`` when:

        - no spec reads a clock (a row-count clock has no position to resume
          from);
        - a spec's clock or group column is not in the frame;
        - a clock is temporal in the frame and was numeric in the bank, or
          the other way round.
        """
        schema = frame.lazy().collect_schema()
        keep: pl.Expr | None = None
        for spec, groups in zip(self.specs, self._native.last_clocks(), strict=True):
            clock = spec.get("clock")
            if clock is None:
                continue
            unlearned = _unlearned(spec["name"], clock, spec.get("group"), groups, schema)
            keep = unlearned if keep is None else keep & unlearned
        if keep is None:
            msg = (
                "no spec reads a clock, so the bank has no position to resume from; a "
                "row-count bank resumes at the next row, which only the caller knows"
            )
            raise ValueError(msg)
        return frame.filter(keep)

    def _spec_index(self, spec: str | int) -> int:
        """A spec's position from its name or index: ``KeyError`` for a name
        the bank has not got, ``IndexError`` for a position it has not got."""
        names = self._native.spec_names()
        if isinstance(spec, int):
            if not -len(names) <= spec < len(names):
                msg = f"spec index {spec} out of range; the bank has {len(names)} spec(s)"
                raise IndexError(msg)
            return spec % len(names)
        if spec not in names:
            msg = f"no spec named {spec!r}; the bank has {names}"
            raise KeyError(msg)
        return names.index(spec)

    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """One chunk in; the chunk plus one struct column per spec out.

        The struct is named after the spec, and its fields are what
        :mod:`polars_online.spec` describes under *What a spec writes* and
        the spec's builder describes under *Output*. For a regression that is
        ``pred_<t>``, ``resid_<t>``, ``weight_sum`` and ``coef``, plus the
        diagnostics switched on (:meth:`output_fields` lists them;
        `docs/OUTPUTS.md
        <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_
        has every model's). ``pred`` is out-of-sample: computed from the
        state before the row updates it. Chunk boundaries never change the
        numbers. By default they move the rows ``coef`` is reported on;
        under ``coef_every`` or ``max_rows_between_coefs`` not even those.

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, ridge=[1e-6, 0.1]
            )
            bank = po.ModelBank([spec])
            out = bank.fit_predict(df)
            preds = out["ridge"].struct.field("pred_y__r0.1")
            flat = out.online.unnest([spec])          # one column per field

        Raises ``TypeError`` for anything but a ``DataFrame`` (a ``LazyFrame``
        is told to collect, or to feed :meth:`fit_predict_batches`), and
        ``ValueError``, naming the spec and the column, when:

        - a column a spec reads (target, feature, clock, session, weight,
          group) is not in the frame;
        - a target, feature or weight column is not numeric;
        - a clock is not a number, a ``Datetime``, a ``Date`` or a
          ``Duration`` (a ``Time`` is refused: a time of day starts again at
          midnight), or its spec gives it the other kind of clock parameter
          -- a plain number to a temporal clock, a duration to a numeric one
          (*Clock units* in :mod:`polars_online.spec`);
        - the clock has a null or non-finite value;
        - a weight is negative (a null weight skips the row);
        - a spec is named like an input column, which the struct would
          replace;
        - a group's clock runs backwards with ``restart_after_step_back``
          unset, the default, or by no more than it, where that is a late
          row rather than a new start. A chunk that overlaps what a loaded
          state has learned is refused the same way; :meth:`skip_learned`
          drops the overlap.

        A chunk that would take a window past a refusing ``window_budget``
        is refused too (``ValueError``, naming the ring's size and its
        cadence, ``window_every`` and ``max_rows_between_snapshots``), found
        by replaying the chunk's clock schedule on the rings before any row
        is learned. A refused chunk leaves the bank
        exactly as it was, so the corrected chunk can be fed. The exception
        is a window past its budget under ``drift_action="reset"``, whose
        resets the replay cannot foresee. That is found as the rows are
        learned, so the chunk is refused after some of it has been. The bank
        then refuses every later ``fit_predict``, ``predict`` and ``save``
        rather than go on from there; rebuild it from its last save.

        ``RuntimeError`` when the bank is in use on another thread (class
        docstring).
        """
        return self._fit_predict_from(df, 0)

    def _fit_predict_from(
        self, df: pl.DataFrame, row_base: int, learn_only: bool = False, what: str = "fit_predict"
    ) -> pl.DataFrame:
        """:meth:`fit_predict` for rows ``row_base..`` of a longer input: an error
        names the input's row, not the chunk's. The chunked surfaces pass the rows
        they have fed before the chunk; ``learn_only`` is :meth:`fit`'s run, which
        keeps no prediction. ``what`` is the public method the caller used, for a
        chunk that is not a frame (review round 4, SF11: it named
        ``fit_predict`` under :meth:`fit_predict_batches` and :meth:`fit`)."""
        self._check_frame(df, what)
        outs = self._native.fit_predict(df, row_base, learn_only)
        self._warn_notices()
        return df.with_columns([pl.Series(s) for s in outs])

    def _warn_notices(self) -> None:
        """Raise each readiness notice the last chunk left as a
        :class:`polars_online.ReadinessWarning`, once per (spec, group): a
        coefficient the ridge determined more than the data did, or a
        ``max_error_inflation`` the stream has settled below
        (docs/WARMUP-AND-CONVERGENCE.md §3)."""
        from polars_online._frame import ReadinessWarning, _user_stacklevel

        for notice in self._native.take_notices():
            warnings.warn(ReadinessWarning(notice), stacklevel=_user_stacklevel())

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Score a frame against the bank as it stands, learning nothing.

        Every row gets the struct :meth:`fit_predict` would give it as the next row of
        its group's stream, computed from the current state. The bank is left exactly
        as it was, so the call is safe from any number of threads at once and row
        order does not matter. This is the serving side of a trained bank: ``load``
        once, ``predict`` per request, and ``fit_predict`` the rows later, in order,
        once their targets arrive.

        The frame needs each spec's feature and clock columns. Target columns are
        optional: present, they give ``resid`` and the standardized residual; absent,
        those are null. The session column is optional and feeds ``session_gap``; a
        weight column is not read. A trend model (``holt``) extrapolates over the
        clock distance from the row it last learned, capped by ``gap_cap``, and a
        ``kalman`` with ``revert_half_life`` shrinks its coefficients over that
        distance exactly as the next ``fit_predict`` row would. The other coefficient
        models predict from their current coefficients regardless of the clock.

        Per field: ``weight_sum``, ``penalty_selected``, ``sigma``, the residual quantiles,
        autocorrelation and the metrics are the values the bank holds, frozen.
        ``coef`` is filled on the last accepted row, since the same coefficients score
        every row. ``drift`` never fires. Rows of a group the bank has never seen, or
        without usable features, are null throughout, as a skipped row is in
        ``fit_predict``. Under ``group_close = "session"``, a row from a session other
        than the group's current one is scored as the first row of a fresh stream,
        null with ``weight_sum`` 0, because ``fit_predict`` would restart the stream there.
        A row that ``session_gap = "reset"`` would restart on is scored the same way.
        A row before the group's last learned clock is scored against the state as it
        stands, whatever ``restart_after_step_back`` says: scoring learns nothing,
        so it neither refuses the row nor starts over. Under an ``embargo``,
        ``predict`` releases nothing: a held row whose delay has passed by the scored
        row's clock is still held, and learned by the next ``fit_predict``, so every
        row is scored against the model the last ``fit_predict`` left, its
        ``weight_sum`` and ``learned_clock`` included.

        .. code-block:: python

            bank = po.ModelBank.load("bank.state")
            scored = bank.predict(today)              # the same struct columns, the bank unmoved

        Raises what :meth:`fit_predict` raises for the same frame (a missing or
        non-numeric column, a bad clock value), except that a missing target is not an
        error, and ``RuntimeError`` only when a ``fit_predict`` is in flight on
        another thread.
        """
        return self._predict_from(df, 0)

    def _predict_from(self, df: pl.DataFrame, row_base: int) -> pl.DataFrame:
        """:meth:`predict` for rows ``row_base..`` of a longer input, as
        :meth:`_fit_predict_from` is to :meth:`fit_predict`."""
        self._check_frame(df, "predict")
        outs = self._native.predict(df, row_base)
        return df.with_columns([pl.Series(s) for s in outs])

    def fit_predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]:
        """:meth:`fit_predict`, with the output handed back as Arrow.

        One struct per spec, in spec order, each exposing ``__arrow_c_array__`` --
        the Arrow PyCapsule interface. The values are :meth:`fit_predict`'s exactly,
        field for field and null for null; what differs is the way out. A polars
        ``Series`` crosses on py-polars' private ``_export``/``_import``, which is
        why this package carries a polars floor and why that interface promises no
        stability. The capsule interface is public and standardised, so a consumer
        of it reads the struct directly: ``pl.Series``, and pyarrow's ``pa.array``
        and ``pa.table``. duckdb reads the stream interface instead, so it takes the
        struct through ``pl.Series(s)``.

        Use this to hand a bank's output to something that is not polars, or to
        avoid a second copy of polars in the process. When you want a frame back,
        :meth:`fit_predict` is the same call with the wrapping done for you.

        Exporting hands the buffers to the consumer, so each struct can be read
        once; reading one twice raises ``ValueError``. The struct's Arrow field
        carries its spec's name, which is where the series below gets its name
        from.

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            structs = bank.fit_predict_arrow(df)       # one per spec, Arrow not polars
            out = df.with_columns([pl.Series(s) for s in structs])

        The frame still goes *in* as a polars frame. Raises what
        :meth:`fit_predict` raises for it.
        """
        self._check_frame(df, "fit_predict_arrow")
        out = self._native.fit_predict_arrow(df)
        self._warn_notices()
        return out

    def predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]:
        """:meth:`predict`, with the output handed back as Arrow.

        To :meth:`predict` what :meth:`fit_predict_arrow` is to
        :meth:`fit_predict`: the same scoring, the same values, and the bank left
        exactly as it was. See :meth:`fit_predict_arrow` for what a struct is and
        why it can be read only once.

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            structs = bank.predict_arrow(df)           # the bank unmoved
            scored = df.with_columns([pl.Series(s) for s in structs])
        """
        self._check_frame(df, "predict_arrow")
        return self._native.predict_arrow(df)

    @staticmethod
    def _check_frame(df: object, what: str) -> None:
        if isinstance(df, pl.DataFrame):
            return
        # A LazyFrame is the common slip, and the attribute error it used to
        # produce named an internal method.
        if isinstance(df, pl.LazyFrame):
            msg = (
                f"ModelBank.{what} takes a DataFrame, not a LazyFrame: "
                "collect it first (lf.collect())"
            )
            if what == "fit_predict":
                msg += ", or hand the plan to fit_predict_batches(lf), which chunks it"
        else:
            msg = f"ModelBank.{what} takes a polars DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    @_renamed_keywords
    def fit_predict_batches(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        closed_groups: str | Path | None = None,
        chunk_size: int | None = None,
    ) -> Iterable[pl.DataFrame]:
        """:meth:`fit_predict` over a plan or an iterator of chunks, lazily.

        Give it a ``LazyFrame`` and it does the chunking: the plan is read
        ``chunk_size`` rows at a time (100,000 by default) and each chunk is fed as
        the generator reaches it, so memory is the state plus a chunk however long
        the plan's input. A ``DataFrame`` is one chunk, or ``chunk_size`` slices of
        it when that is given. An iterator of frames is fed as it comes, and
        ``chunk_size`` does not re-chunk it. Whatever
        ``fit_predict`` raises for a chunk, this raises there; the chunks before it
        have been learned from.

        A plan runs through polars' streaming engine, and an online model learns
        in row order, so the order the engine delivers is part of the result. A
        ``join``, ``group_by`` or ``unique`` without an order guarantee delivers a
        different stream there than ``lf.collect()`` gives, and may differ between
        runs: give a join ``maintain_order="left"``, a ``group_by``
        ``maintain_order=True``, and sort after a ``unique``, or sort before the
        bank. A plan with such a node raises
        :class:`polars_online.OrderNotGuaranteedWarning`, naming the node.

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            for out in bank.fit_predict_batches(lf, chunk_size=100):
                pass    # each `out` is a chunk plus the struct columns

        ``closed_groups``, a path, drains the bank's closed groups after every chunk,
        so the queue is empty whenever a chunk is handed on, and writes what it
        drained to that file, in the format its extension names, when the chunks stop.
        A bank's queue is bounded only by draining it: without this, or
        :meth:`closed_groups` between chunks, every closed group waits in the bank,
        and :meth:`save` writes them all. The file is written however the chunks stop
        -- at their end, at a ``break``, or at an error -- because a drained row has
        left the bank, and the file is then the only place it is. The file and the
        bank's queue together always hold every group that closed.
        """
        return self._batches(batches, closed_groups, chunk_size, "fit_predict_batches")

    def _batches(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        closed_groups: str | Path | None,
        chunk_size: int | None,
        what: str,
    ) -> Iterable[pl.DataFrame]:
        """The run behind :meth:`fit_predict_batches` and :meth:`fit`: the
        arguments checked eagerly, then a generator. ``what`` is the public
        method the caller used, for the messages that name it."""
        from polars_online._frame import (
            _closed_path,
            _order_free,
            _plan_text,
            _warn_if_order_unspecified,
        )

        if chunk_size is not None and chunk_size < 1:
            msg = f"chunk_size must be at least 1, got {chunk_size}"
            raise ValueError(msg)
        path = _closed_path(closed_groups, self._specs)
        # Bound as a narrowed local rather than an `is_plan` flag: a bool does
        # not carry the type, so mypy could not see that the three calls below
        # get a `LazyFrame`. Only a frame or an iterator of frames reaches
        # `None`, and neither has a plan to read.
        plan = batches if isinstance(batches, pl.LazyFrame) else None
        # Both checks below read the plan through `explain`. Read it once and
        # hand it to both: reading it twice costs more than the JSON scan the
        # order check skips on a small plan (0.105 ms against 0.074 ms), so
        # sharing is what makes the skip a saving rather than a wash.
        plan_text = _plan_text(plan) if plan is not None else None
        # The order a plan delivers is the model, so a plan is inspected
        # before a row moves. The exception is `fit`, whose product is the
        # state alone: over accumulator-only specs with no decay the same
        # rows reach the same state in any order (to rounding), so warning
        # there would be a false positive. `fit_predict_batches` never
        # qualifies, because it hands back predictions and every prediction
        # is out-of-sample -- reordering moves all of them.
        if plan is not None and not (what == "fit" and _order_free(self._specs)):
            _warn_if_order_unspecified(plan, f"ModelBank.{what}", plan_text)
        # A plan whose source is a Python scan may be reading a single-use
        # Arrow stream, which yields nothing the second time and says nothing
        # about it. Decided here, before a row moves, because `explain` must
        # be read off the plan the caller gave.
        from polars_online._frame import _is_python_scan, _source_runs

        guard = (
            f"ModelBank.{what}" if plan is not None and _is_python_scan(plan, plan_text) else None
        )
        # Read before the plan is: one of this package's own plan forms over
        # an empty input is a Python scan with no rows too, and the run
        # counter is how it is told from a spent stream (task 159, P2), here
        # as in the sources (review 2026-10-05, YB6).
        started = _source_runs()
        return self._feed(
            self._chunks(batches, chunk_size), path, guard, what == "fit", started, what
        )

    def _feed(
        self,
        source: Iterable[pl.DataFrame],
        path: str | None,
        guard: str | None = None,
        learn_only: bool = False,
        started: int | None = None,
        what: str = "fit_predict_batches",
    ) -> Iterable[pl.DataFrame]:
        """The loop behind :meth:`fit_predict_batches`, once its arguments are
        checked: a generator, so nothing here runs until a caller asks.

        ``guard`` names the calling method when the source is a plan that might
        be reading a single-use Arrow stream; a run that then sees no rows at
        all is reported rather than passed off as a finished fit, unless one of
        this package's own sources ran since ``started``: then the plan is
        one of its plan forms over an input that was empty. ``what`` names the
        calling method for a chunk that is not a frame."""
        from polars_online._frame import (
            ConsumedSourceWarning,
            _sources_ran_since,
            _user_stacklevel,
            _write_closed,
        )

        drained: list[pl.DataFrame] = []
        fed = 0
        try:
            for chunk in source:
                out = self._fit_predict_from(chunk, fed, learn_only, what)
                fed += chunk.height
                if path is not None:
                    rows = self.closed_groups()
                    if rows.height:
                        drained.append(rows)
                yield out
        finally:
            if path is not None:
                _write_closed(path, drained, self.closed_groups(drop=False).clear())
        # After the loop and outside the `finally`, so an error on the way
        # through is the thing the caller hears about, not this.
        ours = started is not None and _sources_ran_since(started)
        if guard is not None and fed == 0 and not ours:
            warnings.warn(
                ConsumedSourceWarning(
                    f"{guard}: the plan yielded no rows, and its source is a Python scan -- "
                    "which is what `pl.scan_arrow_c_stream` builds over a DuckDB relation, a "
                    "pyarrow reader, or anything exposing `__arrow_c_stream__`. Such a stream "
                    "is consumed once: collected a second time it yields nothing, with no "
                    "error, so this bank has learned from nothing. Rebuild the plan per run "
                    "(call `pl.scan_arrow_c_stream(...)` inside the loop, not outside it). If "
                    "the source really is empty, silence this with "
                    'warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning).'
                ),
                stacklevel=_user_stacklevel(),
            )

    @staticmethod
    def _chunks(
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        chunk_size: int | None,
    ) -> Iterable[pl.DataFrame]:
        """The frames to feed, from a plan, a frame, or an iterator of frames.

        The plan's order is inspected by :meth:`_batches`, which can see the
        specs; whether the warning applies depends on them, and this cannot."""
        if isinstance(batches, pl.LazyFrame):
            rows = _native.default_chunk_size() if chunk_size is None else chunk_size
            return batches.collect_batches(chunk_size=rows, maintain_order=True)
        if isinstance(batches, pl.DataFrame):
            if chunk_size is None:
                return [batches]
            # `iter_slices` is zero-copy: views of the one frame, in order.
            return batches.iter_slices(n_rows=chunk_size)
        return batches

    @_renamed_keywords
    def fit(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        closed_groups: str | Path | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """Learn from every row and keep nothing: the run whose product is the state.

        :meth:`fit_predict_batches` with the output dropped as it comes, so no
        chunk's result is ever held and no frame is assembled from them. An
        accumulator-only spec emits ``weight_sum`` a row and nothing else,
        which over a billion rows is gigabytes written so they can be deleted.
        A fit whose product is its coefficients need not keep its predictions
        either.

        The plan runs through polars' streaming engine, whose row order for a
        ``join``, ``group_by`` or ``unique`` without an order guarantee is not
        the order ``lf.collect()`` gives. An online model learns in row order.
        Give a join ``maintain_order="left"`` or sort before the bank; such a
        plan raises :class:`polars_online.OrderNotGuaranteedWarning` naming the
        node (:meth:`fit_predict_batches` says more).

        **With one exception, and it is this method's alone.** A fit whose
        every spec is an accumulator with no decay -- ``ewridge`` or ``rls``
        at ``lam=1.0`` or ``half_life=inf``, no ``window_size``, no session,
        no drift reset -- reaches the same coefficients whatever order the
        rows arrived in, because its sums commute. Measured to rounding, not
        to the bit: 3.3e-16 over 200 rows. Since :meth:`fit` returns nothing
        and keeps only the state, the order genuinely does not matter there,
        and no warning is raised. It still is for :meth:`fit_predict_batches` over
        the same specs, whose predictions are out-of-sample and so move with
        the order (1.33 on those same rows). It is for every model whose
        update does not commute: ``sgd``, ``pa``, ``ftrl``, ``quantile`` and a
        reweighting ``huber`` differ materially with no decay at all. And it
        is for a model that selects by an out-of-sample error, as ``lasso``
        does its penalty.

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(lf, chunk_size=100)
            bank.save("bank.state")

        The state this leaves is the state :meth:`fit_predict_batches` leaves
        over the same rows, byte for byte. What it saves is the result, not
        the work: each row is still predicted before it is learned from,
        because that is what makes the fit out-of-sample. The output columns
        are still built before they are dropped.
        """
        # A formula target's window need not fit the embargo here: the run
        # keeps the state alone, and each row is learned once its window
        # closes (docs/PLAN.md task 104). `fit_predict` refuses that spec.
        # Said per call (review R2, P1): a flag on the bank could be left
        # set when a `predict` on another thread held the bank.
        for _ in self._batches(batches, closed_groups, chunk_size, "fit"):
            pass

    def coef(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """The coefficients behind every fit: one row per (spec, group, instance,
        position).

        A bank loaded from a state file answers "what are the betas?" without a row of
        data. Like :meth:`last_row`, :meth:`summary` and :meth:`describe`, it takes
        every spec by default and leads with a ``spec`` column, so the frames of
        several banks stack with a plain ``concat``. Sweeping this way skips a spec
        that has no coefficients (an ``ew_cov``, which writes statistics, or a
        ``seqtest``, which writes evidence), since the question was "the coefficients
        in this bank" and those have none. Naming one of them still raises, because
        then the question was about that spec. A bank with no coefficients at all
        gives an empty frame.

        The values are the fit after the last row each stream learned from, which the
        next row's ``pred`` is computed from: what the output's ``coef`` field shows on
        that row where it is written there. The columns:

        ``spec``, ``group``, ``instance``
            The spec's name; the group key as :meth:`groups` reports it; the decay
            instance's field suffix (``"@h500"``, or ``""`` for a single one).
        ``weight_sum``
            The accumulated weight behind the fit, what the next row's ``weight_sum`` field
            reports. The solve schedule decides when a stream first solves, not
            ``min_weight``. ``pred`` waits for ``min_weight`` and ``coef`` does not,
            so a fit with ``weight_sum`` below it is over fewer rows than the spec asks
            for; a solve over fewer rows than terms is a jittered one, counted by
            :meth:`solve_failures`.
        ``position``, ``target``, ``ridge``, ``feature_set``, ``penalty``, ``term``
            :func:`polars_online.spec.coef_index`'s columns: ``position`` indexes the
            flat ``coef`` list, ``term`` is ``"intercept"``, a feature name, or
            ``"level"`` / ``"trend"`` for ``holt``.
        ``coef``
            The value, in the features' original units; null until the stream's first
            solve, as ``coef`` is on those rows.

        .. code-block:: python

            bank = po.ModelBank.load("bank.state")
            betas = bank.coef()                       # every spec that has coefficients
            just_one = bank.coef("ridge")             # or name one
            one_ridge = betas.filter(pl.col("ridge") == 0.1)   # one (ridge, set) slot of a grid
            wide = one_ridge.pivot("term", index=["group", "instance"], values="coef")


        ``spec`` and ``group`` are as for :meth:`gram`: ``KeyError`` / ``IndexError``
        for a spec the bank has not got, and a group it has never seen gives an empty
        frame with the same columns. Groups come in :meth:`groups`' order.
        ``ValueError`` for a named ``ew_cov`` spec, which
        writes statistics, not coefficients, and for a named ``seqtest`` spec, which
        writes evidence.
        """
        names = self._native.spec_names()
        keys = _group_keys(group)
        if spec is not None:
            i = self._spec_index(spec)
            return self._coef_one(i, keys, names[i])
        frames = [
            f
            for i in range(len(names))
            # A spec that emits statistics or evidence rather than coefficients
            # is skipped when sweeping, and still refused when named.
            if (f := self._coef_try(i, keys, names[i])) is not None
        ]
        if not frames:
            # Every `coef()` frame's columns, `coef_index`'s among them, so
            # this one stacks with the others: it had seven columns of the
            # eleven, and a plain concat raised (review round 4, SF3).
            return pl.DataFrame(
                schema={
                    "spec": pl.String,
                    "group": pl.String,
                    "instance": pl.String,
                    "weight_sum": pl.Float64,
                    **_coef_index_schema(),
                    "coef": pl.Float64,
                }
            )
        return pl.concat(frames, how="diagonal_relaxed")

    def _coef_try(self, idx: int, group: list[str | None] | None, name: str) -> pl.DataFrame | None:
        """[`_coef_one`] or ``None`` for a spec that has no coefficients."""
        try:
            return self._coef_one(idx, group, name)
        except ValueError:
            return None

    def _coef_one(self, idx: int, group: list[str | None] | None, name: str) -> pl.DataFrame:
        layout = coef_index(self._specs[idx])
        n = layout.height
        groups: list[str | None] = []
        instances: list[str] = []
        n_effs: list[float] = []
        values: list[float | None] = []
        for g, instance, weight_sum, coef in self._native.coef(idx, group):
            if coef is not None and len(coef) != n:
                msg = (
                    f"spec {self._specs[idx]['name']!r}: {len(coef)} coefficients for {n} positions"
                )
                raise AssertionError(msg)
            groups += [g] * n
            instances += [instance] * n
            n_effs += [weight_sum] * n
            values += coef if coef is not None else [None] * n
        k = len(instances) // n
        body = pl.concat([layout] * k) if k else layout.clear()
        return body.with_columns(
            pl.Series("group", groups, pl.String),
            pl.Series("instance", instances, pl.String),
            pl.Series("weight_sum", n_effs, pl.Float64),
            # Finite-or-null, as the output's `coef` field is: an `ew_class`
            # class no row has carried yet has NaN means.
            pl.Series("coef", values, pl.Float64).fill_nan(None),
            pl.Series("spec", [name] * len(instances), pl.String),
        ).select("spec", "group", "instance", "weight_sum", *layout.columns, "coef")

    def last_row(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """The output struct as it stood on the last row each stream learned from: one
        row per (spec, group).

        It is the row :meth:`fit_predict` reported for that row, field for
        field, unnested after ``spec`` and ``group``: ``pred``, ``resid``,
        ``sigma``, the metrics, the residual quantiles, ``weight_sum``, and
        ``coef`` when the row carried it. A group's last accepted row in a chunk does by default;
        under ``coef_every`` or ``max_rows_between_coefs`` only a row on the
        cadence does. :meth:`coef` has the coefficients whichever row was last. It travels
        with the state, so a bank loaded from a file says how each model was
        doing without its output frame, and a directory of fits compares
        without keeping the last row of every output:

        .. code-block:: python

            files = ["bank.state"]
            table = pl.concat(
                [po.ModelBank.load(f).last_row().with_columns(file=pl.lit(f)) for f in files],
                how="diagonal_relaxed",
            )

        ``spec``, a name or a position, narrows the table to one spec
        (``KeyError`` / ``IndexError`` for one the bank has not got, as
        :meth:`groups`). ``group`` narrows it to one group's key, or to a list
        of keys with ``None`` among them for the null group (``group=None`` is
        every group), and a key the bank has never seen gives nothing. Groups
        come in :meth:`groups`' order. Specs with different fields
        are stacked ``diagonal_relaxed``, so a field one spec has not got is
        null on its rows. A group with no learned row yet (every row skipped
        so far) is a row of nulls.
        :meth:`predict` does not move it, and a chunk that ends in skipped
        rows leaves the row before them.
        """
        names = self._native.spec_names()
        picked = range(len(names)) if spec is None else [self._spec_index(spec)]
        wanted = _group_keys(group)
        frames: list[pl.DataFrame] = []
        for i in picked:
            keys, struct = self._native.last_row(i, wanted)
            frames.append(
                pl.DataFrame(
                    [
                        pl.Series("spec", [names[i]] * len(keys), pl.String),
                        pl.Series("group", keys, pl.String),
                        struct,
                    ]
                ).unnest(names[i])
            )
        return pl.concat(frames, how="diagonal_relaxed")

    def summary(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """What each stream has been fed: one row per (spec, group).

        Counts and ranges over every row routed to the group since its state
        began, undecayed, so they say what the model was trained on rather
        than what it still remembers. They are kept in the state file, so a
        bank loaded from a file says it too. The columns:

        ``spec``, ``group``
            As :meth:`groups` reports them.
        ``rows_fed``
            Rows routed to the group, skipped or not.
        ``rows_processed``
            Rows the models saw: every feature and the weight usable.
        ``rows_skipped``
            ``rows_fed - rows_processed``: a null, NaN, infinite or
            out-of-bound feature or weight.
        ``rows_learned``
            Processed rows with a positive weight and, for a model with
            targets, at least one usable target -- a target that is a window
            expression counted when the row is released with a value, once
            its window has closed.
        ``rows_zero_weight``
            Processed rows with weight 0 (the clock moved; nothing learned).
        ``weight_sum``
            The sum of the processed rows' weights (1 per row without a
            weight column).
        ``clock_min``, ``clock_max``, ``last_clock``
            The clock range fed and the last value, in the clock column's
            own dtype, exactly, as :meth:`groups` gives ``last_clock``; null
            on a row-count clock.
        ``session_changes``
            Rows whose session differed from the previous row's.
        ``clock_backwards``
            Rows whose clock fell below the previous row's within a session
            (what ``restart_after_step_back`` decided about).
        ``resets``
            Rows at which ``session_gap = "reset"`` or
            ``restart_after_step_back`` restarted the stream.
        ``settled_frac``, ``error_inflation``
            The warm-up readings of the stream's first instance after the
            last row (docs/WARMUP-AND-CONVERGENCE.md): how full the decay
            window is, and the largest noise-gate ratio over its slots. Null
            where the model has no such reading.
        ``min_support_coef``, ``min_support_coef_feature``, ``n_coef``
            The smallest coefficient's data share, the feature it belongs to,
            and the coefficients per target, likewise.

        ``spec`` narrows to one spec (``KeyError`` / ``IndexError`` for one
        the bank has not got), ``group`` to one group's key or a list of keys,
        ``None`` among them for the null group (``group=None`` is every
        group); a key never seen gives nothing. Groups come in
        :meth:`groups`' order, and a bank whose specs read clocks of
        different dtypes is read one spec at a time, as :meth:`groups` is.
        :meth:`predict` moves none of it, and feeding the same rows in one
        chunk or a thousand gives the same numbers to the bit.
        """
        n = len(self._native.spec_names())
        picked = range(n) if spec is None else [self._spec_index(spec)]
        return self._native.summary(list(picked), _group_keys(group))

    def describe(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Per-column statistics of what each stream has been fed: one row per (spec,
        group, input column), in spec order -- features, then targets, then the weight
        column.

        ``column``, ``role``
            The column's name, and ``"feature"``, ``"target"`` or ``"weight"``.
        ``count``, ``null_count``
            A partition of the rows fed: a value counts when finite and within the
            input bound, as the models take it, and is a null otherwise -- polars
            nulls, NaN, infinities and magnitudes beyond the bound alike.
        ``mean``, ``std``, ``min``, ``max``
            Over the counted values, undecayed and in row order, so chunking cannot
            move them; ``std`` is the sample one (``ddof=1``), null below two values.

        An unsupervised model lists no targets, an ``ew_class`` label column has its
        counts only, and a comparison's target is the difference of residuals it
        tests, named as the spec names it. A window expression given as a target
        counts the value each row is learned from, when the row is learned: it is
        a null until the row's window closes and its ``embargo`` passes, and stays
        one for a row still held when the input ends. ``spec`` and ``group``
        narrow the frame as in :meth:`summary`.
        """
        names = self._native.spec_names()
        picked = range(len(names)) if spec is None else [self._spec_index(spec)]
        keys = _group_keys(group)
        frames = [
            self._native.describe(i, keys).select(pl.lit(names[i]).alias("spec"), pl.all())
            for i in picked
        ]
        return pl.concat(frames)

    def gram(
        self, spec: str | int, group: str | Iterable[str | None] | None = None
    ) -> list[dict[str, Any]]:
        """The running sums behind a spec's fit, per group and instance, as numpy arrays.

        One dict per (group, decay instance), and per Gram where a spec reads more
        than one. Under ``target_gaps = "own_rows"``, the default, targets that have
        been missing on different rows are fitted from Grams of their own, one per set
        of targets present on the same rows, and each dict names its ``targets``. Only
        the models that keep a co-moment matrix report -- ``ewridge``, ``lasso`` and
        ``ew_cov``; the others yield nothing (``rls`` and ``kalman`` track an inverse,
        the gradient models keep no second moment). Each dict has:

        ``group``, ``instance``
            The group key as :meth:`groups` reports it (``""`` for a spec without a
            ``group`` column, ``None`` for a null key) and the instance's field suffix
            (``"@h500"``, or ``""`` for a single instance).
        ``columns``, ``targets``
            What the axes mean: the spec's features, with ``"intercept"`` first when
            the spec has one (the ``term`` names of
            :func:`polars_online.spec.coef_index`), and the target names the
            per-target arrays are indexed by -- the spec's, or the ones this Gram's
            rows belong to. ``targets`` is empty for ``ew_cov``, which learns from
            none. They are what makes the mapping self-describing, so
            :mod:`polars_online.gram` can take a column or a target by name.
        ``weight_sum``
            The accumulated weight behind these moments.
        ``n_kish``
            Kish's effective sample size, ``weight_sum**2 / sum(w**2)``: the number of
            equally weighted rows these moments are worth, and what a standard error
            computed from them divides by. ``weight_sum`` counts weight, not rows, so it is
            not a sample size; ``(1 + lam**d) / (1 - lam**d)`` is the Kish size of
            an exponentially weighted window of unit rows ``d`` clock units apart,
            ``lam = 0.5 ** (1 / half_life)``. It is
            scale-free: decay divides ``weight_sum`` and ``sum(w**2)`` by the same factor,
            so it does not shrink when a stream goes quiet. It says how many rows
            these moments average, not how old they are; ``weight_sum`` and
            ``target_weights`` say that. ``None`` before the first row, and
            under a window whose sum of squared weights the subtraction keeps
            no digit of (within ``64 eps`` of the history's): a window of rows
            far lighter than the history, where the ratio would be rounding.
        ``means``
            EW column means, shape ``(k,)``.
        ``comoments``
            Centred co-moments, shape ``(k, k)``: the EW analogue of a centred ``X'X /
            n``. Centred is what makes it accurate at large offsets.
        ``cross_moments``
            Per-target uncentred cross-moments ``E[z*y]`` over the rows each target
            was present on, shape ``(n_targets, k)``. Empty for ``ew_cov``.
        ``means_by_target``
            Per target, the column means over the rows it was present on, shape
            ``(n_targets, k)``: what its cross-moments are centred at and its
            intercept is recovered from. ``means`` again, to rounding, for a target
            present on every row the Gram learned (every target under ``target_gaps =
            "own_rows"``), and under ``"pairwise"`` a target's own means where it has
            gaps. :func:`polars_online.gram.solve` reads them. Empty for ``ew_cov``.
        ``cross_centred``
            Per target, the cross-moments centred at its column means and its own
            mean, ``E[(z - m_t) * (y - ybar_t)]``, shape ``(n_targets, k)``: what the
            model solves from, and what :func:`polars_online.gram.solve` reads. It
            equals ``cross_moments[t] - means_by_target[t] * ybar_t`` in exact
            arithmetic; formed in floating point, that difference keeps ``L**2 * eps``
            of it at a level ``L``, so the model keeps the centred form itself. Empty
            for ``ew_cov``.
        ``target_weights``
            Per-target accumulated weight, shape ``(n_targets,)``. Differs from
            ``weight_sum`` when targets have different null patterns.
        ``target_means``, ``target_vars``
            Per-target EW mean and centred variance of the target itself, shape
            ``(n_targets,)``, in the same arithmetic as ``comoments``: a target's
            variance here is the variance an ``ew_cov`` over that column would report,
            to the bit. Empty for ``ew_cov``.
        ``target_n_kish``
            Per-target Kish effective sample size, ``target_weights**2 / sum(w**2)``
            over that target's rows; ``nan`` for a target that has not seen a weighted
            row, and under a window, as ``n_kish``, where the window's squared weights
            keep no digit. Empty for ``ew_cov``.
        ``lags``, ``lag_comoments``
            The lags an ``ew_cov(lags=[...])`` accumulates at, in the order given, and
            their cross-moments as an ``(L, k, k)`` array: ``lag_comoments[l][a][b]``
            is ``E_w[d_a(t) * d_b(t - lags[l])]``, both deviations against the mean
            before the row. Both ``None`` for a spec without ``lags``. The matrix is
            not symmetric for a lag above zero (``a`` leading ``b`` is not ``b``
            leading ``a``), and lag 0 would be ``comoments`` exactly.

        The target moments are what makes the export a complete sufficient statistic:
        with the cross-moments alone there is no residual variance, no R², no
        information criterion and no standard error to be had from a saved Gram,
        because every one of them needs ``Var[y]``.

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0,
                ridge=0.1, standardize=True, group="stock_id",
            )
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            g = bank.gram("ridge", group="b0")[0]     # one dict per (group, instance, Gram)
            beta = po.gram.solve(g, target="y", ridge=0.1, standardize=True)   # the model's own fit
            var_y = g["target_vars"][0]
            r2 = 1 - (var_y - beta[1:] @ g["comoments"][1:, 1:] @ beta[1:]) / var_y

        Under a ``window_size`` everything here is the window's, the target moments
        included, since the window's snapshots carry them too, so ``po.gram.solve``
        on it fits the window as it stood after the last row.

        The two moment forms differ, and mixing them gives a silently wrong answer
        rather than an error, so the bridging identities are worth stating. For a
        target whose rows are the Gram's, which is every target under ``target_gaps =
        "own_rows"``:

        .. code-block:: text

            raw = comoments + outer(means, means)
            raw @ beta[t] == cross_moments[t]                       (up to the ridge term)

        and for every target, the centred form the model solves, with ``m =
        means_by_target[t]`` and ``ybar = cross_moments[t][0]``:

        .. code-block:: text

            comoments[1:, 1:] @ beta[t][1:] == cross_centred[t][1:]
            beta[t][0] == ybar - m[1:] @ beta[t][1:]

        Values are in the features' original units. The intercept, when the spec has
        one, is column 0: a constant 1, so it has zero variance in ``comoments`` and
        ``raw[0] == means``.

        Why this exists: the accumulators are the expensive part, and they are already
        exact, centred, decayed on the model's own clock with session and
        ``gap_cap`` handling, and resumable. Anyone wanting to do something other
        than the model's solve with them (a custom penalty, an information criterion,
        ``cond(G)``, a scree plot, forward stepwise, orthogonal matching pursuit, or a
        fit checked by hand) would otherwise recompute ``X'X`` from raw data in a
        second pass. These come from one pass over data that is never materialized, at
        every point in the stream rather than one, and they are the same matrices the
        deployed model solves against. It is not a speed claim: for a single batch
        Gram over materialized data, BLAS ``dgemm`` is blocked, vectorized, and
        comfortably faster.

        ``spec`` is a spec name or position (``KeyError`` / ``IndexError`` for one the
        bank has not got, as for :meth:`groups`); ``group`` narrows the list to one
        group's key, or to a list of keys with ``None`` among them for the null group
        (``group=None`` is every group), in :meth:`groups`' order. A group the bank
        has never seen gives an empty list, as does a model
        that keeps no co-moments; neither is an error. Requires numpy, which is not a
        dependency of this package (polars does not require it either): ``pip install
        polars-online[numpy]`` adds it, and without it the call raises
        ``ModuleNotFoundError`` saying so.
        """
        try:
            import numpy as np
        except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
            msg = (
                "ModelBank.gram() returns numpy arrays, and numpy is not installed. "
                "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
            )
            raise ModuleNotFoundError(msg) from e

        idx = self._spec_index(spec)
        spec_dict = self._specs[idx]
        # `ew_cov` accumulates over the features alone -- no target, and so no
        # constant column to regress one on.
        unsupervised = spec_dict["model"]["type"] in _NO_TARGET_GRAM
        columns = list(spec_dict["features"])
        if spec_dict.get("fit_intercept", True) and not unsupervised:
            columns = [_INTERCEPT, *columns]
        names = [] if unsupervised else [target_name(t) for t in spec_dict["targets"]]
        out = []
        for row, lag, (tidx, by_target, centred) in self._native.gram(idx, _group_keys(group)):
            g, instance, k, weight_sum, n_kish, means, como, cross, tw = row[:9]
            tmeans, tvars, tkish = row[9:]
            lags = None if lag is None else lag[0]
            out.append(
                {
                    "group": g,
                    "instance": instance,
                    "columns": columns,
                    "targets": [names[j] for j in tidx],
                    "weight_sum": weight_sum,
                    "n_kish": n_kish,
                    "means": np.asarray(means),
                    "comoments": np.asarray(como).reshape(k, k),
                    "cross_moments": np.asarray(cross).reshape(len(cross), k)
                    if cross
                    else np.zeros((0, k)),
                    "means_by_target": np.asarray(by_target, dtype=float).reshape(len(tidx), k)
                    if tidx
                    else np.zeros((0, k)),
                    "cross_centred": np.asarray(centred, dtype=float).reshape(len(tidx), k)
                    if tidx
                    else np.zeros((0, k)),
                    "target_weights": np.asarray(tw),
                    "target_means": np.asarray(tmeans, dtype=float),
                    "target_vars": np.asarray(tvars, dtype=float),
                    # A target with no weighted row yet has no Kish size; the
                    # array says `nan` where the Rust side says `None`, as
                    # every other float array here does.
                    "target_n_kish": np.asarray(
                        [np.nan if v is None else v for v in tkish], dtype=float
                    ),
                    "lags": None if lags is None else list(lags),
                    "lag_comoments": None
                    if lag is None
                    else np.asarray(lag[1]).reshape(len(lag[0]), k, k),
                }
            )
        return out

    def marginal(
        self, spec: str | int, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """The pairs a ``marginal`` spec keeps: one row per (group, instance, feature,
        target).

        Groups sorted, targets in spec order, features in spec order within each. The
        pairs are read from the state, so a bank loaded from a file reports them as
        the bank that saved it would, and feeding the rows in one chunk or a thousand
        gives the same numbers to the bit. The columns:

        ``group``, ``instance``
            As :meth:`gram` reports them (``""`` for a spec without a ``group``
            column; ``""`` for a single decay instance, else the field suffix such as
            ``"@h500"``).
        ``feature``, ``target``
            The pair's feature column and the target's name (its column, unless
            a ``po.target`` table gave one).
        ``weight_sum``
            The target's accumulated weight ``W_t``: rows where the target was
            present, weighted and decayed. Differs from the struct's ``weight_sum`` when
            targets have different null patterns.
        ``n_kish``
            ``W_t^2 / Q_t`` with ``Q_t`` the accumulated squared weight: the Kish
            effective sample size, the number of equally weighted rows that carry the
            same information. Null before the target's first row, and under a
            window whose ``Q_t`` the subtraction keeps no digit of (within
            ``64 eps`` of the history's).
        ``mean_x``, ``var_x``, ``mean_y``, ``var_y``, ``cov``
            The pair's EW moments, population form, over the decayed weights, so
            ``var`` is never negative.
        ``corr``, ``beta``, ``t_stat``
            ``cov / sqrt(var_x var_y)``; ``cov / var_x``, the slope of the target on
            that feature alone; and ``corr * sqrt((n_kish - 2) / (1 - corr^2))``, the
            t-statistic of that correlation at the Kish sample size. Read ``t_stat`` as
            a scale for comparing pairs, not a p-value: the rows are neither independent
            nor Gaussian. Null until ``weight_sum`` reaches the target's ``min_weight``,
            and null where undefined (a constant column, ``n_kish <= 2``).

        With ``lags``, four more list columns and four numbers:

        ``lag_corr_xx``, ``lag_corr_yy``
            Each series' own autocorrelation at the configured lags.
        ``lag_corr_xy``, ``lag_corr_yx``
            The feature now against the target ``l`` rows back, and the target now
            against the feature ``l`` rows back, at each of ``cross_lags`` (every lag
            by default). For two series that describe the same moment, a feature
            whose ``lag_corr_yx[0]`` exceeds its ``corr`` leads the target, and one
            whose ``lag_corr_xy[0]`` does follows it. That reading does not hold
            against a forward-looking target, one built from the rows after its own.
            There the target ``l`` rows back is built partly from the feature's
            newest ``l`` rows, so a feature built from the same news shows
            ``lag_corr_xy`` above ``corr`` however it is sampled. Absent under
            ``cross_lags=[]``.
        ``n_serial``, ``t_serial``
            ``n_kish`` divided by Bartlett's serial-dependence factor, and the
            statistic against it. Null without ``serial_rule``.
        ``phi_x``, ``phi_y``
            The per-row decays fitted under ``serial_rule = "geometric"``; null
            otherwise, and null when fewer than two kept lags are positive.

        With ``bins`` or ``bin_edges``, the nonlinear view. These columns are present
        whenever the spec asked for bins, holding empty lists and nulls until the
        edges are fixed:

        ``bin_edges``
            The feature's interior edges, fixed once and never moved. Ragged: a
            feature keeps only the bins it can support, so a binary feature has two
            bins and a constant one has a single bin.
        ``bin_n``, ``bin_mean_y``, ``bin_var_y``
            The target's weight, mean and variance inside each bin: the feature's
            response curve. One more entry than ``bin_edges``, since the outer two
            bins are open. A bin no row has landed in has ``bin_n = 0`` and null for
            the two moments.
        ``split_gain``, ``split_at``
            The fraction of the target's variance removed by the best single cut of
            the feature, and where that cut falls. Directly comparable with ``corr **
            2``, so ``split_gain - corr ** 2`` is the nonlinear surplus.
        ``split_gain_t``
            The ``t_stat`` a ``corr`` would need to match that gain, against ``n_serial``
            where there is one. Optimistic, because the cut was chosen by maximising
            over the candidates: a ranking, not a p-value.

        .. code-block:: python

            pairs = po.spec.marginal(
                "pairs", targets=["y"], features=["x0", "x1", "x2"], half_life=500.0
            )
            bank = po.ModelBank([pairs])
            bank.fit_predict(df)
            table = bank.marginal("pairs").sort("t_stat", descending=True)   # strongest first

        ``group`` narrows the frame to one group's key, or to a list of keys with
        ``None`` among them for the null group (``group=None`` is every group). A
        group the bank has never seen gives an empty frame; a spec that is not a
        ``marginal`` is refused (``ValueError``); an ``ew_cov``'s moments are read
        with :meth:`gram`. ``spec`` is a name or position (``KeyError`` /
        ``IndexError`` for one the bank has not got).
        """
        return self._native.marginal(self._spec_index(spec), _group_keys(group))

    def closed_groups(self, spec: str | int | None = None, *, drop: bool = True) -> pl.DataFrame:
        """The groups that have finished and not yet been read, oldest first, as one long
        frame.

        A spec with ``group_close`` emits a group's accumulators at the moment
        the bank can prove no further row will join it, and then drops the
        stream. That moment is a key smaller than the largest one fed so far
        under ``"monotone"``, or a session that has ended under
        ``"session"``. That is what keeps a bank over an unbounded key space
        bounded: without it, every key ever seen stays in memory. The rows it
        emits wait here until they are read, so the queue is bounded only by
        reading it: drain it between chunks, or pass ``closed_groups=`` to
        :meth:`fit_predict_batches`, which does.

        One row per (group, decay instance), and per Gram where a spec reads
        several (under ``target_gaps = "own_rows"`` targets that have been
        missing on different rows are fitted from Grams of their own). The
        common columns:

        ``spec``, ``group``, ``instance``, ``session``
            Which stream closed. ``session`` is the value of the span that
            ended under ``group_close = "session"``, and null under
            ``"monotone"``.
        ``weight_sum``, ``n_kish``
            As :meth:`gram` reports them, at the moment of the close.
        ``rows_fed``, ``rows_learned``, ``clock_min``, ``clock_max``
            The span's own :meth:`summary` counts, ``UInt64`` as there, and clock
            range, in the clock column's own dtype as there.

        Then a block per kind, present when any spec of the bank closes
        groups and is of that kind, null on the rows of other kinds:

        .. list-table::
           :header-rows: 1
           :widths: 24 36 40

           * - kind
             - columns
             - what they hold
           * - a kind that keeps accumulators
             - ``columns``, ``means``, ``comoments``, ``targets``,
               ``target_means``, ``target_vars``, ``target_weights``,
               ``target_n_kish``, ``cross_moments``, ``means_by_target`` and
               ``cross_centred``
             - packed: ``comoments`` is the upper triangle with the diagonal,
               row by row (``k(k+1)/2`` numbers), and the per-target arrays
               are row-major ``(n_targets, k)``.
               :func:`polars_online.gram.from_row` expands them and hands
               back exactly what :meth:`gram` would have returned for that
               group, so the exact solve on a closed group is
               ``po.gram.solve(po.gram.from_row(row))``
           * - every kind that reports coefficients
             - ``coef``
             - on a Gram's row, the coefficients of that Gram's ``targets``
           * - an ``ew_cov`` with ``pca``
             - ``eig_vals`` and ``eig_vecs``
             - the vectors signed for continuity with the previous closed
               row of the same (spec, instance): the previous group under
               ``"monotone"``, the same group's previous close under
               ``"session"``, never the previous chunk. So a sign flip
               between two rows is a real rotation rather than an
               eigensolver's arbitrary choice
           * - an ``rcov``
             - ``rcov``, ``rcorr``, ``rcov_n``, ``rcov_kind``,
               ``rcov_bandwidth_used``, ``rcov_omega2``, ``rcov_iv_sparse``,
               ``rcov_iq`` and ``rcov_psd_repaired``
             - :func:`polars_online.spec.rcov` explains each
           * - a ``marginal``
             - ``pair_*``
             - :meth:`marginal`'s frame turned on its side: ``pair_feature``
               and ``pair_target`` naming the pairs, and every other
               ``marginal`` column becoming ``pair_<column>`` with one entry
               per pair in the same order

        A ``marginal`` column that is a list per pair there (the
        ``lag_corr_*`` family with ``lags``; ``bin_edges``, ``bin_n``,
        ``bin_mean_y`` and ``bin_var_y`` with ``bins``) is a list of lists
        here, present when any closing ``marginal`` asked for it. A nested
        list has no CSV form, so a closed frame with them is for parquet or
        the frame itself.

        .. code-block:: python

            blocks = po.spec.ew_cov(
                "cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone"
            )
            by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # four blocks
            bank = po.ModelBank([blocks])              # an ew_cov closing on "block"
            bank.fit_predict(by_block)
            closed = bank.closed_groups()              # one row per finished block
            g = po.gram.from_row(closed.head(1))       # the block's moments, as gram() gives them

        ``drop`` (the default) removes what it returns from the queue, which
        is what a driver draining per chunk wants; ``drop=False`` peeks. The
        streams are dropped when they close, never when this is called: a
        bank's memory must not depend on the caller polling. What is
        undrained is saved with the state, so a driver that saves between
        chunks does not lose rows silently. ``spec`` narrows the frame to one
        spec's rows (a name or a position; ``KeyError`` / ``IndexError`` for
        one the bank has not got). A bank whose closing specs read clocks of
        different dtypes is drained one spec at a time, as :meth:`groups` is
        read: ``ValueError`` names them otherwise, and drains nothing.
        :meth:`predict` never closes anything.
        """
        idx = None if spec is None else self._spec_index(spec)
        return self._native.closed_groups(idx, drop)

    def solve_failures(self) -> dict[str, dict[str | None, int]]:
        """Jittered or failed matrix factorizations so far, per spec and group.

        A solve never returns NaN silently: a near-singular system is retried with
        escalating diagonal jitter, and total failure keeps the previous coefficients.
        Both cases are counted here, so a nonzero value means the inputs are
        degenerate (constant or collinear features, or far too few observations for
        the feature count), not that anything crashed. What each model counts:

        ``ewridge``, ``huber``, ``quantile``
            A solve that needed jitter, or failed at every jitter and kept the
            previous fit, the standardized solve included.
        ``lasso``
            A coordinate descent that ran out of ``max_iter`` sweeps before every
            coefficient moved less than ``tol``, one per target and path point.
        ``ew_class``
            A row on which a class covariance could not be factorized.
        ``hmm``
            A row whose state densities could not be evaluated; the filter is left
            where it stands.
        ``bocpd``
            A row whose predictive could not be evaluated: it reports nulls and leaves
            the run-length posterior where it stands, and the count is what makes a
            run of them visible.

        Every other model reports 0 because it counts nothing, not because nothing can
        fail: ``rls`` and ``kalman`` track an inverse, and the gradient models keep no
        second moment to factorize. Returns ``{spec name: {group key: count}}``.
        """
        names = self._native.spec_names()
        return {
            name: dict(pairs)
            for name, pairs in zip(names, self._native.solve_failures(), strict=True)
        }

    def output_fields(self) -> dict[str, list[str]]:
        """Spec name to the field names of its output struct, in order:
        :func:`polars_online.spec.output_fields` for every spec in the bank.

        The fields are fixed by the spec, so this is the output schema before any row
        is fed.
        """
        return dict(zip(self._native.spec_names(), self._native.output_fields(), strict=True))

    def save(self, path: str | Path) -> None:
        """Write the state to a file: versioned msgpack, loadable on any supported OS.

        Written to a temporary sibling and renamed into place, so an interrupted save
        leaves the previous state where it was rather than truncating it. The rename
        is preceded by a filesystem sync, which is what a resumable file costs: about
        4 ms on macOS, against about 0.5 ms for serializing 500 groups. Save every
        chunk and the sync dominates; save every hundredth and it disappears.

        Raises the ``OSError`` for what went wrong, with the path in the message:
        ``FileNotFoundError`` for a directory that is not there, ``PermissionError``
        for one that cannot be written. The file, if it existed, is untouched.
        ``RuntimeError`` while a ``fit_predict`` is in flight on another thread.
        """
        self._native.save(str(path))

    def save_bytes(self) -> bytes:
        """What :meth:`save` writes, as bytes, for a store that is not a file
        (:meth:`load_bytes` reads them back).

        This is also what pickle and ``copy.deepcopy`` carry. ``RuntimeError`` while a
        ``fit_predict`` is in flight on another thread.
        """
        return bytes(self._native.save_bytes())

    def to_json(self, *, pretty: bool = True) -> str:
        """The state as JSON: everything :meth:`save` writes, in a form that can be read
        without this library.

        An export, not a second state format: :meth:`load` reads msgpack and only
        msgpack. Use it to look at a state, diff two of them, or hand one to something
        that is not Python.

        It carries every value, the ones JSON has no literal for included. A NaN or an
        infinity is written as the string ``"nan"``, ``"inf"`` or ``"-inf"``, the
        spelling a spec's ``half_life`` takes. Every export is read back, and its
        msgpack must match the state's byte for byte. One that does not raises
        ``ValueError`` naming the problem, instead of returning a file that is quietly
        wrong. ``RuntimeError`` while a ``fit_predict`` is in flight on another
        thread, as :meth:`save` does.
        """
        return str(self._native.save_json_string(pretty))

    def save_json(self, path: str | Path, *, pretty: bool = True) -> None:
        """Write :meth:`to_json` to ``path``.

        A plain ``write_text``, not :meth:`save`'s atomic rename: this is an export of
        a state that lives elsewhere, so a half-written one costs nothing but a
        re-run.
        """
        Path(path).write_text(self.to_json(pretty=pretty), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        """A bank from a file :meth:`save` wrote, on this or any other OS.

        The file carries the specs, so none need be given; passing ``specs`` asserts
        they are the file's, which is how a resuming job checks that the state it
        found is the state of the bank it is about to run.

        .. code-block:: python

            # refused if the file's specs differ from these
            bank = po.ModelBank.load("bank.state", specs=[spec])
            bank.fit_predict(today)                    # and the stream goes on

        Raises ``FileNotFoundError`` (or the ``OSError`` for what went wrong) when the
        file cannot be read, and ``ValueError`` when it can but is not a bank this
        build loads:

        - not a bank state file at all;
        - written by a newer build (the file's format or state schema version is above
          this build's);
        - written under a state schema below this build's floor -- before 1.0 a schema
          change is not carried across, and such a state is refit rather than loaded
          (:func:`polars_online.schema_version` is the current schema);
        - a state that contradicts its own spec, such as an ``sgd`` state without the
          scaler its ``standardize`` needs;
        - ``specs`` that differ from the file's.
        """
        return cls.load_bytes(Path(path).read_bytes(), specs)

    @classmethod
    def load_bytes(cls, data: bytes, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        """:meth:`load` from the bytes :meth:`save_bytes` gave, with the same ``specs``
        check and the same ``ValueError`` for bytes that are not a bank this build
        loads.
        """
        specs_json = _json(list(specs)) if specs is not None else None
        native = _native.ModelBank.load_bytes(data, specs_json)
        return cls._wrap(native)

    @classmethod
    def _wrap(cls, native: Any) -> ModelBank:
        obj = cls.__new__(cls)
        obj._native = native
        # The state file carries the specs; they come back as the same dicts
        # the builders made.
        obj._specs = _from_json(native.specs_json())
        return obj


#: A temporal clock's physical value in nanoseconds, per unit, as the bank
#: reads it (a ``Date`` counts days).
_NANOS_PER = {"ms": 1_000_000, "us": 1_000, "ns": 1}
_NANOS_PER_DAY = 86_400 * 1_000_000_000


def _unlearned(
    name: str,
    clock: str,
    group: str | None,
    groups: list[tuple[str | None, float | None, int | None]],
    schema: pl.Schema,
) -> pl.Expr:
    """True on the rows of one spec's input its state has not learned: a clock
    after the group's last one, a group the bank has not seen, or a null clock
    (:meth:`ModelBank.skip_learned`)."""
    for column, role in ((clock, "clock"), (group, "group")):
        if column is not None and column not in schema:
            msg = f"spec {name!r}: {role} column {column!r} is not in the frame"
            raise ValueError(msg)
    dtype = schema[clock]
    value: pl.Expr
    # A temporal column is compared in its own unit, its physical integer
    # against the saved nanoseconds scaled down: `v * per > last` is
    # `v > last // per` for integers, and scaling the column up instead
    # wrapped past 2262 in polars arithmetic, so such a row was dropped
    # where the bank refuses it by name (review 2026-09-28).
    per = 1
    if dtype == pl.Date:
        per = _NANOS_PER_DAY
        value = pl.col(clock).cast(pl.Int64)
    elif isinstance(dtype, pl.Datetime | pl.Duration):
        per = _NANOS_PER[dtype.time_unit or "us"]
        value = pl.col(clock).cast(pl.Int64)
    elif dtype.is_numeric():
        value = pl.col(clock).cast(pl.Float64)
    else:
        msg = f"spec {name!r}: clock column {clock!r} has dtype {dtype}, which is not a clock"
        raise ValueError(msg)
    temporal = not dtype.is_numeric()
    kind: type[pl.DataType] = pl.Int64 if temporal else pl.Float64
    lasts: dict[str | None, float | int] = {}
    for key, number, nanos in groups:
        if number is None and nanos is None:
            continue
        if (nanos is not None) != temporal:
            was, now = ("temporal", "numeric") if nanos is not None else ("numeric", "temporal")
            msg = (
                f"spec {name!r}: clock column {clock!r} is {now} in the frame and was "
                f"{was} in the bank"
            )
            raise ValueError(msg)
        seen: float | int
        if nanos is not None:
            seen = nanos // per
        else:
            assert number is not None
            seen = number
        lasts[key] = seen
    last: pl.Expr
    if group is None:
        last = pl.lit(lasts.get(""), dtype=kind)
    else:
        named = {k: v for k, v in lasts.items() if k is not None}
        keys = pl.col(group)
        keyed = schema[group]
        if isinstance(keyed, pl.Datetime) and keyed.time_zone is not None:
            # The bank keys a zoned Datetime by its instant, the UTC wall
            # time without the zone; cast to text, the zone's own wall time
            # and offset matched no key (review 2026-10-05, PA4b).
            keys = keys.cast(pl.Datetime(keyed.time_unit))
        last = (
            keys.cast(pl.String).replace_strict(
                list(named), list(named.values()), default=None, return_dtype=kind
            )
            if named
            else pl.lit(None, dtype=kind)
        )
        if None in lasts:
            last = (
                pl.when(pl.col(group).is_null())
                .then(pl.lit(lasts[None], dtype=kind))
                .otherwise(last)
            )
    return value.is_null() | last.is_null() | (value > last)
