"""The model bank, fed one chunk at a time, whose memory holds its models' state and one chunk."""

from __future__ import annotations

import copy
import io
import numbers
import operator
import os
import warnings
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, overload

import polars as pl

from polars_online import _polars_online as _native
from polars_online._errors import ColumnNotFoundError, WrongTypeError
from polars_online._polars_online import ArrowStruct
from polars_online._renamed import renamed_method
from polars_online._spec import (
    _coef_index_schema,
    _from_json,
    _json,
    _renamed_keywords,
    coef_index,
    output_index,
    target_name,
)
from polars_online._warnings import (
    PolarsOnlineDeprecationWarning,
    deprecated_alias,
    deprecated_keywords,
    keyword_only_forwarded,
    warn_unstable,
)

if TYPE_CHECKING:
    from polars_online.gram import Gram

#: What `gram()` calls the constant column a spec's `fit_intercept` puts in
#: front of the features -- the `term` name `coef_index` gives it.
_INTERCEPT = "intercept"

#: Models whose Gram is over the features alone: they learn from no target,
#: so there is no cross-moment and no intercept column in the accumulator.
_NO_TARGET_GRAM = ("ew_cov",)

#: The columns of :meth:`ModelBank.coef` before :func:`coef_index`'s: what
#: every reader leads with -- the spec, the group, the instance with its
#: decay -- and the weight behind the fit.
_COEF_SCHEMA_HEAD: dict[str, Any] = {
    "spec": pl.String,
    "group": pl.String,
    "instance": pl.String,
    "half_life": pl.Float64,
    "lam": pl.Float64,
    "weight_sum": pl.Float64,
}

#: The last column of :meth:`ModelBank.coef`: the value.
_COEF_VALUE: dict[str, Any] = {"coef": pl.Float64}

#: Why :meth:`ModelBank.coef` refuses a named spec of a kind with no named
#: coefficient, where saying so helps; every other kind "emits no
#: coefficients".
_NO_COEF = {
    "ew_cov": (
        "an ew_cov model gives statistics, not coefficients. Read them from its output "
        "column, or its Gram with ModelBank.gram."
    ),
    "seqtest": (
        "a seqtest gives evidence (log e-values and counts), not coefficients. Read it from "
        "its output column."
    ),
    "micro": (
        "a micro model keeps a varying number of clusters, so its coef field has no fixed "
        "positions to index. Read it from the coef field of the spec's output column."
    ),
}

__all__ = ["ModelBank"]


class _CalledInt(int):
    """:attr:`ModelBank.rows_fed`'s value, which still answers the call it
    was before it became a property, with the deprecation warning."""

    def __call__(self) -> int:
        _warn_called("rows_fed")
        return int(self)


class _CalledDict(dict[str, list[str]]):
    """:attr:`ModelBank.output_fields`'s value, which still answers the call
    it was before it became a property, with the deprecation warning."""

    def __call__(self) -> dict[str, list[str]]:
        _warn_called("output_fields")
        return dict(self)


def _warn_called(name: str) -> None:
    who = f"ModelBank.{name}"
    warnings.warn(
        f"{who}: it is a property, so calling it is deprecated, and the call works only until "
        f"the next major version, which refuses it. Read bank.{name} without the call.",
        PolarsOnlineDeprecationWarning,
        stacklevel=3,
    )


def _group_keys(
    group: str | Iterable[str | None] | None, who: str = "ModelBank"
) -> list[str | None] | None:
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
            f"{who}: group must be a group key (a str), a list of group keys, or None for every "
            f"group, but it is of type {type(group).__name__} ({group!r}). Pass a group key, or "
            "a list of group keys with None in it for the null group, such as [None]."
        )
        raise TypeError(msg)
    keys = list(group)
    for k in keys:
        if k is not None and not isinstance(k, str):
            msg = (
                f"{who}: each group key in group must be a str, or None for the null group, but "
                f"one is of type {type(k).__name__} ({k!r}). Pass each group key as a str, or "
                "None."
            )
            raise TypeError(msg)
    return keys


def _refuse_uint128(schema: pl.Schema, who: str, clocks: Collection[str | None] = ()) -> None:
    """Refuse a ``UInt128`` column by name, before a frame with one crosses
    to the extension. The extension's Polars is built without 128-bit
    unsigned integers, and converting such a column panicked there
    (``PanicException: activate 'dtype-u128' feature``) before any of this
    library's code ran, whether a spec read the column or not (task 208's
    worker). A clock is told what an ``Int128`` clock is told (review round
    5, B3); any other column, the cast to make."""
    for name, dtype in schema.items():
        if dtype != pl.UInt128:
            continue
        if name in clocks:
            msg = (
                f"{who}: clock column {name!r} is UInt128, wider than the Int64 an integer "
                "clock is held in. Cast it to Int64 if its values fit, or after subtracting "
                "an origin."
            )
        else:
            msg = (
                f"{who}: column {name!r} is UInt128, which polars-online cannot read. Cast it "
                "to Int64 if its values fit, or to Float64, or leave it out of the frame."
            )
        raise WrongTypeError(msg)


def _threads_refusal(shown: str) -> str:
    """The text that refuses a ``threads`` argument: the Rust side's, word for
    word (``online_polars::threads_refusal``), so either side says the same."""
    return (
        f"threads: {shown} is not a number of threads, and 0 is not one thread per core here, "
        "as it is in POLARS_ONLINE_MAX_THREADS. Give an integer of 1 or more, or leave threads "
        "out (None in Python) to take the count from POLARS_ONLINE_MAX_THREADS, else one thread "
        "per core."
    )


def _check_threads(threads: object) -> int | None:
    """``threads`` checked where it is given, before anything is built: ``None``
    or an integer of 1 or more -- anything ``operator.index`` reads as one,
    numpy's integers included (task 269, D-6). A bool, a float or another
    type is a ``TypeError``, and 0, a negative integer or a count above the
    bound one pool may have a ``ValueError``, each with the message the
    native side raises for the same value, and each before any pool is
    built."""
    if threads is None:
        return None
    if isinstance(threads, bool):
        raise TypeError(_threads_refusal(repr(threads)))
    try:
        count = operator.index(threads)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(_threads_refusal(repr(threads))) from None
    if count < 1:
        raise ValueError(_threads_refusal(repr(threads)))
    if count > _native.max_threads():
        raise ValueError(_native.threads_too_many(repr(threads)))
    return count


def thread_pool_size(*, threads: int | None = None) -> int:
    """The number of threads a bank given this ``threads`` argument runs on.

    Use it to see how many threads a bank will use before you build one, or
    what ``POLARS_ONLINE_MAX_THREADS`` gives. Polars' own threads are
    separate, set by ``POLARS_MAX_THREADS``, and ``pl.thread_pool_size()``
    reports those.

    .. rubric:: Parameters

    ``threads``
        The count a bank would be given, as :class:`ModelBank` takes it.
        ``None``, the default, names the process's shared threads, which
        every bank given no ``threads`` runs on. ``POLARS_ONLINE_MAX_THREADS``
        sets their number when they are first used (unset or ``0``: one
        thread per core), and the number never changes after that. This call
        starts them if nothing has yet. An integer ``n`` names ``n`` threads,
        which this call checks without starting them.

    .. rubric:: Returns

    ``int``: the number of threads.

    .. rubric:: Raises

    ``TypeError``
        ``threads`` is not an integer, a bool included.
    ``ValueError``
        ``threads`` is below 1, or above 1024 (or the machine's core count,
        where that is larger); or ``threads`` is ``None`` and
        ``POLARS_ONLINE_MAX_THREADS`` is set to anything but a whole number,
        or to a count past the same bound. More threads than cores run no
        faster, and a pool starts all of its threads at once. A bank refuses
        the same values: a ``threads`` when it is made, and a bad
        ``POLARS_ONLINE_MAX_THREADS`` at its first call that runs. Both are
        checked before any thread starts.

    .. rubric:: Examples

    .. code-block:: python

        print(po.thread_pool_size(threads=2))   # 2: checked, and no thread starts

    .. code-block:: text

        2

    .. rubric:: See also

    - :class:`ModelBank`, whose ``threads`` argument this reads.
    """
    return int(_native.thread_pool_size(_check_threads(threads)))


class ModelBank:
    """A set of models fitted together over the same rows, fed one chunk at a time.

    Use a model bank when you feed the rows yourself, from frames you hold or a
    query you hand it to read in chunks, and when you want to read, save or
    serve what it has learned between chunks. To run the same bank inside a
    Polars query, use ``lf.online.fit_predict(specs)``, and to run it outside
    Python, from a file to a file, use the ``polars-online`` command line.

    .. rubric:: Parameters

    ``specs``
        A list of the specs the :mod:`polars_online.spec` builders make, at
        least one, with distinct names. The bank keeps one model for each
        spec and each group of the spec's ``group`` column, and adds one
        output column per spec to each chunk it learns.
    ``threads``
        How many threads the bank's work runs on. ``None``, the default, is the
        process's shared threads, whose number ``POLARS_ONLINE_MAX_THREADS``
        sets when they are first used (unset or ``0``: one thread per core). An
        integer ``n`` is ``n`` threads, which every bank given the same ``n``
        shares and which end with the last of those banks, unless the process's
        shared threads already number ``n``, in which case the bank runs on
        those. The output and the saved state are the same to the bit at any
        count (see Notes). The count is not saved with the state, so
        :meth:`load` takes its own. A spec's ``gram_threads`` caps the threads
        it uses to learn each row into its Gram, and ``marginal``'s ``shards``
        splits its pair work across the bank's threads. Neither adds a thread.

    .. rubric:: Raises

    ``ValueError``
        When:

        - ``specs`` is empty, two specs share a name, or an item is not a
          spec (the message names the field);
        - a spec's settings are refused, as its builder would refuse them;
        - ``threads`` is below 1 (``None``, not 0, gives the shared threads:
          one per core unless ``POLARS_ONLINE_MAX_THREADS`` is set), or above
          1024 (or the machine's core count, where that is larger), refused
          before any thread starts;
        - from the bank's first call that runs, with ``threads=None``:
          ``POLARS_ONLINE_MAX_THREADS`` is not a whole number, or is past
          the same bound.
    :class:`~polars_online.WrongTypeError`
        ``specs`` is one spec, a string or bytes, in place of a list of specs. It
        is both a ``TypeError`` and a ``ValueError``.
    ``TypeError``
        ``specs`` is not iterable, or ``threads`` is not an integer, a bool
        included.
    ``RuntimeError``
        From any method: the bank is in use on another thread (see Notes).

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        for chunk in df.iter_slices(100):          # chunks in stream order
            out = bank.fit_predict(chunk)          # the chunk, with the output column "ridge" added
        coefs = bank.coef()                        # what it has learned, read without the rows
        bank.save("bank.state")                    # a state file, to resume from on any OS
        print(bank)

    .. code-block:: text

        ModelBank(['ridge'], groups=1, rows_fed=400)

    .. rubric:: See also

    - :meth:`fit_predict`, :meth:`fit_predict_batches` and :meth:`fit`, which
      learn rows.
    - :meth:`predict`, which predicts rows from the state and learns nothing.
    - :meth:`coef`, :meth:`gram`, :meth:`marginal`, :meth:`last_row`,
      :meth:`summary` and :meth:`describe`, which read what the bank has
      learned without the rows.
    - :meth:`save` and :meth:`load`, which write and read the state file.
    - ``lf.online.fit_predict``, the same bank inside a query.

    .. rubric:: Notes

    A bank feeds each chunk to its models in stream order. Every prediction
    is made before the model learns the row it is for. Memory holds the
    models' state and one chunk, however long the stream, and the numbers do
    not depend on how the stream is cut into chunks.

    The output and the state do not depend on ``threads``: each piece of work
    is the same arithmetic whichever thread runs it, and the work is divided
    by the data, not by the number of threads.

    A bank is one ordered stream, so two threads cannot use it at once. A
    method that finds the bank in use on another thread raises
    ``RuntimeError``, so that two calls never interleave their rows. Without
    the check they could, since :meth:`fit_predict` releases Python's global
    interpreter lock while it works. :meth:`predict` learns nothing, so any
    number of ``predict`` calls may run at once. Only a call that learns, in
    flight, refuses them, and they refuse it.
    """

    def __init__(self, specs: Iterable[dict[str, Any]], *, threads: int | None = None) -> None:
        threads = _check_threads(threads)
        # One spec, or a string, is iterable too: read as a list it was its
        # keys or its characters, refused as `invalid type: string "name",
        # expected struct Spec`, which named neither the mistake nor the fix.
        if isinstance(specs, dict):
            msg = (
                "ModelBank: specs must be a list of specs, but it is one spec. Pass the spec in a "
                "list, as [spec], for a bank of one."
            )
            raise WrongTypeError(msg)
        if isinstance(specs, (str, bytes)):
            msg = (
                "ModelBank: specs must be a list of specs, such as [po.spec.ewridge(...)], but "
                f"it is of type {type(specs).__name__}. Pass a list of specs."
            )
            raise WrongTypeError(msg)
        # A hand-written dict may put a window expression in `targets`, as
        # the builders take one; `_json` writes it as the builders would
        # (review R1 D8, R2 P5).
        self._native = _native.ModelBank(_json(list(specs)), threads)
        # The specs as the bank runs them -- filled, as a state file carries
        # them -- so they read the same before a round trip as after it. The
        # caller's own dicts were kept, and a dict the builders did not write
        # was one spec here and another after a save (review 2026-09-12, S21).
        self._specs = _from_json(self._native.specs_json())

    @property
    def specs(self) -> list[dict[str, Any]]:
        """The specs this bank runs, with every field filled in.

        These are the dicts the builders made, with the fields a hand-written
        dict may leave out filled in, such as ``targets`` for a model with no
        target, and ``drift_action``. A state file carries them, so a bank
        loaded from one reports its specs without being told what they are.
        With :meth:`groups`, :attr:`output_fields` and the tables the bank's
        readers return, a caller who knows nothing about a state file in
        advance can find what it holds.

        The list is a read-only copy. The bank runs from the state it built
        when it was made, not from this list, so a change here could only
        disagree with it, and a stale copy would put the wrong names on
        coefficients. Assigning to ``bank.specs`` raises ``AttributeError``,
        and changing what it returns changes nothing in the bank.
        """
        return copy.deepcopy(self._specs)

    @property
    def threads(self) -> int | None:
        """The ``threads`` count the bank was made or loaded with.

        ``None`` means the process's shared threads (:class:`ModelBank`). A
        bank that is pickled or copied with ``copy.deepcopy`` keeps its count.
        """
        return self._native.threads

    def __repr__(self) -> str:
        names = ", ".join(repr(s["name"]) for s in self._specs)
        n_groups = max(self._native.group_counts(), default=0)
        return f"ModelBank([{names}], groups={n_groups}, rows_fed={self.rows_fed})"

    @property
    def rows_fed(self) -> int:
        """The rows fed to the bank so far, over every chunk and group.

        Each row counts once, however many specs the bank holds. Skipped rows
        count, as they do in :meth:`summary`'s ``rows_fed``, which gives the
        count for each spec and group. So do the rows of groups since dropped with
        :meth:`drop_groups`, or closed under ``group_close``, which
        :meth:`summary` no longer lists.

        It is a property, as :attr:`specs` is, and as Polars' ``height`` is:
        a count the bank keeps, read without work. Calling it, as
        ``bank.rows_fed()``, is deprecated: it still returns the count, with
        a :class:`~polars_online.PolarsOnlineDeprecationWarning`, until the
        next major version refuses it.
        """
        return _CalledInt(self._native.rows_fed())

    rows_seen = renamed_method("ModelBank", "rows_seen", "rows_fed", read_as="bank.rows_fed")

    def groups(self, spec: str | int | None = None) -> pl.DataFrame:
        """The groups the bank holds a model for, one row for each spec and group.

        Use it to see which groups a bank has learned, such as a bank loaded
        from a state file, and to find the groups that have gone quiet, whose
        models stay in the bank until :meth:`drop_groups` removes them.

        .. rubric:: Parameters

        ``spec``
            A spec's name or position, which narrows the table to that spec.
            ``None``, the default, lists every spec.

        .. rubric:: Returns

        A ``DataFrame`` with one row for each spec and group, in the bank's
        spec order, each spec's groups sorted by key: integer keys as
        numbers, text keys as text, the null group first. Its columns:

        ``spec``, ``group``
            The spec's name, and the group key as a string: ``""`` for a spec
            without a ``group`` column, and null for the rows whose key was
            null. A ``Datetime`` key with a time zone is written as its
            instant's UTC wall time.
        ``rows_fed``
            The rows the bank routed to the group, skipped rows included, as
            :meth:`summary` counts them.
        ``clock_last``
            The group's last clock value, exactly, in the clock column's own
            type: a ``Datetime`` in its unit and time zone, a ``Date``, a
            ``Duration``, an integer clock in its own width, and a
            ``Float64`` for a float clock (the type ``emit_clocks`` gives
            ``scored_clock``). It is null before the group's first row, and on
            a spec without a clock. The column is a ``Float64`` of nulls only
            when no spec read has a clock.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not hold. The message lists the
            names.
        ``IndexError``
            ``spec`` is a position the bank does not hold.
        ``ValueError``
            ``spec`` is ``None`` and the specs read clocks of different
            types, which one ``clock_last`` column cannot hold. Read such a
            bank one spec at a time.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0"], group="stock_id",
                clock="ts", gap_cap="5m", half_life="1h",
            )
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            print(bank.groups())
            # the groups with no row in the last 2 minutes of the stream
            cutoff = pl.lit(df["ts"].max()) - pl.duration(minutes=2)
            quiet = bank.groups().filter(pl.col("clock_last") < cutoff)
            print(bank.drop_groups(quiet["group"]))

        .. code-block:: text

            shape: (4, 4)
            ┌───────┬───────┬──────────┬─────────────────────┐
            │ spec  ┆ group ┆ rows_fed ┆ clock_last          │
            │ ---   ┆ ---   ┆ ---      ┆ ---                 │
            │ str   ┆ str   ┆ i64      ┆ datetime[μs]        │
            ╞═══════╪═══════╪══════════╪═════════════════════╡
            │ ridge ┆ b0    ┆ 100      ┆ 2024-01-02 16:06:00 │
            │ ridge ┆ b1    ┆ 100      ┆ 2024-01-02 16:07:00 │
            │ ridge ┆ b2    ┆ 100      ┆ 2024-01-02 16:08:00 │
            │ ridge ┆ b3    ┆ 100      ┆ 2024-01-02 16:09:00 │
            └───────┴───────┴──────────┴─────────────────────┘
            1

        .. rubric:: See also

        - :meth:`drop_groups`, which removes groups' models.
        - :meth:`summary`, which counts each group's rows by what happened
          to them.
        - :meth:`closed_groups`, for the groups ``group_close`` ends.
        """
        n = len(self._native.spec_names())
        picked = range(n) if spec is None else [self._spec_index(spec, "ModelBank.groups")]
        return self._native.groups(list(picked))

    def drop_groups(self, keys: Iterable[str | None], spec: str | int | None = None) -> int:
        """Remove the models of these groups, in every spec or in one.

        Use it to free the memory a long-running bank holds for groups that
        have gone quiet, which :meth:`groups` finds. To have the bank end a
        group's model itself, as its rows end, set the spec's
        ``group_close``.

        .. rubric:: Parameters

        ``keys``
            The group keys to drop, a list of keys and not one key, as
            :meth:`groups` writes them in its ``group`` column: strings, with
            ``None`` for the null group. A key the bank does not hold drops
            nothing, and is not an error. A string given alone is read as its
            characters, each a key.
        ``spec``
            A spec's name or position, which drops the groups from that spec
            alone. ``None``, the default, drops them from every spec.

        .. rubric:: Returns

        ``int``: the number of models dropped, one for each spec and group.
        A dropped group that appears again starts with nothing learned, as a
        group the bank has never seen does. Nothing else in the bank changes,
        and :attr:`rows_fed` still counts the dropped groups' rows.

        .. rubric:: Raises

        ``TypeError``
            A key is neither a string nor ``None``.
        ``KeyError``
            ``spec`` is a name the bank does not hold.
        ``IndexError``
            ``spec`` is a position the bank does not hold.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0"], group="stock_id", half_life=50.0
            )
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            print(bank.drop_groups(["b0", "b9"]))   # b9 was never seen, so one model is dropped

        .. code-block:: text

            1

        .. rubric:: See also

        - :meth:`groups`, which lists the groups and their last clock values.
        """
        index = None if spec is None else self._spec_index(spec, "ModelBank.drop_groups")
        return self._native.drop_groups(list(keys), index)

    @overload
    def skip_learned(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def skip_learned(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def skip_learned(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """The rows of a frame or query that the bank has not learned.

        Use it to resume a saved bank on input that overlaps what the bank
        learned before it was saved, such as a rerun of the day or a file
        that overlaps the last one. A loaded bank goes on from the next row,
        so the overlap steps each group's clock back, which :meth:`fit_predict`
        refuses unless ``restart_after_step_back`` reads it as a new start.
        Without a clock in any spec, pass the bank only the rows after the
        last one it learned.

        .. rubric:: Parameters

        ``frame``
            The input, a ``DataFrame`` or a ``LazyFrame``, holding each spec's
            clock column and ``group`` column.

        .. rubric:: Returns

        ``frame``'s rows that the bank has not learned, of the same type, in
        the same order: a ``LazyFrame`` stays lazy. A row is kept when, in
        every spec that reads a clock, its clock is after its group's last
        clock, its group is one that spec has not seen, or its clock is null.
        A null clock is left for :meth:`fit_predict` to judge. A row at its
        group's last clock counts as learned.

        .. rubric:: Raises

        ``ValueError``
            When:

            - no spec reads a clock, so there is no position on a clock to
              resume from;
            - a clock column is not a number, a ``Date``, a ``Datetime`` or a
              ``Duration``;
            - a clock is temporal in the frame and was numeric in the bank,
              or the other way round.
        :class:`~polars_online.ColumnNotFoundError`
            A spec's clock or group column is not in the frame. It is both a
            ``ValueError`` and Polars' ``ColumnNotFoundError``.
        ``AttributeError``
            ``frame`` is neither a ``DataFrame`` nor a ``LazyFrame``, such as
            a dict.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "m", targets=["y"], features=["x0"], clock="t", half_life=50.0, gap_cap=10.0
            )
            saved = po.ModelBank([spec])
            saved.fit_predict(df.head(300))   # the stream as far as the save
            saved.save("bank.state")

            bank = po.ModelBank.load("bank.state")
            rerun = df                        # the whole day again: its first 300 rows overlap
            out = bank.fit_predict(bank.skip_learned(rerun))
            print(f"{out.height} rows learned, from t = {out['t'][0]}")

        .. code-block:: text

            100 rows learned, from t = 300.0

        .. rubric:: See also

        - :meth:`load`, which loads the saved bank.
        - `Resume from a saved state
          <https://github.com/hgilde/polars-online/blob/main/docs/how-to/resume-from-a-saved-state.md#resume-on-input-that-overlaps-the-state>`_.

        .. rubric:: Notes

        The comparison is exact. A temporal clock is compared in the integer
        nanoseconds the bank keeps, whatever its unit or time zone, and an
        integer clock in integers. Since a row at its group's last clock
        counts as learned, a stream that repeats a clock value and was saved
        between two rows with that value loses the later ones.
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
                "ModelBank.skip_learned: no spec reads a clock, so the bank has no place on a "
                "clock to resume from. Without a clock the bank resumes at the next row. Pass "
                "the bank only the rows after the last one it learned."
            )
            raise ValueError(msg)
        return frame.filter(keep)

    def _spec_index(self, spec: str | int, who: str = "ModelBank") -> int:
        """A spec's position from its name or index: ``KeyError`` for a name
        the bank has not got, ``IndexError`` for a position it has not got."""
        names = self._native.spec_names()
        # Any whole number is an index, numpy's integers included (task 352);
        # a bool is not one, and is looked up as a name.
        at: int | None = None
        if isinstance(spec, numbers.Integral) and not isinstance(spec, bool):
            at = operator.index(spec)
        if at is not None:
            if not -len(names) <= at < len(names):
                msg = (
                    f"{who}: spec index {at} is out of range, since the bank has {len(names)} "
                    f"specs. Pass an index from {-len(names)} to {len(names) - 1}, or a spec's "
                    "name."
                )
                raise IndexError(msg)
            return at % len(names)
        if spec not in names:
            msg = (
                f"{who}: the bank has no spec named {spec!r}, only {names}. Pass one of those "
                "names, or an index."
            )
            raise KeyError(msg)
        return names.index(spec)

    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Learn a frame's rows, and return them with each spec's output.

        Use it for rows held in memory as a frame, when you want the output
        back as one frame. For a query or a stream too large to hold, use
        ``lf.online.fit_predict`` or :meth:`fit_predict_batches`. To learn
        without keeping the output, use :meth:`fit`, and to predict without
        learning, :meth:`predict`.

        .. rubric:: Parameters

        ``df``
            The next rows of the stream, as one chunk: a ``DataFrame``
            holding every column the bank's specs read. Those are the specs'
            targets and features, the clock, session, weight, group and
            ``label_clock`` columns they name, and the columns a formula
            target's expression reads. In each group of a spec with a
            clock, the rows continue the group's clock within each session.
            A step back is refused, unless ``restart_after_step_back`` is
            set below it, where the model resets.

        .. rubric:: Returns

        ``df`` with one output column added for each spec, named after the
        spec, whose value in each row is a record of named fields. For a
        regression they are ``pred_<t>``, ``resid_<t>``, ``weight_sum``,
        ``settled_frac``, ``withheld_reason`` and ``coef``, with
        ``support_coef`` where the model reports it, ``lasso``'s
        ``penalty_selected_<t>``, and the fields of the diagnostics switched
        on. Under a grid of half-lives every field read from one instance
        carries its suffix, such as ``weight_sum@h100`` (an instance is the
        part of the model that fits one half-life). The fields that choose
        across instances (``emit_selected``, ``emit_averaged``) and those
        ``emit_clocks`` writes carry none. Under a grid of ``ridge`` values,
        feature sets or penalties, each slot's fields, its ``pred``, its
        ``resid`` and its diagnostics, carry the slot's suffix, such as
        ``pred_y__r0.1``. :attr:`output_fields` lists a
        bank's fields, :mod:`polars_online.spec` describes them under *What
        a spec writes*, and `docs/OUTPUTS.md
        <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md>`_
        gives every model's. Each prediction is out-of-sample: the model
        makes it from what it learned before the row, and only then learns
        the row. ``coef``, ``support_coef`` and the standard errors are the
        fit after the row.

        .. rubric:: Raises

        ``TypeError``
            ``df`` is not a ``DataFrame``, such as a ``LazyFrame``.
        :class:`~polars_online.ColumnNotFoundError`
            A column a spec reads is not in ``df``. It is both a
            ``ValueError`` and Polars' ``ColumnNotFoundError``.
        :class:`~polars_online.WrongTypeError`
            A column of a type its role cannot take. It is both a
            ``TypeError`` and a ``ValueError``. When:

            - a column of ``df`` is ``UInt128``, which polars-online cannot
              read;
            - a target column (other than ``ew_class``'s), a feature or a
              weight column is neither numeric nor Boolean;
            - a clock is not a number, a Boolean, a ``Datetime``, a ``Date``
              or a ``Duration`` (a ``Time`` starts again at midnight), or is
              an ``Int128``;
            - a temporal column is read as a feature, target or weight, or
              by a formula target, other than as the clock;
            - a ``label_clock`` column's type does not match the clock's, a
              temporal one for a temporal clock and a numeric one for a
              numeric clock;
            - a group or session column, or an ``ew_class`` target, has a
              type with no text form, such as a nested one;
            - under ``group_close="monotone"``, the group column is not an
              integer, String or Categorical type.
        ``ValueError``
            When:

            - a clock is a ``UInt64`` past the largest ``Int64``;
            - between chunks, an integer clock column changes its width or
              becomes a float, a float clock becomes an integer, or a group
              or session column changes between integer, float, text and
              Boolean keys;
            - a spec gives its clock the other kind of clock parameter: a
              plain number to a temporal clock, or a duration to a numeric
              one (*Clock units* in :mod:`polars_online.spec`), or a
              ``gap_cap`` or ``restart_after_step_back`` finer than the
              clock's smallest step;
            - the clock has a null or non-finite value, or a step between
              two of its values that is not a finite number;
            - a weight is negative (a null weight skips the row);
            - a spec is named like an input column, which its output column
              would replace;
            - a group's clock steps back within a session, with
              ``restart_after_step_back`` not set, the default, or by no
              more than it. A chunk that overlaps what a loaded state has
              learned is refused the same way, and :meth:`skip_learned`
              drops the overlap;
            - under ``group_close="monotone"``, a group key is null or below
              the key before it;
            - a ``label_clock`` time is before its row's clock, or a group's
              rows waiting for their ``label_clock`` times would pass
              ``max_held_rows`` under ``held_overflow="refuse"``, the
              default;
            - a formula target's ``embargo`` is shorter than its longest
              look-ahead ``window_size``;
            - a chunk would take a hard window's snapshots past a
              ``window_budget`` that refuses;
            - a model refuses a value of its own, such as a ``strict_binary``
              target that is not 0 or 1, as its builder's docstring says;
            - the bank refused an earlier chunk after learning part of it
              (see Notes);
            - ``POLARS_ONLINE_MAX_THREADS`` is not a whole number, or is
              above 1024 (or the machine's core count, where that is
              larger), on a bank
              with ``threads=None``, at the first call that runs on the
              process's threads.
        ``RuntimeError``
            The bank is in use on another thread (:class:`ModelBank`).
        :class:`~polars_online.ReadinessWarning`
            A warning, not an error: issued when a finding that casts doubt
            on what a model reports first holds, such as a gate still unmet
            once the model has learned long enough to judge. That class's
            docstring lists each finding and how often it is issued.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, ridge=[1e-6, 0.1]
            )
            bank = po.ModelBank([spec])
            out = bank.fit_predict(df)                         # df, with the column "ridge" added
            pred = out["ridge"].struct.field("pred_y__r0.1")   # the slot with ridge 0.1
            first = pred.is_not_null().arg_true()[0]           # the first row after its warm-up
            print(f"{pred.count()} of {out.height} rows predicted, the first at row {first}")

        .. code-block:: text

            396 of 400 rows predicted, the first at row 4

        .. rubric:: See also

        - ``lf.online.fit_predict``, the same over a query, run as the query
          runs.
        - :meth:`fit_predict_batches`, the same over an iterator of chunks.
        - :meth:`fit`, which learns and keeps no output.
        - :meth:`predict`, which predicts and learns nothing.
        - `Why every prediction is out-of-sample
          <https://github.com/hgilde/polars-online/blob/main/docs/explanation/out-of-sample.md>`_.

        .. rubric:: Notes

        Chunk boundaries never change the numbers: a stream fed as one chunk
        or a thousand gives the same predictions to the bit. By default they
        move the rows ``coef`` is reported on, each group's last row in the
        chunk; under ``coef_every`` or ``max_rows_between_coefs`` not even
        those.

        A refused chunk leaves the bank as it was, so the corrected chunk can
        be fed. The bank checks every group's clock steps in every spec over
        the whole chunk, and plays the chunk's clock through each hard
        window's snapshot schedule, before it learns a row. Two overruns of
        a ``window_budget`` are found only as the rows are learned: one under
        ``drift_action="reset"``, whose resets the check cannot foresee, and
        one where a snapshot grows within the chunk. Then the chunk is
        refused after some of it has been learned, and the bank refuses
        every later call that learns, predicts or saves: load its last saved
        state with :meth:`load` and feed it the rows since that save.
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
        :class:`polars_online.ReadinessWarning`, once per (spec, group), or
        per session span under ``group_close="session"``, which it names: a
        coefficient the ridge determined more than the data did, or a
        ``max_error_inflation`` the stream has settled below
        (docs/WARMUP-AND-CONVERGENCE.md §3)."""
        from polars_online._frame import ReadinessWarning, _user_stacklevel

        for notice in self._native.take_notices():
            warnings.warn(ReadinessWarning(notice), stacklevel=_user_stacklevel())

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict a frame's rows from the bank as it stands, learning nothing.

        Use it to serve a bank that has learned: :meth:`load` it once, call ``predict``
        for each request, and feed the same rows to :meth:`fit_predict` later,
        in order, once their targets arrive. The bank is left exactly as it
        was, so any number of threads may call ``predict`` at once, and row
        order does not matter. For a query, use ``lf.online.predict``.

        .. rubric:: Parameters

        ``df``
            The rows to predict, as a ``DataFrame``. Each row is predicted as
            the next row of its group's stream would be. ``df`` needs each
            spec's feature, clock and ``group`` columns. A target column is
            optional: present, it gives ``resid`` and the standardized
            residual, and absent, they are null. The session column is
            optional, and is read for ``session_gap`` and
            ``group_close="session"``. A weight column is not read.

        .. rubric:: Returns

        ``df`` with each spec's output column added, with the fields
        :meth:`fit_predict` gives, computed from the current state. Some
        fields and rows differ from what ``fit_predict`` would give:

        .. list-table::
           :header-rows: 1
           :widths: 40 60

           * - the field or the row
             - what ``predict`` gives
           * - ``weight_sum``, ``penalty_selected_<t>``, ``sigma``, the residual
               quantiles, the autocorrelation and the metrics
             - the values the bank holds, unchanged by the rows
           * - ``coef``
             - filled on each group's last row that is not skipped, since the
               same coefficients predict every row. ``coef_every`` and
               ``max_rows_between_coefs`` are not read
           * - ``drift``
             - never fires
           * - a row of a group the bank has never seen, or a row whose
               features are null
             - null throughout, as a skipped row is in ``fit_predict``
           * - under ``group_close="session"``, a row from a session other
               than the group's current one
             - the first row of a new stream, null with ``weight_sum`` 0,
               since ``fit_predict`` would start the stream again there
           * - a row on which ``session_gap="reset"`` would reset the model
             - the same
           * - a row before the group's last learned clock
             - predicted from the state as it stands, whatever
               ``restart_after_step_back`` says: predicting learns nothing,
               so it neither refuses the row nor resets

        .. rubric:: Raises

        What :meth:`fit_predict` raises for a frame's columns and their
        types, such as a missing column or a bad clock value, with these
        differences:

        - it does not refuse what only learning would: a step back, a
          negative weight, a ``window_budget`` overrun or held rows past
          ``max_held_rows``;
        - a missing target column is not an error;
        - ``RuntimeError`` comes only when a call that learns is in flight on
          another thread, since ``predict`` calls may run at once.

        .. rubric:: Examples

        This code reads ``df`` and ``today`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)                        # learn df's rows
            served = bank.predict(today.drop("y"))      # the next rows, their targets not yet known
            pred = served["ridge"].struct.field("pred_y")
            print(f"{pred.count()} of {served.height} rows predicted, {bank.rows_fed} rows learned")

        .. code-block:: text

            400 of 400 rows predicted, 400 rows learned

        .. rubric:: See also

        - :meth:`fit_predict`, which predicts each row and then learns it.
        - :meth:`predict_arrow`, the same with the output as Arrow.
        - ``lf.online.predict``, the same over a query.

        .. rubric:: Notes

        A trend model (``holt``) extrapolates over the clock distance from the
        row it last learned, capped by ``gap_cap``. A ``kalman`` with
        ``revert_half_life`` shrinks its coefficients over that distance
        exactly as the next ``fit_predict`` row would. The other models that
        predict from coefficients use their current coefficients whatever the
        clock says.

        Under an ``embargo``, ``predict`` releases nothing. A held row whose
        delay has passed by the predicted row's clock stays held, and the next
        ``fit_predict`` learns it. So every row is predicted from the model
        the last ``fit_predict`` left, its ``weight_sum`` and
        ``learned_clock`` included.
        """
        return self._predict_from(df, 0)

    def _predict_from(self, df: pl.DataFrame, row_base: int) -> pl.DataFrame:
        """:meth:`predict` for rows ``row_base..`` of a longer input, as
        :meth:`_fit_predict_from` is to :meth:`fit_predict`."""
        self._check_frame(df, "predict")
        outs = self._native.predict(df, row_base)
        return df.with_columns([pl.Series(s) for s in outs])

    def fit_predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]:
        """Learn a frame's rows, and return each spec's output as an Arrow array.

        Use it to hand a bank's output to a library other than Polars, or to
        take the output through the public Arrow PyCapsule interface, without
        Polars' private series boundary (see Notes). For a frame
        back, use :meth:`fit_predict`, the same call with the output added to
        the frame for you.

        .. warning::
            The Arrow output (this method, :meth:`predict_arrow` and
            :class:`~polars_online.ArrowStruct`) is **unstable**: it may change
            in any release without that counting as a breaking change. Under
            ``POLARS_ONLINE_WARN_UNSTABLE=1``, each call issues
            :class:`~polars_online.UnstableWarning`.

        .. rubric:: Parameters

        ``df``
            The next rows of the stream, as one chunk, a Polars ``DataFrame``,
            as :meth:`fit_predict` takes it. Only the output is Arrow.

        .. rubric:: Returns

        A list of :class:`~polars_online.ArrowStruct`, one per spec, in the
        order of the bank's specs. Each holds the spec's output column as an
        Arrow struct array, with :meth:`fit_predict`'s values exactly, field
        for field and null for null, and is named after its spec. A consumer
        reads it through the Arrow PyCapsule interface
        (``__arrow_c_array__``): Polars' ``pl.Series``, and pyarrow's
        ``pa.array`` and ``pa.table``. DuckDB reads Arrow streams instead, so
        give it ``pl.Series(s)``. Each struct can be read once (see Notes).

        .. rubric:: Raises

        What :meth:`fit_predict` raises for the same frame.

        :class:`~polars_online.UnstableWarning`
            A warning, not an error: issued by each call under
            ``POLARS_ONLINE_WARN_UNSTABLE=1``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            structs = bank.fit_predict_arrow(df)            # one per spec, as Arrow
            out = df.with_columns([pl.Series(s) for s in structs])
            print(out["ridge"].struct.field("pred_y").count())

        .. code-block:: text

            396

        .. rubric:: See also

        - :meth:`fit_predict`, the same with the output added to the frame.
        - :meth:`predict_arrow`, which predicts and learns nothing.

        .. rubric:: Notes

        A Polars ``Series`` crosses from the extension to Python through
        Polars' private ``_export`` and ``_import``, an interface with no
        promise of stability, which is why :class:`ModelBank` needs Polars
        1.28.1 or later. The Arrow PyCapsule interface is a public standard, so a
        consumer of it reads the struct directly. Reading a struct hands its
        buffers to the consumer, so a second read raises ``ValueError``. The
        struct's Arrow field carries its spec's name, which ``pl.Series``
        gives the series it builds.
        """
        warn_unstable("ModelBank.fit_predict_arrow and the ArrowStruct it returns")
        self._check_frame(df, "fit_predict_arrow")
        out = self._native.fit_predict_arrow(df)
        self._warn_notices()
        return out

    def predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]:
        """Predict a frame's rows without learning, and return each spec's output as an Arrow array.

        Use it to serve a bank's predictions to a library other than Polars.
        It is to :meth:`predict` what :meth:`fit_predict_arrow` is to
        :meth:`fit_predict`: the same predictions, the same values, and the
        bank left exactly as it was.

        .. warning::
            The Arrow output (:meth:`fit_predict_arrow`, this method and
            :class:`~polars_online.ArrowStruct`) is **unstable**: it may change
            in any release without that counting as a breaking change. Under
            ``POLARS_ONLINE_WARN_UNSTABLE=1``, each call issues
            :class:`~polars_online.UnstableWarning`.

        .. rubric:: Parameters

        ``df``
            The rows to predict, as a Polars ``DataFrame``, as :meth:`predict`
            takes it.

        .. rubric:: Returns

        A list of :class:`~polars_online.ArrowStruct`, one per spec, as
        :meth:`fit_predict_arrow` returns, each readable once.

        .. rubric:: Raises

        What :meth:`predict` raises for the same frame.

        :class:`~polars_online.UnstableWarning`
            A warning, not an error: issued by each call under
            ``POLARS_ONLINE_WARN_UNSTABLE=1``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df.head(300))
            structs = bank.predict_arrow(df.tail(100))      # the bank is not changed
            served = df.tail(100).with_columns([pl.Series(s) for s in structs])
            print(served["ridge"].struct.field("pred_y").count(), bank.rows_fed)

        .. code-block:: text

            100 300

        .. rubric:: See also

        - :meth:`predict`, the same with the output added to the frame.
        - :meth:`fit_predict_arrow`, which learns the rows too.
        """
        warn_unstable("ModelBank.predict_arrow and the ArrowStruct it returns")
        self._check_frame(df, "predict_arrow")
        return self._native.predict_arrow(df)

    def _clocks(self) -> set[str | None]:
        """The clock columns this bank's specs read."""
        return {s.get("clock") for s in self._specs}

    def _check_frame(self, df: object, what: str) -> None:
        if isinstance(df, pl.DataFrame):
            _refuse_uint128(df.schema, f"ModelBank.{what}", self._clocks())
            return
        # A LazyFrame is the common slip, and the attribute error it used to
        # produce named an internal method.
        if isinstance(df, pl.LazyFrame):
            who = f"ModelBank.{what}"
            msg = f"{who}: it takes a frame, but it was given a query (a LazyFrame). "
            if what == "fit_predict":
                msg += (
                    "Pass the query's result as a DataFrame, or pass the query to "
                    "fit_predict_batches(lf), which reads it in chunks."
                )
            else:
                msg += "Pass the query's result as a DataFrame."
        else:
            who = f"ModelBank.{what}"
            msg = (
                f"{who}: it takes a Polars DataFrame, but it was given an object of type "
                f"{type(df).__name__}. Pass a DataFrame."
            )
        raise TypeError(msg)

    @keyword_only_forwarded("ModelBank.fit_predict_batches", ("closed_groups", "chunk_size"))
    @_renamed_keywords
    def fit_predict_batches(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        *,
        closed_groups: str | os.PathLike[str] | None = None,
        chunk_size: int | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Learn a stream one chunk at a time, and yield each chunk with each spec's output.

        Use it for a stream too large to hold in memory, when you want the
        output one chunk at a time, as a generator. For a frame in memory, use
        :meth:`fit_predict`. To keep only the state, use :meth:`fit`, and to
        run the bank as a step of a query, use ``lf.online.fit_predict``.

        .. rubric:: Parameters

        ``batches``
            The stream: a ``LazyFrame``, a ``DataFrame`` or an iterator of
            ``DataFrame`` chunks. A query is run ``chunk_size`` rows at a
            time, and each chunk is learned when the generator reaches it. A
            frame is one chunk, or slices of ``chunk_size`` rows when
            ``chunk_size`` is given. An iterator's frames are learned as they
            come.
        ``closed_groups``
            A path to which to write the closed groups' rows, in the format
            its extension names (see Notes). It takes a spec that closes
            groups, under ``group_close``. ``None``, the default, leaves the
            closed groups in the bank, for :meth:`closed_groups` to read.
        ``chunk_size``
            How many rows of a query or a frame make one chunk. ``None``, the
            default, reads a query 100,000 rows at a time and takes a frame
            whole. It does not cut an iterator's frames.

        .. rubric:: Returns

        A generator of ``DataFrame`` chunks, each the chunk's rows with one
        output column added for each spec, as :meth:`fit_predict` returns
        them.

        .. rubric:: Raises

        When called, before a row is read:

        ``ValueError``
            ``chunk_size`` is below 1, or ``closed_groups`` names no format
            by its extension, or names a file beside specs none of which sets
            ``group_close``, or names a CSV file while the closed rows hold a
            nested column that CSV cannot hold.
        ``FileNotFoundError``
            ``closed_groups`` names a file in a folder that does not exist.
        :class:`~polars_online.OrderNotGuaranteedWarning`
            A warning, not an error: issued when the query holds a step that
            does not keep row order, naming the step (see Notes).

        As the generator runs:

        ``TypeError``
            An iterator yields something other than a ``DataFrame``.
        Any error :meth:`fit_predict` raises
            For the chunk where it happens. The chunks before it have been
            learned.
        :class:`~polars_online.ConsumedSourceWarning`
            A warning, not an error: the query gave no rows, and it reads a
            source written in Python, which may give its rows only once.

        .. rubric:: Examples

        This code reads ``lf`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            for out in bank.fit_predict_batches(lf, chunk_size=100):
                print(out.height, out["ridge"].struct.field("pred_y").count())   # each chunk

        .. code-block:: text

            100 96
            100 100
            100 100
            100 100

        .. rubric:: See also

        - :meth:`fit_predict`, the same over one frame.
        - :meth:`fit`, which learns the same rows and keeps no output.
        - :meth:`closed_groups`, which reads the closed groups between chunks.
        - ``lf.online.fit_predict``, the bank as a step of a query.

        .. rubric:: Notes

        Memory holds the state and one chunk, however long the stream.

        A model learns rows in their order, so the order in which Polars
        delivers a query's rows is part of the result. A ``join``,
        ``group_by`` or ``unique`` that does not keep order may deliver the
        rows in another order than the query's whole result has, and in
        another order from one run to the next. Give a join
        ``maintain_order="left"``, a ``group_by`` ``maintain_order=True``,
        and sort after a ``unique``, or sort before the bank.

        Under ``closed_groups``, the bank's closed groups are read out after
        every chunk, so none waits in the bank when a chunk is handed on,
        and the rows read out are written to the file when the chunks stop.
        Without it, or a call to :meth:`closed_groups` between chunks, every
        closed group waits in the bank, and :meth:`save` writes them all. The
        file is written however the chunks stop, at their end, at a
        ``break`` or at an error, since a row read out of the bank is then in
        no other place. The file and the bank together always hold every
        group that closed. A CSV file holds each list column, such as
        ``coef``, as JSON text, the bytes the command line writes. Read it
        back with Polars' ``str.json_decode``.
        """
        return self._batches(batches, closed_groups, chunk_size, "fit_predict_batches")

    def _batches(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        closed_groups: str | os.PathLike[str] | None,
        chunk_size: int | None,
        what: str,
    ) -> Iterator[pl.DataFrame]:
        """The run behind :meth:`fit_predict_batches` and :meth:`fit`: the
        arguments checked eagerly, then a generator. ``what`` is the public
        method the caller used, for the messages that name it."""
        from polars_online._frame import (
            _check_closed_format,
            _closed_path,
            _order_free,
            _plan_text,
            _warn_if_order_unspecified,
        )

        if chunk_size is not None and chunk_size < 1:
            who = f"ModelBank.{what}"
            msg = (
                f"{who}: chunk_size must be at least 1, but it is {chunk_size}. Set chunk_size "
                "to 1 or more, or leave it out."
            )
            raise ValueError(msg)
        path = _closed_path(closed_groups, self._specs, f"ModelBank.{what}")
        if path is not None:
            _check_closed_format(path, self.closed_groups(drop=False).clear(), f"ModelBank.{what}")
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
    ) -> Iterator[pl.DataFrame]:
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
                    f"{guard}: the query gave no rows, and it reads a source written in Python, "
                    "such as `pl.scan_arrow_c_stream` over a DuckDB relation, a pyarrow reader or "
                    "anything with `__arrow_c_stream__`, which gives its rows once, so a second "
                    "run of the query reads nothing and this bank learns from nothing. Build the "
                    "query again for each run, calling "
                    "`pl.scan_arrow_c_stream(...)` inside the loop, or, if the source "
                    "really is empty, silence this warning with "
                    '`warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning)`.'
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

    @keyword_only_forwarded("ModelBank.fit", ("closed_groups", "chunk_size"))
    @_renamed_keywords
    def fit(
        self,
        batches: pl.LazyFrame | pl.DataFrame | Iterable[pl.DataFrame],
        *,
        closed_groups: str | os.PathLike[str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """Learn every row of a query, a frame or an iterator of frames, and keep only the state.

        Use it when the product of a run is the state, such as a bank you
        will save and serve, or coefficients you will read, and the
        predictions on the way are not needed. It goes on from the state the
        bank holds, as scikit-learn's ``partial_fit`` does, not as its
        ``fit``, which starts over: for a bank that starts with nothing
        learned, make a new :class:`ModelBank`. To keep the output too, use
        :meth:`fit_predict_batches`.

        .. rubric:: Parameters

        ``batches``
            The stream: a ``LazyFrame``, a ``DataFrame`` or an iterator of
            ``DataFrame`` chunks, as :meth:`fit_predict_batches` takes it.
        ``closed_groups``
            A path to which to write the closed groups' rows, as
            :meth:`fit_predict_batches` takes it. ``None``, the default,
            leaves them in the bank.
        ``chunk_size``
            How many rows of a query or a frame make one chunk, as
            :meth:`fit_predict_batches` takes it. ``None``, the default, reads
            a query 100,000 rows at a time and takes a frame whole.

        .. rubric:: Raises

        What :meth:`fit_predict_batches` raises, with two differences:

        - :class:`~polars_online.OrderNotGuaranteedWarning` is not issued
          when every spec reaches the same state in any row order (see
          Notes);
        - a spec whose ``embargo`` is shorter than its formula target's
          longest look-ahead ``window_size`` is not refused: ``fit`` learns
          each of its rows once the window the row's target reads has
          closed, while :meth:`fit_predict`, which would give predictions
          for them, refuses it.

        .. rubric:: Examples

        This code reads ``lf`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(lf, chunk_size=100)        # learn every row, keep no output
            bank.save("bank.state")
            print(bank.rows_fed)

        .. code-block:: text

            400

        .. rubric:: See also

        - :meth:`fit_predict_batches`, the same run with its output yielded.
        - :meth:`save`, which writes the state the run leaves.

        .. rubric:: Notes

        ``fit`` is :meth:`fit_predict_batches` with each chunk's output
        dropped as it comes, so no output is held and no frame is assembled
        from the chunks. Over a long stream that saves the memory the output
        would take. A spec whose output on each row is ``weight_sum`` alone
        still writes gigabytes over a billion rows. A fit whose product is
        its coefficients has no use for its predictions. The state ``fit`` leaves is
        the state :meth:`fit_predict_batches` leaves over the same rows, byte
        for byte. Each row is still predicted before it is learned, and the
        output columns are still built before they are dropped. Two calls
        over two days of rows leave the state one call over both would.

        **A fit whose specs reach the same state in any row order is not
        warned about row order.** Such a spec is an ``ewridge`` or ``rls``
        with no decay (``lam=1.0`` or ``half_life=float("inf")``) and none
        of the settings that read the row order, which are:

        - a hard window, a session, a weight, an embargo or ``label_clock``;
        - a drift reset, a ``coef`` cadence or a formula target;
        - ``gram_block_rows``, ``restart_after_step_back`` or ``group_close``;
        - a ``ridge_scale`` other than ``"mean"``, or a ``target_gaps`` other
          than ``"own_rows"``;
        - ``conformal``, ``long_half_life``, ``session_shrink`` or
          ``emit_averaged``;
        - any diagnostic but ``emit_drift``, ``emit_clocks`` and
          ``emit_error_inflation``;
        - any other setting not known to keep the state free of the row
          order.

        Such a spec's running sums are the same whatever order the rows are
        added in, to rounding. Measured
        over 200 rows, the coefficients differed by at most 6.7e-16 for
        ``ewridge`` and 1.4e-15 for ``rls``. The warning stays for
        :meth:`fit_predict_batches` over the same specs, whose predictions are
        out-of-sample and so move with the order: by up to 0.087 for
        ``ewridge`` on those rows. It stays for every model whose fit
        depends on the order it learns rows (``sgd``, ``pa``, ``ftrl``,
        ``quantile`` and a reweighting ``huber`` differ materially with no decay at all), and
        for a model that chooses by its out-of-sample error, as ``lasso``
        chooses its penalty.
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
        """Return the coefficients of every fit the bank holds, one row per coefficient.

        Use it to read the coefficients from a bank, or from one loaded from a
        state file, without a row of data. For the coefficients on each row of a
        stream, read the ``coef`` field of :meth:`fit_predict`'s output, and for
        the sums behind a fit, :meth:`gram`.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. The default, ``None``,
            reads every spec and skips each one that has no coefficients (see
            Raises). A bank whose specs have no coefficients at all gives an
            empty frame, with the same columns.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with one row per (spec, group, instance, position), in
        spec order, then in :meth:`groups`' order. It leads with ``spec``, as
        :meth:`last_row`, :meth:`summary` and :meth:`describe` do. Every frame
        it returns has the same columns, so the frames of several banks stack
        with a plain ``pl.concat``. The values are the fit after the last row
        each model learned, which the next row's
        ``pred_<t>`` is computed from. That is the fit the output's ``coef`` field
        shows on the rows where it is written. The columns:

        ``spec``, ``group``, ``instance``
            The spec's name, the group key as :meth:`groups` reports it, and the
            instance's field suffix (``"@h500"``, or ``""`` for a spec with one
            instance). An instance is the part of a model that fits one
            half-life of a grid.
        ``half_life``, ``lam``
            The instance's decay, as :func:`polars_online.spec.output_index`
            gives it: the half-life in clock units (``inf`` for no decay), or
            the ``lam``. The other one is null.
        ``weight_sum``
            The accumulated weight behind the fit, which the next row's
            ``weight_sum`` field reports.
        ``position``, ``target``, ``ridge``, ``feature_set``, ``penalty``
            :func:`polars_online.spec.coef_index`'s columns: ``position``
            indexes the flat ``coef`` list, and the others name the
            slot it belongs to.
        ``cluster``, ``state``, ``class``
            ``coef_index``'s columns too: the ``kmeans`` center, the ``hmm``
            state or the ``ew_class`` class a position belongs to. Null for
            every other model.
        ``term``
            ``"intercept"``, a feature's name, ``"level"`` or ``"trend"`` for
            ``holt``, or ``"rho"`` for ``deco``.
        ``coef``
            The coefficient, in the features' original units. Null until the
            model's first solve, and null wherever the model reports no
            number, as the ``coef`` field is on those rows: a fit withheld as
            too ill-conditioned to keep a digit, or an ``ew_class`` class no
            row has carried yet.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.
        ``ValueError``
            ``spec`` names a spec whose model has no coefficients to index:
            ``ew_cov``, which gives statistics; ``seqtest``, which gives
            evidence; ``micro``, whose number of clusters varies; and
            ``marginal``, ``rcov``, ``corrchange``, ``bocpd`` and ``audit``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(df)
            coefs = bank.coef()                                 # one row per coefficient
            for term, value in coefs.select("term", "coef").iter_rows():
                print(f"{term}: {value:.2f}")                   # the truth: 1, 2 and -1

        .. code-block:: text

            intercept: 0.99
            x0: 1.97
            x1: -1.00

        To read one slot of a grid, filter on its columns, such as
        ``coefs.filter(pl.col("ridge") == 0.1)``. To lay the terms out as
        columns, pivot the frame on every column that tells two fits apart:
        ``pivot("term", index=["spec", "group", "instance", "target",
        "ridge", "feature_set", "penalty", "cluster", "state", "class"],
        values="coef")``.

        .. rubric:: See also

        - :func:`polars_online.spec.coef_index`, the layout of a spec's
          ``coef`` list.
        - :meth:`last_row`, the whole output of each model's last row.
        - :meth:`gram`, the Gram a fit is solved from.
        - :meth:`solve_failures`, the solves that needed help.

        .. rubric:: Notes

        ``coef`` can be filled before ``pred_<t>`` is. ``pred_<t>`` waits for
        ``min_weight``, while ``coef`` reads each solve the model has run, and
        a model's solve schedule can run one sooner. For example, an
        ``ewridge`` with three coefficients, ``min_weight=10``, a half-life of
        100 and the default schedule carries coefficients on its third row,
        while its ``weight_sum`` reads near 2. :func:`polars_online.spec.ewridge`
        gives its schedule (``solve_every``). A coefficient read while ``weight_sum`` is below
        ``min_weight`` comes from fewer rows than the spec asks of a
        prediction.

        Sweeping every spec skips the ones with no coefficients, since the
        question was about the coefficients in the bank. Naming one of them
        still raises, since then the question was about that spec.
        """
        names = self._native.spec_names()
        keys = _group_keys(group, "ModelBank.coef")
        if spec is not None:
            i = self._spec_index(spec, "ModelBank.coef")
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
            return pl.DataFrame(schema=_COEF_SCHEMA_HEAD | _coef_index_schema() | _COEF_VALUE)
        return pl.concat(frames, how="diagonal_relaxed")

    def _coef_try(self, idx: int, group: list[str | None] | None, name: str) -> pl.DataFrame | None:
        """[`_coef_one`] or ``None`` for a spec that has no coefficients."""
        try:
            return self._coef_one(idx, group, name)
        except ValueError:
            return None

    def _coef_one(self, idx: int, group: list[str | None] | None, name: str) -> pl.DataFrame:
        spec = self._specs[idx]
        layout = coef_index(spec)
        if layout.is_empty():
            kind = spec["model"]["type"]
            why = _NO_COEF.get(
                kind,
                f"a {kind} model has no coefficients. Read its fields from the spec's output "
                "column.",
            )
            msg = f'spec "{name}": {why}'
            raise ValueError(msg)
        n = layout.height
        # Each instance's decay, by its suffix, as `output_index` gives it.
        decay = {
            i: (h, lam)
            for i, h, lam in output_index(spec)
            .filter(pl.col("kind") == "weight_sum")
            .select("instance", "half_life", "lam")
            .iter_rows()
        }
        groups: list[str | None] = []
        instances: list[str] = []
        n_effs: list[float] = []
        values: list[float | None] = []
        for g, instance, weight_sum, coef in self._native.coef(idx, group):
            if coef is not None and len(coef) != n:
                msg = (
                    f'spec "{self._specs[idx]["name"]}": the model returned {len(coef)} '
                    f"coefficients for {n} positions, which is a defect in polars-online. Open "
                    "an issue at https://github.com/hgilde/polars-online/issues with this message."
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
            pl.Series("half_life", [decay[i][0] for i in instances], pl.Float64),
            pl.Series("lam", [decay[i][1] for i in instances], pl.Float64),
            pl.Series("weight_sum", n_effs, pl.Float64),
            # Finite-or-null, as the output's `coef` field is: an `ew_class`
            # class no row has carried yet has NaN means.
            pl.Series("coef", values, pl.Float64).fill_nan(None),
            pl.Series("spec", [name] * len(instances), pl.String),
        ).select(*_COEF_SCHEMA_HEAD, *layout.columns, *_COEF_VALUE)

    def last_row(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Return each model's output on the last row it did not skip, one row per spec and group.

        Use it to see how each model was doing at the end of a stream, from the
        bank or from one loaded from a state file, without keeping the output
        frame. For the coefficients alone, whichever row was last, use
        :meth:`coef`, and for what each model was fed, :meth:`summary`.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. The default, ``None``,
            reads every spec.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with ``spec`` and ``group``, then the row
        :meth:`fit_predict` reported for that row, field for field, unnested:
        ``pred_<t>``, ``resid_<t>``, ``weight_sum``, ``settled_frac``,
        ``withheld_reason``, the diagnostics' fields, and ``coef`` where the
        row carried it. Rows come in
        spec order, then in :meth:`groups`' order. Specs with different fields
        are stacked into one frame, so a field a spec does not write is null
        on its rows. The row is the last one the model predicted, whether or
        not it learned from it: a row of weight 0 or with a null target
        replaces it, and a skipped row does not. A group whose every row so
        far was skipped is a row of nulls.

        ``coef`` is on the row where the output carried it: by default each
        group's last row in a chunk that was not skipped, and under ``coef_every`` or
        ``max_rows_between_coefs`` only a row on that schedule. So it can be
        null here where :meth:`coef` has the fit.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, group="stock_id"
            )
            bank = po.ModelBank([spec])
            bank.fit(df)
            last = bank.last_row()                              # one row per (spec, group)
            for group, pred in last.select("group", "pred_y").iter_rows():
                print(f"{group}: {pred:.2f}")                   # each stock's last prediction

        .. code-block:: text

            b0: 4.72
            b1: -0.11
            b2: 2.92
            b3: 1.80

        To compare saved banks, stack their last rows, each marked with its
        file:

        .. code-block:: python

            bank.save("bank.state")                             # one saved bank; list more
            files = ["bank.state"]
            table = pl.concat(
                [po.ModelBank.load(f).last_row().with_columns(file=pl.lit(f)) for f in files],
                how="diagonal_relaxed",
            )

        .. rubric:: See also

        - :meth:`coef`, the coefficients of each fit, one row per
          coefficient.
        - :meth:`summary`, what each model was fed.
        - :meth:`fit_predict`, whose output fields this row repeats.

        .. rubric:: Notes

        The row travels with the state, so a bank loaded from a file gives the
        row its saved bank would. :meth:`predict` does not move it, and a chunk
        that ends in skipped rows leaves the row before them.
        """
        names = self._native.spec_names()
        picked = (
            range(len(names)) if spec is None else [self._spec_index(spec, "ModelBank.last_row")]
        )
        wanted = _group_keys(group, "ModelBank.last_row")
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
        """Return what each model has been fed, one row per spec and group.

        Use it after a run to see how many rows each model learned, skipped or
        reset on, and whether it is warmed up, from the bank or from one
        loaded from a state file. For each input column's statistics use
        :meth:`describe`, and for findings about the data, :meth:`check`.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. The default, ``None``,
            reads every spec.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with one row per (spec, group), in spec order, then in
        :meth:`groups`' order. The counts and ranges cover every row routed to
        the group since its state began, undecayed, so they say what the model
        learned from, not what it still remembers. Every count is an
        ``Int64``. The columns:

        ``spec``, ``group``
            As :meth:`groups` reports them.
        ``rows_fed``
            The rows routed to the group, skipped or not, as :meth:`groups`
            counts them.
        ``rows_processed``
            The rows the models saw: every feature and the weight usable.
        ``rows_skipped``
            ``rows_fed - rows_processed``: the rows with a null, NaN, infinite
            or out-of-bound feature or weight.
        ``rows_learned``
            The processed rows with a positive weight and, for a model with
            targets, at least one usable target. A formula target's row counts
            when it is released with a value, once its window has closed.
        ``rows_zero_weight``
            The processed rows with weight 0: the clock moved, and nothing was
            learned.
        ``weight_sum``
            The sum of the processed rows' weights, 1 for a row without a
            weight column. It is undecayed, so it is not the output's
            ``weight_sum`` field.
        ``clock_min``, ``clock_max``, ``clock_last``
            The range of the clock fed, and its last value, exactly, in the
            clock column's own dtype, as :meth:`groups` gives ``clock_last``.
            Null for a spec without a clock.
        ``session_changes``
            The rows whose session differed from the row before.
        ``clock_backwards``
            The rows whose clock fell below the row before within a session,
            which ``restart_after_step_back`` decided about.
        ``resets``
            The rows at which ``session_gap = "reset"`` or
            ``restart_after_step_back`` reset the model.
        ``solve_failures``
            The solves that needed jitter or kept the previous fit, which
            :meth:`solve_failures` reports.
        ``refit_given_up``
            The targets whose refit of their first rows was given up, past the
            spec's ``refit_budget`` or past the cap on how long its buffer of
            rows ages. Four models refit their first rows and count it:
            ``huber``, ``quantile``, ``sgd(loss="huber")``, and a ``marginal``
            whose bins refit (``bin_refit_rows``). 0 on every other spec.
        ``labels_missing``, ``labels_dropped``
            Under ``label_clock``, the rows whose targets were never learned.
            ``labels_missing`` counts the rows not skipped whose
            ``label_clock`` is null.
            ``labels_dropped`` counts the rows held for their ``label_clock``
            time and dropped, by ``held_overflow="drop_oldest"`` or by a clock
            that reset the model. 0 on every other spec.
        ``settled_frac``
            How far the first instance (the part of the model that fits the
            spec's first half-life) has come toward its steady state under
            decay, after the last row: the fraction the next row's ``settled_frac`` field
            reads. Under ``embargo`` it counts the clock the held rows have
            covered too. Null where the model has no such reading.
        ``weight_sum_settled``
            The weight the model's ``weight_sum`` tops out at, projected from
            where it stands (see Notes). It says, before the model gets there,
            whether a ``min_weight`` can be met. Null after a single row, with
            no decay, and under ``window_size``.
        ``error_inflation``
            The largest ratio the ``max_error_inflation`` gate checks, over the
            first instance's slots (see Notes). Null where the model has no
            such gate.
        ``support_coef_min``, ``support_coef_min_feature``, ``n_coef``
            The smallest coefficient's data share, the feature it belongs to,
            and the number of coefficients a target, read from the model's
            first instance after the last row, as ``settled_frac`` and
            ``error_inflation`` are.

        An ``audit`` (:func:`polars_online.spec.audit`) reads every row and
        learns from none: its ``rows_processed`` is ``rows_fed``, its
        ``rows_skipped`` is 0, and its ``rows_learned`` and ``n_coef`` are
        null. What it counted is :meth:`audit`'s.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.
        ``ValueError``
            ``spec`` is ``None`` and two specs read clocks of different
            dtypes, which one clock column cannot hold. Read them one spec at a
            time, as for :meth:`groups`.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0,
                clock="t", gap_cap=300.0,
            )
            bank = po.ModelBank([spec])
            holes = pl.when(pl.col("t") % 10 == 0).then(None).otherwise(pl.col("x1"))
            bank.fit(df.with_columns(x1=holes))                 # x1 null on every tenth row
            row = bank.summary().row(0, named=True)             # one row: the spec has no groups
            print(row["rows_fed"], row["rows_processed"], row["rows_skipped"], row["rows_learned"])

        .. code-block:: text

            400 360 40 360

        .. rubric:: See also

        - :meth:`describe`, each input column's statistics.
        - :meth:`check`, findings about the data, read from this summary and
          more.
        - :meth:`solve_failures`, one of these columns alone.
        - `Is it ready?
          <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready>`__,
          on ``settled_frac``, ``error_inflation`` and ``weight_sum_settled``.

        .. rubric:: Notes

        The counts are kept in the state file, so a bank loaded from a file
        gives them too. :meth:`predict` moves none of them, and feeding the
        same rows in one chunk or a thousand gives the same numbers to the
        bit.

        ``settled_frac`` reads the model's own decay, ``half_life``, as
        ``min_settled_frac`` and the readiness warnings do. It does not read a
        second memory the model keeps beside it, such as ``holt``'s trend: the
        diagnostics wait for the longest of those instead (`One memory per
        diagnostic
        <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#one-memory-per-diagnostic-run-once-or-windowed>`__).

        ``error_inflation`` is ``ewridge``'s ``sqrt(1 + edf / n_kish)``,
        ``rls``'s ``sqrt(1 + k_total / n_kish)`` and ``lasso``'s
        ``sqrt(1 + df / n_kish)``. ``kalman``'s gate reads each row's own
        value, so its reading here is the gate's mean over the design,
        ``sqrt(1 + sum_i P_ii E[z_i^2] / R)``.

        ``weight_sum_settled`` is ``(W - w1 * (1 - s)) / s``. ``W`` is the
        accumulated weight a next row would report as ``weight_sum``, ``w1`` a
        row's weight (1 without a weight column, the rows' mean with one), and
        ``s`` how far the learned rows have come toward their steady state.
        ``s`` is ``settled_frac`` except under ``embargo``, where the held rows' clock
        is left out: ``W`` has neither decayed by nor accumulated them, and the
        projection pairs a weight with the clock it has covered. On a regular
        stream, rows ``d`` apart of one weight, it is exactly the ceiling
        ``1 / (1 - 2 ** (-d / half_life))`` times that weight, from the second
        row on. It is exactly 0 where ``W - w1 * (1 - s)`` is within its own
        rounding: a model whose weight is still its first row's alone, which
        would otherwise read a few 1e-16 above or below 0, by the platform's
        math library. The rounding allowed is ``8 * 2**-53`` times the rows
        read times ``W + w1``.
        """
        n = len(self._native.spec_names())
        picked = range(n) if spec is None else [self._spec_index(spec, "ModelBank.summary")]
        return self._native.summary(list(picked), _group_keys(group, "ModelBank.summary"))

    def describe(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Return statistics of each input column each model was fed, one row per column.

        Use it to see the counts and moments of the features, targets and
        weight each model was fed, from the bank or from one loaded from a
        state file. For each model's row counts use :meth:`summary`, and for
        findings about the data, :meth:`check`.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. The default, ``None``,
            reads every spec.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with one row per (spec, group, input column), in spec
        order, then in :meth:`groups`' order, and within each group the
        features, then the targets, then the weight column. The columns:

        ``spec``, ``group``
            As :meth:`groups` reports them.
        ``column``, ``role``
            The column's name, and ``"feature"``, ``"target"`` or ``"weight"``.
        ``count``, ``unusable_count``
            The rows fed, split in two. A value counts when it is finite and
            within the input bound, as the models take it, and is unusable
            otherwise: a Polars null, NaN, an infinity and a magnitude beyond
            the bound alike. So ``unusable_count`` is not Polars'
            ``null_count``, which counts the nulls alone.
        ``mean``, ``std``, ``min``, ``max``
            Over the counted values of every row fed, skipped rows included,
            undecayed and in row order, so chunking
            cannot move them. ``std`` is the sample standard deviation
            (``ddof=1``), null below two values.

        A model with no targets lists none. An ``ew_class`` label column has
        its counts only. A ``seqtest`` that compares two specs (``a=`` and
        ``b=``) lists as its target the value it tests on each row,
        ``|resid_b| - |resid_a|``, named as the spec names it. A formula target counts the
        value each row is learned from, when the row is learned: a null until
        the row's window closes and its ``embargo`` passes, and a null for a
        row still held when the input ends.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(df)
            stats = bank.describe().select("column", "role", "count", "mean")
            for column, role, count, mean in stats.iter_rows():
                print(f"{column} ({role}): {count} values, mean {mean:.2f}")

        .. code-block:: text

            x0 (feature): 400 values, mean -0.04
            x1 (feature): 400 values, mean -0.02
            y (target): 400 values, mean 0.94

        .. rubric:: See also

        - :meth:`summary`, each model's row counts.
        - :meth:`check`, which reads these statistics for its findings.
        - :meth:`audit`, the fuller counts an ``audit`` spec keeps.
        """
        names = self._native.spec_names()
        picked = (
            range(len(names)) if spec is None else [self._spec_index(spec, "ModelBank.describe")]
        )
        keys = _group_keys(group, "ModelBank.describe")
        frames = [
            self._native.describe(i, keys).select(pl.lit(names[i]).alias("spec"), pl.all())
            for i in picked
        ]
        return pl.concat(frames)

    def gram(
        self,
        spec: str | int,
        group: str | Iterable[str | None] | None = None,
        *,
        dtype: Any = "float64",
        layout: Literal["full", "packed"] = "full",
    ) -> list[Gram]:
        """Return the Gram behind a spec's fit, per group and instance, as numpy arrays.

        Use it to do with a fit's Gram what the model does not: a custom
        penalty, an information criterion, a condition number, standard errors,
        or a fit pooled over groups, with the functions of
        :mod:`polars_online.gram`. For the coefficients alone use :meth:`coef`,
        and for the sums of a group that has closed, :meth:`closed_groups`.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. Only ``ewridge``,
            ``lasso`` and ``ew_cov`` report. Every other model gives an empty
            list, which is not an error: ``huber`` and ``quantile`` keep a
            reweighted Gram per target that is not exported, ``rls`` keeps a
            Cholesky factor, ``kalman`` a covariance, and the gradient models
            no second moment.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group, in :meth:`groups`' order. A key
            the bank has never seen gives an empty list.
        ``dtype``
            The dtype of ``comoments`` and ``lag_comoments``: ``"float64"``, the
            default, or ``"float32"``, which halves their bytes (see Notes). A
            numpy or Polars dtype of either is taken too. Every other array
            stays float64.
        ``layout``
            ``"full"``, the default, for ``comoments`` as a ``(k, k)`` matrix,
            or ``"packed"`` for its upper triangle with the diagonal, row by row,
            a 1-D array of ``k * (k + 1) // 2``: about half the bytes (see
            Notes). ``lag_comoments`` is never packed, since a lagged matrix is
            not symmetric.

        .. rubric:: Returns

        A list of dicts, one per (group, instance), an instance being the
        part of the model that fits one half-life, and one per Gram where a
        spec reads several. Under ``target_gaps = "own_rows"``, the default,
        targets missing on different rows are fitted from Grams of their own,
        one per set of targets present on the same rows, and each dict names
        its ``targets``. ``k`` is the number of columns, and the arrays are
        exponentially weighted (EW) and decayed on the model's own clock. Each
        dict has:

        ``group``, ``instance``
            The group key as :meth:`groups` reports it (``""`` for a spec
            without a ``group`` column, ``None`` for a null key), and the
            instance's field suffix (``"@h500"``, or ``""`` for a single
            instance).
        ``columns``, ``targets``
            What the axes mean. ``columns`` is the spec's features, with
            ``"intercept"`` first when the spec has one: the ``term`` names of
            :func:`polars_online.spec.coef_index`. ``targets`` is the target
            names the per-target arrays are indexed by: the spec's, or the ones
            this Gram's rows belong to. It is empty for ``ew_cov``, which
            learns from none. The names make the dict self-describing, so
            :mod:`polars_online.gram` takes a column or a target by name.
        ``weight_sum``
            The accumulated weight behind these moments.
        ``n_kish``
            Kish's effective sample size, ``weight_sum**2 / sum(w**2)``: the
            number of equally weighted rows these moments are worth, which a
            standard error divides by (see Notes). ``None`` before the first
            row, and under a hard window where subtracting the rows that leave
            keeps no digit of the sum of squared weights.
        ``means``
            The EW column means, shape ``(k,)``.
        ``comoments``
            The centered co-moments, the EW analogue of a centered ``X'X / n``,
            shape ``(k, k)``, or packed under ``layout="packed"``. Centering is
            what keeps them accurate at a large offset.
        ``cross_moments``
            Per target, the uncentered cross-moments ``E[z*y]``, with ``z`` a
            row's columns and ``y`` the target, over the rows the target was
            present on, shape ``(n_targets, k)``. Empty for
            ``ew_cov``.
        ``means_by_target``
            Per target, the column means over the rows it was present on, shape
            ``(n_targets, k)``: what its cross-moments are centered at, and its
            intercept is recovered from. They equal ``means``, to rounding, for
            a target present on every row the Gram learned, which every target
            is under ``target_gaps = "own_rows"``. Under ``"pairwise"`` a
            target with gaps has its own. :func:`polars_online.gram.solve` reads
            them. Empty for ``ew_cov``.
        ``cross_centered``
            Per target, the cross-moments centered at its column means and its
            own mean, ``E[(z - m_t) * (y - ybar_t)]``, shape
            ``(n_targets, k)``: what the model solves from, and what
            :func:`polars_online.gram.solve` reads. Empty for ``ew_cov``.
        ``target_weights``
            Per target, the accumulated weight, shape ``(n_targets,)``. It
            differs from ``weight_sum`` when targets have different null
            patterns.
        ``target_means``, ``target_vars``
            Per target, the EW mean and centered variance of the target itself,
            shape ``(n_targets,)``, in the same arithmetic as ``comoments``: a
            target's variance here is, to the bit, the one an ``ew_cov`` over
            that column would report. Empty for ``ew_cov``.
        ``target_n_kish``
            Per target, Kish's effective sample size over that target's rows,
            ``target_weights**2 / sum(w**2)``. ``nan`` for a target that has not
            seen a weighted row, and under a hard window where the squared
            weights keep no digit, as for ``n_kish``. Empty for ``ew_cov``.
        ``lags``, ``lag_comoments``
            The lags an ``ew_cov(lags=[...])`` accumulates at, in the order
            given, and their cross-moments as an ``(L, k, k)`` array, ``L`` the
            number of lags. ``lag_comoments[l][a][b]`` is
            ``E_w[d_a(t) * d_b(t - lags[l])]``, both deviations taken against
            the mean before the row. The matrix is not symmetric for a lag
            above 0, since ``a`` leading ``b`` is not ``b`` leading ``a``, and
            at lag 0 it would be ``comoments`` exactly. Both are ``None`` for a
            spec without ``lags``.

        Values are in the features' original units. The intercept, when the
        spec has one, is column 0: a constant 1, so it has zero variance in
        ``comoments``. Under a ``window_size`` every array is the hard
        window's, the target moments included, since the window's snapshots
        carry them too. So :func:`polars_online.gram.solve` on it fits the
        hard window as it stood after the last row.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.
        ``ValueError``
            ``dtype`` is not float64 or float32, or ``layout`` is not
            ``"full"`` or ``"packed"``. Under ``dtype="float32"``, an entry
            float32 cannot hold to its precision, which the message names by
            its column (see Notes).
        ``ModuleNotFoundError``
            numpy is not installed. It is not a dependency of this package, as
            it is not of Polars: ``pip install polars-online[numpy]`` adds it.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, ridge=1e-6
            )
            bank = po.ModelBank([spec])
            bank.fit(df)
            g = bank.gram("ridge")[0]                           # one dict: one group, one instance
            coef = po.gram.solve(g, target="y", ridge=1e-6)     # solved again from the Gram
            print(g["columns"], coef.round(2))
            print(bank.coef()["coef"].round(2).to_list())       # the model's own fit
            var_y = g["target_vars"][0]                         # the target's moments give its R²
            r2 = 1 - (var_y - coef[1:] @ g["comoments"][1:, 1:] @ coef[1:]) / var_y
            print(f"R² {r2:.2f}")

        .. code-block:: text

            ['intercept', 'x0', 'x1'] [ 0.99  1.97 -1.  ]
            [0.99, 1.97, -1.0]
            R² 0.94

        .. rubric:: See also

        - :mod:`polars_online.gram`, the functions that pool, narrow and solve
          these dicts.
        - :meth:`closed_groups`, the same sums for each group that has closed.
        - :meth:`coef`, the fit the model solved.
        - `How a Gram stands in for the rows
          <https://github.com/hgilde/polars-online/blob/main/docs/GRAM.md#how-a-gram-stands-in-for-the-rows>`__.

        .. rubric:: Notes

        **Why the Gram is exported.** The Gram is the expensive part of a
        fit, and it is already exact, centered, decayed on the
        model's own clock with its session and ``gap_cap`` handling, and
        resumable. Without it, anything other than the model's own solve
        (a custom penalty, an information criterion, ``cond(G)``, a scree
        plot, forward stepwise, orthogonal matching pursuit, or a fit checked
        by hand) would recompute ``X'X`` from the raw data in a second pass.
        It comes from one pass over data that is never materialized, at
        every point in the stream rather than one, and it holds the matrices
        the deployed model solves against. It is not a speed claim: for one
        Gram computed at once over materialized data, BLAS ``dgemm`` is blocked,
        vectorized, and comfortably faster. The target moments make the
        export a complete sufficient statistic. With the cross-moments alone
        a saved Gram gives no residual variance, no R², no information
        criterion and no standard error, since each of them needs ``Var[y]``.

        **Kish's size.** ``weight_sum`` counts weight, not rows, so it is not
        a sample size. ``(1 + lam**d) / (1 - lam**d)`` is the Kish size of an
        EW window of unit rows ``d`` clock units apart, with
        ``lam = 0.5 ** (1 / half_life)``. It is scale-free: decay scales
        ``weight_sum**2`` and ``sum(w**2)`` by the same factor, so it does not
        shrink when a stream goes quiet. It says how many rows these moments average, not how old
        they are, which ``weight_sum`` and ``target_weights`` say. Under a
        hard window it is ``None`` where the window's sum of squared weights is
        no more than ``64 eps`` times the history's: a window of rows far lighter than
        the history, where the ratio would be rounding.

        **The centered cross-moments.** ``cross_centered[t]`` equals
        ``cross_moments[t] - means_by_target[t] * ybar_t`` in exact
        arithmetic. Formed in floating point at a level ``L``, that difference
        carries a rounding error of about ``L**2 * eps``, so the model keeps
        the centered form itself.

        **The two moment forms.** Mixing the raw and the centered forms gives
        a wrong answer, not an error, so the identities between them are
        stated here, with ``coef[t]`` target ``t``'s coefficients, intercept
        first. For a target present on every row the Gram holds, which is
        every target under ``target_gaps = "own_rows"``:

        .. code-block:: text

            raw = comoments + outer(means, means)
            raw @ coef[t] == cross_moments[t]                       (up to the ridge term)

        and, for a spec with an intercept, every target's centered form the
        model solves, up to the ridge term, with ``m = means_by_target[t]``
        and ``ybar = cross_moments[t][0]``:

        .. code-block:: text

            comoments[1:, 1:] @ coef[t][1:] == cross_centered[t][1:]
            coef[t][0] == ybar - m[1:] @ coef[t][1:]

        With an intercept, ``raw[0] == means``.

        **A compact Gram.** ``comoments``, and ``lag_comoments`` under
        ``lags``, are the arrays that grow as ``k²``: ``comoments`` is 800 MB
        a Gram in float64 at 10,000 columns. Together, the two options make
        ``comoments`` a quarter of its bytes and ``lag_comoments`` half, since
        it is never packed. Every function of
        :mod:`polars_online.gram` takes either, merging and solving in float64
        whatever it is given. The state itself stays float64 and whole: the
        options shape only what is handed back.

        Under ``dtype="float32"`` each entry is the float64 rounded to
        nearest, so it is within ``2**-24`` of it, relative, about 6e-8, and a
        solve read from it carries that error times the system's condition
        number. The other arrays stay float64: they are ``k`` long, and the
        means are what a merge's spread term is made of. An entry is refused
        when float32 cannot hold it to that precision: past float32's largest
        number, about 3.4e38, which would read ``inf``, or nonzero and below
        its smallest normal number, about 1.2e-38, which would read 0 or keep
        fewer digits. A column spread 1e20 wide, or 1e-20, gets there, since
        its variance squares the scale. Rescale it with Polars before the
        bank, or read the Gram in float64.

        Under ``layout="packed"`` entry ``(i, j)`` with ``i <= j`` is at
        ``i * (2 * k - i + 1) // 2 + (j - i)``, the order of
        ``numpy.triu_indices(k)``, and ``(j, i)`` is the same entry. Laid back
        into a ``k x k`` array ``m`` by ``m[np.triu_indices(k)] = packed`` and
        mirrored, it is the full layout's matrix, bit for bit, since the
        model keeps the matrix symmetric. A closed row's ``comoments`` is
        packed the same way.
        """
        try:
            import numpy as np
        except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
            msg = (
                "ModelBank.gram: it returns numpy arrays, and numpy is not installed. Install "
                "it with `pip install numpy` or `pip install polars-online[numpy]`."
            )
            raise ModuleNotFoundError(msg) from e

        if layout not in ("full", "packed"):
            msg = (
                f'ModelBank.gram: layout must be "full" or "packed", but it is {layout!r}. '
                "Set layout to one of those words."
            )
            raise ValueError(msg)
        polars_dtypes = {pl.Float64: "float64", pl.Float32: "float32"}
        try:
            as_dtype = np.dtype(polars_dtypes.get(dtype, dtype))
        except TypeError as e:
            msg = (
                f'ModelBank.gram: dtype must be "float64" or "float32", but it is {dtype!r}. '
                "Set dtype to one of those words."
            )
            raise ValueError(msg) from e
        if as_dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            msg = (
                f'ModelBank.gram: dtype must be "float64" or "float32", but it is {dtype!r}. '
                "Set dtype to one of those words."
            )
            raise ValueError(msg)
        float32 = as_dtype == np.dtype(np.float32)
        packed = layout == "packed"
        idx = self._spec_index(spec, "ModelBank.gram")
        spec_dict = self._specs[idx]
        # `ew_cov` accumulates over the features alone -- no target, and so no
        # constant column to regress one on.
        unsupervised = spec_dict["model"]["type"] in _NO_TARGET_GRAM
        columns = list(spec_dict["features"])
        if spec_dict.get("fit_intercept", True) and not unsupervised:
            columns = [_INTERCEPT, *columns]
        names = [] if unsupervised else [target_name(t) for t in spec_dict["targets"]]
        out: list[Gram] = []
        native = self._native.gram(
            idx, _group_keys(group, "ModelBank.gram"), float32, packed, columns
        )
        for row, lag, (tidx, by_target, centred) in native:
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
                    "comoments": np.frombuffer(como, dtype=as_dtype)
                    if packed
                    else np.frombuffer(como, dtype=as_dtype).reshape(k, k),
                    "cross_moments": np.asarray(cross).reshape(len(cross), k)
                    if cross
                    else np.zeros((0, k)),
                    "means_by_target": np.asarray(by_target, dtype=float).reshape(len(tidx), k)
                    if tidx
                    else np.zeros((0, k)),
                    "cross_centered": np.asarray(centred, dtype=float).reshape(len(tidx), k)
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
                    else np.frombuffer(lag[1], dtype=as_dtype).reshape(len(lag[0]), k, k),
                }
            )
        return out

    def marginal(
        self, spec: str | int, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Return the feature and target pairs a ``marginal`` spec keeps, one row per pair.

        Use it to rank features by how each one alone relates to a target, from
        the bank or from one loaded from a state file. For a spec of another
        model use :meth:`gram`, which gives an ``ew_cov``'s moments, and for
        what an ``audit`` counted, :meth:`audit`.

        .. rubric:: Parameters

        ``spec``
            The ``marginal`` spec to read, by name or by position.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives an empty frame.

        .. rubric:: Returns

        A ``DataFrame`` with one row per (group, instance, feature, target),
        an instance being the part of the model that fits one half-life:
        groups sorted, targets in the spec's order, and features in the spec's
        order within each. Its moments are exponentially weighted (EW). The
        columns:

        ``spec``, ``group``, ``instance``
            The spec's name, and the group and instance as :meth:`gram` reports
            them: ``""`` for a spec without a ``group`` column, and ``""`` for a
            single instance, else the field suffix, such as ``"@h500"``.
        ``half_life``, ``lam``
            The instance's decay, as :meth:`coef` gives it.
        ``feature``, ``target``
            The pair's feature column, and the target's name: its column,
            unless :func:`polars_online.target` gave it another name.
        ``weight_sum``
            The target's accumulated weight ``W_t``: the rows where the target
            was present, weighted and decayed. It differs from the output's
            ``weight_sum`` when targets have different null patterns.
        ``n_kish``
            ``W_t^2 / Q_t``, with ``Q_t`` the accumulated squared weight:
            Kish's effective sample size, the number of equally weighted rows
            that carry the same information. Null before the target's first
            row, and under a hard window where subtracting the rows that leave
            keeps no digit of ``Q_t`` (at most ``64 eps`` times the history's).
        ``mean_x``, ``var_x``, ``mean_y``, ``var_y``, ``cov``
            The pair's EW moments, in population form over the decayed
            weights, so a variance is never negative.
        ``corr``, ``beta``, ``t_stat``
            ``cov / sqrt(var_x var_y)``; ``cov / var_x``, the slope of the
            target on that feature alone; and
            ``corr * sqrt((n_kish - 2) / (1 - corr^2))``, the t-statistic of
            that correlation at Kish's size.
            Read ``t_stat`` as a scale to compare pairs by, not as a p-value:
            the rows are neither independent nor Gaussian. Each is null until
            ``weight_sum`` reaches the target's ``min_weight``, null while the
            instance's ``settled_frac`` is below ``min_settled_frac``, and null
            where it is undefined (a constant column, ``n_kish <= 2``). Both gates
            withhold ``t_stat_serial``, the ``lag_corr_*`` columns, ``phi_x``,
            ``phi_y`` and the split columns too. ``n_serial``, a count as
            ``n_kish`` is, stays.

        With ``lags``, four more list columns and four numbers:

        ``lag_corr_xx``, ``lag_corr_yy``
            Each series' own autocorrelation at the spec's lags (see Notes on
            the first rows).
        ``lag_corr_xy``, ``lag_corr_yx``
            The feature now against the target ``l`` rows back, and the target
            now against the feature ``l`` rows back, at each of ``cross_lags``
            (every lag by default). Absent under ``cross_lags=[]``. Notes say
            how to read them for a lead or a lag.
        ``n_serial``, ``t_stat_serial``
            ``n_kish`` divided by Bartlett's serial-dependence factor, and the
            statistic against it. Null without ``serial_rule``.
        ``phi_x``, ``phi_y``
            The per-row decays fitted under ``serial_rule = "geometric"``. Null
            otherwise, and null when fewer than two kept lags are positive.
            They are fitted from the lag correlations, so both gates withhold
            them where they withhold those.

        With ``bins`` or ``bin_edges``, the nonlinear view. These columns are
        present whenever the spec asked for bins, and hold empty lists and
        nulls until the edges are fixed:

        ``bin_edges``
            The feature's interior edges, fixed once and never moved. Ragged: a
            feature keeps only the bins it can support, so a binary feature has
            two bins and a constant one has a single bin.
        ``bin_n``, ``bin_mean_y``, ``bin_var_y``
            The target's weight, mean and variance inside each bin: the
            feature's response curve. Each has one more entry than
            ``bin_edges``, since the outer two bins are open. A bin no row has
            landed in has ``bin_n = 0`` and null for the two moments. Under
            ``bin_stat="huber"`` the mean is Huber's M-location of the bin's
            rows, and the variance the mean square of their deviations from
            it, clipped at the cut. The weight is the rows' either way
            (:func:`polars_online.spec.marginal`).
        ``split_gain``, ``split_at``
            The fraction of the target's variance removed by the best single
            cut of the feature, and where that cut falls. It compares directly
            with ``corr ** 2``, so ``split_gain - corr ** 2`` is the nonlinear
            surplus.
        ``split_gain_t``
            The ``t_stat`` a ``corr`` would need to match that gain, against
            ``n_serial`` where there is one. It is optimistic, since the cut
            was chosen as the best of the candidates: a ranking, not a p-value.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.
        ``ValueError``
            ``spec`` is not a ``marginal`` spec.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            pairs = po.spec.marginal(
                "pairs", targets=["y"], features=["x0", "x1", "x2"], half_life=500.0
            )
            bank = po.ModelBank([pairs])
            bank.fit(df)
            table = bank.marginal("pairs")
            table = table.sort(pl.col("t_stat").abs(), descending=True)   # strongest first
            for feature, corr, beta in table.select("feature", "corr", "beta").iter_rows():
                print(f"{feature}: corr {corr:.2f}, beta {beta:.2f}")

        .. code-block:: text

            x0: corr 0.86, beta 1.89
            x1: corr -0.37, beta -0.81
            x2: corr 0.04, beta 0.09

        .. rubric:: See also

        - :func:`polars_online.spec.marginal`, the spec, its lags and its
          bins.
        - :meth:`closed_groups`, the same columns for each group that has
          closed, as ``pair_<column>``.
        - :meth:`gram`, the moments of an ``ewridge``, ``lasso`` or
          ``ew_cov``.

        .. rubric:: Notes

        The pairs are read from the state, so a bank loaded from a file
        reports them as the bank that saved it would. Feeding the rows in one
        chunk or a thousand gives the same numbers to the bit.

        As in ``ew_cov``, a lagged moment starts at 0 and takes its first
        product when a learned row has one ``l`` rows back. So each of the
        four lag columns reads 0.0, not null, at a lag ``l`` while a group has
        learned no more than ``l`` rows and ``weight_sum`` has reached
        ``min_weight``. Below ``min_weight`` each is null, as ``corr`` is.

        For two series that describe the same moment, a feature whose
        ``lag_corr_yx[0]`` exceeds its ``corr`` leads the target, and one
        whose ``lag_corr_xy[0]`` does follows it. That reading does not hold
        against a forward-looking target, one built from the rows after its
        own. There the target ``l`` rows back is built partly from the
        feature's newest ``l`` rows, so a feature built from the same rows'
        information shows ``lag_corr_xy`` above ``corr`` however it is sampled.
        """
        return self._native.marginal(
            self._spec_index(spec, "ModelBank.marginal"), _group_keys(group, "ModelBank.marginal")
        )

    def audit(
        self,
        spec: str | int | None = None,
        group: str | Iterable[str | None] | None = None,
        *,
        table: Literal["columns", "pairs", "clock"] = "columns",
        pooled: bool = False,
    ) -> pl.DataFrame:
        """Return what an ``audit`` spec counted, one row per column, pair of columns or clock.

        Use it to read the counts an ``audit`` keeps about the data itself:
        sentinel values, frozen feeds, few distinct values, heavy tails,
        duplicate columns and the clock's spacing. For the findings drawn from
        them use :meth:`check`, and for the statistics every spec keeps,
        :meth:`describe`.

        .. rubric:: Parameters

        ``spec``
            The ``audit`` spec to read, by name or by position
            (:func:`polars_online.spec.audit`). The default, ``None``, reads the
            bank's one ``audit`` spec.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.
        ``table``
            Which table to return: ``"columns"``, the default, one row per
            (group, column); ``"pairs"``, one row per (group, pair of columns),
            for a spec built with ``pairs=True``; or ``"clock"``, one row per
            group.
        ``pooled``
            ``True`` merges the groups the frame covers into one audit, as if
            they were one stream, and drops the ``group`` column. The counts
            merge exactly and the moments to rounding, and the counters past the cap
            and the digests approximately, as
            :func:`polars_online.spec.audit` says. The default, ``False``, gives
            a row per group.

        .. rubric:: Returns

        A ``DataFrame`` led by ``spec``, as every reader of the bank's is, with
        every count an ``Int64``. ``table="columns"`` has ``spec``, ``group``
        and ``column``, then:

        ``rows_fed``, ``null_count``, ``nan_count``
            The rows read, then two kinds of value a model cannot learn from:
            the nulls, as Polars' ``null_count`` counts them, and the NaNs.
        ``pos_inf_count``, ``neg_inf_count``, ``beyond_bound_count``, ``count``
            The other kinds, each counted apart (``beyond_bound_count`` is
            finite but past ``1e100`` in magnitude), and ``count``, the usable
            values. The six counts add up to ``rows_fed``.
        ``mean``, ``std``, ``skew``, ``kurtosis``, ``min``, ``max``
            Over the usable values: ``std`` with ``ddof = 1``, and ``skew`` and
            the excess ``kurtosis`` as scipy's biased estimators. Null where
            undefined: no usable value, one, or no spread.
        ``median``, ``mad``, ``robust_z``
            The median and the median absolute deviation, exact while
            ``n_unique`` is not null and from a t-digest past the cap (about 1%
            of the rows in rank), and the largest robust z,
            ``max(max - median, median - min) / (1.4826 mad)``.
        ``n_unique``
            The distinct usable values, as Polars' ``n_unique`` counts them,
            with ``-0`` read as ``0``. Null past the spec's ``distinct_cap``.
        ``top_value``, ``top_count``, ``second_count``, ``count_error``
            The most repeated value, its count and the next value's count.
            They are exact while ``n_unique`` is, and past the cap each is at
            most ``count_error`` short of the truth, where a value not held has
            a count of at most ``count_error``.
        ``longest_run``, ``equal_prev``, ``adjacent``, ``equal_by_chance``
            ``longest_run`` is the longest run of one value on consecutive
            rows. ``equal_prev`` counts the rows equal to the row before, out
            of ``adjacent``, the rows whose row before was usable too.
            ``equal_by_chance`` is ``sum p_v^2`` over the values' shares
            ``p_v``: the share of ``adjacent`` rows that independent rows would
            make equal, a lower bound past the cap.
        ``min_change``, ``min_change_count``
            The smallest nonzero change ``|x_t - x_(t-1)|`` between two
            consecutive usable rows (null before there is one), and how many
            changes sit at it. Two changes count as one size when they differ
            by no more than the rounding of one step of a grid, ``8 eps`` of
            the largest magnitude seen. :meth:`check`'s ``frozen`` weighs
            ``min_change_count`` against ``equal_prev``, the changes at 0.
        ``autocorr``, ``unit_root_t``
            ``autocorr`` is the correlation of each usable row with the usable
            row before it. ``unit_root_t`` is the Dickey-Fuller statistic of
            that regression with a constant. It reads:

            - above about -3 for a random walk;
            - about ``-sqrt(n)`` for rows that do not persist;
            - ``+inf`` for a fit with no residual and a slope of 1, such as a
              row counter;
            - ``-inf`` for a fit with no residual and a slope below 1.

        ``table="pairs"`` has ``spec``, ``group``, ``column_a``, ``column_b``,
        and, over the rows where both are usable, ``count``, ``corr`` and
        ``equal``, the rows on which the two are equal.

        ``table="clock"`` has no rows for a spec without a ``clock``. Its
        columns:

        ``spec``, ``group``
            As :meth:`groups` reports them.
        ``steps``
            The number of steps between consecutive rows.
        ``duplicates``
            How many steps are 0.
        ``gaps``
            How many steps are at or past ``gap_cap`` (a clock requires
            ``gap_cap``).
        ``regular``
            How many steps are neither.
        ``step_mean``, ``step_std``, ``step_cv``
            The regular steps' mean, population standard deviation and
            coefficient of variation.
        ``step_max``
            The largest step of all.

        On a temporal clock ``step_mean``, ``step_std`` and ``step_max`` are a
        ``Duration`` in the clock column's unit (milliseconds for a ``Date``),
        on a number clock numbers in its units, and a ``Float64`` before a
        chunk has said which.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.
        ``ValueError``
            When:

            - ``spec`` is ``None`` and the bank holds no ``audit`` spec, or
              several;
            - ``spec`` is not an ``audit`` spec;
            - ``table="pairs"`` reads a spec built without ``pairs=True``;
            - ``table`` is not ``"columns"``, ``"pairs"`` or ``"clock"``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            audit = po.spec.audit(
                "audit", columns=["x0", "x1"], clock="t", gap_cap=300.0, pairs=True
            )
            bank = po.ModelBank([audit])
            holes = pl.when(pl.col("t") < 10).then(None).otherwise(pl.col("x1"))
            bank.fit(df.with_columns(x1=holes))                 # x1 null on the first 10 rows
            columns = bank.audit()                              # one row per (group, column)
            pairs = bank.audit(table="pairs")                   # one row per (group, pair)
            clock = bank.audit(table="clock")                   # one row per group
            print(columns.select("column", "rows_fed", "null_count", "count").rows())
            print(pairs.select("column_a", "column_b", "count").rows())
            print(clock.select("steps", "regular", "gaps").rows())

        .. code-block:: text

            [('x0', 400, 0, 400), ('x1', 400, 10, 390)]
            [('x0', 'x1', 390)]
            [(399, 399, 0)]

        ``bank.audit(pooled=True)`` gives every group as one stream, with no
        ``group`` column.

        .. rubric:: See also

        - :func:`polars_online.spec.audit`, the spec and what it keeps.
        - :meth:`check`, the findings read from these tables.
        - :meth:`describe`, the statistics every spec keeps.

        .. rubric:: Notes

        The counts are read from the state, so a bank loaded from its file
        reports what the bank that saved it would, and the stream's chunking
        moves none of them.

        A clock step is the time that passed between the two rows, uncapped:
        a model ages by at most ``gap_cap`` of it, and the audit counts all of
        it (across a session change under a numeric ``session_gap``, that
        gap). A step back is refused, as for every spec. With
        ``restart_after_step_back`` each one resets the audit's steps
        instead (``0`` resets at every one), and :meth:`summary`'s
        ``clock_backwards`` and :meth:`check`'s ``step_back`` count them.
        """
        if spec is None:
            audits = [i for i, s in enumerate(self._specs) if s["model"]["type"] == "audit"]
            if len(audits) != 1:
                names = [self._specs[i]["name"] for i in audits]
                msg = (
                    f"ModelBank.audit: the bank has {len(audits)} audit specs, {names}. Name the "
                    "one to read with spec=."
                    if audits
                    else "ModelBank.audit: the bank has no audit spec. Add po.spec.audit(...) to "
                    "its specs."
                )
                raise ValueError(msg)
            idx = audits[0]
        else:
            idx = self._spec_index(spec, "ModelBank.audit")
        return self._native.audit(idx, _group_keys(group, "ModelBank.audit"), table, pooled)

    def closed_groups(self, spec: str | int | None = None, *, drop: bool = True) -> pl.DataFrame:
        """Return the groups that have closed and not yet been read, oldest first, as one frame.

        Use it with a spec that sets ``group_close``, to read each group's sums
        and fit once no further row can join it, and to keep a bank over an
        unbounded number of keys bounded. To have each chunk's closed groups
        read for you, pass ``closed_groups=`` to :meth:`fit_predict_batches`.
        For the sums of a group still open, use :meth:`gram`.

        .. rubric:: Parameters

        ``spec``
            The spec whose closed groups to read, by name or by position. The
            default, ``None``, reads every spec's.
        ``drop``
            ``True``, the default, removes what it returns from the queue of
            closed groups not yet read, as a
            loop that reads after each chunk wants. ``False`` reads the queue and
            leaves it as it is.

        .. rubric:: Returns

        A ``DataFrame`` with one row per (group, instance), an instance being
        the part of the model that fits one half-life, and one per Gram
        where a spec reads several: under ``target_gaps = "own_rows"`` targets
        missing on different rows are fitted from Grams of their own. The
        columns every row has:

        ``spec``, ``group``, ``instance``, ``half_life``, ``lam``, ``session``
            Which group closed, and its instance with the instance's decay, as
            :meth:`coef` gives it. ``session`` is the value of the session that
            ended under ``group_close = "session"``, and null under
            ``"monotone"``.
        ``weight_sum``, ``n_kish``
            As :meth:`gram` reports them, at the moment the group closed.
        ``rows_fed``, ``rows_learned``, ``clock_min``, ``clock_max``
            The span's own :meth:`summary` counts, ``Int64`` as there, and its
            clock range, in the clock column's own dtype as there.
        ``coef``
            On a Gram's row, the coefficients of that Gram's ``targets``, and
            on a kind that reports coefficients, its coefficients. Null on
            the rows of a kind that reports none.

        Then, for each kind of model among the specs that close groups, a block
        of columns, null on the rows of the other kinds:

        .. list-table::
           :header-rows: 1
           :widths: 24 36 40

           * - kind
             - columns
             - what they hold
           * - a kind that keeps a Gram
             - ``columns``, ``means``, ``comoments``, ``targets``,
               ``target_means``, ``target_vars``, ``target_weights``,
               ``target_n_kish``, ``cross_moments``, ``means_by_target`` and
               ``cross_centered``
             - :meth:`gram`'s arrays, packed: ``comoments`` is the upper
               triangle with the diagonal, row by row (``k(k+1)/2`` numbers),
               and the per-target arrays are row-major ``(n_targets, k)``.
               :func:`polars_online.gram.from_row` expands them, and hands
               back exactly what :meth:`gram` would have returned for that
               group, so the exact solve on a closed group is
               ``po.gram.solve(po.gram.from_row(row))``
           * - an ``ew_cov`` with ``lags``
             - ``lags`` and ``lag_comoments``
             - :meth:`gram`'s, the co-moments at each lag flattened lag by
               lag, row-major: ``L*k*k`` numbers for ``L`` lags
           * - an ``ew_cov`` with ``pca``
             - ``eig_vals`` and ``eig_vecs``
             - the vectors signed to continue those of the previous closed row
               of the same (spec, instance): the previous group under
               ``"monotone"``, and the same group's previous close under
               ``"session"``, never the previous chunk. So a sign flip between
               two rows is a real rotation, not an eigensolver's arbitrary
               choice
           * - an ``rcov``
             - ``rcov``, ``rcorr``, ``rcov_n``, ``rcov_kind``,
               ``rcov_bandwidth_used``, ``rcov_omega2``, ``rcov_iv_sparse``,
               ``rcov_iq`` and ``rcov_psd_repaired``
             - :func:`polars_online.spec.rcov` explains each
           * - a ``marginal``
             - ``pair_*``
             - :meth:`marginal`'s frame turned on its side: ``pair_feature``
               and ``pair_target`` name the pairs, and every other
               ``marginal`` column becomes ``pair_<column>``, with one entry
               per pair in the same order

        A ``marginal`` column that is a list per pair there (the
        ``lag_corr_*`` columns with ``lags``, and ``bin_edges``, ``bin_n``,
        ``bin_mean_y`` and ``bin_var_y`` with ``bins``) is a list of lists
        here, present when any closing ``marginal`` asked for it. A nested
        list has no CSV form, so write a closed frame with them to parquet, or
        keep it as a frame.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``ValueError``
            ``spec`` is ``None`` and the specs that close groups read clocks of
            different dtypes, which one clock column cannot hold. The message
            names them, and nothing is removed from the queue. Read them one
            spec at a time, as for :meth:`groups`.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            blocks = po.spec.ew_cov(
                "cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone"
            )
            by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # 4 blocks of 100 rows
            bank = po.ModelBank([blocks])
            bank.fit(by_block)
            closed = bank.closed_groups()               # blocks 0 to 2: block 3 is still open
            print(closed.select("group", "rows_fed").rows())
            g = po.gram.from_row(closed.head(1))        # block 0's sums, as gram() gives them
            print(g["comoments"].shape)

        .. code-block:: text

            [('0', 100), ('1', 100), ('2', 100)]
            (2, 2)

        .. rubric:: See also

        - :meth:`fit_predict_batches`, whose ``closed_groups=`` reads the
          queue after each chunk.
        - :func:`polars_online.gram.from_row`, a closed row's sums as
          :meth:`gram` gives them.
        - `Closed groups in the README
          <https://github.com/hgilde/polars-online#one-row-per-finished-group>`__,
          a worked example.

        .. rubric:: Notes

        A spec with ``group_close`` gives a group's sums at the moment the
        bank can prove no further row will join it, and then drops its
        model. Under ``"monotone"`` that is at the end of the chunk in which a
        larger key arrives, and
        under ``"session"`` when the group's session ends. Without
        ``group_close``, every key ever seen stays in memory. A closed group's
        model is freed when the group closes, whether or not the queue is
        read. Its rows (see Returns) stay in the queue until
        they are read, so read it between chunks. What is unread is saved with
        the state, so a loop that saves between chunks loses no row.
        :meth:`predict` never closes a group.
        """
        idx = None if spec is None else self._spec_index(spec, "ModelBank.closed_groups")
        return self._native.closed_groups(idx, drop)

    def check(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Return findings about the data each model was fed, one row per problem, the worst first.

        Run it after a first pass over the data, before trusting a fit: it
        names features null on many rows, constant or collinear features, leakage, a clock
        that steps back, models that never warmed up, and more. To find what
        no model keeps, such as sentinel values or a frozen feed, put an
        :func:`polars_online.spec.audit` in the bank, and read its counts with
        :meth:`audit`.

        .. rubric:: Parameters

        ``spec``
            The spec to check, by name or by position. The default, ``None``,
            checks every spec.
        ``group``
            The group to check: one group key, as :meth:`groups` reports it, or
            a list of keys, with ``None`` in the list for the null group. The
            default, ``None``, checks every group. A key the bank has never
            seen gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with one row per finding, in severity order, then spec
        order, then :meth:`groups`' order. A bank that has seen no rows gives
        an empty frame. The columns:

        ``severity``
            ``error`` (the model cannot learn), ``warning`` (the fit is
            unreliable) or ``info``: an ``Enum`` in that order, worst first.
        ``code``
            Which check found it: a short name that stays the same across
            releases, one of the codes listed under Returns. It is a ``String``, not an
            ``Enum`` as ``severity`` is. A minor release may add a check, and
            an ``Enum``'s categories are part of its dtype, so a new code would
            change the column's type under code that pins it.
        ``spec``, ``group``
            The spec and group, as :meth:`summary` names them. ``group`` is null for
            ``not_checked``, which is about the spec as a whole.
        ``column``
            The input column the finding is about, where there is one.
        ``value``, ``threshold``
            The measured number, and the line it crossed. Null for
            ``withheld`` and ``not_checked``.
        ``message``
            What is wrong and, for most codes, what to do. It ends with
            a link to the code's section of `docs/DATA-QUALITY.md
            <https://github.com/hgilde/polars-online/blob/main/docs/DATA-QUALITY.md>`_,
            which shows what the problem does to each model and how to fix it
            with Polars, before the bank, in a program that runs.

        The codes every spec can find:

        .. list-table::
           :header-rows: 1
           :widths: 18 10 72

           * - code
             - severity
             - fires when
           * - ``nothing_learned``
             - error
             - a model was fed rows and learned from none; the column null
               most often is named
           * - ``missing``
             - warning
             - a feature or the weight column is null, NaN, infinite or past
               1e100 on 10% of the rows or more, each of them skipped; an
               error at 100%, and information for a target missing on half the
               rows or more, since sparse targets are a design, not a fault
           * - ``few_learned``
             - warning
             - more than half the rows were skipped for a missing feature or
               weight, whichever columns caused it
           * - ``constant``
             - error, warning
             - a target (error) or a feature (warning) took one value on
               every row
           * - ``level_over_spread``
             - warning, info
             - a feature's ``|mean| / std`` passes the limit of a model that
               does not center it (Notes); information past 1e6 for the models
               that do
           * - ``scales_apart``
             - warning
             - the widest feature's spread over the narrowest's passes the
               limit of a model that takes one step for every column (Notes)
           * - ``ridge_shrinks``
             - warning
             - ``ewridge``, ``huber`` or ``quantile`` without
               ``standardize``, at ``ridge_scale="mean"`` (the default): the
               smallest ridge is a quarter of a feature's variance or more,
               so it takes a fifth of that coefficient
           * - ``collinear``
             - warning
             - the largest variance inflation factor, or the condition index
               of the design the model solves, over the features passes 30,
               from the Gram of an ``ewridge`` or ``lasso`` (information for
               an ``ew_cov``). A model with an intercept centers, so its index
               is the features' correlation matrix's, ``sqrt(d_max / d_min)``;
               one fitted through the origin reads Belsley's uncentered index
               (:func:`polars_online.gram.condition`)
           * - ``leakage``
             - warning
             - a feature correlates with a target at 0.9999 or more, from
               the Gram or a ``marginal``: it contains the target, both are
               one random walk, or the relation is nearly exact (a
               correlation of 0.9999 is an R² of 0.9998 for that feature
               alone)
           * - ``step_back``
             - warning
             - the clock stepped back, a reset or a late row under
               ``restart_after_step_back``
           * - ``resets``
             - info
             - a model reset at a ``session_gap = "reset"`` or a step back
           * - ``few_rows``
             - warning
             - a regression learned from fewer rows than it has
               coefficients
           * - ``group_sizes``
             - info
             - the largest group learned from 100 times the rows of the
               smallest or more
           * - ``never_settled``
             - warning
             - a model has seen less than one half-life of clock
               (``settled_frac`` below 0.5)
           * - ``below_min_weight``
             - warning
             - the weight a model's ``weight_sum`` tops out at
               (``weight_sum_settled``) is
               below the spec's ``min_weight``, so its predictions stay null
           * - ``withheld``
             - warning
             - a model's last row had its predictions withheld, with the
               reason
           * - ``low_support``
             - warning
             - the smallest coefficient's data share is below 0.5, the
               command line's readiness line; not repeated for a column
               already found constant, collinear or shrunk by the ridge
           * - ``solve_failures``
             - warning, info
             - a model's :meth:`solve_failures` count is above 0
               (warning); for ``lasso``, more than 10 (information)
           * - ``not_checked``
             - info
             - numpy is not installed, so ``collinear`` and ``leakage`` were
               not checked on a spec with a Gram and a group to read it from;
               said only of a spec with a group to check

        An ``audit`` knows no roles, so its column's ``missing`` share is
        information (an error at every row), and its ``constant`` a warning.
        ``step_back`` and ``resets`` are read from its :meth:`summary` as for
        any spec. Its own codes, read from :meth:`audit`:

        .. list-table::
           :header-rows: 1
           :widths: 18 10 72

           * - code
             - severity
             - fires when
           * - ``sentinel``
             - warning, info
             - a column holds ``+inf``, ``-inf`` or a value past ``1e100``
               (warning); holds NaN, not null (information); or one value is
               1% of its usable rows or more, at least 5 times as often as any
               other can be (warning), unless the column has 10 values or
               fewer or is frozen
           * - ``frozen``
             - warning
             - the rows equal to the row before, less those the column's
               value frequencies give by chance, outnumber the changes at its
               smallest step by five standard errors,
               ``(repeats - chance - at_step) / sqrt(repeats + at_step)``,
               where that step is less than half the column's moves: an
               atom of changes at zero, which a stale feed adds and a column
               on a coarse grid (an hour, a regime, a price in ticks) does
               not show
           * - ``few_values``
             - info
             - 10 distinct values or fewer over 100 usable rows or more
           * - ``random_walk``
             - warning
             - the Dickey-Fuller statistic is above -3.5 (the 1% critical
               value is -3.43) over 100 pairs of rows or more
           * - ``heavy_tails``
             - info
             - an excess kurtosis of 5 or more, or a value 10 robust standard
               deviations or more from the median; ``value`` is the robust z
               where it fired, the kurtosis otherwise, each beside its own
               threshold
           * - ``duplicate``
             - warning
             - with ``pairs=True``, two columns correlate at 0.999 or more
           * - ``duplicate_stamps``
             - info
             - a row shares the clock of the row before
           * - ``gaps``
             - info
             - a step reaches ``gap_cap``
           * - ``irregular_clock``
             - info
             - the regular steps' coefficient of variation is 0.5 or more

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge(
                "ridge", targets=["y"], features=["x0", "x1", "x2"], half_life=100.0
            )
            bank = po.ModelBank([spec])
            bank.fit(df.with_columns(x2=pl.lit(1.0)))           # x2 made constant
            problems = bank.check()                             # the worst first
            print(problems.select("severity", "code", "column").rows())

        .. code-block:: text

            [('warning', 'constant', 'x2')]

        To keep only what needs action, drop the information:
        ``problems.filter(pl.col("severity") != "info")``.

        .. rubric:: See also

        - `Data quality
          <https://github.com/hgilde/polars-online/blob/main/docs/DATA-QUALITY.md>`__,
          each problem, what it does to each model, and its fix.
        - :meth:`summary`, :meth:`describe` and :meth:`audit`, the counts the
          findings are read from.
        - :func:`polars_online.spec.audit`, the spec that keeps what no model
          keeps.

        .. rubric:: Notes

        ``check`` reads only what the bank keeps anyway: :meth:`summary`,
        :meth:`describe`, :meth:`solve_failures`, :meth:`last_row`, and
        :meth:`gram`, :meth:`marginal` or :meth:`audit` where the spec has
        one. So it adds no work to the run, a bank loaded from its file gives
        the same findings as the bank that saved it, and the input's chunking
        moves none of them.

        **Every threshold was measured before it shipped**, for how often it
        finds a planted problem and how often it fires on clean data. `How
        the checks work
        <https://github.com/hgilde/polars-online/blob/main/docs/DATA-QUALITY.md#how-the-checks-work>`__
        gives the clean streams, the planted problems and the numbers each
        line was set from.

        ``level_over_spread`` and ``scales_apart`` read a limit for each
        model: the largest value at which its out-of-sample R² fell by no more
        than 0.05, at half-lives of 20, 200 and infinite (for ``rls``'s level
        limit, at half-lives of 200, 2,000 and infinite, over 2,000 and
        20,000 rows).

        .. list-table::
           :header-rows: 1

           * - model
             - level limit
             - spread-ratio limit
           * - ``ftrl``
             - 0.5
             - 1.5
           * - ``kalman``, ``standardize=False``
             - 2
             - 2
           * - ``sgd``, ``standardize=False``
             - 3
             - 3
           * - ``pa``, ``standardize=False``
             - 3
             - 2
           * - ``rls``
             - ``0.9 sqrt(W / delta_left)``
             - 30

        ``rls``'s level limit is read from the summary. A level lowers its
        accuracy only through the part of its prior ``delta I`` it still
        holds, ``delta_left = delta (1 - settled_frac)`` (all of ``delta``
        without decay), set against the weight its fit holds, ``W``
        (``settled_frac * weight_sum_settled``, or ``weight_sum`` without
        decay). So a decaying ``rls`` that has forgotten its prior is told
        nothing, at any level. The models that center their features
        (``ewridge``, ``lasso``, ``huber``, ``quantile``, and ``kalman``,
        ``sgd`` and ``pa`` at their default ``standardize=True``) lost nothing
        at a level of 1e14 times the spread. Nor did they at spreads 1e8
        apart, except where a narrow feature met an unstandardized ridge,
        which ``ridge_shrinks`` finds.

        A level or a spread ratio counts only when it is five standard errors
        from what a centered column, or two of equal spread, would show by
        chance in that many rows. The Gram's checks wait for ten Kish rows a
        coefficient. Both guards keep a small group quiet.

        **What it cannot see.** :meth:`describe` keeps every row's moments
        undecayed, so a feature that drifted slowly can sit far above its
        recent spread without showing as a level. The Gram's checks cover the
        three models whose Gram the bank exports (:meth:`gram`), and
        ``leakage`` also reads a ``marginal``. An ``audit`` measures what no
        model keeps: sentinel values counted apart from nulls, a frozen feed, a
        column of few distinct values, a level fed as a feature, heavy tails,
        duplicate columns, and the clock's duplicate stamps, gaps and irregular
        spacing. Without one in the bank, nothing is said about them. A group a
        spec closes (``group_close``) leaves the bank with its model, so
        nothing is said about it either: the findings cover the groups still
        open. An ``audit`` refuses ``group_close`` for that reason.
        """
        from polars_online._check import check

        return check(self, spec, group)

    def solve_failures(
        self, spec: str | int | None = None, group: str | Iterable[str | None] | None = None
    ) -> pl.DataFrame:
        """Return how many of each model's solves needed help, one row per spec and group.

        Use it to find the models whose inputs are degenerate (constant or
        collinear features, or far too few rows for the number of features),
        or whose refit or seeding stopped at its pass cap. A
        count above 0 means a solve needed help, not that anything crashed.
        For every count of a model at once, use :meth:`summary`, which
        carries this one as its ``solve_failures`` column.

        .. rubric:: Parameters

        ``spec``
            The spec to read, by name or by position. The default, ``None``,
            reads every spec.
        ``group``
            The group to read: one group key, as :meth:`groups` reports it, or a
            list of keys, with ``None`` in the list for the null group. The
            default, ``None``, reads every group. A key the bank has never seen
            gives no rows.

        .. rubric:: Returns

        A ``DataFrame`` with ``spec``, ``group`` (as :meth:`groups` reports it)
        and ``solve_failures``, an ``Int64``, in spec order, then in
        :meth:`groups`' order. What each model counts:

        ``ewridge``, ``huber``, ``quantile``
            A solve that needed jitter, or failed at every jitter and kept the
            previous fit, the standardized solve included. ``huber`` and
            ``quantile`` also count each target whose refit of its first rows
            took all 200 of its passes, where the fit of the last pass stands.
        ``sgd(loss="huber")``
            Each target whose refit of its first rows took all 200 of its
            passes.
        ``marginal``
            Each target whose bins' refit of their first rows took all 200 of
            its passes, under ``bin_refit_rows``.
        ``lasso``
            A coordinate descent that ran out of ``max_iter`` sweeps before
            every coefficient moved less than ``tol``, one per target and path
            point.
        ``kmeans``
            Each run of a ``"lloyd"`` start that was still moving rows after
            300 passes, and stopped there.
        ``ew_class``
            A row on which a class covariance could not be factorized.
        ``hmm``
            A row whose state densities could not be evaluated, where the
            filter is left where it stands, and, as for ``kmeans``, each run
            of a ``"lloyd"`` start of its seeding that was still moving rows
            after 300 passes.
        ``bocpd``
            A row whose predictive density could not be evaluated. It reports
            nulls and leaves the run-length posterior where it stands, and the
            count makes a run of them visible.

        Every other model reports 0 because it counts nothing, not because
        nothing can fail: ``rls`` keeps a Cholesky factor and ``kalman`` a
        covariance, and the gradient models keep no second moment to
        factorize.

        .. rubric:: Raises

        ``KeyError``
            ``spec`` is a name the bank does not have.
        ``IndexError``
            ``spec`` is a position the bank does not have.
        ``TypeError``
            ``group`` is not a str, a list of them with ``None`` allowed, or
            ``None``.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(df)
            print(bank.solve_failures().rows())                 # no solve needed help

        .. code-block:: text

            [('ridge', '', 0)]

        To find the models that need a look, keep the rows above 0:
        ``bank.solve_failures().filter(pl.col("solve_failures") > 0)``.

        .. rubric:: See also

        - :meth:`summary`, which carries this count beside the rest.
        - :meth:`check`, whose ``solve_failures`` finding reads it.
        - :meth:`coef`, the fits the solves gave.

        .. rubric:: Notes

        A solve never returns NaN silently. A near-singular system is tried
        again with a larger diagonal jitter at each try, and a solve that
        fails at every jitter keeps the previous coefficients. Both are
        counted here.

        Past about 20% of outliers on one side, a Huber refit converges
        slowly. Over 3,000 simulated targets of 50 to 500 rows, with up to 45%
        of the rows on one side moved by a level, 50 passes ran out on 149 of
        them, and 200 passes on 9, all nine past 39%.
        """
        names = self._native.spec_names()
        picked = (
            range(len(names))
            if spec is None
            else [self._spec_index(spec, "ModelBank.solve_failures")]
        )
        wanted = _group_keys(group, "ModelBank.solve_failures")
        native = self._native.solve_failures()
        rows = [
            (names[i], key, count)
            for i in picked
            for key, count in native[i]
            if wanted is None or key in wanted
        ]
        return pl.DataFrame(
            rows,
            schema={"spec": pl.String, "group": pl.String, "solve_failures": pl.Int64},
            orient="row",
        )

    @property
    def output_fields(self) -> dict[str, list[str]]:
        """Each spec's name, mapped to the names of its output column's fields, in order.

        For each spec, the list is :func:`polars_online.spec.output_fields`
        of that spec. The spec fixes its fields, so this is the output's
        layout before any row is fed. It is a property, as :attr:`specs` is.
        Calling it, as ``bank.output_fields()``, is deprecated: it still
        returns the mapping, with a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`, until the
        next major version refuses it.
        """
        return _CalledDict(
            zip(self._native.spec_names(), self._native.output_fields(), strict=True)
        )

    @deprecated_keywords("ModelBank.save", {"path": "file"})
    def save(self, file: str | os.PathLike[str] | IO[bytes]) -> None:
        """Write the bank's state to a file that :meth:`load` reads on any supported OS.

        Use it to keep what a bank has learned, to resume it later or to
        serve from it. For the bytes without a file, use :meth:`save_bytes`,
        and to read a state outside this library, use :meth:`write_json`.

        .. warning::
            A spec's formula target (an expression in ``targets`` holding a
            window operator of :mod:`polars_online.ops` that looks ahead) is
            written into the file in a form that is
            **unstable**: it may change in any release without that counting
            as a breaking change. The rest of the file's form is kept from
            release to release.

        .. rubric:: Parameters

        ``file``
            A path, or a file object opened for binary writing, such as
            ``open(..., "wb")`` or an ``io.BytesIO``, as Polars' ``write_*``
            methods take either. A path is replaced safely: an interrupted
            save leaves the file that was there before (see Notes). A file
            object is handed the bytes in one ``write``, and whether that
            write can be interrupted is up to whoever opened it.

        .. rubric:: Raises

        ``OSError``
            The file cannot be written, with the path in the message:
            ``FileNotFoundError`` for a folder that does not exist,
            ``PermissionError`` for one that cannot be written. A file that
            was there is left untouched.
        ``TypeError``
            ``file`` is neither a path nor a binary file object.
        ``ValueError``
            A refused chunk left the bank unable to go on
            (:meth:`fit_predict`, Notes).
        ``RuntimeError``
            A call that learns is in flight on another thread.
        :class:`~polars_online.UnstableWarning`
            A warning, not an error, issued only under
            ``POLARS_ONLINE_WARN_UNSTABLE=1``: a spec holds a formula target,
            whose written form is unstable.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            bank.save("bank.state")                       # a path, replaced safely
            print(po.ModelBank.load("bank.state").rows_fed)

        .. code-block:: text

            400

        .. rubric:: See also

        - :meth:`load`, which reads the file back.
        - :meth:`save_bytes` and :meth:`write_json`, the state as bytes and as
          JSON.
        - `Resume from a saved state
          <https://github.com/hgilde/polars-online/blob/main/docs/how-to/resume-from-a-saved-state.md>`_.

        .. rubric:: Notes

        The state is versioned MessagePack. A path is written to a temporary
        file beside it, synced to the disk, and renamed into place, so a
        power loss cannot leave an empty file where a good state was. The
        sync takes about 4 ms on macOS, against about 0.1 ms to write a state
        of 396 KiB. In the same measurement, on a Mac, a whole save of 500
        groups (encoding the state, writing it and renaming it) took 5.0 ms
        with the sync and 0.5 ms without it. A save after every chunk is
        dominated by the sync, and a save after every hundredth chunk is not.
        """
        if isinstance(file, (str, os.PathLike)):
            # One frame more than `save_bytes`: the keyword forwarding's.
            self._warn_formula_form(stacklevel=5)
            self._native.save(os.fspath(file))
            return
        if not callable(getattr(file, "write", None)):
            msg = (
                "ModelBank.save: file must be a path or a file object opened for binary "
                f"writing, but it is of type {type(file).__name__}. Pass a path, or a file opened "
                "with 'wb'."
            )
            raise TypeError(msg)
        self._warn_formula_form(stacklevel=5)
        file.write(bytes(self._native.save_bytes()))

    def _warn_formula_form(self, stacklevel: int = 4) -> None:
        """:class:`~polars_online.UnstableWarning` for a bank whose specs hold a
        formula target, whose written form is not promised (docs/PLAN.md task
        198, D5). ``stacklevel`` 4 is the caller's caller, the user's line under
        a method that calls this; a method under a decorator passes 5."""
        if any(
            isinstance(t, dict) and "formula" in t
            for spec in self._specs
            for t in spec.get("targets") or ()
        ):
            warn_unstable("a formula target's written form in a state file", stacklevel=stacklevel)

    def save_bytes(self) -> bytes:
        """The bytes :meth:`save` writes, for a store that is not a file.

        Use it to keep a state in a database or an object store, and read it
        back with :meth:`load_bytes`. Pickle and ``copy.deepcopy`` carry a
        bank as these bytes.

        .. rubric:: Returns

        ``bytes``: the state, as :meth:`save` writes it to a file.

        .. rubric:: Raises

        ``RuntimeError``
            A call that learns is in flight on another thread.
        ``ValueError``
            A refused chunk left the bank unable to go on
            (:meth:`fit_predict`, Notes).
        :class:`~polars_online.UnstableWarning`
            Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, for a formula target, as
            :meth:`save` says.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            blob = bank.save_bytes()                     # to keep wherever bytes can be kept
            print(po.ModelBank.load_bytes(blob).rows_fed)

        .. code-block:: text

            400

        .. rubric:: See also

        - :meth:`load_bytes`, which reads the bytes back.
        - :meth:`save`, which writes them to a file.
        """
        self._warn_formula_form()
        return bytes(self._native.save_bytes())

    @overload
    def write_json(self, file: None = None, *, pretty: bool = True) -> str: ...

    @overload
    def write_json(
        self, file: str | os.PathLike[str] | IO[str] | IO[bytes], *, pretty: bool = True
    ) -> None: ...

    def write_json(
        self,
        file: str | os.PathLike[str] | IO[str] | IO[bytes] | None = None,
        *,
        pretty: bool = True,
    ) -> str | None:
        """Write the bank's state as JSON, which can be read without this library.

        Use it to look at a state, to compare two states, or to hand one to a
        program that is not Python. It is an export, not a second form of
        state file: :meth:`load` reads only what :meth:`save` writes.

        .. rubric:: Parameters

        ``file``
            Where to write: a path, or a file object, text or binary (the
            bytes are UTF-8). ``None``, the default, returns the text, as
            Polars' ``DataFrame.write_json()`` does. A path is written
            directly, without :meth:`save`'s safe replacement: an export of a
            state kept elsewhere is mended by writing it again.
        ``pretty``
            Whether to indent the JSON over several lines. ``True`` by
            default. ``False`` writes it on one line.

        .. rubric:: Returns

        The JSON text when ``file`` is ``None``, and otherwise ``None``. It
        holds everything :meth:`save` writes, every value included: NaN and
        the infinities, which JSON has no literal for, are written as the
        strings ``"nan"``, ``"inf"`` and ``"-inf"``, the spellings a spec's
        ``half_life`` takes.

        .. rubric:: Raises

        ``ValueError``
            The export, read back, does not give the state's MessagePack
            byte for byte (see Notes), or a refused chunk left the bank
            unable to go on.
        ``TypeError``
            ``file`` is neither a path, a file object nor ``None``.
        ``OSError``
            A path cannot be written, such as ``FileNotFoundError`` for a
            folder that does not exist.
        ``RuntimeError``
            A call that learns is in flight on another thread.
        :class:`~polars_online.UnstableWarning`
            Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, for a formula target, as
            :meth:`save` says. The export carries the same written form.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            import json

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            state = json.loads(bank.write_json())     # the state, as Python objects
            bank.write_json("bank.json")               # the same text, in a file
            print(state["rows_fed"], [s["name"] for s in state["specs"]])

        .. code-block:: text

            400 ['ridge']

        .. rubric:: See also

        - :meth:`save`, which writes the state :meth:`load` reads.

        .. rubric:: Notes

        Every export is read back and converted to MessagePack, which must
        match the state's byte for byte. One that does not raises
        ``ValueError`` naming the problem, so the export is never a file that
        is wrong without saying so.
        """
        self._warn_formula_form()
        text = str(self._native.save_json_string(pretty))
        if file is None:
            return text
        if isinstance(file, (str, os.PathLike)):
            Path(file).write_text(text, encoding="utf-8")
        elif isinstance(file, io.TextIOBase):
            file.write(text)
        elif callable(getattr(file, "write", None)):
            # A binary file object, or one that does not say which: bytes, as
            # UTF-8, and the text where it takes no bytes.
            writer: Any = file
            try:
                writer.write(text.encode("utf-8"))
            except TypeError:
                writer.write(text)
        else:
            msg = (
                "ModelBank.write_json: file must be a path, a file object or None, but it is "
                f"of type {type(file).__name__}. Pass a path or a file object, or None to get the "
                "text back."
            )
            raise TypeError(msg)
        return None

    @deprecated_alias("ModelBank.to_json", "ModelBank.write_json", ", which returns the same text")
    def to_json(self, *, pretty: bool = True) -> str:
        """Return the state as JSON text: the old name of :meth:`write_json`.

        Use :meth:`write_json` without a file, which returns the same text.
        Each call issues a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`.
        """
        return self.write_json(pretty=pretty)

    @deprecated_alias("ModelBank.save_json", "ModelBank.write_json", ", which takes the same path")
    def save_json(self, path: str | os.PathLike[str], *, pretty: bool = True) -> None:
        """Write the state as JSON to a path: the old name of :meth:`write_json`.

        Use :meth:`write_json`, which takes the same path. Each call issues a
        :class:`~polars_online.PolarsOnlineDeprecationWarning`.
        """
        self.write_json(path, pretty=pretty)

    @classmethod
    @deprecated_keywords("ModelBank.load", {"path": "source"})
    def load(
        cls,
        source: str | os.PathLike[str] | bytes | IO[bytes],
        specs: Iterable[dict[str, Any]] | None = None,
        *,
        threads: int | None = None,
    ) -> ModelBank:
        """Load a bank from a state file :meth:`save` wrote, on this OS or any other.

        Use it to resume a saved bank, or to serve predictions from it with
        :meth:`predict`. The file carries the bank's specs, so you need not
        give them. For bytes from a store that is not a file, this call takes
        them as well, as :meth:`load_bytes` does.

        .. rubric:: Parameters

        ``source``
            A path, the bytes :meth:`save_bytes` gave, or a file object opened
            for binary reading, such as ``open(..., "rb")`` or an
            ``io.BytesIO``, as Polars' ``read_*`` functions take any of the
            three.
        ``specs``
            The specs you expect the file to hold. Given, the load is refused
            unless they are the file's, which is how a job that resumes
            checks that the state it found belongs to the bank it is about to
            run. ``None``, the default, takes the file's specs unchecked.
        ``threads``
            The loaded bank's ``threads``, as :class:`ModelBank` takes it.
            The file carries no count, so a bank saved on eight threads and
            loaded with none runs on the process's shared threads, with the
            same numbers.

        .. rubric:: Returns

        A :class:`ModelBank` holding the saved state, which goes on from the
        row after the last one it learned before the save.

        .. rubric:: Raises

        ``OSError``
            The file cannot be read, such as ``FileNotFoundError`` for a path
            that does not exist.
        ``TypeError``
            ``source`` is not a path, bytes or a file object, or the file
            object gives text rather than bytes. ``threads`` is not an
            integer.
        ``ValueError``
            The file can be read but is not a bank this build loads:

            - it is not a bank's state file, or it is damaged in a way the
              checks find (see Notes);
            - a newer build wrote it, under a format or state version above
              this build's;
            - a build of 0.13.0 or earlier wrote it, below the oldest state
              version this build loads, which is 1.0's
              (:func:`polars_online.schema_version` gives the current one):
              fit the bank again from its input;
            - a state contradicts its own spec, such as an ``sgd`` state
              without the scaler its ``standardize`` needs;
            - ``specs`` are given and differ from the file's;
            - ``threads`` is below 1, or above 1024 (or the machine's core
              count, where that is larger).
        :class:`~polars_online.UnstableWarning`
            Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, for a file holding a
            formula target, as :meth:`save` says.

        .. rubric:: Examples

        This code reads ``df`` and ``today`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            saved = po.ModelBank([spec])
            saved.fit_predict(df)
            saved.save("bank.state")

            bank = po.ModelBank.load("bank.state", specs=[spec])   # refused if the specs differ
            bank.fit_predict(today)                                # and the stream goes on
            print(bank.rows_fed)

        .. code-block:: text

            800

        .. rubric:: See also

        - :meth:`save`, which writes the state file.
        - :meth:`skip_learned`, for input that overlaps what the state has
          learned.
        - `Resume from a saved state
          <https://github.com/hgilde/polars-online/blob/main/docs/how-to/resume-from-a-saved-state.md>`_.

        .. rubric:: Notes

        **The file carries no checksum.** The load checks the file's envelope
        (its magic number, its format and state versions, its number of
        states) and each state's shape against its spec, every vector at the
        length its spec gives it. A damaged file that still decodes and passes
        those checks loads: a flipped bit in a stored number is a slightly
        different state, consistent with itself, and the model goes on from
        it. Keep a state file where it cannot be damaged, or keep a checksum
        beside it.
        """
        threads = _check_threads(threads)
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        elif isinstance(source, (str, os.PathLike)):
            data = Path(source).read_bytes()
        elif callable(getattr(source, "read", None)):
            data = source.read()
            if not isinstance(data, (bytes, bytearray)):
                msg = (
                    "ModelBank.load: the file object gave an object of type "
                    f"{type(data).__name__}, not bytes. Open the file in binary mode ('rb')."
                )
                raise TypeError(msg)
            data = bytes(data)
        else:
            msg = (
                "ModelBank.load: source must be a path, bytes or a file object opened for "
                f"binary reading, but it is of type {type(source).__name__}. Pass one of those."
            )
            raise TypeError(msg)
        # One frame more than `load_bytes`: the keyword forwarding's.
        return cls._load_bytes(data, specs, stacklevel=5, threads=threads)

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        specs: Iterable[dict[str, Any]] | None = None,
        *,
        threads: int | None = None,
    ) -> ModelBank:
        """Load a bank from the bytes :meth:`save_bytes` gave.

        Use it for a state kept in a store that is not a file. :meth:`load`
        takes bytes as well, with the same checks.

        .. rubric:: Parameters

        ``data``
            The bytes :meth:`save_bytes` returned.
        ``specs``
            The specs you expect the bytes to hold, checked as :meth:`load`
            checks them. ``None``, the default, takes their specs unchecked.
        ``threads``
            The loaded bank's ``threads``, as :class:`ModelBank` takes it.

        .. rubric:: Returns

        A :class:`ModelBank` holding the saved state.

        .. rubric:: Raises

        ``ValueError`` and ``TypeError``
            As :meth:`load` raises them, for bytes that are not a bank this
            build loads, and for a ``threads`` it refuses. ``TypeError`` also
            when ``data`` is not ``bytes``.
        :class:`~polars_online.UnstableWarning`
            Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, for bytes holding a
            formula target, as :meth:`save` says.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit_predict(df)
            again = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
            print(again.rows_fed)

        .. code-block:: text

            400

        .. rubric:: See also

        - :meth:`save_bytes`, which gives the bytes.
        - :meth:`load`, which reads a file too.
        """
        return cls._load_bytes(data, specs, stacklevel=4, threads=_check_threads(threads))

    @classmethod
    def _load_bytes(
        cls,
        data: bytes,
        specs: Iterable[dict[str, Any]] | None,
        stacklevel: int,
        threads: int | None = None,
    ) -> ModelBank:
        """:meth:`load_bytes`, with the formula form's warning pointed
        ``stacklevel`` frames up from :meth:`_warn_formula_form`, so a public
        method's caller is named however many frames it adds."""
        specs_json = _json(list(specs)) if specs is not None else None
        native = _native.ModelBank.load_bytes(data, specs_json, threads)
        bank = cls._wrap(native)
        bank._warn_formula_form(stacklevel=stacklevel + 1)
        return bank

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
    groups: list[tuple[str | None, float | None, int | None, int | None]],
    schema: pl.Schema,
) -> pl.Expr:
    """True on the rows of one spec's input its state has not learned: a clock
    after the group's last one, a group the bank has not seen, or a null clock
    (:meth:`ModelBank.skip_learned`)."""
    for column, role in ((clock, "clock"), (group, "group")):
        if column is not None and column not in schema:
            msg = (
                f'spec "{name}": {role} column {column!r} is not in the input. Pass an input '
                f"with the {role} column the spec reads."
            )
            raise ColumnNotFoundError(msg)
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
        msg = (
            f'spec "{name}": clock column {clock!r} has dtype {dtype}, which is not a clock. '
            "Pass the clock as a number, a Date, a Datetime or a Duration."
        )
        raise ValueError(msg)
    temporal = not dtype.is_numeric()
    lasts: dict[str | None, float | int] = {}
    whole = True
    for key, number, nanos, integer in groups:
        if number is None and nanos is None and integer is None:
            continue
        if (nanos is not None) != temporal:
            was, now = ("temporal", "numeric") if nanos is not None else ("numeric", "temporal")
            msg = (
                f'spec "{name}": clock column {clock!r} is {now} in the frame, but it was {was} '
                f"in the bank. Cast the clock back to the {was} type the bank learned it as."
            )
            raise ValueError(msg)
        seen: float | int
        if nanos is not None:
            seen = nanos // per
        elif integer is not None:
            seen = integer
        else:
            assert number is not None
            seen = number
            whole = False
        lasts[key] = seen
    # An integer clock the bank held as one is compared in integers (task
    # 200): as doubles, two epoch nanoseconds near 1.8e18 a step apart tie,
    # and the later row was dropped as learned. A value past an `Int64` is
    # kept, null, for the bank to refuse by name.
    integer_clock = dtype.is_integer() and whole
    if integer_clock:
        value = pl.col(clock).cast(pl.Int64, strict=False)
    elif not temporal:
        lasts = {k: float(v) for k, v in lasts.items()}
    kind: type[pl.DataType] = pl.Int64 if temporal or integer_clock else pl.Float64
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
