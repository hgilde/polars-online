"""Chunk-fed model bank: memory O(state), not O(data)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from polars_online import _polars_online as _native
from polars_online._spec import _from_json, _json, coef_index

__all__ = ["ModelBank"]


class ModelBank:
    """Runs a list of specs over ordered chunks, one state per (spec, group).

    Chunks must arrive in stream order within each group (the clock must not
    run backwards unless the spec's reset semantics say so). Feed
    ``LazyFrame.collect_batches()`` for out-of-core streams.

    ``specs`` are the dicts the :mod:`polars_online.spec` builders make.
    ``ValueError`` when there are none, when two share a name, or when a
    dict is not a spec (the message names the field).

    A bank is one ordered stream, so it is not for two threads at once: a
    method that finds the bank in use on another thread raises
    ``RuntimeError`` saying so rather than interleave with it
    (:meth:`fit_predict` releases the GIL while it works). :meth:`predict`
    learns nothing, so any number of ``predict`` calls may overlap; only a
    ``fit_predict`` in flight refuses them, and they refuse it.
    """

    def __init__(self, specs: Iterable[dict[str, Any]]) -> None:
        self.specs = list(specs)
        self._native = _native.ModelBank(_json(self.specs))

    def __repr__(self) -> str:
        names = ", ".join(repr(s["name"]) for s in self.specs)
        n_groups = max((len(g) for g in self._native.groups()), default=0)
        return f"ModelBank([{names}], groups={n_groups}, rows_seen={self.rows_seen()})"

    def rows_seen(self) -> int:
        """Rows fed so far, over every chunk and group -- skipped rows and
        dropped groups included, so not the sum of :meth:`groups`."""
        return self._native.rows_seen()

    def groups(self, spec: str | int | None = None) -> pl.DataFrame:
        """The groups the bank holds state for: one row per (spec, group).

        ``group`` is the key as a string -- ``""`` for a spec without a
        ``group`` column, null for rows whose key was null, as in
        :meth:`solve_failures`. ``rows_processed`` counts the group's rows
        that the null policy did not skip, and ``last_clock`` is its last
        clock value (null before the first row, or on a row-count clock).
        State lives until :meth:`drop_groups` removes it, so this is how a
        long-running bank finds the groups that have gone quiet::

            stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
            bank.drop_groups(stale["group"])

        ``spec``, a name or a position, narrows the table to one spec:
        ``KeyError`` for a name the bank has not got (the message lists the
        names), ``IndexError`` for a position it has not got.
        """
        names = self._native.spec_names()
        per_spec = self._native.groups()
        if spec is not None:
            i = self._spec_index(spec)
            names, per_spec = [names[i]], [per_spec[i]]
        rows = [
            (name, key, n, clock)
            for name, groups in zip(names, per_spec, strict=True)
            for key, n, clock in groups
        ]
        return pl.DataFrame(
            rows,
            schema={
                "spec": pl.String,
                "group": pl.String,
                "rows_processed": pl.UInt64,
                "last_clock": pl.Float64,
            },
            orient="row",
        )

    def drop_groups(self, keys: Iterable[str | None], spec: str | int | None = None) -> int:
        """Forget the state of these groups, in every spec or in one, and
        return how many streams were dropped. Keys are as :meth:`groups`
        reports them; a key the bank does not hold is not an error, it just
        drops nothing. A dropped group starts cold if it appears again,
        exactly as a never-seen one would; nothing else in the bank changes,
        and :meth:`rows_seen` still counts the rows it was fed. ``spec`` is
        as for :meth:`groups` (``KeyError`` / ``IndexError`` for one the bank
        has not got)."""
        index = None if spec is None else self._spec_index(spec)
        return self._native.drop_groups(list(keys), index)

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

        The struct is named after the spec (``out["m"]``; its fields are
        :meth:`output_fields`), and ``pred`` in it is out-of-sample: computed
        from the state *before* the row updates it. Chunk boundaries never
        change the numbers, only the cadence at which ``coef`` is reported.

        Raises ``TypeError`` for anything but a ``DataFrame`` (a ``LazyFrame``
        is told to collect, or to feed :meth:`fit_predict_batches`), and
        ``ValueError`` -- naming the spec and the column -- when a column a
        spec reads (target, feature, clock, session, weight, group) is not
        in the frame; a target, feature, clock or weight column is not
        numeric (a datetime clock is refused rather than read as its epoch
        integer: cast it to the unit ``halflife`` and ``max_dclock`` are in);
        the clock has a null or non-finite value; a weight is negative (null
        skips the row); a spec is named like an input column, which the
        struct would replace; or a group's clock runs backwards under
        ``on_clock_reset="error"`` (the other policies absorb it). A refused
        chunk leaves the bank exactly as it was, so the corrected chunk can
        be fed. ``RuntimeError`` when the bank is in use on another thread
        (class docstring).
        """
        self._check_frame(df, "fit_predict")
        outs = self._native.fit_predict(df)
        return df.with_columns([pl.Series(s) for s in outs])

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Score a frame against the bank as it stands, learning nothing.

        Every row gets the struct :meth:`fit_predict` would give it as the
        next row of its group's stream, computed from the current state; the
        bank is left exactly as it was, so the call is safe from any number
        of threads at once and row order does not matter. This is the serving
        side of a trained bank: ``load`` once, ``predict`` per request, and
        ``fit_predict`` the rows later, in order, once their targets arrive.

        The frame needs each spec's feature and clock columns. Target columns
        are optional -- present, they give ``resid`` and the standardized
        residual; absent, those are null. The session column is optional and
        feeds ``session_gap``; a weight column is not read. A trend model
        (``holt``) extrapolates over the clock distance from the row it last
        learned, capped by ``max_dclock``, and a ``kalman`` with
        ``revert_halflife`` shrinks its coefficients over that distance
        exactly as the next ``fit_predict`` row would; the other coefficient
        models predict from their current coefficients regardless of the
        clock.

        Per field: ``n_eff``, ``lam_selected``, ``sigma``, the residual
        quantiles, autocorrelation and the metrics are the values the bank
        holds, frozen; ``coef`` is filled on the last accepted row (the same
        coefficients score every row); ``drift`` never fires; rows of a group
        the bank has never seen, or without usable features, are null
        throughout, as a skipped row is in ``fit_predict``.

        Raises what :meth:`fit_predict` raises for the same frame -- a
        missing or non-numeric column, a bad clock value -- except that a
        missing target is not an error, and ``RuntimeError`` only when a
        ``fit_predict`` is in flight on another thread.
        """
        self._check_frame(df, "predict")
        outs = self._native.predict(df)
        return df.with_columns([pl.Series(s) for s in outs])

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
                msg += ", or feed it in chunks with fit_predict_batches(lf.collect_batches())"
        else:
            msg = f"ModelBank.{what} takes a polars DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    def fit_predict_batches(self, batches: Iterable[pl.DataFrame]) -> Iterable[pl.DataFrame]:
        """Lazily map :meth:`fit_predict` over an iterator of chunks: each is
        fed as the generator reaches it, so ``lf.collect_batches()`` streams
        through the bank one chunk at a time. Whatever ``fit_predict`` raises
        for a chunk, this raises there; the chunks before it have been
        learned from."""
        for chunk in batches:
            yield self.fit_predict(chunk)

    def coef(self, spec: str | int, group: str | None = None) -> pl.DataFrame:
        """The coefficients behind a spec's fit: one row per (group, instance,
        position), so a bank loaded from a state file answers "what are the
        betas?" without a row of data::

            bank = po.ModelBank.load("state.bin")
            betas = bank.coef("ols")
            wide = betas.pivot("term", index=["group", "instance"], values="coef")

        The values are what the output's ``coef`` field reported on the last
        row each stream learned from: the fit *after* that row, which the
        next row's ``pred`` is computed from. Laid out by
        :func:`polars_online.spec.coef_index`:

        ``group``, ``instance``
            As :meth:`groups` and :meth:`gram` report them: the key as a
            string (``""`` for a spec without a ``group`` column) and the
            decay instance's field suffix (``"@h500"``, or ``""`` for a
            single one).
        ``n_eff``
            The accumulated weight behind the fit -- what the next row's
            ``n_eff`` field reports. The solve schedule (``solve_every``,
            default halflife/50 or every row for ``halflife=inf``, and
            ``max_rows_between_solves``) decides
            when a stream first solves, not ``min_periods``: ``pred`` waits
            for ``min_periods``, ``coef`` does not, so a fit with ``n_eff``
            below it is over fewer rows than the spec asks for (a solve over
            fewer rows than terms is a jittered one, counted by
            :meth:`solve_failures`).
        ``position``, ``target``, ``ridge``, ``feature_set``, ``lambda``, ``term``
            :func:`~polars_online.spec.coef_index`'s columns: ``position``
            indexes the flat ``coef`` list, ``term`` is ``"intercept"``, a
            feature name, or ``"level"``/``"trend"`` for ``holt``.
        ``coef``
            The value, in the features' original units; null until the
            stream's first solve, as ``coef`` is on those rows.

        ``spec`` and ``group`` are as for :meth:`gram`: ``KeyError`` /
        ``IndexError`` for a spec the bank has not got, and a group it has
        never seen gives an empty frame with the same columns. ``ValueError``
        for an ``ew_cov`` spec, which emits statistics, not coefficients,
        and for a ``seqtest`` spec, which emits evidence.
        """
        idx = self._spec_index(spec)
        layout = coef_index(self.specs[idx])
        n = layout.height
        groups: list[str | None] = []
        instances: list[str] = []
        n_effs: list[float] = []
        values: list[float | None] = []
        for g, instance, n_eff, coef in self._native.coef(idx, group):
            if coef is not None and len(coef) != n:
                msg = (
                    f"spec {self.specs[idx]['name']!r}: {len(coef)} coefficients for {n} positions"
                )
                raise AssertionError(msg)
            groups += [g] * n
            instances += [instance] * n
            n_effs += [n_eff] * n
            values += coef if coef is not None else [None] * n
        k = len(instances) // n
        body = pl.concat([layout] * k) if k else layout.clear()
        return body.with_columns(
            pl.Series("group", groups, pl.String),
            pl.Series("instance", instances, pl.String),
            pl.Series("n_eff", n_effs, pl.Float64),
            # Finite-or-null, as the output's `coef` field is: an `ew_class`
            # class no row has carried yet has NaN means.
            pl.Series("coef", values, pl.Float64).fill_nan(None),
        ).select("group", "instance", "n_eff", *layout.columns, "coef")

    def last_row(self, spec: str | int | None = None, group: str | None = None) -> pl.DataFrame:
        """The output struct as it stood on the last row each stream learned
        from: one row per (spec, group), the struct's fields unnested after
        ``spec`` and ``group``.

        It is the row :meth:`fit_predict` reported for that row, field for
        field -- ``pred``, ``resid``, ``sigma``, the metrics, the residual
        quantiles, ``n_eff``, and ``coef`` when the row carried it (a chunk's
        last row does; :meth:`coef` has the coefficients whichever row was
        last). It travels with the state, so a bank loaded from a file says
        how each model was doing without its output frame, and a directory
        of fits compares without keeping the last row of every output::

            fits = sorted(Path("fits").glob("*.bin"))
            table = pl.concat(
                [po.ModelBank.load(f).last_row().with_columns(file=pl.lit(f.name)) for f in fits],
                how="diagonal_relaxed",
            )
            table.sort("ic_y", descending=True)  # with emit_metrics=True

        ``spec``, a name or a position, narrows the table to one spec
        (``KeyError`` / ``IndexError`` for one the bank has not got, as
        :meth:`groups`); ``group`` to one group, and a group the bank has
        never seen gives an empty frame. Specs with different fields are
        stacked ``diagonal_relaxed``, so a field one spec has not got is
        null on its rows. A group with no learned row yet -- every row
        skipped so far, or a state file written before 0.2.0 -- is a row of
        nulls. :meth:`predict` does not move it, and a chunk that ends in
        skipped rows leaves the row before them.
        """
        names = self._native.spec_names()
        picked = range(len(names)) if spec is None else [self._spec_index(spec)]
        frames: list[pl.DataFrame] = []
        for i in picked:
            keys, struct = self._native.last_row(i, group)
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

    def summary(self, spec: str | int | None = None, group: str | None = None) -> pl.DataFrame:
        """What each stream has been fed: one row per (spec, group).

        Counts and ranges over every row routed to the group since its state
        began -- undecayed, so they say what the model was trained on rather
        than what it still remembers -- and kept in the state file, so a bank
        loaded from a file says it too. The columns:

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
            targets, at least one usable target.
        ``rows_zero_weight``
            Processed rows with weight 0 (the clock moved; nothing learned).
        ``weight_sum``
            Sum of the processed rows' weights (1 per row without a weight
            column).
        ``clock_min``, ``clock_max``, ``last_clock``
            The clock range fed and the last value; null on a row-count clock.
        ``session_changes``
            Rows whose session differed from the previous row's.
        ``clock_backwards``
            Rows whose clock fell below the previous row's within a session
            (what ``on_clock_reset`` decided about).
        ``resets``
            Rows at which ``session_gap="reset"`` or
            ``on_clock_reset="reset_state"`` restarted the stream.

        ``spec`` narrows to one spec (``KeyError`` / ``IndexError`` for one
        the bank has not got), ``group`` to one group; a group never seen
        gives an empty frame. A state file written before 0.2.0 carries no
        summary: its groups report ``spec``, ``group``, ``rows_processed``
        and ``last_clock``, and nulls elsewhere -- for good, since a count
        that began at the load would read as the whole history.
        :meth:`predict` moves none of it, and feeding the same rows in one
        chunk or a thousand gives the same numbers to the bit.
        """
        names = self._native.spec_names()
        picked = range(len(names)) if spec is None else [self._spec_index(spec)]
        frames = [
            self._native.summary(i, group).select(pl.lit(names[i]).alias("spec"), pl.all())
            for i in picked
        ]
        return pl.concat(frames)

    def describe(self, spec: str | int | None = None, group: str | None = None) -> pl.DataFrame:
        """Per-column statistics of what each stream has been fed: one row per
        (spec, group, input column), in spec order -- features, then targets,
        then the weight column.

        ``column`` and ``role`` (``"feature"``, ``"target"``, ``"weight"``)
        name the column; ``count`` and ``null_count`` partition the rows fed
        (a value counts when finite and within the input bound, as the models
        take it, and is a null otherwise -- polars nulls, NaN, infinities and
        magnitudes beyond the bound alike); ``mean``, ``std`` (sample,
        ``ddof=1``; null below two values), ``min`` and ``max`` are over the
        counted values, undecayed and in row order, so chunking cannot move
        them. An unsupervised model lists no targets, an ``ew_class`` label
        column has its counts only, and a comparison's target is the
        difference of residuals it tests, named as the spec names it.

        ``spec`` and ``group`` narrow the frame as in :meth:`summary`. A
        state file written before 0.2.0 lists its columns with every number
        null; see :meth:`summary`.
        """
        names = self._native.spec_names()
        picked = range(len(names)) if spec is None else [self._spec_index(spec)]
        frames = [
            self._native.describe(i, group).select(pl.lit(names[i]).alias("spec"), pl.all())
            for i in picked
        ]
        return pl.concat(frames)

    def gram(self, spec: str | int, group: str | None = None) -> list[dict[str, Any]]:
        """The EW accumulators behind a spec's fit, per group and instance.

        Returns one dict per (group, decay instance) with:

        ``group``, ``instance``
            The group key as :meth:`groups` reports it (``""`` for a spec
            without a ``group`` column, ``None`` for a null key) and the
            instance's field suffix (``"@h500"``, or ``""`` for a single
            instance).
        ``n_eff``
            Accumulated weight behind these moments.
        ``n_kish``
            Kish's effective sample size, ``n_eff**2 / sum(w**2)`` -- the
            number of *equally* weighted rows these moments are worth, and
            what a standard error computed from them divides by. ``n_eff``
            counts weight, not rows, so it is not a sample size:
            ``(1 + lam) / (1 - lam)`` is the Kish size of an exponentially
            weighted window, whatever the halflife's units. ``None`` before
            the first row.

            It is scale-free: decay divides ``n_eff`` and ``sum(w**2)`` by
            the same factor, so ``n_kish`` does not shrink when a stream goes
            quiet. It says how many rows these moments average, not how old
            they are -- ``n_eff`` and ``target_weights`` are what say that.
        ``means``
            EW column means, shape ``(k,)``.
        ``comoments``
            **Centered** co-moments, shape ``(k, k)`` -- the EW analogue of a
            centered ``X'X / n``. Centered is what makes it accurate at large
            offsets (E11b).
        ``cross_moments``
            Per-target **uncentered** cross-moments ``E[z*y]``, shape
            ``(n_targets, k)``. Empty for ``ew_cov``.
        ``target_weights``
            Per-target accumulated weight, shape ``(n_targets,)``. Differs from
            ``n_eff`` when targets have different null patterns.
        ``target_means``, ``target_vars``
            Per-target EW mean and **centered** variance of the target itself,
            shape ``(n_targets,)``, in the same arithmetic as ``comoments`` --
            a target's variance here is the variance an ``ew_cov`` over that
            column would report, to the bit. Empty for ``ew_cov``.
        ``target_n_kish``
            Per-target Kish effective sample size, ``target_weights**2 /
            sum(w**2)`` over that target's rows; ``nan`` for a target that has
            not seen a weighted row. Empty for ``ew_cov``.

        The target moments are what makes the export a *complete* sufficient
        statistic (ENHANCEMENTS E45). With the cross-moments alone there is no
        residual variance, no ``R^2``, no information criterion and no
        standard error to be had from a saved Gram, because every one of them
        needs ``Var[y]``::

            beta = solve(raw, cross_moments[t])           # the model's own fit
            resid_var = target_vars[t] - beta[1:] @ comoments[1:, 1:] @ beta[1:]
            r2 = 1 - resid_var / target_vars[t]

        A state saved before task 38 has none of them: ``n_kish``,
        ``target_means``, ``target_vars`` and ``target_n_kish`` are ``None``
        there, for that state's whole remaining life. The weight sums behind
        them cannot be replayed, and a ``sum(w**2)`` accumulated from the
        resume point against an ``n_eff`` from the whole stream would report
        an effective size too large by the length of the history -- a wrong
        number where ``None`` is the true answer.

        The two moment forms differ, and mixing them gives a silently wrong
        answer rather than an error, so the bridging identity is worth stating
        plainly::

            raw = comoments + np.outer(means, means)
            raw @ beta[t] == cross_moments[t]     # up to the ridge term

        Values are in the features' original units. The intercept, when the
        spec has one, is column 0: a constant 1, so it has zero variance in
        ``comoments`` and ``raw[0] == means``.

        Only models that keep a co-moment matrix report -- ``ewridge``,
        ``lasso`` and ``ew_cov``. The others yield nothing: ``rls`` and
        ``kalman`` track an inverse, and the gradient models keep no second
        moment at all.

        **Why this exists** (ENHANCEMENTS E30): the accumulators are the
        expensive part, and they are already exact, centered, decayed on the
        model's own clock with session and ``max_dclock`` handling, and
        resumable. Anyone wanting to do something *other than* our solve with
        them -- a custom penalty, an information criterion, ``cond(G)``, a
        scree plot, forward stepwise, orthogonal matching pursuit, or simply
        to check a fit by hand -- previously had to recompute ``X'X`` from raw
        data in a second pass. These come from one pass over data that is
        never materialized, at every point in the stream rather than one, and
        they are the same matrices the deployed model solves against.

        This is not a speed claim: for a single batch Gram over materialized
        data, BLAS ``dgemm`` is blocked, vectorized, and comfortably faster.

        ``spec`` is a spec name or position (``KeyError`` / ``IndexError``
        for one the bank has not got, as for :meth:`groups`); ``group``
        narrows the list to one group. A group the bank has never seen gives
        an empty list, as does a model that keeps no co-moments -- neither is
        an error.

        Requires numpy, which is *not* a dependency of this package -- polars
        does not require it either, and one optional accessor is no reason to
        put it on every install. ``pip install polars-online[numpy]`` adds it;
        without it the call raises ``ModuleNotFoundError`` saying so.
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
        out = []
        for row in self._native.gram(idx, group):
            g, instance, k, n_eff, n_kish, means, como, cross, tw = row[:9]
            tmeans, tvars, tkish = row[9:]
            out.append(
                {
                    "group": g,
                    "instance": instance,
                    "n_eff": n_eff,
                    "n_kish": n_kish,
                    "means": np.asarray(means),
                    "comoments": np.asarray(como).reshape(k, k),
                    "cross_moments": np.asarray(cross).reshape(len(cross), k)
                    if cross
                    else np.zeros((0, k)),
                    "target_weights": np.asarray(tw),
                    "target_means": None if tmeans is None else np.asarray(tmeans),
                    "target_vars": None if tvars is None else np.asarray(tvars),
                    # A target with no weighted row yet has no Kish size; the
                    # array says `nan` where the Rust side says `None`, as
                    # every other float array here does.
                    "target_n_kish": None
                    if tkish is None
                    else np.asarray([np.nan if v is None else v for v in tkish], dtype=float),
                }
            )
        return out

    def marginal(self, spec: str | int, group: str | None = None) -> pl.DataFrame:
        """The pairs a ``marginal`` spec keeps: one row per (group, instance,
        feature, target), in spec order -- groups sorted, targets in spec
        order, features in spec order within each.

        ``group`` and ``instance``
            As :meth:`gram` reports them (``""`` for a spec without a
            ``group`` column; ``""`` for a single decay instance, else the
            field suffix such as ``"@h500"``).
        ``feature``, ``target``
            The pair's columns, by name.
        ``n_eff``
            The target's accumulated weight ``W_t``: rows where the target
            was present, weighted and decayed. Differs from the struct's
            ``n_eff`` when targets have different null patterns.
        ``n_kish``
            ``W_t^2 / Q_t`` with ``Q_t`` the accumulated squared weight: the
            Kish effective sample size, the number of equally weighted rows
            that carry the same information. Null before the target's first
            row.
        ``mean_x``, ``var_x``, ``mean_y``, ``var_y``, ``cov``
            The pair's EW moments -- population form, over the decayed
            weights, so ``var`` is never negative.
        ``corr``, ``beta``, ``t``
            ``cov / sqrt(var_x var_y)``, ``cov / var_x`` and ``corr *
            sqrt((n_kish - 2) / (1 - corr^2))``; null until ``n_eff`` reaches
            the target's ``min_periods``, and null where undefined (a
            constant column, ``n_kish <= 2``).

        The pairs are read from the state, so a bank loaded from a file
        reports them as the bank that saved it would, and feeding the rows
        in one chunk or a thousand gives the same numbers to the bit. A
        group the bank has never seen gives an empty frame; a spec that is
        not a ``marginal`` is refused (``ValueError``) -- an ``ew_cov``'s
        moments are read with :meth:`gram`. ``spec`` is a name or position
        (``KeyError`` / ``IndexError`` for one the bank has not got).
        """
        return self._native.marginal(self._spec_index(spec), group)

    def solve_failures(self) -> dict[str, dict[str | None, int]]:
        """Jittered or failed matrix factorizations so far, per spec and group.

        A solve never returns NaN silently (docs/PLAN.md section 7): a
        near-singular system is retried with escalating diagonal jitter, and
        total failure keeps the previous coefficients. Both cases are counted
        here, so a nonzero value means the inputs are degenerate (constant or
        collinear features, or far too few observations for the feature count),
        not that anything crashed. Models that do not factorize -- rls, kalman,
        ftrl -- always report 0.
        """
        names = self._native.spec_names()
        return {
            name: dict(pairs)
            for name, pairs in zip(names, self._native.solve_failures(), strict=True)
        }

    def output_fields(self) -> dict[str, list[str]]:
        """Spec name -> the field names of its output struct, in order --
        :func:`polars_online.spec.output_fields` for every spec in the bank.
        The fields are fixed by the spec, so this is the output schema before
        any row is fed."""
        return dict(zip(self._native.spec_names(), self._native.output_fields(), strict=True))

    def save(self, path: str | Path) -> None:
        """Versioned msgpack state; loads on any supported OS.

        Written to a temporary sibling and renamed into place, so an
        interrupted save leaves the previous state where it was rather than
        truncating it (docs/IMPROVEMENTS.md C6). The rename is preceded by a
        filesystem sync, which is what a resumable file costs: ~4 ms on macOS,
        against ~0.5 ms for serializing 500 groups. Save every chunk and the
        sync dominates; save every hundredth and it disappears.

        Raises the ``OSError`` for what went wrong -- ``FileNotFoundError``
        for a directory that is not there, ``PermissionError`` for one that
        cannot be written -- with the path in the message; the file, if it
        existed, is untouched. ``RuntimeError`` while a ``fit_predict`` is in
        flight on another thread.
        """
        self._native.save(str(path))

    def save_bytes(self) -> bytes:
        """What :meth:`save` writes, as bytes -- for a store that is not a
        file (:meth:`load_bytes` reads them back). This is also what pickle
        and ``copy.deepcopy`` carry. ``RuntimeError`` while a ``fit_predict``
        is in flight on another thread."""
        return bytes(self._native.save_bytes())

    @classmethod
    def load(cls, path: str | Path, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        """A bank from a file :meth:`save` wrote, on this or any other OS.

        The file carries the specs, so none need be given; passing ``specs``
        asserts they are the file's, which is how a resuming job checks that
        the state it found is the state of the bank it is about to run.

        Raises ``FileNotFoundError`` (or the ``OSError`` for what went wrong)
        when the file cannot be read, and ``ValueError`` when it can but is
        not a bank this build loads: not a bank state file at all, written by
        a newer build (the file's format or state schema version is above
        this build's), or ``specs`` differ from the file's.
        """
        return cls.load_bytes(Path(path).read_bytes(), specs)

    @classmethod
    def load_bytes(cls, data: bytes, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        """:meth:`load` from the bytes :meth:`save_bytes` gave, with the same
        ``specs`` check and the same ``ValueError`` for bytes that are not a
        bank this build loads."""
        specs_json = _json(list(specs)) if specs is not None else None
        native = _native.ModelBank.load_bytes(data, specs_json)
        return cls._wrap(native)

    @classmethod
    def _wrap(cls, native: Any) -> ModelBank:
        obj = cls.__new__(cls)
        obj._native = native
        # The state file carries the specs; they come back as the same dicts
        # the builders made.
        obj.specs = _from_json(native.specs_json())
        return obj
