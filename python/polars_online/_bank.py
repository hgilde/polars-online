"""Chunk-fed model bank: memory O(state), not O(data)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from polars_online import _polars_online as _native
from polars_online._spec import _json

__all__ = ["ModelBank"]


class ModelBank:
    """Runs a list of specs over ordered chunks, one state per (spec, group).

    Chunks must arrive in stream order within each group (the clock must not
    run backwards unless the spec's reset semantics say so). Feed
    ``LazyFrame.collect_batches()`` for out-of-core streams.
    """

    def __init__(self, specs: Iterable[dict[str, Any]]) -> None:
        self.specs = list(specs)
        self._native = _native.ModelBank(_json(self.specs))

    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """One chunk in; the chunk plus one struct column per spec out."""
        outs = self._native.fit_predict(df)
        return df.with_columns([pl.Series(s) for s in outs])

    def fit_predict_batches(self, batches: Iterable[pl.DataFrame]) -> Iterable[pl.DataFrame]:
        """Lazily map ``fit_predict`` over an iterator of chunks."""
        for chunk in batches:
            yield self.fit_predict(chunk)

    def gram(self, spec: str | int, group: str | None = None) -> list[dict[str, Any]]:
        """The EW accumulators behind a spec's fit, per group and instance.

        Returns one dict per (group, decay instance) with:

        ``group``, ``instance``
            The group key (``None`` when ungrouped) and the instance's field
            suffix (``"@h500"``, or ``""`` for a single instance).
        ``n_eff``
            Accumulated weight behind these moments.
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

        ``spec`` is a spec name or index.

        Requires numpy, which is *not* a dependency of this package -- polars
        does not require it either, and one optional accessor is no reason to
        put it on every install. ``pip install polars-online[numpy]`` adds it.
        """
        try:
            import numpy as np
        except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
            msg = (
                "ModelBank.gram() returns numpy arrays, and numpy is not installed. "
                "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
            )
            raise ModuleNotFoundError(msg) from e

        names = self._native.spec_names()
        idx = names.index(spec) if isinstance(spec, str) else spec
        out = []
        for g, instance, k, n_eff, means, como, cross, tw in self._native.gram(idx, group):
            out.append(
                {
                    "group": g,
                    "instance": instance,
                    "n_eff": n_eff,
                    "means": np.asarray(means),
                    "comoments": np.asarray(como).reshape(k, k),
                    "cross_moments": np.asarray(cross).reshape(len(cross), k)
                    if cross
                    else np.zeros((0, k)),
                    "target_weights": np.asarray(tw),
                }
            )
        return out

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
        return dict(zip(self._native.spec_names(), self._native.output_fields(), strict=True))

    def save(self, path: str | Path) -> None:
        """Versioned msgpack state; loads on any supported OS."""
        self._native.save(str(path))

    def save_bytes(self) -> bytes:
        return bytes(self._native.save_bytes())

    @classmethod
    def load(cls, path: str | Path, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        """Load a saved bank. Passing ``specs`` asserts they match the file."""
        specs_json = _json(list(specs)) if specs is not None else None
        native = _native.ModelBank.load(str(path), specs_json)
        return cls._wrap(native)

    @classmethod
    def load_bytes(cls, data: bytes, specs: Iterable[dict[str, Any]] | None = None) -> ModelBank:
        specs_json = _json(list(specs)) if specs is not None else None
        native = _native.ModelBank.load_bytes(data, specs_json)
        return cls._wrap(native)

    @classmethod
    def _wrap(cls, native: Any) -> ModelBank:
        obj = cls.__new__(cls)
        obj._native = native
        obj.specs = []  # not round-tripped as dicts; the native side holds them
        return obj
