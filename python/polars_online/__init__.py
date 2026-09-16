"""Streaming and online regression models for Polars.

One Rust core, reached three ways:

1. :class:`ModelBank`, a bank fed one chunk at a time, with memory O(state)
   rather than O(data); and the same bank as a plan,
   ``lf.online.fit_predict(specs)``, a ``LazyFrame`` that streams, or
   ``df.online.fit_predict(specs)`` for a frame in memory;
2. :func:`run`, or the ``online`` CLI: parquet, ipc, csv or ndjson in and out,
   with the state saved and resumed between runs;
3. the expression namespace, ``pl.col("y").online.<model>(...)``, for a frame
   in memory only: polars hands a user expression its whole column in either
   engine, so every use warns (:class:`InMemoryExpressionWarning`).

A spec names a model and the columns it reads (:mod:`polars_online.spec`);
every way in takes a list of them and writes one struct column per spec.
Around them: :mod:`polars_online.eval` scores the output;
:mod:`polars_online.gram` solves and diagnoses the running sums a bank
exports; :mod:`polars_online.corr` repairs and reads correlation matrices;
:mod:`polars_online.prep` prepares late labels and asynchronous series;
:mod:`polars_online.sim` simulates streams whose truth is known.

Errors follow one contract throughout, and each docstring says which of it
applies. A file that cannot be read or written is the ``OSError`` subclass for
what went wrong (``FileNotFoundError``, ``PermissionError``, ...), with the
path in the message. A value that is refused (a spec parameter, a config key,
what a column holds) is ``ValueError`` naming the spec and the parameter or
column. A wrong type is ``TypeError``; a spec name or position a bank has not
got is ``KeyError`` or ``IndexError``; a bank fed from two threads at once is
``RuntimeError``. Inside a polars plan the same messages arrive as polars'
``ComputeError``. A refused chunk never changes a bank, and a failed run never
replaces an output or a state file.
"""

from polars_online import (
    _expr,  # noqa: F401  (registers the expression namespace)
    _frame,  # noqa: F401  (registers the frame namespaces)
    corr,
    eval,
    gram,
    prep,
    sim,
    spec,
)
from polars_online._bank import ModelBank
from polars_online._expr import InMemoryExpressionWarning, online
from polars_online._frame import fit_predict, predict, unnest
from polars_online._polars_online import native_version, schema_version, thread_pool_size
from polars_online._runner import run

__version__ = "0.6.0"

__all__ = [
    "InMemoryExpressionWarning",
    "ModelBank",
    "__version__",
    "corr",
    "eval",
    "fit_predict",
    "gram",
    "native_version",
    "online",
    "predict",
    "prep",
    "run",
    "schema_version",
    "sim",
    "spec",
    "thread_pool_size",
    "unnest",
]
