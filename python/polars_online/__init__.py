"""Streaming / online regression models for Polars.

Three entry points share one Rust core (see ``docs/PLAN.md``):

1. :class:`ModelBank`, chunk-fed, memory O(state) not O(data) -- and as a
   plan, ``lf.online.fit_predict(specs)``, a ``LazyFrame`` that streams
   (``df.online.fit_predict(specs)`` for a frame in memory);
2. :func:`run`, or the ``online`` CLI: parquet, ipc, csv or ndjson in and out;
3. the expression namespace, ``pl.col("y").online.<model>(...)``, for a frame
   in memory only: polars calls it with the whole column in either engine, so
   every use warns (:class:`InMemoryExpressionWarning`, ``polars_online._expr``).

Errors follow one contract throughout, and each docstring says which of it
applies: a file that cannot be read or written is the ``OSError`` subclass
for what went wrong (``FileNotFoundError``, ``PermissionError``, ...), with
the path in the message; a value that is refused -- a spec parameter, a
config key, what a column holds -- is ``ValueError`` naming the spec and
the parameter or column; a wrong type is ``TypeError``; a spec name or
position a bank has not got is ``KeyError`` or ``IndexError``; a bank fed
from two threads at once is ``RuntimeError``. Inside a polars plan the same
messages arrive as polars' ``ComputeError``. A refused chunk never changes
a bank, and a failed run never replaces an output or a state file.
"""

from polars_online import (
    _expr,  # noqa: F401  (registers the expression namespace)
    _frame,  # noqa: F401  (registers the frame namespaces)
    eval,
    gram,
    prep,
    spec,
)
from polars_online._bank import ModelBank
from polars_online._expr import InMemoryExpressionWarning, online
from polars_online._frame import fit_predict, predict, unnest
from polars_online._polars_online import native_version, schema_version, thread_pool_size
from polars_online._runner import run

__version__ = "0.2.0"

__all__ = [
    "InMemoryExpressionWarning",
    "ModelBank",
    "__version__",
    "eval",
    "fit_predict",
    "gram",
    "native_version",
    "online",
    "predict",
    "prep",
    "run",
    "schema_version",
    "spec",
    "thread_pool_size",
    "unnest",
]
