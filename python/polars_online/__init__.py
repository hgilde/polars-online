"""Streaming / online regression models for Polars.

Three entry points share one Rust core (see ``docs/PLAN.md``):

1. :class:`ModelBank`, chunk-fed, memory O(state) not O(data) -- and as a
   plan, ``lf.online.fit_predict(specs)``, a ``LazyFrame`` that streams
   (``df.online.fit_predict(specs)`` for a frame in memory);
2. :func:`run`, or the ``online`` CLI: parquet, ipc, csv or ndjson in and out;
3. the expression namespace, ``pl.col("y").online.<model>(...)``, for a frame
   in memory only: polars calls it with the whole column in either engine, so
   every use warns (:class:`InMemoryExpressionWarning`, ``polars_online._expr``).
"""

from polars_online import (
    _expr,  # noqa: F401  (registers the expression namespace)
    _frame,  # noqa: F401  (registers the frame namespaces)
    eval,
    spec,
)
from polars_online._bank import ModelBank
from polars_online._expr import InMemoryExpressionWarning, online
from polars_online._frame import fit_predict, predict
from polars_online._polars_online import native_version, schema_version
from polars_online._runner import run

__version__ = "0.1.0"

__all__ = [
    "InMemoryExpressionWarning",
    "ModelBank",
    "__version__",
    "eval",
    "fit_predict",
    "native_version",
    "online",
    "predict",
    "run",
    "schema_version",
    "spec",
]
