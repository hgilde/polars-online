"""Streaming / online regression models for Polars.

Two entry points share one Rust core (see ``docs/PLAN.md``):

1. :class:`ModelBank`, chunk-fed, memory O(state) not O(data) -- and as a
   plan, ``lf.online.fit_predict(specs)``, a ``LazyFrame`` that streams
   (``df.online.fit_predict(specs)`` for a frame in memory);
2. :func:`run`, or the ``online`` CLI: parquet, ipc, csv or ndjson in and out.

A third, the expression namespace ``pl.col("y").online.<model>(...)``, is
dormant: it exists only in a build with the ``expr-plugin`` feature
(``polars_online._expr``).
"""

from polars_online import (
    _frame,  # noqa: F401  (registers the frame namespaces)
    eval,
    spec,
)
from polars_online._bank import ModelBank
from polars_online._frame import fit_predict, predict
from polars_online._polars_online import has_expr_plugin, native_version, schema_version
from polars_online._runner import run

__version__ = "0.1.0"

__all__ = [
    "ModelBank",
    "__version__",
    "eval",
    "fit_predict",
    "native_version",
    "predict",
    "run",
    "schema_version",
    "spec",
]

if has_expr_plugin():
    # The dormant expression surface: registers `pl.Expr.online` and exports
    # its typed spelling, in a build that carries the plugin symbol.
    from polars_online._expr import online as online

    __all__ += ["online"]
