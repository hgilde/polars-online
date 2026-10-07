"""Streaming and online regression models for Polars.

One Rust core, reached two ways:

1. :class:`ModelBank`, a bank fed one chunk at a time, with memory O(state)
   rather than O(data); hand it a ``LazyFrame`` and it does the chunking
   (:meth:`ModelBank.fit_predict_batches`, :meth:`ModelBank.fit`). The same
   bank runs as a plan, ``lf.online.fit_predict(specs)``, a ``LazyFrame`` that
   streams, or ``df.online.fit_predict(specs)`` for a frame in memory;
2. the ``online`` command line: parquet, ipc, csv or ndjson in and out, with
   the state saved and resumed between runs (``docs/RUNNER.md``).

A spec names a model and the columns it reads (:mod:`polars_online.spec`);
every way in takes a list of them and writes one struct column per spec. A
target is a column, a column under a name of its own (:func:`target`), or a
window expression looking ahead; a target computed from its own row's
columns, a price against the mid, is a column made with ``with_columns``.
Around them: :mod:`polars_online.eval` scores the output;
:mod:`polars_online.gram` solves and diagnoses the running sums a bank
exports; :mod:`polars_online.corr` repairs and reads correlation matrices;
:mod:`polars_online.stream` transforms a plan: late labels, asynchronous series,
windowed means, sums and rates looking back or ahead (the operators of :mod:`polars_online.ops`);
:mod:`polars_online.sim` simulates streams whose truth is known.

Errors follow one contract throughout, and each docstring says which of it
applies. A file that cannot be read or written is the ``OSError`` subclass for
what went wrong (``FileNotFoundError``, ``PermissionError``, ...), with the
path in the message. A value that is refused (a spec parameter, a config key,
what a column holds) is ``ValueError`` naming the spec and the parameter or
column. A wrong type is ``TypeError``; a spec name or position a bank has not
got is ``KeyError`` or ``IndexError``; a bank fed from two threads at once is
``RuntimeError``. Inside a running polars plan the same exception arrives
wrapped in polars' ``ComputeError`` under py-polars 1.x, carrying its
message, and as itself under 2.0: how a plan wraps an error is Polars' to
change. A refused chunk never changes a bank, and a failed run never replaces
an output or a state file.
"""

from polars_online import (
    _frame,  # noqa: F401  (registers the frame namespaces)
    corr,
    eval,
    gram,
    ops,
    sim,
    spec,
    stream,
)
from polars_online._bank import ModelBank
from polars_online._frame import (
    ConsumedSourceWarning,
    OrderNotGuaranteedWarning,
    ReadinessWarning,
    fit_predict,
    predict,
    unnest,
)
from polars_online._polars_online import (
    ArrowStruct,
    native_version,
    schema_version,
    thread_pool_size,
)
from polars_online._spec import FormulaTarget, Target, target
from polars_online._warnings import PolarsOnlineDeprecationWarning, UnstableWarning
from polars_online.ops import ewm_mean, ewm_rate, ewm_sum, increment, rewm_mean, rewm_rate, rewm_sum

__version__ = "0.13.0"

__all__ = [
    "ewm_mean",
    "ewm_rate",
    "ewm_sum",
    "increment",
    "ops",
    "rewm_mean",
    "rewm_rate",
    "rewm_sum",
    "ArrowStruct",
    "ConsumedSourceWarning",
    "ModelBank",
    "OrderNotGuaranteedWarning",
    "PolarsOnlineDeprecationWarning",
    "ReadinessWarning",
    "UnstableWarning",
    "__version__",
    "corr",
    "eval",
    "fit_predict",
    "gram",
    "native_version",
    "predict",
    "schema_version",
    "sim",
    "spec",
    "stream",
    "target",
    "FormulaTarget",
    "Target",
    "thread_pool_size",
    "unnest",
]
