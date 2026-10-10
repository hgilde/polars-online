"""Streaming and online regression models for Polars.

One Rust core, run three ways, with the same numbers from each:

1. inside a Polars query: ``lf.online.fit_predict(specs)`` is a
   ``LazyFrame`` that streams its rows through a bank when it runs, and
   ``df.online.fit_predict(specs)`` fits a frame already in memory;
2. :class:`ModelBank`, a bank fed one chunk at a time, with memory O(state)
   rather than O(data); hand it a ``LazyFrame`` and it does the chunking
   (:meth:`ModelBank.fit_predict_batches`, :meth:`ModelBank.fit`);
3. the ``online`` command line: parquet, ipc, csv or ndjson in and out, with
   the state saved and resumed between runs (`docs/RUNNER.md
   <https://github.com/hgilde/polars-online/blob/main/docs/RUNNER.md>`_).

A spec names a model and the columns it reads (:mod:`polars_online.spec`).
Every way in takes a list of specs and writes one struct column per spec. A
target is a column, a column under a name of its own (:func:`target`), or a
window expression looking ahead. To fit a target computed from its own row's
columns, such as a price against the mid, make the column with Polars'
``with_columns`` in the query before the bank.

The modules around the bank:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - module
     - what it does
   * - :mod:`polars_online.eval`
     - scores a bank's output
   * - :mod:`polars_online.gram`
     - solves and diagnoses the running sums a bank exports
   * - :mod:`polars_online.stream`
     - transforms a query: labels that arrive late, series that tick at
       their own times, and windowed means, sums and rates looking back or
       ahead
   * - :mod:`polars_online.ops`
     - the window operators that :func:`polars_online.stream.with_windows`
       computes
   * - :mod:`polars_online.corr`
     - repairs and reads correlation matrices
   * - :mod:`polars_online.sim`
     - simulates streams whose truth is known
   * - :mod:`polars_online.gbdt`
     - builds XGBoost's and LightGBM's training inputs from files or a
       ``LazyFrame``

**Errors follow one contract throughout**, and each docstring says which
part of it applies:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - what went wrong
     - the exception
   * - a file that cannot be read or written
     - the ``OSError`` subclass for what went wrong (``FileNotFoundError``,
       ``PermissionError``, ...), with the path in the message
   * - a value that is refused: a spec parameter, a config key, what a
       column holds
     - ``ValueError``, naming the spec and the parameter or column
   * - a wrong type
     - ``TypeError``
   * - a spec name or position a bank has not got
     - ``KeyError`` or ``IndexError``
   * - a bank fed from two threads at once
     - ``RuntimeError``

Inside a running Polars query the same exception arrives wrapped in
Polars' ``ComputeError`` under py-polars 1.x, carrying its message, and as
itself under 2.0: how a query wraps an error is Polars' to change. A failed
run never replaces an output or a state file. A refused chunk leaves a bank
as it was, apart from the one case :meth:`ModelBank.fit_predict` names.
"""

from polars_online import (
    _frame,  # noqa: F401  (registers the frame namespaces)
    corr,
    eval,
    gbdt,
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
from polars_online.ops import (
    ewm_mean,
    ewm_rate,
    ewm_std,
    ewm_sum,
    ewm_var,
    increment,
    rewm_mean,
    rewm_rate,
    rewm_sum,
)

__version__ = "0.13.0"

__all__ = [
    "ewm_mean",
    "ewm_rate",
    "ewm_std",
    "ewm_sum",
    "ewm_var",
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
    "gbdt",
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
