"""Streaming and online regression models for Polars.

One Rust core, run three ways, with the same numbers from each:

1. inside a Polars query: ``lf.online.fit_predict(specs)`` is a query
   that feeds its rows through a bank when it runs, and
   ``df.online.fit_predict(specs)`` fits a frame already in memory;
2. :class:`ModelBank`, a model bank fed one chunk at a time, whose memory
   grows with the models' state and never with the number of rows that
   have passed. Handed a query, it reads the query a chunk at a time
   (:meth:`ModelBank.fit_predict_batches`, :meth:`ModelBank.fit`);
3. the ``polars-online`` command line: Parquet, Arrow IPC, CSV or NDJSON
   in and out, with the state saved and resumed between runs (`docs/RUNNER.md
   <https://github.com/hgilde/polars-online/blob/main/docs/RUNNER.md>`_).

A spec names a model and the columns it reads (:mod:`polars_online.spec`).
Every way in takes a list of specs and writes one output column per spec. A
target is a column, a column under a name of its own (:func:`target`), or a
formula target, an expression over the window operators that looks ahead.
To fit a target computed from its own row's columns, such as a price
against the mid, make the column with Polars' ``with_columns`` in the query
before the bank.

The modules around the bank:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - module
     - what it does
   * - :mod:`polars_online.eval`
     - scores a bank's output
   * - :mod:`polars_online.gram`
     - solves and diagnoses the Grams a bank exports
   * - :mod:`polars_online.stream`
     - transforms a query: targets whose values arrive late, series that tick at
       their own times, and exponentially weighted means, sums, rates,
       variances and quantiles looking back or ahead
   * - :mod:`polars_online.ops`
     - the window operators that :func:`polars_online.stream.with_windows`
       computes, and that a formula target is built from
   * - :mod:`polars_online.corr`
     - repairs and reads correlation matrices
   * - :mod:`polars_online.sim`
     - simulates streams whose truth is known
   * - :mod:`polars_online.gbdt`
     - builds XGBoost's and LightGBM's training inputs from files or a
       query

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
   * - a column a spec reads that the frame does not have
     - :class:`ColumnNotFoundError`, naming the spec, the column's role and
       the column: Polars' ``ColumnNotFoundError`` and a ``ValueError``
       both, so either ``except`` catches it
   * - a spec's output column that a scoring function's frame does not
       have
     - :class:`OutputColumnNotFoundError`, a :class:`ColumnNotFoundError`
   * - a wrong type
     - ``TypeError``; where the same refusal once raised ``ValueError``,
       :class:`WrongTypeError`
   * - a value out of range, or a combination of settings refused, that
       once raised ``TypeError``
     - :class:`RefusedValueError`
   * - a spec name or position a bank does not have
     - ``KeyError`` or ``IndexError``
   * - a bank fed from two threads at once
     - ``RuntimeError``

**Classes with two parents.** A refusal that once raised a class the
table above no longer gives it raises a class of this library's own that
inherits from both: the class the table gives, and the class it raised
before. :class:`WrongTypeError` is a ``TypeError`` and a ``ValueError``,
:class:`RefusedValueError` a ``ValueError`` and a ``TypeError``, and
:class:`OutputColumnNotFoundError` a :class:`ColumnNotFoundError` and a
``KeyError``. So a handler written for either class keeps catching the
refusal, and new code can catch the class that names the problem. Changing
the class outright would let such a refusal slip past an ``except`` that
catches it today.
The older parent is dropped at the next major version: catch ``TypeError``,
``ValueError`` and ``ColumnNotFoundError`` respectively.

Inside a running Polars query the same exception arrives wrapped in
Polars' ``ComputeError`` under py-polars 1.x, carrying its message, and as
itself under 2.0: how a query wraps an error is Polars' to change. A run
the bank refuses, or one abandoned before its last row, never replaces a
state file. A refused chunk leaves a bank
as it was, apart from the two cases :meth:`ModelBank.fit_predict` names.
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
from polars_online._bank import ModelBank, thread_pool_size
from polars_online._errors import (
    ColumnNotFoundError,
    OutputColumnNotFoundError,
    RefusedValueError,
    WrongTypeError,
)
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
)
from polars_online._spec import FormulaTarget, Target, target
from polars_online._warnings import (
    PolarsOnlineDeprecationWarning,
    PolarsOnlineWarning,
    UnstableWarning,
)
from polars_online.ops import (
    ewm_mean,
    ewm_quantile,
    ewm_rate,
    ewm_std,
    ewm_sum,
    ewm_var,
    increment,
    # The old names of `ewm_*(reverse=True)`, forwarded with a deprecation
    # warning and kept out of `__all__`, so the reference shows the new form.
    rewm_mean,  # noqa: F401
    rewm_rate,  # noqa: F401
    rewm_sum,  # noqa: F401
)

__version__ = "1.4.0"

__all__ = [
    "ewm_mean",
    "ewm_quantile",
    "ewm_rate",
    "ewm_std",
    "ewm_sum",
    "ewm_var",
    "increment",
    "ops",
    "ArrowStruct",
    "ColumnNotFoundError",
    "ConsumedSourceWarning",
    "ModelBank",
    "OrderNotGuaranteedWarning",
    "OutputColumnNotFoundError",
    "PolarsOnlineDeprecationWarning",
    "PolarsOnlineWarning",
    "ReadinessWarning",
    "RefusedValueError",
    "UnstableWarning",
    "WrongTypeError",
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
