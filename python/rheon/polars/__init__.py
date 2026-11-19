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

A spec names a model and the columns it reads (:mod:`rheon.spec`).
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
   * - :mod:`rheon.polars.eval`
     - scores a bank's output
   * - :mod:`rheon.gram`
     - solves and diagnoses the Grams a bank exports
   * - :mod:`rheon.polars.stream`
     - transforms a query: targets whose values arrive late, series that tick at
       their own times, and exponentially weighted means, sums, rates,
       variances and quantiles looking back or ahead
   * - :mod:`rheon.polars.ops`
     - the window operators that :func:`rheon.polars.stream.with_windows`
       computes, and that a formula target is built from
   * - :mod:`rheon.corr`
     - repairs and reads correlation matrices
   * - :mod:`rheon.polars.sim`
     - simulates streams whose truth is known

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
     - ``TypeError``; for a column of a type its role cannot take, or a
       value of the wrong type in a spec, its subclass
       :class:`WrongTypeError`
   * - a value out of range, or a combination of settings refused
     - ``ValueError``; for a combination, its subclass
       :class:`RefusedValueError`
   * - a spec name or position a bank does not have
     - ``KeyError`` or ``IndexError``
   * - a bank fed from two threads at once
     - ``RuntimeError``

**Subclasses of this library's own.** :class:`WrongTypeError` is a
``TypeError``, :class:`RefusedValueError` a ``ValueError`` and
:class:`OutputColumnNotFoundError` a :class:`ColumnNotFoundError`, each
with the one parent the table above gives its problem, so catch the class
the table names.

Inside a running Polars query the same exception arrives wrapped in
Polars' ``ComputeError`` under py-polars 1.x, carrying its message, and as
itself under 2.0: how a query wraps an error is Polars' to change. A run
the bank refuses, or one abandoned before its last row, never replaces a
state file. A refused chunk leaves a bank
as it was, apart from the two cases :meth:`ModelBank.fit_predict` names.

.. rubric:: See also

- The `feature index
  <https://github.com/hgilde/polars-online/blob/main/docs/FEATURES.md>`_,
  every feature of the library with a line each.
- The `highlights page
  <https://github.com/hgilde/polars-online/blob/main/docs/HIGHLIGHTS.md>`_,
  four use cases run end to end.
"""

# ruff: noqa: E402  (the imports below follow the docstring on purpose)
from rheon import __version__, corr, gram, spec
from rheon._errors import (
    ColumnNotFoundError,
    OutputColumnNotFoundError,
    RefusedValueError,
    WrongTypeError,
)
from rheon._rheon import (
    ArrowStruct,
    native_version,
    schema_version,
)
from rheon._spec import FormulaTarget, Target, target
from rheon._threads import thread_pool_size
from rheon._warnings import (
    PolarsOnlineDeprecationWarning,
    PolarsOnlineWarning,
    ReadinessWarning,
    UnstableWarning,
)
from rheon.polars import (
    _frame,  # noqa: F401  (registers the frame namespaces)
    eval,
    ops,
    sim,
    stream,
)
from rheon.polars._bank import ModelBank
from rheon.polars._frame import (
    ConsumedSourceWarning,
    OrderNotGuaranteedWarning,
    fit_predict,
    predict,
    unnest,
)
from rheon.polars.ops import (
    ewm_mean,
    ewm_quantile,
    ewm_rate,
    ewm_std,
    ewm_sum,
    ewm_var,
    increment,
    # The old names of `ewm_*(reverse=True)`, which refuse a call naming the
    # new form and are kept out of `__all__`, so the reference shows it alone.
    rewm_mean,  # noqa: F401
    rewm_rate,  # noqa: F401
    rewm_sum,  # noqa: F401
)

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
