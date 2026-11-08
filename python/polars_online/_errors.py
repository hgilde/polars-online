"""This library's own exception classes.

Until the next major version, three of them, :class:`WrongTypeError`,
:class:`RefusedValueError` and :class:`OutputColumnNotFoundError`, keep a
second parent: the class the same refusal raised before. An ``except`` written
for either parent catches them. In new code, catch the class the problem
calls for, which each one's docstring names.
"""

# Nothing here imports the rest of the package, so the extension can raise
# these by name (crates/online-py/src/lib.rs, `column_not_found`).

from __future__ import annotations

import polars as pl

__all__ = [
    "ColumnNotFoundError",
    "OutputColumnNotFoundError",
    "RefusedValueError",
    "WrongTypeError",
]


class ColumnNotFoundError(pl.exceptions.ColumnNotFoundError, ValueError):
    """The error raised when a column a call reads is not in its input.

    For a column a spec reads, its message names the spec, the role the
    column plays (target, feature, clock, session, weight, group or label)
    and the column, and lists the input's columns. ``po.stream.embargo``,
    ``po.stream.refresh_time`` and ``po.gbdt`` raise it too, for a column
    they name. ``po.stream.with_windows`` raises a plain ``ValueError``
    instead.
    ``po.eval`` raises it for a target column when ``spec`` is the spec
    dict, and a plain ``ValueError`` when ``spec`` is the spec's name
    alone. For a missing ``group``, ``clock``
    or ``weight`` column, ``po.eval`` raises Polars' plain
    ``ColumnNotFoundError``, which is not a ``ValueError``.

    ``po.ColumnNotFoundError`` is a subclass of Polars'
    ``ColumnNotFoundError``, which Polars raises for a column a query names
    that its input does not have, and of ``ValueError``, the class of every
    other value the bank refuses. So either ``except`` catches it.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        try:
            bank.fit_predict(df.drop("x1"))
        except pl.exceptions.ColumnNotFoundError as e:   # or po.ColumnNotFoundError, or ValueError
            print(type(e).__name__, str(e).split(",")[0])

    .. code-block:: text

        ColumnNotFoundError spec "ridge": feature column "x1" is not in the input

    .. rubric:: See also

    - :class:`OutputColumnNotFoundError`, for a spec's output column that a
      scoring function's frame does not have.
    """


class OutputColumnNotFoundError(ColumnNotFoundError, KeyError):
    """The error raised when a spec's output column is not in the frame a scoring function reads.

    It is a :class:`ColumnNotFoundError`, so Polars' own
    ``ColumnNotFoundError`` and ``ValueError`` catch it, as they catch any
    other missing column. It is a ``KeyError`` too, the class this refusal
    raised before. That parent is dropped at the next major version, so
    catch ``ColumnNotFoundError``. Its text, like any ``KeyError``'s, prints
    in quotes.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        try:
            po.eval.metrics(df, "ridge")   # a frame without the bank's output
        except pl.exceptions.ColumnNotFoundError as e:   # or KeyError, until then
            print(type(e).__name__)

    .. code-block:: text

        OutputColumnNotFoundError

    .. rubric:: See also

    - :func:`polars_online.eval.metrics` and the other scoring functions of
      :mod:`polars_online.eval`, which raise it.
    """


class WrongTypeError(TypeError, ValueError):
    """The error raised for a value of the wrong type where ``ValueError`` was raised before.

    The value may be a float where a whole number is needed, one spec where
    a list of them is, a text column where a number is, or a ``UInt128``
    column, which polars-online cannot read. It is a ``TypeError``, the
    class for a value of the wrong type, and a ``ValueError`` too, the
    class most of these refusals raised before. The ``ValueError`` parent
    is dropped at the next major version, so catch ``TypeError``.

    .. rubric:: Examples

    .. code-block:: python

        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)
        try:
            po.ModelBank(spec)                   # one spec, not a list of them
        except TypeError as e:   # or ValueError, until then
            print(type(e).__name__, str(e).split(".")[0])

    .. code-block:: text

        WrongTypeError ModelBank: specs must be a list of specs, but it is one spec

    .. rubric:: See also

    - :class:`RefusedValueError`, its counterpart for a value out of range.
    """


class RefusedValueError(ValueError, TypeError):
    """The error raised for a refused value where ``TypeError`` was raised before.

    The value is out of range, or a combination of settings is refused, such
    as ``reverse=True`` with no ``window_size`` to end the rows read ahead.
    It is a ``ValueError``, the class for a value out of range
    or a combination refused, and a ``TypeError`` too, the class these
    refusals raised before. The ``TypeError`` parent is dropped at the next
    major version, so catch ``ValueError``.

    .. rubric:: Examples

    .. code-block:: python

        try:
            po.ewm_mean("x", half_life=10.0, reverse=True)   # no window_size
        except ValueError as e:   # or TypeError, until then
            print(type(e).__name__, str(e).split(",")[0])

    .. code-block:: text

        RefusedValueError po.ewm_mean: reverse=True needs window_size

    .. rubric:: See also

    - :class:`WrongTypeError`, its counterpart for a value of the wrong
      type.
    """
