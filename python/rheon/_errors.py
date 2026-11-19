"""This library's own exception classes.

Each is a subclass of the one class the problem calls for, which its docstring
names: :class:`WrongTypeError` is a ``TypeError``, :class:`RefusedValueError` a
``ValueError`` and :class:`OutputColumnNotFoundError` a
:class:`ColumnNotFoundError`.
"""

# Nothing here imports the rest of the package, so the extension can raise
# these by name (crates/rheon-py/src/lib.rs, `column_not_found`).

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
    and the column, and lists the input's columns. ``rp.stream.embargo``
    and ``rp.stream.refresh_time`` raise it too, for a column they name.
    ``rp.stream.with_windows`` raises a plain ``ValueError`` instead.
    ``rp.eval`` raises it for a missing ``group``, ``clock`` or ``weight``
    column, and for a target column when ``spec`` is the spec dict; a
    plain ``ValueError`` when ``spec`` is the spec's name alone.

    ``rp.ColumnNotFoundError`` is a subclass of Polars'
    ``ColumnNotFoundError``, which Polars raises for a column a query names
    that its input does not have, and of ``ValueError``, the class of every
    other value the bank refuses. So either ``except`` catches it.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        spec = rp.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = rp.ModelBank([spec])
        try:
            bank.fit_predict(df.drop("x1"))
        except pl.exceptions.ColumnNotFoundError as e:   # or rp.ColumnNotFoundError, or ValueError
            print(type(e).__name__, str(e).split(",")[0])

    .. code-block:: text

        ColumnNotFoundError spec "ridge": feature column "x1" is not in the input

    .. rubric:: See also

    - :class:`OutputColumnNotFoundError`, for a spec's output column that a
      scoring function's frame does not have.
    """


class OutputColumnNotFoundError(ColumnNotFoundError):
    """The error raised when a spec's output column is not in the frame a scoring function reads.

    It is a :class:`ColumnNotFoundError`, so Polars' own
    ``ColumnNotFoundError`` and ``ValueError`` catch it, as they catch any
    other missing column.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        try:
            rp.eval.metrics(df, "ridge")   # a frame without the bank's output
        except pl.exceptions.ColumnNotFoundError as e:
            print(type(e).__name__)

    .. code-block:: text

        OutputColumnNotFoundError

    .. rubric:: See also

    - :func:`rheon.polars.eval.metrics` and the other scoring functions of
      :mod:`rheon.polars.eval`, which raise it.
    """


class WrongTypeError(TypeError):
    """The error raised for a value of the wrong type.

    The value may be a float where a whole number is needed, one spec where
    a list of them is, a text column where a number is, or a ``UInt128``
    column, which polars-online cannot read. It is a ``TypeError``, the
    class for a value of the wrong type.

    .. rubric:: Examples

    .. code-block:: python

        spec = rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)
        try:
            rp.ModelBank(spec)                   # one spec, not a list of them
        except TypeError as e:
            print(type(e).__name__, str(e).split(".")[0])

    .. code-block:: text

        WrongTypeError ModelBank: specs must be a list of specs, but it is one spec

    .. rubric:: See also

    - :class:`RefusedValueError`, its counterpart for a value out of range.
    """


class RefusedValueError(ValueError):
    """The error raised for a refused value.

    The value is out of range, or a combination of settings is refused, such
    as ``reverse=True`` with no ``window_size`` to end the rows read ahead.
    It is a ``ValueError``, the class for a value out of range
    or a combination refused.

    .. rubric:: Examples

    .. code-block:: python

        try:
            rp.ewm_mean("x", half_life=10.0, reverse=True)   # no window_size
        except ValueError as e:
            print(type(e).__name__, str(e).split(",")[0])

    .. code-block:: text

        RefusedValueError po.ewm_mean: reverse=True needs window_size

    .. rubric:: See also

    - :class:`WrongTypeError`, its counterpart for a value of the wrong
      type.
    """
