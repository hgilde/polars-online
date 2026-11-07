"""The exception classes of this library's own.

Nothing here imports the rest of the package, so the extension can raise
these by name (``crates/online-py/src/lib.rs``, ``column_not_found``).

Three of them carry two parents for a while: the class the problem calls
for, and the class the same refusal raised before. An ``except`` written
for either catches them. The older parent is dropped at the next major
version, so new code catches the class each one's docstring names first.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "ColumnNotFoundError",
    "OutputColumnNotFoundError",
    "RefusedValueError",
    "WrongTypeError",
]


class ColumnNotFoundError(pl.exceptions.ColumnNotFoundError, ValueError):
    """A column a spec reads is not in the frame.

    The message names the spec, the role the column plays (target, feature,
    clock, session, weight or group) and the column, and lists the input's
    columns. It is Polars' own ``ColumnNotFoundError``, as Polars raises for
    a column a query names that its input has not got, and a ``ValueError``
    too, the class every other value the bank refuses raises. So either
    ``except`` catches it:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        try:
            bank.fit_predict(df.drop("x1"))
        except pl.exceptions.ColumnNotFoundError as e:   # or po.ColumnNotFoundError, or ValueError
            print(e)   # spec "ridge": feature column "x1" not found; the input has columns [...]
    """


class OutputColumnNotFoundError(ColumnNotFoundError, KeyError):
    """A spec's output column is not in the frame a scoring function reads.

    It is a :class:`ColumnNotFoundError`, so Polars' own
    ``ColumnNotFoundError`` and ``ValueError`` catch it, as they catch any
    other missing column. It is a ``KeyError`` too, the class this refusal
    raised before; that parent is dropped at the next major version, so
    catch ``ColumnNotFoundError``. Its text, like any ``KeyError``'s, prints
    in quotes.

    .. rubric:: Examples

    .. code-block:: python

        try:
            po.eval.metrics(df, "ridge")   # a frame without the bank's output
        except pl.exceptions.ColumnNotFoundError as e:   # or KeyError, until then
            print(e)
    """


class WrongTypeError(TypeError, ValueError):
    """A value or a column of the wrong type, such as a float where a whole
    number is needed, one spec where a list of them is, or a text column
    where a number is.

    It is a ``TypeError``, the class for a value of the wrong type, and a
    ``ValueError`` too, the class most of these refusals raised before. The
    ``ValueError`` parent is dropped at the next major version, so catch
    ``TypeError``.

    .. rubric:: Examples

    .. code-block:: python

        try:
            po.ModelBank(po.spec.ewridge("m", targets=["y"], features=["x"], half_life=50.0))
        except TypeError as e:   # or ValueError, until then
            print(e)   # ModelBank: specs must be a list of specs, but it is one spec. ...
    """


class RefusedValueError(ValueError, TypeError):
    """A value out of its range, or a combination of settings refused, such
    as ``reverse=True`` with no ``window_size`` to end the rows read ahead.

    It is a ``ValueError``, the class for a value out of range or a
    combination refused, and a ``TypeError`` too, the class these refusals
    raised before. The ``TypeError`` parent is dropped at the next major
    version, so catch ``ValueError``.

    .. rubric:: Examples

    .. code-block:: python

        try:
            po.ewm_mean("x", half_life=10.0, reverse=True)   # no window_size
        except ValueError as e:   # or TypeError, until then
            print(e)   # po.ewm_mean: reverse=True needs window_size, but it is not set, ...
    """
