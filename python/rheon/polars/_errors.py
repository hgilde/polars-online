"""The Polars-parented exception classes ``rheon.polars`` raises.

:class:`ColumnNotFoundError` and :class:`OutputColumnNotFoundError` here are
subclasses of ``rheon``'s classes of the same names and of Polars' own
``ColumnNotFoundError``, so ``except rheon.ColumnNotFoundError``, ``except
rp.ColumnNotFoundError``, ``except pl.exceptions.ColumnNotFoundError`` and
``except ValueError`` each catch what a Polars call of this library raises.
``WrongTypeError`` and ``RefusedValueError`` need no Polars parent, and are
``rheon``'s own, the same objects under both names.
"""

# Importing this module is what turns `rheon._errors.flavoured` to these
# classes: `rheon.polars` imports it first, so the extension raises them on
# every Polars path (crates/rheon-py/src/lib.rs, `errors_class`).

from __future__ import annotations

import polars as pl

from rheon import _errors
from rheon._errors import RefusedValueError, WrongTypeError

__all__ = [
    "ColumnNotFoundError",
    "OutputColumnNotFoundError",
    "RefusedValueError",
    "WrongTypeError",
]


class ColumnNotFoundError(_errors.ColumnNotFoundError, pl.exceptions.ColumnNotFoundError):
    """The error raised when a column a call reads is not in its input.

    For a column a spec reads, its message names the spec, the role the
    column plays (target, feature, clock, session, weight, group or label)
    and the column, and lists the input's columns. ``rp.stream.embargo``
    and ``rp.stream.refresh_time`` raise it too, for a column they name.
    ``rp.stream.with_windows`` raises a plain ``ValueError`` instead.
    ``rp.eval`` raises it for a missing ``group``, ``clock`` or ``weight``
    column, and for a target column when ``spec`` is the spec dict; a
    plain ``ValueError`` when ``spec`` is the spec's name alone.

    ``rp.ColumnNotFoundError`` is a subclass of :class:`rheon.ColumnNotFoundError`,
    a ``ValueError``, the class of every other value the bank refuses, and
    of Polars' ``ColumnNotFoundError``, which Polars raises for a column a
    query names that its input does not have. So any of the four
    ``except`` catches it.

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


class OutputColumnNotFoundError(ColumnNotFoundError, _errors.OutputColumnNotFoundError):
    """The error raised when a spec's output column is not in the frame a scoring function reads.

    It is a :class:`ColumnNotFoundError`, so Polars' own
    ``ColumnNotFoundError`` and ``ValueError`` catch it, as they catch any
    other missing column, and a :class:`rheon.OutputColumnNotFoundError`.

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
