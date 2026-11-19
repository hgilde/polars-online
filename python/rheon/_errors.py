"""This library's own exception classes.

Each is a subclass of the one class the problem calls for, which its docstring
names: :class:`WrongTypeError` is a ``TypeError``, :class:`RefusedValueError` a
``ValueError`` and :class:`OutputColumnNotFoundError` a
:class:`ColumnNotFoundError`.

These need no Polars. ``rheon.polars`` adds a subclass of
:class:`ColumnNotFoundError` and of :class:`OutputColumnNotFoundError` that
is Polars' ``ColumnNotFoundError`` too (``rheon/polars/_errors.py``), and
raises those, so an ``except`` on Polars' class catches what a Polars call
of this library raises.
"""

# Nothing here imports the rest of the package, so the extension can raise
# these by name (crates/rheon-py/src/lib.rs, `errors_class`, through
# `flavoured`).

from __future__ import annotations

import sys
from typing import cast

__all__ = [
    "ColumnNotFoundError",
    "OutputColumnNotFoundError",
    "RefusedValueError",
    "WrongTypeError",
]

#: The module that holds the Polars-parented classes, once `rheon.polars`
#: has imported it.
_POLARS_ERRORS = "rheon.polars._errors"


def flavoured(name: str) -> type[Exception]:
    """The class ``name`` that a raise should use: the Polars-parented one
    of ``rheon.polars`` once that is imported, where it defines one, and
    this module's otherwise. A dict lookup, so the extension can call it on
    every refusal (crates/rheon-py/src/lib.rs, ``errors_class``)."""
    polars_errors = sys.modules.get(_POLARS_ERRORS)
    cls = getattr(polars_errors, name, None) if polars_errors is not None else None
    return cast(type[Exception], cls if cls is not None else globals()[name])


class ColumnNotFoundError(ValueError):
    """The error raised when a column a call reads is not in its input.

    For a column a spec reads, its message names the spec, the role the
    column plays (target, feature, clock, session, weight, group or label)
    and the column, and lists the input's columns. ``rp.stream.embargo``
    and ``rp.stream.refresh_time`` raise it too, for a column they name.
    ``rp.stream.with_windows`` raises a plain ``ValueError`` instead.
    ``rp.eval`` raises it for a missing ``group``, ``clock`` or ``weight``
    column, and for a target column when ``spec`` is the spec dict; a
    plain ``ValueError`` when ``spec`` is the spec's name alone.

    It is a ``ValueError``, the class of every other value the bank
    refuses. ``rp.ColumnNotFoundError``, the class ``rheon.polars`` raises,
    is a subclass of this one and of Polars' ``ColumnNotFoundError``, which
    Polars raises for a column a query names that its input does not have.
    So ``except rheon.ColumnNotFoundError``, ``except
    rp.ColumnNotFoundError``, Polars' class and ``ValueError`` each catch
    what a call of ``rheon.polars`` raises.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        spec = rp.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = rp.ModelBank([spec])
        try:
            bank.fit_predict(df.drop("x1"))
        except rheon.ColumnNotFoundError as e:   # or rp.ColumnNotFoundError, or ValueError
            print(type(e).__name__, str(e).split(",")[0])

    .. code-block:: text

        ColumnNotFoundError spec "ridge": feature column "x1" is not in the input

    .. rubric:: See also

    - :class:`OutputColumnNotFoundError`, for a spec's output column that a
      scoring function's frame does not have.
    """


class OutputColumnNotFoundError(ColumnNotFoundError):
    """The error raised when a spec's output column is not in the frame a scoring function reads.

    It is a :class:`ColumnNotFoundError`, so ``ValueError`` catches it,
    and, from ``rheon.polars``, Polars' own ``ColumnNotFoundError``, as they
    catch any other missing column.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/hgilde/polars-online/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        try:
            rp.eval.metrics(df, "ridge")   # a frame without the bank's output
        except rheon.OutputColumnNotFoundError as e:
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
