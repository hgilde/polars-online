"""The exception classes of this library's own.

Nothing here imports the rest of the package, so the extension can raise
these by name (``crates/online-py/src/lib.rs``, ``column_not_found``).
"""

from __future__ import annotations

import polars as pl

__all__ = ["ColumnNotFoundError"]


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
