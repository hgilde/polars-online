"""A stub of rheon.polars: ModelBank, a few modules, and the `.rheon` namespace.

There is no gbdt module here, as in a 2.0.0 that leaves it out.
"""

import polars as pl

from rheon.polars import _bank, eval, gram, ops, sim, stream  # noqa: F401
from rheon.polars._bank import ModelBank, thread_pool_size

__all__ = ["ModelBank", "eval", "gram", "ops", "sim", "stream", "thread_pool_size"]


class _Rheon:
    def __init__(self, frame):
        self._frame = frame

    def fit(self, alpha=0.5):
        return ModelBank(alpha).fit(self._frame)


for _register in (pl.api.register_dataframe_namespace, pl.api.register_lazyframe_namespace):
    _register("rheon")(_Rheon)
