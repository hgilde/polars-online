"""A stub of rheon.polars: ModelBank, a few modules, and the `.rheon` namespace.

Its layout is the real one. As in rheon 2.0.0 there is no public ``gbdt``,
only the private ``_gbdt``, and ``gram``, ``corr`` and ``spec`` are the
top-level modules re-exported.
"""

import polars as pl

from rheon import corr, gram, spec  # noqa: F401
from rheon._warnings import RheonDeprecationWarning, RheonWarning
from rheon.polars import _bank, eval, ops, sim, stream  # noqa: F401
from rheon.polars._bank import ModelBank, thread_pool_size

__version__ = "2.0.0"
__all__ = [
    "ModelBank",
    "RheonDeprecationWarning",
    "RheonWarning",
    "__version__",
    "corr",
    "eval",
    "gram",
    "ops",
    "sim",
    "spec",
    "stream",
    "thread_pool_size",
]


class _Rheon:
    def __init__(self, frame):
        self._frame = frame

    def fit(self, alpha=0.5):
        return ModelBank(alpha).fit(self._frame)


for _register in (pl.api.register_dataframe_namespace, pl.api.register_lazyframe_namespace):
    _register("rheon")(_Rheon)
