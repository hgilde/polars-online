"""A stub of rheon 2.0.0's top level: neutral, and it must not import polars.

The layout is the real one (``python/rheon`` in rheon 2.0.0): what needs no
Polars is a module here, what does is under ``rheon.polars``.
"""

from rheon import corr, gram, spec  # noqa: F401
from rheon._warnings import RheonDeprecationWarning, RheonWarning

__version__ = "2.0.0"
__all__ = ["RheonDeprecationWarning", "RheonWarning", "__version__", "corr", "gram", "spec"]
