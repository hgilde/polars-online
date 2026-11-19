"""Streaming and online regression models.

The names that need no dataframe library: the model builders
(:mod:`rheon.spec`), the Gram tools (:mod:`rheon.gram`), the correlation
tools (:mod:`rheon.corr`), the library's errors and warnings, and the
versions. The Polars interface (the model bank, the frame namespaces, the
window operators and the scoring functions) is :mod:`rheon.polars`, which
re-exports every name here.
"""

# ruff: noqa: E402  (the imports below follow the CPU check on purpose)
# First, before anything loads the compiled module: a CPU that lacks a feature
# the wheel was built for gets a warning here, where the alternative is an
# illegal instruction with no Python frame to say why (rheon._cpu_check).
from rheon import _cpu_check

_cpu_check.run()

from rheon import corr, gram, spec
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

__version__ = "1.6.0"

__all__ = [
    "ArrowStruct",
    "ColumnNotFoundError",
    "FormulaTarget",
    "OutputColumnNotFoundError",
    "PolarsOnlineDeprecationWarning",
    "PolarsOnlineWarning",
    "ReadinessWarning",
    "RefusedValueError",
    "Target",
    "UnstableWarning",
    "WrongTypeError",
    "__version__",
    "corr",
    "gram",
    "native_version",
    "schema_version",
    "spec",
    "target",
    "thread_pool_size",
]
