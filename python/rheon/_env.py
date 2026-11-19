"""The one reader of the library's environment variables.

Each is read by its ``RHEON_`` name. The name it had before the library was
renamed, ``POLARS_ONLINE_...``, is read while the new one is not set at all
(an empty value is set), and a :class:`~rheon.RheonDeprecationWarning` says
so once per process, naming the new variable. When both are set the new one
wins and nothing is said. The old names are read until the next major
release. The native side reads the same pairs (``rheon_polars::env``) and
says nothing: this module warns, once, for the process.

Nothing here imports the rest of the package at import time, so every module
can use it.
"""

from __future__ import annotations

import os
import warnings

#: Each variable the library reads, by its current name, with the name it had
#: before the rename.
RENAMED: dict[str, str] = {
    "RHEON_MAX_THREADS": "POLARS_ONLINE_MAX_THREADS",
    "RHEON_SKIP_CPU_CHECK": "POLARS_ONLINE_SKIP_CPU_CHECK",
    "RHEON_TIMING": "POLARS_ONLINE_TIMING",
    "RHEON_WARN_UNSTABLE": "POLARS_ONLINE_WARN_UNSTABLE",
}

#: The old names this process has already warned about.
_warned: set[str] = set()


def get(name: str) -> str | None:
    """The value of ``name`` (a key of :data:`RENAMED`), or ``None`` when
    neither it nor its old name is set. The old name is read only when ``name``
    is not set, and then warns once."""
    value = os.environ.get(name)
    if value is not None:
        return value
    old = RENAMED[name]
    value = os.environ.get(old)
    if value is not None and old not in _warned:
        _warned.add(old)
        _warn(name, old)
    return value


def warn_old_names() -> None:
    """Read every variable once, so that each old name set without its new one
    warns now, at ``import rheon``, and not at the first call that reads it."""
    for name in RENAMED:
        get(name)


def _warn(name: str, old: str) -> None:
    from rheon._warnings import RheonDeprecationWarning, _user_stacklevel

    warnings.warn(
        f"rheon: the environment variable {old} was renamed {name}. The old name is read for now, "
        f"and will not be read from the next major release. Set {name} instead.",
        RheonDeprecationWarning,
        stacklevel=_user_stacklevel(),
    )
