"""The warnings of the library's promise: a name on its way out, and a surface
that is not yet promised (docs/PLAN.md task 198; review round 4, D2 and D5).

Nothing here imports the rest of the package, so every module can use it.
"""

from __future__ import annotations

import functools
import os
import threading
import warnings
from collections.abc import Callable, Mapping
from typing import Any

#: The environment variable that turns :class:`UnstableWarning` on, as
#: ``POLARS_WARN_UNSTABLE`` does Polars' own: ``1`` and nothing else.
UNSTABLE_VAR = "POLARS_ONLINE_WARN_UNSTABLE"


class PolarsOnlineDeprecationWarning(DeprecationWarning):
    """A name renamed after 1.0, used by its old spelling.

    The old name still works -- read as the new one -- until the next major
    version, which refuses it naming the new one; the message names both.
    A rename made before 1.0 was never forwarded: the old name is refused
    outright (``TypeError`` from a builder, ``ValueError`` from a spec dict),
    so no name warns today. Python shows a ``DeprecationWarning`` raised from
    a library only in ``__main__`` and under test runners; to see it
    everywhere, ``warnings.simplefilter("default",
    polars_online.PolarsOnlineDeprecationWarning)``, and to make it an error,
    ``"error"``.
    """


class UnstableWarning(UserWarning):
    """A surface that is not yet promised, used.

    Raised only when the environment variable ``POLARS_ONLINE_WARN_UNSTABLE``
    is ``1``, as Polars raises its own ``UnstableWarning`` under
    ``POLARS_WARN_UNSTABLE``. Each such surface says so in its docstring. It
    may change in any release without that counting as a breaking change:
    the state file of :func:`polars_online.stream.with_windows`, the formula
    tree's written form (a formula target in a TOML file or a saved state),
    :meth:`polars_online.ModelBank.fit_predict_arrow`,
    :meth:`polars_online.ModelBank.predict_arrow` and
    :class:`polars_online.ArrowStruct`, and the modules
    :mod:`polars_online.sim` and :mod:`polars_online.corr`. Everything else in
    ``__all__`` is promised.
    """


def _unstable_on() -> bool:
    return os.environ.get(UNSTABLE_VAR, "") == "1"


def warn_unstable(what: str, *, stacklevel: int = 3) -> None:
    """An :class:`UnstableWarning` about ``what``, when the variable is ``1``.
    ``stacklevel`` counts from the caller of this function: 3 is the caller's
    caller, the user's line under a wrapper."""
    if not _unstable_on():
        return
    warnings.warn(
        f"{what} is considered unstable. It may be changed at any point without it being "
        f"considered a breaking change (set {UNSTABLE_VAR}=0 or unset it to silence this).",
        UnstableWarning,
        stacklevel=stacklevel,
    )


#: How deep this thread is in calls to functions labelled unstable, so that
#: only the outermost warns: one of a module's functions calling another is
#: one call of the user's.
_inside = threading.local()


def unstable[**P, R](what: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A function that raises :class:`UnstableWarning` about ``what`` on each
    call, under the variable, and otherwise runs as it is. A call made from
    inside another labelled call does not warn again."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            depth = getattr(_inside, "depth", 0)
            if depth == 0:
                warn_unstable(what)
            _inside.depth = depth + 1
            try:
                return fn(*args, **kwargs)
            finally:
                _inside.depth = depth

        return wrapper

    return decorate


def forward_deprecated(
    who: str,
    given: Mapping[str, Any],
    table: Mapping[str, str],
    *,
    stacklevel: int = 3,
) -> dict[str, Any]:
    """``given`` with each name ``table`` deprecates read as its new name,
    and a :class:`PolarsOnlineDeprecationWarning` for each. Refuses an old
    name beside its new one, naming both."""
    out = dict(given)
    for old, new in table.items():
        if old not in out:
            continue
        if new in out:
            raise TypeError(f"{who}: {old} was renamed {new}, and both are given: give {new} alone")
        warnings.warn(
            f"{who}: {old} is deprecated: it was renamed {new}. It is read as {new} until the "
            f"next major version, which refuses it",
            PolarsOnlineDeprecationWarning,
            stacklevel=stacklevel,
        )
        out[new] = out.pop(old)
    return out
