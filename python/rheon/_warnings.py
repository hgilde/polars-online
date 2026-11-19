"""The warnings of the library's promise: a name on its way out, and a surface
that is not yet promised. Nothing here imports the rest of the package, so
every module can use it.
"""

from __future__ import annotations

import functools
import os
import threading
import warnings
from collections.abc import Callable

from polars.exceptions import UnstableWarning as _PolarsUnstableWarning

#: The environment variable that turns :class:`UnstableWarning` on, as
#: ``POLARS_WARN_UNSTABLE`` does Polars' own: ``1`` and nothing else.
UNSTABLE_VAR = "POLARS_ONLINE_WARN_UNSTABLE"


class PolarsOnlineWarning(Warning):
    """The base class of every warning this library issues.

    Filter on it to silence or surface every warning of this library at
    once, as Polars' ``PolarsWarning`` covers Polars' own. Each warning keeps
    its standard category as a second parent, so a filter on that category
    goes on working:

    .. list-table::
       :header-rows: 1
       :widths: 55 45

       * - the warning
         - its standard category
       * - :class:`~rheon.ReadinessWarning`,
           :class:`~rheon.OrderNotGuaranteedWarning`,
           :class:`~rheon.ConsumedSourceWarning` and
           :class:`~rheon.UnstableWarning`
         - ``UserWarning``
       * - :class:`~rheon.PolarsOnlineDeprecationWarning`
         - ``DeprecationWarning``

    .. rubric:: Examples

    .. code-block:: python

        import warnings

        warnings.simplefilter("ignore", po.PolarsOnlineWarning)   # every one of them
        print(issubclass(po.ReadinessWarning, po.PolarsOnlineWarning))

    .. code-block:: text

        True
    """


class PolarsOnlineDeprecationWarning(PolarsOnlineWarning, DeprecationWarning):
    """A warning that a name still read is on its way out.

    The library issues it for a name it still reads under a spelling it will
    stop reading at the next major version. It reads no name that way at
    present: a renamed name is refused outright, naming the new one
    (``TypeError`` from a builder or a function, ``ValueError`` from a spec
    dict). The class stays so that a filter on it keeps working when one is.
    Python shows a ``DeprecationWarning`` issued from a library only in
    ``__main__`` and under test runners, so to see it everywhere, filter it
    with ``warnings.simplefilter("default", rheon.PolarsOnlineDeprecationWarning)``,
    and to make it an error, with ``"error"``.

    .. rubric:: Examples

    .. code-block:: python

        import warnings

        warnings.simplefilter("error", po.PolarsOnlineDeprecationWarning)   # an error from here
        print(issubclass(po.PolarsOnlineDeprecationWarning, DeprecationWarning))

    .. code-block:: text

        True

    .. rubric:: See also

    - :class:`PolarsOnlineWarning`, the base class every warning of this
      library shares.
    """


class UnstableWarning(PolarsOnlineWarning, UserWarning, _PolarsUnstableWarning):
    """A warning that a part of the API not yet promised to stay the same was used.

    It is issued only when the environment variable
    ``POLARS_ONLINE_WARN_UNSTABLE`` is ``1``, as Polars issues its own
    ``UnstableWarning`` under ``POLARS_WARN_UNSTABLE``. It is a subclass of
    Polars' class too, so a filter on ``polars.exceptions.UnstableWarning``
    covers this library's unstable parts as well.

    Each unstable part says so in its docstring, and may change in any
    release without that counting as a breaking change:

    - the state file of :func:`rheon.stream.with_windows`;
    - the written form of a formula target's tree, in a TOML file or a
      saved state;
    - :meth:`rheon.ModelBank.fit_predict_arrow`,
      :meth:`rheon.ModelBank.predict_arrow` and
      :class:`rheon.ArrowStruct`;
    - the modules :mod:`rheon.sim`, :mod:`rheon.corr` and
      :mod:`rheon.gbdt`.

    Everything else in ``__all__`` is promised.

    .. rubric:: Examples

    .. code-block:: python

        print(issubclass(po.UnstableWarning, pl.exceptions.UnstableWarning))

    .. code-block:: text

        True

    .. rubric:: See also

    - :class:`PolarsOnlineWarning`, the base class every warning of this
      library shares.
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
        f"{what}: this is unstable, so it may change in any release without that counting "
        f"as a breaking change. Set {UNSTABLE_VAR}=0, or unset it, to silence this warning.",
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


#: The attribute that marks a name kept only to refuse its old spelling: the
#: API snapshot and the reference leave such a name out, so both show the new
#: name only (``_renamed.is_renamed`` reads it).
RENAMED_MARK = "__polars_online_renamed__"
