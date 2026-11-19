"""The Polars-parented :class:`UnstableWarning` that ``rheon.polars`` issues.

It is a subclass of ``rheon``'s :class:`~rheon.UnstableWarning` and of
Polars' own ``UnstableWarning``, so a filter on either class, or on
``polars.exceptions.UnstableWarning``, covers this library's unstable parts.
Every other warning of this library needs no Polars parent and is
``rheon``'s own.
"""

# Importing this module is what turns `rheon._warnings.warn_unstable` to this
# class: `rheon.polars` imports it first.

from __future__ import annotations

from polars.exceptions import UnstableWarning as _PolarsUnstableWarning

from rheon import _warnings

__all__ = ["UnstableWarning"]


class UnstableWarning(_warnings.UnstableWarning, _PolarsUnstableWarning):
    """A warning that a part of the API not yet promised to stay the same was used.

    It is issued only when the environment variable
    ``RHEON_WARN_UNSTABLE`` is ``1``, as Polars issues its own
    ``UnstableWarning`` under ``POLARS_WARN_UNSTABLE``. It is a subclass of
    :class:`rheon.UnstableWarning` and of Polars' class, so a filter on
    ``polars.exceptions.UnstableWarning`` covers this library's unstable
    parts as well. :class:`rheon.UnstableWarning` lists them.

    .. rubric:: Examples

    .. code-block:: python

        print(issubclass(rp.UnstableWarning, pl.exceptions.UnstableWarning))

    .. code-block:: text

        True

    .. rubric:: See also

    - :class:`rheon.RheonWarning`, the base class every warning of
      this library shares.
    """
