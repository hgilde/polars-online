"""Names the helper modules renamed before 1.0, each refused by its old name.

Task 144's rule (docs/PLAN.md): an old name is not an alias. It is refused,
and the message names the new one, so a caller learns the new spelling from
the error rather than from a changelog. A spec's keyword goes through
``_spec._RENAMED``; a helper function's keyword goes through
:func:`renamed_keywords` on that function, and a renamed function leaves a
stub from :func:`renamed_function` under its old name (task 197). A keyword
for something removed, such as ``po.target``'s ``relative_to`` (task 201),
is refused by :func:`removed_keywords`, saying what replaces it.

These are the refusals every module uses: the stream functions'
``_spec._renamed_keywords`` (``chunk_size``, task 196) is
:func:`renamed_keywords`. From 1.0 a rename is forwarded instead, with a
warning, until the next major version refuses it here:
:mod:`polars_online._warnings` keeps that table and its warning.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping
from typing import NoReturn

from polars_online._warnings import RENAMED_MARK


def renamed_keywords[**P, R](
    who: str, table: Mapping[str, str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator refusing each keyword of ``table`` by name, naming the
    keyword that replaced it, before ``fn`` sees the call. The signature
    ``inspect`` and the reference show is ``fn``'s own (``functools.wraps``)."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for old, new in table.items():
                if old in kwargs:
                    raise TypeError(f"{who}: {old} was renamed {new}")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def removed_keywords[**P, R](
    who: str, keywords: Iterable[str], why: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator refusing each of ``keywords``, which named something the
    library no longer has, with ``why``: what replaces it. As
    :func:`renamed_keywords`, the signature shown is ``fn``'s own, so the
    removed keywords are in neither the reference nor the API snapshot."""
    gone = tuple(keywords)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if any(k in kwargs for k in gone):
                raise TypeError(f"{who}: {why}")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def renamed_function(old: str, new: str, more: str = "") -> Callable[..., NoReturn]:
    """A stub for a function renamed ``new``, raising ``TypeError`` naming it
    (and ``more``, what else the rename changed) whatever it is called with.
    It is kept out of its module's ``__all__``, so the reference and the API
    snapshot show the new name only."""

    def stub(*args: object, **kwargs: object) -> NoReturn:
        raise TypeError(f"{old} was renamed {new}{more}")

    stub.__name__ = stub.__qualname__ = old.rsplit(".", 1)[-1]
    stub.__doc__ = f"Renamed {new}: calling it raises TypeError, naming the new name."
    setattr(stub, RENAMED_MARK, new)
    return stub


def is_renamed(obj: object) -> bool:
    """Whether ``obj`` is kept only under an old name: a refusal stub from
    :func:`renamed_function` or :func:`renamed_method`, or a forwarding from
    :func:`polars_online._warnings.deprecated_alias`. The API snapshot and the
    reference leave such a name out, so both show the new name only."""
    target = obj.fget if isinstance(obj, property) else obj
    return getattr(target, RENAMED_MARK, None) is not None


def renamed_method(cls: str, old: str, new: str) -> Callable[..., NoReturn]:
    """A method stub for a method renamed ``new``, raising ``AttributeError``
    naming it, as a missing attribute does. Marked, as
    :func:`renamed_function`'s stubs are, so the reference and the API
    snapshot show the new name only."""

    def stub(self: object, *args: object, **kwargs: object) -> NoReturn:
        raise AttributeError(f"{cls}.{old} was renamed {new}")

    stub.__name__ = old
    stub.__qualname__ = f"{cls}.{old}"
    stub.__doc__ = f"Renamed :meth:`{new}`: calling it raises AttributeError, naming the new name."
    setattr(stub, RENAMED_MARK, new)
    return stub
