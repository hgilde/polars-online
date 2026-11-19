"""The thread count a bank runs on: the ``threads`` argument checked where it is
given, and the pool size it names. Native calls only, so every module can use it."""

from __future__ import annotations

import operator

from rheon import _rheon as _native


def _threads_refusal(shown: str) -> str:
    """The text that refuses a ``threads`` argument: the Rust side's, word for
    word (``rheon_polars::threads_refusal``), so either side says the same."""
    return (
        f"threads: {shown} is not a number of threads, and 0 is not one thread per core here, "
        "as it is in RHEON_MAX_THREADS. Give an integer of 1 or more, or leave threads "
        "out (None in Python) to take the count from RHEON_MAX_THREADS, else one thread "
        "per core."
    )


def _check_threads(threads: object) -> int | None:
    """``threads`` checked where it is given, before anything is built: ``None``
    or an integer of 1 or more -- anything ``operator.index`` reads as one,
    numpy's integers included (task 269, D-6). A bool, a float or another
    type is a ``TypeError``, and 0, a negative integer or a count above the
    bound one pool may have a ``ValueError``, each with the message the
    native side raises for the same value, and each before any pool is
    built."""
    if threads is None:
        return None
    if isinstance(threads, bool):
        raise TypeError(_threads_refusal(repr(threads)))
    try:
        count = operator.index(threads)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(_threads_refusal(repr(threads))) from None
    if count < 1:
        raise ValueError(_threads_refusal(repr(threads)))
    if count > _native.max_threads():
        raise ValueError(_native.threads_too_many(repr(threads)))
    return count


def thread_pool_size(*, threads: int | None = None) -> int:
    """The number of threads a bank given this ``threads`` argument runs on.

    Use it to see how many threads a bank will use before you build one, or
    what ``RHEON_MAX_THREADS`` gives. Polars' own threads are
    separate, set by ``POLARS_MAX_THREADS``, and ``pl.thread_pool_size()``
    reports those.

    .. rubric:: Parameters

    ``threads``
        The count a bank would be given, as :class:`ModelBank` takes it.
        ``None``, the default, names the process's shared threads, which
        every bank given no ``threads`` runs on. ``RHEON_MAX_THREADS``
        sets their number when they are first used (unset or ``0``: one
        thread per core), and the number never changes after that. This call
        starts them if nothing has yet. An integer ``n`` names ``n`` threads,
        which this call checks without starting them.

    .. rubric:: Returns

    ``int``: the number of threads.

    .. rubric:: Raises

    ``TypeError``
        ``threads`` is not an integer, a bool included.
    ``ValueError``
        ``threads`` is below 1, or above 1024 (or the machine's core count,
        where that is larger); or ``threads`` is ``None`` and
        ``RHEON_MAX_THREADS`` is set to anything but a whole number,
        or to a count past the same bound. More threads than cores run no
        faster, and a pool starts all of its threads at once. A bank refuses
        the same values: a ``threads`` when it is made, and a bad
        ``RHEON_MAX_THREADS`` at its first call that runs. Both are
        checked before any thread starts.

    .. rubric:: Examples

    .. code-block:: python

        print(rp.thread_pool_size(threads=2))   # 2: checked, and no thread starts

    .. code-block:: text

        2

    .. rubric:: See also

    - :class:`ModelBank`, whose ``threads`` argument this reads.
    """
    return int(_native.thread_pool_size(_check_threads(threads)))
