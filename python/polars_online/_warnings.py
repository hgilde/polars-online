"""The warnings of the library's promise: a name on its way out, and a surface
that is not yet promised (docs/PLAN.md task 198; review round 4, D2 and D5).

The rename mechanism has two halves: a name renamed before the API was
settled for 1.0 is refused, naming the new one (:mod:`polars_online._renamed`),
and one renamed since (task 244 on) is forwarded here, with a warning, until
the next major version moves it to the refusals. Nothing here imports the
rest of the package, so every module can use it.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from polars.exceptions import UnstableWarning as _PolarsUnstableWarning

#: The parameters renamed after 1.0, old -> new: a builder's keyword or a
#: spec dict's key under the old name is read as the new one, with a
#: :class:`PolarsOnlineDeprecationWarning`, until the next major version
#: moves the entry to ``_spec._RENAMED``, which refuses it naming the new one
#: (docs/PLAN.md task 198; review round 4, D2). Empty: every rename so far
#: was made before 1.0 and stays refused by name
#: (:mod:`polars_online._renamed`). Its twin for a TOML file is the Rust
#: side's ``online_polars::DEPRECATED``, which ``tests/test_deprecation.py``
#: holds equal to this one. A function's own keyword renamed after 1.0
#: forwards through :func:`forward_deprecated` at its top. Read where it is
#: used, as ``_warnings._DEPRECATED``, so one table serves every caller.
_DEPRECATED: dict[str, str] = {}

#: The keys renamed in one model, model type -> old -> new: a builder's
#: keyword, or a key of a model table whose ``type`` is the model's, is read
#: as the new name with a :class:`PolarsOnlineDeprecationWarning` until the
#: next major version refuses it. A model's own table, not
#: :data:`_DEPRECATED`, because one old name went two ways (``kmeans``' ``k``
#: is ``n_clusters``, ``hmm``'s ``n_components``) and because a name another
#: model still takes, or a feature called ``k`` in a table keyed by feature,
#: must not move. The names taken at 1.0 (docs/PLAN.md task 244): sklearn's
#: for the mixture and clustering models, and one name per concept across the
#: models. Its twin is the Rust side's ``online_polars::DEPRECATED_IN``, which
#: ``tests/test_deprecation.py`` holds equal to this one.
_DEPRECATED_IN: dict[str, dict[str, str]] = {
    "kmeans": {"k": "n_clusters", "seed_rule": "init"},
    "hmm": {"k": "n_components", "seed_rule": "init", "covariance": "covariance_type"},
    "ew_class": {"covariance": "covariance_type", "window_every": "snapshot_every"},
    "ewridge": {"window_every": "snapshot_every"},
    "lasso": {"window_every": "snapshot_every", "lasso_path": "penalties"},
    "ew_cov": {"window_every": "snapshot_every", "max_rows_between_pca": "max_rows_between_pcas"},
    "marginal": {"window_every": "snapshot_every"},
    "bocpd": {
        "prior_mean": "mean_prior",
        "prior_kappa": "mean_precision_prior",
        "prior_nu": "degrees_of_freedom_prior",
        "prior_scale": "covariance_prior",
    },
    "corrchange": {"span_rows": "window_rows"},
}

#: The words renamed in one model's parameter, model type -> key -> old ->
#: new, read as :data:`_DEPRECATED_IN` reads a key. Its twin is
#: ``online_polars::DEPRECATED_WORDS_IN``.
_DEPRECATED_WORDS_IN: dict[str, dict[str, dict[str, str]]] = {
    "kmeans": {"init": {"kmeanspp": "k-means++"}},
    "hmm": {
        "init": {"kmeanspp": "k-means++"},
        "covariance_type": {"shared": "tied", "diagonal": "diag"},
    },
    "ew_class": {"covariance_type": {"shared": "tied", "diagonal": "diag"}},
}

#: ``bocpd``'s ``emission`` word -> its ``covariance_type`` and whether it
#: was the tempered predictive, which ``robust_beta`` turns on now (0.1, its
#: default then, where none is given).
_EMISSION = {"gaussian": ("full", False), "diag": ("diag", False), "robust": ("diag", True)}

#: The environment variable that turns :class:`UnstableWarning` on, as
#: ``POLARS_WARN_UNSTABLE`` does Polars' own: ``1`` and nothing else.
UNSTABLE_VAR = "POLARS_ONLINE_WARN_UNSTABLE"


class PolarsOnlineWarning(Warning):
    """The base of every warning this library raises.

    Each warning keeps the standard category it belongs to as well:
    :class:`~polars_online.ReadinessWarning`,
    :class:`~polars_online.OrderNotGuaranteedWarning`,
    :class:`~polars_online.ConsumedSourceWarning` and
    :class:`~polars_online.UnstableWarning` are each a ``UserWarning``, and
    :class:`~polars_online.PolarsOnlineDeprecationWarning` is a
    ``DeprecationWarning``. So one filter covers every warning of this
    library, as Polars' ``PolarsWarning`` covers Polars' own, and a filter
    on the standard category goes on working:

    .. code-block:: python

        import warnings

        warnings.simplefilter("ignore", po.PolarsOnlineWarning)   # every one of them
    """


class PolarsOnlineDeprecationWarning(PolarsOnlineWarning, DeprecationWarning):
    """A renamed name, used by its old spelling.

    The old name still works -- read as the new one -- until the next major
    version, which refuses it naming the new one; the message names both.
    A spec parameter goes through a forwarding table, at any depth of a
    spec. Any other renamed name ships a forwarding of its own that raises
    this warning: a check at the function's top, a stub, an alias or a
    second column. That covers another function's keyword, a function, a
    word a parameter takes, a flag, an environment variable, an output field
    and a frame column.

    The names renamed as the API was settled for 1.0 are forwarded this way,
    and the changelog lists them. The renames made earlier were never
    forwarded: each old name is refused outright, naming the new one
    (``TypeError`` from a builder, ``ValueError`` from a spec dict).

    Python shows a ``DeprecationWarning`` raised from a library only in
    ``__main__`` and under test runners; to see it everywhere,
    ``warnings.simplefilter("default",
    polars_online.PolarsOnlineDeprecationWarning)``, and to make it an error,
    ``"error"``.
    """


class UnstableWarning(PolarsOnlineWarning, UserWarning, _PolarsUnstableWarning):
    """A surface that is not yet promised, used.

    Raised only when the environment variable ``POLARS_ONLINE_WARN_UNSTABLE``
    is ``1``, as Polars raises its own ``UnstableWarning`` under
    ``POLARS_WARN_UNSTABLE``. It is a subclass of Polars' class too, so a
    filter on ``polars.exceptions.UnstableWarning`` covers this library's
    unstable surfaces as well. Each such surface says so in its docstring,
    and may change in any release without that counting as a breaking
    change:

    - the state file of :func:`polars_online.stream.with_windows`;
    - the formula tree's written form (a formula target in a TOML file or a
      saved state);
    - :meth:`polars_online.ModelBank.fit_predict_arrow`,
      :meth:`polars_online.ModelBank.predict_arrow` and
      :class:`polars_online.ArrowStruct`;
    - the modules :mod:`polars_online.sim`, :mod:`polars_online.corr` and
      :mod:`polars_online.gbdt`.

    Everything else in ``__all__`` is promised.
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


def forward_model(
    who: str,
    kind: str,
    given: Mapping[str, Any],
    *,
    stacklevel: int = 3,
) -> dict[str, Any]:
    """``given``, a builder's keywords or a model table of type ``kind``,
    with each key and word renamed in that model read as the new one
    (:data:`_DEPRECATED_IN`, :data:`_DEPRECATED_WORDS_IN`), and the two keys
    that became others with a value of their own: ``bocpd``'s ``emission``
    (``"gaussian"`` is ``covariance_type="full"``, ``"diag"`` is ``"diag"``,
    ``"robust"`` is ``"diag"`` with ``robust_beta``, 0.1 where none is given)
    and ``corrchange``'s ``reset_on_flag`` (``True`` is
    ``change_action="reset"``, ``False`` is ``"flag"``). A
    :class:`PolarsOnlineDeprecationWarning` for each; an old name beside its
    new one is refused, naming both. The Rust side forwards a TOML file's
    the same way (``online_polars::forward_deprecated_in_models``)."""
    out = forward_deprecated(who, given, _DEPRECATED_IN.get(kind, {}), stacklevel=stacklevel + 1)

    def warn(message: str) -> None:
        warnings.warn(
            f"{who}: {message} until the next major version, which refuses it",
            PolarsOnlineDeprecationWarning,
            stacklevel=stacklevel + 1,
        )

    def both(old: str, new: str) -> TypeError:
        return TypeError(f"{who}: {old} was renamed {new}, and both are given: give {new} alone")

    if kind == "bocpd" and "emission" in out:
        if "covariance_type" in out:
            raise both("emission", "covariance_type")
        word = out.pop("emission")
        covariance, tempered = (
            _EMISSION.get(word, (word, False)) if isinstance(word, str) else (word, False)
        )
        out["covariance_type"] = covariance
        robust = ""
        if tempered:
            out.setdefault("robust_beta", 0.1)
            robust = " with robust_beta, 0.1 where none is given"
        warn(
            f"emission = {_quoted(word)} is deprecated: it is covariance_type = "
            f"{_quoted(covariance)}{robust}. It is read so"
        )
    if kind == "corrchange" and "reset_on_flag" in out:
        if "change_action" in out:
            raise both("reset_on_flag", "change_action")
        value = out.pop("reset_on_flag")
        action = "reset" if value is True else "flag"
        out["change_action"] = action if isinstance(value, bool) else value
        warn(
            f"reset_on_flag = {value!r} is deprecated: it is change_action = "
            f"{_quoted(action)}. It is read so"
        )
    for key, words in _DEPRECATED_WORDS_IN.get(kind, {}).items():
        word = out.get(key)
        if isinstance(word, str) and word in words:
            out[key] = words[word]
            warn(
                f"{key} = {_quoted(word)} is deprecated: it was renamed "
                f"{_quoted(words[word])}. It is read as {_quoted(words[word])}"
            )
    return out


def _quoted(v: Any) -> str:
    return f'"{v}"' if isinstance(v, str) else repr(v)


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


#: The attribute that marks a name kept only to forward or refuse its old
#: spelling: the API snapshot and the reference leave such a name out, so
#: both show the new name only (``_renamed.is_renamed`` reads it).
RENAMED_MARK = "__polars_online_renamed__"


def deprecated_keywords[**P, R](
    who: str, table: Mapping[str, str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator reading each old keyword of ``table`` as its new name, with
    a :class:`PolarsOnlineDeprecationWarning` pointed at the caller's line
    (:func:`forward_deprecated`). The signature ``inspect`` and the
    reference show is ``fn``'s own, the new names alone."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if any(old in kwargs for old in table):
                kwargs = forward_deprecated(who, kwargs, table)  # type: ignore[assignment]
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def keyword_only_forwarded[**P, R](
    who: str, names: tuple[str, ...]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator for a function whose ``names``, once positional, became
    keyword-only: a call that still passes them by position has each read as
    its keyword, with a :class:`PolarsOnlineDeprecationWarning` naming it.
    The signature shown is ``fn``'s own, with the names keyword-only."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        params = inspect.signature(fn).parameters.values()
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        n_pos = sum(p.kind in kinds for p in params)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if len(args) > n_pos:
                extra = args[n_pos:]
                if len(extra) > len(names):
                    raise TypeError(
                        f"{who}() takes {n_pos} positional arguments but {len(args)} were given"
                    )
                for name, value in zip(names, extra, strict=False):
                    if name in kwargs:
                        raise TypeError(f"{who}() got multiple values for argument {name!r}")
                    warnings.warn(
                        f"{who}: {name} given by position is deprecated: pass it by keyword, "
                        f"as {name}=... It is read until the next major version, which "
                        "refuses it",
                        PolarsOnlineDeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[name] = value
                args = args[:n_pos]  # type: ignore[assignment]
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def deprecated_alias[**P, R](
    old: str, new: str, more: str = ""
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator for a function kept under its old name ``old`` after it was
    renamed ``new``: each call raises a :class:`PolarsOnlineDeprecationWarning`
    naming the new name (and ``more``, what else the rename changed), then
    runs. The function is marked (:data:`RENAMED_MARK`), so the API snapshot
    and the reference show the new name only."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(
                f"{old} is deprecated: it was renamed {new}{more}. It works as before until "
                "the next major version, which refuses it",
                PolarsOnlineDeprecationWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

        setattr(wrapper, RENAMED_MARK, new)
        return wrapper

    return decorate
