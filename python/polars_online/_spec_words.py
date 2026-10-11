"""The words a builder's parameter takes, read off its ``Literal``
annotation, and the one refusal of a word outside them: a part of
:mod:`polars_online._spec` moved to a file of its own, keeping that one
under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Sequence
from typing import Any, Literal


def _one_of(words: Sequence[str]) -> str:
    """``one of "a", "b"``: how every refusal names the words a parameter
    takes, here and on the Rust side (``online_polars::words_refusal``)."""
    return "one of " + ", ".join(json.dumps(w) for w in words)


def _bad_word(v: Any, hint: Any) -> tuple[str, tuple[str, ...]] | None:
    """The word ``v`` gives outside the words ``hint`` takes, and those
    words, where ``v`` has the shape the hint asks for but a word it does
    not take: a str for a ``Literal``, an entry of a list of them, or a key
    of a dict keyed by them. ``None`` where the shape itself is wrong, which
    is a ``TypeError``, not a word outside a list."""
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        for a in typing.get_args(hint):
            if (found := _bad_word(v, a)) is not None:
                return found
        return None
    if origin is Literal:
        words = typing.get_args(hint)
        if isinstance(v, str) and v not in words:
            return v, words
        return None
    if origin in (list, Sequence) and isinstance(v, (list, tuple)):
        (inner,) = typing.get_args(hint)
        if typing.get_origin(inner) is Literal and all(isinstance(x, str) for x in v):
            words = typing.get_args(inner)
            return next(((x, words) for x in v if x not in words), None)
        return None
    if origin is dict and isinstance(v, dict):
        key, _ = typing.get_args(hint)
        if typing.get_origin(key) is Literal and all(isinstance(k, str) for k in v):
            words = typing.get_args(key)
            return next(((k, words) for k in v if k not in words), None)
    return None


#: The words renamed before 1.0, refused naming the new one: ``ew_cov``'s
#: statistic, the twin of the Rust side's ``RENAMED_VALUES`` (whose other
#: entry is a model ``type``, which no builder takes).
_RENAMED_WORDS = {"lagcorr": "lag_corr"}

#: Why a parameter takes the words it does, where a caller would ask: said
#: after the refusal, as the Rust side says it.
_WORD_REASONS = {
    ("rcov", "kernel"): (
        "the Bartlett kernel is not consistent for this estimator, and Parzen beats "
        "the quadratic spectral on efficiency (BNHLS 2011)"
    ),
}


#: The words a parameter refuses for a reason of its own, said in place
#: of the words it takes: a windowed model's ``closed`` takes two of Polars'
#: four, and the Rust side says why the other two cannot be.
_CLOSED_OUT = (
    'closed = "{word}" leaves out the current row, the one the window ends at; a model '
    "reads its fit after it has learned that row, so its window always holds it: give "
    '"right" (the default) or "both"'
)
_REFUSED_FOR_A_REASON = {("closed", "left"): _CLOSED_OUT, ("closed", "none"): _CLOSED_OUT}


def word_reason(who: str, key: str, got: str) -> str | None:
    """The refusal of ``got`` for ``key`` where it has a reason of its own,
    as the Rust side words it; ``None`` for any other word."""
    why = _REFUSED_FOR_A_REASON.get((key, got))
    return f"{who}: {why.format(word=got)}" if why else None


def words_refusal(who: str, key: str, got: str, words: Sequence[str], hint: Any = None) -> str:
    """The one refusal of a word outside the words a parameter takes
    (task 244, P-4): ``spec "m": <key> must be one of "a", "b"; got "x"``,
    the shape the Rust side gives a spec dict and a TOML file too. A
    parameter that also takes something other than a word (``hint``, its
    annotation) names that first: ``an int or one of "auto"``."""
    others = [a for a in typing.get_args(hint) if a in (int, float)]
    kinds = " or ".join({int: "an int", float: "a number"}[a] for a in others)
    taken = f"{kinds} or {_one_of(words)}" if kinds else _one_of(words)
    # A word renamed before 1.0 is named, as the Rust side names it
    # (`online_polars::RENAMED_VALUES`).
    renamed = f"; {got} was renamed {_RENAMED_WORDS[got]}" if got in _RENAMED_WORDS else ""
    return f"{who}: {key} must be {taken}; got {json.dumps(got)}{renamed}"
