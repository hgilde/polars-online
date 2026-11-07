"""The words a builder's parameter takes, read off its ``Literal``
annotation, and the one refusal of a word outside them; and the parameters
where infinity means something: a part of
:mod:`polars_online._spec` moved to a file of its own, keeping that one
under ``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Sequence
from typing import Any, Literal


def quoted(x: Any) -> str:
    """A name or a word as a message quotes it: in double quotes, as JSON
    writes it, with any letter outside ASCII kept as it is, as the Rust side
    (``{:?}``) keeps it."""
    return json.dumps(x, ensure_ascii=False)


def _listed(items: Sequence[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``: how a refusal lists what a
    parameter takes."""
    *rest, last = items
    return f"{', '.join(rest)} or {last}" if rest else last


def _one_of(words: Sequence[str]) -> str:
    """``"a"``, ``"a" or "b"``, ``"a", "b" or "c"``: how every refusal
    names the words a parameter takes, here and on the Rust side
    (``online_polars::words_listed``)."""
    return _listed([quoted(w) for w in words])


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

#: The refusal of a word for a parameter that takes one word alone, where
#: its model says why, as the Rust side words it (``online_core::RcovCfg``).
_WORD_REASONS = {
    ("rcov", "kernel"): (
        'kernel must be "parzen", the one kernel this model takes, but it is {got}. '
        'Set kernel to "parzen".'
    ),
}


#: The words a parameter refuses for a reason of its own, said beside the
#: words it takes: a model's ``closed`` takes two of Polars' four, and the
#: Rust side says why the other two cannot be.
_CLOSED_OUT = (
    "which leaves out the current row: a hard window ends at the last row the model learned, "
    "so that row is always inside it"
)
_REFUSED_FOR_A_REASON = {("closed", "left"): _CLOSED_OUT, ("closed", "none"): _CLOSED_OUT}


def word_reason(who: str, key: str, got: str) -> str | None:
    """The refusal of ``got`` for ``key`` where it has a reason of its own,
    as the Rust side words it; ``None`` for any other word."""
    why = _REFUSED_FOR_A_REASON.get((key, got))
    if why is None:
        return None
    return (
        f'{who}: {key} must be "right" or "both", but it is {quoted(got)}, {why}. '
        "Use one of those words."
    )


def words_refusal(who: str, key: str, got: str, words: Sequence[str], hint: Any = None) -> str:
    """The one refusal of a word outside the words a parameter takes
    (task 244, P-4), in the shape every message takes:
    ``spec "m": <key> must be "a", "b" or "c", but it is "x". Use one of
    those words.``, the shape the Rust side gives a spec dict and a TOML
    file too (``online_polars::words_body``). A parameter that also takes
    something other than a word (``hint``, its annotation) names that
    first: ``an int or "auto"``."""
    others = [a for a in typing.get_args(hint) if a in (int, float)]
    kinds = [{int: "an int", float: "a number"}[a] for a in others]
    taken = _listed(kinds + [quoted(w) for w in words])
    # A word renamed before 1.0 is told its new name, as the Rust side
    # tells it (`online_polars::RENAMED_VALUES`).
    new = _RENAMED_WORDS.get(got)
    if new is not None and new in words:
        return (
            f"{who}: {key} must be {taken}, but it is {quoted(got)}, the old name of "
            f"{quoted(new)}. Use {quoted(new)}."
        )
    if kinds:
        remedy = "Use one of those."
    elif len(words) == 1:
        remedy = f"Use {quoted(words[0])}."
    else:
        remedy = "Use one of those words."
    return f"{who}: {key} must be {taken}, but it is {quoted(got)}. {remedy}"


def refusal(who: str, problems: Sequence[Exception]) -> Exception:
    """Every problem a builder's check found, as one exception to raise,
    in the shape the Rust side gives a spec's (``render_refusal``): one
    problem is its own message, and several are listed one to a line under
    ``spec "m": N problems.``, the spec named once. A ``TypeError`` if any
    problem is one, since a value of the wrong kind is the first to fix,
    and a ``ValueError`` otherwise."""
    prefix = f"{who}: "
    lines: list[str] = []
    for p in problems:
        line = str(p).removeprefix(prefix)
        if line not in lines:
            lines.append(line)
    if len(lines) == 1:
        return problems[0]
    kind = TypeError if any(isinstance(p, TypeError) for p in problems) else ValueError
    listed = "".join(f"\n- {line}" for line in lines)
    return kind(f"{prefix}{len(lines)} problems.{listed}")


# The parameters where ``inf`` means something: no decay, no ceiling, a
# pinned coefficient, no clip, least squares (``huber_delta``), a step nothing
# caps (``pa``'s ``c``), the argmin (``average_eta``). Rust parses these as
# ``Num`` and ``validate`` takes them at ``inf``; everywhere else ``validate``
# says ``finite``, and an infinity is refused here by name first, because
# once it is inside the model's tagged union serde can only say `expected
# f64`. The table was the fields whose Rust *type* admits ``inf``, which named
# ``ridge`` and ``q``, both refused by ``validate``, and missed six that mean
# something (review 2026-09-12, S27). Keyed by builder, since one name can be
# a limit for one builder and no setting for another; ``"*"`` is the shared
# parameters. tests/test_error_messages.py checks this table against the
# Rust side.
#: Where ``inf`` meant something until task 120 (2026-09-28), and the Rust
#: side now refuses it with a message that names what to write instead:
#: ``half_life = "inf"`` or a finite cap, and ``"reset"``. The builder passes
#: these to it rather than say only "must be finite". ``min_weight`` joined
#: them in task 278 (the user, 2026-10-25): a floor at ``inf`` withheld every
#: row for good, and the Rust side's refusal says why.
_INF_REFUSED_BY_RUST = frozenset({"gap_cap", "session_gap", "min_weight"})

_INF_OK: dict[str, frozenset[str]] = {
    "*": frozenset(
        {
            "half_life",
            "average_eta",
            # The noise gate at `inf` is off: no ratio is above it.
            "max_error_inflation",
            # A window budget at `inf` is no bound (`{"thin": inf}`).
            "window_budget",
            # A solve cadence at `inf` never comes due: the sums are
            # collected and nothing is solved (task 278).
            "solve_every",
            # A diagnostic's memory at `inf` is the run-once form (task 221).
            "calibration_half_life",
            "breaks_half_life",
            "robust_se_half_life",
            "specification_half_life",
            "tails_half_life",
            "influence_half_life",
            "feature_health_half_life",
        }
    ),
    "ewridge": frozenset({"long_half_life"}),
    "lasso": frozenset({"select_half_life"}),
    "kalman": frozenset({"coef_half_life", "revert_half_life"}),
    # A refit budget at `inf` is no bound (task 277).
    "huber": frozenset({"huber_delta", "huber_scale_half_life", "refit_budget"}),
    "sgd": frozenset(
        {
            "clip_gradient",
            "coef_min",
            "coef_max",
            "huber_delta",
            "huber_scale_half_life",
            "refit_budget",
        }
    ),
    "quantile": frozenset({"refit_budget"}),
    "pa": frozenset({"c", "coef_min", "coef_max"}),
    "holt": frozenset({"trend_half_life"}),
    # No bound on the bins' memory (docs/PLAN.md task 131).
    "marginal": frozenset({"bin_budget", "bin_huber_delta", "refit_budget"}),
}
