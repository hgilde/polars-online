"""Spec builders: plain dicts, validated eagerly by the Rust core.

The public module is :mod:`polars_online.spec`; its docstring says what a
spec is, what every model shares and what a builder raises. Here: the
checker each builder is wrapped in (:func:`_checked`, a ``TypeError`` per
keyword whose value has the wrong shape, a ``ValueError`` for a count below
0 or a non-finite value where infinity means nothing), the shared
parameters (:func:`_common`, which has the Rust side validate the finished
dict), and the JSON the two sides exchange (:func:`_json`, where
``inf`` becomes the string the Rust side reads).
"""

from __future__ import annotations

import functools
import json
import math
import numbers
import types
import typing
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, NotRequired, TypedDict, Unpack

import polars as pl

from polars_online import _warnings
from polars_online._duration import Duration, duration_text, infinity_as_number
from polars_online._kwargs import CommonKwargs
from polars_online._polars_online import (
    spec_coef_fields,
    spec_output_fields,
    spec_output_index,
    validate_spec,
)
from polars_online._renamed import removed_keywords, renamed_keywords
from polars_online._warnings import forward_deprecated


def _who(name: Any) -> str:
    """How a refusal names a spec: ``spec "<name>"``, or, for a hand-built
    dict with no name, ``a spec with no name`` -- which read ``spec null``
    (review 2026-10-06, YA6)."""
    return "a spec with no name" if name is None else f"spec {json.dumps(name)}"


def _json(spec: dict[str, Any] | list[dict[str, Any]]) -> str:
    """JSON has no infinity literal, but ``half_life=inf`` is meaningful (it pins
    a coefficient), so infinities are encoded as strings the Rust side
    understands. A NaN is never meaningful in a spec and is refused here, by
    parameter name, rather than by the JSON offset serde would report. NumPy
    scalars are plain numbers here (``json`` alone refuses them).

    A raw dict may hold what a builder takes: a window expression in
    ``targets`` is a formula target, and anywhere else an expression or a
    ``timedelta`` is a duration, written as its text as the builders write it
    (review 2026-10-05, YA11: an expression under any key was read as a
    formula, so ``half_life=pl.duration(minutes=10)`` was refused as a formula
    literal that named no key)."""

    def enc(v: Any, key: str, who: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, numbers.Integral):
            return int(v)
        if isinstance(v, numbers.Real):
            v = float(v)
            if math.isnan(v):
                raise ValueError(f"{_who(who)}: {key} must not be NaN")
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return v
        if isinstance(v, dict):
            # A key renamed after 1.0 is read as its new name, with a
            # warning, at any depth -- a model's own keys are a level down --
            # as the Rust side forwards a TOML file's (`DEPRECATED`).
            v = forward_deprecated(_who(who), v, _warnings._DEPRECATED, stacklevel=5)
            return {k: enc(x, k, who) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x, key, who) for x in v]
        if isinstance(v, pl.Expr) and key == "targets":
            # A window expression in `targets`, as the builders take one
            # (review R2, P5: every surface that writes a spec dict).
            return enc(formula_target(_who(who), v), key, who)
        if isinstance(v, (pl.Expr, timedelta)):
            if key in _CLOCK_KEYS:
                return duration_text(v, _who(who), key)
            raise TypeError(
                f"{_who(who)}: {key} takes no {type(v).__name__}; a duration is "
                "a clock parameter's, and a window expression is a target's"
            )
        return v

    def name(s: Any) -> Any:
        return s.get("name") if isinstance(s, dict) else None

    # A list of specs is `ModelBank`'s path: each spec names its own NaN,
    # where the message read `spec null` (review 2026-09-12, D8).
    if isinstance(spec, list):
        return json.dumps([enc(s, "spec", name(s)) for s in spec])
    return json.dumps(enc(spec, "spec", name(spec)))


def _from_json(text: str) -> Any:
    """The inverse of :func:`_json`: the ``"inf"`` strings the Rust side writes
    for infinite numeric parameters become floats again, so a loaded bank's
    ``specs`` compare equal to the dicts that built it (a ``po.target`` table
    that names its own column comes back as the column, which is what it is).
    Only the numeric parameters are touched -- a feature column may be called
    ``"inf"``."""

    def dec(v: Any, numeric: bool) -> Any:
        if isinstance(v, dict):
            # Under a numeric parameter a dict's values are numbers too:
            # `window_budget = {"thin": inf}` (review 2026-09-12, P4).
            return {k: dec(x, numeric or k in _NUMERIC_KEYS) for k, x in v.items()}
        if isinstance(v, list):
            return [dec(x, numeric) for x in v]
        if numeric and v in ("inf", "-inf"):
            return math.inf if v == "inf" else -math.inf
        return v

    return dec(json.loads(text), False)


# The builders' annotations are the contract, and these two functions read
# them, so a wrong shape is reported by parameter name ("half_life must be a
# number or a list of numbers, got str '10'") before anything is serialized.
# The Rust side checks the same things, but serde cannot name the field once it
# is inside the model's tagged union, and "expected f64" with no name is not
# much of a message.


def _matches(v: Any, hint: Any) -> bool:
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        return any(_matches(v, a) for a in typing.get_args(hint))
    if hint is type(None):
        return v is None
    if origin in (list, Sequence):
        # A list or a tuple: a str is a sequence of strs too, and never what
        # a list parameter means.
        (inner,) = typing.get_args(hint)
        return isinstance(v, (list, tuple)) and all(_matches(x, inner) for x in v)
    if origin is dict:
        key, val = typing.get_args(hint)
        return isinstance(v, dict) and all(
            _matches(k, key) and _matches(x, val) for k, x in v.items()
        )
    if hint is bool:
        return isinstance(v, bool)
    if hint is float:
        return isinstance(v, numbers.Real) and not isinstance(v, bool)
    if hint is int:
        return isinstance(v, numbers.Integral) and not isinstance(v, bool)
    if hint is str:
        return isinstance(v, str)
    if hint in (timedelta, pl.Expr):
        return isinstance(v, hint)
    if typing.is_typeddict(hint):
        # A table (`po.target`): its keys and no other, the required ones
        # present, each value of its annotated shape (review 2026-09-26, F4:
        # a table fell through to "not read", and a wrong entry was named
        # by JSON path).
        if not isinstance(v, dict):
            return False
        # `__required_keys__` reads every key as required under
        # `from __future__ import annotations`; the annotations say which.
        hints = typing.get_type_hints(hint, include_extras=True)
        qualified = (typing.NotRequired, typing.Required)
        required = {k for k, h in hints.items() if typing.get_origin(h) is not typing.NotRequired}
        plain = {
            k: typing.get_args(h)[0] if typing.get_origin(h) in qualified else h
            for k, h in hints.items()
        }
        return (
            set(v) <= set(plain)
            and required <= set(v)
            and all(_matches(x, plain[k]) for k, x in v.items())
        )
    return True  # an annotation this does not read; the Rust side still checks


def _describe(hint: Any, plural: bool = False) -> str:
    if hint == TargetList:
        return (
            "lists of strs, po.target tables or window expressions looking ahead"
            if plural
            else "a list of strs, po.target tables or window expressions looking ahead"
        )
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        # A clock parameter's three duration forms read as one word.
        duration = set(typing.get_args(Duration))
        if duration <= set(args):
            args = [a for a in args if a not in duration]
            parts = [_describe(a, plural) for a in args]
            parts.insert(1, "durations" if plural else "a duration")
        else:
            parts = [_describe(a, plural) for a in args]
        return " or ".join(parts)
    if origin is list:
        (inner,) = typing.get_args(hint)
        return ("lists of " if plural else "a list of ") + _describe(inner, plural=True)
    if origin is dict:
        key, val = typing.get_args(hint)
        return f"a dict of {_describe(key)} -> {_describe(val)}"
    nouns = {float: ("a number", "numbers"), int: ("an int", "ints")}
    nouns |= {bool: ("a bool", "bools"), str: ("a str", "strs")}
    one, many = nouns.get(hint, (str(hint), str(hint)))
    return many if plural else one


def _got(v: Any) -> str:
    r = repr(v)
    return f"{type(v).__name__} {r if len(r) <= 60 else r[:57] + '...'}"


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
#: these to it rather than say only "must be finite".
_INF_REFUSED_BY_RUST = frozenset({"gap_cap", "session_gap"})

_INF_OK: dict[str, frozenset[str]] = {
    "*": frozenset(
        {
            "half_life",
            "min_weight",
            "average_eta",
            # The noise gate at `inf` is off: no ratio is above it.
            "max_error_inflation",
            # A window budget at `inf` is no bound (`{"thin": inf}`).
            "window_budget",
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
    "huber": frozenset({"huber_delta"}),
    "sgd": frozenset({"clip_gradient", "coef_min", "coef_max", "huber_delta"}),
    "pa": frozenset({"c", "coef_min", "coef_max"}),
    "holt": frozenset({"trend_half_life"}),
    # No bound on the bins' memory (docs/PLAN.md task 131).
    "marginal": frozenset({"bin_budget"}),
}


#: The model keys the Rust spec skips when they are absent
#: (``skip_serializing_if`` in ``crates/online-polars/src/spec.rs``): a key
#: written as null there would move every spec's bytes and cost a schema
#: bump. A builder writes each of these only when it is given, so the bank
#: reports back the dict it made: a ``None`` the bank never writes made
#: ``ModelBank([s]).specs[0] != s`` for every ``marginal`` (review
#: 2026-10-05, the TB5 leg). ``holt``'s ``trend`` is written only when off.
_SKIPPED_WHEN_ABSENT: dict[str, frozenset[str]] = {
    "holt": frozenset({"trend"}),
    "ewridge": frozenset({"gram_threads"}),
    "ew_cov": frozenset({"gram_threads"}),
    "marginal": frozenset(
        {
            "lags",
            "serial_rule",
            "cross_lags",
            "bins",
            "bin_rule",
            "bin_warm_rows",
            "bin_edges",
            "bin_budget",
            "shards",
            "window_lags",
            "feature_moments",
        }
    ),
    "audit": frozenset({"pairs", "distinct_cap"}),
}


def _takes_duration(hint: Any) -> bool:
    """Whether a parameter's annotation admits a duration: the clock
    parameters, and nothing else."""
    if hint is timedelta:
        return True
    return any(_takes_duration(a) for a in typing.get_args(hint))


def _finite(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_finite(x) for x in value)
    if isinstance(value, dict):
        return all(_finite(x) for x in value.values())
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return not math.isinf(value)
    return True


#: The int parameters whose Rust floor is 1, not 0 (review 2026-09-12, D8);
#: for a list of ints, each entry's floor.
_AT_LEAST_ONE = frozenset(
    {
        "lags",
        "cross_lags",
        "shards",
        "gram_threads",
        "update_every_rows",
        "split_merge_every_rows",
        "max_clusters",
        "resid_autocorr_lag",
        "ljung_box_lags",
        "k",
        # A lasso of no sweeps never descends: every solve is a failure, and
        # the coefficients read like a fit (review 2026-10-05, PB7).
        "max_iter",
        # A cap of no rows is no schedule: every row is `coef_every = 0`
        # (docs/PLAN.md task 178), and in every row cap the clock form's `0`
        # is every row (task 196, U7).
        "max_rows_between_coefs",
        "max_rows_between_solves",
        "max_rows_between_snapshots",
        "max_rows_between_pca",
        "max_rows_between_prunes",
    }
)


#: The int parameters that are a ``u32`` on the Rust side; every other count
#: is a ``u64`` (``usize`` on the 64-bit platforms the package ships for).
#: A count past the width was named by serde as ``model`` (review
#: 2026-10-06, YA4); ``test_an_int_past_the_rust_width_is_refused_by_name``
#: holds this table to the Rust side.
_U32 = frozenset(
    {
        "max_rows_between_coefs",
        "max_rows_between_solves",
        "max_rows_between_snapshots",
        "max_rows_between_pca",
        "max_rows_between_prunes",
        "max_iter",
        "update_every_rows",
        "split_merge_every_rows",
    }
)

#: The counts whose ceiling is tighter than their width: each sizes an
#: allocation before the first row -- a ring of that many rows, or the
#: permutation draws held for a quantile -- and Rust refuses past it too
#: (``online_core::MAX_LAG``, ``MAX_PERM``; review 2026-10-06, CD10). For a
#: list of counts, each entry's ceiling.
_CEILING = {
    "lags": 2**20,
    "resid_autocorr_lag": 2**20,
    "robust_se_lags": 2**20,
    "ljung_box_lags": 2**20,
    "n_perm": 2**20,
}

#: A ceiling one builder's count has under a name another builder shares:
#: ``hmm``'s ``k`` sizes its transition matrix, ``k^2`` cells, and its states
#: before the first row (``online_core::Hmm::MAX_K``), where ``kmeans``' ``k``
#: is held to 2^16 by the Rust side.
_CEILING_OF = {"hmm": {"k": 2**10}}


def _int_ceiling(key: str, builder: str = "") -> int:
    """The most an int parameter of ``builder`` may be: its own ceiling,
    else its width."""
    own = _CEILING_OF.get(builder, {})
    if key in own:
        return own[key]
    return _CEILING.get(key, 2**32 - 1 if key in _U32 else 2**64 - 1)


def _lists(value: Any) -> Any:
    """``value`` with every tuple a list, at any depth: the dict a builder
    returns is the one the bank reports back, and ``features=("x0",)`` kept
    its tuple, so ``ModelBank([s]).specs[0] != s`` (review 2026-10-06,
    YA5)."""
    if isinstance(value, (list, tuple)):
        return [_lists(v) for v in value]
    if isinstance(value, dict):
        return {k: _lists(v) for k, v in value.items()}
    return value


#: The parameters task 144 renamed (docs/PLAN.md): an old name is refused
#: naming the new one, with no alias.
_RENAMED = {
    "halflife": "half_life",
    "long_halflife": "long_half_life",
    "coef_halflife": "coef_half_life",
    "revert_halflife": "revert_half_life",
    "select_halflife": "select_half_life",
    "level_halflife": "half_life",
    "trend_halflife": "trend_half_life",
    "label_delay": "embargo",
    "max_dclock": "gap_cap",
    "window": "window_size",
    "min_periods": "min_weight",
    "emit_resid_z": "emit_zscore",
    "scale_features": "standardize",
    "on_clock_reset": "restart_after_step_back",
    "min_backwards_jump": "restart_after_step_back",
    "ridge_decay": "ridge_scale",
    "add_intercept": "fit_intercept",
    "max_cd_iters": "max_iter",
    "cd_tol": "tol",
    "reset": "reset_on_flag",
    # Task 196 (docs/PLAN.md §18, N14 and N16): a count of rows says so, and
    # holt's level takes the spec's `half_life`, one knob under one name.
    "update_every": "update_every_rows",
    "split_merge_every": "split_merge_every_rows",
    "permute_every": "permute_every_rows",
    "level_half_life": "half_life",
}

#: The parameters renamed in one builder whose old name another keeps
#: (docs/PLAN.md task 195, N11): ``rls``'s ``ridge`` is ``delta``, and
#: ``ridge`` is still :func:`ewridge`'s, :func:`huber`'s and :func:`quantile`'s.
_RENAMED_IN = {"rls": {"ridge": "delta"}}

#: The keywords of the functions that run a stream that task 196 renamed
#: (docs/PLAN.md §18, N2): ``chunk_size`` is Polars' name on the call it
#: feeds, ``collect_batches(chunk_size=)``. :func:`_renamed_keywords`
#: refuses the old one naming the new.
_RENAMED_KEYWORDS = {"chunk_rows": "chunk_size"}


def _renamed_keywords[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """``fn``, refusing a keyword :data:`_RENAMED_KEYWORDS` names by the new
    name, where Python would say only that it is unexpected: the refusal
    :func:`polars_online._renamed.renamed_keywords` gives every helper, as
    ``<qualified name>(): <old> was renamed <new>``."""
    return renamed_keywords(f"{fn.__qualname__}()", _RENAMED_KEYWORDS)(fn)


def _checked[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Check each keyword against ``fn``'s annotations (and ``_common``'s for
    the shared parameters) so a wrong shape names the parameter."""
    skip = {"return", "common"}
    own = {k: v for k, v in typing.get_type_hints(fn).items() if k not in skip}
    shared = typing.get_type_hints(_common)
    shared = {k: v for k, v in shared.items() if k not in {"return", "name", "model"}}
    inf_ok = _INF_OK["*"] | _INF_OK.get(fn.__name__, frozenset())
    clock = frozenset(k for k, v in {**shared, **own}.items() if _takes_duration(v))

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        name = args[0] if args else kwargs.get("name")
        if not isinstance(name, str):
            raise TypeError(f"spec name must be a str, got {_got(name)}")
        who = f"spec {json.dumps(name)}"
        # A keyword renamed after 1.0, read as its new name with a warning
        # pointed at the caller's line.
        given = forward_deprecated(who, kwargs, _warnings._DEPRECATED, stacklevel=3)
        for key, value in given.items():
            if key == "name":
                continue
            if key == "targets" and isinstance(value, (list, tuple)):
                # A table written by hand with a relative target's keys
                # (task 201): refused by name, where the table's shape check
                # would say only that `targets` is not a list of targets.
                for t in value:
                    if isinstance(t, dict) and any(k in t for k in _RELATIVE_KEYS):
                        what = t.get("column", t.get("name"))
                        raise TypeError(f"{who}: target {json.dumps(what)}: {_RELATIVE_REMOVED}")
            hint = own.get(key, shared.get(key))
            if hint is None:
                if key in _RENAMED:
                    raise TypeError(f"{who}: {key} was renamed {_RENAMED[key]}")
                if key in _RENAMED_IN.get(fn.__name__, {}):
                    raise TypeError(f"{who}: {key} was renamed {_RENAMED_IN[fn.__name__][key]}")
                raise TypeError(
                    f"{who}: {fn.__name__}() got an unexpected keyword argument {key!r}"
                )
            if not _matches(value, hint):
                raise TypeError(f"{who}: {key} must be {_describe(hint)}, got {_got(value)}")
            # Every int parameter is a count (u32 on the Rust side), and some
            # are counts of something there must be one of: the Rust side's
            # floor, said here, where `0` passed with nothing said (review
            # 2026-09-12, D8).
            if (
                isinstance(value, numbers.Integral)
                and not isinstance(value, bool)
                and int in (hint, *typing.get_args(hint))
            ):
                floor = 1 if key in _AT_LEAST_ONE else 0
                if value < floor:
                    raise ValueError(f"{who}: {key} must be >= {floor}, got {value}")
                # And at most its width, or the tighter ceiling of a count
                # that sizes an allocation (review 2026-10-06, YA4 and CD10).
                ceiling = _int_ceiling(key, fn.__name__)
                if int(value) > ceiling:
                    raise ValueError(f"{who}: {key} must be <= {ceiling}, got {value}")
            # And each entry of a list of counts (review 2026-09-26, F7: a
            # negative lag was named by serde as `model`).
            if (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in value)
                and list[int] in (hint, *typing.get_args(hint))
            ):
                floor = 1 if key in _AT_LEAST_ONE else 0
                if min(value) < floor:
                    raise ValueError(f"{who}: {key} must be >= {floor}, got {_got(value)}")
                ceiling = _int_ceiling(key, fn.__name__)
                if max(value) > ceiling:
                    raise ValueError(f"{who}: {key} must be <= {ceiling}, got {_got(value)}")
            if key not in inf_ok and key not in _INF_REFUSED_BY_RUST and not _finite(value):
                raise ValueError(f"{who}: {key} must be finite, got {_got(value)}")
        # A clock parameter's duration, however it was written, is kept as
        # the text a TOML config writes and a state file stores (task 88),
        # and an infinity word as the number it names (YA7). Every tuple is a
        # list, as the bank reports the dict back (YA5).
        written = {
            key: infinity_as_number(duration_text(_lists(value), who, key))
            if key in clock
            else _lists(value)
            for key, value in given.items()
        }
        return typing.cast(Callable[..., R], fn)(*args, **written)

    return wrapper


__all__ = [
    "audit",
    "bocpd",
    "corrchange",
    "deco",
    "hmm",
    "rcov",
    "ew_class",
    "ew_cov",
    "ewridge",
    "ftrl",
    "holt",
    "huber",
    "kalman",
    "kmeans",
    "lasso",
    "marginal",
    "micro",
    "output_fields",
    "pa",
    "quantile",
    "rls",
    "seqtest",
    "sgd",
]


class Target(TypedDict):
    """A target written as a table: what :func:`polars_online.target` returns,
    and a ``[[specs]]`` target table in the CLI's TOML. ``column`` is read as
    it is; ``name`` is what its output fields carry, ``column`` when not
    given."""

    column: str
    name: NotRequired[str]


class FormulaTarget(TypedDict):
    """A target that is a formula of the row's future (docs/PLAN.md task 104):
    what a window expression in ``targets`` becomes, and a ``[[specs]]`` target
    table with ``name`` and ``formula`` in the CLI's TOML. ``formula`` is the
    expression's compact tree (:mod:`polars_online.ops`), holding at least one
    operator looking ahead."""

    name: str
    formula: list[Any]


#: What a builder's ``targets`` takes: column names, with
#: :func:`polars_online.target` tables and window expressions
#: (:mod:`polars_online.ops`, looking ahead) among them, in a list or a tuple.
#: A ``Sequence``, which is covariant, so a ``list[str]`` a caller already has
#: type-checks, and so do a list of tables, a list of expressions and a mix:
#: ``list`` is invariant, and a union of two lists refused all three (review
#: 2026-10-05, YA4). A ``str`` is a sequence of strs to a type checker; the
#: builders refuse one by name when they run.
TargetList = Sequence[str | Target | FormulaTarget | pl.Expr]

#: The keys of task 107a's relative target, removed in task 201: a target
#: table naming one is refused by name, with :data:`_RELATIVE_REMOVED`.
_RELATIVE_KEYS = ("relative_to", "relative")

#: What a relative target is told on the Python side; the Rust side says the
#: same of a spec dict, a TOML file and a saved state.
_RELATIVE_REMOVED = (
    "relative targets were removed: compute the target as a column, e.g. "
    '`lf.with_columns(ret=pl.col("p") - pl.col("mid"))`, and name it in `targets`; for a '
    'return, prefer a log ratio, `(pl.col("p") / pl.col("mid")).log()`'
)


def target_name(t: str | Target | FormulaTarget) -> str:
    """The name a target's output fields carry: a string target's own, a
    table's ``name``, or its ``column`` when it gives none."""
    if isinstance(t, str):
        return t
    if "formula" in t:
        return typing.cast(FormulaTarget, t)["name"]
    return t.get("name", t["column"])


def target_columns(t: str | Target | FormulaTarget) -> list[str]:
    """The columns a target reads: a string target's own; a table's
    ``column``; a formula's columns. What a plan must keep for the bank
    (review 2026-09-26, D1/F1)."""
    if isinstance(t, str):
        return [t]
    if "formula" in t:
        from polars_online import _formula

        return _formula.columns(typing.cast(FormulaTarget, t)["formula"])
    return [t["column"]]


def formula_target(who: str, expr: pl.Expr) -> FormulaTarget:
    """A window expression as a target table (docs/PLAN.md task 104): named by
    its alias, holding at least one operator looking ahead."""
    from polars_online import _formula

    tree = _formula.to_tree(expr)
    name = expr.meta.output_name()
    # The alias names the target; the tree kept is the formula under it.
    if isinstance(tree, list) and tree and tree[0] == "alias":
        name = tree[2]
        tree = tree[1]
    if name.startswith(_formula.PREFIX):
        raise ValueError(
            f"{who}: a target expression whose name would be an operator's needs a name: "
            f'give it with .alias("...")'
        )
    if name in _formula.columns(tree):
        # Unaliased, a positional expression is named after its leftmost
        # column, here one the formula itself reads; the bank could never
        # add the target beside it (task 159, P1).
        raise ValueError(
            f"{who}: target {name!r} is named after a column its formula reads, so it could "
            f'never be added beside that column: give it a name of its own with .alias("...")'
        )
    if not _formula.looks_ahead(tree):
        # Name the call that makes the column: with_windows for a formula over
        # operators looking back, Polars' with_columns for one of the row alone,
        # which with_windows refuses in turn (README-ITERATIONS, E11).
        if _formula.operators(tree):
            make = "add it as a column with po.stream.with_windows"
        else:
            make = "it holds no operator at all, so add it as a column with Polars' with_columns"
        raise ValueError(
            f"{who}: target {name!r} holds no operator looking ahead (rewm_mean, rewm_sum or "
            f"rewm_rate), so it is known at its own row: {make}, and name the column as the "
            f"target"
        )
    return {"name": name, "formula": tree}


@removed_keywords("po.target", _RELATIVE_KEYS, _RELATIVE_REMOVED)
def target(column: str, *, name: str | None = None) -> Target:
    """A target column under a name of its own.

    ``po.target("p", name="price")`` in a spec's ``targets`` has the model
    learn the column ``p`` as it is, and name its output fields after
    ``price``: ``pred_price`` and the rest. A string target is the same as
    ``po.target`` with no ``name``, and so is a ``name`` equal to the column.
    A target's column used as a feature is refused, whatever its name.

    A target computed from its own row's columns, such as a price against
    the mid, is a column: make it with Polars' ``with_columns`` in the query
    before the bank, and name it in ``targets``. For a return, a log ratio
    (``(pl.col("p") / pl.col("mid")).log()``) or a difference sits about
    zero, where ``hit_rate`` takes its sign; a plain ratio sits about 1,
    where two positive numbers always agree. Before 1.0 this function took
    ``relative_to`` and ``relative`` for that; they are refused, and the
    error says this. A target computed from the rows after its own is a
    window expression looking ahead (:mod:`polars_online.ops`), named by
    its alias.

    In the CLI's TOML the same target is a table:
    ``targets = ["ret_5m", { column = "p", name = "price" }]``.
    """
    if not isinstance(column, str):
        raise TypeError(f"target: column must be a str, got {type(column).__name__}")
    if not column:
        raise ValueError("target: column must not be empty")
    out: Target = {"column": column}
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(f"target {column!r}: name must be a str, got {type(name).__name__}")
        if not name:
            raise ValueError(f"target {column!r}: name must not be empty")
        out["name"] = name
    return out


def _common(
    name: str,
    model: dict[str, Any],
    *,
    targets: TargetList,
    features: list[str],
    fit_intercept: bool = True,
    clock: str | None = None,
    half_life: float | Duration | list[float | Duration] | None = None,
    lam: float | None = None,
    gap_cap: float | Duration | None = None,
    restart_after_step_back: float | Duration | None = None,
    session: str | None = None,
    session_gap: float | Duration | None = None,
    weight: str | None = None,
    min_weight: float | list[float] | None = None,
    min_settled_frac: float | None = None,
    max_error_inflation: float | None = None,
    emit_error_inflation: bool = False,
    emit_se_coef: bool = False,
    emit_clocks: bool = False,
    coef_every: float | Duration | None = None,
    max_rows_between_coefs: int | None = None,
    emit_sigma: bool = False,
    emit_zscore: bool = False,
    emit_selected: bool = False,
    emit_averaged: bool = False,
    average_eta: float | None = None,
    emit_metrics: bool = False,
    conformal: float | None = None,
    conformal_rate: float | None = None,
    resid_quantiles: list[float] | None = None,
    emit_autocorr: bool = False,
    resid_autocorr_lag: int | None = None,
    emit_drift: bool = False,
    drift_delta: float | None = None,
    drift_threshold: float | Duration | None = None,
    drift_action: str = "flag",
    emit_calibration: bool = False,
    calibration_half_life: float | Duration | None = None,
    emit_breaks: bool = False,
    breaks_half_life: float | Duration | None = None,
    emit_robust_se: bool = False,
    robust_se_half_life: float | Duration | None = None,
    robust_se_lags: int | None = None,
    emit_specification: bool = False,
    specification_half_life: float | Duration | None = None,
    ljung_box_lags: int | None = None,
    emit_tails: bool = False,
    tails_half_life: float | Duration | None = None,
    emit_influence: bool = False,
    influence_half_life: float | Duration | None = None,
    emit_feature_health: bool = False,
    feature_health_half_life: float | Duration | None = None,
    embargo: float | Duration | None = None,
    group: str | None = None,
    group_close: str | None = None,
) -> dict[str, Any]:
    written: list[str | Target | FormulaTarget] = [
        formula_target(f"spec {json.dumps(name)}", t) if isinstance(t, pl.Expr) else t
        for t in targets
    ]
    spec = {
        "name": name,
        "model": model,
        "targets": written,
        "features": features,
        "fit_intercept": fit_intercept,
        "clock": clock,
        "half_life": half_life,
        "lam": lam,
        "gap_cap": gap_cap,
        "restart_after_step_back": restart_after_step_back,
        "session": session,
        "session_gap": session_gap,
        "weight": weight,
        "min_weight": min_weight,
        "min_settled_frac": min_settled_frac,
        "max_error_inflation": max_error_inflation,
        "emit_error_inflation": emit_error_inflation,
        "emit_se_coef": emit_se_coef,
        "emit_clocks": emit_clocks,
        "coef_every": coef_every,
        "max_rows_between_coefs": max_rows_between_coefs,
        "emit_sigma": emit_sigma,
        "emit_zscore": emit_zscore,
        "emit_selected": emit_selected,
        "emit_averaged": emit_averaged,
        "average_eta": average_eta,
        "emit_metrics": emit_metrics,
        "conformal": conformal,
        "conformal_rate": conformal_rate,
        "resid_quantiles": resid_quantiles,
        "emit_autocorr": emit_autocorr,
        "resid_autocorr_lag": resid_autocorr_lag,
        "emit_drift": emit_drift,
        "drift_delta": drift_delta,
        "drift_threshold": drift_threshold,
        "drift_action": drift_action,
        "emit_calibration": emit_calibration,
        "calibration_half_life": calibration_half_life,
        "emit_breaks": emit_breaks,
        "breaks_half_life": breaks_half_life,
        "emit_robust_se": emit_robust_se,
        "robust_se_half_life": robust_se_half_life,
        "robust_se_lags": robust_se_lags,
        "emit_specification": emit_specification,
        "specification_half_life": specification_half_life,
        "ljung_box_lags": ljung_box_lags,
        "emit_tails": emit_tails,
        "tails_half_life": tails_half_life,
        "emit_influence": emit_influence,
        "influence_half_life": influence_half_life,
        "emit_feature_health": emit_feature_health,
        "feature_health_half_life": feature_health_half_life,
        "embargo": embargo,
        "group": group,
        "group_close": group_close,
    }
    validate_spec(_json(spec))
    return spec


def _mirror_target(
    name: str, kind: str, features: list[str], common: Any, statistic: str
) -> list[str]:
    """The ``targets`` a model with no target hands the plumbing: its first
    feature. ``targets=`` is refused by name, and so is an empty
    ``features``, which the Rust side names for every other model and these
    indexed before it could (review 2026-10-05, YA1 and YA5: an
    ``IndexError``, and a ``TypeError`` naming ``_common()``)."""
    if "targets" in common:
        msg = f"spec {json.dumps(name)}: {kind}() takes no targets; {statistic} the features"
        raise TypeError(msg)
    if not features:
        raise ValueError(f"spec {json.dumps(name)}: features must be non-empty")
    return [features[0]]


@_checked
def ewridge(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    ridge: float | list[float] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    standardize: bool = False,
    ridge_scale: str = "mean",
    coef_prior: list[list[float]] | None = None,
    session_shrink: float | None = None,
    long_half_life: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    gram_block_rows: int | None = None,
    gram_threads: int | None = None,
    target_gaps: str = "own_rows",
    window_size: float | Duration | None = None,
    closed: str = "right",
    window_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted ridge regression on running sums -- the workhorse.

    The model keeps the exponentially weighted means of ``z z'`` and ``z y``,
    with ``z`` the features and the intercept in front, and solves the ridge
    normal equations from them on a schedule. The sums are the whole state. So
    several ridge values and feature subsets are fitted from the same sums at
    almost no extra cost (a list of half-lives keeps one set of sums each),
    and the sums can be read back (:meth:`polars_online.ModelBank.gram`),
    pooled and solved offline (:mod:`polars_online.gram`). Reach for it first.
    The other regressions here each cover a case it does not: coefficients
    that drift (:func:`kalman`), outliers (:func:`huber`), a quantile
    (:func:`quantile`), a sparse fit (:func:`lasso`), a row cost of O(k)
    (:func:`sgd`, :func:`pa`).

    .. rubric:: The fit

    Per row, with ``w`` the row's weight and ``lam`` its decay ``0.5 **
    (d_clock / half-life)``. ``W_j`` is the weight behind target ``j`` over
    the rows that target is present on:

    .. code-block:: text

        W_j' = lam * W_j + w
        S_j' = (lam * W_j * S_j + w * z z') / W_j'
        r_j' = (lam * W_j * r_j + w * z * y_j) / W_j'
        on the schedule:  (S_j + ridge * D) beta_j = r_j      D = I with a 0 in the intercept slot

    The sums are means, not totals, so they stay bounded over a stream of any
    length. The second moments are kept centred (a weighted Welford update),
    so a feature far from zero loses no precision. The solve is a Cholesky
    factorization: a near-singular system is retried with a small diagonal
    jitter, and a solve that needed one is counted in
    :meth:`polars_online.ModelBank.solve_failures`. A prediction uses the
    coefficients of the last solve and the state before the row. A wide
    solve runs on the bank's threads, and its result is the same to the bit
    whatever their number (``POLARS_ONLINE_MAX_THREADS``, or the machine's
    cores): at 2,000 features a factorization and solve took 18 ms on eight
    threads against 68 on one.

    .. rubric:: Parameters

    ``ridge``
        The penalty on the slopes, in the features' squared units unless
        ``standardize``; on the intercept too under ``ridge_scale = "sum"``,
        and never otherwise. Default ``1e-6``. A list fits one instance per
        value from the same sums, reported side by side as
        ``pred_<t>__r<ridge>``.
    ``feature_sets``
        Named subsets of ``features``, each a fit of its own from the same
        sums, reported as ``pred_<t>__<set>``. The full set is fitted only
        when it is one of them; ``emit_selected`` then reports the set doing
        best.
    ``standardize``
        Solve in correlation form and unscale afterwards, so ``ridge`` means
        the same thing whatever the features' units, and a feature whose
        variance is zero is dropped from the solve rather than blowing it up.
        Default ``False``. Without an intercept nothing is centred. The system
        is scaled by each column's root mean square instead, a fit through the
        origin is least squares through the origin, and the column dropped is
        one that is all zero. ``lasso``, ``kalman``, ``huber``, ``quantile``
        and ``sgd`` standardize the same way.
    ``ridge_scale``
        What the ridge is scaled against. ``S`` is a weighted mean, so the two
        settings mean two different priors:

        .. list-table::
           :header-rows: 1
           :widths: 14 40 46

           * - setting
             - the prior
             - the system
           * - ``"mean"`` (the default)
             - a fixed per-observation penalty whose pull is permanent:
               "always stay near this belief"
             - ``(S + ridge * D) b = r``, the intercept unpenalized
           * - ``"sum"``
             - sits on the decaying sum scale and fades as data arrives, the
               usual warm start: "begin at yesterday's fit and let evidence
               take over"
             - RLS's, ``(W S + prior_scale * ridge * I) b = W r``, penalizing
               every slot, **the intercept's included**

        Under ``"sum"`` a constant target of 5 reads an intercept of 3.5 at
        row 20 under ``ridge=10``, ``half_life=50``, and 4.95 at row 200,
        until the prior fades. A ``coef_prior`` intercept is what it shrinks
        toward.
    ``coef_prior``
        Shrink toward these coefficients instead of toward zero: one vector
        per target, in the features' original units, ``len(features) + 1``
        long when there is an intercept. The intercept slot is read only
        under ``ridge_scale = "sum"``, the one solve that penalizes the
        intercept.
    ``session_shrink``, ``long_half_life``
        A middle way between a session's decay and a full reset. A second
        accumulator follows the long-run relationship at ``long_half_life``
        (``inf``: the whole history). On a session boundary the fit's moments
        become a mixture of the two data sets, ``1 - f`` of today's and ``f``
        of the long run's:

        .. code-block:: text

            m' = (1 - f) * m_fast + f * m_slow
            C' = (1 - f) * C_fast + f * C_slow + f * (1 - f) * (m_fast - m_slow)(m_fast - m_slow)'

        for the means and the centred moments. The weight stays today's, so
        ``weight_sum``, the warm-up gates and the solve schedule do not move.
        ``0`` keeps today's fit, ``1`` takes the long run's moments at today's
        weight, and ``f`` between fits on that share of the long run. Unlike
        ``session_gap`` this changes what the model believes, not how
        confident it is. With ``ridge_scale = "sum"`` the prior keeps today's
        scale too, so at ``1`` the fit is the twin's moments under today's
        prior.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule: every ``solve_every`` clock units, and at least
        every ``max_rows_between_solves`` rows. Left out, the schedule is by
        weight: a solve once the weight learned since the last reaches ``ln 2
        / 50`` of the weight the fit holds. In steady state that is every
        ``half_life / 50`` of clock at any row spacing, and more often during
        warm-up and after a gap, where the fit moves most. A half-life far
        longer than the stream still solves, at rows further apart as the
        weight grows. ``half_life = inf`` and ``lam`` solve every row.
        ``max_rows_between_solves`` is off by default. The coefficients are
        the sums' as of the last solve. A row of weight 0 is clock alone: it
        never solves and is no row of the cap, so a solve that comes due on
        one waits for the next row with weight, and the solves fall where
        they would without it.
    ``gram_block_rows``
        Hold that many rows back and bring the ``k x k`` matrix up to date
        once per block, by one matrix product instead of one rank-one update
        per row: ``256`` measured 6.6x faster at a thousand features. The
        matrix is brought up to date before every solve too, so the block
        never exceeds the solve cadence. The option is refused where there is
        no cadence (``solve_every <= 0`` or ``max_rows_between_solves <= 1``)
        and with ``window_size``. ``weight_sum``, the timing of every
        prediction and chunk invariance are unchanged to the bit. The blocked
        sum is the same sum in another order, so a blocked fit agrees with an
        unblocked one to rounding. The held rows travel in the state file, and
        are refused over 256 MiB.
    ``gram_threads``
        Run the update of the ``k x k`` matrix on up to that many threads:
        the block's matrix product under ``gram_block_rows``, and each row's
        rank-one update without it. Default 1. The output is the same to the
        bit at every count. Each entry of the matrix is still summed by one
        thread in one order, because the product is cut into pieces by ``k``
        alone, never by the count. The threads come from the pool the bank
        runs its groups on (``POLARS_ONLINE_MAX_THREADS``). So a bank whose
        groups already keep every core busy gains little, and loses nothing.
        Measured on 14 cores with ``gram_block_rows=256``: at 2,000 features
        a bank learned 3.9 times as fast on eight threads, and on eight
        threads the product alone ran 6 to 7 times as fast from 2,000 to
        10,000 features. Without a block the update is bound by memory and
        gains 2 to 3 times. Up to
        256 features the product is one piece, and below 725 a row's update
        stays on one thread, so the option changes nothing there. Being the
        same to the bit at every count, it is a setting and not part of the
        state: a saved bank resumes under the count of the specs given to
        ``load``, or under the saved one when given none.
    ``target_gaps``
        Which rows a target's fit is read from where the target is null on
        some. Target ``j`` keeps its mean ``ybar_j``, the column means ``m_j``
        and the centred cross-moments ``c_j = EW[(x - m_j)(y_j - ybar_j)]``
        over the rows it is present on. Its slopes solve ``(C + ridge * I) b =
        c_j``, with ``b_0 = ybar_j - m_j . b``. The option is which rows the
        feature covariance ``C`` is taken over:

        .. list-table::
           :header-rows: 1
           :widths: 16 44 40

           * - setting
             - ``C`` is taken over
             - what that gives
           * - ``"own_rows"`` (the default)
             - the target's own rows, so its fit is the fit of the frame with
               its null rows dropped; targets present on the same rows share
               one ``C``, and one that goes missing where the others are
               present takes a copy and keeps its own from then on
             - a bank of targets keeps one ``k x k`` matrix per pattern of
               missing rows, and a single target none
           * - ``"pairwise"``
             - every row, the way pandas' ``DataFrame.cov`` takes a
               pairwise-complete covariance: one matrix whatever the gaps
             - exact when the gaps have nothing to do with the features;
               where they do, each slope is scaled by the ratio of the
               feature's variance on the target's rows to its variance on all
               of them

        ``weight_sum`` counts every row either way, and a null target is
        still predicted.
    ``window_size``, ``closed``, ``window_every``, ``max_rows_between_snapshots``, ``window_budget``
        A hard cutoff on the history the fit is solved from, in clock units:
        a row ``window_size`` old or older is not in the sums at all, where
        the exponential weight alone would leave ``0.5 ** (age / half_life)``
        of it. Inside the window the weights are still exponential, so this
        is a windowed exponentially weighted regression, not a rolling least
        squares. It is exact. The sums are sums of per-row contributions, so
        everything at or before a time ``u`` is ``lam ** (t - u)`` times the
        sums as they stood then, and subtracting that leaves the window. The
        model keeps a ring of snapshots to do it, which is the one place here
        where memory grows with a window rather than with the state. A
        half-life grid is one instance per entry, each with its own ring.

        ``closed`` is which edge holds the row exactly ``window_size`` old,
        in Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the
        default, keeps the rows less than ``window_size`` old, so that row
        has left, as it has in the window operators (:func:`polars_online.ewm_mean`)
        and in ``rolling_sum_by``. ``"both"`` keeps it too. ``"left"`` and
        ``"none"`` would leave out the row the window ends at, which a model
        cannot do, since it reads its fit after it has learned that row: both
        are refused, as ``"both"`` is without ``window_size``.

        The ring takes a snapshot on every row unless told otherwise.
        ``window_every`` spaces them on the clock, as ``solve_every`` spaces
        solves: a number of the clock's units, a duration on a temporal clock
        (``"1m"``), or ``0`` for every row. ``max_rows_between_snapshots``
        caps the rows between them, as ``max_rows_between_solves`` caps the
        rows between solves, and whichever comes first takes a snapshot. A
        row of weight zero counts, a gap capped at ``gap_cap`` counts as the
        cap, and without a clock column the clock is the row's number. A
        coarser cadence divides the memory and can only shorten the effective
        window. Under a clock spacing, every row at most
        ``window_size - window_every`` old is kept and none older than
        ``window_size``. So the effective window is in
        ``[window_size - window_every, window_size]`` in clock units, exactly,
        with ``window_every`` the spacing in force: a thinning
        ``window_budget`` doubles it.
        A burst of rows inside one spacing takes no snapshot of its own.

        ``window_budget`` bounds each ring in MiB, and says what happens when
        a ring reaches the bound:

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - ``window_budget``
             - at the bound
           * - ``{"thin": mib}``
             - drops every other snapshot and doubles the spacing between the
               rest, the clock's or the rows' or both, as often as it takes;
               like ``window_every``, that can only shorten the window
           * - ``{"refuse": mib}``
             - refuses the chunk, naming the ring's size and its cadence
           * - unset
             - refuses past 256 MiB per ring; ``{"refuse": float("inf")}`` is
               no bound

        The bank replays each chunk's clock schedule on its rings before it
        learns a row, so such a chunk is refused whole and the bank goes on as
        it was. Under ``drift_action="reset"``, whose resets the replay cannot
        foresee, the ring finds the overrun as the rows go in instead, and the
        bank then refuses every later ``fit_predict``, ``predict`` and
        ``save``. Rebuild it from its last save
        (:meth:`polars_online.ModelBank.fit_predict`).

        ``weight_sum``, ``sigma`` and ``zscore`` are the window's too, so the
        spread describes the rows the fit describes. The spread keeps a ring
        of its own for it, a pair of floats a slot, bounded with the fit's.
        Everything that reads the spread is the window's with it: drift's
        scale, the conformal band, and the ranking ``emit_selected`` and
        ``emit_averaged`` take. The coefficients are the window's as of the
        last solve, so a coarse ``solve_every`` reports a window that has
        since moved on. ``window_size`` is refused with ``ridge_scale =
        "sum"``: the decaying prior's scale is the product of every decay the
        stream applied, which a window truncates the data of but not the
        prior. It is refused with ``session_shrink`` too: the slow twin is a
        second accumulator under a longer half-life.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#ewridge
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ewridge>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per (target, ridge value, feature set) slot in the order the ``pred``
        fields declare them, the intercept then one entry per feature, zero
        for a feature outside the slot's set. :func:`coef_index` maps each
        position to its term, and :func:`coef_fields` names the column each
        becomes when the struct is unnested.
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data determined
        rather than the ridge (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. Under a grid each field is suffixed per instance. The sums
    behind the fit are :meth:`polars_online.ModelBank.gram`'s, one entry per
    Gram (under ``target_gaps = "own_rows"``, one per set of targets present
    on the same rows). A group that closes writes one row per Gram to
    :meth:`polars_online.ModelBank.closed_groups`.

    .. rubric:: Example

    .. code-block:: python

        rr = po.spec.ewridge(
            "rr", targets=["y"], features=["x0", "x1", "x2"],
            clock="t", gap_cap=300.0, half_life=600.0,
            ridge=[1e-6, 0.1],                # one fit per value, from the same sums
            feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # subsets, likewise
            standardize=True,
            emit_selected=True,               # which (ridge, set) is doing best
        )
        out = po.ModelBank([rr]).fit_predict(df)
        fields = po.spec.output_index(rr)     # every field, with what its name encodes
        best = out["rr"].struct.field("pred_y__selected")

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError``
    naming the problem for a ``feature_sets`` entry naming a column not in
    ``features``, a ``coef_prior`` vector of the wrong length, and
    ``session_shrink`` without ``long_half_life``.
    """
    model: dict[str, Any] = {
        "type": "ewridge",
        "ridge": ridge,
        # `{}` is written as `[]`, which the Rust side refuses by name, as it
        # refuses a dict spec's `[]`: read as `None` it was no sets, in
        # silence (review 2026-10-06, YA5).
        "feature_sets": (
            [[k, list(v)] for k, v in feature_sets.items()] if feature_sets is not None else None
        ),
        "standardize": standardize,
        "ridge_scale": ridge_scale,
        "coef_prior": coef_prior,
        "session_shrink": session_shrink,
        "long_half_life": long_half_life,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "gram_block_rows": gram_block_rows,
        "gram_threads": gram_threads,
        "target_gaps": target_gaps,
        "window_size": window_size,
        "closed": closed,
        "window_every": window_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    # Written only when given, as the Rust spec skips it when absent.
    if gram_threads is None:
        del model["gram_threads"]
    if ridge_scale not in ("mean", "sum"):
        raise ValueError(
            f'spec {json.dumps(name)}: ridge_scale must be "mean" or "sum", got {ridge_scale!r}'
        )
    return _common(name, model, targets=targets, features=features, **common)


def _numeric_keys() -> frozenset[str]:
    """Every parameter, across the builders, whose annotation admits a float.

    Every builder in ``__all__``: a list kept by hand named sixteen of
    twenty-one, and a float of the other five -- ``bocpd``'s ``hazard``, say
    -- came back from a state file as the string ``"inf"`` (review
    2026-09-12, D8)."""
    keys = set()
    helpers = {"output_fields", "output_index", "coef_fields", "coef_index"}
    builders = [globals()[name] for name in __all__ if name not in helpers]
    for fn in (_common, *builders):
        for key, hint in typing.get_type_hints(getattr(fn, "__wrapped__", fn)).items():
            leaves = {hint, *typing.get_args(hint)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            leaves |= {a for h in list(leaves) for a in typing.get_args(h)}
            if float in leaves:
                keys.add(key)
    return frozenset(keys)


def _clock_keys() -> frozenset[str]:
    """Every parameter, across the builders, whose annotation admits a
    duration: the clock parameters, where a raw dict may hold an expression
    or a ``timedelta`` as a builder takes one (:func:`_json`)."""
    helpers = {"output_fields", "output_index", "coef_fields", "coef_index"}
    builders = [globals()[name] for name in __all__ if name not in helpers]
    return frozenset(
        key
        for fn in (_common, *builders)
        for key, hint in typing.get_type_hints(getattr(fn, "__wrapped__", fn)).items()
        if _takes_duration(hint)
    )


def output_fields(spec: dict[str, Any]) -> list[str]:
    """The names of the struct fields a spec writes, in order.

    The same list :meth:`polars_online.ModelBank.output_fields` gives for a whole
    bank, for one spec and before any row is fed: the fields are fixed by the
    spec. ``ValueError`` for a spec that is not valid, with the builders' message
    (:mod:`polars_online.spec`); a dict that is not a spec at all is told what it
    lacks (``invalid spec: missing field `name```).

    .. code-block:: python

        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], half_life=100.0, ridge=[1e-6, 0.1]
        )
        fields = po.spec.output_fields(spec)
        # ['pred_y__r0.000001', 'resid_y__r0.000001', 'pred_y__r0.1', 'resid_y__r0.1',
        #  'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']

    """
    return spec_output_fields(_json(spec))


def output_index(spec: dict[str, Any]) -> pl.DataFrame:
    """Every struct field a spec writes, with the machine values its name encodes.

    One row per field, in the struct's order:

    ``field``
        The name, as it appears in the struct.
    ``kind``
        What it is: ``pred``, ``resid``, ``sigma``, ``weight_sum``, ``coef``,
        ``penalty_selected``, ``selected``, a statistic's stem, and so on.
    ``target``
        The target the field is about, or null.
    ``half_life``, ``lam``
        The decay of the instance the field belongs to, filled for a single
        instance as for a grid. ``half_life`` is the instance's half-life in
        clock units, in seconds under a duration (``300.0`` for ``"5m"``),
        and null under ``lam`` or for an infinite half-life, since the table
        travels as JSON, which has no infinity. ``lam`` is the spec's
        ``lam``, and null under a half-life.
    ``ridge``, ``feature_set``, ``penalty``
        The grid combination: the ridge value, the feature set's name, the lasso
        path point; null where the spec has no such grid.
    ``quantile``
        The level of a ``resid_quantiles`` or ``mahal_quantiles`` field.
    ``columns``
        The pair an ``ew_cov`` statistic is over.
    ``dtype``
        ``f64``, ``bool``, ``str``, ``list[f64]``, ``enum`` (``withheld_reason``),
        ``i32`` (a cluster or state index), ``i64`` (a count, a row index or an
        id) or ``clock`` (``scored_clock`` and ``learned_clock``, in the clock
        column's own type): the type the bank declares to polars before the
        first row is read.

    This is how to reach a field without constructing its name; the string grammar
    (:mod:`polars_online.spec`, "What a spec writes") stays an implementation
    detail:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        idx = po.spec.output_index(grid)
        name = idx.filter(
            (pl.col("kind") == "pred")
            & (pl.col("target") == "y")
            & (pl.col("ridge") == 0.5)
            & (pl.col("half_life") == 500.0)
        )["field"].item()
        column = out["m"].struct.field(name)

    Produced by the Rust code that renders the names, so the table can never drift
    from the strings. ``ValueError`` for a spec that is not valid, as for
    :func:`output_fields`.
    """
    rows = json.loads(spec_output_index(_json(spec)))
    return pl.DataFrame(
        rows,
        schema={
            "field": pl.String,
            "kind": pl.String,
            "target": pl.String,
            "half_life": pl.Float64,
            "lam": pl.Float64,
            "ridge": pl.Float64,
            "feature_set": pl.String,
            "penalty": pl.Float64,
            "quantile": pl.Float64,
            "columns": pl.List(pl.String),
            "dtype": pl.String,
        },
    )


def coef_fields(spec: dict[str, Any]) -> pl.DataFrame:
    """Every coefficient a spec reports, in ``coef`` list order, with the column it
    becomes when the struct is unnested.

    One row per (instance, target, grid combination, term):

    ``field``
        The ``coef`` list the coefficient sits in: ``coef``, or ``coef@h500`` per
        half-life instance.
    ``position``
        Its index in that list.
    ``name``
        The column :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest`
        gives it: ``coef_{target}_{term}{combo}{instance}``, so
        ``coef_y_x1__r0.5@h500`` sits beside ``pred_y__r0.5@h500``. The ``coef_``
        prefix and the target are there because a bare ``x1`` would collide with
        the feature column of that name in the same frame.
    ``target``, ``half_life``, ``lam``, ``ridge``, ``feature_set``, ``penalty``
        As :func:`output_index` reports them.
    ``term``
        ``"intercept"``, a feature name, or ``"level"`` / ``"trend"`` for
        :func:`holt`.

    Empty for the kinds that report no coefficients: ``ew_cov``, ``seqtest``,
    ``marginal``, ``rcov``, ``corrchange`` and ``bocpd``. To reach one coefficient
    in the nested output without writing either name:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        cf = po.spec.coef_fields(grid)
        row = cf.filter(
            (pl.col("target") == "y") & (pl.col("term") == "x1") & (pl.col("half_life") == 500.0)
        ).row(0, named=True)
        slope = out["m"].struct.field(row["field"]).list.get(row["position"])

    Rendered by the Rust code that names the fields, from the slot order the
    models lay the list out in. That order is the intercept first when the spec
    has one, then every feature -- zero for a feature outside a feature set --
    per (target, combination) slot, slots in the order the ``pred`` fields
    declare them.
    ``ValueError`` for a spec that is not valid, as for :func:`output_fields`.
    """
    rows = json.loads(spec_coef_fields(_json(spec)))
    return pl.DataFrame(rows, schema=_COEF_FIELDS_SCHEMA)


#: The columns of :func:`coef_fields`, in order.
_COEF_FIELDS_SCHEMA: dict[str, Any] = {
    "field": pl.String,
    "position": pl.UInt32,
    "name": pl.String,
    "target": pl.String,
    "half_life": pl.Float64,
    "lam": pl.Float64,
    "ridge": pl.Float64,
    "feature_set": pl.String,
    "penalty": pl.Float64,
    "term": pl.String,
}


def _index_columns(fields: pl.DataFrame) -> pl.DataFrame:
    """:func:`coef_index`'s columns of a :func:`coef_fields` frame: the one
    place they are chosen, so a bank with no coefficients lays out the empty
    frame :meth:`ModelBank.coef` gives with them too (review round 4, SF3)."""
    return fields.select(
        pl.col("position").cast(pl.Int64), "target", "ridge", "feature_set", "penalty", "term"
    )


def _coef_index_schema() -> pl.Schema:
    """The schema of every :func:`coef_index` frame, whatever the spec."""
    return _index_columns(pl.DataFrame(schema=_COEF_FIELDS_SCHEMA)).schema


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """The layout of each ``coef`` list, one row per position.

    ``coef`` is flat: (target x grid combination) slots, each contributing its
    terms in order. This maps ``position`` to ``target``, the combination's
    ``ridge``, ``feature_set`` and ``penalty``, and ``term`` -- ``"intercept"``, a
    feature name, or ``"level"`` / ``"trend"`` for :func:`holt`. For
    :func:`kmeans` the slots are the centres, so ``target`` reads ``"cluster0"``,
    ``"cluster1"``, ... and ``term`` is the feature whose coordinate the position
    holds; for :func:`ew_class` and :func:`hmm` a class or state likewise.

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        ci = po.spec.coef_index(grid)
        pos = ci.filter(
            (pl.col("target") == "y") & (pl.col("ridge") == 0.5) & (pl.col("term") == "x1")
        )["position"].item()
        slope = out["m"].struct.field("coef@h100").list.get(pos)

    Every instance's list has this layout; :func:`coef_fields` is the same table
    per instance, with the field each list sits in and the column name each entry
    unnests to. ``ValueError`` for a spec that is not valid, as for
    :func:`output_fields`. ``ValueError`` naming the kind, too, for a spec with no
    fixed layout to index: an ``ew_cov`` spec, which has no coefficients; a
    ``seqtest`` spec, which reports evidence; a ``micro`` spec, whose ``coef`` has
    as many rows as there are established summaries; and ``marginal``, ``rcov``,
    ``corrchange`` and ``bocpd``, which report none.
    """
    kind = spec.get("model", {}).get("type")
    if kind == "ew_cov":
        msg = "ew_cov emits statistics, not coefficients"
        raise ValueError(msg)
    if kind == "seqtest":
        msg = "seqtest emits evidence (log e-values and counts), not coefficients"
        raise ValueError(msg)
    if kind == "micro":
        msg = (
            "micro's coef is one [id, label, n, radius, c_1, ..., c_p] row per established "
            "summary, as many as there are; it has no fixed layout to index"
        )
        raise ValueError(msg)
    cf = coef_fields(spec)
    if cf.is_empty():
        # Every kind with no ``coef``, not only the three named above:
        # ``marginal``, ``rcov``, ``corrchange`` and ``bocpd`` reached the line
        # below on an empty frame and raised whatever polars raises there
        # (review 2026-09-12, S26).
        msg = f"{kind} emits no coefficients"
        raise ValueError(msg)
    first = cf["field"][0]
    return _index_columns(cf.filter(pl.col("field") == first))


@_checked
def rls(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    delta: float | None = None,
    coef_prior: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Recursive least squares: the ridge fit re-solved exactly on every row.

    Where :func:`ewridge` solves on a schedule, this moves the coefficients on
    every row, so nothing is ever out of date. The cost is the same O(k²) per row,
    and the state is the Cholesky factor of the decayed normal matrix rather than
    the matrix itself, which is what keeps the recursion stable.

    .. rubric:: The fit

    .. code-block:: text

        A <- lam * A + w * z z'        b_j <- lam * b_j + w * y_j * z        beta_j = A^-1 b_j
        A_0 = delta * I                b_0 = delta * coef_prior

    What is stored is the factor ``R`` of ``A = R'R`` and ``u_j = R^-T b_j``: a
    row is folded in by Givens rotations and ``beta`` read off by one
    back-substitution (the square-root form). The textbook recursion on the
    inverse ``P`` loses symmetry to rounding by ``1 / lam`` a row, and one extreme
    row can cancel it and freeze a coefficient for good; this form has neither
    failure. The result is :func:`ewridge` with ``ridge_scale = "sum"`` solved on every
    row, to better than 1e-9.

    **A feature that holds one value for long winds the fit up: use**
    :func:`ewridge` **for a long-running stream whose features can go quiet.**
    The prior fades with the sums, so nothing holds a coefficient in a direction
    the data stop exciting. A feature held at a value ``c`` other than 0 shows
    the fit only ``intercept + c * slope``. The information that tells the slope
    from the intercept then decays by ``lam`` a row, with nothing to renew it.
    Once it is under a rounding step, rounding sets the slope, and the slope
    wanders. A feature constant from its first row winds up the same way. Held
    at exactly 0, it excites nothing in that direction, and the slope stays
    where it was. :func:`ewridge`'s default ``ridge`` is a penalty on the means
    that never fades, and it centres each feature, so it does not wind up.
    Measured with ``half_life=20`` (a row a clock unit), ``delta=1`` and noise
    0.17, on a feature held for 150 half-lives after 300 rows of moving, then
    moving again for 10 (docs/PLAN.md task 217):

    .. list-table::
       :header-rows: 1

       * - the run
         - the slope while held (0.5 in the data)
         - it first passes 1
         - ``pred - y`` once it moves: rms / worst
       * - ``rls``, held at 0.87
         - -3.1e13 to 3.0e13
         - 54 half-lives in
         - 3.1e11 / 4.4e12
       * - ``rls``, held at 1,000
         - -8.3e9 to 8.1e9
         - 43
         - 8.3e7 / 1.2e9
       * - ``rls``, held at 1e8
         - -4.6e5 to 4.5e5
         - 62
         - 4.6e3 / 6.5e4
       * - ``rls``, constant at 0.37 from the first row
         - -3.0e13 to 3.1e13
         - 50
         - 7.0e11 / 9.9e12
       * - ``rls``, held at 0
         - 0.41 to 0.53
         - never
         - 0.20 / 0.74
       * - ``ewridge`` (``ridge=1e-6``, solved every row), any of these
         - 0 to 0.51
         - never
         - 0.19 to 0.20 / 0.37 to 0.74

    The fit held for some 30 half-lives in every case measured. Constant from
    the first row at 1,000 and 1e8, the slope first passed 1 at 59 and 76
    half-lives.

    .. rubric:: Parameters

    ``delta``
        The prior's strength: ``A_0 = delta * I``, which is ``P_0 = I / delta``,
        the classic RLS name. Default 1.0, in the units of ``A``, a decayed sum of
        ``w * z z'``: the features' squared, summed over rows. It is
        :func:`ewridge`'s ``ridge`` under ``ridge_scale = "sum"``, not its default
        ``ridge``, a penalty on the means that never fades: this one penalizes the
        intercept too and fades as data arrives. It was called ``ridge`` until
        task 195, and that name is refused naming this one.
    ``coef_prior``
        The coefficients the fit starts from and is shrunk toward, one vector per
        target in the features' original units; zeros when left out.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the model learned from, those with every
    target present, at their raw weights, decayed, where ``weight_sum`` counts every
    row, so rows with a null target do not warm up a fit they never reached
    (docs/PLAN.md task 115 (d)).

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#rls
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rls>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. The factor ``R`` is shared between the targets, so a row with
    any null target is scored and learned from for no target.

    .. rubric:: Example

    .. code-block:: python

        rls = po.spec.rls(
            "rls", targets=["y"], features=["x0", "x1"],
            clock="t", gap_cap=300.0, half_life=600.0,
            delta=1e-3,    # A starts at delta * I: this penalizes the intercept too
        )
        out = po.ModelBank([rls]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`).
    """
    model: dict[str, Any] = {"type": "rls", "delta": delta, "coef_prior": coef_prior}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def lasso(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    lasso_path: list[float],
    l1_ratio: float | None = None,
    select_half_life: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    window_size: float | Duration | None = None,
    closed: str = "right",
    window_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[str, float] | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    target_gaps: str = "own_rows",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A lasso or elastic-net path on running sums, with the penalty chosen as the
    stream runs.

    Coordinate descent on the standardized, centred sums :func:`ewridge` keeps,
    warm-started from the previous solution both along the path of penalties and
    from one solve to the next. Every point of the path is predicted, so choosing
    among them takes no extra solve: ``penalty_selected_<t>`` is the point with
    the lowest exponentially weighted out-of-sample squared error so far.

    .. rubric:: The fit

    For each penalty ``l`` in ``lasso_path``, with ``C`` the feature correlation
    matrix and ``c_i = cov(x_i, y) / s_i``, each feature's covariance with the
    target over the feature's standard deviation. The descent runs until no
    coefficient moves by more than ``tol``:

    .. code-block:: text

        rho_i = c_i - sum_{j != i} C_ij b_j
        b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))
        soft(v, t) = sign(v) * max(|v| - t, 0)

    then unscaled, with the intercept recovered as ``ybar - m . beta``. The target
    is centred but not scaled, so the threshold ``l * l1_ratio`` is in the
    target's units, while the ridge part ``l * (1 - l1_ratio)`` is added to a
    correlation and has none. So for a pure lasso, ``y`` and ``l`` scaled by 10
    scale the predictions by 10 and zero the same coefficients; for an elastic
    net they do not. ``l1_ratio < 1`` is an elastic net.

    .. rubric:: Parameters

    ``lasso_path``
        The penalties, decreasing; required. One instance per point, reported as
        ``pred_<t>__l<lambda>``.
    ``l1_ratio``
        The share of the penalty that is L1. Default 1, the lasso; below 1 an
        elastic net.
    ``select_half_life``
        The half-life of the EW squared out-of-sample error each path point is
        ranked by. Default: the model's half-life; ``inf`` ranks on the plain mean
        over every row so far. ``penalty_selected_<t>`` is reported as it stood before
        the row -- the point this row was scored with, not the one its own error
        then elected. A row of weight 0 adds no error and ages the errors so far,
        so the selection moves only by what the ageing forgets. A target's
        errors count from the row where its own weight reaches its own
        ``min_weight``. A row that threshold withholds from the output adds
        no error, so one target's choice does not depend on another's
        threshold. Before the first scored row ``penalty_selected_<t>`` is
        the path's last point, and a tie between points keeps the first in
        ``lasso_path`` order.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule, as for :func:`ewridge`.
    ``max_iter``, ``tol``
        Within a solve, the descent stops after ``max_iter`` sweeps (default
        100, at least 1) or when no coefficient moves by more than ``tol``
        (default ``1e-10``). A descent that runs out of sweeps first is
        counted in :meth:`polars_online.ModelBank.solve_failures`, one per
        target and path point.
    ``target_gaps``
        Which rows a target's feature correlations are taken over where the target
        is null on some: ``"own_rows"``, the default, or ``"pairwise"``, as for
        :func:`ewridge`. The cross-correlations are centred at the target's own
        means either way.
    ``window_size``, ``closed``, ``window_every``, ``max_rows_between_snapshots``, ``window_budget``
        A hard cutoff on the history the path is fitted from, in clock units, as
        for :func:`ewridge`. A row ``window_size`` old or older is not in the
        sums, and ``closed="both"`` keeps the one exactly that old, as for
        :func:`ewridge`.
        ``window_every`` spaces the snapshots on the clock and
        ``max_rows_between_snapshots`` caps the rows between them, whichever
        comes first, as for :func:`ewridge`; ``window_budget`` bounds each ring
        in MiB and thins or refuses past the bound. The selection error
        is truncated with the sums, so the ``lambda`` chosen is the one that fits
        the window rather than rows the fit has dropped. So a window can change
        the support, not just the coefficients: a feature with no evidence inside
        it goes to exactly zero. ``weight_sum``, ``sigma`` and ``zscore`` are the
        window's too.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#lasso
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#lasso>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per (target, path point) slot, the intercept then one entry per feature
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. The ``pred`` and ``resid`` fields are per target and path
    point, ``pred_<t>__l<lambda>``, and ``penalty_selected_<t>`` is the path point in
    force for the target, by lowest EW out-of-sample error. The sums behind the
    path are :meth:`polars_online.ModelBank.gram`'s, as for :func:`ewridge`.

    .. rubric:: Example

    .. code-block:: python

        las = po.spec.lasso(
            "las", targets=["y"], features=["x0", "x1", "x2"],
            clock="t", gap_cap=300.0, half_life=600.0,
            lasso_path=[0.1, 0.01, 0.001],   # decreasing; every point is predicted
            l1_ratio=1.0,                    # below 1: an elastic net
        )
        out = po.ModelBank([las]).fit_predict(df)
        chosen = out["las"].struct.field("penalty_selected_y")   # the point in force, per row

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``lasso_path`` is required.
    """
    model: dict[str, Any] = {
        "type": "lasso",
        "lasso_path": lasso_path,
        "l1_ratio": l1_ratio,
        "select_half_life": select_half_life,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "max_iter": max_iter,
        "tol": tol,
        "target_gaps": target_gaps,
        "window_size": window_size,
        "closed": closed,
        "window_every": window_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def kalman(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    coef_half_life: float | Duration | list[float | Duration] | None = None,
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float | None = None,
    share_p: bool = False,
    revert_half_life: float | Duration | list[float | Duration] | None = None,
    standardize: bool = True,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A regression whose coefficients are allowed to drift, tracked by a Kalman
    filter.

    Each target's coefficients are a state with a mean and a covariance. A row
    moves them by the Kalman gain, and between rows they drift as a random walk,
    or shrink toward zero under ``revert_half_life``. Use it where a relationship
    moves faster than a half-life can follow, or where a coefficient should be
    forgotten when nothing supports it.

    .. rubric:: The fit

    Per target, per row, with clock delta ``d`` and row weight ``w``:

    .. code-block:: text

        b_j <- Phi b_j                     Phi = diag(2 ** (-d / r_i))      the reversion
        P_j <- Phi P_j Phi                                                  every row
        D_j <- D_j + d                     the clock since P_j's last observation
        P_j <- P_j + Q * D_j ** 2, D_j <- 0                                 the drift, on a row
                                                                            that observes y_j
        s    = z' P_j z + R_j / w
        k    = P_j z / s
        b_j <- b_j + k (y_j - z' b_j)
        P_j <- P_j - k z' P_j

    ``Q`` is the process noise and ``R_j`` the observation noise, the target's EW
    residual variance unless ``obs_var`` is given. That variance is per target,
    and its weight ages on every row, a row with no prediction or with weight 0
    included. Such a row, and a row whose target is null, ages the target's weight
    the same way and learns nothing for it.

    The drift is charged once, on the row that next observes the target, for
    the whole clock since the last one (``D_j``; under ``share_p`` any target's
    row). ``Q * D ** 2`` is not additive over a split gap, so charged per row it
    let a row of weight 0 inside a gap shrink the gap's noise and move every
    later prediction (0.197 at ``coef_half_life = 20``), and a target seen on
    one row in 10 forgot 3.6 times slower on the clock than one seen on every
    row (docs/PLAN.md task 211).

    Until a target has a residual variance, each row takes its noise from its own
    innovation, ``R_j = (y_j - z' b_j) ** 2``, computed before the update, so the
    prediction stays out of sample. The process noise from ``coef_half_life``
    reads the same number. The residual variance starts at the target's first
    prediction. A row whose innovation is exactly 0 says nothing about the size
    of the noise, and changes nothing. Under ``share_p``, until some target has a
    residual variance, the noise is the mean of the squared innovations of the
    targets the row observes. The prior is sized from the squared innovations
    too (``p0`` below), so the filter never sees the target's units: scaling a
    target by ``c`` scales its predictions by ``c``.

    .. rubric:: Parameters

    ``coef_half_life``
        How fast a coefficient may drift, as a half-life on the clock, on
        standardized features. An observation ``D`` clock units after the
        target's last adds the process noise ``sigma^2 * (ln 2 * D / h_i) ** 2``,
        which matches EW-RLS's steady-state gain at that spacing: the same
        half-life whether rows come every unit or every hundredth (docs/PLAN.md
        task 150), and whether the target is seen on every row or one in ten
        (task 211). That is each standardized direction's memory with the
        others held still. Under correlated features a direction of the design's
        correlation with eigenvalue ``lambda`` is learned with a memory of about
        ``h / sqrt(lambda)`` (Ljung and Gunnarsson 1990; measured 37.5 and 160.5
        clock units at a correlation of 0.9 and ``h = 50``, against 36.3 and
        158.1), where EW-RLS forgets every direction at ``h``; and with the EW
        residual variance as ``R``, which a step in the truth inflates, a step
        is learned about 1.45 times slower than with the true noise. A scalar,
        or one value per slot with the intercept first; ``inf`` pins that
        coefficient. Required unless ``q`` is given, and refused beside it. Not
        the spec's ``half_life``, which drives the standardization and the
        residual variance.

        Under ``standardize`` the drift is per standard deviation of each
        feature as the moments stand: a slope's random walk in the caller's
        units has a variance of ``q_i * D ** 2 / s_i ** 2`` per observation,
        a drift measured in the feature's current spread.
    ``q``
        The process noise given outright, ``q_i`` per slot, in place of the
        derivation from ``coef_half_life``; added as ``q_i * D ** 2`` on each
        row that observes the target. Exactly one of the two is given: the
        half-life was required, and ignored beside ``q``.
    ``obs_var``
        A fixed observation noise, in place of the EW residual variance.
    ``p0``
        The prior variance of each coefficient, as a multiple of the noise.
        Unstandardized, ``P_0 = p0 * R * I``, set on the target's first row with
        a noise, from that row's noise, before its correction; until then ``P``
        is unsized, and no process noise is added to it. With ``obs_var``
        given, ``P_0 = p0 * obs_var * I`` from the start, and with no process
        noise the filter is the ridge regression with penalty ``1 / p0``.
        Standardized, each coefficient's prior is ``p0 * R`` on the first row
        that observes the target on which its feature's scale is usable (the
        intercept's always is): ``R`` is the weighted median of the first three
        rows' squared innovations over 0.4549, the median of a chi-squared of
        one degree, so it reads the noise variance as a mean would; and with
        ``obs_var`` it is ``obs_var`` (the intercept then sized from the
        start). Until then the coefficient is 0 and takes no correction. A
        median, so one wild row among the first three, a target at the input
        bound, does not size the prior from itself; such a row once left
        ``P`` with a negative variance and the filter stuck for good. One
        squared innovation falls below 1% of the noise one time in twelve,
        and with no process noise a prior that small pins the fit for good;
        the median of three, about one time in 120. A covariance whose
        diagonal goes below 0 is sized again from the noise as it stands.
        Default 1.0, a prior as uncertain as one observation.
    ``standardize``
        Run the filter on standardized features, so ``coef_half_life`` and ``p0``
        mean the same thing whatever the columns' scale; the reported coefficients
        are in the original units either way. Default ``True``. Each row is
        standardized against the EW moments of the rows before it, ``z_i = (x_i
        - m_i) / s_i``. Without an intercept the features are scaled by their
        root mean square and not centred, as for :func:`ewridge`. With
        ``standardize = False``, ``q = 0`` and a fixed ``obs_var`` the filter is
        exactly Bayesian linear regression.

        The filter's state is held in the coordinates of an *anchor*, the
        moments at its last re-map, and maps the process noise and the priors,
        defined in the moments as they stand, into them exactly. When the
        moments drift from the anchor -- a scale by more than twice or less
        than half, a mean by more than the anchor's scale -- the state follows
        them to their new coordinates, ``b <- A b`` and ``P <- A P A'``:

        .. code-block:: text

            A_00 = 1,    A_0i = (m_i - m^a_i) / s^a_i,    A_ii = s_i / s^a_i

        with ``m^a``, ``s^a`` the anchor and ``m``, ``s`` the moments after the
        row. So the moments' moving moves no prediction, no predictive variance
        and no coefficient in the original units, from the first row: no row's
        raw reading of a feature reaches the state, and features scaled by a
        power of two predict the same to the bit. Read through the moments as
        they stood, a finite half-life's own wander moved every prediction: at
        a half-life of 50 and R² 0.99998 the out-of-sample error was 223 noise
        variances above the noise (docs/PLAN.md task 206). Task 206 re-mapped
        on every row; the anchor is the same filter to rounding, a fifth
        cheaper a row at ten features (task 211). A re-map past what a stretch
        of data makes -- a
        scale by more than 1024 times either way, or a mean by more than 1024
        of the anchor's scale -- is not followed, and the state is read in the
        new coordinates.

        The model before task 206, a coefficient per current standard deviation
        that moves with the scale, is this one on a column z-scored in the
        stream, given with ``standardize = False``: the README's *Features in
        units of their spread* builds the column with
        :func:`polars_online.ewm_mean` and :func:`polars_online.ewm_std`
        (docs/PLAN.md task 212).
    ``revert_half_life``
        A reversion half-life ``r_i`` per slot: between observations the
        coefficient shrinks toward zero by ``2 ** (-d / r_i)``, so a coefficient
        no row has supported for a while is forgotten rather than carried. Default
        ``inf``, the random walk, which adds nothing. A scalar applies to every
        slot, the intercept included; a list gives one per slot, intercept first,
        and ``[inf, r, r]`` leaves the intercept a random walk. The pull is toward
        zero in the standardized coordinates when ``standardize`` is on: a slope
        toward "no effect", the intercept toward "the target averages zero". A
        reverting slot's process noise for an observation ``D`` after the last
        is ``q_i * g_i(D) ** 2`` with ``g_i(D) = (1 - 2 ** (-D / r_i)) /
        theta_i`` and ``theta_i = ln 2 / r_i``: ``q_i * D ** 2`` for a gap well
        under ``r_i``, and never more than ``q_i / theta_i ** 2``, so a
        coefficient's uncertainty stays bounded across a run of rows that
        observe nothing. At observations one clock unit apart that is short of
        ``q_i`` by about ``ln 2 / r_i``. A reverting slot settles, at
        observations ``D`` apart, at the prior variance ``q_i * g_i(D) ** 2 /
        (1 - phi_i ** 2)``, a stationary AR(1) instead of an unbounded walk.
        The reversion runs on every row, a row that
        observes nothing included, being the same over a gap however it is
        cut. A prediction propagates the state by the same ``Phi`` over the
        row's clock gap.
    ``share_p``
        Keep one ``P`` for every target, driven by the mean ``sigma^2`` over the
        targets that have one, where by default ``P`` is per target because the
        recursion depends on ``sigma^2_j``. A target with no residual variance
        yet is left out of the mean, not counted as a noise of 0. ``P``'s
        recursion never reads ``y``, so targets that share the noise would each
        carry the same ``P``: every target observed on a row takes its gain from
        ``P`` as the row finds it, and ``P`` takes the row once. The order of
        the targets moves no prediction, and a target beside a copy of itself
        present on the same rows predicts as it would alone. A copy null on
        some of those rows moves nothing until it has a residual variance; from
        then its variance, learned from fewer rows, enters the mean. With a
        noise ``sigma^2_j`` of its own, a target's own filter would keep ``P``
        times ``sigma^2_j`` over the mean, and that is the covariance its
        ``se_coef`` reads (docs/PLAN.md task 211): read off
        ``P`` as it stands, two targets of noise 0.01 and 1 had standard errors
        7.3 times too large and 1.39 times too small. Default ``False``: one
        ``P`` per target, the exact filter for each target under its own noise.

        When sharing helps. With a constant noise, ``P / sigma^2`` and the gain
        never read ``sigma^2``, since the process noise is ``sigma^2`` times
        ``(ln 2 / coef_half_life) ** 2``; so the shared filter and the
        per-target ones part only through how the ``sigma^2_j`` paths move.
        Sharing helps where the targets are related, so their noises move
        together; where each target's own ``sigma^2_j`` is a noisy estimate, a
        short or sparse history, which the mean over the targets smooths; and
        for cost, one ``k x k`` update a row for every target. On
        ``docs/VALIDATION.md`` section 4's two targets, the next minute's
        return and the next five minutes', sharing scored R² -0.0110 against
        -0.0168 on the first and -0.0880 against -0.0746 on the second.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#kalman
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kalman>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the original
        units (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` moves the coefficients
    by ``Phi`` over the clock distance from the last learned row, capped by
    ``gap_cap``. So however far past the data, a coefficient has shrunk by
    ``2 ** (-gap_cap / r_i)`` at most, and the prediction is the intercept
    alone only where ``gap_cap`` spans several of the slopes' reversion
    half-lives and the intercept's is ``inf``.

    ``se_coef`` reads the filter's own covariance ``P`` and ``error_inflation``
    its prior ``z' P z / R``: both exact *under the model*, which says the
    coefficients walk at the noise ``coef_half_life`` implies. On a truth that
    does not move, that walk is variance the coefficients do not have, and the
    standard errors come out about ``sqrt(2)`` too large (a mean squared error
    over ``se_coef ** 2`` of 0.47 to 0.50 at ``coef_half_life`` 50 and 200;
    0.95 to 1.06 on a walk of the implied variance). And with ``R`` the EW
    variance of the out-of-sample residuals, which already carry the
    estimation error, ``error_inflation`` counts that error twice: the mean
    squared error over ``R * error_inflation ** 2`` is 0.96 at
    ``coef_half_life`` 50.

    .. rubric:: Example

    .. code-block:: python

        revert = po.spec.kalman(
            "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", gap_cap=10.0,
            half_life=200.0,          # the observation-noise estimate forgets at this rate
            coef_half_life=100.0,     # how fast a coefficient may drift
            revert_half_life=[float("inf"), 50.0, 50.0],  # the slopes shrink toward zero unobserved
        )
        out = po.ModelBank([revert]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` for
    neither or both of ``coef_half_life`` and ``q``.
    """
    model: dict[str, Any] = {
        "type": "kalman",
        "coef_half_life": coef_half_life,
        "q": q,
        "obs_var": obs_var,
        "p0": p0,
        "share_p": share_p,
        "revert_half_life": revert_half_life,
        "standardize": standardize,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def huber(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    huber_delta: float | None = None,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Huber regression: :func:`ewridge` with the rows that miss by far
    down-weighted.

    Each row's weight is scaled by the Huber weight of its prior residual, the
    residual of the prediction made before the row is learned, so the reweighting
    stays out-of-sample. A row that misses by more than ``huber_delta`` standard
    deviations pulls the fit less the further it misses.

    .. rubric:: The fit

    With ``d = huber_delta`` and ``s`` the EW standard deviation of the target's
    residuals:

    .. code-block:: text

        w_robust = 1              if |r| <= d * s
                 = d * s / |r|    otherwise
        the ewridge update, at weight w * w_robust

    The weights are per target, so the sums are per target here (one accumulator
    each), where :func:`ewridge` shares one across targets.

    ``s`` is not itself robust. It is the plain EW standard deviation of the
    residuals, in which the rows the cut down-weights count at full weight, so a
    burst of outliers widens the cut for the rows after it until the EW mean
    forgets them. ``huber_delta`` is in units of that std, not of a robust one
    such as a MAD. Its weight ages on every row the model sees, a row with no
    prediction or with weight 0 included, so it forgets across a gap as the clock
    says.

    The cut stays in the residual's std where the insensitivity bands of
    :func:`pa` and :func:`sgd` are in the target's own. A row beyond the cut
    still teaches, at a weight that shrinks with its miss, so a cut that early
    residuals widen does not stop the fit learning. A row inside a band
    teaches nothing, so a band that early residuals widen does: a fit from
    zero coefficients has the target's whole level for its first residuals.

    Until a target has an ``s`` above 0, no row is down-weighted. That is before
    its first residual, or while every residual so far is exactly zero, and
    there is then nothing to judge an outlier against. ``s`` was taken as 1
    there, a cut in the target's own units, so a target in millions had its
    first predicted rows down-weighted and one in millionths none.

    .. rubric:: Parameters

    ``huber_delta``
        The cut, in units of ``s``: a residual within it counts at full weight.
        Default 1.345, the cut at which Huber's estimator is 95% as efficient as
        least squares when the errors are normal (Huber 1981): with ``Z``
        standard normal and ``ψ`` the residual clipped at ``±k``, the efficiency
        is ``(2Φ(k) − 1)² / E[ψ(Z)²]``, which is 0.95 at ``k = 1.345``.
        statsmodels' ``HuberT`` takes 1.345 and scikit-learn's
        ``HuberRegressor`` 1.35. ``inf`` cuts nothing, which is least squares.
        :func:`sgd`'s ``huber_delta`` is the same constant in the same unit.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``), correlation-form
        solving, and the solve schedule. ``ridge`` is on the mean-form Gram, in
        the features' squared units, and dimensionless under ``standardize``,
        which solves in correlation form.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed. The reweighted sum, which an outlier lowers, is not what it
    reads, so a stream whose warm-up meets outliers reports its first prediction
    when the rows say to.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#huber
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#huber>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them.

    .. rubric:: Example

    .. code-block:: python

        hub = po.spec.huber(
            "hub", targets=["y"], features=["x0", "x1"],
            clock="t", gap_cap=300.0, half_life=600.0,
            huber_delta=2.0,    # a residual beyond 2 sigma is down-weighted
        )
        out = po.ModelBank([hub]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`).
    """
    model: dict[str, Any] = {
        "type": "huber",
        "huber_delta": huber_delta,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def quantile(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    quantile: float,
    ridge: float | None = None,
    standardize: bool = False,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    quantile_eps: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantile regression at level ``quantile``: the fit that a share ``tau`` of the
    targets fall below.

    One Newton step per row on the check loss smoothed by a uniform kernel,
    linearised at the fit the row was scored with, so it stays out-of-sample. Use
    it for a conditional median, which outliers cannot pull, or for a tail, such
    as the 0.9 quantile of a return, where a mean says nothing.

    .. rubric:: The fit

    With ``psi(r) = tau - 1{r < 0}``, ``s`` the EW standard deviation of the
    target's residuals and ``h`` the band's half-width:

    .. code-block:: text

        |r| <  h:  a least-squares row, with target y + 2h (tau - 1/2)
        |r| >= h:  no weight in the sums; 2h * psi(r) * z into the cross-moment

    The smoothed loss has the same curvature on either side of the fit, so the
    rows' own history cancels and the fit converges to the quantile regression.
    Reweighting the rows by the check loss's IRLS weight, ``psi(r) / r``, does not
    converge to it. That weight is the secant of this step where the step is its
    tangent, and it is unbounded as ``r`` goes to zero; a row whose prior residual
    happened to be near zero then holds the fit near the fit it was scored by.
    Measured at the median of a skewed noise after 20 000 rows against
    ``statsmodels``' ``QuantReg``, the step is 0.005 away, inside ``QuantReg``'s
    own standard error of 0.007, where the weights settled 0.164 short (review
    2026-09-12, N9).

    The band. ``h`` is ``quantile_eps * s``. The rows inside the band are the
    curvature the step leans on, so a much narrower band converges more slowly and
    a much wider one smooths the quantile toward the mean. The band is never
    narrower than ``(k / n) ** 0.4`` of ``s``, for ``k`` coefficients and the
    target's effective sample ``n`` -- its present rows counted one each,
    decayed, whatever their weights -- which is the smoothed-quantile bandwidth
    rate. Under a half-life the band's share of the sample is a few rows, and the
    floor is what keeps the step fed there; a long stream leaves the floor behind.
    Coverage at ``quantile = 0.9`` on normal noise reads 0.895 at ``half_life =
    30`` and 0.900 from 200 up (the second review of 2026-09-15, F3).

    Warm-up and rebuilding. Under three rows per coefficient of the rows the
    target was present on, the fit is ordinary least squares: a Newton step needs
    a Hessian, and a band around a fit built from a handful of rows is not one.
    The same rule rebuilds the fit after a gap or a reset has aged the weight
    away. A band holding under one row per coefficient, in rows of the target's
    mean weight, takes least-squares rows too, until it holds rows again. That is
    what rebuilds a fit a row at the input bound leaves behind. Such a row sets
    the sums at its own scale, and every later row is outside the band. Only
    nudges arrive, each ``2h * psi * z`` over the band's weight; they cannot move
    the fit until that weight has decayed to nothing, and each is then a step that
    outgrows the band. From one row up an outside row's step lands inside the
    band, and the floor keeps a settled band well clear of one row, so the rule
    does not fire in steady state. Both rules count rows, so a weight's scale
    reaches neither: they compared the rows' weight with a row count, and rows at
    weight 100 left the warm-up on their first row, where the fit reached 1e51
    (docs/PLAN.md task 147).

    Until a target has an ``s`` above 0, its rows are least squares too. That
    is before its first residual, or while every residual so far is exactly
    zero, and the band, a width in units of ``s``, has nothing to be drawn in.
    ``s`` was taken as 1 there, so a band drawn past the warm-up was in the
    target's own units, and a target in millionths was thrown far off.

    .. rubric:: Parameters

    ``quantile``
        The level ``tau``, in ``(0, 1)``; required. 0.5 is a median regression.
    ``quantile_eps``
        The band's half-width in units of ``s``. Default 0.2, at which the band
        holds about a fifth of a stream at the median and a fifteenth at the 0.9
        quantile.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``), correlation-form
        solving, and the solve schedule. ``ridge`` is on the mean-form Gram, in
        the features' squared units, and dimensionless under ``standardize``,
        which solves in correlation form.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed. The band's weight, which a half-life caps at the band's share
    of the sample, is not what it reads.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#quantile
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#quantile>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).
    ``support_coef``
        On ``coef``'s rows, each coefficient's data share, laid out like
        ``coef``: how much of the fit after the row the data, as the loss
        weighs the rows, determined rather than the ridge
        (:mod:`polars_online.spec`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is the conditional quantile, and a ``resid`` is
    signed against it, so a share ``tau`` of them are positive once the fit has
    settled.

    .. rubric:: Example

    .. code-block:: python

        med = po.spec.quantile(
            "med", targets=["y"], features=["x0", "x1"],
            clock="t", gap_cap=300.0, half_life=600.0,
            quantile=0.5,        # the level: a median regression
            quantile_eps=0.2,    # the band the step leans on, in units of sigma
        )
        out = po.ModelBank([med]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``quantile`` is required
    and must be strictly between 0 and 1.
    """
    model: dict[str, Any] = {
        "type": "quantile",
        "quantile": quantile,
        "ridge": ridge,
        "standardize": standardize,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "quantile_eps": quantile_eps,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def ftrl(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    alpha: float | None = None,
    beta: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    strict_binary: bool = False,
    loss: str = "logistic",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """FTRL-proximal: online logistic regression for a 0/1 target, or a sparse linear
    regression, with per-coordinate learning rates.

    McMahan et al.'s (2013) algorithm, the standard for click prediction: a
    gradient method whose L1 penalty zeroes a coefficient with too little
    evidence, so the fit is sparse, and whose per-coordinate rates adapt to
    each feature's history. Its sums decay on the model's clock like every
    other model's.

    .. rubric:: The fit

    With ``z`` the row's feature vector, intercept included, and ``zz``,
    ``n`` and ``d`` the per-coordinate sums:

    .. code-block:: text

        b_i   = 0 if |zz_i| <= l1 else -(zz_i - sign(zz_i) l1) / (r_i + l2)
        r_i   = beta / alpha + d_i                 (under a half-life)
              = (beta + sqrt(n_i)) / alpha         (without one: river's closed form)
        p     = sigmoid(z . b)                     (z . b itself under loss="squared")
        n_i   <- lam * n_i ;  zz_i <- lam * zz_i ;  d_i <- lam * d_i
                                                   (on a row that teaches the target,
                                                    over the clock since the last one)
        g_i   = (p - y) * z_i * w
        s_i   = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
        zz_i += g_i - s_i * b_i ;  n_i += g_i^2 ;  d_i += s_i

    Under a half-life the fit after a row that teaches the target minimizes
    FTRL's objective: a fixed regularizer, ``l1 |b|_1 + (beta / alpha + l2)
    |b|^2 / 2``, against the rows' linearized losses and proximal terms,
    each weighted by its age on the clock. So old evidence counts for less
    against the prior. In steady state the penalties act as a mean-scale
    ridge of ``(1 - lam) * (beta / alpha + l2)``: a constant 5 settles at
    4.65 at ``half_life = 100`` and 4.96 at 1000.

    A row that teaches the target nothing -- absent, at weight 0, a label
    ``strict_binary`` refuses, or a row whose squared gradient would
    overflow, which is skipped -- leaves the sums as the last row that
    taught it left them. So the fit does not move, as :func:`ewridge`'s does
    not, and a row of weight 0 is clock alone, as in every model. The next
    row that teaches is scored with that fit, then ages the sums by the
    whole clock since and learns. The gap ages the evidence and not the
    prior: a constant 5 settled at ``half_life = 100`` stands at 0.75 of
    itself after that row when the gap was one half-life, and at 0.15 when
    it was five. Without a half-life the fit is river's ``FTRLProximal`` to
    the bit.

    A row's weight is an importance weight, as Vowpal Wabbit's: the gradient
    carries it, against penalties in absolute weight. So a heavier stream
    overcomes ``l1`` and ``l2`` sooner (``tests/test_second_opinion.py``
    holds the fit to VW's, weighted). ``l1``, ``l2`` and ``beta`` are a prior of fixed
    mass against evidence that grows with weight and density, which FTRL's
    regret bound rests on. So under a half-life the effective penalty is
    ``l1 / W`` for the weight ``W`` the window holds, and more rows in a
    half-life, or heavier ones, outweigh it sooner. For a penalty on the
    mean scale, invariant to both, use :func:`lasso`.

    .. rubric:: Parameters

    ``loss``
        ``"logistic"`` (the default) for a 0/1 target: ``pred`` is a
        probability. ``"squared"`` for a continuous target: ``pred`` is the
        linear prediction, and the model is a sparse linear regression with
        no solves and the L1 support :func:`ewridge` has not got. The two
        differ only in the link; the gradient is ``(p - y) * z`` either way.
    ``alpha``, ``beta``
        The learning-rate scale and its smoothing. Defaults 0.1 and 1.0.
    ``l1``, ``l2``
        The penalties: ``l1`` zeroes a coefficient whose evidence is below
        it. Defaults 0.0 and 1.0.

    None of the four is free of units. A coefficient is ``zz_i`` over
    ``r_i + l2``, and ``g_i`` is the target (a probability, under the
    logistic loss) times the feature, at the row's weight. So ``beta`` is in
    the units of ``sqrt(n_i)``, a gradient, ``alpha`` in the coefficient's,
    the target per unit of the feature, ``l1`` in the summed gradient's, and
    ``l2`` in ``r_i``'s, the feature squared. A feature scaled by ``c`` fits
    the same model at ``alpha / c``, ``beta * c``, ``l1 * c`` and ``l2 * c**2``.

    ``ftrl`` does not centre a feature. A coefficient's error reaches the
    prediction multiplied by the feature's value, so a feature that sits at a
    level ``L`` carries it at ``L`` times its size. On the stream of
    ``crates/online-core/tests/held_values.rs``, with one of three features at
    ``L``, the squared loss's error is about ``0.018 * L``: 18.3 at 1,000 and
    at -1,000, 1.8e6 at 1e8, where it is 0.75 at 0.5. Standardize or z-score
    such a feature before the bank, as the README's *Features in units of
    their spread* shows. There is no ``standardize`` here because ``l1``
    zeroes a coefficient in the feature's own units: a scale inside the model
    would move which coefficients it zeroes as the feature's spread moved.

    ``strict_binary``
        Refuse a chunk whose target is not 0 or 1, naming the row, before any
        stream is touched. Default ``False``: such a target is clamped into
        ``[0, 1]``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row. So rows with a
    null target do not warm up coefficients they never moved (docs/PLAN.md
    task 115 (d)). A label ``strict_binary`` refuses is not one of them, nor
    a row skipped because its squared gradient would overflow.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#ftrl
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ftrl>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is a probability under ``"logistic"``, and
    ``emit_metrics`` then reads the fit as probabilities against labels:
    accuracy at 0.5, the Brier skill score and the point-biserial
    correlation, under their usual names.

    .. rubric:: Example

    .. code-block:: python

        click = po.spec.ftrl(
            "click", targets=["y"], features=["x0", "x1"], half_life=500.0,
            loss="squared",           # y here is continuous; "logistic" for a 0/1 target
            alpha=0.1, beta=1.0,      # the learning-rate scale and its smoothing
            l1=0.01, l2=0.0,          # l1 zeroes a coefficient whose evidence is below it
        )
        out = po.ModelBank([click]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``alpha``, ``beta``,
    ``l1`` and ``l2`` refuse ``inf`` and ``NaN``.
    """
    model: dict[str, Any] = {
        "type": "ftrl",
        "alpha": alpha,
        "beta": beta,
        "l1": l1,
        "l2": l2,
        "strict_binary": strict_binary,
        "loss": loss,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def ew_cov(
    name: str,
    *,
    features: list[str],
    stats: list[str] | None = None,
    precision_prior: float | None = None,
    mahal_quantiles: list[float] | None = None,
    pca: int | None = None,
    pca_every: float | Duration | None = None,
    max_rows_between_pca: int | None = None,
    lags: list[int] | None = None,
    gram_threads: int | None = None,
    window_size: float | Duration | None = None,
    closed: str = "right",
    window_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted moments of the feature columns: means, variances,
    correlations, and what is read off them.

    Not a regression: there are no targets and no coefficients, only running
    statistics of the columns named, decayed on the same clock as every
    model here. Values are read from the state before each row, so an
    ``ew_cov`` output can be a feature for that same row without leaking it.

    .. rubric:: The recursion

    The means and the centred co-moments ``C`` are kept as weighted means, in
    Welford's form:

    .. code-block:: text

        W'     = lam * W + w
        a      = lam * W / W'        b = w / W'        (a + b = 1)
        delta  = x - m
        m'     = m + b * delta
        C'_ij  = a * C_ij + a * b * delta_i * delta_j

    so ``var_i = C_ii``, ``cov_ij = C_ij`` and ``corr_ij = C_ij / sqrt(C_ii
    C_jj)`` are read off directly. A variance stays accurate when the columns
    sit on a large offset, where the raw ``E[x^2] - m^2`` form loses it. One
    O(k²) update per row, which replaces the O(k²) passes a pure-Polars
    pairwise EW correlation needs.

    .. rubric:: Parameters

    ``stats``
        Which statistics to write, from ``mean``, ``var``, ``std``, ``cov``,
        ``corr``, ``partial_corr``, ``mahal`` and ``lag_corr``. Default
        ``["mean", "std", "corr"]``. ``[]`` writes no statistic, only
        ``weight_sum``, ``settled_frac`` and ``withheld_reason``, and
        accumulates all the same. The spec's value is then its state,
        read back with :meth:`polars_online.ModelBank.gram` and
        :meth:`polars_online.ModelBank.describe`. That is the form for a wide
        set of columns, where even the means are k values per row nobody
        reads.
    ``precision_prior``
        A ridge on the co-moments, needed by ``partial_corr`` and ``mahal``:
        the precision matrix is ``(C + s * prior * I)^-1``, solved on each
        row it is read (O(k³), only when asked for). Like an RLS prior it
        fades as data accumulates.
    ``mahal_quantiles``
        Levels at which to keep the exponentially weighted quantile of the
        past ``mahal`` scores (``mahal_q<p>``), at the model's half-life and
        each row's weight, within ``tanh(1/128)`` (0.78%) of the exact one.
        That is a threshold from the stream's own history instead of a
        table, so ``mahal > mahal_q0.99`` is one row in a hundred without
        assuming a distribution. The row's own score joins after it is read.
    ``pca``
        Track the top ``pca`` principal components of the covariance, per
        component ``j``:

        .. list-table::
           :header-rows: 1
           :widths: 34 66

           * - field
             - what it holds
           * - ``pc<j>_var``
             - its eigenvalue
           * - ``pc<j>_share``
             - its share of the total variance
           * - ``pc<j>_loading_<feature>``
             - its unit loading on each column, largest entry positive
           * - ``pc<j>_score``
             - the row's coordinate ``v_j . (x - m)``

        Each refresh keeps the previous sign, so a loading never flips. The
        components are the same to the bit whatever the bank's thread count
        (``POLARS_ONLINE_MAX_THREADS``): the eigensolver runs at a fixed
        parallel degree of eight.
    ``pca_every``, ``max_rows_between_pca``
        The eigendecomposition, O(k³), is refreshed after the row is folded
        in, every ``pca_every`` clock units or every ``max_rows_between_pca``
        rows, whichever comes first, as ``solve_every`` and
        ``max_rows_between_solves`` schedule a regression's solve.
        ``pca_every`` is a number of the clock's units, a duration on a
        temporal clock (``"5m"``), or ``0`` for every row; without a clock
        column the clock is the row's number. With neither given, every row.
        A row of weight zero advances both, as the window's snapshot cadence
        counts it, and a gap capped at ``gap_cap`` counts as the cap. Between
        refreshes the loadings are frozen, so a row's scores never depend on
        chunking.
    ``lags``
        Lagged cross-moments beside the contemporaneous ones. With ``W`` and
        ``m`` the weight and mean before the row, and both deviations against
        that mean:

        .. code-block:: text

            C_l' = a * C_l + a * b * (x_t - m) (x_{t-l} - m)'

        the same ``a`` and ``b`` the co-moments use, so lag 0 would be
        ``comoments`` exactly. The lagged statistics are defined by this
        recursion, both legs centred at the mean before each row, not as a
        batch EW lagged covariance about the final mean, which a frame in
        memory would compute. Lags are counted in learned rows within the
        group, not clock units, and must be strictly increasing, ``>= 1`` and
        at most 2^20 (1,048,576), the ring being sized before the first row;
        the list order is the output order. The ring of past rows is emptied
        on a session change and on a clock gap beyond ``gap_cap``, one row's
        or a run of skipped rows' whose total the ceiling cut. Those are the
        events after which "the row ``l`` back" is not a row ``l`` ago. A
        zero-weight row ages the matrices and does not enter the ring. Add
        ``"lag_corr"`` to ``stats`` to write ``lag_corr_<a>_<b>_l<l>`` =
        ``C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`` per lag and ordered pair, the
        auto terms included. That is ``k²`` slots a lag, since a lagged
        matrix is not symmetric. Or read ``lags`` and ``lag_comoments`` (an
        ``(L, k, k)`` array) from :meth:`polars_online.ModelBank.gram`.
    ``gram_threads``
        Run the per-row update of the ``k x k`` co-moments on up to that
        many threads, as :func:`ewridge`'s ``gram_threads`` does. Default 1.
        The output is the same to the bit at every count. The update is bound
        by memory: at 2,000 features a bank learned 2.5 times as fast on eight
        threads and 3.1 times on fourteen. Below 725 features a row's update
        stays on one thread, and the lagged cross-moments of ``lags`` always
        do.
    ``window_size``, ``closed``, ``window_every``, ``max_rows_between_snapshots``, ``window_budget``
        A hard cutoff on the history, in clock units: a row ``window_size``
        old or older contributes nothing at all (``closed="both"`` keeps the
        one exactly that old, as for :func:`ewridge`), where the exponential
        weight alone would still leave ``0.5 ** (age / half_life)`` of it --
        12.5% at three half-lives. Inside the window the weights are still
        exponential, so this is not a rolling flat mean: the newest row
        dominates exactly as it does without a window. It is exact, because
        an exponentially weighted sum contains its own past. Everything at or
        before a time ``u`` is ``lam ** (t - u)`` times the accumulator as it
        stood at ``u``, so subtracting that leaves precisely the rest. What
        the model keeps is a ring of snapshots, one per row by default: the
        one place in this library where memory grows with a window rather
        than with the state. A snapshot is ``k² + k + 3`` eight-byte numbers,
        so a 1,000-row window over 20 columns is about 3 MB per group.
        ``window_every`` spaces the snapshots on the clock (``"1m"``, or a
        number of the clock's units) and ``max_rows_between_snapshots`` caps
        the rows between them, whichever comes first, as for :func:`ewridge`.
        Either divides the memory, and a clock spacing alone holds a ring to
        at most ``window_size / window_every + 2`` snapshots whatever the row
        rate. ``window_budget`` bounds each ring in MiB and thins or refuses
        past the bound, as for :func:`ewridge`.

        Four things to know before reading windowed numbers:

        .. list-table::
           :header-rows: 1
           :widths: 34 66

           * - the caveat
             - why
           * - the guarantee is one-sided
             - the boundary is the oldest snapshot still inside the window,
               so what is dropped is always a superset of what the window
               excludes; under a coarser cadence the effective window is
               shorter than asked by at most one snapshot's spacing, never
               longer: in ``[window_size - window_every, window_size]`` in
               clock units, exactly, under a clock spacing, with
               ``window_every`` the spacing in force (a thinning budget
               doubles it)
           * - the clock is the decayed one
             - ``window_size`` is measured on the clock the decay uses, after
               ``gap_cap`` caps a gap and after a ``session_gap`` is applied
           * - the edge is a discontinuity
             - a row ageing out drops its whole weight at once, so a windowed
               series has small steps an EWMA does not
           * - it is a subtraction
             - precision falls with the fraction discarded: negligible at
               ``window = 3 * half-life`` (an eighth of the mass), worse as
               the window shortens toward the half-life

        ``weight_sum`` becomes the weight inside the window, which stops
        growing once the window fills, so ``min_weight`` gates on a quantity
        with a ceiling. A clock gap longer than ``window_size`` empties it.
        The row after the gap reads the window as the row before the gap
        left it, since a row's fields are read before its own decay, as
        ``weight_sum`` is; the rows after it report nulls rather than stale
        numbers, until the window holds ``min_weight`` again. ``mahal``,
        ``partial_corr`` and the PCA read the window's moments too, and the
        PCA refresh is gated on the window's weight. ``window_size`` does not
        combine with ``lags`` or ``mahal_quantiles``, which accumulate over a
        history it does not truncate; both are refused by name.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by
    name: ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, holding the statistics ``stats``
    asks for: ``mean_<column>``, ``var_<column>`` and ``std_<column>`` per
    column; ``cov_<column>_<column>``, ``corr_<column>_<column>`` and
    ``partial_corr_<column>_<column>`` per pair, unordered, ``i < j``
    (``mean_x0``, ``corr_x0_x1``); ``lag_corr_<a>_<b>_l<l>`` per lag and
    ordered pair; ``mahal`` and ``mahal_q<p>``, the ``pc<j>_*`` fields,
    ``weight_sum``, and ``settled_frac`` and ``withheld_reason`` as
    everywhere. The statistics are
    null until ``min_weight``, which the bank floors at 2: a variance needs two
    rows, so a lower ``min_weight``, ``0`` included, is raised to 2 rather than
    refused. The plain spec's fields are listed in
    `docs/OUTPUTS.md#ew_cov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_cov>`_.
    The moments themselves are :meth:`polars_online.ModelBank.gram`'s
    ``means``, ``comoments``, ``lags`` and ``lag_comoments``. A group that
    closes writes them to :meth:`polars_online.ModelBank.closed_groups` with
    ``eig_vals`` and ``eig_vecs`` under ``pca``.

    .. rubric:: Example

    .. code-block:: python

        mv = po.spec.ew_cov(
            "mv", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
            stats=["mean", "std", "corr", "partial_corr", "mahal"],
            precision_prior=1e-6,        # needed by partial_corr and mahal
            mahal_quantiles=[0.99],      # mahal_q0.99: one row in a hundred, from the history
            pca=1, pca_every=20,         # refreshed every 20 units of t; pc0_var, pc0_share, ...
        )
        scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")
        odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers

    For moments on data that fits in memory, polars already does this --
    ``df.rolling(clock, period=...).agg(...)`` with an exponential weight
    agrees to 1e-14. The reason to reach for the spec is a stream, a saved
    state, or the work: polars recomputes each window, ``O(n * W)``, where
    this is ``O(n)``.

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got.
    """
    model: dict[str, Any] = {
        "type": "ew_cov",
        "stats": stats,
        "precision_prior": precision_prior,
        "mahal_quantiles": mahal_quantiles,
        "pca": pca,
        "pca_every": pca_every,
        "max_rows_between_pca": max_rows_between_pca,
        "lags": lags,
        "gram_threads": gram_threads,
        "window_size": window_size,
        "closed": closed,
        "window_every": window_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    # Written only when given, as the Rust spec skips it when absent.
    if gram_threads is None:
        del model["gram_threads"]
    targets = _mirror_target(name, "ew_cov", features, common, "its statistics are over")
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def sgd(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    loss: str = "squared",
    huber_delta: float | None = None,
    quantile: float | None = None,
    eps: float | None = None,
    learning_rate: float | None = None,
    schedule: str = "constant",
    power: float | None = None,
    l2: float | None = None,
    clip_gradient: float | None = None,
    standardize: bool = True,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    strict_binary: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Stochastic gradient descent with a choice of losses: one gradient step per
    row, no solves.

    The cheap baseline, O(k) per row where the exact solvers are O(k²), and the
    only model here that takes count targets (``loss = "poisson"``). The
    coefficients can be constrained (bounded, summed to a constant, held on the
    simplex), which makes it the model for portfolio and mixing weights.

    .. rubric:: The fit

    With ``eta = z . b``, ``p = link(eta)`` and ``d = dL / d eta``:

    .. list-table::
       :header-rows: 1

       * - loss
         - link
         - ``p``
         - ``d``
       * - ``squared``
         - identity
         - ``eta``
         - ``p - y``
       * - ``huber``
         - identity
         - ``eta``
         - ``clamp(p - y, +/- delta * s)``
       * - ``quantile``
         - identity
         - ``eta``
         - ``1{y < p} - tau``
       * - ``epsilon_insensitive``
         - identity
         - ``eta``
         - 0 within ``eps * s_y`` of ``y``, else ``sign(p - y)``
       * - ``poisson``
         - log
         - ``exp(clamp(eta, +/- 30))``
         - ``p - y``
       * - ``logistic``
         - sigmoid
         - ``sigmoid(eta)``
         - ``p - clamp(y, 0, 1)``

    then ``g_i = d * z_i * w + l2 * b_i`` for a slope, and ``g_0 = d * w`` for
    the intercept, which is not penalised; each ``g_i`` is clamped to ``+/-
    clip_gradient``; then ``b_i -= lr_i * g_i``. With ``standardize=False``,
    ``z = [1, x]`` and ``b`` is in the caller's units; under ``standardize``,
    the default, ``z`` is the row standardized and the step, once the scaler
    has warmed up, is mapped into the caller's units (``standardize`` below).
    The Poisson link clamps ``eta`` to ``+/- 30`` before the ``exp``, so a
    Poisson prediction never exceeds ``e ** 30``, about ``1.07e13``.

    ``s`` is the EW standard deviation of the target's out-of-sample
    residuals, as the row arrives, as for :func:`huber`. Its square is the EW
    mean of ``(y - p) ** 2`` over the rows with the target, a weight above 0
    and a prediction, each joining after its own step; its weight ages on
    every row. ``s_y`` is the target's own EW standard deviation as the row
    arrives: the spread of ``y`` around its EW mean, over the rows with the
    target and a weight above 0, whatever the fit. Until the target has an
    ``s`` above 0 the Huber row is a squared-loss row, and until it has an
    ``s_y`` above 0, two weighted rows of different values, the tube has no
    width. In the target's own units, a cut and a tube fitted one scale and
    failed the others: a target in thousandths never left a tube of 0.1, so
    it was never learned, and the cut never bound on it. With ``s`` and
    ``s_y``, the squared and Huber losses fit a target scaled by ``c`` as the
    unscaled one, scaled by ``c``.

    The cut and the tube are in different units because the fit starts from
    zero coefficients, so its first residuals are the target's whole level.
    Beyond the cut the Huber gradient is clipped, not zero, so a cut those
    residuals widen still lets every row teach. Inside the tube the gradient
    is zero. A tube drawn in the residual's std on a target at 1,000 with a
    spread of 2 was about 100 wide and held every row. Without decay it never
    narrowed, and the fit stopped where it stood.

    .. rubric:: Parameters

    ``loss``
        One of the table's. Default ``"squared"``. ``epsilon_insensitive`` has a
        sign-valued subgradient, so a constant rate oscillates in a band around
        the optimum; use ``schedule = "inv_scaling"`` with it. ``poisson`` takes
        counts: a chunk with a negative target is refused, naming the row,
        before any stream is touched, as scikit-learn's ``PoissonRegressor``
        refuses one. Taken as it stood, ``p - y`` drove the prediction down to
        the link's floor, ``e ** -30``, and held it there. ``-0.0`` is the
        count 0.
    ``huber_delta``
        The Huber cut, in units of ``s``, the residual's std, as for
        :func:`huber`, with its default, 1.345. ``inf`` clips nothing, which is the
        squared loss.
    ``quantile``
        The level for ``loss = "quantile"``; required for it.
    ``eps``
        The half-width of the insensitive tube, in units of ``s_y``, the
        target's own std. Default 0.01: errors under 1% of the target's own
        spread do not move the fit. In units of the noise, on a target a fit
        predicts to R², the tube is ``eps / sqrt(1 - R²)`` noise standard
        deviations wide: at 0.01, 0.067 of them at R² 0.978 and 2.2 at
        0.99998. Inside it the gradient is zero, so a tube many noise standard
        deviations wide holds the fit wherever it first lands inside it.
        Measured (one feature, no decay; the out-of-sample error above the
        noise, in noise variances, median over seeds; docs/PLAN.md task 207):

        .. list-table::
           :header-rows: 1

           * - R²
             - ``inv_scaling`` at 0.5, ``eps`` 0.01
             - at 0.1
             - ``constant`` at 0.01, ``eps`` 0.01
             - at 0.1
           * - 0.978
             - 0.015
             - 0.010
             - 0.042
             - 0.027
           * - 0.99998
             - 0.20
             - 24 (0.7 to 214)
             - 0.81
             - 17 (3.4 to 30)

        At 0.1 a fit of a target predicted to within 1% of its spread stopped
        as soon as every error was inside the tube, about 0.08 off the truth
        in intercept and slope together.
    ``learning_rate``, ``schedule``, ``power``
        The rate (default 0.01) and its schedule: ``"constant"``,
        ``"inv_scaling"`` (``lr / (1 + weight_sum) ** power``, ``power`` default 0.5)
        or ``"adagrad"`` (``lr / (sqrt(G_i) + 1e-8)``). The rate is per gradient
        coordinate: a step is ``lr * g_i``, so ``lr`` is in the coefficient's
        units over the gradient's -- one over the feature squared for the squared
        and Huber losses, and the target over the feature squared for the
        sign-valued quantile and epsilon-insensitive ones -- in standardized
        units under ``standardize``. A sign-valued step moves the fit's level by
        the rate times the row's weight, so at the default 0.01 a target at a
        level of 1,000 is 100,000 unit-weight rows away; under ``"inv_scaling"``
        at 0.5, without decay, about 1,000,000. AdaGrad's sum of squared
        gradients and ``weight_sum`` both decay on the model's clock, so an
        annealed or adapted rate opens up again after a long gap instead of
        staying frozen. The coefficients themselves do not decay: every row's
        step moves them, so under ``"constant"`` their memory is in rows, about
        ``1 / (lr * E[z**2])`` of them, whatever the clock between rows.
        ``half-life`` reaches ``weight_sum`` and ``min_weight``, the scaler and
        AdaGrad's sum, not the coefficients.
    ``l2``
        A ridge on every step, on the slopes only: the intercept is not
        penalised. Under ``standardize`` it is on the slope in the row's
        standardized coordinates, ``s_i * b_i``. Default 0.0.
    ``clip_gradient``
        A cap on each coordinate of the gradient, which clamps each ``g_i`` to
        ``+/- clip_gradient``: a box, not a cap on the gradient's norm, in the
        gradient's units, the target times the feature at the row's weight
        (the standardized feature under ``standardize``). Default ``1e3``, not
        off, because ``poisson`` needs it: ``p = exp(eta)``, so a row that
        pushes ``eta`` up makes the next gradient exponentially larger and a
        constant rate diverges within a few thousand rows. It does not bind at
        ordinary scales for an identity-link fit.
    ``standardize``
        Take the step in standardized coordinates, which is the difference between
        one learning rate for every column and one per scale. Default ``True``,
        as for :func:`kalman`: raw, features times 100 took the defaults' fit from
        an R² of 0.96 to -71847.
        Each row is standardized against the running moments with the row
        admitted, sklearn's ``StandardScaler.partial_fit`` then ``transform``:
        ``z_i = (x_i - m_i) / s_i``, ``m_i`` and ``s_i`` the EW mean and standard
        deviation, ``s_i = 1`` for a feature with no spread yet. Without an
        intercept the row is scaled by each column's root mean square and not
        centred, ``z_i = x_i / s_i``, as for :func:`ewridge`. That bounds a
        standardized value by ``sqrt(weight_sum)``, and it is not a leak: the
        rule is about the target, and the features of the row being predicted
        are known. Against the moments from before the row, a variance a few rows
        old can be tiny by chance, and one step throws a coefficient the rest of a
        short group never brings back.

        The moments warm up first. Until Kish's count of the rows they have
        learned, ``(sum w) ** 2 / sum w ** 2`` undecayed, reaches 22, the
        coefficients are held in the standardized coordinates, the same
        standardized row serves the prediction and the step, and the
        coefficients are read out in the caller's units through the moments as
        they stand. On the row the count reaches 22 they are read out so once,
        and from then on held in the caller's units, the step taken in ``z``
        mapped into them by the row's own ``m_i`` and ``s_i``:

        .. code-block:: text

            eta   = b_0 + sum_i b_i * x_i
            g_i   = d * z_i * w + l2 * s_i * b_i,    g_0 = d * w      (each clamped)
            b_i  -= lr_i * g_i / s_i
            b_0  -= lr_0 * g_0 - sum_i m_i * lr_i * g_i / s_i

        So a step moves the row's own prediction as the step in ``z`` does, and
        one learning rate suits every column. The prediction reads no moment,
        so the moments' moving moves none of it. Read through the moments as
        they stood, a finite half-life's own wander moved every prediction: at
        a half-life of 10 and R² 0.978 the slope came out 3.03 for a truth of
        2, and at a half-life of 50 and R² 0.99998 the out-of-sample error was
        248 noise variances above the noise, where the raw fit's was 0.010
        (docs/PLAN.md task 206). The warm-up is why the first rows keep their
        old behaviour: held in the caller's units from the first row, a step
        taken against a scale a few rows old was kept, and short groups lost
        everything (R² -207 at rows 25-50 of 200-row groups).
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes; the intercept is always free. ``coef_min`` and
        ``coef_max`` bound each slope (one number for every feature, or a list
        with one entry per feature; ``-inf`` / ``inf`` for no bound), and
        ``coef_sum`` fixes the slopes' total. After each update the slopes are
        replaced by the nearest point of the feasible set, the Euclidean
        projection: ``b_i = clamp(b_i - mu, lo_i, hi_i)``. ``mu`` is 0 for a box
        alone and, with a sum, the one value at which the sum holds, found exactly
        by sorting the ``2k`` breakpoints where a coordinate meets a bound.
        ``coef_min = 0, coef_sum = 1`` puts the slopes on the simplex: portfolio
        weights, mixing weights, an ensemble over forecasts. The starting point
        (all zero) is projected too, so a simplex starts uniform. With
        ``standardize`` the step and the projection are taken in standardized
        coordinates with the bounds and the sum carried over exactly, and the
        reported coefficients satisfy the constraint in the caller's units after
        every learned row. Past the scaler's warm-up the coefficients are held
        in the caller's units and the projection is the same one written in
        them, ``b_i = clamp(b_i - mu / s_i ** 2, lo_i, hi_i)``, the bounds met
        exactly; the intercept keeps its standardized value, taking ``sum_i m_i
        * (b_i - b_i')`` from the slopes' move. A sum the bounds cannot reach, a
        floor above a cap, or an infinite bound on the wrong side is refused by
        name; floors of ``[0.1, 0.2, 0.3]`` accept a sum of ``0.6`` although
        they add up to ``0.6000000000000001``.
    ``strict_binary``
        Under ``loss = "logistic"``, refuse a chunk whose target is not 0 or 1,
        naming the row, before any stream is touched, as :func:`ftrl` does.
        Default ``False``: such a target is clamped into ``[0, 1]``. Taken as it
        stood, a label of 5 pushed the linear predictor up on every row for
        ever. Refused beside another loss.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row, so rows with a null
    target do not warm up coefficients they never moved (docs/PLAN.md task
    115 (d)). A label ``strict_binary`` refuses is not one of them.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#sgd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#sgd>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the caller's
        units, the constraints satisfied after every learned row
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is a probability under ``"logistic"`` and a rate
    under ``"poisson"``, and ``emit_metrics`` reads a logistic fit as
    probabilities against labels. A Poisson fit's ``hit_rate_<t>`` is null: a rate
    is positive and a count is never negative, so there is no sign to hit, and
    the sign test about zero read 1.0 whatever the fit.

    .. rubric:: Example

    .. code-block:: python

        weights = po.spec.sgd(
            "w", targets=["y"], features=["signal_a", "signal_b", "x0"], half_life=200.0,
            loss="squared",
            learning_rate=0.01, schedule="constant",
            coef_min=0.0, coef_sum=1.0,    # the slopes on the simplex: long-only, fully invested
            coef_every=1,
        )
        fit = po.ModelBank([weights]).fit_predict(df)
        last = fit["w"].struct.field("coef").drop_nulls()[-1]
        assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``quantile`` is required
    for ``loss = "quantile"``, ``learning_rate`` refuses ``inf``, and the
    constraints are checked as above. ``huber_delta``, ``quantile``, ``eps``
    and ``strict_binary`` belong to the losses ``"huber"``, ``"quantile"``,
    ``"epsilon_insensitive"`` and ``"logistic"``, and ``power`` to ``schedule =
    "inv_scaling"``: each is refused beside another (``ValueError``), rather than
    ignored.
    """
    model: dict[str, Any] = {
        "type": "sgd",
        "loss": loss,
        "huber_delta": huber_delta,
        "quantile": quantile,
        "eps": eps,
        "learning_rate": learning_rate,
        "schedule": schedule,
        "power": power,
        "l2": l2,
        "clip_gradient": clip_gradient,
        "standardize": standardize,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
        "strict_binary": strict_binary,
    }
    spec = _common(name, model, targets=targets, features=features, **common)
    # A parameter of a loss or a schedule the spec does not use is refused, as
    # a switch that is off is; each was taken and ignored (review 2026-10-05,
    # YA8). After the Rust side's checks, so an unknown loss is named first.
    for key, value, what, owner, chosen in (
        ("huber_delta", huber_delta, "loss", "huber", loss),
        ("quantile", quantile, "loss", "quantile", loss),
        ("eps", eps, "loss", "epsilon_insensitive", loss),
        ("power", power, "schedule", "inv_scaling", schedule),
    ):
        if value is not None and chosen != owner:
            raise ValueError(
                f"spec {json.dumps(name)}: sgd {key} is for {what} {json.dumps(owner)}; "
                f"{what} {json.dumps(chosen)} does not use it"
            )
    return spec


@_checked
def pa(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    mode: str = "pa1",
    c: float | None = None,
    eps: float | None = None,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    standardize: bool = True,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Passive-aggressive regression (Crammer et al. 2006): each row asks the fit to
    come within ``eps`` standard deviations of the target (its own EW spread), and
    the update is the smallest change that does so.

    Passive when the constraint already holds, aggressive when it does not, and
    there is no learning rate to tune. O(k) per row, like :func:`sgd`.

    .. rubric:: The fit

    With ``p = z . b``, ``loss = max(0, |y - p| - eps * sigma)`` and ``s =
    ||z||^2``, the intercept's 1 inside ``z``:

    .. code-block:: text

        pa    tau = loss / s                 (unbounded)
        pa1   tau = min(c, loss / s)         (capped at c)
        pa2   tau = loss / (s + 1 / (2c))    (damped by c)
        b    += min(w, 1) * tau * sign(y - p) * z

    That is the step with ``standardize=False``, ``z = [1, x]``. Under
    ``standardize``, the default, ``z`` is the row standardized, and once the
    scaler has warmed up the step is mapped into the caller's units
    (``standardize`` below).

    ``sigma`` is the target's own EW standard deviation as the row arrives, as
    for :func:`sgd`'s tube: the spread of ``y`` around its EW mean, over the
    rows with the target and a weight above 0, whatever the fit, each ``y``
    joining after its own row is judged. Before the target has a ``sigma`` above
    0, two weighted rows of different values, the tube has no width and every
    row teaches. In the target's own units a tube of 0.1 held a target in
    hundredths on every row: passive for ever, every prediction 0.0 and R² -0.05
    where the same target unscaled scored 0.96.

    **c is in the target's units.** ``pa1``'s ``tau`` is in the target's units
    over ``s``'s, so a cap of 1 caps every step of a target in thousands and
    none of a target in thousandths, and the figures below are for a target of
    spread about 2. ``pa2``'s ``c`` is added to ``s`` as ``1 / (2c)``, in
    ``s``'s units, so it is free of the target's. Under ``"pa"`` and ``"pa2"`` a
    target scaled by ``k`` fits as the unscaled one, scaled by ``k``; under
    ``"pa1"`` only while the cap does not bind.

    **On a level that moves, raise ``c`` or difference the target.** The fit
    starts from zero coefficients, and a ``pa1`` cap of 1 moves the intercept
    about one unit a row. On a target falling from 1,000 to -1,000 at 0.5 a
    row (half-life 200, noise 0.3), the intercept stood at 51.5 at row 50 and
    401 at row 400, and met the level near row 670. With the level as a
    feature too, the capped steps, mapped into the caller's units through the
    scaler, swung the fit far past the level instead. A drift that the
    intercept carries alone keeps a lag at any ``c``: 2.2 on average over
    rows 800 to 1,600 at ``c = 1000``. About half of it is the tube,
    ``eps * sigma``, whose ``sigma`` holds the drift (97 to 143 here); with
    ``eps = 0`` the lag was 1.1. Differencing the target and the moving
    features, ``y - y.shift(1)``, takes the level out of the fit: the level
    is then the last row's ``y`` plus the predicted change, and the change
    carries noise of 0.42 here. Measured as ``pred - y`` (docs/PLAN.md task
    216):

    .. list-table::
       :header-rows: 1

       * - the run
         - row 800
         - row 1,600
         - every row within 3 from row
       * - the defaults, the level a feature
         - 1,790
         - -3.2
         - 2,795
       * - ``c = 10``, the level a feature
         - 0.67
         - 1.1
         - 467
       * - ``c = 1000``, the level a feature
         - -0.52
         - -0.60
         - 11
       * - the defaults, the level the intercept's alone
         - 1.3
         - 2.1
         - never: 2.7 rms over rows 800 to 1,600
       * - differenced, the level read back
         - -0.73
         - -0.05
         - 12

    In units of the noise, on a target a fit predicts to R², ``sigma`` is the
    noise's standard deviation over ``sqrt(1 - R²)``, so the tube is ``eps /
    sqrt(1 - R²)`` noise standard deviations wide. At the default 0.01:

    .. list-table::
       :header-rows: 1

       * - R²
         - 0.5
         - 0.978
         - 0.9975
         - 0.9998
         - 0.99998
       * - the tube, in noise standard deviations
         - 0.014
         - 0.067
         - 0.20
         - 0.71
         - 2.2

    A tube well inside the noise damps nothing: every row outside it is
    projected onto its edge, and only a cap that binds damps then. A wider tube
    damps too, while the cap does not bind: the fit moves only on the rows in
    the noise's tails, and they keep pulling it back. But a tube many noise
    standard deviations wide holds the fit wherever it first lands inside it.

    ``sigma`` is not the residual's std, as :func:`huber`'s cut is. The fit
    starts from zero coefficients, so its first residuals are the target's
    whole level. On a target at 1,000 with a spread of 2, a tube drawn from
    them was about 100 wide and held every row. Without decay it never
    narrowed: R² -52 at a half-life of 1e9.

    The coefficients keep no accumulators, so there is nothing in them for the
    clock to decay: each step fully satisfies the current row, and older rows
    survive only through the coefficients they left behind. The clock decays
    ``weight_sum``, so ``min_weight`` means the same thing as elsewhere, and
    it decays the scaler and ``sigma``; the coefficients have no half-life.

    .. rubric:: Parameters

    ``mode``
        ``"pa"``, ``"pa1"`` (the default) or ``"pa2"``. Prefer the bounded modes
        when outliers are possible: plain ``"pa"`` moves the fit as far as it
        takes to satisfy a single bad row.
    ``c``
        Under ``pa1``, the cap on ``tau``, in the target's units over ``s``'s:
        the features' squared, or standardized ones under ``standardize``, so
        ``c`` is in the target's units alone there, and what it does depends on
        the target's scale. Under ``pa2``, the damping ``1 / (2c)`` beside
        ``s``, in ``s``'s units and free of the target's. Default 1.0; ``inf``
        caps nothing, so either bounded mode is then ``"pa"``.
    ``eps``
        The margin, in units of ``sigma``: the row is close enough inside it and
        nothing moves. Default 0.01: errors under 1% of the target's own spread
        do not move the fit. Measured (``pa1``, one feature, a target of spread
        about 2, no decay; the out-of-sample error above the noise, in noise
        variances, median over seeds; docs/PLAN.md task 207):

        .. list-table::
           :header-rows: 1

           * - R²
             - the defaults
             - ``c = 0.1``
             - ``eps = 0.1``
             - ``eps = 0.5``
           * - 0.978
             - 1.15
             - 0.33
             - 0.57
             - 0.06
           * - 0.99998
             - 0.14
             - 0.14
             - 65 (25 to 347)
             - 1,496

        At R² 0.978 the default tube, 0.067 noise standard deviations, damps
        nothing, and a cap of 1 binds on few rows of a target of spread 2: a
        cap of 0.1 or a wider tube damps more. At R² 0.99998 the default tube
        is 2.2 noise standard deviations and no cap binds, and a tube of 0.1
        (22 of them) or 0.5 holds the fit wherever it first lands, from 25 to
        347 noise variances over 20 seeds at 0.1. The default had the smallest
        worst regret over 108 streams (R² from 0.5 to 0.99998, a level of 0 or
        plus or minus 1,000, a half-life of 50, 500 or none, one feature or
        five) and the three modes, against a tube in the residual's spread
        capped by the target's: without decay its start-up residuals stayed in
        it for ever, and under ``pa2`` it left a target at a level of 1,000 at
        R² 0.76 to 0.80.
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes, exactly as for :func:`sgd`. The projection
        follows each update, so the step does not meet the row's margin exactly:
        it is the closest feasible coefficient to the one that would. A truth
        outside the feasible set is never realizable, so the model keeps stepping
        against the walls; a small ``c`` keeps those steps small.
    ``standardize``
        Take the step in standardized coordinates, :func:`sgd`'s scaler, its
        rule and its warm-up: each row standardized against the running moments
        with the row admitted, ``z_i = (x_i - m_i) / s_i`` (``z_i = x_i / s_i``,
        the root mean square, without an intercept), and a box or a sum
        projected with its bounds carried over. Default ``True``: raw, ``s`` and
        ``c`` are in the features' units, so one feature in thousands makes
        every step tiny and one in thousandths every step the cap.

        Until Kish's count of the rows the moments have learned reaches 22,
        the coefficients are held in the standardized coordinates and read
        back in the caller's units through the moments as they stand. On the
        row it reaches 22 they are read out so once, and from then on held in
        the caller's units, the step mapped into them by the row's own ``m_i``
        and ``s_i``:

        .. code-block:: text

            step  = min(w, 1) * tau * sign(y - p)
            b_i  += step * z_i / s_i
            b_0  += step - sum_i m_i * step * z_i / s_i

        The step still puts the row where the update says: the row's change of
        prediction is ``step * ||z||^2``, so an uncapped step at weight 1 leaves
        it on the tube's edge, and one at weight ``w < 1`` goes that fraction
        of the way. The prediction reads no moment, so the moments' moving moves
        none of it. Read through the moments as they stood, a finite
        half-life's own wander moved every prediction: at a half-life of 50 and
        R² 0.99998, 31.8 noise variances of out-of-sample error above the noise
        where the raw fit paid 0.18 (docs/PLAN.md task 206). Past the warm-up a
        box or a sum bounds the coefficients in the caller's units and is
        projected as for :func:`sgd`.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row, so rows with a null
    target do not warm up coefficients they never moved (docs/PLAN.md task
    115 (d)).

    A row weight below 1 scales ``tau``, so a half-weight row moves the fit half
    as far; a weight above 1 counts as 1. The update is a projection onto the
    row's constraint, and repeating a projection changes nothing, so there is no
    "two observations" to emulate.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#pa
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#pa>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them.

    .. rubric:: Example

    .. code-block:: python

        pa = po.spec.pa(
            "pa", targets=["y"], features=["x0", "x1"], half_life=200.0,
            mode="pa1", c=0.1,   # the step is capped at c
            eps=0.05,            # within 0.05 of the target's std nothing moves
        )
        out = po.ModelBank([pa]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``eps`` refuses ``inf``,
    and the constraints are checked as for :func:`sgd`.
    """
    model: dict[str, Any] = {
        "type": "pa",
        "mode": mode,
        "c": c,
        "eps": eps,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
        "standardize": standardize,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def holt(
    name: str,
    *,
    targets: TargetList,
    trend_half_life: float | Duration | None = None,
    trend: bool = True,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Holt's linear trend: the target's own level and slope, extrapolated.

    It takes no features, as only :func:`seqtest` among the other models
    does -- the forecasting baseline a feature-based model should have to
    beat. If a regression cannot
    outperform "the series is going up at about this rate", its features are
    not earning their place. Run it in the same bank and compare the two
    ``sigma``, or let a :func:`seqtest` with ``a`` and ``b`` say which
    predicts closer.

    .. rubric:: The fit

    Per row and target, with ``s`` the clock since the target was last
    observed: this row's delta included, so ``s`` is the row's own delta on
    a stream with no gaps. ``w`` is the row's weight, and ``W``, ``V`` the
    weight the level and the trend have gathered, each decayed on its own
    half-life (``lam_l = 0.5 ** (s / half_life)``, the spec's own, and
    ``lam_b`` likewise at ``trend_half_life``):

    .. code-block:: text

        pred = l + b * s                                   extrapolate s clock units ahead
        l'   = (lam_l * W * pred + w * y) / (lam_l * W + w)
        b'   = (lam_b * V * b + w * (l' - l) / s) / (lam_b * V + w)

    Level and trend are weighted means, as every accumulator here is: a row
    at weight ``w`` counts ``w`` times, and an infinite half-life forgets
    nothing and fits the whole history. The gains ``w / (lam * W + w)``
    start at 1 and fall to the textbook's fixed ``1 - lam`` as the weight
    saturates. So a new series is followed sooner, and the fit is
    statsmodels' ``Holt`` from there. A row at the last row's clock is a
    second observation the level takes in; the trend holds, since a move
    over no clock has no slope. A row with a null target or a zero weight
    leaves ``l`` and ``b`` where the last observation put them and carries
    its clock to the next one. So it gives the same numbers as if it were
    absent. The trend is per clock unit, so an irregular clock extrapolates
    the right distance.

    .. rubric:: Parameters

    The level forgets at the spec's ``half_life`` (or ``lam``), in clock
    units, ``inf`` included: one knob under one name, as for every model.
    ``level_half_life``, the second name it once had, is refused naming
    it.

    ``trend_half_life``
        How fast the trend forgets, in clock units. Default four times the
        level's half-life; ``inf`` is the whole history's drift, not a trend
        pinned at zero.
    ``trend``
        ``False`` fits the level alone: the trend is held at zero and the
        forecast is flat, which is simple exponential smoothing. The level is
        then the weighted mean of the observations, each at its weight times
        ``0.5 ** (age / half_life)``, which is pandas'
        ``ewm(adjust=True)`` for unit weights. ``trend_half_life`` is refused
        beside it. Default ``True``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    There is no seasonal term: a seasonal index is a ``group`` on the phase,
    which the bank already does.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#holt
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#holt>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and why the row's
        predictions are null where they are, as everywhere.
    ``coef``
        ``[level, trend]`` per target, the whole state; :func:`coef_index`
        names the two. Null for a target not yet observed, which has no
        level to report. The trend is 0 under ``trend=False``.

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` extrapolates
    over the clock distance from the row the model last learned, capped by
    ``gap_cap``.

    .. rubric:: Example

    .. code-block:: python

        baseline = po.spec.holt(
            "baseline", targets=["y"], clock="t", gap_cap=600.0,
            half_life=200.0,          # how fast the level forgets
            trend_half_life=2000.0,   # how fast the trend forgets; inf is the whole history's drift
        )
        out = po.ModelBank([baseline]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`).
    """
    model: dict[str, Any] = {
        "type": "holt",
        "trend_half_life": trend_half_life,
    }
    if not trend:
        # Absent when on, as the Rust spec leaves it out, so a bank reports
        # the dict this made (`ModelBank.specs`).
        model["trend"] = False
    return _common(name, model, targets=targets, features=features or [], **common)


@_checked
def kmeans(
    name: str,
    *,
    features: list[str],
    k: int,
    warm_rows: int | None = None,
    seed_rule: str | None = None,
    seed: int | None = None,
    update_every_rows: int | None = None,
    split_merge: float | None = None,
    split_merge_every_rows: int | None = None,
    dead_frac: float | None = None,
    standardize: bool | None = None,
    scale_floor: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted k-means over the feature columns.

    Not a regression: there are no targets. Each row is assigned to the
    nearest of ``k`` centres before it is learned, so the label is
    out-of-sample like every prediction here. Each centre is the decayed
    weighted mean of the rows assigned to it: :func:`ew_cov`'s mean
    recursion, per cluster.

    .. rubric:: The recursion

    .. code-block:: text

        j*   = argmin_j |x - c_j|^2          distances in units of each feature's EW sd
        n'_j = lam * n_j + w
        c'_j = (lam * n_j * c_j + w * x) / n'_j       for the nearest j

    Alongside each centre the EW squared radius ``r2_j``, the mean of ``|x -
    c_j|^2`` over the rows assigned there, which the split-merge rule reads.
    Rows are folded into per-centre batches and applied every
    ``update_every_rows`` learned rows, so ``update_every_rows = 1`` is plain
    sequential k-means and a larger value a mini-batch one.

    .. rubric:: Parameters

    ``k``
        The number of centres; required, and at most 65,536 (2^16): every row
        is scored against every centre, and seeding is ``O(k^2)``.
    ``warm_rows``, ``seed_rule``, ``seed``
        Seeding. The first ``warm_rows`` learned rows are buffered: at least
        ``k``, and a smaller value is refused; default 500, or ``k`` where
        that is more. The buffer is held to 256 MiB, each row its values and
        its weight. Then the centres are placed by ``seed_rule``:

        .. list-table::
           :header-rows: 1
           :widths: 24 76

           * - ``seed_rule``
             - the centres
           * - ``"lloyd"`` (the default)
             - k-means++ then ten weighted Lloyd iterations over the buffer
           * - ``"kmeanspp"``
             - k-means++
           * - ``"farthest"``
             - Gonzalez, from the first row
           * - ``"first"``
             - the first ``k`` distinct rows

        ``seed`` (default 0) drives the two random rules; the same seed gives
        the same centres. The buffer is replayed into the centres and freed,
        so the model is O(state) again from that row on. Outputs are null
        until seeding and until ``weight_sum`` reaches ``min_weight``.
    ``update_every_rows``
        Learned rows between applications of the per-centre batches. Default
        1.
    ``split_merge``, ``split_merge_every_rows``
        A row farther from its centre than a blob of the typical radius
        produces (about four standard deviations of ``|x - c|^2`` above it)
        is far. It is scored, but summarised instead of learned, so it
        neither drags the centre nor widens the radius. Every
        ``split_merge_every_rows`` learned rows (default 100) the two closest
        centres are compared. If their distance is under ``split_merge``
        (default 0.5; ``0`` disables) times the sum of their radii, and
        enough far rows have gathered somewhere (at least three, and five per
        cent of the window's weight), they are merged. The freed centre is
        placed at the far rows' mean. So a cluster that appears after seeding
        gets a centre without anyone restarting. Far rows still count in the
        radius at each check as if they sat at the cut, so a cut the data has
        outgrown widens until the rows are learned again.
    ``dead_frac``
        Re-place a centre whose weight has decayed below ``dead_frac *
        weight_sum / k`` the same way, on whatever far rows there are
        (default 0.05; ``0`` disables). A centre whose cluster vanished is
        re-placed ``log2(1 / dead_frac)`` half-lives later (4.3 at the
        default, 2 at 0.25). A cluster lighter than ``dead_frac / k`` of the
        stream loses its centre whenever any row is far. The rule runs at a
        split-merge check, so beside ``split_merge = 0``, which runs none,
        the default is 0 and a value above 0 is refused.
    ``standardize``
        Measure distances in units of each feature's EW standard deviation,
        tracked alongside the centres; the coordinates themselves are never
        rescaled, so the centres stay in the features' units. Default
        ``True``.
    ``scale_floor``
        The metric's variance is floored at this fraction of the feature's
        long-run variance. ``1 / var`` alone grows as ``2^Q`` over ``Q``
        half-lives of a feature going quiet (a flag that stops firing, a
        sensor at rest): a million at twenty, and the row on which the
        feature moves again is then infinitely far from every centre.
        Floored, the weight grows as ``2^(Q/8) / scale_floor``, about 57 at
        twenty. The long-run variance is tracked at eight times the
        half-life, a feature at a time. Each row's weight and deviation are
        clipped against it, and its start is the medians of the feature's
        first five rows, so a row at the input bound moves it by a factor of
        26 at most, which a few of its half-lives undo. Default ``0.1``;
        ``0`` is the EW variance alone, and what a state saved before the
        floor loads with.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by
    name: ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#kmeans
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kmeans>`_):

    ``cluster``
        The nearest centre's index (``i32``), before the row is learned from;
        null until seeding.
    ``dist``, ``dist_second``
        The distance to that centre, and to the runner-up, the second-nearest
        (null when ``k == 1``), so ``dist_second - dist`` is the margin.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.
    ``coef``
        The centres: ``k`` rows of ``len(features)``, flattened
        cluster-major. :func:`coef_index` lays it out, with ``target``
        reading ``"cluster0"``, ``"cluster1"``, ... and ``term`` the feature
        whose coordinate the position holds.

    .. rubric:: Example

    .. code-block:: python

        km = po.spec.kmeans(
            "km", features=["x0", "x1", "x2"], clock="t", half_life=2000.0, gap_cap=300.0,
            k=3,
            warm_rows=100,           # seeding waits for this many rows, then replays them
            seed_rule="lloyd",       # k-means++ then ten Lloyd iterations over the buffer
            split_merge=0.5,         # two centres in one blob: one moves to the far rows
            dead_frac=0.05,         # a centre whose blob vanished is re-placed 4.3 half-lives later
        )
        out = po.ModelBank([km]).fit_predict(df).unnest("km")
        layout = po.spec.coef_index(km)    # target = "cluster0".., term = the feature

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got; ``ValueError`` for ``k`` past
    2^16, ``warm_rows`` below ``k`` or a warm-up buffer past 256 MiB, and
    ``dead_frac`` above 0 beside ``split_merge = 0``.
    """
    model: dict[str, Any] = {
        "type": "kmeans",
        "k": k,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
        "seed": seed,
        "update_every_rows": update_every_rows,
        "split_merge": split_merge,
        "split_merge_every_rows": split_merge_every_rows,
        "dead_frac": dead_frac,
        "standardize": standardize,
        "scale_floor": scale_floor,
    }
    targets = _mirror_target(name, "kmeans", features, common, "its clusters are over")
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def micro(
    name: str,
    *,
    features: list[str],
    eps: float,
    beta_mu: float | None = None,
    max_clusters: int | None = None,
    prune_every: float | Duration | None = None,
    max_rows_between_prunes: int | None = None,
    macro_link: float | None = None,
    standardize: bool | None = None,
    scale_floor: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Density-based clustering over the feature columns: DenStream-style
    micro-clusters with a linkage step over them.

    Not a regression: there are no targets, and unlike :func:`kmeans` there is
    no fixed number of clusters. The model keeps a bounded set of small
    summaries (micro-clusters), each a decayed weight, centre and radius, and
    reads clusters off them as the chains of summaries that touch. That finds
    clusters of any shape (moons, rings), reports rows that belong to none,
    and lets clusters appear and vanish as the stream moves.

    .. rubric:: The recursion

    A summary is ``(n, c, r2)``: decayed weight, centre, and the EW mean
    squared distance of its rows from the centre (DenStream's radius, with the
    fading function being the decay). Distances are measured in the metric
    ``mw_i = 1 / var_i`` when ``standardize`` (the default), so ``eps`` is a
    bound per standardized coordinate and the bound in the metric is ``E =
    eps² p`` for ``p`` features. Each row:

    .. code-block:: text

        n_j <- lam n_j                                   every summary
        j   = the nearest potential summary, if absorbing a unit row keeps
              a r2_j + a b |x - c_j|^2 <= E,   a = n_j/(n_j+1), b = 1/(n_j+1)
              else the nearest outlier summary by the same test
              else a new summary at x with the next id
        n_j <- n_j + w,  c_j <- c_j + w/n_j (x - c_j),  r2_j <- min(., E)

    A summary is potential (established) once ``n >= beta_mu * w_bar`` and
    outlier below, ``w_bar`` the EW mean weight of the rows learned from. A
    new one is opened at the cap ``max_clusters`` by evicting the lightest
    outlier summary, else the lightest potential one. A checkpoint prunes
    and links, on the schedule ``prune_every`` and ``max_rows_between_prunes``
    set (below). It drops potential summaries
    lighter than ``beta_mu * w_bar``, and outlier summaries lighter than
    DenStream's ``xi(age) * w_bar``, with ``xi(age) = sum_{i <= age / Tp} 2 **
    (-i Tp / h)``. That is the weight of a summary that had taken one row of
    the mean weight every ``Tp`` clock units since it opened, with ``Tp =
    ceil(h log2(beta_mu / (beta_mu - 1)))`` for half_life ``h``. Both count
    rows, as DenStream's do: a stream with more rows to a clock unit fills its
    summaries faster. With no decay nothing is pruned, only capped. Then it
    links the potential summaries by single linkage: centres within ``L`` of
    each other share a label. Ids are monotone and never reused; a label is
    the smallest id in its chain, so it survives everything but the loss of
    that summary.

    .. rubric:: Parameters

    ``eps``
        The within-cluster spread the model should read as one cluster, per
        standardized coordinate; required. About 0.07 for two-dimensional
        shapes, 0.3 for well-separated Gaussians in twenty dimensions. Both
        ways to get it wrong show in the outputs:

        .. list-table::
           :header-rows: 1
           :widths: 34 36 30

           * - what you see
             - what it means
             - what to do
           * - nearly every row is an ``outlier`` and ``cluster`` stays null
             - ``eps`` is too small: no summary reaches ``beta_mu`` before it
               is pruned
             - raise ``eps``
           * - ``n_micro`` is about the number of clusters
             - ``eps`` is too coarse for the derived link, which then bridges
               them
             - lower ``eps``, or set ``macro_link``
    ``beta_mu``
        The weight at which a summary is established, in rows of the
        stream's mean weight. Default 3. It is DenStream's point density
        (DBSCAN's ``MinPts``), so it is set against the arrival rate. With
        half-life ``h`` and ``v`` rows in each unit of the clock the stream's
        steady-state weight is about ``1.44 * v * h``. So a summary meant to
        hold a share ``s`` of the stream needs ``beta_mu`` of about
        ``1.44 * s * v * h``.
    ``max_clusters``
        The cap on live summaries. Default 200.
    ``prune_every``, ``max_rows_between_prunes``
        A checkpoint every ``prune_every`` clock units or every
        ``max_rows_between_prunes`` learned rows, whichever comes first, as
        ``solve_every`` and ``max_rows_between_solves`` schedule a
        regression's solve; with neither, every 100 learned rows.
        ``prune_every`` is a number of the clock's units, a duration on a
        temporal clock (``"10m"``), or ``0`` for every row; without a clock
        column the clock is the row's number. A row of weight zero advances
        the clock, so a quiet spell still prunes on time, and a gap capped at
        ``gap_cap`` counts as the cap. DenStream checks every ``Tp`` clock
        units (above): give ``prune_every`` that to follow the paper.
    ``macro_link``
        ``L`` as a multiple of ``eps sqrt(p)``: ``0`` links nothing, so each
        summary is its own cluster; ``2`` links summaries that touch. Left
        out, ``L`` is derived at each checkpoint from the spacing the
        summaries already show: 1.5 times the 90th percentile of the
        nearest-neighbour distance, never below ``2 eps sqrt(p)``. So a chain
        along a shape holds without a constant that fragments one shape and
        bridges another.
    ``standardize``
        Measure in units of each feature's EW standard deviation. Default
        ``True``.
    ``scale_floor``
        The metric's variance is floored at this fraction of the feature's
        long-run variance. ``1 / var`` alone grows as ``2^Q`` over ``Q``
        half-lives of a feature going quiet (a flag that stops firing, a
        sensor at rest): a million at twenty, and the row on which the
        feature moves again is then infinitely far from every centre.
        Floored, the weight grows as ``2^(Q/8) / scale_floor``, about 57 at
        twenty. The long-run variance is tracked at eight times the
        half-life, a feature at a time. Each row's weight and deviation are
        clipped against it, and its start is the medians of the feature's
        first five rows, so a row at the input bound moves it by a factor of
        26 at most, which a few of its half-lives undo. Default ``0.1``;
        ``0`` is the EW variance alone, and what a state saved before the
        floor loads with.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    A row of weight ``w`` is admitted where a row of the mean weight would be
    and absorbed with its full weight, so a constant multiple of every weight
    moves nothing. A zero-weight row advances the clock and learns nothing.
    Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is learned
    (`docs/OUTPUTS.md#micro
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#micro>`_):

    ``cluster``
        The label of the nearest established summary (``i64``); null while
        there is none.
    ``dist``
        The distance to that summary's centre.
    ``micro_id``
        The id of the summary this row goes to (``i64``) -- the one it opens,
        when none can take it.
    ``outlier``
        Whether no established summary takes it (``bool``).
    ``n_clusters``, ``n_micro``
        Live clusters and live summaries (``i32``), so churn is visible
        without diffing labels.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.
    ``coef``
        The established summaries, one ``[id, label, n, radius, c_1, ...,
        c_p]`` row each, flattened -- as many rows as there are, so
        :func:`coef_index` does not apply.

    .. rubric:: Example

    .. code-block:: python

        mc = po.spec.micro(
            "mc", features=["x0", "x1"], clock="t", half_life=2000.0, gap_cap=300.0,
            min_weight=50.0,
            eps=0.3,             # the spread read as one cluster, per standardized coordinate
            beta_mu=5.0,         # a summary with at least this much weight is established
            prune_every=100,     # every 100 units of t: prune the light summaries, link the rest
        )
        out = po.ModelBank([mc]).fit_predict(df).unnest("mc")
        churn = out.select("cluster", "outlier", "n_clusters", "n_micro").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got.
    """
    model: dict[str, Any] = {
        "type": "micro",
        "eps": eps,
        "beta_mu": beta_mu,
        "max_clusters": max_clusters,
        "prune_every": prune_every,
        "max_rows_between_prunes": max_rows_between_prunes,
        "macro_link": macro_link,
        "standardize": standardize,
        "scale_floor": scale_floor,
    }
    targets = _mirror_target(name, "micro", features, common, "its clusters are over")
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def ew_class(
    name: str,
    *,
    features: list[str],
    label: str,
    classes: list[str],
    covariance: str | None = None,
    precision_prior: float,
    window_size: float | Duration | None = None,
    closed: str = "right",
    window_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian classifier on :func:`ew_cov`'s moments, one set per class:
    quadratic discriminant analysis, linear discriminant analysis or Gaussian
    naive Bayes.

    A label column takes the place of a numeric target. The model keeps one
    ``ew_cov`` state per class: a weight ``n_c``, a mean ``mu_c`` and a
    centred covariance ``C_c``. It scores a row by Bayes' rule over Gaussian
    classes before the row's own label is learned, so a row's probabilities
    never saw its label. That is also how a stream whose labels arrive late
    is scored: null the label, keep the features.

    .. rubric:: The recursion

    Before a row ``x`` is learned it is scored against every seen class:

    .. code-block:: text

        pi_c = n_c / sum_c' n_c'
        r_c  = precision_prior * s_c         s_c: the class's prior scale
        M_c  = C_c + r_c I                                  ("full", QDA)
        M    = sum_c pi_c (C_c + r_c I)                     ("shared", LDA)
        M_c  = diag(C_c) + r_c I                        ("diagonal", naive Bayes)
        l_c  = ln pi_c - 1/2 ln det M_c - 1/2 (x - mu_c)' M_c^-1 (x - mu_c)
        p_c  = exp(l_c - max_c' l_c') / sum_c' exp(l_c' - max l)

    then the labelled row's class learns it:

    .. code-block:: text

        n_c   <- lam n_c + w
        mu_c  <- mu_c + (w / n_c) (x - mu_c)
        C_c   <- weighted Welford on (x - mu_c_old)(x - mu_c_new)'

    and ``weight_sum <- lam weight_sum + w`` counts every accepted row,
    labelled or not, so ``min_weight`` means the same number of rows as
    everywhere else. A row with a non-finite feature is null and learns
    nothing; a zero-weight row advances the clock.

    .. rubric:: Parameters

    ``label``
        The column that holds the class of each row, read as a key -- any
        dtype with a string form, so ``["0", "1"]`` for an integer column and
        ``["true", "false"]`` for a boolean one. A null label is a row to
        score but not to learn from: the model classifies it and ticks its
        clock, and no class moves.
    ``classes``
        Every value the label can hold, in the order the ``p_<class>`` fields
        are written; required. A non-null value it does not list is an error
        naming the row.
    ``covariance``
        The shape, and the work a row takes:

        .. list-table::
           :header-rows: 1
           :widths: 20 36 44

           * - ``covariance``
             - the shape
             - per row
           * - ``"full"`` (the default)
             - a covariance per class
             - each class keeps its Cholesky factor, and only the class a row
               teaches is refactored: one ``k x k`` factorization per learned
               row; under a ``window_size`` every class's covariance moves on
               every row, so every class is refactored
           * - ``"shared"``
             - the weight-averaged one, so the decision boundaries are linear
             - one factorization
           * - ``"diagonal"``
             - the variances alone, which cannot see a correlation
             -
    ``precision_prior``
        A ridge on every class covariance, in the features' units, so the
        first rows of a class, whose sample covariance is singular, are
        scored with a finite, isotropic one; required. It is scaled by
        ``s_c``, the class's own prior scale, which starts at 1 and decays by
        ``lam * n_c / (lam * n_c + w)`` on every row the class learns. So the
        ridge washes out as the class fills in, exactly as :func:`ew_cov`'s
        ``precision_prior`` does.
    ``window_size``, ``closed``, ``window_every``, ``max_rows_between_snapshots``, ``window_budget``
        A hard cutoff on the history each class's moments are computed from,
        in clock units, as for :func:`ewridge`: a row ``window_size`` old or
        older contributes to no class (``closed="both"`` keeps the one
        exactly that old, as for :func:`ewridge`). That is what lets a
        classifier follow class means that move; over a long history two
        regimes average together and the labels go to chance.
        ``window_every`` spaces the snapshots on the clock and
        ``max_rows_between_snapshots`` caps the rows between them, whichever
        comes first, as for :func:`ewridge`; ``window_budget`` bounds each
        ring in MiB. ``weight_sum`` is the weight inside the window, in
        the struct and in the ``min_weight`` gate, and the class moments a
        row is scored against are the window's.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by
    name: ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is
    learned (`docs/OUTPUTS.md#ew_class
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_class>`_):

    ``class``
        The class with the largest posterior (``str``; the first, on a tie),
        null before ``min_weight`` and while no class has been seen.
    ``p_<class>``
        One per class in ``classes`` order: its posterior probability, so the
        ``p_`` fields sum to 1 -- exactly 0 for a class no row has carried
        yet.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.
    ``coef``
        The class means, one row per class in ``classes`` order, each one
        entry per feature; :func:`coef_index` lays the list out as ``(class,
        feature)``, and a class not yet seen is null.

    .. rubric:: Example

    .. code-block:: python

        labelled = df.with_columns(
            pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
        )
        cl = po.spec.ew_class(
            "cl", features=["x0", "x1", "x2"], clock="t", half_life=200.0, gap_cap=300.0,
            label="dir", classes=["down", "up"],
            covariance="shared",         # pooled by class weight: linear boundaries
            precision_prior=0.1,         # the ridge that makes a class scoreable from its first row
            min_weight=20.0,
        )
        out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
        calls = out.select("dir", "class", "p_up", "weight_sum").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets`` -- this model takes ``label``.
    """
    model: dict[str, Any] = {
        "type": "ew_class",
        "classes": classes,
        "covariance": covariance,
        "precision_prior": precision_prior,
        "window_size": window_size,
        "closed": closed,
        "window_every": window_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    if "targets" in common:
        msg = f"spec {json.dumps(name)}: ew_class() takes `label`, not targets"
        raise TypeError(msg)
    return _common(name, model, targets=[label], features=features, **common)


@_checked
def seqtest(
    name: str,
    *,
    targets: TargetList,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str | None = None,
    b_suffix: str | None = None,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A sequential test of a sign, by betting: an e-process, read at any row.

    Not a regression. Per target the model keeps the wealth of two gamblers,
    one betting that the next sign is positive and one that it is negative.
    The answer is evidence that can be read at every row, as often as
    wanted, and acted on the first time it is enough. A p-value cannot be:
    checking it repeatedly inflates its error rate. With ``a`` and ``b`` it
    asks instead whether one spec of the bank predicts closer than another.

    .. rubric:: The recursion

    Per target the row's value is reduced to its sign ``s`` in ``{-1, 0,
    +1}``, and with ``n_pos`` and ``n_neg`` the counts of positive and
    negative rows before this one and ``n = n_pos + n_neg``:

    .. code-block:: text

        lam_pos = max(0, (n_pos - n_neg) / (n + 1))    lam_neg = max(0, (n_neg - n_pos) / (n + 1))
        E_pos  *= 1 + lam_pos * s                     E_neg  *= 1 - lam_neg * s

    ``(n_pos - n_neg) / (n + 1)`` is ``2p - 1`` for the Krichevsky-Trofimov
    estimate ``p = (n_pos + 1/2) / (n + 1)`` of ``P(s = +1)``. That is the
    stake a gambler with a ``Beta(1/2, 1/2)`` prior puts on the next sign,
    clipped so that each side bets only on the direction it tests. Both
    stakes are computed from the rows before. Under the null (given
    everything before it, a row is at least as likely to be negative as
    positive) ``E_pos`` is a non-negative supermartingale with ``E_pos[0] =
    1``. Ville's inequality then gives ``P(max_t E_pos[t] >= 1/alpha) <=
    alpha``. That is the whole guarantee. ``log_e_pos >= log(1/alpha)`` on
    any row rejects "no more positives than negatives" at level ``alpha``,
    and the stream can be read at every row and stopped the moment it
    crosses. Nothing about ``y`` but its sign is assumed: no independence of
    the sizes, no bound, no variance. What it does not test is the size: a
    stream up by a hair 60% of the time and down by a mile the rest rejects.
    ``(E_pos + E_neg) / 2`` is the two-sided e-value.

    .. rubric:: Parameters

    ``a``, ``b``, ``a_suffix``, ``b_suffix``
        Compare two specs of the same bank. Each target ``t`` names a
        residual field both carry: ``resid_<t>`` on each side, plus the
        side's grid suffix when it is a grid (``a_suffix="@h50"`` picks
        ``resid_<t>@h50`` of ``a``; ``"__r0.1"`` a ridge instance). The sign
        tested is that of ``|resid_b| - |resid_a|``, positive when ``a`` was
        closer on the row. Any loss that grows with ``|resid|`` (squared,
        absolute, Huber) gives the same sign, so this is a test of "``a``
        beats ``b``" under any of them. The bank runs ``a`` and ``b`` first,
        so the comparison reads the same out-of-sample residuals their
        structs report. A row where either side is null (warm-up, a skipped
        row) is a row the test sits out. A spec named against itself with
        the same suffix, a side that is not in the bank or is itself a
        ``seqtest``, and a target neither side has a residual for are
        refused by name. Two instances of one grid (``a_suffix="@h20"``
        against ``b_suffix="@h400"``) are a comparison like any other.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    A trial is a row, so ``weight`` is refused and there is no
    ``half_life``/``lam``: a process that forgot its losses would not be an
    e-process. A session change restarts it under ``session_gap = "reset"``
    or ``group_close = "session"``, and so does a step back past
    ``restart_after_step_back``.
    ``min_weight`` defaults to 0. No ``features`` (the column is the test;
    the keyword is taken so that a frame namespace can pass ``[]``), no
    ``coef``, and nothing residual-based applies -- there is no prediction.
    :func:`polars_online.eval.seqtest` is the same computation in polars
    expressions over a frame in memory.

    .. rubric:: Output

    One struct column named after the spec, per target ``t`` and read before
    the row is learned (`docs/OUTPUTS.md#seqtest
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#seqtest>`_):

    ``log_e_pos_<t>``, ``log_e_neg_<t>``
        The two gamblers' log wealth (``log E``, so 0 is no evidence and
        ``log(20) = 3.0`` is level 0.05).
    ``n_pos_<t>``, ``n_neg_<t>``
        The signs counted so far (``Int64``). A zero or null target bets
        nothing and counts nothing.
    ``weight_sum``
        As everywhere, decayed by nothing.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.

    With ``a`` and ``b`` the fields read ``log_e_a_<t>``, ``log_e_b_<t>``,
    ``wins_a_<t>`` and ``wins_b_<t>`` instead.

    .. rubric:: Example

    .. code-block:: python

        common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0)
        ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
        kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)
        sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")   # is y usually positive?
        # does kalman predict closer than ridge?
        closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge")
        out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
        verdict = out["closer"].struct.field("log_e_a_y").max()   # >= log(20): kalman won at 5%

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` for
    ``weight``, ``half_life`` or ``lam``, and for the comparison refusals
    above.
    """
    model: dict[str, Any] = {
        "type": "seqtest",
        "a": a,
        "b": b,
        "a_suffix": a_suffix,
        "b_suffix": b_suffix,
    }
    return _common(name, model, targets=targets, features=features or [], **common)


@_checked
def marginal(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    lags: list[int] | None = None,
    cross_lags: list[int] | None = None,
    serial_rule: str | None = None,
    bins: int | None = None,
    bin_rule: str | None = None,
    bin_warm_rows: int | None = None,
    bin_edges: dict[str, list[float]] | list[list[float]] | None = None,
    bin_budget: float | None = None,
    shards: int | str | None = None,
    window_size: float | Duration | None = None,
    closed: str = "right",
    window_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[str, float] | None = None,
    window_lags: bool = False,
    feature_moments: str = "per_target",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Every (feature, target) pair's exponentially weighted moments, kept in the
    state and read back as a table.

    Not a regression and not a joint fit: each pair ``(x_j, y_t)`` is its own
    two-column :func:`ew_cov`. A wide feature set against a few targets then
    takes ``O(p * T)`` per row, not the ``O((p + T)²)`` of one ``ew_cov`` over
    all the columns. Nothing is written per row but ``weight_sum``; the pairs
    live in the state and :meth:`polars_online.ModelBank.marginal` reads them.
    Use it to screen thousands of features against a few targets in one pass.

    .. rubric:: The recursion

    Per target ``t`` on a row where ``y_t`` is present, with ``W_t`` the target's
    accumulated weight before the row:

    .. code-block:: text

        W'_t = lam * W_t + w        a = lam * W_t / W'_t        b = w / W'_t
        Q'_t = lam^2 * Q_t + w^2
        S'_yy      = a * S_yy      + a * b * (y_t - m_y)^2
        S'_xx[t,j] = a * S_xx[t,j] + a * b * (x_j - m_x[t,j])^2
        S'_xy[t,j] = a * S_xy[t,j] + a * b * (x_j - m_x[t,j]) * (y_t - m_y)
        m'  = m + b * (value - m)                     for m_y and each m_x[t,j]

    The deviations are from the means before the row. The arithmetic is
    ``ew_cov``'s, so a pair's ``corr`` equals the ``corr`` an ``ew_cov`` over
    the two columns reports, to the bit. A null target ages that target
    (``W_t *= lam``, ``Q_t *= lam^2``) and learns nothing for it; a null
    feature skips the row, as everywhere.

    .. rubric:: Parameters

    ``lags``, ``cross_lags``, ``serial_rule``
        The pair's moments at those lags too, each at most 2^20 (1,048,576),
        the ring being sized before the first row. A lag counts learned rows
        within the group, not rows where that target was present, since the
        ring is shared. For a sparsely present target the lag is a row distance, not
        an observation distance. They add four list columns per pair to the
        table. ``lag_corr_xx`` and ``lag_corr_yy`` are the two series' own
        autocorrelations. ``lag_corr_xy`` is the feature now against the target
        ``l`` rows back, and ``lag_corr_yx`` the target now against the feature
        ``l`` rows back. For two series that describe the same moment, a
        feature whose ``lag_corr_yx[0]`` exceeds its ``corr`` leads the target,
        and one whose ``lag_corr_xy[0]`` does follows it. That reading does not
        hold against a forward-looking target, one built from the rows after
        its own. There the target ``l`` rows back is built partly from the
        feature's newest ``l`` rows, so a feature built from the same news
        shows ``lag_corr_xy`` above ``corr`` however it is sampled. The pair is
        the same statistic ``ew_cov(lags=)`` computes, to the bit.

        ``cross_lags`` keeps the two cross-correlations at fewer lags:
        strictly increasing, each one of ``lags``, and ``[]`` for none, which
        leaves the ``lag_corr_xy`` and ``lag_corr_yx`` columns out. By default
        they are kept at every lag. ``lag_corr_xy`` and ``lag_corr_yx`` are then
        lists over ``cross_lags``, in its order. The autocorrelations, and
        ``n_serial`` built from them, are kept at every lag whatever it says,
        and are the same to the bit. A lead or lag of a row or two is the
        usual question, and the cross terms are two thirds of the lag work.
        ``lags=[1, 2, 5, 10, 20, 50], cross_lags=[1]`` keeps the serial
        correction over fifty rows and the lead and lag at one row, at eight
        lagged moments per pair instead of eighteen.

        ``serial_rule`` turns them into an honest count. ``t`` is built on
        ``n_kish``, which is right for unequal weights and silent about serial
        dependence. On a smooth stream consecutive rows are nearly the same
        observation, and the variance of a sample correlation is not ``1/n``
        but ``[1 + 2 * sum_l rho_x(l) * rho_y(l)] / n`` (Bartlett 1935).
        ``n_serial`` is ``n_kish`` divided by that bracket, and ``t_serial``
        the statistic against it. The rule says how the bracket sums the kept
        lags:

        .. list-table::
           :header-rows: 1
           :widths: 16 50 34

           * - rule
             - the bracket
             - ``n_serial`` and ``t_serial`` null when
           * - ``"truncated"``
             - the kept lags as they are
             - the bracket is zero or below (two series whose
               autocorrelations have opposite signs)
           * - ``"bartlett"``
             - lag ``l`` weighted by ``1 - l / (L + 1)``, ``L`` the longest
               kept lag, as Newey and West do: the long lags, whose estimates
               are the noisiest, count less, and over every lag ``1..L`` the
               bracket stays positive where the lagged products form a
               positive-definite sequence
             - with lags missing it can still reach zero, as under
               ``"truncated"``
           * - ``"geometric"``
             - ``rho(l) = phi^l`` fitted per series by least squares on
               ``log rho`` over the kept lags with ``rho > 0``, and the tail
               summed in closed form: the right choice when both series are
               exponentially weighted, and the reason the lags need not be
               dense; the fitted ``phi_x`` and ``phi_y`` are reported
             - fewer than two kept lags are positive on either side

        Measured on two independent AR(1) series with ``phi_x = 0.9`` and
        ``phi_y = 0.8``: ``t = 2.39``, significance that is not there, against
        ``t_serial = 1.03``, with ``n_kish = 3000`` becoming ``n_serial =
        557``. The lags add ``(L + 2C) * p * T + L * T`` doubles beside the
        pair moments, for ``L`` lags of which ``C`` keep the cross terms. They
        also hold a ring of ``max(lags)`` learned rows: the one place
        ``marginal`` holds rows rather than state.
    ``bins``, ``bin_rule``, ``bin_warm_rows``, ``bin_edges``, ``bin_budget``
        The nonlinear view. Every statistic above is linear, and a feature
        can be strongly related to a target with ``corr`` at zero: a
        threshold, a V, a saturation. With ``bins = 16`` each pair also
        reports the target's weight, mean and variance inside each of the
        feature's bins: its response curve, in ``bin_edges``, ``bin_n``,
        ``bin_mean_y`` and ``bin_var_y``. It reports the best single cut of
        that curve too. ``split_gain`` is the fraction of the target's
        variance the cut removes, a regression stump's gain, directly
        comparable with ``corr²``. ``split_at`` is where it falls.
        ``split_gain_t`` is the ``t`` a ``corr`` would need to match it. Read
        ``split_gain_t`` as a ranking, not a p-value: the cut was chosen by
        maximising over ``bins - 1`` candidates, which the statistic does not
        know. It uses ``n_serial`` in place of ``n_kish`` when ``serial_rule``
        gives one. The bins take ``O(bins)`` of state per pair, one search of
        the edges per feature per row, and a constant per pair.

        The edges are fixed once and never move, so a bin means the same
        thing for the life of the stream. ``bin_edges`` sets them outright,
        as a list per feature in ``features`` order or a dict keyed by
        feature name: exact, and comparable across runs and groups. ``bins``,
        ``bin_rule`` and ``bin_warm_rows`` describe learning them, and are
        refused beside it. Otherwise they are learned from the first
        ``bin_warm_rows`` learned rows (default 1,000) under ``bin_rule``.
        ``"quantile"`` (the default) gives equal weight per bin; a value that
        carries more than a bin's share, an indicator's zero say, fills a bin
        of its own, and the remaining bins share what is left. ``"fixed"``
        gives equal widths between the smallest and largest value seen. Those
        warm-up rows are held, not spent. The moment the edges exist every
        one of them is replayed with its own decay, so the histogram is what
        it would have been had the edges been known before the first row.
        That holds to the bit, or to about 1e-15 of the data's scale where
        rows of weight zero fall inside the warm-up, whose decays are carried
        to the next held row as one product. A feature keeps only the bins it can
        support, so the lists are ragged: a binary feature gets two bins
        whatever ``bins`` says, and a constant one a single bin and no split.
        Until the edges are fixed the bin columns are empty and the split
        columns null. Decay reaches the histogram as it reaches the pair
        moments, so a clock gap past ``gap_cap`` empties it along with them.

        The memory, per group, and per half-life when ``half_life`` is a list
        (every one keeps its own):

        .. list-table::
           :header-rows: 1
           :widths: 44 56

           * - what
             - about
           * - the hold, until the edges are fixed
             - ``bin_warm_rows * (8 * features + 16 * targets)`` bytes
           * - the histogram, for good
             - ``32 * features * targets * bins`` bytes

        Each is refused up front past ``bin_budget`` MiB, 256 by default and
        ``float("inf")`` for no bound. At the warm-up's last row the two
        exist at once, while the held rows are replayed into the histogram,
        so that row's peak is their sum.
    ``shards``
        Split the pair work across the bank's threads: a count of ranges of
        features, or ``"auto"`` for as many as the width keeps busy. The pool
        runs a bank's groups and specs in parallel already, so one wide spec
        on one group is one thread's work while the rest wait. With
        ``shards`` its pairs are cut into ranges of features, and each range
        runs on a thread of its own, a batch of rows at a time. Every number
        is the same to the bit whatever the count, so it is a setting and not
        part of the state. A saved bank resumes under the count of the specs
        given to ``load``, or under the saved one when given none.
        ``"auto"`` sizes itself to the machine it runs on: it estimates a
        batch's pair work from the width, the lags and the bins, and splits
        the batch into as many ranges as hold a tenth of a millisecond each,
        up to twice the pool's threads. The moments alone of fewer than about
        1,300 pairs stay whole. By default the pairs run row by row on the
        group's thread. Through the bank at 10,000 features on 14
        threads, ``"auto"`` ran 1.2 times as fast at one target and 4.9 times
        with nine targets, lags and bins; the bank's own work on each row
        does not split (``docs/PERFORMANCE.md`` §25).
    ``window_size``, ``closed``, ``window_every``, ``max_rows_between_snapshots``, ``window_budget``
        A hard cutoff on the history the pairs are computed from, in
        clock units, as for :func:`ewridge`: a row ``window_size`` old or
        older contributes nothing (``closed="both"`` keeps the one exactly
        that old, as for :func:`ewridge`), and inside the window the weights are still
        exponential. Every moment a pair is built from is truncated (the
        weight, both means and the three centred second moments), so
        ``corr``, ``beta`` and ``t`` describe the window and nothing else.
        That matters most for a screen: two regimes of opposite sign average
        to nothing over a long history. ``window_every`` spaces the snapshots
        on the clock and ``max_rows_between_snapshots`` caps the rows between
        them, whichever comes first, as for :func:`ewridge`;
        ``window_budget`` bounds each ring in MiB. The
        ``weight_sum`` the struct writes and the one the table reports are
        the weight inside the window. ``lags`` under a window take
        ``window_lags=True``, below. ``bins`` and ``window_size`` are refused
        together (a snapshot of the histogram is ``bins`` times the size of
        one).
    ``window_lags``
        Accept ``lags`` under a ``window_size``, and the memory that takes.
        Each of the window's snapshots then also holds the lag moments, in
        doubles, for ``L`` lags, ``C`` cross lags, ``p`` features and ``T``
        targets:

        .. list-table::
           :header-rows: 1
           :widths: 44 56

           * - a snapshot holds
             - doubles
           * - without lags
             - ``(3*p + 5)*T``
           * - the lag moments beside them
             - ``L*T + (L + 2*C)*p*T``, about ``(L + 2*C) / 3`` times its size
           * - with ``cross_lags=[]``, the least
             - ``L*(p + 1)*T``; ``n_serial`` does not read the cross terms

        So a snapshot is twice the size at one lag, and six times at five
        lags with the default cross lags (every lag). The snapshots count in
        ``window_budget``. Without it, ``lags`` and ``window_size`` together
        are refused with this spec's own numbers; with it, and without both,
        it is refused. Each windowed lag moment is the sum of the increments
        made inside the window, each centred at the mean as it stood when it
        was made. A lagged moment has no re-centring identity, so against
        the rows inside the window it is a statistical estimate, as the whole
        history's is. Default ``False``.
    ``feature_moments``
        Where the feature's mean and variance are kept. ``"per_target"``, the
        default, keeps them per pair, over the rows the pair's target was
        present. ``"shared"`` keeps one per feature, over every learned row,
        and each pair only its covariance. Where every target is on every
        learned row the two report the same numbers, to the bit. Where a
        target is absent on some rows, ``"shared"`` is a different estimator.
        ``mean_x`` and ``var_x`` are the feature's over every learned row,
        and ``cov`` is the target's rows centred on that mean, so ``corr``,
        ``beta`` and ``t`` move with it. That is sound where the absence says
        nothing about the feature. With a tenth of a target's rows absent at
        a half-life of 69 rows, measured, ``corr`` moved by at most 0.0065
        and ``var_x`` by 2.5% at the median. At 20,000 pairs it runs 2.7
        times as fast at ten targets and 3.2 times at thirty, the same at
        one, and a ten-target state is under half the size
        (docs/PERFORMANCE.md §27). With ``lags`` it keeps the feature's
        autocovariance per feature too, and at ten targets runs 4.6 times as
        fast with no cross lags. Refused with a ``window_size``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``fit_intercept`` and ``coef_every`` have nothing to act on here. Nothing
    residual-based applies (there is no prediction), so the residual switches
    are refused by name. A column may not be both a target and a feature.
    ``min_weight`` defaults to 3: two rows give a correlation of ±1 whatever
    the data, three the first one with content.

    .. rubric:: Output

    One struct column named after the spec holding ``weight_sum`` alone
    (`docs/OUTPUTS.md#marginal
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#marginal>`_).
    The pairs are read from the state with
    :meth:`polars_online.ModelBank.marginal`, one row per (group, instance,
    feature, target) with the moments, ``corr``, ``beta``, ``t``, ``n_kish``,
    and the lag and bin columns above. That method documents every column. A
    group that closes writes the same pairs to
    :meth:`polars_online.ModelBank.closed_groups` as ``pair_*`` columns, one
    entry per pair.

    .. rubric:: Example

    .. code-block:: python

        pairs = po.spec.marginal(
            "pairs", targets=["y", "ret"], features=["x0", "x1", "x2", "signal_a", "signal_b"],
            clock="t", gap_cap=300.0, half_life=500.0, group="stock_id",
        )
        bank = po.ModelBank([pairs])
        bank.fit_predict(df)                          # the struct holds weight_sum alone
        table = bank.marginal("pairs")   # a row per (group, instance, feature, target)
        one_stock = bank.marginal("pairs", group="b0")

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError``
    naming the problem for:

    - ``bin_edges`` that miss or add a feature;
    - ``bins`` beside ``bin_edges``, or beside ``window_size``;
    - ``lags`` beside ``window_size`` without ``window_lags``;
    - ``bin_budget`` without bins or not above 0, and bins past ``bin_budget``;
    - ``shards`` below 1, or a string other than ``"auto"``.
    """
    edges: list[list[float]] | None
    # `spec "m":` as every other refusal, where these read `marginal 'm':`
    # (review 2026-10-06, YA6).
    who = _who(name)
    if isinstance(bin_edges, dict):
        missing = [f for f in features if f not in bin_edges]
        if missing:
            raise ValueError(
                f"{who}: bin_edges is missing {missing}; give a list of "
                "edges for every feature, or pass a list of lists in features order"
            )
        extra = [f for f in bin_edges if f not in features]
        if extra:
            raise ValueError(f"{who}: bin_edges has {extra}, which are not features")
        edges = [list(bin_edges[f]) for f in features]
    elif bin_edges is not None:
        edges = [list(e) for e in bin_edges]
        if len(edges) != len(features):
            raise ValueError(
                f"{who}: bin_edges has {len(edges)} lists for "
                f"{len(features)} features; one list per feature, in features order"
            )
    else:
        edges = None
    if isinstance(shards, str) and shards != "auto":
        raise ValueError(
            f'{who}: shards must be a number of shards of at least 1 or "auto", got {shards!r}'
        )
    model: dict[str, Any] = {
        "type": "marginal",
        "lags": lags,
        "cross_lags": cross_lags,
        "serial_rule": serial_rule,
        "bins": bins,
        "bin_rule": bin_rule,
        "bin_warm_rows": bin_warm_rows,
        "bin_edges": edges,
        "bin_budget": bin_budget,
        "shards": shards,
        "window_size": window_size,
        "closed": closed,
        "window_every": window_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
        "window_lags": window_lags or None,
        "feature_moments": None if feature_moments == "per_target" else feature_moments,
    }
    # The keys the Rust spec skips when absent are written only when given,
    # as `holt`'s `trend` is, so `ModelBank.specs` is the dict made here.
    skipped = _SKIPPED_WHEN_ABSENT["marginal"]
    model = {k: v for k, v in model.items() if v is not None or k not in skipped}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def deco(
    name: str,
    *,
    features: list[str],
    dynamics: str = "ew",
    alpha: float | None = None,
    beta: float | None = None,
    blocks: dict[str, list[str]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Dynamic equicorrelation: one number for the whole correlation matrix, ``O(m)``
    a row (Engle & Kelly 2012).

    A correlation matrix of ``m`` series has ``m(m-1)/2`` free entries, and a
    stream cannot keep them all moving without ``O(m²)`` a row. DECO replaces
    them with their average and estimates that. Not a regression: no
    targets, and nothing residual-based applies. Needs at least two features.

    .. rubric:: The recursion

    The row is standardised against the pre-row means and variances of an EW
    accumulator, ``r_i = (x_i - m_i) / sqrt(v_i)``. With ``S1 = sum(r)`` and
    ``S2 = sum(r * r)`` over the ``n`` features the row's estimate is their
    Lemma 2.3:

    .. code-block:: text

        u = (S1**2 - S2) / ((n - 1) * S2)      = mean_{i != j} r_i r_j / mean_i r_i**2

    which lies in ``(-1 / (n - 1), 1)``. The level then follows one of two
    dynamics, on the model's own clock with decay factor ``lam`` and row
    weight ``w``:

    .. code-block:: text

        "ew":      W' = lam * W + w,  a = lam * W / W',  b = w / W'
                   rho' = a * rho + b * u                     (rho = u at W = 0)
        "linear":  rho' = (1 - alpha - beta) * rho_bar' + alpha * u + beta * rho
                   rho' = rho                                  (at w = 0)

    where ``rho_bar`` is the ``"ew"`` recursion run alongside as the target
    of the linear one. The ``"ew"`` dynamics run on the clock. The linear one
    steps once per row, as a DCC model's does, so a gap capped at ``gap_cap``
    moves ``rho`` by one row's ``alpha * u``, as a millisecond does. A row of
    weight 0 moves neither: it advances the clock and learns nothing, as
    everywhere. Otherwise the linear recursion has no row weights, as the
    paper's has none: a positive weight reaches ``rho`` only through
    ``rho_bar``, so rows of weight 0.5 and 2 move it by the same ``alpha *
    u``.

    Two departures from the paper, on purpose. Its eq. 21 has a free
    intercept and applies correlation targeting to the DCC ``Q`` recursion,
    not to the linear one. Writing the intercept as ``(1 - alpha - beta) *
    rho_bar`` is this library's reparameterisation, chosen because a
    streaming model has no sample to fit a free intercept on. And the paper
    permits ``alpha + beta`` slightly above 1 under numerical bounds, where
    this refuses it. The paper also notes that ``u`` is a downward biased
    estimate of the equicorrelation (``E[u]`` is about 0.20 for a true 0.30
    at ``m = 6``). It offers an alternative, ``1 - (1 / (n - 1)) * sum((r_i -
    rbar)**2)``, which this does not compute. Use ``u`` as a signal that
    moves with the market's correlation, not as the correlation. ``rho`` is
    not an ``ew_cov``'s ``corr`` over the columns, since the mean of a ratio
    is not the ratio of means.

    A column whose variance is exactly zero -- constant from its first row,
    the only way an EW variance is exactly zero -- has no standardised
    value. It is left out of its block's sums, so ``n`` counts the columns
    that have one. Its block reads the correlation among its other columns,
    and so does a pair it is in; a block left with fewer than two has no
    ``u`` on the row. Each value keeps its own weight ``W``: one with no
    ``u`` on a row learns nothing and does not decay, while the others
    learn. ``loglik`` needs every column, and is null on such a row.

    .. rubric:: Parameters

    ``dynamics``
        ``"ew"`` (the default) or ``"linear"``. ``"linear"`` needs ``alpha``
        and ``beta``, both ``>= 0`` with ``alpha + beta < 1``; ``"ew"``
        refuses them.
    ``alpha``, ``beta``
        The linear dynamics' weights on the row's estimate and on the
        previous level.
    ``blocks``
        A name to a subset of ``features``: the model then estimates one
        number per block and one per pair of blocks, which is the useful
        middle between one correlation and all of them. Every feature must
        be in exactly one block, and a block needs at least two.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``half_life`` or ``lam`` is required: both the standardiser and the level
    decay on it.

    .. rubric:: Output

    One struct column named after the spec, all read from the state before
    the row, so they are safe as features for that same row
    (`docs/OUTPUTS.md#deco
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#deco>`_):

    ``u``
        The row's own estimate of the equicorrelation.
    ``rho``
        The level as it stood before the row.
    ``loglik``
        The row's Gaussian log-density in standardised coordinates under
        that level.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where
        nothing decays, and why the row's fields are null where they are, as
        everywhere.
    ``coef``
        The correlation values, in the order of the ``u`` fields.

    With ``K`` named blocks the first two become ``u_<A>`` per block then
    ``u_<A>_<B>`` per pair, and ``rho_*`` likewise, with one ``loglik`` over
    all of them. ``u`` is null on a row where fewer than two of its block's
    columns have a positive pre-row variance, and ``loglik`` on a row where
    any column has none.

    .. rubric:: Example

    .. code-block:: python

        eq = po.spec.deco(
            "eq", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
            dynamics="ew",    # the EW mean of u; "linear" needs alpha and beta
        )
        blocked = po.spec.deco(
            "blocks", features=["x0", "x1", "x2", "signal_a"], half_life=500.0,
            # one number per block and one per pair of blocks: u_fast, u_slow, u_fast_slow
            blocks={"fast": ["x0", "x1"], "slow": ["x2", "signal_a"]},
        )
        out = po.ModelBank([eq, blocked]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``; ``ValueError`` for:

    - fewer than two features;
    - ``alpha``/``beta`` under ``"ew"``, or missing under ``"linear"``;
    - blocks that do not partition the features.
    """
    model: dict[str, Any] = {
        "type": "deco",
        "dynamics": dynamics,
        "alpha": alpha,
        "beta": beta,
        # `{}` as `[]`, refused by name as `feature_sets={}` is (YA5).
        "blocks": [[k, list(v)] for k, v in blocks.items()] if blocks is not None else None,
    }
    targets = _mirror_target(name, "deco", features, common, "its equicorrelation is over")
    return _common(name, model, targets=targets, features=features, **common)


#: The model types with no target column: their outputs are read from the
#: state before each row, their ``targets`` mirror ``features[0]`` for the
#: plumbing, and nothing residual-based applies to them. ``ew_class`` is
#: not one -- its label column travels as the target -- though it predicts
#: no number either, and refuses the residual switches the same way.
# The changepoint detectors', the regimes' and the realised covariance's
# builders live in files of their own (the 250 KB cap); read back here,
# after everything they use, for `__all__`, the key tables below and
# `polars_online.spec`.
from polars_online._spec_audit import audit  # noqa: E402
from polars_online._spec_changepoint import bocpd, corrchange  # noqa: E402
from polars_online._spec_regimes import hmm, rcov  # noqa: E402

UNSUPERVISED = frozenset(
    {"ew_cov", "kmeans", "micro", "deco", "rcov", "hmm", "corrchange", "bocpd", "audit"}
)

_NUMERIC_KEYS = _numeric_keys()
_CLOCK_KEYS = _clock_keys()
