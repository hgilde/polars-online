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
from typing import Any, Literal, NotRequired, TypedDict, Unpack

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
from polars_online._spec_words import (
    _INF_OK,
    _INF_REFUSED_BY_RUST,
    _WORD_REASONS,
    _bad_word,
    _one_of,
    word_reason,
    words_refusal,
)
from polars_online._warnings import forward_deprecated, forward_model


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
            # A model table's keys and words renamed in that model alone.
            if isinstance(v.get("type"), str):
                v = forward_model(_who(who), v["type"], v, stacklevel=5)
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
    if origin is Literal:
        # The words a parameter takes (task 244, P-1): a str among them.
        return isinstance(v, str) and v in typing.get_args(hint)
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
    if origin is Literal:
        words = _one_of(typing.get_args(hint))
        return f"strs, each {words}" if plural else words
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
    "huber": frozenset(
        {
            "huber_scale",
            "huber_scale_half_life",
            "feature_sets",
            "ridge_scale",
            "emit_huber_scale",
            "huber_refit_rows",
            "refit_budget",
        }
    ),
    "sgd": frozenset({"huber_scale", "huber_scale_half_life", "huber_refit_rows", "refit_budget"}),
    "quantile": frozenset({"quantile_refit_rows", "refit_budget"}),
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
            "bin_stat",
            "bin_huber_delta",
            "bin_refit_rows",
            "refit_budget",
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
        "n_clusters",
        "n_components",
        # A lasso of no sweeps never descends: every solve is a failure, and
        # the coefficients read like a fit (review 2026-10-05, PB7).
        "max_iter",
        # A cap of no rows is no schedule: every row is `coef_every = 0`
        # (docs/PLAN.md task 178), and in every row cap the clock form's `0`
        # is every row (task 196, U7).
        "max_rows_between_coefs",
        "max_rows_between_solves",
        "max_rows_between_snapshots",
        "max_rows_between_pcas",
        "max_rows_between_prunes",
        # rcov's: a jitter of 1 is none, and a stride of 1 every return; a
        # negative one was told ">= 0", and 0 then refused (task 278, b11).
        "jitter",
        "noise_stride",
        "iv_stride",
    }
)


#: The int parameters whose floor the Rust side refuses with its reason,
#: and the floor as this side's refusal of a negative one states it: a
#: negative one was told ">= 0", which is not the floor (task 278, b11).
#: Between 0 and the floor the Rust refusal, which gives the reason, is
#: left to speak.
_FLOOR = {
    "block_rows": "1",
    "preavg_rows": "2 (3 under psd = False)",
}


def _int_floor(key: str) -> tuple[int, str]:
    """The least an int parameter may be here, and how its refusal states
    the floor."""
    if key in _FLOOR:
        return 0, _FLOOR[key]
    floor = 1 if key in _AT_LEAST_ONE else 0
    return floor, str(floor)


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
        "max_rows_between_pcas",
        "max_rows_between_prunes",
        "max_iter",
        "update_every_rows",
        "split_merge_every_rows",
        "huber_refit_rows",
        "bin_refit_rows",
        "quantile_refit_rows",
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
    # Newey and West's lags are twice it.
    "horizon_rows": 2**19,
    "n_perm": 2**20,
}

#: Why a count's ceiling is where it is, said after the refusal where the
#: Rust side says it too (``online_core::MAX_PERM``'s message).
_CEILING_WHY = {
    "n_perm": "the draws are held to take their quantile, and each is a pass over both blocks",
}

#: A ceiling one builder's count has that its width does not say:
#: ``hmm``'s ``n_components`` sizes its transition matrix, its square in
#: cells, and its states before the first row (``online_core::Hmm::MAX_K``),
#: where ``kmeans``' ``n_clusters`` is held to 2^16 by the Rust side.
_CEILING_OF = {"hmm": {"n_components": 2**10}}


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
    "reset": "change_action",
    # Task 196 (docs/PLAN.md §18, N14 and N16): a count of rows says so, and
    # holt's level takes the spec's `half_life`, one knob under one name.
    "update_every": "update_every_rows",
    "split_merge_every": "split_merge_every_rows",
    "permute_every": "permute_every_rows",
    "level_half_life": "half_life",
}

#: The builders with no ``features`` keyword (task 244, P-6), and what each
#: reads instead. A spec dict or a TOML table still says ``features = []``:
#: every spec has the key there.
_NO_FEATURES = {
    "holt": "it reads only its targets, their own level and trend",
    "seqtest": "it reads only its targets, or the residuals of `a` and `b`",
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
        # And a keyword or a word renamed in this model alone (task 244).
        given = forward_model(who, fn.__name__, given, stacklevel=3)
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
            if key == "features" and fn.__name__ in _NO_FEATURES:
                raise TypeError(
                    f"{who}: {fn.__name__}() takes no features; {_NO_FEATURES[fn.__name__]}"
                )
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
                # A word outside the ones the parameter takes is a value
                # error, in the shape every surface gives it (task 244, P-4);
                # anything else of the wrong shape is a type error.
                if (bad := _bad_word(value, hint)) is not None:
                    if (reason := word_reason(who, key, bad[0])) is not None:
                        raise ValueError(reason)
                    why = _WORD_REASONS.get((fn.__name__, key))
                    said = words_refusal(who, key, *bad, hint=hint)
                    raise ValueError(f"{said}: {why}" if why else said)
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
                floor, stated = _int_floor(key)
                if value < floor:
                    raise ValueError(f"{who}: {key} must be >= {stated}, got {value}")
                # And at most its width, or the tighter ceiling of a count
                # that sizes an allocation (review 2026-10-06, YA4 and CD10).
                ceiling = _int_ceiling(key, fn.__name__)
                if int(value) > ceiling:
                    said = f"{who}: {key} must be <= {ceiling}, got {value}"
                    why = _CEILING_WHY.get(key)
                    raise ValueError(f"{said}: {why}" if why else said)
            # And each entry of a list of counts (review 2026-09-26, F7: a
            # negative lag was named by serde as `model`).
            if (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(x, numbers.Integral) and not isinstance(x, bool) for x in value)
                and list[int] in (hint, *typing.get_args(hint))
            ):
                floor, stated = _int_floor(key)
                if min(value) < floor:
                    raise ValueError(f"{who}: {key} must be >= {stated}, got {_got(value)}")
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
    """A target that is a formula of the row's future:
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


def formula_target(who: str, expr: pl.Expr) -> Target | FormulaTarget:
    """A window expression as a target table (docs/PLAN.md task 104): named by
    its alias, holding at least one operator looking ahead. A bare column
    under an alias, ``pl.col("p").alias("price")``, is Polars' spelling of
    ``po.target("p", name="price")``, and is read as it (task 244, M-11)."""
    from polars_online import _formula

    tree = _formula.to_tree(expr)
    name = expr.meta.output_name()
    # The alias names the target; the tree kept is the formula under it.
    if isinstance(tree, list) and tree and tree[0] == "alias":
        name = tree[2]
        tree = tree[1]
    if isinstance(tree, list) and len(tree) == 2 and tree[0] == "col":
        column = tree[1]
        if name == column:
            raise ValueError(
                f"{who}: target pl.col({column!r}) is the column itself: write the string "
                f"{column!r} in targets, or name it with .alias(...), which "
                f'po.target({column!r}, name="...") also does'
            )
        return target(column, name=name)
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
            f"{who}: target {name!r} holds no operator looking ahead (ewm_mean, ewm_sum, "
            f"ewm_rate or ewm_quantile with reverse=True), so it is known at its own row: "
            f"{make}, and name the column as the target"
        )
    return {"name": name, "formula": tree}


@removed_keywords("po.target", _RELATIVE_KEYS, _RELATIVE_REMOVED)
def target(column: str, *, name: str | None = None) -> Target:
    """A target column under a name of its own.

    ``po.target("p", name="price")`` in a spec's ``targets`` has the model
    learn the column ``p`` as it is, and name its output fields after
    ``price``: ``pred_price`` and the rest. Polars' own spelling of a column
    under another name, ``pl.col("p").alias("price")``, is read in
    ``targets`` as this same target. A string target is the same as
    ``po.target`` with no ``name``, and so is a ``name`` equal to the column.
    A target's column used as a feature is refused, whatever its name.

    A target computed from its own row's columns, such as a price against the
    mid, is a column: make it with Polars' ``with_columns`` in the query before
    the bank, and name it in ``targets``. For a return, a log ratio
    (``(pl.col("p") / pl.col("mid")).log()``) or a difference sits about zero,
    where ``hit_rate`` takes its sign; a plain ratio sits about 1, where two
    positive numbers always agree. ``relative_to`` and ``relative`` are
    refused, with a message that says this. A target computed from the rows
    after its own is a window expression looking ahead
    (:mod:`polars_online.ops`), named by its alias.

    In the CLI's TOML the same target is a table: ``targets = ["ret_5m", {
    column = "p", name = "price" }]``.

    Raises ``TypeError`` for a ``column`` or a ``name`` that is not a str, and
    ``ValueError`` for an empty one.
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
    drift_action: Literal["flag", "reset"] = "flag",
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
    horizon_rows: int | None = None,
    emit_tails: bool = False,
    tails_half_life: float | Duration | None = None,
    emit_influence: bool = False,
    influence_half_life: float | Duration | None = None,
    emit_feature_health: bool = False,
    feature_health_half_life: float | Duration | None = None,
    embargo: float | Duration | None = None,
    group: str | None = None,
    group_close: Literal["monotone", "session"] | None = None,
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
        "horizon_rows": horizon_rows,
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
    name: str,
    kind: str,
    features: list[str],
    common: Any,
    statistic: str,
    noun: str = "features",
) -> list[str]:
    """The ``targets`` a model with no target hands the plumbing: its first
    feature. ``targets=`` is refused by name, and so is an empty
    ``features``, which the Rust side names for every other model and these
    indexed before it could (review 2026-10-05, YA1 and YA5: an
    ``IndexError``, and a ``TypeError`` naming ``_common()``). ``noun`` is
    what the builder calls its features: ``audit``'s are ``columns``."""
    if "targets" in common:
        msg = f"spec {json.dumps(name)}: {kind}() takes no targets; {statistic} the {noun}"
        raise TypeError(msg)
    if not features:
        raise ValueError(f"spec {json.dumps(name)}: {noun} must be non-empty")
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
    ridge_scale: Literal["mean", "sum"] = "mean",
    coef_prior: list[list[float]] | None = None,
    session_shrink: float | None = None,
    long_half_life: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    gram_block_rows: int | None = None,
    gram_threads: int | Literal["auto"] | None = None,
    target_gaps: Literal["own_rows", "pairwise"] = "own_rows",
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
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
    (:func:`sgd`, :func:`pa`). There is no Huber loss here: :func:`huber` is
    these sums with each row down-weighted by its residual, and takes this
    model's ``feature_sets`` and ``ridge_scale``.

    .. rubric:: The fit

    Per row, with ``w`` the row's weight and ``lam`` its decay ``0.5 **
    (d_clock / half-life)``. ``W_j`` is the weight behind target ``j`` over
    the rows that target is present on:

    .. code-block:: text

        W_j' = lam * W_j + w
        S_j' = (lam * W_j * S_j + w * z z') / W_j'
        eta_j' = (lam * W_j * eta_j + w * z * y_j) / W_j'
        on the schedule:  (S_j + ridge * D) beta_j = eta_j      D = I with a 0 in the intercept slot

    The sums are means, not totals, so they stay bounded over a stream of any
    length. The second moments are kept centred (a weighted Welford update),
    so a feature far from zero loses no precision. The solve is a Cholesky
    factorization: a near-singular system is retried with a small diagonal
    jitter, and a solve that needed one is counted in
    :meth:`polars_online.ModelBank.solve_failures`.

    A system can also factorize and still be too nearly singular for its
    solve to keep a digit: rounding can leave a matrix past ``1/u`` in
    condition positive definite, and its coefficients then come out wrong by
    more than their own size. So each factorization that needed no jitter
    estimates the condition of the system with each column scaled by the
    root of its diagonal, ``H = D^-1 A D^-1``, which is what a Cholesky solve
    is accurate to, whatever the features' units. The estimate is
    ``||H^-1||_1`` by Hager and Higham's method (LAPACK's ``dlacn2``), from
    the factor in ``O(k^2)``. The factorization and solve are backward
    stable with a backward error of at most ``k * gamma_{3k+1}`` in ``H``'s
    entries, ``gamma_m = m u / (1 - m u)``, so the forward error of the
    scaled coefficients is at most ``2 kappa eta / (1 - kappa eta)`` with
    ``kappa eta = k * gamma_{3k+1} * ||H^-1||_1``. Where that bound reaches
    1, no digit is promised:

    .. code-block:: text

        withheld where   3 * k * gamma_{3k+1} * ||H^-1||_1 >= 1
        (k = 3: ||H^-1||_1 >= 1.0e14;  k = 10: 9.7e12;  k = 50: 4.0e11)

    There the fit is withheld: ``coef``, ``support_coef`` and the
    predictions are null, ``withheld_reason`` reads ``ill_conditioned``,
    and the solve is counted in ``solve_failures``. The next solve whose
    system is within the limit reports again. A ridge, ``standardize``
    with a ridge, or fewer collinear features keep a system inside it;
    ``rls``, which rotates the design rather than squaring it, loses half
    as many digits to the same design. A prediction uses the
    coefficients of the last solve and the state before the row. A wide
    solve runs on the bank's threads, and its result is the same to the bit
    whatever their number (``POLARS_ONLINE_MAX_THREADS``, or the machine's
    cores): at 2,000 features a factorization and solve took 18 ms on eight
    threads against 68 on one.

    .. rubric:: Parameters

    ``ridge``
        The penalty on the slopes, in the features' squared units unless
        ``standardize``; on the intercept too under ``ridge_scale = "sum"``,
        and never otherwise. Default ``1e-6``, finite and at least 0. A list
        fits one slot per value from the same sums, reported side by side as
        ``pred_<t>__r<ridge>``; a value listed twice is refused.
    ``feature_sets``
        Named subsets of ``features``, each a fit of its own from the same
        sums, reported as ``pred_<t>__<set>``. The full set is fitted only when
        it is one of them; ``emit_selected`` then reports the set doing best. A
        single set is no grid: it is fitted alone, and its fields take no
        suffix.
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

        Under ``"sum"`` a constant target of 5 reads an intercept of 3.5 at row
        20 under ``ridge=10``, ``half_life=50``, and 4.95 at row 200, until the
        prior fades. A ``coef_prior`` intercept is what it shrinks toward.
        ``"sum"`` is refused beside ``standardize``, beside a list of ``ridge``
        values and beside two or more ``feature_sets``.
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
        The blend changes the moments and nothing else: the coefficients stay
        as last solved until the next solve the schedule brings due, which
        reads the blended moments together with every row learned since. So
        the output is continuous in ``f``, and ``0`` is the same as no
        ``session_shrink``, to the bit. For the new session's first rows to
        read the blend at once, solve on every row
        (``max_rows_between_solves = 1``): the solve on the open's row reads
        it, and the row after is predicted from it.
        ``0`` keeps today's fit, ``1`` takes the long run's moments at today's
        weight, and ``f`` between fits on that share of the long run. Unlike
        ``session_gap`` this changes what the model believes, not how
        confident it is. With ``ridge_scale = "sum"`` the prior keeps today's
        scale too, so at ``1`` the fit is the twin's moments under today's
        prior. ``session_shrink`` is a share in ``[0, 1]`` and needs a
        ``session`` column; it is refused beside ``session_gap = "reset"`` and
        ``group_close = "session"``, which start the model over instead.
        ``long_half_life`` is above 0.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule: every ``solve_every`` clock units (finite and at
        least 0, ``0`` every row), and at least every
        ``max_rows_between_solves`` rows (at least 1). Left out, the schedule
        is by weight: a solve once the weight learned since the last reaches
        ``ln 2 / 50`` of the weight the fit holds. In steady state that is
        every ``half_life / 50`` of clock at any row spacing, and more often
        during warm-up and after a gap, where the fit moves most. A half-life
        far longer than the stream still solves, at rows further apart as the
        weight grows. ``half_life = inf`` and ``lam`` solve every row.
        ``max_rows_between_solves`` is off by default. The coefficients are the
        sums' as of the last solve. A target's first solve falls on the row
        its weight reaches ``k + 1`` (``k`` without an intercept), whatever
        ``min_weight`` above that says, which withholds the output alone: the
        fit is the same at any ``min_weight`` from ``k + 1`` up, on every
        schedule. A row of weight 0 is clock alone: it never
        solves and is no row of the cap, so a solve that comes due on one waits
        for the next row with weight, and the solves fall where they would
        without it.
    ``gram_block_rows``
        Hold that many rows back and bring the ``k x k`` matrix up to date once
        per block, by one matrix product instead of one rank-one update per
        row: ``256`` measured 6.6x faster at a thousand features. Default 0,
        off. The matrix is brought up to date before every solve too, so the
        block never exceeds the solve cadence. The option is refused where
        there is no cadence (``solve_every <= 0``, the default under ``lam``
        and ``half_life = inf``, or ``max_rows_between_solves <= 1``) and with
        ``window_size``. ``weight_sum``, the timing of every prediction and
        chunk invariance are unchanged to the bit. The blocked sum is the same
        sum in another order, so a blocked fit agrees with an unblocked one to
        rounding. Each target's cross-moments with the features wait for the
        block as well, and are merged from the same pass over its rows, so a
        target adds little beside the product: at 200 features about 0.15 ns
        per feature and row, where updating its cross-moments row by row
        took 0.35 (one fit, one core of an Apple M4 Pro). A target's weight,
        mean and variance still take every row. The held rows travel in the
        state file, with each target's value on them, and are refused over
        256 MiB, counting ``session_shrink``'s slow twin and, under
        ``target_gaps = "own_rows"``, a block per target.
    ``gram_threads``
        Run the update of the ``k x k`` matrix on up to that many threads: the
        block's matrix product under ``gram_block_rows``, and each row's
        rank-one update without it. A whole number of at least 1, or
        ``"auto"``. Default ``"auto"``. The threads come from the bank's
        pool, which the bank's ``threads`` sizes (``POLARS_ONLINE_MAX_THREADS``
        when it is ``None``). This option caps how much of that pool one
        spec's update of the ``k x k`` matrix may take, so a count above the
        pool's size runs on the pool's size, and it never adds a thread. A
        solve is not capped by this count: its factorization can use every
        thread of the pool. The output is the same to
        the bit at every count. Each entry of the matrix is still summed by
        one thread in one order, because the product is cut into pieces by
        ``k`` alone, never by the count.

        ``"auto"`` splits only where the pool has idle threads. On each
        chunk the bank counts the tasks it runs side by side, one per spec
        and group, ``T``, on its pool of ``P`` threads. When ``2 * T <= P``,
        each task's update may use ``2 * P // T`` threads, at most ``P``;
        otherwise it uses one. Twice the even share lets a thread whose group
        is done take pieces of another group's update. The rule reads the two
        counts and nothing else, so it does not depend on timing. Measured on
        14 cores against a count of 1: one group at 2,000 features learned
        3.0 times as fast, and 3.5 times with ``gram_block_rows=256``. Four
        groups at 800 features learned 1.6 times as fast, and seven were even
        to 1.1 times. From eight groups up, a split of every count was even
        or up to 10% slower, so ``"auto"`` does not split there. With every
        fast core already running a group, splitting each row's update only
        adds the cost of handing it out.
        Up to 256 columns (the features and the intercept) the product is one
        piece, and below 725 a row's update stays on one thread, so the option
        changes nothing there. Being the same to the bit at every count, it is
        a setting and not part of the state: a saved bank resumes under the
        count of the specs given to ``load``, or under the saved one when given
        none.
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
    ``window_size``, ``closed``, ``snapshot_every``, ``window_budget``
        A hard cutoff on the history the fit is solved from, in clock units,
        finite and above 0. A row ``window_size`` old or older is not in the
        sums at all, where the exponential weight alone would leave
        ``0.5 ** (age / half_life)`` of it. Inside the window the weights are
        still exponential, so this is a windowed exponentially weighted
        regression, not a rolling least squares. It is exact. The sums are sums
        of per-row contributions, so everything at or before a time ``u`` is
        ``lam ** (t - u)`` times the sums as they stood then, and subtracting
        that leaves the window. The model keeps a ring of snapshots to do it,
        which is the one place here where memory grows with a window rather
        than with the state. A half-life grid is one instance per entry, each
        with its own ring.

        ``closed`` is which edge holds the row exactly ``window_size`` old, in
        Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the default,
        keeps the rows less than ``window_size`` old, so that row has left, as
        it has in the window operators (:func:`polars_online.ewm_mean
        <polars_online.ops.ewm_mean>`) and in ``rolling_sum_by``. ``"both"``
        keeps it too. ``"left"`` and ``"none"`` would leave out the row the
        window ends at, which a model cannot do, since it reads its fit after
        it has learned that row: both are refused, as ``"both"`` is without
        ``window_size``.

        The ring takes a snapshot on every row unless told otherwise.
        ``snapshot_every`` spaces them on the clock, as ``solve_every`` spaces
        solves: a number of the clock's units, a duration on a temporal clock
        (``"1m"``), or ``0`` for every row. ``max_rows_between_snapshots`` caps
        the rows between them, as ``max_rows_between_solves`` caps the rows
        between solves, and whichever comes first takes a snapshot. A row of
        weight zero counts, a gap capped at ``gap_cap`` counts as the cap, and
        without a clock column the clock is the row's number. A coarser cadence
        divides the memory and can only shorten the effective window. Under a
        clock spacing, every row at most ``window_size - snapshot_every`` old is
        kept and none older than ``window_size``. So the effective window is in
        ``[window_size - snapshot_every, window_size]`` in clock units, exactly,
        with ``snapshot_every`` the spacing in force: a thinning
        ``window_budget`` doubles it. A burst of rows inside one spacing takes
        no snapshot of its own. ``snapshot_every``,
        ``max_rows_between_snapshots`` and ``window_budget`` are refused
        without ``window_size``.

        ``window_budget`` bounds each ring in MiB, above 0, and says what
        happens when a ring reaches the bound:

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - ``window_budget``
             - at the bound
           * - ``{"thin": mib}``
             - drops every other snapshot and doubles the spacing between the
               rest, the clock's or the rows' or both, as often as it takes;
               like ``snapshot_every``, that can only shorten the window
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
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
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

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them. Under a half-life grid every
    field is suffixed per instance, and under a ridge or feature-set grid each
    slot's fields (:mod:`polars_online.spec`, "What a spec writes"). The sums
    behind the fit are :meth:`polars_online.ModelBank.gram`'s, one entry per
    Gram (under ``target_gaps = "own_rows"``, one per set of targets present on
    the same rows). A group that closes writes one row per Gram to
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
    naming the problem for:

    - a ``feature_sets`` entry naming a column not in ``features``, and a
      ``coef_prior`` vector of the wrong length;
    - ``ridge_scale = "sum"`` beside ``standardize`` or a grid;
    - ``session_shrink`` without ``long_half_life`` or ``session``, or beside
      ``session_gap = "reset"`` or ``group_close = "session"``;
    - ``gram_block_rows`` beside ``window_size``, without a solve cadence, or
      past 256 MiB of held rows;
    - ``window_size`` beside ``ridge_scale = "sum"`` or ``session_shrink``, and
      the window's other parameters without it.
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
        "snapshot_every": snapshot_every,
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
    ``instance``
        The instance the field belongs to, by its suffix, as
        :meth:`~polars_online.ModelBank.coef`, :meth:`~polars_online.ModelBank.gram`
        and :meth:`~polars_online.ModelBank.closed_groups` name it: ``"@h500"``,
        or ``""`` for a single instance. Null on a field that chooses across
        the instances (``emit_selected``, ``emit_averaged``) and on the clocks,
        which are the spec's own.
    ``half_life``, ``lam``
        The decay of the instance the field belongs to, filled for a single
        instance as for a grid. ``half_life`` is the instance's half-life in
        clock units, in seconds under a duration (``300.0`` for ``"5m"``),
        ``inf`` for an infinite one, and null under ``lam``. ``lam`` is the
        spec's ``lam``, and null under a half-life.
    ``ridge``, ``feature_set``, ``penalty``
        The grid combination: the ridge value, the feature set's name, the lasso
        path point; null where the spec has no such grid.
    ``quantile``
        The level of a ``resid_quantiles`` or ``mahal_quantiles`` field.
    ``lag``
        The lag, in learned rows, of an ``ew_cov`` ``lag_corr`` field.
    ``component``
        The principal component of an ``ew_cov`` ``pc<j>_*`` field.
    ``state``
        The hidden state an ``hmm`` posterior is about (``filtered_state<s>``,
        ``predicted_state<s>``).
    ``class``
        The class an ``ew_class`` posterior is about (``p_<class>``).
    ``columns``
        The columns the field is computed from:

        - one for an ``ew_cov`` mean, variance or standard deviation, for a
          feature-health field, and for ``bocpd``'s predictive mean of a
          column (``pred_<column>``);
        - the pair for a correlation or a covariance, ordered for ``lag_corr``;
        - every feature for ``mahal``, the principal components and the other
          fields of ``kmeans``, ``micro``, ``ew_class``, ``hmm``, ``bocpd`` and
          ``corrchange``;
        - a block or a pair of blocks for ``deco``.

        Null where no column set describes the field, as for a prediction.
    ``dtype``
        The type the bank declares to polars before the first row is read:
        ``f64``, ``bool``, ``list[f64]``, ``enum`` (an ``Enum`` of
        its values in their order: ``withheld_reason``'s, an ``ew_class``
        spec's ``classes`` for ``class``, a grid's slots for
        ``selected_<t>``), ``i64`` (every integer: a count, a state, a row
        index, an id or a cluster label), or ``clock`` (``scored_clock`` and
        ``learned_clock``, in the clock column's own type).

    **Every name follows one grammar, which is stable.** Read left to right, a
    name is the kind (``pred``, ``abs_resid``, ``lag_corr``), then its
    qualifier, then the target or the columns it is about, then the grid point
    after ``__`` and the instance after ``@``. A qualifier is a number glued to
    the letter that says what it is: ``q`` a quantile level, ``l`` a lag
    (``abs_resid_q0.05_y``, ``mahal_q0.99``, ``lag_corr_l1_x0_x1``). The grid
    point uses letters too, ``r`` a ridge and ``p`` a lasso penalty, and the
    instance ``h``, a half-life. A thing the spec has several of is numbered
    where it is named, glued to its word: ``pc0_var``, ``filtered_state0``,
    ``coef_cluster0_x1``.

    This table is how to reach a field without constructing its name. No two
    rows hold the same values apart from ``field``, so a filter on the other
    columns reaches each field alone:

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
    return spec_output_index(_json(spec))


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
        The column
        :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest` gives it:
        ``coef_{target}_{term}{combo}{instance}``, so ``coef_y_x1__r0.5@h500``
        sits beside ``pred_y__r0.5@h500``. The ``coef_`` prefix and the target
        are there because a bare ``x1`` would collide with the feature column
        of that name in the same frame. :func:`deco`'s are
        ``coef_{slot}{instance}``, its slot ``rho`` or a block or pair of
        blocks (``coef_tech``, ``coef_tech_banks``).
    ``target``, ``instance``, ``half_life``, ``lam``, ``ridge``, ``feature_set``, ``penalty``
        As :func:`output_index` reports them. ``target`` is :func:`ew_class`'s
        label column, a :func:`deco` slot, and null for a :func:`kmeans`
        centre and an :func:`hmm` state, which belong to no target.
    ``cluster``, ``state``, ``class``
        The :func:`kmeans` centre, the :func:`hmm` state or the
        :func:`ew_class` class whose coordinates the slot holds; null for
        every other kind.
    ``term``
        ``"intercept"``, a feature name, ``"level"`` / ``"trend"`` for
        :func:`holt`, or ``"rho"`` for :func:`deco`.

    Empty for the kinds that report no coefficients: ``ew_cov``, ``seqtest``,
    ``marginal``, ``rcov``, ``corrchange``, ``bocpd`` and ``audit``, and for
    ``micro``, whose ``coef`` has no fixed layout. To reach one coefficient in
    the nested output without writing either name:

    .. code-block:: python

        grid = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], half_life=[100.0, 500.0], ridge=[1e-6, 0.5]
        )
        out = po.ModelBank([grid]).fit_predict(df)    # a half-life grid by a ridge grid
        cf = po.spec.coef_fields(grid)
        row = cf.filter(
            (pl.col("target") == "y")
            & (pl.col("term") == "x1")
            & (pl.col("ridge") == 0.5)
            & (pl.col("half_life") == 500.0)
        ).row(0, named=True)
        slope = out["m"].struct.field(row["field"]).list.get(row["position"])

    Rendered by the Rust code that names the fields, from the slot order the
    models lay the list out in. That order is the intercept first when the spec
    has one, then every feature -- zero for a feature outside a feature set --
    per (target, combination) slot, slots in the order the ``pred`` fields
    declare them.
    ``ValueError`` for a spec that is not valid, as for :func:`output_fields`.
    """
    return spec_coef_fields(_json(spec))


#: :func:`coef_index`'s columns, of a :func:`coef_fields` frame.
_INDEX_COLUMNS = (
    "position",
    "target",
    "ridge",
    "feature_set",
    "penalty",
    "cluster",
    "state",
    "class",
    "term",
)


def _index_columns(fields: pl.DataFrame) -> pl.DataFrame:
    """:func:`coef_index`'s columns of a :func:`coef_fields` frame: the one
    place they are chosen, so a bank with no coefficients lays out the empty
    frame :meth:`ModelBank.coef` gives with them too (review round 4, SF3)."""
    return fields.select(_INDEX_COLUMNS)


def _coef_index_schema() -> pl.Schema:
    """The schema of every :func:`coef_index` frame, whatever the spec: the
    empty layout of a spec with no coefficients."""
    return _index_columns(
        coef_fields(ew_cov("_", features=["_"], stats=["mean"], half_life=1.0))
    ).schema


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """The layout of each ``coef`` list, one row per position.

    ``coef`` is flat: (target x grid combination) slots, each contributing its
    terms in order. This maps ``position`` to ``target``, the combination's
    ``ridge``, ``feature_set`` and ``penalty``, and ``term`` --
    ``"intercept"``, a feature name, or ``"level"`` / ``"trend"`` for
    :func:`holt`. For :func:`kmeans` the slots are the centres, so ``cluster``
    reads ``0``, ``1``, ... and ``term`` is the feature whose coordinate the
    position holds; for :func:`hmm` a state likewise, in ``state``, and for
    :func:`ew_class` a class, in ``class``. For :func:`deco` a slot is
    ``rho``, or a block or a pair of blocks, in ``target``, and ``term``
    reads ``"rho"``.

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

    Every instance's list has this layout; :func:`coef_fields` is the same
    table per instance, with the field each list sits in and the column name
    each entry unnests to. ``ValueError`` for a spec that is not valid, as for
    :func:`output_fields`. Empty, with the same columns, for a spec with no
    fixed layout to index, as :func:`coef_fields` is:

    - ``ew_cov``, which has no coefficients;
    - ``seqtest``, which reports evidence;
    - ``micro``, whose ``coef`` has as many rows as there are established
      summaries;
    - ``marginal``, ``rcov``, ``corrchange``, ``bocpd`` and ``audit``, which
      report none.
    """
    cf = coef_fields(spec)
    if cf.is_empty():
        return _index_columns(cf)
    return _index_columns(cf.filter(pl.col("field") == cf["field"][0]))


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
    row, to better than 1e-9. For a Huber loss use :func:`huber` with
    ``ridge_scale = "sum"``, which is this fit with each row down-weighted by
    its residual, solved on a schedule.

    **A feature that holds one value for long winds the fit up: use**
    :func:`ewridge` **for a long-running stream whose features can go quiet.**
    The prior fades with the sums, so nothing holds a coefficient in a
    direction the data stop exciting. A feature held at a value ``c`` other
    than 0 shows the fit only ``intercept + c * slope``. The information that
    tells the slope from the intercept then decays by ``lam`` a row, with
    nothing to renew it. Once it is under a rounding step, rounding sets the
    slope, and the slope wanders. A feature constant from its first row winds
    up the same way. Held at exactly 0, it excites nothing in that direction,
    and the slope stays where it was. :func:`ewridge`'s default ``ridge`` is a
    penalty on the means that never fades, and it centres each feature, so it
    does not wind up. Measured with ``half_life=20`` (a row a clock unit),
    ``delta=1`` and noise 0.17, on a feature held for 150 half-lives after 300
    rows of moving, then moving again for 10:

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
        The prior's strength: ``A_0 = delta * I``, which is
        ``P_0 = I / delta``, the classic RLS name. Default 1.0, in the units of
        ``A``, a decayed sum of ``w * z z'``: the features' squared, summed
        over rows. It is :func:`ewridge`'s ``ridge`` under
        ``ridge_scale = "sum"``, not its default ``ridge``, a penalty on the
        means that never fades: this one penalizes the intercept too and fades
        as data arrives. ``ridge`` is refused here, with a message naming
        ``delta``.
    ``coef_prior``
        The coefficients the fit starts from and is shrunk toward, one vector
        per target in the features' original units, ``len(features) + 1`` long
        when there is an intercept; zeros when left out.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the model learned from, those with every
    target present, at their raw weights, decayed, where ``weight_sum`` counts
    every row, so rows with a null target do not warm up a fit they never
    reached.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#rls
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rls>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
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

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` for a
    ``coef_prior`` of the wrong shape.
    """
    model: dict[str, Any] = {"type": "rls", "delta": delta, "coef_prior": coef_prior}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def lasso(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    penalties: list[float],
    l1_ratio: float | None = None,
    select_half_life: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    target_gaps: Literal["own_rows", "pairwise"] = "own_rows",
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

    For each penalty ``l`` in ``penalties``, with ``C`` the feature correlation
    matrix and ``c_i = cov(x_i, y) / s_i``, each feature's covariance with the
    target over the feature's standard deviation. The descent runs until a
    sweep moves no coefficient ``b_i`` by ``tol * s_y`` or more, ``s_y`` the
    target's spread:

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

    ``penalties``
        The penalties, strictly decreasing, each finite and at least 0;
        required, and not empty. One slot per point, reported as
        ``pred_<t>__l<lambda>``.
    ``l1_ratio``
        The share of the penalty that is L1, in ``[0, 1]``. Default 1, the
        lasso; below 1 an elastic net.
    ``select_half_life``
        The half-life of the EW squared out-of-sample error each path point is
        ranked by, above 0. Default: the model's half-life; ``inf`` ranks on
        the plain mean over every row so far. ``penalty_selected_<t>`` is
        reported as it stood before the row -- the point this row was scored
        with, not the one its own error then elected. A row of weight 0 adds no
        error and ages the errors so far, so the selection moves only by what
        the ageing forgets. A target's errors count once the fit holds a row
        of it per coefficient: ``k + 1`` rows of positive weight with an
        intercept, ``k`` without, inside the window under one. They count
        whatever ``min_weight`` says, which withholds the output alone: the
        selection scores the prediction on a row the gate withholds, so the
        penalty chosen, as the fit, is the same at any ``min_weight``, and
        one target's choice does not depend on another's threshold. Before
        the first scored row
        ``penalty_selected_<t>`` is the path's last point, and a tie between
        points keeps the first in ``penalties`` order.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule, as for :func:`ewridge`. A target's first solve
        falls on the row it reaches ``k + 1`` rows of positive weight (``k``
        without an intercept), whatever ``min_weight`` says, where the
        schedule has not solved it sooner.
    ``max_iter``, ``tol``
        Within a solve, the descent stops after ``max_iter`` sweeps (default
        100, at least 1), or after a sweep that moved no coefficient in the
        standardized coordinates, ``b_i = beta_i * s_i``, by ``tol * s_y`` or
        more (default ``1e-10``, finite and above 0), or moved none at all.
        ``s_y`` is the target's spread over the rows the fit reads: its
        standard deviation, or its root mean square without an intercept,
        where the fit is not centred. The standardized coefficients are in
        the target's units, so the rule has none: ``y`` and ``penalties``
        scaled by a power of two give the same fit scaled, to the bit,
        stopped on the same sweeps. A descent that runs out of sweeps first is
        counted in :meth:`polars_online.ModelBank.solve_failures`, one per
        target and path point. :func:`polars_online.gram.lasso_path` runs the
        same descent with ten times the sweeps by default, 1,000, because it
        starts its path from zero. A solve here starts its first path point
        from the last solve's fit, which the rows since then have moved only a
        little once the moments settle, and each later point from the one
        before it. On 15 features correlated at ``0.8 ** |i - j|``, solved on
        every row of 20,000 at seven penalties, 798 of the 140,000 descents
        ran out of their 100 sweeps; 1,000 sweeps left 44, at 1% more sweeps
        in all. A cyclic descent contracts the error along two correlated
        columns by about ``1 - 1 / kappa`` a sweep, ``kappa`` the
        correlation matrix's condition number, so where columns nearly repeat
        each other most descents can run out: on 800 rows of two columns
        0.1 apart (``kappa`` near 400) at a penalty of 0.001, 798 of 799.
    ``target_gaps``
        Which rows a target's feature correlations are taken over where the target
        is null on some: ``"own_rows"``, the default, or ``"pairwise"``, as for
        :func:`ewridge`. The cross-correlations are centred at the target's own
        means either way.
    ``window_size``, ``closed``, ``snapshot_every``, ``window_budget``
        A hard cutoff on the history the path is fitted from, in clock units,
        as for :func:`ewridge`. A row ``window_size`` old or older is not in
        the sums, and ``closed="both"`` keeps the one exactly that old, as for
        :func:`ewridge`. ``snapshot_every`` spaces the snapshots on the clock and
        ``max_rows_between_snapshots`` caps the rows between them, whichever
        comes first, as for :func:`ewridge`; ``window_budget`` bounds each ring
        in MiB and thins or refuses past the bound. Each of the three is
        refused without ``window_size``. The selection error is truncated with
        the sums, so the ``lambda`` chosen is the one that fits the window
        rather than rows the fit has dropped. So a window can change the
        support, not just the coefficients: a feature with no evidence inside
        it goes to exactly zero. ``weight_sum``, ``sigma`` and ``zscore`` are
        the window's too.

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
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        Per (target, path point) slot, the intercept then one entry per feature
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as
    :mod:`polars_online.spec` describes them. The ``pred`` and ``resid`` fields
    are per target and path point, ``pred_<t>__l<lambda>``, and
    ``penalty_selected_<t>`` is the path point in force for the target, by
    lowest EW out-of-sample error. ``weight_sum``, ``coef`` and
    ``penalty_selected_<t>`` take no path suffix. The sums behind the path are
    :meth:`polars_online.ModelBank.gram`'s, as for :func:`ewridge`.

    .. rubric:: Example

    .. code-block:: python

        las = po.spec.lasso(
            "las", targets=["y"], features=["x0", "x1", "x2"],
            clock="t", gap_cap=300.0, half_life=600.0,
            penalties=[0.1, 0.01, 0.001],   # decreasing; every point is predicted
            l1_ratio=1.0,                    # below 1: an elastic net
        )
        out = po.ModelBank([las]).fit_predict(df)
        chosen = out["las"].struct.field("penalty_selected_y")   # the point in force, per row

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``penalties`` is
    required, and ``ValueError`` names a path that is empty or not strictly
    decreasing, an ``l1_ratio`` outside ``[0, 1]``, and a ``select_half_life``
    or ``tol`` that is not above 0.
    """
    model: dict[str, Any] = {
        "type": "lasso",
        "penalties": penalties,
        "l1_ratio": l1_ratio,
        "select_half_life": select_half_life,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "max_iter": max_iter,
        "tol": tol,
        "target_gaps": target_gaps,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
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

        b_j <- Phi b_j                     Phi = diag(2 ** (-d / kappa_i))  the reversion
        P_j <- Phi P_j Phi                                                  every row
        D_j <- D_j + d                     the clock since P_j's last observation
        P_j <- P_j + Q * D_j ** 2, D_j <- 0                                 the drift, on a row
                                                                            that observes y_j
        s    = z' P_j z + 1 / w
        k    = P_j z / s
        b_j <- b_j + k (y_j - z' b_j)
        P_j <- P_j - k z' P_j

    ``P_j`` is held in units of the observation noise ``R_j``, so the
    coefficients' covariance is ``R_j * P_j``, and ``Q`` is the process noise
    in the same units. ``R_j`` is the target's EW residual variance, unless
    given as ``obs_var``. That variance is per target, and its weight ages on
    every row, a row with no prediction or with weight 0 included. It learns
    from the filter's own prediction on every row the filter predicts,
    whether or not ``min_weight`` lets that prediction out: ``min_weight``
    withholds the output alone, so the fit, the variance and an explicit
    ``q`` read over it are the same at any ``min_weight``. A row with no
    prediction, before the filter has sized the target's coefficients,
    still moves them and adds to the target's weight; only the residual
    variance learns nothing from it. A row of weight 0, or one whose target
    is null, ages the target's weight and learns nothing for it.

    The drift is charged once, on the row that next observes the target, for
    the whole clock since the last one (``D_j``; under ``share_p`` any target's
    row), because ``Q * D ** 2`` is not additive over a split gap. Charged per
    row, a row of weight 0 inside a gap would shrink the gap's noise and move
    every later prediction (by 0.197 at ``coef_half_life = 20``). And a target
    seen on one row in 10 would forget 3.6 times slower on the clock than one
    seen on every row: 270 clock units against 75, measured at the same
    ``coef_half_life``.

    The noise is a scale, not a weight: the gain never reads the noise
    estimate. The model's noise is one constant per target, and the prior
    (``p0`` times it) and the drift (``(ln 2 * D / h) ** 2`` times it) are
    multiples of it, so it cancels from the gain, and the estimate scales
    only what is read out: ``pred_var``, ``se_coef`` and the readiness
    gates. An estimate inside the gain weighs each row by the inverse of the
    noise as it stood before the row, so a burst of large residuals inflates
    it and the rows after count for less than the burst did. With no drift, a
    fifth of the first 500 rows 50 too high held the slopes at ``[-25.5,
    33.6]`` where least squares on the same rows gives ``[4.1, -1.4]``. The
    residual variance starts at the target's first prediction. Neither the
    gain nor the prior reads the target's units: scaling a target by ``c``
    scales its predictions by ``c``. A state saved by a release before this
    form goes on as it was fitted, with the estimate in the gain.

    The intercept's prior is diffuse: the first row that observes the target
    fills the intercept with its whole innovation, ``b_0 = y - z' b``, at a
    variance of ``z' P z + 1 / w``, the limit of a prior that knows nothing of
    the level. So a level in the target moves every prediction by the level,
    to the rounding of a number of its size, from the first.

    .. rubric:: Parameters

    ``coef_half_life``
        How fast a coefficient may drift, as a half-life on the clock, on
        standardized features. An observation ``D`` clock units after the
        target's last adds the process noise ``(ln 2 * D / h_i) ** 2`` in
        units of the noise, which matches EW-RLS's steady-state gain at that
        spacing. So the half-life is the same whether
        rows come every unit or every hundredth, and whether the target is seen
        on every row or one in ten, since the noise is charged for the whole
        clock since the target's last observation. That is each standardized
        direction's memory with the others held still. Under correlated
        features, a direction of the design's correlation with eigenvalue
        ``lambda`` is learned with a memory of about ``h / sqrt(lambda)``
        (Ljung and Gunnarsson 1990), where EW-RLS forgets every direction at
        ``h``. At a correlation of 0.9 and ``h = 50`` the memories measured
        37.5 and 160.5 clock units, against 36.3 and 158.1 predicted. The
        gain reads no noise estimate, so a step in the truth is learned at
        that memory however much it inflates the residual variance. A
        scalar, or one value per slot with the intercept first; ``inf`` pins
        that coefficient; every value above 0. Required unless ``q`` is given,
        and refused beside it. Not the spec's ``half_life``, which drives the
        standardization and the residual variance. The diagnostics' default
        memory is the shortest finite entry; the windowed CUSUM's null share
        reads the intercept's entry, the memory at which a level in the
        target is absorbed, and ``inf`` there leaves the plain sum.

        Under ``standardize`` the drift is per standard deviation of each
        feature as the moments stand. A slope's random walk in the caller's
        units has a variance of ``q_i * D ** 2 / s_i ** 2`` per observation, a
        drift measured in the feature's current spread.
    ``q``
        The process noise given outright, ``q_i`` per slot, in place of the
        derivation from ``coef_half_life``; added as ``q_i * D ** 2`` on each
        row that observes the target. One value per slot, each finite and at
        least 0. Exactly one of the two is given. Its units are a
        coefficient's squared per clock unit squared: under ``standardize`` a
        slope is in the target's units per standard deviation of its feature,
        so ``q_i`` is in the target's squared units per clock unit squared;
        without it, in the target's units over the feature's, squared, per
        clock unit squared. The intercept's is the target's squared units per
        clock unit squared either way. Since ``P`` is held in units of the
        noise, ``q`` is measured against the noise estimate, ``q_i / R_j``,
        the one place the estimate reaches the gain. Before a target's first
        residual the row's own squared innovation stands in for it, and a row
        whose innovation is exactly 0 adds none.
    ``obs_var``
        A fixed observation noise, in place of the EW residual variance: the
        variance of the target about its prediction, in the target's squared
        units, finite and above 0. It leaves the gain as it is and scales the
        readouts; ``P`` is then absolute and starts at ``p0 * obs_var`` on
        every coefficient, the intercept included.
    ``p0``
        The prior variance of each slope, in units of the noise: a ratio, with
        no units under ``standardize``, where a coefficient and the noise are
        both in the target's units. Without ``standardize`` a slope's prior
        ``p0 * R`` carries its feature's units, so ``p0`` is in one over the
        feature's squared units there. Unstandardized, each slope's prior is
        ``p0`` from the start; standardized, on the first row that observes
        the target on which its feature's scale is usable, and until then
        the slope is 0 and takes no correction. The intercept's prior is
        diffuse (above), so it needs neither ``p0`` nor a noise: with no
        process noise the filter is least squares with the ridge ``1 / p0`` on
        each slope and none on the intercept, the noise's estimate playing no
        part. With no slot sized the target is not predicted, which with an
        intercept no row after the first meets. With ``obs_var`` given,
        ``P_0 = p0 * obs_var * I`` from the start, the intercept included,
        and unstandardized, with no process noise, the filter is the ridge
        regression with penalty ``1 / p0``. A covariance whose diagonal goes
        below 0 is sized again at ``p0``. The covariance form loses digits at
        a huge prior: as a stand-in for least squares it was 4.2e-6 off at
        ``p0=1e12``, against 3.4e-10 at 1e8, so keep ``p0`` well below 1e10.
        Default 1.0, a prior as uncertain as one observation; finite and above
        0.
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
        and no coefficient in the original units, from the first row. No row's
        raw reading of a feature reaches the state, and features scaled by a
        power of two predict the same to the bit. A filter read through the
        moments as they stand instead has a finite half-life's own wander move
        every prediction. Release 0.13.0 read it that way: on ``y = 2 x + e``
        with one feature ``x ~ N(0, 1)``, R² 0.99998, 30,000 rows and both
        half-lives 50, its out-of-sample error over rows 10,000 on stood 223
        noise variances above the noise (the median of seeds 0 to 4, 212 to
        244). This filter stands 0.014 above it there, as the unstandardized
        one does. The anchor gives the same filter, to rounding, as a re-map on
        every row, and is a fifth cheaper a row at ten features (221 against
        277 nanoseconds). A re-map past what a stretch of data makes -- a scale
        by more than 1024 times either way, or a mean by more than 1024 of the
        anchor's scale -- is not followed, and the state is read in the new
        coordinates.

        For a coefficient per current standard deviation, which moves with the
        scale, z-score the column in the stream and give it with
        ``standardize = False``: ``docs/WINDOWS.md``'s *Features in units of
        their spread* builds the column with :func:`polars_online.ewm_mean
        <polars_online.ops.ewm_mean>` and :func:`polars_online.ewm_std
        <polars_online.ops.ewm_std>`.
    ``revert_half_life``
        A reversion half-life ``kappa_i`` per slot, above 0: between
        observations the coefficient shrinks toward zero by ``2 ** (-d /
        kappa_i)``, so a
        coefficient no row has supported for a while is forgotten rather than
        carried. Default ``inf``, the random walk, which adds nothing. A scalar
        applies to every slot, the intercept included; a list gives one per
        slot, intercept first, and ``[inf, h, h]`` leaves the intercept a
        random walk. The pull is toward zero in the standardized coordinates
        when ``standardize`` is on: a slope toward "no effect", the intercept
        toward "the target averages zero". A reverting slot's process noise for
        an observation ``D`` after the last is ``q_i * g_i(D) ** 2``, with
        ``g_i(D) = (1 - 2 ** (-D / kappa_i)) / theta_i`` and
        ``theta_i = ln 2 / kappa_i``. That is ``q_i * D ** 2`` for a gap well
        under ``kappa_i``, and never more than ``q_i / theta_i ** 2``, so a coefficient's
        uncertainty stays bounded across a run of rows that observe nothing. At
        observations one clock unit apart that is short of ``q_i`` by about
        ``ln 2 / kappa_i``. A reverting slot settles, at observations ``D`` apart,
        at the prior variance ``q_i * g_i(D) ** 2 / (1 - phi_i ** 2)`` with
        ``phi_i = 2 ** (-D / kappa_i)``, a stationary AR(1) instead of an unbounded
        walk. The reversion runs on every row, a row that observes nothing
        included, being the same over a gap however it is cut. A prediction
        propagates the state by the same ``Phi`` over the row's clock gap.
    ``share_p``
        Keep one ``P`` for every target, where by default each target has its
        own. ``P``'s recursion reads neither the targets nor the noise (it is
        held in units of the noise), so targets observed on the same rows
        would each carry the same ``P``: one ``k x k`` update a row then serves
        every target, and every prediction is the per-target filter's, to the
        bit. Every target observed on a row takes its gain from ``P`` as the
        row finds it, and ``P`` takes the row once, so the order of the
        targets moves no prediction. Where a target is null on rows another is
        present on, the shared ``P`` takes those rows too, and that target's
        memory follows the others' rows; a target first seen after the
        others has its intercept filled by its first innovation, as the
        diffuse prior fills each target's. With an explicit ``q`` the shared
        ``P`` measures it against the mean noise over the targets that have
        one. A target's ``se_coef`` reads ``P`` times its own noise. Default
        ``False``: one ``P`` per target.

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
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the original
        units (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` moves the coefficients
    by ``Phi`` over the clock distance from the last learned row, capped by
    ``gap_cap``. So however far past the data, a coefficient has shrunk by
    ``2 ** (-gap_cap / kappa_i)`` at most, and the prediction is the intercept
    alone only where ``gap_cap`` spans several of the slopes' reversion
    half-lives and the intercept's is ``inf``.

    ``se_coef`` reads the filter's own covariance ``P`` and ``error_inflation``
    its prior ``z' P z / R``: both exact *under the model*, which says the
    coefficients walk at the noise ``coef_half_life`` implies. On a truth that
    does not move, that walk is variance the coefficients do not have. The
    standard errors then come out about ``sqrt(2)`` too large: a mean squared
    error over ``se_coef ** 2`` of 0.47 to 0.50 at ``coef_half_life`` 50 and
    200, against 0.95 to 1.06 on a walk of the implied variance. And with ``R``
    the EW variance of the out-of-sample residuals, which already carry the
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
    loss: Literal["logistic", "squared"] = "logistic",
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
    FTRL's objective. That is a fixed regularizer, ``l1 |b|_1 + (beta / alpha +
    l2) |b|^2 / 2``, against the rows' linearized losses and proximal terms,
    each weighted by its age on the clock. So old evidence counts for less
    against the prior. In steady state the penalties act as a mean-scale ridge
    of ``(1 - lam) * (beta / alpha + l2)``: a constant 5 settles at 4.65 at
    ``half_life = 100`` and 4.96 at 1000.

    Under a half-life the fit also follows a break at the pace of its steps,
    not of its half-life. A coordinate steps by its gradient over the root of
    its recent squared gradients, so a settled coefficient moves at most about
    ``2 * alpha * sqrt(1 - lam)`` a row, ``lam`` the decay over one row's
    clock, however large the error. A break that moves a coefficient by ``D``
    takes about ``D / (2 * alpha * sqrt(1 - lam))`` rows to follow, and the
    large gradients just after it raise the sums that divide every step for
    about a half-life. A larger ``alpha`` follows a break sooner and settles
    noisier; ``docs/MODELS.md`` has the measurements.

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
        The learning-rate scale and its smoothing. Defaults 0.1 and 1.0;
        ``alpha`` above 0 and ``beta`` at least 0.
    ``l1``, ``l2``
        The penalties: ``l1`` zeroes a coefficient whose evidence is below it.
        Defaults 0.0 and 1.0; each at least 0.

    None of the four is free of units. A coefficient is ``zz_i`` over
    ``r_i + l2``, and ``g_i`` is the target (a probability, under the logistic
    loss) times the feature, at the row's weight. So ``beta`` is in the units
    of ``sqrt(n_i)``, a gradient, ``alpha`` in the coefficient's, the target
    per unit of the feature, ``l1`` in the summed gradient's, and ``l2`` in
    ``r_i``'s, the feature squared. With ``fit_intercept=False``, every feature
    scaled by ``c`` fits the same model at ``alpha / c``, ``beta * c``,
    ``l1 * c`` and ``l2 * c**2``. The intercept's 1 does not scale, and the
    four are shared by every coordinate, so with an intercept there is no such
    equivalence.

    ``ftrl`` does not centre a feature. A coefficient's error reaches the
    prediction multiplied by the feature's value, so a feature that sits at a
    level ``L`` carries it at ``L`` times its size. On the stream of
    ``crates/online-core/tests/held_values.rs``, with one of three features at
    ``L``, the squared loss's error is about ``0.018 * L``: 18.3 at 1,000 and
    at -1,000, 1.8e6 at 1e8, where it is 0.75 at 0.5. Standardize or z-score
    such a feature before the bank, as ``docs/WINDOWS.md``'s *Features in
    units of their spread* shows. There is no ``standardize`` here because ``l1``
    zeroes a coefficient in the feature's own units: a scale inside the model
    would move which coefficients it zeroes as the feature's spread moved.

    ``strict_binary``
        Under ``loss = "logistic"``, refuse a chunk whose target is not 0 or 1,
        naming the row, before any stream is touched. Default ``False``: such a
        target is clamped into ``[0, 1]``. Refused beside ``"squared"``, which
        reads the target as it stands.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row. So rows with a
    null target do not warm up coefficients they never moved. A label
    ``strict_binary`` refuses is not one of them, nor a row skipped because its
    squared gradient would overflow.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#ftrl
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ftrl>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The accumulated weight before the row, as everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
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
    ``l1`` and ``l2`` refuse ``inf``, ``NaN`` and the values above, and
    ``strict_binary`` is refused beside ``"squared"``. Under ``"logistic"`` a
    window target is refused: the model fits a probability to a 0/1 target.
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
    stats: list[Literal["mean", "var", "std", "cov", "corr", "partial_corr", "mahal", "lag_corr"]]
    | None = None,
    bias: bool = False,
    precision_prior: float | None = None,
    mahal_quantiles: list[float] | None = None,
    pca: int | None = None,
    pca_every: float | Duration | None = None,
    max_rows_between_pcas: int | None = None,
    lags: list[int] | None = None,
    gram_threads: int | Literal["auto"] | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
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
    Welford's form, beside the sum of the squared weights ``Q``:

    .. code-block:: text

        W'     = lam * W + w
        Q'     = lam**2 * Q + w**2
        a      = lam * W / W'        b = w / W'        (a + b = 1)
        delta  = x - m
        m'     = m + b * delta
        C'_ij  = a * C_ij + a * b * delta_i * delta_j

    The second moments are read off ``C`` times a factor ``c``, and
    ``corr_ij = C_ij / sqrt(C_ii C_jj)``, from which ``c`` cancels:

    .. code-block:: text

        c       = W**2 / (W**2 - Q)       (1 under bias=True)
        var_i   = c * C_ii        std_i = sqrt(var_i)        cov_ij = c * C_ij

    A variance stays accurate when the columns sit on a large offset, where
    the raw ``E[x^2] - m^2`` form loses it. With ``P = (c * (C + s *
    precision_prior * I))^-1``, read before the row, ``s`` the prior's scale
    (1 on the first row, times ``a`` on each row after):

    .. code-block:: text

        partial_corr_ij = -P_ij / sqrt(P_ii P_jj)
        mahal           = sqrt((x - m)' P (x - m))

    One O(k²) update per row, which replaces the O(k²) passes a pure-Polars
    pairwise EW correlation needs. For moments on data that fits in memory,
    polars already does this under a ``window_size``:
    ``df.rolling(clock, period=window_size).agg(...)``, each row weighted
    ``2 ** (-age / half_life)`` from the window's last row, ``c`` from those
    weights, and shifted one row, since a row's moments are read before the
    row. On 20,000 rows about a minute apart, a three-hour window (about 180
    rows) and a half-life of 30 minutes, the mean of a column near 100 agreed
    to 8.5e-14, its variance to 6.4e-14 and its correlation with a second
    column to 4.3e-14. The reason to reach for the spec is a stream, a saved
    state, or the work: polars recomputes each window, ``O(n * W)``, where
    this is ``O(n)``.

    .. rubric:: Parameters

    ``stats``
        Which statistics to write, from ``mean``, ``var``, ``std``, ``cov``,
        ``corr``, ``partial_corr``, ``mahal`` and ``lag_corr``. Default
        ``["mean", "std", "corr"]``, so a single feature needs ``stats``
        without ``cov``, ``corr`` and ``partial_corr``, which need two columns.
        ``lag_corr`` needs ``lags``, and ``mahal_quantiles`` needs ``mahal``.
        ``[]`` writes no statistic, only ``weight_sum``, ``settled_frac`` and
        ``withheld_reason``, and accumulates all the same. The spec's value is
        then its state, read back with :meth:`polars_online.ModelBank.gram` and
        :meth:`polars_online.ModelBank.describe`. That is the form for a wide
        set of columns, where even the means are k values per row nobody reads.
    ``bias``
        Polars' ``bias``, with its default ``False``: ``var``, ``std`` and
        ``cov`` are then Polars' ``ewm_var`` and ``ewm_std`` and the
        covariance they imply, on a clock of one row a step and with Polars'
        default ``adjust=True``. The factor ``c`` above is ``n / (n - 1)`` at
        Kish's effective size ``n = W**2 / Q``, the number of equally weighted
        rows the moments are worth: Bessel's correction, ``ddof=1`` for ``n``
        unit rows with no decay. At a half-life of 20 rows it is 1.018 in
        steady state, at 2 rows 1.207. A second moment is null while the
        moments hold one row of weight, which has no spread to correct, as
        Polars' is. ``True`` reports the weighted means as kept, which is what
        :meth:`polars_online.ModelBank.gram` returns under either. ``c``
        multiplies the ridge on the co-moments with them, so it divides
        ``mahal`` by ``sqrt(c)``, multiplies ``pc<j>_var`` by its value at the
        refresh, and moves no correlation, share, loading or score.
    ``precision_prior``
        A ridge on the co-moments, in the features' squared units, above 0:
        the precision matrix ``P`` above, solved on each row it is read
        (O(k³), only when asked for). Like an RLS prior it fades as data
        accumulates, with ``s``. Optional here, and needed by
        ``partial_corr`` and ``mahal`` alone: the other statistics read the
        co-moments themselves, which need no inverse, so ``None`` adds no
        ridge to them. :func:`ew_class` and :func:`hmm` require it, since
        every row they score reads an inverse.
    ``mahal_quantiles``
        Levels in ``(0, 1)`` at which to keep the exponentially weighted
        quantile of the past ``mahal`` scores (``mahal_q<p>``), at the model's
        half-life and each row's weight, within ``tanh(1/128)`` (0.78%) of the
        exact one. That is a threshold from the stream's own history instead of
        a table, so ``mahal > mahal_q0.99`` is one row in a hundred without
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
             - its unit loading on each column
           * - ``pc<j>_score``
             - the row's coordinate ``v_j . (x - m)``

        The first refresh makes each component's largest loading positive, and
        each later refresh keeps the previous sign, so a loading never flips.
        ``pca`` is at least 1 and at most the number of features. The
        components are the same to the bit whatever the bank's thread count
        (``POLARS_ONLINE_MAX_THREADS``): the eigensolver runs at a fixed
        parallel degree of eight.
    ``pca_every``, ``max_rows_between_pcas``
        The eigendecomposition, O(k³), is refreshed after the row is folded in,
        every ``pca_every`` clock units or every ``max_rows_between_pcas`` rows,
        whichever comes first (both refused without ``pca``), as
        ``solve_every`` and ``max_rows_between_solves`` schedule a regression's
        solve. ``pca_every`` is a number of the clock's units, a duration on a
        temporal clock (``"5m"``), or ``0`` for every row; without a clock
        column the clock is the row's number. With neither given, every row. A
        row of weight zero advances both, as the window's snapshot cadence
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
        ``"lag_corr"`` to ``stats`` to write ``lag_corr_l<l>_<a>_<b>`` =
        ``C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`` per lag and ordered pair, the
        auto terms included. That is ``k²`` slots a lag, since a lagged
        matrix is not symmetric. ``C_l`` starts at 0 and takes its first
        product when a learned row has one ``l`` rows back, so ``lag_corr``
        reads 0.0, not null, until then: through the ``l``-th learned row of
        a group, wherever ``min_weight`` lets those rows through. Or read
        ``lags`` and ``lag_comoments`` (an ``(L, k, k)`` array) from
        :meth:`polars_online.ModelBank.gram`.
    ``gram_threads``
        Run the per-row update of the ``k x k`` co-moments on up to that many
        threads of the bank's pool, or as ``"auto"`` shares the pool out, as
        :func:`ewridge`'s ``gram_threads`` does: a cap on the pool the
        bank's ``threads`` sizes, never more threads. Default ``"auto"``. The
        output is the same to the bit at every count. The update is bound by
        memory: at 2,000 features a spec that wrote no statistic learned 2.5
        times as fast on eight threads and 3.1 times on fourteen
        (``docs/PERFORMANCE.md`` §37). Below 725 features a row's update stays
        on one thread, and the lagged cross-moments of ``lags`` always do.
    ``window_size``, ``closed``, ``snapshot_every``, ``window_budget``
        A hard cutoff on the history, in clock units: a row ``window_size`` old
        or older contributes nothing at all, where the exponential weight alone
        would still leave ``0.5 ** (age / half_life)`` of it, 12.5% at three
        half-lives. ``closed="both"`` keeps the row exactly that old, as for
        :func:`ewridge`. Inside the window the weights are still exponential,
        so this is not a rolling flat mean: the newest row dominates exactly as
        it does without a window. It is exact, because an exponentially
        weighted sum contains its own past. Everything at or before a time
        ``u`` is ``lam ** (t - u)`` times the accumulator as it stood at ``u``,
        so subtracting that leaves precisely the rest. What the model keeps is
        a ring of snapshots, one per row by default: the one place in this
        library where memory grows with a window rather than with the state. A
        snapshot is ``k² + k + 3`` eight-byte numbers, so a 1,000-row window
        over 20 columns is about 3 MB per group. ``snapshot_every`` spaces the
        snapshots on the clock (``"1m"``, or a number of the clock's units) and
        ``max_rows_between_snapshots`` caps the rows between them, whichever
        comes first, as for :func:`ewridge`. Either divides the memory, and a
        clock spacing alone holds a ring to at most
        ``window_size / snapshot_every + 2`` snapshots whatever the row rate.
        ``window_budget`` bounds each ring in MiB and thins or refuses past the
        bound, as for :func:`ewridge`.

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
               longer: in ``[window_size - snapshot_every, window_size]`` in
               clock units, exactly, under a clock spacing, with
               ``snapshot_every`` the spacing in force (a thinning budget
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
        with a ceiling. A clock gap longer than ``window_size`` empties it. The
        row after the gap reads the window as the row before the gap left it,
        since a row's fields are read before its own decay, as ``weight_sum``
        is. The rows after it report nulls rather than stale numbers, until the
        window holds ``min_weight`` again. ``mahal``, ``partial_corr`` and the
        PCA read the window's moments too, ``bias``'s factor is the window's
        ``W`` and ``Q``, and the PCA refresh is gated on the window's weight.
        ``window_size`` does not combine with ``lags`` or
        ``mahal_quantiles``, which accumulate over a history it does not
        truncate; both are refused by name.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.
    Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, holding the statistics ``stats``
    asks for: ``mean_<column>``, ``var_<column>`` and ``std_<column>`` per
    column; ``cov_<column>_<column>``, ``corr_<column>_<column>`` and
    ``partial_corr_<column>_<column>`` per pair, unordered, ``i < j``
    (``mean_x0``, ``corr_x0_x1``); ``lag_corr_l<l>_<a>_<b>`` per lag and
    ordered pair; ``mahal`` and ``mahal_q<p>``, the ``pc<j>_*`` fields,
    ``weight_sum``, and ``settled_frac`` and ``withheld_reason`` as everywhere.
    The statistics are null until ``min_weight``, which the bank floors at 2: a
    variance needs two rows, so a lower ``min_weight``, ``0`` included, is
    raised to 2 rather than refused. The rows the floor holds back read
    ``below_min_weight``, as the gate's do. ``coef_every`` and ``max_rows_between_coefs`` are
    refused, as there are no coefficients. The plain spec's fields are listed
    in `docs/OUTPUTS.md#ew_cov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_cov>`_.
    The moments themselves are :meth:`polars_online.ModelBank.gram`'s
    ``means``, ``comoments``, ``lags`` and ``lag_comoments``, the weighted
    means as kept whatever ``bias`` is, with ``n_kish`` to correct them by. A
    group that closes writes them to
    :meth:`polars_online.ModelBank.closed_groups` with ``eig_vals`` and
    ``eig_vecs`` of those ``comoments`` under ``pca``.

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

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got; ``ValueError`` for:

    - a statistic not in the list, ``cov``, ``corr`` or ``partial_corr`` over
      one column, and ``lag_corr`` without ``lags``;
    - a ``precision_prior`` not above 0;
    - an empty ``lags``, or one that does not increase;
    - an empty ``mahal_quantiles``, one without ``mahal``, or a level outside
      ``(0, 1)``;
    - a ``pca`` of 0 or past the features, and ``pca_every`` or
      ``max_rows_between_pcas`` without ``pca``;
    - a ``window_size`` not above 0, and ``window_size`` beside ``lags`` or
      ``mahal_quantiles``.
    """
    model: dict[str, Any] = {
        "type": "ew_cov",
        "stats": stats,
        "bias": bias,
        "precision_prior": precision_prior,
        "mahal_quantiles": mahal_quantiles,
        "pca": pca,
        "pca_every": pca_every,
        "max_rows_between_pcas": max_rows_between_pcas,
        "lags": lags,
        "gram_threads": gram_threads,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    # Written only when given, as the Rust spec skips it when absent.
    if gram_threads is None:
        del model["gram_threads"]
    targets = _mirror_target(name, "ew_cov", features, common, "its statistics are over")
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def pa(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    mode: Literal["pa", "pa1", "pa2"] = "pa1",
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
    for :func:`sgd`'s tube. It is the spread of ``y`` around its EW mean, over
    the rows with the target and a weight above 0, whatever the fit, each ``y``
    joining after its own row is judged. Before the target has a ``sigma``
    above 0, two weighted rows of different values, the tube has no width and
    every row teaches. A tube of 0.1 in the target's own units would hold a
    target in hundredths on every row: passive for ever, every prediction 0.0
    and R² -0.05, where the same target unscaled scored 0.96.

    **c is in the target's units.** ``pa1``'s ``tau`` is in the target's units
    over ``s``'s, so a cap of 1 caps every step of a target in thousands and
    none of a target in thousandths. The figures below are for a target of
    spread about 2. ``pa2``'s ``c`` is added to ``s`` as ``1 / (2c)``, in
    ``s``'s units, so it is free of the target's. Under ``"pa"`` and ``"pa2"``
    a target scaled by ``k`` fits as the unscaled one, scaled by ``k``; under
    ``"pa1"`` only while the cap does not bind.

    **On a level that moves, raise c or difference the target.** The fit
    starts from zero coefficients, and a ``pa1`` cap of 1 moves the intercept
    about one unit a row. On a target falling from 1,000 to -1,000 at 0.5 a row
    (half-life 200, noise 0.3), the intercept stood at 51.5 at row 50 and 401
    at row 400, and met the level near row 670. With the level as a feature
    too, the capped steps, mapped into the caller's units through the scaler,
    swung the fit far past the level instead. A drift that the intercept
    carries alone keeps a lag at any ``c``: 2.2 on average over rows 800 to
    1,600 at ``c = 1000``. About half of it is the tube, ``eps * sigma``, whose
    ``sigma`` holds the drift (97 to 143 here); with ``eps = 0`` the lag was
    1.1. Differencing the target and the moving features, ``y - y.shift(1)``,
    takes the level out of the fit: the level is then the last row's ``y`` plus
    the predicted change, and the change carries noise of 0.42 here. Measured
    as ``pred - y``:

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
    whole level. On a target at 1,000 with a spread of 2 they put the
    residual's std near 1,000, so a tube of ``eps = 0.1`` drawn in it would be
    about 100 wide and hold every row. Without decay it would never narrow:
    R² -52 at a half-life of 1e9.

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
        ``s``, in ``s``'s units and free of the target's. ``"pa"`` reads no
        ``c`` and refuses one. Default 1.0, above 0; ``inf`` caps nothing, so
        either bounded mode is then ``"pa"``.
    ``eps``
        The margin, in units of ``sigma``, at least 0: the row is close enough
        inside it and nothing moves. Default 0.01: errors under 1% of the
        target's own spread do not move the fit. Measured (``pa1``, one
        feature, a target of spread about 2, no decay; the out-of-sample error
        above the noise, in noise variances, median over seeds):

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
        347 noise variances over 20 seeds at 0.1. Over 108 streams and the
        three modes, the default had the smallest worst regret (R² from 0.5 to
        0.99998, a level of 0 or plus or minus 1,000, a half-life of 50, 500 or
        none, one feature or five). It was measured against a tube in the
        residual's spread capped by the target's, whose start-up residuals
        stayed in it for ever without decay: under ``pa2`` that tube left a
        target at a level of 1,000 at R² 0.76 to 0.80.
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes, exactly as for :func:`sgd`. The projection
        follows each update, so the step does not meet the row's margin exactly:
        it is the closest feasible coefficient to the one that would. A truth
        outside the feasible set is never realizable, so the model keeps stepping
        against the walls; a small ``c`` keeps those steps small.
    ``standardize``
        Take the step in standardized coordinates, with :func:`sgd`'s scaler,
        its rule and its warm-up. Each row is standardized against the running
        moments with the row admitted, ``z_i = (x_i - m_i) / s_i``
        (``z_i = x_i / s_i``, the root mean square, without an intercept), and
        a box or a sum is projected with its bounds carried over. Default
        ``True``: raw, ``s`` and ``c`` are in the features' units, so one
        feature in thousands makes every step tiny and one in thousandths every
        step the cap.

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

        The step still puts the row where the update says. The row's change of
        prediction is ``step * ||z||^2``, so an uncapped step at weight 1
        leaves it on the tube's edge, and one at weight ``w < 1`` goes that
        fraction of the way. The prediction reads no moment, so the moments'
        moving moves none of it. A fit read through the moments as they stand
        instead has a finite half-life's own wander move every prediction. At a
        half-life of 50 and R² 0.99998 that cost 31.8 noise variances of
        out-of-sample error above the noise, where the raw fit's was 0.18. Past
        the warm-up a box or a sum bounds the coefficients in the caller's
        units and is projected as for :func:`sgd`.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_weight`` counts the rows the target was present on, at their raw
    weights, decayed, where ``weight_sum`` counts every row, so rows with a
    null target do not warm up coefficients they never moved.

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
        How far the decay window had filled before the row, and which gate
        withheld the row's predictions, null where none did, as everywhere.
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

    As every builder does (:mod:`polars_online.spec`); ``eps`` refuses ``inf``
    and a value below 0, ``c`` a value not above 0, and the constraints are
    checked as for :func:`sgd`. ``c`` belongs to the modes ``"pa1"`` and
    ``"pa2"``, and is refused beside ``mode="pa"`` (``ValueError``), rather
    than ignored.
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
def kmeans(
    name: str,
    *,
    features: list[str],
    n_clusters: int,
    warm_rows: int | None = None,
    init: Literal["first", "farthest", "k-means++", "lloyd"] | None = None,
    seed: int | None = None,
    update_every_rows: int | None = None,
    split_merge: float | None = None,
    split_merge_every_rows: int | None = None,
    dead_frac: float | None = None,
    standardize: bool = True,
    scale_floor: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted k-means over the feature columns.

    Not a regression: there are no targets. Each row is assigned to the
    nearest of ``n_clusters`` centres before it is learned, so the label is
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

    ``n_clusters``
        The number of centres; required, and at most 65,536 (2^16): every row
        is scored against every centre.
    ``warm_rows``, ``init``, ``seed``
        Seeding. The first ``warm_rows`` learned rows are buffered: at least
        ``n_clusters``, and a smaller value is refused; default 500, or ``n_clusters`` where
        that is more. The buffer is held to 256 MiB, each row its values and
        its weight. Then the centres are placed by ``init``:

        .. list-table::
           :header-rows: 1
           :widths: 24 76

           * - ``init``
             - the centres
           * - ``"lloyd"`` (the default)
             - the best, by inertia, of ten ``"k-means++"`` starts, each
               followed by weighted Lloyd passes over the buffer until no row
               changes centre: once per group, 56 to 91 ms at 400 features
               and 8 centres (``"k-means++"`` 3.6 ms)
           * - ``"k-means++"``
             - greedy k-means++, scikit-learn's: each centre after the first
               is the best of ``2 + floor(ln n_clusters)`` rows drawn by their
               squared distance from the centres so far, the one that leaves
               the least inertia
           * - ``"farthest"``
             - Gonzalez, from the first row
           * - ``"first"``
             - the first ``n_clusters`` distinct rows

        ``seed`` (default 0) drives the two random rules; the same seed gives
        the same centres. ``None`` is 0, so a run draws the same centres
        every time. A ``"lloyd"`` start that is still moving rows after 300
        passes stops there and is counted in
        :meth:`polars_online.ModelBank.solve_failures`; the starts measured
        on 500 rows needed at most 54. Under ``"first"``, seeding waits past
        ``warm_rows`` if it must for ``n_clusters`` distinct rows, holding up
        to ``max(warm_rows, 1000)`` rows. The buffer is replayed into the centres
        and freed, so the model is O(state) again from that row on. Outputs are
        null until seeding and until ``weight_sum`` reaches ``min_weight``.
    ``update_every_rows``
        Learned rows between applications of the per-centre batches, at least
        1. Default 1.
    ``split_merge``, ``split_merge_every_rows``
        A row farther from its centre than a blob of the typical radius
        produces (about four standard deviations of ``|x - c|^2`` above it) is
        far. It is scored, but summarised instead of learned, so it neither
        drags the centre nor widens the radius. Every
        ``split_merge_every_rows`` learned rows (default 100, at least 1) the
        two closest centres are compared, where there are at least three. If
        their distance is under ``split_merge`` (default 0.5) times the sum of
        their radii, and enough far rows have gathered somewhere (at least
        three, and five per cent of the weight learned since the last check),
        they are merged. ``split_merge = 0`` turns off the merge and the far
        rows both, so every row is learned. The freed centre is placed at the
        far rows' mean. So a cluster that appears after seeding gets a centre
        without anyone restarting. Far rows still count in the radius at each
        check as if they sat at the cut, so a cut the data has outgrown widens
        until the rows are learned again.
    ``dead_frac``
        Re-place a centre whose weight has decayed below ``dead_frac *
        weight_sum / n_clusters`` the same way, on whatever far rows there are
        (default 0.05; ``0`` disables; below 0.5). The freed centre takes half
        the weight of the cluster it is split from, which keeps the other half,
        so from a cluster at an equal share both halves start at ``weight_sum /
        (2 n_clusters)``: the floor itself at 0.5. Each check then re-places
        one of them on the far rows, and the clustering is lost. Over 50
        streams with a check every 25 rows, a centre whose blob vanished was
        re-placed once on every stream at 0.45 and below; 3 streams cascaded at
        0.5, 29 at 0.6 and all 50 at 0.9. A centre whose cluster vanished is
        re-placed ``log2(1 / dead_frac)`` half-lives later (4.3 at the default,
        2 at 0.25). A cluster lighter than ``dead_frac / n_clusters`` of the
        stream loses its centre whenever any row is far. The rule runs at a
        split-merge check, so beside ``split_merge = 0``, which runs none, the
        default is 0 and a value above 0 is refused.
    ``standardize``
        Measure distances in units of each feature's EW standard deviation,
        tracked alongside the centres; the coordinates themselves are never
        rescaled, so the centres stay in the features' units. Default
        ``True``; ``False`` measures in the features' own units.
    ``scale_floor``
        Under ``standardize``, the metric's variance is floored at this
        fraction of the feature's long-run variance. ``1 / var`` alone grows
        as ``2^Q`` over ``Q`` half-lives of a feature going quiet (a flag that
        stops firing, a sensor at rest): a million at twenty, and the row on
        which the feature moves again is then infinitely far from every
        centre. Floored, the weight
        grows as ``2^(Q/8) / scale_floor``, about 57 at twenty. The long-run
        variance is tracked at eight times the half-life, a feature at a time.
        Each row's weight and deviation are clipped against it, and its start
        is the medians of the feature's first five rows, so a row at the input
        bound moves it by a factor of 26 at most, which a few of its half-lives
        undo. Default ``0.1``; ``0`` is the EW variance alone. Refused
        beside ``standardize=False``, which reads no variance to floor.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.
    Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#kmeans
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kmeans>`_):

    ``cluster``
        The nearest centre's index (``i64``), before the row is learned from;
        null until seeding.
    ``dist``, ``dist_second``
        The distance to that centre, and to the runner-up, the second-nearest
        (null when ``n_clusters == 1``), so ``dist_second - dist`` is the margin.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where nothing
        decays, and which gate withheld the row's fields, null where none did:
        ``below_warm_rows`` on the rows held to seed the centres from.
    ``coef``
        The centres: ``n_clusters`` rows of ``len(features)``, flattened
        cluster-major. :func:`coef_index` lays it out, with ``cluster``
        reading ``0``, ``1``, ... and ``term`` the feature whose coordinate
        the position holds.

    .. rubric:: Example

    .. code-block:: python

        km = po.spec.kmeans(
            "km", features=["x0", "x1", "x2"], clock="t", half_life=2000.0, gap_cap=300.0,
            n_clusters=3,
            warm_rows=100,           # seeding waits for this many rows, then replays them
            init="lloyd",       # ten k-means++ starts, each refined by Lloyd's passes
            split_merge=0.5,         # two centres in one blob: one moves to the far rows
            dead_frac=0.05,         # a centre whose blob vanished is re-placed 4.3 half-lives later
        )
        out = po.ModelBank([km]).fit_predict(df).unnest("km")
        layout = po.spec.coef_index(km)    # cluster = 0, 1, 2; term = the feature

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got; ``ValueError`` for:

    - ``n_clusters`` of 0 or past 2^16, ``warm_rows`` below ``n_clusters``, and a warm-up buffer
      past 256 MiB;
    - a ``init`` not in the list;
    - ``split_merge``, ``dead_frac`` or ``scale_floor`` below 0 or not finite,
      a ``dead_frac`` of 0.5 or more,
      ``dead_frac`` above 0 beside ``split_merge = 0``, and ``scale_floor``
      beside ``standardize=False``.
    """
    model: dict[str, Any] = {
        "type": "kmeans",
        "n_clusters": n_clusters,
        "warm_rows": warm_rows,
        "init": init,
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
    standardize: bool = True,
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

    A summary is ``(n, c, nu2)``: decayed weight, centre, and the EW mean
    squared distance of its rows from the centre (DenStream's radius, with the
    fading function being the decay). Distances are measured in the metric
    ``mw_i = 1 / var_i`` when ``standardize`` (the default), so ``eps`` is a
    bound per standardized coordinate and the bound in the metric is ``E =
    eps² p`` for ``p`` features. Each row:

    .. code-block:: text

        n_j <- lam n_j                                   every summary
        j   = the nearest potential summary, if absorbing a row of the mean weight keeps
              a nu2_j + a b |x - c_j|^2 <= E,  a = n_j/(n_j+w_bar), b = w_bar/(n_j+w_bar)
              else the nearest outlier summary by the same test
              else a new summary at x with the next id
        n_j <- n_j + w,  c_j <- c_j + w/n_j (x - c_j),  nu2_j <- min(., E)

    A summary is potential (established) once ``n >= beta_mu * w_bar`` and
    outlier below, ``w_bar`` the EW mean weight of the rows learned from. A new
    one is opened at the cap ``max_clusters`` by evicting the lightest outlier
    summary, else the lightest potential one. A checkpoint prunes and links, on
    the schedule ``prune_every`` and ``max_rows_between_prunes`` set (below).
    It drops potential summaries lighter than ``beta_mu * w_bar``, and outlier
    summaries lighter than DenStream's ``xi(age) * w_bar``, with ``xi(age) = (2
    ** (-(age + Tp) / h)
    - 1) / (2 ** (-Tp / h) - 1)``, ``sum_{i <= age / Tp} 2 ** (-i Tp / h)``
    where ``age / Tp`` is whole. That is the weight of a summary that had taken
    one row of the mean weight every ``Tp`` clock units since it opened, with
    ``Tp = ceil(h log2(beta_mu / (beta_mu - 1)))`` for half_life ``h``. Both
    count rows, as DenStream's do: a stream with more rows to a clock unit
    fills its summaries faster. With no decay, or with ``beta_mu`` at most 1,
    where ``Tp`` has no value, no outlier summary is pruned; a potential
    summary is still dropped below ``beta_mu * w_bar``, as ``w_bar`` rises.
    Then it links the potential summaries by single linkage: centres within
    ``L`` of each other share a label. Ids are monotone and never reused; a
    label is the smallest id in its chain, so it survives everything but the
    loss of that summary.

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
        column the clock is the row's number, the first row being 1, so
        ``prune_every=n`` checkpoints on the rows ``max_rows_between_prunes=n``
        does while every row has weight. A row of weight zero advances
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
        ``True``; ``False`` measures in the features' own units.
    ``scale_floor``
        Under ``standardize``, the metric's variance is floored at this
        fraction of the feature's long-run variance. ``1 / var`` alone grows
        as ``2^Q`` over ``Q`` half-lives of a feature going quiet (a flag that
        stops firing, a sensor at rest): a million at twenty, and the row on
        which the feature moves again is then infinitely far from every
        centre. Floored, the weight
        grows as ``2^(Q/8) / scale_floor``, about 57 at twenty. The long-run
        variance is tracked at eight times the half-life, a feature at a time.
        Each row's weight and deviation are clipped against it, and its start
        is the medians of the feature's first five rows, so a row at the input
        bound moves it by a factor of 26 at most, which a few of its half-lives
        undo. Default ``0.1``; ``0`` is the EW variance alone. Refused
        beside ``standardize=False``, which reads no variance to floor.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.

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
        Live clusters and live summaries (``i64``), so churn is visible
        without diffing labels.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where nothing
        decays, and which gate withheld the row's fields, null where none did:
        a warm-up of the model's own carries no reason.
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
    ``targets``, which this model has not got; ``ValueError`` for an ``eps``
    or ``beta_mu`` not above 0, a ``max_clusters`` of 0, a
    ``prune_every``, ``macro_link`` or ``scale_floor`` below 0 or not finite,
    and ``scale_floor`` beside ``standardize=False``.
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
    covariance_type: Literal["full", "tied", "diag"] = "full",
    precision_prior: float,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
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
        omega_c = precision_prior * s_c      s_c: the class's prior scale
        M_c  = C_c + omega_c I                              ("full", QDA)
        M    = sum_c pi_c (C_c + omega_c I)                 ("tied", LDA)
        M_c  = diag(C_c) + omega_c I                    ("diag", naive Bayes)
        l_c  = ln pi_c - 1/2 ln det M_c - 1/2 (x - mu_c)' M_c^-1 (x - mu_c)
        p_c  = exp(l_c - max_c' l_c') / sum_c' exp(l_c' - max l)

    then the labelled row's class learns it:

    .. code-block:: text

        n_c   <- lam n_c + w
        mu_c  <- mu_c + (w / n_c) (x - mu_c)
        C_c   <- weighted Welford on (x - mu_c_old)(x - mu_c_new)'
        n_c'  <- lam n_c'                         every other class c'

    The other classes' means, covariances and prior scales stay where they
    were, but their weights decay, which moves ``pi_c``. And
    ``weight_sum <- lam weight_sum + w`` counts every accepted row, labelled or
    not, so ``min_weight`` means the same number of rows as everywhere else. A
    row with a non-finite feature is null and learns nothing; a zero-weight row
    advances the clock.

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
    ``covariance_type``
        The shape, and the work a row takes:

        .. list-table::
           :header-rows: 1
           :widths: 20 36 44

           * - ``covariance_type``
             - the shape
             - per row
           * - ``"full"`` (the default)
             - a covariance per class
             - each class keeps its Cholesky factor, and only the class a row
               teaches is refactored: one ``k x k`` factorization per learned
               row; under a ``window_size`` every class's covariance moves on
               every row, so every class is refactored
           * - ``"tied"``
             - the weight-averaged one, so the decision boundaries are linear
             - one factorization
           * - ``"diag"``
             - the variances alone, which cannot see a correlation
             -
    ``precision_prior``
        A ridge on every class covariance, in the features' squared units and
        above 0, so the first rows of a class, whose sample covariance is
        singular, are scored with a finite, isotropic one; required, because
        every row is scored through each class's inverse covariance, where
        :func:`ew_cov` inverts only for ``partial_corr`` and ``mahal``. It is
        scaled by ``s_c``, the class's own prior scale, which starts at 1 and
        decays by ``lam * n_c / (lam * n_c + w)`` on every row the class
        learns. So the ridge washes out as the class fills in, exactly as
        :func:`ew_cov`'s ``precision_prior`` does.
    ``window_size``, ``closed``, ``snapshot_every``, ``window_budget``
        A hard cutoff on the history each class's moments are computed from,
        in clock units, as for :func:`ewridge`: a row ``window_size`` old or
        older contributes to no class (``closed="both"`` keeps the one
        exactly that old, as for :func:`ewridge`). That is what lets a
        classifier follow class means that move; over a long history two
        regimes average together and the labels go to chance.
        ``snapshot_every`` spaces the snapshots on the clock and
        ``max_rows_between_snapshots`` caps the rows between them, whichever
        comes first, as for :func:`ewridge`; ``window_budget`` bounds each
        ring in MiB. ``weight_sum`` is the weight inside the window, in
        the struct and in the ``min_weight`` gate, and the class moments a
        row is scored against are the window's.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.
    Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is
    learned (`docs/OUTPUTS.md#ew_class
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_class>`_):

    ``class``
        The class with the largest posterior (the first, on a tie), an
        ``Enum`` of ``classes`` in their order; null before ``min_weight`` and
        while no class has been seen.
    ``p_<class>``
        One per class in ``classes`` order: its posterior probability, so the
        ``p_`` fields sum to 1 -- exactly 0 for a class no row has carried
        yet.
    ``weight_sum``
        As everywhere.
    ``settled_frac``, ``withheld_reason``
        How far the decay window had filled before the row, null where nothing
        decays, and which gate withheld the row's fields, null where none did:
        a warm-up of the model's own carries no reason.
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
            covariance_type="tied",         # pooled by class weight: linear boundaries
            precision_prior=0.1,         # the ridge that makes a class scoreable from its first row
            min_weight=20.0,
        )
        out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
        calls = out.select("dir", "class", "p_up", "weight_sum").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets`` -- this model takes ``label``; ``ValueError`` for:

    - fewer than two ``classes``, and a class listed twice or named ``""``;
    - a ``covariance_type`` not in the list;
    - a ``precision_prior`` or a ``window_size`` not above 0;
    - a ``label`` also listed as a feature.
    """
    model: dict[str, Any] = {
        "type": "ew_class",
        "classes": classes,
        "covariance_type": covariance_type,
        "precision_prior": precision_prior,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
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
        Compare two specs of the same bank. Each target ``t`` names a residual
        field both carry: ``resid_<t>`` on each side, plus the side's grid
        suffix when it is a grid (``a_suffix="@h50"`` picks ``resid_<t>@h50``
        of ``a``; ``"__r0.1"`` a ridge instance). The sign tested is that of
        ``|resid_b| - |resid_a|``, positive when ``a`` was closer on the row.
        Any loss that grows with ``|resid|`` (squared, absolute, Huber) gives
        the same sign, so this is a test of "``a`` beats ``b``" under any of
        them. The bank runs ``a`` and ``b`` first, so the comparison reads the
        same out-of-sample residuals their structs report. A row where either
        side is null (warm-up, a skipped row) is a row the test sits out. ``a``
        and ``b`` are given together, and a suffix only beside them. When the
        bank is built, a spec named against itself with the same suffix, a side
        that is not in the bank or is itself a ``seqtest``, and a target either
        side has no residual for are refused by name. Two instances of one grid
        (``a_suffix="@h20"`` against ``b_suffix="@h400"``) are a comparison
        like any other.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.

    A trial is a row, so ``weight`` is refused and there is no
    ``half_life``/``lam``: a process that forgot its losses would not be an
    e-process. A session change restarts it under ``session_gap = "reset"`` or
    ``group_close = "session"``, and so does a step back past
    ``restart_after_step_back``. ``min_weight`` defaults to 0. No ``features``
    (the column is the test; a spec dict or a TOML table says
    ``features = []``), no ``coef``, and nothing
    residual-based applies -- there is no prediction.
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
        How far the decay window had filled before the row, null where nothing
        decays, and which gate withheld the row's fields, null where none did:
        a warm-up of the model's own carries no reason.

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

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``features``, which this model has not got; and ``ValueError`` for
    ``weight``, ``half_life`` or ``lam``, ``a`` without ``b`` or the reverse,
    and a suffix without them. The bank refuses
    the comparisons above when it is built.
    """
    model: dict[str, Any] = {
        "type": "seqtest",
        "a": a,
        "b": b,
        "a_suffix": a_suffix,
        "b_suffix": b_suffix,
    }
    return _common(name, model, targets=targets, features=[], **common)


@_checked
def marginal(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    lags: list[int] | None = None,
    cross_lags: list[int] | None = None,
    serial_rule: Literal["truncated", "bartlett", "geometric"] | None = None,
    bins: int | None = None,
    bin_rule: Literal["quantile", "fixed"] | None = None,
    bin_warm_rows: int | None = None,
    bin_edges: dict[str, list[float]] | list[list[float]] | None = None,
    bin_budget: float | None = None,
    bin_stat: Literal["mean", "huber"] = "mean",
    bin_huber_delta: float | None = None,
    bin_refit_rows: int | None = None,
    refit_budget: float | None = None,
    shards: int | Literal["auto"] | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    window_lags: bool = False,
    feature_moments: Literal["per_target", "shared"] = "per_target",
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
        the same statistic ``ew_cov(lags=)`` computes, to the bit. As there, a
        lagged moment starts at 0 and takes its first product when a learned row
        has one ``l`` rows back, so each lag reads 0.0, not null, while a group
        has learned no more than ``l`` rows. Below ``min_weight`` the lag
        correlations are null, as ``corr`` is, and so are ``phi_x`` and
        ``phi_y``, which are fitted from them.

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

        ``serial_rule`` turns them into an honest count. ``t_stat`` is built on
        ``n_kish``, which is right for unequal weights and silent about serial
        dependence. On a smooth stream consecutive rows are nearly the same
        observation, and the variance of a sample correlation is not ``1/n``
        but ``[1 + 2 * sum_l rho_x(l) * rho_y(l)] / n`` (Bartlett 1935).
        ``n_serial`` is ``n_kish`` divided by that bracket, and ``t_stat_serial``
        the statistic against it. The rule says how the bracket sums the kept
        lags:

        .. list-table::
           :header-rows: 1
           :widths: 16 50 34

           * - rule
             - the bracket
             - ``n_serial`` and ``t_stat_serial`` null when
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

        Measured on 1,000 pairs of independent AR(1) series, ``phi_x = 0.9``
        and ``phi_y = 0.8``, each 3,000 rows with no decay,
        ``lags = [1, 2, 3, 5, 8, 13]`` and ``"geometric"``: ``t_stat`` spread
        with a standard deviation of 2.52, near Bartlett's
        ``sqrt(1 + 2 * 0.72 / (1 - 0.72)) = 2.48``, and passed 1.96 on 45% of
        them: significance that is not there. ``t_stat_serial`` spread 1.04 and
        passed on 5.4%, its ``n_serial`` a median 511 of ``n_kish = 3000``. The
        lags add ``(L + 2C) * p * T + L * T`` doubles beside the pair moments,
        for ``L`` lags of which ``C`` keep the cross terms. They also hold a
        ring of ``max(lags)`` learned rows: the one place ``marginal`` holds
        rows rather than state.
    ``bins``, ``bin_rule``, ``bin_warm_rows``, ``bin_edges``, ``bin_budget``
        The nonlinear view. Every statistic above is linear, and a feature can
        be strongly related to a target with ``corr`` at zero: a threshold, a
        V, a saturation. With ``bins = 16`` each pair also reports the target's
        weight, mean and variance inside each of the feature's bins: its
        response curve, in ``bin_edges``, ``bin_n``, ``bin_mean_y`` and
        ``bin_var_y``. It reports the best single cut of that curve too.
        ``split_gain`` is the fraction of the target's variance the cut
        removes, a regression stump's gain, directly comparable with ``corr²``.
        ``split_at`` is where it falls. ``split_gain_t`` is the ``t_stat`` a
        ``corr`` would need to match it. Read ``split_gain_t`` as a ranking,
        not a p-value: the cut was chosen by maximising over ``bins - 1``
        candidates, which the statistic does not know. It uses ``n_serial`` in
        place of ``n_kish`` when ``serial_rule`` gives one. The bins take
        ``O(bins)`` of state per pair, one search of the edges per feature per
        row, and a constant per pair.

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
    ``bin_stat``, ``bin_huber_delta``
        Each bin's location. ``"mean"``, the default, is the weighted mean,
        which one spike in a thin bin moves by its size times its share: a row
        of 50 among twenty rows near 0 moves it by 2.4, a shape where there is
        none. ``"huber"`` is Huber's M-estimate of location, learned in one
        pass. A row's weight is cut back by how far it lies from the bin's
        location before it, as :func:`huber` weighs a regression's rows, and
        the location is the mean over those weights:

        .. code-block:: text

            c      = bin_huber_delta * sigma_t          the cut
            omega  = w * min(1, c / |y - m|)            Omega' = lam * Omega + omega
            m'     = m + (omega / Omega') * (y - m)

        Its fixed point is the Huber M-location of the bin's decayed sample,
        and one row moves it by at most ``c * w / Omega'``, that is
        ``c * w / (lam * Omega + omega)``. ``sigma_t`` is the
        target's robust scale, read before the row: the exponentially
        weighted median of ``|y - m_t|`` over 0.6745, ``m_t`` the target's own
        location by the same recursion. It is one per target, not per
        feature: a cut is for a row far from the target's spread, the same row
        whichever feature bins it, and per feature it would take a quantile
        sketch per pair. The sketch keeps each deviation in units of the
        first, as :func:`huber`'s does, so a rescaled target rescales the
        cuts and the locations to rounding. A bin's first row is judged against ``m_t``. Until
        the target has a scale above 0, its first two rows, nothing is
        clipped, as :func:`huber` down-weights nothing before its scale
        exists. ``bin_huber_delta`` is the cut in units of ``sigma_t``,
        default 1.345, :func:`huber`'s; ``float("inf")`` clips nothing, which
        is the mean to the bit.

        Under ``"huber"``, ``bin_mean_y`` is the location and ``bin_n`` the
        rows' weight, as under the mean. ``bin_var_y`` is the weighted mean
        square of the deviations clipped at the cut, the variance where
        nothing is clipped. ``split_gain`` is read from those. Against
        statsmodels' batch M-location of each bin, measured on 20,000 rows in
        16 bins with no decay and spikes of 50 on 1%, 5% and 12% of the rows,
        it lands within 0.001, 0.002 and 0.03 of the target's scale, where
        the mean is off by 0.45, 1.5 and 3. It keeps a fifth more per cell, 40
        bytes in place of 32, and up to 9 KB per target, and a row takes 0.75
        to 0.9 times the mean's throughput.
    ``bin_refit_rows``, ``refit_budget``
        The refit of the first rows under ``"huber"``. One pass weighs each
        row against the location and the scale before it, and over the first
        rows those are still settling. A share of spikes then pulls the first
        locations, the scale learns the pulled deviations, the cut is drawn
        wide, and the spikes go on entering at nearly full weight. Without
        decay the bins stay there: captured. Measured on 16 bins with spikes
        of about 20 of the target's scales and no decay, the worst of 50
        streams had a bin 1.3 scales from statsmodels' M-location after
        2,000 rows at 12% spikes, and 3.5 at 20%.

        So once ``bin_refit_rows`` learned rows (rows of positive weight) are
        in, every target's location and scale and every bin's location are
        refit to the converged batch fit of those rows, once. The target's
        location and MAD scale are fitted jointly, as statsmodels' ``RLM``
        fits a constant with ``scale_est="mad"``. Its scale is rebuilt from
        the converged deviations, and each bin's location is solved exactly
        at the cut that scale gives. Then the stream goes on in one pass. The
        refit runs after the row's own update, on a count of rows, so it is
        the same whatever the chunking. It moves no weight: ``bin_n``,
        ``weight_sum`` and every pair moment are those of the one pass. With
        it the same worst streams were 0.015 and 0.14 scales away after
        2,000 rows, and 0.004 and 0.057 after 20,000.

        The default is ``max(500, 30 * bins)``, ``bins`` the most any
        feature has: a bin needs about 30 rows at the refit. A bin whose rows
        are then mostly spikes has its location among them, rightly for those
        rows, and the rows after it bring it back only slowly. At 20% spikes,
        12.5 rows a bin left 6 of 50 streams half a scale off after 20,000
        rows, and 31 rows a bin left none. These figures are at the default
        cut, ``bin_huber_delta=1.345``. A wider cut converges more slowly: at
        2.5 with 12% spikes, the worst of 50 streams was 0.30 scales off after
        1,000 rows with the refit, against 1.25 without it. ``0`` turns the
        refit off, and is the one pass alone. Under ``"mean"``, or
        ``bin_huber_delta=inf``, there is nothing to refit, so the default is
        0, and either key given is refused.

        With learned edges the warm-up already holds its rows: the refit runs
        when the edges are fixed, over every held row, if the count is within
        ``bin_warm_rows``, and holds no row of its own. With ``bin_edges``, or
        a count past the warm-up, the model holds the rows until the count:
        ``bin_refit_rows * (8 * features + 16 * targets + 64)`` bytes per
        group, 4 MB at 1,000 features. ``refit_budget`` bounds those holds
        over all of a spec's groups, in MiB, 256 by default and
        ``float("inf")`` for no bound. Each new group reserves its hold when
        it first appears; the group that would pass the bound keeps the one
        pass, as does every new group after it. A notice says so once, and
        ``summary()``'s ``refit_given_up`` counts their targets. The refit
        itself runs once, in about 1 ms at 10 features, 13 ms at 1,000 and
        110 ms at 10,000 over 500 rows, and a row after it costs what it did.
    ``shards``
        Split the pair work across the bank's threads: a count of ranges of
        features, or ``"auto"`` for as many as the width keeps busy. The
        threads are the bank's pool, which the bank's ``threads`` sizes;
        ``shards`` cuts one spec's work so that more of the pool can take
        it, and never adds a thread. The pool
        runs a bank's groups and specs in parallel already, so one wide spec on
        one group is one thread's work while the rest wait. With ``shards`` its
        pairs are cut into ranges of features, and each range runs on a thread
        of its own, a batch of rows at a time. Every number is the same to the
        bit whatever the count, so it is a setting and not part of the state. A
        saved bank resumes under the count of the specs given to ``load``, or
        under the saved one when given none. ``"auto"`` sizes itself to the
        machine it runs on. It estimates a batch's pair work from the width,
        the lags and the bins, and splits the batch into as many ranges as hold
        a tenth of a millisecond each, up to twice the pool's threads. The
        moments alone of fewer than about 1,300 pairs stay whole. By default
        the pairs run row by row on the group's thread. Through the bank at
        10,000 features on 14 threads, ``"auto"`` ran 1.2 times as fast at one
        target and 4.9 times with nine targets, lags and bins; the bank's own
        work on each row does not split (``docs/PERFORMANCE.md`` §25).
    ``window_size``, ``closed``, ``snapshot_every``, ``window_budget``
        A hard cutoff on the history the pairs are computed from, in
        clock units, as for :func:`ewridge`: a row ``window_size`` old or older
        contributes nothing (``closed="both"`` keeps the one exactly that old,
        as for :func:`ewridge`), and inside the window the weights are still
        exponential. Every moment a pair is built from is truncated (the
        weight, both means and the three centred second moments), so ``corr``,
        ``beta`` and ``t_stat`` describe the window and nothing else. That
        matters most for a screen: two regimes of opposite sign average to
        nothing over a long history. ``snapshot_every`` spaces the snapshots on
        the clock and ``max_rows_between_snapshots`` caps the rows between
        them, whichever comes first, as for :func:`ewridge`; ``window_budget``
        bounds each ring in MiB. The ``weight_sum`` the struct writes and the
        one the table reports are the weight inside the window. ``lags`` under
        a window take ``window_lags=True``, below. ``bins`` and ``window_size``
        are refused together (a snapshot of the histogram is ``bins`` times the
        size of one).
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
        and each pair only its covariance. With ``a`` and ``b`` the row's mix
        over the model's weight ``W``, taken on every learned row, and ``a_t``,
        ``b_t`` the target's:

        .. code-block:: text

            dx_j       = x_j - m_x[j]                    the shared mean before the row
            S'_xx[j]   = a * S_xx[j] + a * b * dx_j**2   m_x[j] += b * dx_j
            S'_xy[t,j] = a_t * S_xy[t,j] + a_t * b_t * dx_j * (y_t - m_y)    on t's rows

        Where every target is on every learned row the two report the same
        numbers, to the bit. Where a target is absent on some rows,
        ``"shared"`` is a different estimator. ``mean_x`` and ``var_x`` are the
        feature's over every learned row, and ``cov`` is the target's rows
        centred on that mean, so ``corr``, ``beta`` and ``t_stat`` move with
        it. That is sound where the absence says nothing about the feature.
        With a tenth of a target's rows absent at a half-life of 69 rows,
        measured, ``corr`` moved by at most 0.0065 and ``var_x`` by 2.5% at the
        median. At 20,000 pairs it runs 2.7 times as fast at ten targets and
        3.2 times at thirty, the same at one, and a ten-target state is under
        half the size (docs/PERFORMANCE.md §27). With ``lags`` it keeps the
        feature's autocovariance per feature too, and at ten targets runs 4.6
        times as fast with no cross lags. Refused with a ``window_size``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    rest.

    ``coef_every`` and ``max_rows_between_coefs`` are refused, as there are no
    coefficients, and ``fit_intercept`` has nothing to act on. Nothing
    residual-based applies (there is no prediction), so the residual switches
    are refused by name. A column may not be both a target and a feature.
    ``min_weight`` defaults to 3: two rows give a correlation of ±1 whatever
    the data, three the first one with content. Below a target's
    ``min_weight`` its pairs' ``corr``, ``beta``, ``t_stat``, ``t_stat_serial``,
    lag correlations, ``phi_x`` and ``phi_y`` and split are null, and so are
    every pair's while the stream is less settled than ``min_settled_frac``;
    the moments and ``n_serial`` are reported either way.

    .. rubric:: Output

    One struct column named after the spec holding ``weight_sum`` alone
    (`docs/OUTPUTS.md#marginal
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#marginal>`_).
    The pairs are read from the state with
    :meth:`polars_online.ModelBank.marginal`, one row per (group, instance,
    feature, target) with the moments, ``corr``, ``beta``, ``t_stat``,
    ``n_kish``, and the lag and bin columns above. That method documents every
    column. A group that closes writes the same pairs to
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

    - ``bin_edges`` that miss or add a feature, that are not finite and
      increasing, or that give no feature an edge (every feature one open
      bin, with no response curve and no cut; one feature without edges
      beside others is taken, and reports a single bin);
    - ``bins`` beside ``bin_edges``, and ``bins`` or ``bin_edges`` beside
      ``window_size``;
    - ``bins`` below 2, or a ``bin_warm_rows`` below ``bins``;
    - an empty ``lags`` (left out, there are none), ``serial_rule`` or
      ``cross_lags`` without ``lags``, ``lags`` or ``cross_lags`` that do not
      increase, and a cross lag not in ``lags``;
    - ``lags`` beside ``window_size`` without ``window_lags``;
    - ``bin_budget`` without bins or not above 0, and bins past ``bin_budget``;
    - ``bin_stat`` without bins, and ``bin_huber_delta`` under ``"mean"`` or
      not above 0;
    - ``bin_refit_rows`` or ``refit_budget`` under ``"mean"`` or
      ``bin_huber_delta=inf``, and a ``refit_budget`` not above 0;
    - ``shards`` below 1, or a string other than ``"auto"``;
    - a ``serial_rule``, ``bin_rule``, ``bin_stat`` or ``feature_moments`` not
      in the list.
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
        "bin_stat": None if bin_stat == "mean" else bin_stat,
        "bin_huber_delta": bin_huber_delta,
        "bin_refit_rows": bin_refit_rows,
        "refit_budget": refit_budget,
        "shards": shards,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
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
    dynamics: Literal["ew", "linear"] = "ew",
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
    accumulator, ``zeta_i = (x_i - m_i) / sqrt(v_i)``. With ``S1 = sum(zeta)``
    and ``S2 = sum(zeta * zeta)`` over the ``n`` features the row's estimate
    is their Lemma 2.3:

    .. code-block:: text

        u = (S1**2 - S2) / ((n - 1) * S2)      = mean_{i != j} zeta_i zeta_j / mean_i zeta_i**2

    which lies in ``[-1 / (n - 1), 1]``, both ends reached. Between two blocks
    ``A`` and ``B`` the estimate is ``u_AB = S1_A S1_B / sqrt(n_A n_B S2_A
    S2_B)``. The level then follows one of two dynamics, on the model's own
    clock with decay factor ``lam`` and row weight ``w``:

    .. code-block:: text

        "ew":      W' = lam * W + w,  a = lam * W / W',  b = w / W'
                   rho' = a * rho + b * u                     (rho = u at W = 0)
        "linear":  rho' = (1 - alpha - beta) * rho_bar' + alpha * u + beta * rho
                   rho' = u                                    (on the first row)
                   rho' = rho                                  (at w = 0)

    where ``rho_bar`` is the ``"ew"`` recursion run alongside as the target of
    the linear one. The ``"ew"`` dynamics run on the clock. The linear one
    steps once per row, as a DCC model's does: across a gap capped at
    ``gap_cap`` its own terms take one row's step, as across a millisecond, and
    only ``rho_bar`` moves on the clock. A row of weight 0 moves neither: it
    advances the clock and learns nothing, as everywhere. Otherwise the linear
    recursion has no row weights, as the paper's has none: a positive weight
    reaches ``rho`` only through ``rho_bar``, so rows of weight 0.5 and 2 move
    it by the same ``alpha * u``.

    Two departures from the paper, on purpose. Its eq. 21 has a free
    intercept and applies correlation targeting to the DCC ``Q`` recursion,
    not to the linear one. Writing the intercept as ``(1 - alpha - beta) *
    rho_bar`` is this library's reparameterisation, chosen because a
    streaming model has no sample to fit a free intercept on. And the paper
    permits ``alpha + beta`` slightly above 1 under numerical bounds, where
    this refuses it. The paper also notes that ``u`` is a downward biased
    estimate of the equicorrelation (``E[u]`` is about 0.20 for a true 0.30
    at ``m = 6``). It offers an alternative, ``1 - (1 / (n - 1)) *
    sum((zeta_i - zeta_bar)**2)``, which this does not compute. Use ``u`` as a signal that
    moves with the market's correlation, not as the correlation. ``rho`` is
    not an ``ew_cov``'s ``corr`` over the columns, since the mean of a ratio
    is not the ratio of means.

    A column whose variance is exactly zero -- constant from its first row, the
    only way an EW variance is exactly zero -- has no standardised value. It is
    left out of its block's sums, so ``n`` counts the columns that have one.
    Its block reads the correlation among its other columns, and so does a pair
    it is in; a block left with fewer than two has no ``u`` on the row, nor a
    pair with an empty side. Nor does a row whose ``S2`` is 0. Each value keeps
    its own weight ``W``: one with no ``u`` on a row learns nothing and does
    not decay, while the others learn. ``loglik`` needs every column, and is
    null on such a row.

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
    rest. ``fit_intercept`` has nothing to act on, as there is no regression.

    ``half_life`` or ``lam`` is required: both the standardiser and the level
    decay on it.

    .. rubric:: Output

    One struct column named after the spec, all but ``coef`` read from the
    state before the row, so they are safe as features for that same row
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
        How far the decay window had filled before the row, null where nothing
        decays, and which gate withheld the row's fields, null where none did:
        a warm-up of the model's own carries no reason.
    ``coef``
        The levels after the row, in the order of the ``rho`` fields: ``coef``
        on a row is ``rho`` on the next.

    With ``K`` named blocks the first two become ``u_<A>`` per block then
    ``u_<A>_<B>`` per pair, and ``rho_*`` likewise, with one ``loglik`` over
    all of them. ``u`` is null on a row where fewer than two of its block's
    columns have a positive pre-row variance, or where ``S2`` is 0, and
    ``loglik`` on a row where any column has none. ``loglik`` is null too
    where the levels hold no correlation matrix, or hold one too close to
    singular for its density to keep a digit: each level can be inside
    ``(-1, 1)`` while the blocks' levels together are not. Then the factor of
    the ``K x K`` system the density solves needs a shift on its diagonal, or
    its condition passes the point where the solve's forward-error bound
    reaches 1. Such levels come from the first rows of a warm-up, when they
    rest on one or two rows. One block never reaches that point.

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
# The changepoint detectors', the regimes', the realised covariance's, the
# robust regressions', `sgd`'s and the trend's builders live in files of their own
# (the 250 KB cap); read back here,
# after everything they use, for `__all__`, the key tables below and
# `polars_online.spec`.
from polars_online._spec_audit import audit  # noqa: E402
from polars_online._spec_changepoint import bocpd, corrchange  # noqa: E402
from polars_online._spec_gradient import sgd  # noqa: E402
from polars_online._spec_regimes import hmm, rcov  # noqa: E402
from polars_online._spec_robust import huber, quantile  # noqa: E402
from polars_online._spec_trend import holt  # noqa: E402

UNSUPERVISED = frozenset(
    {"ew_cov", "kmeans", "micro", "deco", "rcov", "hmm", "corrchange", "bocpd", "audit"}
)

_NUMERIC_KEYS = _numeric_keys()
_CLOCK_KEYS = _clock_keys()
