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
from collections.abc import Callable
from datetime import timedelta
from typing import Any, Unpack

import polars as pl

from polars_online._duration import Duration, duration_text
from polars_online._kwargs import CommonKwargs
from polars_online._polars_online import (
    spec_coef_fields,
    spec_output_fields,
    spec_output_index,
    validate_spec,
)


def _json(spec: dict[str, Any] | list[dict[str, Any]]) -> str:
    """JSON has no infinity literal, but ``halflife=inf`` is meaningful (it pins
    a coefficient), so infinities are encoded as strings the Rust side
    understands. A NaN is never meaningful in a spec and is refused here, by
    parameter name, rather than by the JSON offset serde would report. NumPy
    scalars are plain numbers here (``json`` alone refuses them)."""

    def enc(v: Any, key: str, who: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, numbers.Integral):
            return int(v)
        if isinstance(v, numbers.Real):
            v = float(v)
            if math.isnan(v):
                raise ValueError(f"spec {json.dumps(who)}: {key} must not be NaN")
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return v
        if isinstance(v, dict):
            return {k: enc(x, k, who) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x, key, who) for x in v]
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
    ``specs`` compare equal to the dicts that built it. Only the numeric
    parameters are touched -- a feature column may be called ``"inf"``."""

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
# them, so a wrong shape is reported by parameter name ("halflife must be a
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
    if origin is list:
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
    return True  # an annotation this does not read; the Rust side still checks


def _describe(hint: Any, plural: bool = False) -> str:
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
_INF_OK: dict[str, frozenset[str]] = {
    "*": frozenset(
        {
            "halflife",
            "min_periods",
            "max_dclock",
            "session_gap",
            "average_eta",
            # The noise gate at `inf` is off: no ratio is above it.
            "max_error_inflation",
        }
    ),
    "ewridge": frozenset({"long_halflife"}),
    "lasso": frozenset({"select_halflife"}),
    "kalman": frozenset({"coef_halflife", "revert_halflife"}),
    "huber": frozenset({"huber_delta"}),
    "sgd": frozenset({"clip_gradient", "coef_min", "coef_max", "huber_delta"}),
    "pa": frozenset({"c", "coef_min", "coef_max"}),
    "holt": frozenset({"level_halflife", "trend_halflife"}),
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
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return not math.isinf(value)
    return True


#: The int parameters whose Rust floor is 1, not 0 (review 2026-09-12, D8).
_AT_LEAST_ONE = frozenset(
    {
        "window_every",
        "pca_every",
        "update_every",
        "split_merge_every",
        "prune_every",
        "max_clusters",
        "resid_autocorr_lag",
        "k",
    }
)


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
        for key, value in kwargs.items():
            if key == "name":
                continue
            hint = own.get(key, shared.get(key))
            if hint is None:
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
            if key not in inf_ok and not _finite(value):
                raise ValueError(f"{who}: {key} must be finite, got {_got(value)}")
        # A clock parameter's duration, however it was written, is kept as
        # the text a TOML config writes and a state file stores (task 88).
        written = {
            key: duration_text(value, who, key) if key in clock else value
            for key, value in kwargs.items()
        }
        return typing.cast(Callable[..., R], fn)(*args, **written)

    return wrapper


__all__ = [
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


def _common(
    name: str,
    model: dict[str, Any],
    *,
    targets: list[str],
    features: list[str],
    add_intercept: bool = True,
    clock: str | None = None,
    halflife: float | Duration | list[float | Duration] | None = None,
    lam: float | None = None,
    max_dclock: float | Duration | None = None,
    on_clock_reset: str = "max",
    min_backwards_jump: float | Duration | None = None,
    session: str | None = None,
    session_gap: float | Duration | None = None,
    weight: str | None = None,
    min_periods: float | list[float] | None = None,
    min_settled_frac: float | None = None,
    max_error_inflation: float | None = None,
    emit_error_inflation: bool = False,
    coef_every: int = 0,
    emit_sigma: bool = False,
    emit_resid_z: bool = False,
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
    drift_threshold: float | None = None,
    drift_action: str = "flag",
    label_delay: float | Duration | None = None,
    group: str | None = None,
    group_close: str | None = None,
) -> dict[str, Any]:
    spec = {
        "name": name,
        "model": model,
        "targets": targets,
        "features": features,
        "add_intercept": add_intercept,
        "clock": clock,
        "halflife": halflife,
        "lam": lam,
        "max_dclock": max_dclock,
        "on_clock_reset": on_clock_reset,
        "min_backwards_jump": min_backwards_jump,
        "session": session,
        "session_gap": session_gap,
        "weight": weight,
        "min_periods": min_periods,
        "min_settled_frac": min_settled_frac,
        "max_error_inflation": max_error_inflation,
        "emit_error_inflation": emit_error_inflation,
        "coef_every": coef_every,
        "emit_sigma": emit_sigma,
        "emit_resid_z": emit_resid_z,
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
        "label_delay": label_delay,
        "group": group,
        "group_close": group_close,
    }
    validate_spec(_json(spec))
    return spec


@_checked
def ewridge(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | list[float] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    standardize: bool = False,
    ridge_decay: bool = False,
    coef_prior: list[list[float]] | None = None,
    session_shrink: float | None = None,
    long_halflife: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    gram_block_rows: int | None = None,
    target_gaps: str = "own_rows",
    window: float | Duration | None = None,
    window_every: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted ridge regression on running sums -- the workhorse.

    The model keeps the exponentially weighted means of ``z z'`` and ``z y``, with
    ``z`` the features and the intercept in front, and solves the ridge normal
    equations from them on a schedule. The sums are the whole state, so several
    ridge values, feature subsets and halflives are fitted from the same sums at
    almost no extra cost, and the sums can be read back
    (:meth:`polars_online.ModelBank.gram`), pooled and solved offline
    (:mod:`polars_online.gram`). Reach for it first. The other regressions here
    each cover a case it does not: coefficients that drift (:func:`kalman`),
    outliers (:func:`huber`), a quantile (:func:`quantile`), a sparse fit
    (:func:`lasso`), a row cost of O(k) (:func:`sgd`, :func:`pa`).

    .. rubric:: The fit

    Per row, with ``w`` the row's weight, ``lam`` its decay ``0.5 ** (d_clock /
    halflife)`` and ``W_j`` the weight behind target ``j`` over the rows that
    target is present on:

    .. code-block:: text

        W_j' = lam * W_j + w
        S_j' = (lam * W_j * S_j + w * z z') / W_j'
        r_j' = (lam * W_j * r_j + w * z * y_j) / W_j'
        on the schedule:  (S_j + ridge * D) beta_j = r_j      D = I with a 0 in the intercept slot

    The sums are means, not totals, so they stay bounded over a stream of any
    length, and the second moments are kept centred (a weighted Welford update),
    so a feature far from zero costs no precision. The solve is a Cholesky
    factorization: a near-singular system is retried with a small diagonal jitter,
    and a solve that needed one is counted in
    :meth:`polars_online.ModelBank.solve_failures`. A prediction uses the
    coefficients of the last solve and the state before the row.

    .. rubric:: Parameters

    ``ridge``
        The penalty on the slopes, never on the intercept, in the features'
        squared units unless ``standardize``. Default ``1e-6``. A list fits one
        instance per value from the same sums, reported side by side as
        ``pred_<t>__r<ridge>``.
    ``feature_sets``
        Named subsets of ``features``, each a fit of its own from the same sums,
        reported as ``pred_<t>__<set>``. The full set is fitted only when it is
        one of them; ``emit_selected`` then reports the set doing best.
    ``standardize``
        Solve in correlation form and unscale afterwards, so ``ridge`` means the
        same thing whatever the features' units and a feature whose variance is
        zero is dropped from the solve rather than blowing it up. Default
        ``False``. Without an intercept nothing is centred: the system is scaled
        by each column's root mean square instead, a fit through the origin is
        least squares through the origin, and the column dropped is one that is
        all zero. ``lasso``, ``kalman``, ``huber``, ``quantile`` and ``sgd``
        standardize the same way.
    ``ridge_decay``
        Whether the ridge fades. ``S`` is a weighted mean, so a plain ``ridge`` is
        a fixed per-observation penalty whose pull is permanent -- "always stay
        near this belief". With ``ridge_decay`` the prior sits on the decaying sum
        scale and fades as data arrives -- the usual warm start, "begin at
        yesterday's fit and let evidence take over". Default ``False``.
    ``coef_prior``
        Shrink toward these coefficients instead of toward zero: one vector per
        target, in the features' original units, ``len(features) + 1`` long when
        there is an intercept.
    ``session_shrink``, ``long_halflife``
        A middle way between a session's decay and a full reset. A second
        accumulator follows the long-run relationship at ``long_halflife``
        (``inf``: the whole history), and on a session boundary the two are mixed
        by weight:

        .. code-block:: text

            W' = (1 - f) * W_fast + f * W_slow
            S' = ((1 - f) * W_fast * S_fast + f * W_slow * S_slow) / W'

        so ``0`` keeps today's fit, ``1`` reverts to the long run, and anything
        between drifts partway back overnight. Unlike ``session_gap`` this changes
        what the model believes, not just how confident it is. With
        ``ridge_decay`` the prior's scale mixes the same way, so ``session_shrink
        = 1`` lands exactly on the twin's fit.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule: every ``solve_every`` clock units, and at least every
        ``max_rows_between_solves`` rows. ``solve_every`` defaults to ``halflife /
        50``, which is every row for ``halflife = inf`` and for ``lam``; with a
        large finite halflife set it, or the default never comes due.
        ``max_rows_between_solves`` is off by default. The coefficients are the
        sums' as of the last solve.
    ``gram_block_rows``
        Hold that many rows back and bring the ``k x k`` matrix up to date once
        per block, by one matrix product instead of one rank-one update per row:
        ``256`` measured 6.6x faster at a thousand features. The matrix is brought
        up to date before every solve too, so the block never exceeds the solve
        cadence; the option is refused where there is none (``solve_every <= 0``
        or ``max_rows_between_solves <= 1``) and with ``window``. ``n_eff``, the
        timing of every prediction and chunk invariance are unchanged to the bit;
        the blocked sum is the same sum in another order, so a blocked fit agrees
        with an unblocked one to rounding. The held rows travel in the state file,
        and are refused over 256 MiB.
    ``target_gaps``
        Which rows a target's fit is read from where the target is null on some.
        Target ``j`` keeps its mean ``ybar_j``, the column means ``m_j`` and the
        centred cross-moments ``c_j = EW[(x - m_j)(y_j - ybar_j)]`` over the rows
        it is present on. Its slopes solve ``(C + ridge * I) b = c_j``, with ``b_0
        = ybar_j - m_j . b``. The option is which rows the feature covariance
        ``C`` is taken over. ``"own_rows"``, the default, takes the target's own,
        so its fit is the fit of the frame with its null rows dropped. Targets
        present on the same rows share one ``C``. One that goes missing where the
        others are present takes a copy and keeps its own from then on, so a bank
        of targets costs a ``k x k`` matrix per pattern of missing rows, and a
        single target nothing. ``"pairwise"`` takes every row, the way pandas'
        ``DataFrame.cov`` takes a pairwise-complete covariance: one matrix
        whatever the gaps. It is exact when the gaps have nothing to do with the
        features; where they do, each slope is scaled by the ratio of the
        feature's variance on the target's rows to its variance on all of them.
        ``n_eff`` counts every row either way, and a null target is still
        predicted.
    ``window``, ``window_every``, ``window_budget``
        A hard cutoff on the history the fit is solved from, in clock units: a row
        older than ``window`` is not in the sums at all, where the exponential
        weight alone would leave ``0.5 ** (age / halflife)`` of it. Inside the
        window the weights are still exponential, so this is a windowed
        exponentially weighted regression, not a rolling least squares. It is
        exact: the sums are sums of per-row contributions, so everything at or
        before a time ``u`` is ``lam ** (t - u)`` times the sums as they stood
        then, and subtracting that leaves the window. The model keeps a ring of
        snapshots to do it, which is the one place here where memory grows with a
        window rather than with the state; a halflife grid is one instance per
        entry, each with its own ring. ``window_every`` snapshots every ``n`` rows
        instead, which divides the memory and can only shorten the effective
        window.

        ``window_budget`` bounds each ring in MiB, and says what happens when a
        ring reaches the bound. ``{"thin": mib}`` drops every other snapshot and
        doubles the spacing between the rest, as often as it takes; like
        ``window_every``, that can only shorten the window. ``{"refuse": mib}``
        keeps the ring at the bound and refuses the chunk, naming the ring's size
        and ``window_every``. The budget is checked as the rows are learned, so a
        refused chunk has been partly learned. The bank then refuses every later
        ``fit_predict``, ``predict`` and ``save``; rebuild it from its last save
        (:meth:`polars_online.ModelBank.fit_predict`). Unset, a window refuses
        past 256 MiB per ring; ``{"refuse": float("inf")}`` is no bound.

        ``n_eff``, ``sigma`` and ``resid_z`` are the window's too, so the spread
        describes the rows the fit describes; the spread keeps a ring of its own
        for it, a pair of floats a slot, bounded with the fit's. Everything that
        reads the spread is the window's with it: drift's scale, the conformal
        band, and the ranking ``emit_selected`` and ``emit_averaged`` take. The
        coefficients are the window's as of the last solve, so a coarse
        ``solve_every`` reports a window that has since moved on. ``window`` is
        refused with ``ridge_decay`` (the decaying prior's scale is the product of
        every decay the stream applied, which a window truncates the data of but
        not the prior) and with ``session_shrink`` (the slow twin is a second
        accumulator under a longer halflife).

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#ewridge
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ewridge>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per (target, ridge value, feature set) slot in the order the ``pred``
        fields declare them, the intercept then one entry per feature, zero for a
        feature outside the slot's set. :func:`coef_index` maps each position to
        its term, and :func:`coef_fields` names the column each becomes when the
        struct is unnested.

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. Under a grid each field is suffixed per instance. The sums
    behind the fit are :meth:`polars_online.ModelBank.gram`'s, one entry per Gram
    (under ``target_gaps = "own_rows"``, one per set of targets present on the
    same rows), and a group that closes writes one row per Gram to
    :meth:`polars_online.ModelBank.closed_groups`.

    .. rubric:: Example

    .. code-block:: python

        rr = po.spec.ewridge(
            "rr", targets=["y"], features=["x0", "x1", "x2"],
            clock="t", max_dclock=300.0, halflife=600.0,
            ridge=[1e-6, 0.1],                # one fit per value, from the same sums
            feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # subsets, likewise
            standardize=True,
            emit_selected=True,               # which (ridge, set) is doing best
        )
        out = po.ModelBank([rr]).fit_predict(df)
        fields = po.spec.output_index(rr)     # every field, with what its name encodes
        best = out["rr"].struct.field("pred_y__selected")

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` naming
    the problem for a ``feature_sets`` entry naming a column not in ``features``,
    a ``coef_prior`` vector of the wrong length, and ``session_shrink`` without
    ``long_halflife``.
    """
    model: dict[str, Any] = {
        "type": "ew_ridge",
        "ridge": ridge,
        "feature_sets": [[k, list(v)] for k, v in feature_sets.items()] if feature_sets else None,
        "standardize": standardize,
        "ridge_decay": ridge_decay,
        "coef_prior": coef_prior,
        "session_shrink": session_shrink,
        "long_halflife": long_halflife,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "gram_block_rows": gram_block_rows,
        "target_gaps": target_gaps,
        "window": window,
        "window_every": window_every,
        "window_budget": window_budget,
    }
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


def output_fields(spec: dict[str, Any]) -> list[str]:
    """The names of the struct fields a spec writes, in order.

    The same list :meth:`polars_online.ModelBank.output_fields` gives for a whole
    bank, for one spec and before any row is fed: the fields are fixed by the
    spec. ``ValueError`` for a spec that is not valid, with the builders' message
    (:mod:`polars_online.spec`); a dict that is not a spec at all is told what it
    lacks (``invalid spec: missing field `name```).

    .. code-block:: python

        fields = po.spec.output_fields(spec)   # ['pred_y__r0.000001', ..., 'n_eff', 'coef']

    """
    return spec_output_fields(_json(spec))


def output_index(spec: dict[str, Any]) -> pl.DataFrame:
    """Every struct field a spec writes, with the machine values its name encodes.

    One row per field, in the struct's order:

    ``field``
        The name, as it appears in the struct.
    ``kind``
        What it is: ``pred``, ``resid``, ``sigma``, ``n_eff``, ``coef``,
        ``lam_selected``, ``selected``, a statistic's stem, and so on.
    ``target``
        The target the field is about, or null.
    ``halflife``, ``lam``
        The decay instance it belongs to, or null for a single one.
    ``ridge``, ``feature_set``, ``lambda``
        The grid combination: the ridge value, the feature set's name, the lasso
        path point; null where the spec has no such grid.
    ``quantile``
        The level of a ``resid_quantiles`` or ``mahal_quantiles`` field.
    ``columns``
        The pair an ``ew_cov`` statistic is over.
    ``dtype``
        ``f64``, ``bool``, ``str`` or ``list[f64]``: the type the bank declares to
        polars before the first row is read.

    This is how to reach a field without constructing its name; the string grammar
    (:mod:`polars_online.spec`, "What a spec writes") stays an implementation
    detail:

    .. code-block:: python

        idx = po.spec.output_index(grid)
        name = idx.filter(
            (pl.col("kind") == "pred")
            & (pl.col("target") == "y")
            & (pl.col("ridge") == 0.5)
            & (pl.col("halflife") == 500.0)
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
            "halflife": pl.Float64,
            "lam": pl.Float64,
            "ridge": pl.Float64,
            "feature_set": pl.String,
            "lambda": pl.Float64,
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
        halflife instance.
    ``position``
        Its index in that list.
    ``name``
        The column :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest`
        gives it: ``coef_{target}_{term}{combo}{instance}``, so
        ``coef_y_x1__r0.5@h500`` sits beside ``pred_y__r0.5@h500``. The ``coef_``
        prefix and the target are there because a bare ``x1`` would collide with
        the feature column of that name in the same frame.
    ``target``, ``halflife``, ``lam``, ``ridge``, ``feature_set``, ``lambda``
        As :func:`output_index` reports them.
    ``term``
        ``"intercept"``, a feature name, or ``"level"`` / ``"trend"`` for
        :func:`holt`.

    Empty for the kinds that report no coefficients: ``ew_cov``, ``seqtest``,
    ``marginal``, ``rcov``, ``corrchange`` and ``bocpd``. To reach one coefficient
    in the nested output without writing either name:

    .. code-block:: python

        cf = po.spec.coef_fields(grid)
        row = cf.filter(
            (pl.col("target") == "y") & (pl.col("term") == "x1") & (pl.col("halflife") == 500.0)
        ).row(0, named=True)
        slope = out["m"].struct.field(row["field"]).list.get(row["position"])

    Rendered by the Rust code that names the fields, from the slot order the
    models lay the list out in: the intercept first when the spec has one, then
    every feature -- zero for a feature outside a feature set -- per (target,
    combination) slot, slots in the order the ``pred`` fields declare them.
    ``ValueError`` for a spec that is not valid, as for :func:`output_fields`.
    """
    rows = json.loads(spec_coef_fields(_json(spec)))
    return pl.DataFrame(
        rows,
        schema={
            "field": pl.String,
            "position": pl.UInt32,
            "name": pl.String,
            "target": pl.String,
            "halflife": pl.Float64,
            "lam": pl.Float64,
            "ridge": pl.Float64,
            "feature_set": pl.String,
            "lambda": pl.Float64,
            "term": pl.String,
        },
    )


def coef_index(spec: dict[str, Any]) -> pl.DataFrame:
    """The layout of each ``coef`` list, one row per position.

    ``coef`` is flat: (target x grid combination) slots, each contributing its
    terms in order. This maps ``position`` to ``target``, the combination's
    ``ridge``, ``feature_set`` and ``lambda``, and ``term`` -- ``"intercept"``, a
    feature name, or ``"level"`` / ``"trend"`` for :func:`holt`. For
    :func:`kmeans` the slots are the centres, so ``target`` reads ``"cluster0"``,
    ``"cluster1"``, ... and ``term`` is the feature whose coordinate the position
    holds; for :func:`ew_class` and :func:`hmm` a class or state likewise.

    .. code-block:: python

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
    return cf.filter(pl.col("field") == first).select(
        pl.col("position").cast(pl.Int64), "target", "ridge", "feature_set", "lambda", "term"
    )


@_checked
def rls(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    ridge: float | None = None,
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
        A_0 = ridge * I                b_0 = ridge * coef_prior

    What is stored is the factor ``R`` of ``A = R'R`` and ``u_j = R^-T b_j``: a
    row is folded in by Givens rotations and ``beta`` read off by one
    back-substitution (the square-root form). The textbook recursion on the
    inverse ``P`` loses symmetry to rounding by ``1 / lam`` a row, and one extreme
    row can cancel it and freeze a coefficient for good; this form has neither
    failure. The result is :func:`ewridge` with ``ridge_decay`` solved on every
    row, to better than 1e-9.

    .. rubric:: Parameters

    ``ridge``
        The prior's strength: ``A_0 = ridge * I``, which is ``P_0 = I / ridge``.
        Default 1.0. Unlike :func:`ewridge` it penalizes the intercept too, and it
        fades as data arrives, since ``A`` is a decayed sum.
    ``coef_prior``
        The coefficients the fit starts from and is shrunk toward, one vector per
        target in the features' original units; zeros when left out.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#rls
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rls>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. The factor ``R`` is shared between the targets, so a row with
    any null target is scored and learned from for no target.

    .. rubric:: Example

    .. code-block:: python

        rls = po.spec.rls(
            "rls", targets=["y"], features=["x0", "x1"],
            clock="t", max_dclock=300.0, halflife=600.0,
            ridge=1e-3,    # A starts at ridge * I: this penalizes the intercept too
        )
        out = po.ModelBank([rls]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`).
    """
    model: dict[str, Any] = {"type": "rls", "ridge": ridge, "coef_prior": coef_prior}
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def lasso(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    lasso_path: list[float],
    l1_ratio: float | None = None,
    select_halflife: float | Duration | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    window: float | Duration | None = None,
    window_every: int | None = None,
    window_budget: dict[str, float] | None = None,
    max_cd_iters: int | None = None,
    cd_tol: float | None = None,
    target_gaps: str = "own_rows",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A lasso or elastic-net path on running sums, with the penalty chosen as the
    stream runs.

    Coordinate descent on the standardized, centred sums :func:`ewridge` keeps,
    warm-started from the previous solution both along the path of penalties and
    from one solve to the next. Every point of the path is predicted, so choosing
    among them costs nothing: ``lam_selected_<t>`` is the point with the lowest
    exponentially weighted out-of-sample squared error so far.

    .. rubric:: The fit

    For each penalty ``l`` in ``lasso_path``, with ``C`` the feature correlation
    matrix and ``c`` the feature-target correlations, until no coefficient moves
    by more than ``cd_tol``:

    .. code-block:: text

        rho_i = c_i - sum_{j != i} C_ij b_j
        b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))
        soft(v, t) = sign(v) * max(|v| - t, 0)

    then unscaled, with the intercept recovered as ``ybar - m . beta``. ``l1_ratio
    < 1`` is an elastic net.

    .. rubric:: Parameters

    ``lasso_path``
        The penalties, decreasing; required. One instance per point, reported as
        ``pred_<t>__l<lambda>``.
    ``l1_ratio``
        The share of the penalty that is L1. Default 1, the lasso; below 1 an
        elastic net.
    ``select_halflife``
        The halflife of the EW squared out-of-sample error each path point is
        ranked by. Default: the model's halflife; ``inf`` ranks on the plain mean
        over every row so far. ``lam_selected_<t>`` is reported as it stood before
        the row -- the point this row was scored with, not the one its own error
        then elected. A row of weight 0 adds no error and ages the errors so far,
        so the selection moves only by what the ageing forgets.
    ``solve_every``, ``max_rows_between_solves``
        The solve schedule, as for :func:`ewridge`.
    ``max_cd_iters``, ``cd_tol``
        Within a solve, the descent stops after ``max_cd_iters`` sweeps (default
        100) or when no coefficient moves by more than ``cd_tol`` (default
        ``1e-10``). A descent that runs out of sweeps first is counted in
        :meth:`polars_online.ModelBank.solve_failures`, one per target and path
        point.
    ``target_gaps``
        Which rows a target's feature correlations are taken over where the target
        is null on some: ``"own_rows"``, the default, or ``"pairwise"``, as for
        :func:`ewridge`. The cross-correlations are centred at the target's own
        means either way.
    ``window``, ``window_every``, ``window_budget``
        A hard cutoff on the history the path is fitted from, in clock units, as
        for :func:`ewridge`: a row older than ``window`` is not in the sums,
        ``window_every`` is the snapshot cadence, and ``window_budget`` bounds
        each ring in MiB and thins or refuses past the bound. The selection error
        is truncated with the sums, so the ``lambda`` chosen is the one that fits
        the window rather than rows the fit has dropped. So a window can change
        the support, not just the coefficients: a feature with no evidence inside
        it goes to exactly zero. ``n_eff``, ``sigma`` and ``resid_z`` are the
        window's too.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#lasso
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#lasso>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per (target, path point) slot, the intercept then one entry per feature
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. The ``pred`` and ``resid`` fields are per target and path
    point, ``pred_<t>__l<lambda>``, and ``lam_selected_<t>`` is the path point in
    force for the target, by lowest EW out-of-sample error. The sums behind the
    path are :meth:`polars_online.ModelBank.gram`'s, as for :func:`ewridge`.

    .. rubric:: Example

    .. code-block:: python

        las = po.spec.lasso(
            "las", targets=["y"], features=["x0", "x1", "x2"],
            clock="t", max_dclock=300.0, halflife=600.0,
            lasso_path=[0.1, 0.01, 0.001],   # decreasing; every point is predicted
            l1_ratio=1.0,                    # below 1: an elastic net
        )
        out = po.ModelBank([las]).fit_predict(df)
        chosen = out["las"].struct.field("lam_selected_y")   # the point in force, per row

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``lasso_path`` is required.
    """
    model: dict[str, Any] = {
        "type": "lasso",
        "lasso_path": lasso_path,
        "l1_ratio": l1_ratio,
        "select_halflife": select_halflife,
        "solve_every": solve_every,
        "max_rows_between_solves": max_rows_between_solves,
        "max_cd_iters": max_cd_iters,
        "cd_tol": cd_tol,
        "target_gaps": target_gaps,
        "window": window,
        "window_every": window_every,
        "window_budget": window_budget,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def kalman(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    coef_halflife: float | Duration | list[float | Duration],
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float | None = None,
    share_p: bool = False,
    revert_halflife: float | Duration | list[float | Duration] | None = None,
    standardize: bool = True,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A regression whose coefficients are allowed to drift, tracked by a Kalman
    filter.

    Each target's coefficients are a state with a mean and a covariance. A row
    moves them by the Kalman gain, and between rows they drift as a random walk,
    or shrink toward zero under ``revert_halflife``. Use it where a relationship
    moves faster than a halflife can follow, or where a coefficient should be
    forgotten when nothing supports it.

    .. rubric:: The fit

    Per target, per row, with clock delta ``d`` and row weight ``w``:

    .. code-block:: text

        b_j <- Phi b_j                     Phi = diag(2 ** (-d / r_i))      the reversion
        P_j <- Phi P_j Phi + Q * d                                          the drift
        s    = z' P_j z + R_j / w
        k    = P_j z / s
        b_j <- b_j + k (y_j - z' b_j)
        P_j <- P_j - k z' P_j

    ``Q`` is the process noise and ``R_j`` the observation noise, the target's EW
    residual variance unless ``obs_var`` is given. That variance is per target,
    and its weight ages on every row, a row with no prediction or with weight 0
    included. Such a row, and a row whose target is null, ages the target's weight
    the same way and learns nothing for it.

    .. rubric:: Parameters

    ``coef_halflife``
        How fast a coefficient may drift, as a halflife on standardized features:
        the process noise is ``q_i = sigma^2 * (ln 2 / h_i) ** 2``, which matches
        EW-RLS's steady-state gain. A scalar, or one value per slot with the
        intercept first; ``inf`` pins that coefficient. Required. Not the spec's
        ``halflife``, which drives the standardization and the residual variance.
    ``q``
        The process noise given outright, one value per slot, which skips the
        derivation from ``coef_halflife``.
    ``obs_var``
        A fixed observation noise, in place of the EW residual variance.
    ``p0``
        The initial coefficient covariance, ``P_0 = p0 * I``. Default 1.0.
    ``standardize``
        Run the filter on standardized features, so ``coef_halflife`` and ``p0``
        mean the same thing whatever the columns' scale; the reported coefficients
        are in the original units either way. Default ``True``. Without an
        intercept the features are scaled by their root mean square and not
        centred, as for :func:`ewridge`. With ``standardize = False``, ``q = 0``
        and a fixed ``obs_var`` the filter is exactly Bayesian linear regression.
    ``revert_halflife``
        A reversion halflife ``r_i`` per slot: between observations the
        coefficient shrinks toward zero by ``2 ** (-d / r_i)``, so a coefficient
        no row has supported for a while is forgotten rather than carried. Default
        ``inf``, the random walk, which costs nothing. A scalar applies to every
        slot, the intercept included; a list gives one per slot, intercept first,
        and ``[inf, r, r]`` leaves the intercept a random walk. The pull is toward
        zero in the standardized coordinates when ``standardize`` is on: a slope
        toward "no effect", the intercept toward "the target averages zero". A
        reverting slot settles at the prior variance ``q_i * d / (1 - phi_i **
        2)``, a stationary AR(1) instead of an unbounded walk. A prediction
        propagates the state by the same ``Phi`` over the row's clock gap.
    ``share_p``
        Keep one ``P`` for every target, driven by the mean ``sigma^2`` across
        them, where by default ``P`` is per target because the recursion depends
        on ``sigma^2_j``. Default ``False``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#kalman
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kalman>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the original
        units (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` moves the coefficients
    by ``Phi`` over the clock distance from the last learned row, capped by
    ``max_dclock``, so a prediction far past the data is the intercept alone.

    .. rubric:: Example

    .. code-block:: python

        revert = po.spec.kalman(
            "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", max_dclock=10.0,
            halflife=200.0,          # the observation-noise estimate forgets at this rate
            coef_halflife=100.0,     # how fast a coefficient may drift
            revert_halflife=[float("inf"), 50.0, 50.0],   # the slopes shrink toward zero unobserved
        )
        out = po.ModelBank([revert]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``coef_halflife`` is
    required.
    """
    model: dict[str, Any] = {
        "type": "kalman",
        "coef_halflife": coef_halflife,
        "q": q,
        "obs_var": obs_var,
        "p0": p0,
        "share_p": share_p,
        "revert_halflife": revert_halflife,
        "standardize": standardize,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def huber(
    name: str,
    *,
    targets: list[str],
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
    says. Until a residual exists ``s`` is taken as 1, so the first rows are
    weighted in the residual's own units.

    .. rubric:: Parameters

    ``huber_delta``
        The cut, in units of ``s``: a residual within it counts at full weight.
        Default 1.5; ``inf`` cuts nothing, which is least squares.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``), correlation-form
        solving, and the solve schedule.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_periods`` counts the rows the target was present on, at their raw
    weights, decayed. The reweighted sum, which an outlier lowers, is not what it
    reads, so a stream whose warm-up meets outliers reports its first prediction
    when the rows say to.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#huber
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#huber>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them.

    .. rubric:: Example

    .. code-block:: python

        hub = po.spec.huber(
            "hub", targets=["y"], features=["x0", "x1"],
            clock="t", max_dclock=300.0, halflife=600.0,
            huber_delta=1.5,    # a residual beyond 1.5 sigma is down-weighted
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
    targets: list[str],
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
    target's effective sample ``n``, which is the smoothed-quantile bandwidth
    rate. Under a halflife the band's share of the sample is a few rows, and the
    floor is what keeps the step fed there; a long stream leaves the floor behind.
    Coverage at ``quantile = 0.9`` on normal noise reads 0.895 at ``halflife =
    30`` and 0.900 from 200 up (the second review of 2026-09-15, F3).

    Warm-up and rebuilding. Under three rows per coefficient of the rows the
    target was present on, the fit is ordinary least squares: a Newton step needs
    a Hessian, and a band around a fit built from a handful of rows is not one.
    The same rule rebuilds the fit after a gap or a reset has aged the weight
    away. A band holding under one row per coefficient takes least-squares rows
    too, until it holds rows again. That is what rebuilds a fit a row at the input
    bound leaves behind. Such a row sets the sums at its own scale, and every
    later row is outside the band. Only nudges arrive, each ``2h * psi * z`` over
    the band's weight; they cannot move the fit until that weight has decayed to
    nothing, and each is then a step that outgrows the band. From one row up an
    outside row's step lands inside the band, and the floor keeps a settled band
    well clear of one row, so the rule does not fire in steady state.

    .. rubric:: Parameters

    ``quantile``
        The level ``tau``, in ``(0, 1)``; required. 0.5 is a median regression.
    ``quantile_eps``
        The band's half-width in units of ``s``. Default 0.2, at which the band
        holds about a fifth of a stream at the median and a fifteenth at the 0.9
        quantile.
    ``ridge``, ``standardize``, ``solve_every``, ``max_rows_between_solves``
        As for :func:`ewridge`: the penalty (default ``1e-6``), correlation-form
        solving, and the solve schedule.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    ``min_periods`` counts the rows the target was present on, at their raw
    weights, decayed. The band's weight, which a halflife caps at the band's share
    of the sample, is not what it reads.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#quantile
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#quantile>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is the conditional quantile, and a ``resid`` is
    signed against it, so a share ``tau`` of them are positive once the fit has
    settled.

    .. rubric:: Example

    .. code-block:: python

        med = po.spec.quantile(
            "med", targets=["y"], features=["x0", "x1"],
            clock="t", max_dclock=300.0, halflife=600.0,
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
    targets: list[str],
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
    evidence, so the fit is sparse, and whose per-coordinate rates adapt to each
    feature's history. Its sums decay on the model's clock like every other
    model's.

    .. rubric:: The fit

    With ``z`` the row's feature vector, intercept included, and ``zz``, ``n`` and
    ``d`` the per-coordinate sums:

    .. code-block:: text

        n_i   <- lam * n_i ;  zz_i <- lam * zz_i ;  d_i <- lam * d_i
        b_i   = 0 if |zz_i| <= l1 else -(zz_i - sign(zz_i) l1) / (r_i + l2)
        r_i   = beta / alpha + d_i                 (under a halflife)
              = (beta + sqrt(n_i)) / alpha         (without one: river's closed form)
        p     = sigmoid(z . b)                     (z . b itself under loss="squared")
        g_i   = (p - y) * z_i * w
        s_i   = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
        zz_i += g_i - s_i * b_i ;  n_i += g_i^2 ;  d_i += s_i

    Under a halflife the penalties are constants on the sums' scale, so they act
    as a mean-scale ridge of ``(1 - lam) * (beta / alpha + l2)``, and a row with
    no target and a clock of ``t`` scales every coefficient by ``lam**t * (d + c)
    / (lam**t * d + c)``, ``c = beta / alpha + l2`` -- about 0.75 over one
    halflife at ``halflife = 100``, where :func:`ewridge`'s fit does not move. For
    forgetting without that shrinkage, reset a ``halflife = inf`` model on a
    ``session``, or use :func:`sgd` or :func:`ewridge`. Without a halflife the fit
    is river's ``FTRLProximal`` to the bit.

    .. rubric:: Parameters

    ``loss``
        ``"logistic"`` (the default) for a 0/1 target: ``pred`` is a probability.
        ``"squared"`` for a continuous target: ``pred`` is the linear prediction,
        and the model is a sparse linear regression with no solves and the L1
        support :func:`ewridge` has not got. The two differ only in the link; the
        gradient is ``(p - y) * z`` either way.
    ``alpha``, ``beta``
        The learning-rate scale and its smoothing. Defaults 0.1 and 1.0.
    ``l1``, ``l2``
        The penalties: ``l1`` zeroes a coefficient whose evidence is below it.
        Defaults 0.0 and 1.0.
    ``strict_binary``
        Refuse a chunk whose target is not 0 or 1, naming the row, before any
        stream is touched. Default ``False``: such a target is clamped into ``[0,
        1]``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#ftrl
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ftrl>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is a probability under ``"logistic"``, and
    ``emit_metrics`` then reads the fit as probabilities against labels: accuracy
    at 0.5, the Brier skill score and the point-biserial correlation, under their
    usual names.

    .. rubric:: Example

    .. code-block:: python

        click = po.spec.ftrl(
            "click", targets=["y"], features=["x0", "x1"], halflife=500.0,
            loss="squared",           # y here is continuous; "logistic" for a 0/1 target
            alpha=0.1, beta=1.0,      # the learning-rate scale and its smoothing
            l1=0.01, l2=0.0,          # l1 zeroes a coefficient whose evidence is below it
        )
        out = po.ModelBank([click]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``alpha``, ``beta``, ``l1``
    and ``l2`` refuse ``inf`` and ``NaN``.
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
    pca_every: int | None = None,
    lags: list[int] | None = None,
    window: float | Duration | None = None,
    window_every: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted moments of the feature columns: means, variances,
    correlations, and what is read off them.

    Not a regression: there are no targets and no coefficients, only running
    statistics of the columns named, decayed on the same clock as every model
    here. Values are read from the state before each row, so an ``ew_cov`` output
    can be a feature for that same row without leaking it.

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
    C_jj)`` are read off directly, and a variance stays accurate when the columns
    sit on a large offset, where the raw ``E[x^2] - m^2`` form loses it. One O(k²)
    update per row, which replaces the O(k²) passes a pure-Polars pairwise EW
    correlation needs.

    .. rubric:: Parameters

    ``stats``
        Which statistics to write, from ``mean``, ``var``, ``std``, ``cov``,
        ``corr``, ``partial_corr``, ``mahal`` and ``lagcorr``. Default ``["mean",
        "std", "corr"]``. ``[]`` writes nothing but ``n_eff`` and accumulates all
        the same. The spec's value is then its state, read back with
        :meth:`polars_online.ModelBank.gram` and
        :meth:`polars_online.ModelBank.describe`; that is the form for a wide set
        of columns, where even the means are k values per row nobody reads.
    ``precision_prior``
        A ridge on the co-moments, needed by ``partial_corr`` and ``mahal``: the
        precision matrix is ``(C + s * prior * I)^-1``, solved on each row it is
        read (O(k³), only when asked for), and like an RLS prior it fades as data
        accumulates.
    ``mahal_quantiles``
        Levels at which to keep a running quantile of the past ``mahal`` scores
        (``mahal_q<p>``, the P² algorithm): a threshold from the stream's own
        history instead of a table, so ``mahal > mahal_q0.99`` is one row in a
        hundred without assuming a distribution. The row's own score joins after
        it is read.
    ``pca``, ``pca_every``
        Track the top ``pca`` principal components of the covariance: per
        component ``j`` the fields ``pc<j>_var`` (its eigenvalue), ``pc<j>_share``
        (of the total variance), ``pc<j>_<feature>`` (its unit loading on each
        column, largest entry positive) and ``pc<j>_score`` (the row's coordinate
        ``v_j . (x - m)``). The eigendecomposition is refreshed every
        ``pca_every`` learned rows (default 1, O(k³) each) after the row is folded
        in; between refreshes the loadings are frozen, so a row's scores never
        depend on chunking, and each refresh keeps the previous sign, so a loading
        never flips.
    ``lags``
        Lagged cross-moments beside the contemporaneous ones. With ``W`` and ``m``
        the weight and mean before the row, and both deviations against that mean:

        .. code-block:: text

            C_l' = a * C_l + a * b * (x_t - m) (x_{t-l} - m)'

        the same ``a`` and ``b`` the co-moments use, so lag 0 would be
        ``comoments`` exactly. Lags are counted in learned rows within the group,
        not clock units, and must be strictly increasing and ``>= 1``; the list
        order is the output order. The ring of past rows is emptied on a session
        change and on a clock gap beyond ``max_dclock``, one row's or a run of
        skipped rows' whose total the ceiling cut: the events after which "the row
        ``l`` back" is not a row ``l`` ago. A zero-weight row ages the matrices
        and does not enter the ring. Add ``"lagcorr"`` to ``stats`` to write
        ``lagcorr_<a>_<b>_l<l>`` = ``C_l[a,b] / sqrt(C_0[a,a] * C_0[b,b])`` per
        lag and ordered pair, the auto terms included: ``k²`` slots a lag, since a
        lagged matrix is not symmetric. Or read ``lags`` and ``lag_comoments`` (an
        ``(L, k, k)`` array) from :meth:`polars_online.ModelBank.gram`.
    ``window``, ``window_every``, ``window_budget``
        A hard cutoff on the history, in clock units: a row older than ``window``
        contributes nothing at all, where the exponential weight alone would still
        leave ``0.5 ** (age / halflife)`` of it -- 12.5% at three halflives.
        Inside the window the weights are still exponential, so this is not a
        rolling flat mean: the newest row dominates exactly as it does without a
        window. It is exact, because an exponentially weighted sum contains its
        own past: everything at or before a time ``u`` is ``lam ** (t - u)`` times
        the accumulator as it stood at ``u``, so subtracting that leaves precisely
        the rest. What the model keeps is a ring of snapshots, one per learned
        row: the one place in this library where memory grows with a window rather
        than with the state. A snapshot is ``k² + k + 2`` doubles, so a 1,000-row
        window over 20 columns is about 3 MB per group. ``window_every`` snapshots
        every ``n`` rows instead and divides that by ``n``; ``window_budget``
        bounds each ring in MiB and thins or refuses past the bound, as for
        :func:`ewridge`.

        Four things to know before reading windowed numbers. The guarantee is
        one-sided: the boundary is the oldest snapshot still inside the window, so
        what is dropped is always a superset of what the window excludes. With
        ``window_every`` above 1 the effective window is shorter than asked by at
        most one snapshot's spacing, never longer. The clock is the decayed one:
        ``window`` is measured on the clock the decay uses, after ``max_dclock``
        caps a gap and after a ``session_gap`` is applied. The edge is a
        discontinuity: a row ageing out drops its whole weight at once, so a
        windowed series has small steps an EWMA does not. And it is a subtraction,
        so precision falls with the fraction discarded: negligible at ``window =
        3 * halflife`` (an eighth of the mass), worse as the window shortens
        toward the halflife.

        ``n_eff`` becomes the weight inside the window, which stops growing once
        the window fills, so ``min_periods`` gates on a quantity with a ceiling. A
        clock gap longer than ``window`` empties it and the row reports nulls
        rather than stale numbers. ``mahal``, ``partial_corr`` and the PCA read
        the window's moments too, and the PCA refresh is gated on the window's
        weight. ``window`` does not combine with ``lags`` or ``mahal_quantiles``,
        which accumulate over a history it does not truncate; both are refused by
        name.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, holding the statistics ``stats`` asks
    for, each named after its column or pair (``mean_x0``, ``std_x0``,
    ``corr_x0_x1``; pairs are unordered, ``i < j``, except ``lagcorr``'s),
    ``mahal`` and ``mahal_q<p>``, the ``pc<j>_*`` fields, and ``n_eff``; all null
    until ``min_periods``. The plain spec's fields are listed in
    `docs/OUTPUTS.md#ew_cov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_cov>`_.
    The moments themselves are :meth:`polars_online.ModelBank.gram`'s ``means``,
    ``comoments``, ``lags`` and ``lag_comoments``, and a group that closes writes
    them to :meth:`polars_online.ModelBank.closed_groups` with ``eig_vals`` and
    ``eig_vecs`` under ``pca``.

    .. rubric:: Example

    .. code-block:: python

        mv = po.spec.ew_cov(
            "mv", features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=500.0,
            stats=["mean", "std", "corr", "partial_corr", "mahal"],
            precision_prior=1e-6,        # needed by partial_corr and mahal
            mahal_quantiles=[0.99],      # mahal_q0.99: one row in a hundred, from the history
            pca=1, pca_every=20,         # pc0_var, pc0_share, pc0_<feature>, pc0_score
        )
        scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")
        odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers

    For moments on data that fits in memory, polars already does this --
    ``df.rolling(clock, period=...).agg(...)`` with an exponential weight agrees
    to 1e-14. The reason to reach for the spec is a stream, a saved state, or the
    cost: polars recomputes each window at ``O(n * W)`` where this is ``O(n)``.

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
        "lags": lags,
        "window": window,
        "window_every": window_every,
        "window_budget": window_budget,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: ew_cov() takes no targets; its statistics are over "
            "the features"
        )
        raise TypeError(msg)
    # `targets` is required by the common-parameter schema but unused here.
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def sgd(
    name: str,
    *,
    targets: list[str],
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
    scale_features: bool = False,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
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
         - ``clamp(p - y, +/- delta)``
       * - ``quantile``
         - identity
         - ``eta``
         - ``1{y < p} - tau``
       * - ``epsilon_insensitive``
         - identity
         - ``eta``
         - 0 inside the tube, else ``sign(p - y)``
       * - ``poisson``
         - log
         - ``exp(eta)``
         - ``p - y``
       * - ``logistic``
         - sigmoid
         - ``sigmoid(eta)``
         - ``p - y``

    then ``g_i = d * z_i * w + l2 * b_i`` and ``b_i -= lr_i * g_i``.

    .. rubric:: Parameters

    ``loss``
        One of the table's. Default ``"squared"``. ``epsilon_insensitive`` has a
        sign-valued subgradient, so a constant rate oscillates in a band around
        the optimum; use ``schedule = "inv_scaling"`` with it.
    ``huber_delta``
        The Huber cut, in target units -- not in units of the residual std, as for
        :func:`huber`. Default 1.0; ``inf`` clips nothing, which is the squared
        loss.
    ``quantile``
        The level for ``loss = "quantile"``; required for it.
    ``eps``
        The half-width of the insensitive tube, in target units. Default 0.1.
    ``learning_rate``, ``schedule``, ``power``
        The rate (default 0.01) and its schedule: ``"constant"``,
        ``"inv_scaling"`` (``lr / (1 + n_eff) ** power``, ``power`` default 0.5)
        or ``"adagrad"`` (``lr / (sqrt(G_i) + 1e-8)``). AdaGrad's sum of squared
        gradients and ``n_eff`` both decay on the model's clock, so an annealed or
        adapted rate opens up again after a long gap instead of staying frozen.
    ``l2``
        A ridge on every step. Default 0.0.
    ``clip_gradient``
        A cap on the gradient's magnitude. Default ``1e3``, not off, because
        ``poisson`` needs it: ``p = exp(eta)``, so a row that pushes ``eta`` up
        makes the next gradient exponentially larger and a constant rate diverges
        within a few thousand rows. It never binds for an identity-link fit.
    ``scale_features``
        Take the step in standardized coordinates, which is the difference between
        one learning rate for every column and one per scale. Default ``False``.
        Each row is standardized against the running moments with the row
        admitted, sklearn's ``StandardScaler.partial_fit`` then ``transform``.
        That bounds a standardized value by ``sqrt(n_eff)``, and it is not a leak:
        the rule is about the target, and the features of the row being predicted
        are known. Against the moments from before the row, a variance a few rows
        old can be tiny by chance, and one step throws a coefficient the rest of a
        short group never brings back. The same standardized row serves the
        prediction and the step, and the coefficients come back in the caller's
        units. Without an intercept the row is scaled by each column's root mean
        square and not centred, as for :func:`ewridge`.
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
        ``scale_features`` the step and the projection are taken in standardized
        coordinates with the bounds and the sum carried over exactly, and the
        reported coefficients satisfy the constraint in the caller's units after
        every learned row. A sum the bounds cannot reach, a floor above a cap, or
        an infinite bound on the wrong side is refused by name; floors of ``[0.1,
        0.2, 0.3]`` accept a sum of ``0.6`` although they add up to
        ``0.6000000000000001``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#sgd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#sgd>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature, in the caller's
        units, the constraints satisfied after every learned row
        (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. ``pred_<t>`` is a probability under ``"logistic"`` and a rate
    under ``"poisson"``, and ``emit_metrics`` reads a logistic fit as
    probabilities against labels.

    .. rubric:: Example

    .. code-block:: python

        weights = po.spec.sgd(
            "w", targets=["y"], features=["signal_a", "signal_b", "x0"], halflife=200.0,
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
    constraints are checked as above.
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
        "scale_features": scale_features,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def pa(
    name: str,
    *,
    targets: list[str],
    features: list[str],
    mode: str = "pa1",
    c: float | None = None,
    eps: float | None = None,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Passive-aggressive regression (Crammer et al. 2006): each row asks the fit to
    come within ``eps`` of its target, and the update is the smallest change that
    does so.

    Passive when the constraint already holds, aggressive when it does not, and
    there is no learning rate to tune. O(k) per row, like :func:`sgd`.

    .. rubric:: The fit

    With ``p = z . b``, ``loss = max(0, |y - p| - eps)`` and ``s = ||z||^2``:

    .. code-block:: text

        pa    tau = loss / s                 (unbounded)
        pa1   tau = min(c, loss / s)         (capped at c)
        pa2   tau = loss / (s + 1 / (2c))    (damped by c)
        b    += tau * sign(y - p) * z

    The model keeps no accumulators, so there is nothing for the clock to decay:
    each step fully satisfies the current row, and older rows survive only through
    the coefficients they left behind. ``n_eff`` still decays, so ``min_periods``
    means the same thing as elsewhere, but the coefficients have no halflife.

    .. rubric:: Parameters

    ``mode``
        ``"pa"``, ``"pa1"`` (the default) or ``"pa2"``. Prefer the bounded modes
        when outliers are possible: plain ``"pa"`` moves the fit as far as it
        takes to satisfy a single bad row.
    ``c``
        The cap (``pa1``) or damping (``pa2``). Default 1.0; ``inf`` caps nothing,
        so either bounded mode is then ``"pa"``.
    ``eps``
        The margin, in target units: the row is close enough inside it and nothing
        moves. Default 0.1.
    ``coef_min``, ``coef_max``, ``coef_sum``
        Constraints on the slopes, exactly as for :func:`sgd`. The projection
        follows each update, so the step does not meet the row's margin exactly:
        it is the closest feasible coefficient to the one that would. A truth
        outside the feasible set is never realizable, so the model keeps stepping
        against the walls; a small ``c`` keeps those steps small.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    A row weight below 1 scales ``tau``, so a half-weight row moves the fit half
    as far; a weight above 1 counts as 1. The update is a projection onto the
    row's constraint, and repeating a projection changes nothing, so there is no
    "two observations" to emulate.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#pa
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#pa>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        Per target, the intercept then one entry per feature (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them.

    .. rubric:: Example

    .. code-block:: python

        pa = po.spec.pa(
            "pa", targets=["y"], features=["x0", "x1"], halflife=200.0,
            mode="pa1", c=0.1,   # the step is capped at c
            eps=0.05,            # inside this margin nothing moves
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
    }
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def holt(
    name: str,
    *,
    targets: list[str],
    level_halflife: float | Duration | None = None,
    trend_halflife: float | Duration | None = None,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Holt's linear trend: the target's own level and slope, extrapolated.

    The one model that takes no features -- the forecasting baseline a
    feature-based model should have to beat. If a regression cannot outperform
    "the series is going up at about this rate", its features are not earning
    their place; run it in the same bank and compare ``sigma``, or let
    ``emit_selected`` choose.

    .. rubric:: The fit

    Per row and target, with ``s`` the clock since the target was last observed
    (this row's delta included, so ``s`` is the row's own delta on a stream with
    no gaps), ``w`` the row's weight, and ``W``, ``V`` the weight the level and
    the trend have gathered, each decayed on its own halflife (``lam_l = 0.5 ** (s
    / level_halflife)``, ``lam_b`` likewise):

    .. code-block:: text

        pred = l + b * s                                   extrapolate s clock units ahead
        l'   = (lam_l * W * pred + w * y) / (lam_l * W + w)
        b'   = (lam_b * V * b + w * (l' - l) / s) / (lam_b * V + w)

    Level and trend are weighted means, as every accumulator here is: a row at
    weight ``w`` counts ``w`` times, and an infinite halflife forgets nothing and
    fits the whole history. The gains ``w / (lam * W + w)`` start at 1 and fall to
    the textbook's fixed ``1 - lam`` as the weight saturates, so a new series is
    followed sooner and the fit is statsmodels' ``Holt`` from there. A row at the
    last row's clock is a second observation the level takes in; the trend holds,
    since a move over no clock has no slope. A row with a null target or a zero
    weight leaves ``l`` and ``b`` where the last observation put them and carries
    its clock to the next one, so it gives the same numbers as if it were absent.
    The trend is per clock unit, so an irregular clock extrapolates the right
    distance.

    .. rubric:: Parameters

    ``level_halflife``
        How fast the level forgets, in clock units. Defaults to the spec's
        ``halflife`` -- one knob under two names, ``inf`` included.
    ``trend_halflife``
        How fast the trend forgets, in clock units. Default four times the level
        halflife; ``inf`` is the whole history's drift, not a trend pinned at
        zero.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group``, the
    diagnostics and the rest, with the fields each diagnostic adds.

    There is no seasonal term: a seasonal index is a ``group`` on the phase, which
    the bank already does.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#holt
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#holt>`_):

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not null.
    ``n_eff``
        The accumulated weight before the row, as everywhere.
    ``coef``
        ``[level, trend]`` per target, the whole state; :func:`coef_index` names
        the two.

    plus the fields of the diagnostics switched on, as :mod:`polars_online.spec`
    describes them. :meth:`polars_online.ModelBank.predict` extrapolates over the
    clock distance from the row the model last learned, capped by ``max_dclock``.

    .. rubric:: Example

    .. code-block:: python

        baseline = po.spec.holt(
            "baseline", targets=["y"], clock="t", max_dclock=600.0,
            level_halflife=200.0,     # how fast the level forgets
            trend_halflife=2000.0,    # how fast the trend forgets; inf is the whole history's drift
        )
        out = po.ModelBank([baseline]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``halflife`` and
    ``level_halflife`` together are refused, being one knob.
    """
    model: dict[str, Any] = {
        "type": "holt",
        "level_halflife": level_halflife,
        "trend_halflife": trend_halflife,
    }
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
    update_every: int | None = None,
    split_merge: float | None = None,
    split_merge_every: int | None = None,
    dead_frac: float | None = None,
    standardize: bool | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted k-means over the feature columns.

    Not a regression: there are no targets. Each row is assigned to the nearest of
    ``k`` centres before it is learned, so the label is out-of-sample like every
    prediction here. Each centre is the decayed weighted mean of the rows assigned
    to it: :func:`ew_cov`'s mean recursion, per cluster.

    .. rubric:: The recursion

    .. code-block:: text

        j*   = argmin_j |x - c_j|^2          distances in units of each feature's EW sd
        n'_j = lam * n_j + w
        c'_j = (lam * n_j * c_j + w * x) / n'_j       for the nearest j

    Alongside each centre the EW squared radius ``r2_j``, the mean of ``|x -
    c_j|^2`` over the rows assigned there, which the split-merge rule reads. Rows
    are folded into per-centre batches and applied every ``update_every`` learned
    rows, so ``update_every = 1`` is plain sequential k-means and a larger value a
    mini-batch one.

    .. rubric:: Parameters

    ``k``
        The number of centres; required.
    ``warm_rows``, ``seed_rule``, ``seed``
        Seeding. The first ``max(warm_rows, k)`` learned rows (default 500) are
        buffered, then the centres are placed by ``seed_rule``: ``"lloyd"`` (the
        default: k-means++ then ten weighted Lloyd iterations over the buffer),
        ``"kmeanspp"``, ``"farthest"`` (Gonzalez, from the first row) or
        ``"first"`` (the first ``k`` distinct rows). ``seed`` (default 0) drives
        the two random rules; the same seed gives the same centres. The buffer is
        replayed into the centres and freed, so the model is O(state) again from
        that row on. Outputs are null until seeding and until ``n_eff`` reaches
        ``min_periods``.
    ``update_every``
        Learned rows between applications of the per-centre batches. Default 1.
    ``split_merge``, ``split_merge_every``
        A row farther from its centre than a blob of the typical radius produces
        (about four standard deviations of ``|x - c|^2`` above it) is far. It is
        scored, but summarised instead of learned, so it neither drags the centre
        nor widens the radius. Every ``split_merge_every`` learned rows (default
        100) the two closest centres are compared. If their distance is under
        ``split_merge`` (default 0.5; ``0`` disables) times the sum of their
        radii, and enough far rows have gathered somewhere (at least three, and
        five per cent of the window's weight), they are merged and the freed
        centre is placed at the far rows' mean. So a cluster that appears after
        seeding gets a centre without anyone restarting. Far rows still count in
        the radius at each check as if they sat at the cut, so a cut the data has
        outgrown widens until the rows are learned again.
    ``dead_frac``
        Re-place a centre whose weight has decayed below ``dead_frac * n_eff / k``
        the same way, on whatever far rows there are (default 0.05; ``0``
        disables). A centre whose cluster vanished is re-placed ``log2(1 /
        dead_frac)`` halflives later (4.3 at the default, 2 at 0.25), and a
        cluster lighter than ``dead_frac / k`` of the stream loses its centre
        whenever any row is far.
    ``standardize``
        Measure distances in units of each feature's EW standard deviation,
        tracked alongside the centres; the coordinates themselves are never
        rescaled, so the centres stay in the features' units. Default ``True``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec (`docs/OUTPUTS.md#kmeans
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#kmeans>`_):

    ``cluster``
        The nearest centre's index (``i32``), before the row is learned from; null
        until seeding.
    ``dist``, ``dist2``
        The distance to that centre, and to the runner-up (null when ``k == 1``),
        so ``dist2 - dist`` is the margin.
    ``n_eff``
        As everywhere.
    ``coef``
        The centres: ``k`` rows of ``len(features)``, flattened cluster-major.
        :func:`coef_index` lays it out, with ``target`` reading ``"cluster0"``,
        ``"cluster1"``, ... and ``term`` the feature whose coordinate the position
        holds.

    .. rubric:: Example

    .. code-block:: python

        km = po.spec.kmeans(
            "km", features=["x0", "x1", "x2"], clock="t", halflife=2000.0, max_dclock=300.0,
            k=3,
            warm_rows=100,           # seeding waits for this many rows, then replays them
            seed_rule="lloyd",       # k-means++ then ten Lloyd iterations over the buffer
            split_merge=0.5,         # two centres in one blob: one moves to the far rows
            dead_frac=0.05,          # a centre whose blob vanished is re-placed 4.3 halflives later
        )
        out = po.ModelBank([km]).fit_predict(df).unnest("km")
        layout = po.spec.coef_index(km)    # target = "cluster0".., term = the feature

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, which this model has not got.
    """
    model: dict[str, Any] = {
        "type": "kmeans",
        "k": k,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
        "seed": seed,
        "update_every": update_every,
        "split_merge": split_merge,
        "split_merge_every": split_merge_every,
        "dead_frac": dead_frac,
        "standardize": standardize,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: kmeans() takes no targets; its clusters are over "
            "the features"
        )
        raise TypeError(msg)
    # `targets` is required by the common-parameter schema but unused here.
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def micro(
    name: str,
    *,
    features: list[str],
    eps: float,
    beta_mu: float | None = None,
    max_clusters: int | None = None,
    prune_every: int | None = None,
    macro_link: float | None = None,
    standardize: bool | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Density-based clustering over the feature columns: DenStream-style
    micro-clusters with a linkage step over them.

    Not a regression: there are no targets, and unlike :func:`kmeans` there is no
    fixed number of clusters. The model keeps a bounded set of small summaries
    (micro-clusters), each a decayed weight, centre and radius, and reads clusters
    off them as the chains of summaries that touch. That finds clusters of any
    shape (moons, rings), reports rows that belong to none, and lets clusters
    appear and vanish as the stream moves.

    .. rubric:: The recursion

    A summary is ``(n, c, r2)``: decayed weight, centre, and the EW mean squared
    distance of its rows from the centre (DenStream's radius, with the fading
    function being the decay). Distances are measured in the metric ``mw_i = 1 /
    var_i`` when ``standardize`` (the default), so ``eps`` is a bound per
    standardized coordinate and the bound in the metric is ``E = eps² p`` for
    ``p`` features. Each row:

    .. code-block:: text

        n_j <- lam n_j                                   every summary
        j   = the nearest potential summary, if absorbing a unit row keeps
              a r2_j + a b |x - c_j|^2 <= E,   a = n_j/(n_j+1), b = 1/(n_j+1)
              else the nearest outlier summary by the same test
              else a new summary at x with the next id
        n_j <- n_j + w,  c_j <- c_j + w/n_j (x - c_j),  r2_j <- min(., E)

    A summary is potential (established) once ``n >= beta_mu`` and outlier below;
    a new one is opened at the cap ``max_clusters`` by evicting the lightest
    outlier summary, else the lightest potential one. Every ``prune_every``
    learned rows a checkpoint prunes and links. It drops potential summaries
    lighter than ``beta_mu``, and outlier summaries lighter than DenStream's
    ``xi(age)``: the weight a summary that had been gathering a row per clock unit
    since it opened would need to reach ``beta_mu`` within ``Tp`` more, with ``Tp
    = ceil(h log2(beta_mu / (beta_mu - 1)))`` for halflife ``h``. With no decay
    nothing is pruned, only capped. Then it links the potential summaries by
    single linkage: centres within ``L`` of each other share a label. Ids are
    monotone and never reused; a label is the smallest id in its chain, so it
    survives everything but the loss of that summary.

    .. rubric:: Parameters

    ``eps``
        The within-cluster spread the model should read as one cluster, per
        standardized coordinate; required. About 0.07 for two-dimensional shapes,
        0.3 for well-separated Gaussians in twenty dimensions. Both ways to get it
        wrong show in the outputs. If nearly every row is an ``outlier`` and
        ``cluster`` stays null, ``eps`` is too small: no summary reaches
        ``beta_mu`` before it is pruned. If ``n_micro`` is about the number of
        clusters, ``eps`` is too coarse for the derived link, which then bridges
        them; lower ``eps``, or set ``macro_link``.
    ``beta_mu``
        The weight at which a summary is established. Default 3.
    ``max_clusters``
        The cap on live summaries. Default 200.
    ``prune_every``
        Learned rows between checkpoints. Default 100.
    ``macro_link``
        ``L`` as a multiple of ``eps sqrt(p)``: ``0`` links nothing, so each
        summary is its own cluster; ``2`` links summaries that touch. Left out,
        ``L`` is derived at each checkpoint from the spacing the summaries already
        show: 1.5 times the 90th percentile of the nearest-neighbour distance,
        never below ``2 eps sqrt(p)``. So a chain along a shape holds without a
        constant that fragments one shape and bridges another.
    ``standardize``
        Measure in units of each feature's EW standard deviation. Default
        ``True``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    A row of weight ``w`` is admitted where a unit row would be and absorbed with
    its full weight; a zero-weight row advances the clock and learns nothing.
    Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is learned
    (`docs/OUTPUTS.md#micro
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#micro>`_):

    ``cluster``
        The label of the nearest established summary (``i64``); null while there
        is none.
    ``dist``
        The distance to that summary's centre.
    ``micro``
        The id of the summary this row goes to (``i64``) -- the one it opens, when
        none can take it.
    ``outlier``
        Whether no established summary takes it (``bool``).
    ``n_clusters``, ``n_micro``
        Live clusters and live summaries (``i32``), so churn is visible without
        diffing labels.
    ``n_eff``
        As everywhere.
    ``coef``
        The established summaries, one ``[id, label, n, radius, c_1, ..., c_p]``
        row each, flattened -- as many rows as there are, so :func:`coef_index`
        does not apply.

    .. rubric:: Example

    .. code-block:: python

        mc = po.spec.micro(
            "mc", features=["x0", "x1"], clock="t", halflife=2000.0, max_dclock=300.0,
            min_periods=50.0,
            eps=0.3,             # the spread read as one cluster, per standardized coordinate
            beta_mu=5.0,         # a summary with at least this much weight is established
            prune_every=100,     # every this many rows: prune the light summaries, link the rest
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
        "macro_link": macro_link,
        "standardize": standardize,
    }
    if "targets" in common:
        msg = (
            f"spec {json.dumps(name)}: micro() takes no targets; its clusters are over the features"
        )
        raise TypeError(msg)
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def ew_class(
    name: str,
    *,
    features: list[str],
    label: str,
    classes: list[str],
    covariance: str | None = None,
    precision_prior: float,
    window: float | Duration | None = None,
    window_every: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian classifier on :func:`ew_cov`'s moments, one set per class:
    quadratic discriminant analysis, linear discriminant analysis or Gaussian
    naive Bayes.

    A label column takes the place of a numeric target. The model keeps one
    ``ew_cov`` state per class: a weight ``n_c``, a mean ``mu_c`` and a centred
    covariance ``C_c``. It scores a row by Bayes' rule over Gaussian classes
    before the row's own label is learned, so a row's probabilities never saw its
    label. That is also how a stream whose labels arrive late is scored: null the
    label, keep the features.

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

    and ``n_eff <- lam n_eff + w`` counts every accepted row, labelled or not, so
    ``min_periods`` means the same number of rows as everywhere else. A row with a
    non-finite feature is null and learns nothing; a zero-weight row advances the
    clock.

    .. rubric:: Parameters

    ``label``
        The column that holds the class of each row, read as a key -- any dtype
        with a string form, so ``["0", "1"]`` for an integer column and ``["true",
        "false"]`` for a boolean one. A null label is a row to score but not to
        learn from: the model classifies it and ticks its clock, and no class
        moves.
    ``classes``
        Every value the label can hold, in the order the ``p_<class>`` fields are
        written; required. A non-null value it does not list is an error naming
        the row.
    ``covariance``
        The shape: ``"full"`` (the default), a covariance per class; ``"shared"``,
        the weight-averaged one, so the decision boundaries are linear; or
        ``"diagonal"``, the variances alone, which cannot see a correlation.
        ``"full"`` costs one ``k x k`` Cholesky per class per row (per class per
        row under a ``window`` too, since a windowed covariance moves every row);
        ``"shared"`` factorizes once per row.
    ``precision_prior``
        A ridge on every class covariance, in the features' units, so the first
        rows of a class, whose sample covariance is singular, are scored with a
        finite, isotropic one; required. It is scaled by ``s_c``, the class's own
        prior scale, which starts at 1 and decays by ``lam * n_c / (lam * n_c +
        w)`` on every row the class learns, so the ridge washes out as the class
        fills in, exactly as :func:`ew_cov`'s ``precision_prior`` does.
    ``window``, ``window_every``, ``window_budget``
        A hard cutoff on the history each class's moments are computed from, in
        clock units, as for :func:`ewridge`: a row older than ``window``
        contributes to no class, which is what lets a classifier follow class
        means that move -- over a long history two regimes average together and
        the labels go to chance. ``window_every`` is the snapshot cadence and
        ``window_budget`` bounds each ring in MiB. ``n_eff`` is the weight inside
        the window, in the struct and in the ``min_periods`` gate, and the class
        moments a row is scored against are the window's.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest. Nothing residual-based applies, and each such switch is refused by name:
    ``emit_sigma``, ``emit_metrics``, ``conformal``, drift and the rest.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is learned
    (`docs/OUTPUTS.md#ew_class
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#ew_class>`_):

    ``class``
        The class with the largest posterior (``str``; the first, on a tie), null
        before ``min_periods`` and while no class has been seen.
    ``p_<class>``
        One per class in ``classes`` order: its posterior probability, so the
        ``p_`` fields sum to 1 -- exactly 0 for a class no row has carried yet.
    ``n_eff``
        As everywhere.
    ``coef``
        The class means, one row per class in ``classes`` order, each one entry
        per feature; :func:`coef_index` lays the list out as ``(class, feature)``,
        and a class not yet seen is null.

    .. rubric:: Example

    .. code-block:: python

        labelled = df.with_columns(
            pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
        )
        cl = po.spec.ew_class(
            "cl", features=["x0", "x1", "x2"], clock="t", halflife=200.0, max_dclock=300.0,
            label="dir", classes=["down", "up"],
            covariance="shared",         # pooled by class weight: linear boundaries
            precision_prior=0.1,         # the ridge that makes a class scoreable from its first row
            min_periods=20.0,
        )
        out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
        calls = out.select("dir", "class", "p_up", "n_eff").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets`` -- this model takes ``label``.
    """
    model: dict[str, Any] = {
        "type": "ew_class",
        "classes": classes,
        "covariance": covariance,
        "precision_prior": precision_prior,
        "window": window,
        "window_every": window_every,
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
    targets: list[str],
    a: str | None = None,
    b: str | None = None,
    a_suffix: str | None = None,
    b_suffix: str | None = None,
    features: list[str] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A sequential test of a sign, by betting: an e-process, read at any row.

    Not a regression. Per target the model keeps the wealth of two gamblers, one
    betting that the next sign is positive and one that it is negative. The answer
    is evidence that can be read at every row, as often as wanted, and acted on
    the first time it is enough, which a p-value cannot be: checking it repeatedly
    inflates its error rate. With ``a`` and ``b`` it asks instead whether one spec
    of the bank predicts closer than another.

    .. rubric:: The recursion

    Per target the row's value is reduced to its sign ``s`` in ``{-1, 0, +1}``,
    and with ``n_pos`` and ``n_neg`` the counts of positive and negative rows
    before this one and ``n = n_pos + n_neg``:

    .. code-block:: text

        lam_pos = max(0, (n_pos - n_neg) / (n + 1))    lam_neg = max(0, (n_neg - n_pos) / (n + 1))
        E_pos  *= 1 + lam_pos * s                     E_neg  *= 1 - lam_neg * s

    ``(n_pos - n_neg) / (n + 1)`` is ``2p - 1`` for the Krichevsky-Trofimov
    estimate ``p = (n_pos + 1/2) / (n + 1)`` of ``P(s = +1)``: the stake a gambler
    with a ``Beta(1/2, 1/2)`` prior puts on the next sign, clipped so that each
    side bets only on the direction it tests. Both stakes are computed from the
    rows before. Under the null (given everything before it, a row is at least as
    likely to be negative as positive) ``E_pos`` is a non-negative supermartingale
    with ``E_pos[0] = 1``, and Ville's inequality gives ``P(max_t E_pos[t] >=
    1/alpha) <= alpha``. That is the whole guarantee. ``log_e_pos >=
    log(1/alpha)`` on any row rejects "no more positives than negatives" at level
    ``alpha``, and the stream can be read at every row and stopped the moment it
    crosses. Nothing about ``y`` but its sign is assumed: no independence of the
    sizes, no bound, no variance. What it does not test is the size: a stream up
    by a hair 60% of the time and down by a mile the rest rejects. ``(E_pos +
    E_neg) / 2`` is the two-sided e-value.

    .. rubric:: Parameters

    ``a``, ``b``, ``a_suffix``, ``b_suffix``
        Compare two specs of the same bank. Each target ``t`` names a residual
        field both carry: ``resid_<t>`` on each side, plus the side's grid suffix
        when it is a grid (``a_suffix="@h50"`` picks ``resid_<t>@h50`` of ``a``;
        ``"__r0.1"`` a ridge instance). The sign tested is that of ``|resid_b| -
        |resid_a|``, positive when ``a`` was closer on the row. Any loss that
        grows with ``|resid|`` (squared, absolute, Huber) gives the same sign, so
        this is a test of "``a`` beats ``b``" under any of them. The bank runs
        ``a`` and ``b`` first, so the comparison reads the same out-of-sample
        residuals their structs report; a row where either side is null (warm-up,
        a skipped row) is a row the test sits out. A spec named against itself
        with the same suffix, a side that is not in the bank or is itself a
        ``seqtest``, and a target neither side has a residual for are refused by
        name. Two instances of one grid (``a_suffix="@h20"`` against
        ``b_suffix="@h400"``) are a comparison like any other.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    A trial is a row, so ``weight`` is refused and there is no
    ``halflife``/``lam``: a process that forgot its losses would not be an
    e-process. ``session`` or ``on_clock_reset = "reset_state"`` restarts it.
    ``min_periods`` defaults to 0. No ``features`` (the column is the test; the
    keyword is taken so that a frame namespace can pass ``[]``), no ``coef``, and
    nothing residual-based applies -- there is no prediction.
    :func:`polars_online.eval.seqtest` is the same computation in polars
    expressions over a frame in memory.

    .. rubric:: Output

    One struct column named after the spec, per target ``t`` and read before the
    row is learned (`docs/OUTPUTS.md#seqtest
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#seqtest>`_):

    ``log_e_pos_<t>``, ``log_e_neg_<t>``
        The two gamblers' log wealth (``log E``, so 0 is no evidence and ``log(20)
        = 3.0`` is level 0.05).
    ``n_pos_<t>``, ``n_neg_<t>``
        The signs counted so far (``Int64``). A zero or null target bets nothing
        and counts nothing.
    ``n_eff``
        As everywhere, decayed by nothing.

    With ``a`` and ``b`` the fields read ``log_e_a_<t>``, ``log_e_b_<t>``,
    ``wins_a_<t>`` and ``wins_b_<t>`` instead.

    .. rubric:: Example

    .. code-block:: python

        common = dict(targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0)
        ridge = po.spec.ewridge("ridge", halflife=500.0, **common)
        kalman = po.spec.kalman("kalman", halflife=500.0, coef_halflife=100.0, **common)
        sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")   # is y usually positive?
        # does kalman predict closer than ridge?
        closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge")
        out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
        verdict = out["closer"].struct.field("log_e_a_y").max()   # >= log(20): kalman won at 5%


    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` for
    ``weight``, ``halflife`` or ``lam``, and for the comparison refusals above.
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
    targets: list[str],
    features: list[str],
    lags: list[int] | None = None,
    serial_rule: str | None = None,
    bins: int | None = None,
    bin_rule: str | None = None,
    bin_warm_rows: int | None = None,
    bin_edges: dict[str, list[float]] | list[list[float]] | None = None,
    window: float | Duration | None = None,
    window_every: int | None = None,
    window_budget: dict[str, float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Every (feature, target) pair's exponentially weighted moments, kept in the
    state and read back as a table.

    Not a regression and not a joint fit: each pair ``(x_j, y_t)`` is its own
    two-column :func:`ew_cov`, so a wide feature set against a few targets costs
    ``O(p * T)`` per row rather than the ``O((p + T)²)`` of one ``ew_cov`` over
    all the columns. Nothing is written per row but ``n_eff``; the pairs live in
    the state and :meth:`polars_online.ModelBank.marginal` reads them. Use it to
    screen thousands of features against a few targets in one pass.

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

    (deviations from the means before the row), the same arithmetic as
    ``ew_cov``'s, so a pair's ``corr`` equals the ``corr`` an ``ew_cov`` over the
    two columns reports, to the bit. A null target ages that target (``W_t *=
    lam``, ``Q_t *= lam^2``) and learns nothing for it; a null feature skips the
    row, as everywhere.

    .. rubric:: Parameters

    ``lags``, ``serial_rule``
        The pair's moments at those lags too, counted in learned rows within the
        group -- not in rows where that target was present, since the ring is
        shared, so for a sparsely present target the lag is a row distance, not an
        observation distance. They add four list columns per pair to the table:
        ``lagcorr_xx`` and ``lagcorr_yy``, the two series' own autocorrelations,
        and ``lagcorr_xy`` and ``lagcorr_yx``, the feature now against the target
        ``l`` rows back and the reverse. A feature whose ``lagcorr_yx[0]`` exceeds
        its ``corr`` leads the target, and one whose ``lagcorr_xy[0]`` does
        follows it. The pair is the same statistic ``ew_cov(lags=)`` computes, to
        the bit.

        ``serial_rule`` turns them into an honest count. ``t`` is built on
        ``n_kish``, which is right for unequal weights and silent about serial
        dependence. On a smooth stream consecutive rows are nearly the same
        observation, and the variance of a sample correlation is not ``1/n`` but
        ``[1 + 2 * sum_l rho_x(l) * rho_y(l)] / n`` (Bartlett 1935). ``n_serial``
        is ``n_kish`` divided by that bracket and ``t_serial`` the statistic
        against it. ``"truncated"`` sums the kept lags as they are, and reports
        null ``n_serial`` and ``t_serial`` when that takes the bracket to zero or
        below (two series whose autocorrelations have opposite signs).
        ``"geometric"`` fits ``rho(l) = phi^l`` per series by least squares on
        ``log rho`` over the kept lags with ``rho > 0`` and sums the tail in
        closed form, which is the right choice when both series are exponentially
        weighted, and the reason the lags need not be dense. It reports the fitted
        ``phi_x`` and ``phi_y``, and gives up (null ``n_serial``) when fewer than
        two kept lags are positive on either side. Measured on two independent
        AR(1) series with ``phi_x = 0.9`` and ``phi_y = 0.8``: ``t = 2.39``,
        significance that is not there, against ``t_serial = 1.03``, with ``n_kish
        = 3000`` becoming ``n_serial = 557``. Cost: ``(3L + 1) * p * T + L * T``
        doubles beside the pair moments and a ring of ``max(lags)`` learned rows
        -- the one place ``marginal`` holds rows rather than state.
    ``bins``, ``bin_rule``, ``bin_warm_rows``, ``bin_edges``
        The nonlinear view. Every statistic above is linear, and a feature can be
        strongly related to a target with ``corr`` at zero: a threshold, a V, a
        saturation. With ``bins = 16`` each pair also reports the target's weight,
        mean and variance inside each of the feature's bins: its response curve,
        in ``bin_edges``, ``bin_n``, ``bin_mean_y`` and ``bin_var_y``. It reports
        the best single cut of that curve too. ``split_gain`` is the fraction of
        the target's variance the cut removes, a regression stump's gain, directly
        comparable with ``corr²``; ``split_at`` is where it falls;
        ``split_gain_t`` is the ``t`` a ``corr`` would need to match it. Read
        ``split_gain_t`` as a ranking, not a p-value: the cut was chosen by
        maximising over ``bins - 1`` candidates, which the statistic does not
        know. It uses ``n_serial`` in place of ``n_kish`` when ``serial_rule``
        gives one. It costs ``O(bins)`` of state per pair and one binary search
        per pair per row.

        The edges are fixed once and never move, so a bin means the same thing for
        the life of the stream. ``bin_edges`` sets them outright, as a list per
        feature in ``features`` order or a dict keyed by feature name, which is
        exact and comparable across runs and groups; ``bins``, ``bin_rule`` and
        ``bin_warm_rows`` describe learning them and are refused beside it.
        Otherwise they are learned from the first ``bin_warm_rows`` learned rows
        (default 1,000) under ``bin_rule``. ``"quantile"`` (the default) gives
        equal weight per bin; a value that carries more than a bin's share, an
        indicator's zero say, fills a bin of its own and the remaining bins share
        what is left. ``"fixed"`` gives equal widths between the smallest and
        largest value seen. Those warm-up rows are held, not spent: the moment the
        edges exist every one of them is replayed with its own decay, so the
        histogram is what it would have been had the edges been known before the
        first row, to the bit. The price is memory, ``bin_warm_rows * (features +
        targets)`` floats, refused up front past 256 MiB. Until the edges are
        fixed the bin columns are empty and the split columns null. A feature
        keeps only the bins it can support, so the lists are ragged: a binary
        feature gets two bins whatever ``bins`` says, and a constant one a single
        bin and no split. Decay reaches the histogram as it reaches the pair
        moments, so a clock gap past ``max_dclock`` empties it along with them.
    ``window``, ``window_every``, ``window_budget``
        A hard cutoff on the history the pairs are computed from, in clock units,
        as for :func:`ewridge`: a row older than ``window`` contributes nothing,
        and inside the window the weights are still exponential. Every moment a
        pair is built from is truncated (the weight, both means and the three
        centred second moments), so ``corr``, ``beta`` and ``t`` describe the
        window and nothing else. That matters most for a screen: two regimes of
        opposite sign average to nothing over a long history. ``window_every`` is
        the snapshot cadence and ``window_budget`` bounds each ring in MiB. The
        ``n_eff`` the struct writes and the one the table reports are the weight
        inside the window. ``lags`` and ``window`` are refused together (the ring
        of past rows is not something a window's snapshot truncates), and so are
        ``bins`` and ``window`` (a snapshot of the histogram is ``bins`` times the
        size of one).

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    ``add_intercept`` and ``coef_every`` have nothing to act on here, and nothing
    residual-based applies (there is no prediction), so the residual switches are
    refused by name. A column may not be both a target and a feature.
    ``min_periods`` defaults to 3: two rows give a correlation of ±1 whatever the
    data, three the first one with content.

    .. rubric:: Output

    One struct column named after the spec holding ``n_eff`` alone
    (`docs/OUTPUTS.md#marginal
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#marginal>`_).
    The pairs are read from the state with
    :meth:`polars_online.ModelBank.marginal`, one row per (group, instance,
    feature, target) with the moments, ``corr``, ``beta``, ``t``, ``n_kish``, and
    the lag and bin columns above; that method documents every column. A group
    that closes writes the same pairs to
    :meth:`polars_online.ModelBank.closed_groups` as ``pair_*`` columns, one entry
    per pair.

    .. rubric:: Example

    .. code-block:: python

        pairs = po.spec.marginal(
            "pairs", targets=["y", "ret"], features=["x0", "x1", "x2", "signal_a", "signal_b"],
            clock="t", max_dclock=300.0, halflife=500.0, group="stock_id",
        )
        bank = po.ModelBank([pairs])
        bank.fit_predict(df)                          # the struct holds n_eff alone
        table = bank.marginal("pairs")   # a row per (group, instance, feature, target)
        one_stock = bank.marginal("pairs", group="b0")

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`), and ``ValueError`` naming
    the problem for ``bin_edges`` that miss or add a feature, for ``bins`` beside
    ``bin_edges``, and for ``lags`` or ``bins`` beside ``window``.
    """
    edges: list[list[float]] | None
    if isinstance(bin_edges, dict):
        missing = [f for f in features if f not in bin_edges]
        if missing:
            raise ValueError(
                f"marginal {name!r}: bin_edges is missing {missing}; give a list of "
                "edges for every feature, or pass a list of lists in features order"
            )
        extra = [f for f in bin_edges if f not in features]
        if extra:
            raise ValueError(f"marginal {name!r}: bin_edges has {extra}, which are not features")
        edges = [list(bin_edges[f]) for f in features]
    elif bin_edges is not None:
        edges = [list(e) for e in bin_edges]
        if len(edges) != len(features):
            raise ValueError(
                f"marginal {name!r}: bin_edges has {len(edges)} lists for "
                f"{len(features)} features; one list per feature, in features order"
            )
    else:
        edges = None
    model: dict[str, Any] = {
        "type": "marginal",
        "lags": lags,
        "serial_rule": serial_rule,
        "bins": bins,
        "bin_rule": bin_rule,
        "bin_warm_rows": bin_warm_rows,
        "bin_edges": edges,
        "window": window,
        "window_every": window_every,
        "window_budget": window_budget,
    }
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
    stream cannot keep them all moving without ``O(m²)`` a row. DECO replaces them
    with their average and estimates that. Not a regression: no targets, and
    nothing residual-based applies. Needs at least two features.

    .. rubric:: The recursion

    The row is standardised against the pre-row means and variances of an EW
    accumulator, ``r_i = (x_i - m_i) / sqrt(v_i)``, and with ``S1 = sum(r)`` and
    ``S2 = sum(r * r)`` over the ``n`` features the row's estimate is their Lemma
    2.3:

    .. code-block:: text

        u = (S1**2 - S2) / ((n - 1) * S2)      = mean_{i != j} r_i r_j / mean_i r_i**2

    which lies in ``(-1 / (n - 1), 1)``. The level then follows one of two
    dynamics, on the model's own clock with decay factor ``lam`` and row weight
    ``w``:

    .. code-block:: text

        "ew":      W' = lam * W + w,  a = lam * W / W',  b = w / W'
                   rho' = a * rho + b * u                     (rho = u at W = 0)
        "linear":  rho' = (1 - alpha - beta) * rho_bar' + alpha * u + beta * rho

    where ``rho_bar`` is the ``"ew"`` recursion run alongside as the target of the
    linear one. Two departures from the paper, on purpose. Its eq. 21 has a free
    intercept and applies correlation targeting to the DCC ``Q`` recursion, not to
    the linear one; writing the intercept as ``(1 - alpha - beta) * rho_bar`` is
    this library's reparameterisation, chosen because a streaming model has no
    sample to fit a free intercept on. And the paper permits ``alpha + beta``
    slightly above 1 under numerical bounds, where this refuses it. The paper also
    notes that ``u`` is a downward biased estimate of the equicorrelation
    (``E[u]`` is about 0.20 for a true 0.30 at ``m = 6``), and offers an
    alternative, ``1 - (1 / (n - 1)) * sum((r_i - rbar)**2)``, which this does not
    compute. Use ``u`` as a signal that moves with the market's correlation, not
    as the correlation; ``rho`` is not an ``ew_cov``'s ``corr`` over the columns,
    since the mean of a ratio is not the ratio of means.

    .. rubric:: Parameters

    ``dynamics``
        ``"ew"`` (the default) or ``"linear"``. ``"linear"`` needs ``alpha`` and
        ``beta``, both ``>= 0`` with ``alpha + beta < 1``; ``"ew"`` refuses them.
    ``alpha``, ``beta``
        The linear dynamics' weights on the row's estimate and on the previous
        level.
    ``blocks``
        A name to a subset of ``features``: the model then estimates one number
        per block and one per pair of blocks, which is the useful middle between
        one correlation and all of them. Every feature must be in exactly one
        block, and a block needs at least two.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    ``halflife`` or ``lam`` is required: both the standardiser and the level decay
    on it.

    .. rubric:: Output

    One struct column named after the spec, all read from the state before the
    row, so they are safe as features for that same row (`docs/OUTPUTS.md#deco
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#deco>`_):

    ``u``
        The row's own estimate of the equicorrelation.
    ``rho``
        The level as it stood before the row.
    ``loglik``
        The row's Gaussian log-density in standardised coordinates under that
        level.
    ``n_eff``
        As everywhere.
    ``coef``
        The correlation values, in the order of the ``u`` fields.

    With ``K`` named blocks the first two become ``u_<A>`` per block then
    ``u_<A>_<B>`` per pair, and ``rho_*`` likewise, with one ``loglik`` over all
    of them. ``u`` and ``loglik`` are null until every feature has a positive
    pre-row variance.

    .. rubric:: Example

    .. code-block:: python

        eq = po.spec.deco(
            "eq", features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=500.0,
            dynamics="ew",    # the EW mean of u; "linear" needs alpha and beta
        )
        blocked = po.spec.deco(
            "blocks", features=["x0", "x1", "x2", "signal_a"], halflife=500.0,
            # one number per block and one per pair of blocks: u_fast, u_slow, u_fast_slow
            blocks={"fast": ["x0", "x1"], "slow": ["x2", "signal_a"]},
        )
        out = po.ModelBank([eq, blocked]).fit_predict(df)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, and ``ValueError`` for fewer than two features, for
    ``alpha``/``beta`` under ``"ew"`` or missing under ``"linear"``, and for
    blocks that do not partition the features.
    """
    model: dict[str, Any] = {
        "type": "deco",
        "dynamics": dynamics,
        "alpha": alpha,
        "beta": beta,
        "blocks": [[k, list(v)] for k, v in blocks.items()] if blocks else None,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def bocpd(
    name: str,
    *,
    features: list[str],
    hazard: float = 250.0,
    hazard_col: str | None = None,
    emission: str = "diag",
    prior_mean: list[float] | None = None,
    prior_kappa: float | None = None,
    prior_nu: float | None = None,
    prior_scale: list[float] | None = None,
    robust_beta: float | None = None,
    prune_below: float | None = None,
    max_run: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Bayesian online changepoint detection (Adams & MacKay 2007): a posterior over
    how long the current regime has lasted.

    Every other detector here answers "has something changed?" with a statistic.
    This one keeps a distribution over the run length, the rows since the last
    break, so the answer carries the age of the regime with it: "we are 40 rows
    into a regime" is different information from "something broke". Not a
    regression: no targets, and ``halflife``/``lam`` are refused, since the
    run-length posterior is what forgets and ``hazard`` is how fast.

    .. rubric:: The recursion

    Their Algorithm 1, in log space, with ``H = 1 / hazard`` and ``pi_r`` run
    ``r``'s posterior predictive for this row:

    .. code-block:: text

        growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1) pi_r (1 - H)
        changepoint: P(r_t = 0,   x_1:t) = sum_r P(r_t-1 = r, x_1:t-1) pi_r H

    Slot ``r`` keeps the conjugate statistics of exactly the ``r`` rows that
    hypothesis says preceded this one in the run -- so slot 0 holds none and its
    predictive is the prior's, which is what makes "a new run starts here"
    something the data can vote on. A row costs ``O(runs * d²)``, and the run
    vector would grow by one every row, so runs below ``prune_below`` of the mass
    are dropped and ``max_run`` folds every longer run into the last kept one.
    That run takes their mass and keeps its own statistics, so ``max_run`` bounds
    how much history any run holds, not only the length of the vector, and
    ``run_mode`` saturates one below it.

    .. rubric:: Parameters

    ``hazard``
        The expected run length: ``H = 1 / hazard`` is the per-row chance of a
        break. Default 250.
    ``hazard_col``
        Read the hazard per row from a column instead, declared in the targets
        slot the way a weight is. A null or non-finite value there falls back to
        ``hazard``; a value of 1 or less is an error naming the row, since a
        hazard is the expected rows between changepoints.
        :meth:`polars_online.ModelBank.predict` reads the column too.
    ``emission``
        ``"diag"`` (the default) is a normal-inverse-gamma per feature;
        ``"gaussian"`` a normal-inverse-Wishart over all of them -- both exact
        conjugate updates, and the second is the one that can see a break in the
        correlation with the marginals unchanged. ``"robust"`` weights each row by
        ``(pi(x) / pi(mode)) ** robust_beta`` in what the run learns and in the
        message it passes on, so a 20-sigma row is atypical under every run, every
        tempered likelihood is about 1, and nothing moves. Without it that one row
        is a changepoint (``p_change`` 0.91), and the run it starts carries the
        outlier in its mean.
    ``robust_beta``
        The tempering under ``"robust"``. A trade: a whole new regime is a run of
        individually forgiven rows, so above about 0.2 nothing is ever detected
        again. The default, 0.1, ignores the outlier and still dates a four-sigma
        shift to the right row.
    ``prior_mean``, ``prior_kappa``, ``prior_nu``, ``prior_scale``
        The conjugate prior. ``prior_mean`` is ``mu_0`` (default zeros);
        ``prior_kappa`` the weight of that mean in rows (default 1.0);
        ``prior_nu`` the degrees of freedom (default ``d + 2``, the smallest that
        gives the Wishart a mean). ``prior_scale`` is the prior scale of the
        variance as a list: one positive number ``[s]`` for ``s`` times the
        identity, or the ``d * d`` entries of a symmetric positive-definite
        matrix, row by row (default: the identity). ``prior_scale`` is the one
        parameter to set from the data: too large and the model goes quiet,
        because no row is ever surprising under a predictive that wide.
        ``prior_nu`` and ``prior_scale`` are ``2a`` and ``2b`` in the gamma
        parametrisation, which is how Adams and MacKay give their own finance
        example (``a = 1``, ``b = 1e-4``, ``hazard = 250``).
    ``prune_below``, ``max_run``
        What keeps the run vector finite: the share of the mass below which a run
        is dropped, and the run length every longer run is folded into.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    ``min_periods`` gates what is reported, never what is learned. A row whose
    predictive cannot be evaluated reports nulls, leaves the posterior where it
    stands, and is counted in :meth:`polars_online.ModelBank.solve_failures`.

    .. rubric:: Output

    One struct column named after the spec, all read before the row updates the
    posterior (`docs/OUTPUTS.md#bocpd
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#bocpd>`_):

    ``p_change``
        ``P(r_t <= 1)`` given this row: the alarm. It is ``P(r <= 1)`` and not
        ``P(r = 0)`` because the changepoint branch and the growth branch share
        the same predictive, which makes the normalised mass at ``r = 0`` exactly
        ``H`` on every row whatever the data. Row one of a group reports nothing:
        ``P(r <= 1)`` is 1 there however the row looks.
    ``run_mode``
        The most likely run length before the row, so ``t - run_mode`` is the row
        the current run began on. This is the answer, and ``p_change`` is the
        alarm; they are not the same quality of signal. ``p_change`` is a per-row
        likelihood ratio, spiky and as big as the break is against the prior
        scale. A tenfold variance step takes it to 0.83 on the row itself; a
        four-sigma mean shift under a diffuse prior barely lifts it; a change in
        correlation alone never moves it. The run length finds all three within a
        few rows and dates them to the right row.
    ``run_mean``
        The posterior mean run length.
    ``pred_<f>``
        The pre-row predictive mean of each feature, mixed over runs.
    ``logscore``
        The row's log predictive density under that mixture.
    ``n_eff``
        As everywhere.

    .. rubric:: Example

    .. code-block:: python

        b = po.spec.bocpd(
            "regime", features=["ret"], group="stock_id",
            hazard=250.0,            # the expected run length
            prior_nu=2.0, prior_scale=[2e-4],   # 2a and 2b: Adams and MacKay's own finance example
            emission="diag",
            prune_below=1e-6,        # what makes the model finite
        )
        out = po.ModelBank([b]).fit_predict(df).unnest("regime")
        # the row each run began on
        began = out.select(pl.int_range(pl.len()) - pl.col("run_mode"))

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, ``ValueError`` for ``halflife`` or ``lam``, and for a
    ``prior_scale`` that is neither ``[s]`` nor ``d * d`` entries.
    """
    model: dict[str, Any] = {
        "type": "bocpd",
        "hazard": hazard,
        "hazard_col": hazard_col,
        "emission": emission,
        "prior_mean": prior_mean,
        "prior_kappa": prior_kappa,
        "prior_nu": prior_nu,
        "prior_scale": prior_scale,
        "robust_beta": robust_beta,
        "prune_below": prune_below,
        "max_run": max_run,
    }
    targets = [hazard_col] if hazard_col is not None else [features[0]]
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def corrchange(
    name: str,
    *,
    features: list[str],
    kind: str = "monitor",
    span_rows: int | None = None,
    alpha: float = 0.05,
    alpha_adjust: str = "bonferroni",
    bandwidth: int | None = None,
    scalar: bool = False,
    crit: float | None = None,
    n_perm: int | None = None,
    permute_every: int | None = None,
    perm_block: int | None = None,
    norm: str = "l1",
    seed: int | None = None,
    reset: bool = False,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Has the correlation structure changed? Two tests, because there are two
    questions.

    ``kind = "monitor"`` asks whether the correlations were constant over a span
    of rows, with a published null. ``kind = "window"`` asks how big the change
    between two adjacent windows is, against a permutation quantile. Not a
    regression: no targets, no decay of anything but the optional standardiser,
    and nothing residual-based applies.

    .. rubric:: The tests

    ``"monitor"`` is the closed-sample constancy test of Wied, Krämer & Dehling
    (2012), run over consecutive spans of ``span_rows`` rows. At the last row of a
    span, per pair:

    .. code-block:: text

        Q = max_{2 <= j <= T} (j / sqrt(T)) * |rho_j - rho_T| / D

    ``rho_j`` is the sample correlation of the span's first ``j`` rows. ``D`` is
    the delta-method long-run standard deviation of ``rho``: the five raw moments
    ``(x², y², x, y, xy)`` centred at their span means, their Bartlett long-run
    covariance at bandwidth ``floor(ln T)``, mapped to ``(var_x, var_y, cov)`` and
    then to ``rho``. Under the null ``Q`` converges to ``sup|B|``, a Brownian
    bridge, so the critical value is the Kolmogorov quantile -- computed from the
    series, not pinned, and it reproduces the published 1.3581 at 5%. Over the
    pairs the statistic is the maximum and the level is ``alpha / npairs``. The
    paper's own sequential form, with a boundary function, is Wied & Galeano
    (2013), which nobody here has read; the closed test run span by span is what
    ships. The cost is a delay of at most ``span_rows`` rows and the benefit a
    null with published tables, which ``tests/test_corrchange.py`` holds it to.

    ``"window"`` is ``norm(vech(R_pre - R_post))`` over two adjacent blocks of
    ``span_rows`` rows -- how big the change is, rather than whether the span was
    constant. ``crit`` is a fixed threshold; without one the critical value is a
    permutation quantile: ``n_perm`` draws of the pooled rows shuffled between the
    two windows, in blocks of ``perm_block`` so that serial dependence does not
    make the null too liberal, redrawn every ``permute_every`` rows. It is not a
    sign-flip null, which a first reading of the literature suggests: negating a
    whole row leaves every ``x x'`` and so every correlation matrix exactly where
    it was, so a sign-flip null has no spread at all. The flag rate per row is not
    ``alpha`` here: two windows that slide by one row are almost the same windows,
    so a statistic above the quantile stays above it for a run of rows.

    .. rubric:: Parameters

    ``kind``
        ``"monitor"`` (the default) or ``"window"``.
    ``span_rows``
        The rows per comparison block; required by both kinds, at least 8 for
        ``"monitor"`` and at least 3 for ``"window"``.
    ``alpha``, ``alpha_adjust``
        The level (default 0.05) and how it is spread over the pairs
        (``"bonferroni"``, the default: ``alpha / npairs``).
    ``bandwidth``
        Overrides the Bartlett bandwidth ``floor(ln T)``.
    ``scalar``
        Run the same CUSUM on the equicorrelation of the standardised row
        (:func:`deco`'s ``u``) instead of every pair: one statistic however many
        columns there are. ``halflife``/``lam`` parametrise that standardiser and
        are accepted only there; neither kind decays anything else, so they are
        refused otherwise.
    ``crit``, ``n_perm``, ``permute_every``, ``perm_block``, ``seed``
        ``"window"``'s threshold, or the permutation quantile in its place:
        ``n_perm`` (default 200) draws, redrawn every ``permute_every`` rows
        (default 50), in blocks of ``perm_block`` rows (default 1); ``seed``
        (default 0) seeds the draws, so two runs with the same seed report the
        same critical values.
    ``norm``
        ``"l1"`` (the default) or ``"linf"`` for the window kind.
    ``reset``
        Empty the windows at a flag (``"window"`` only; ``"monitor"``'s spans are
        disjoint already). Default ``False``.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    A row is always part of the span reported on it: the report comes before the
    update, which is what makes the flag out of sample. So a zero-weight row is
    reported as if it would be learned, and then does not enter the span, does not
    advance ``since_flag`` for the rows after it, and does not reset it if it
    flags.

    .. rubric:: Output

    One struct column named after the spec, all null except where a statistic is
    due (`docs/OUTPUTS.md#corrchange
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#corrchange>`_):

    ``stat``
        The test statistic for the span.
    ``crit``
        The critical value it is compared against.
    ``flag``
        True on the row where ``stat`` crossed ``crit``.
    ``since_flag``
        Learned rows since the last flag.
    ``n_eff``
        As everywhere.

    .. rubric:: Example

    .. code-block:: python

        c = po.spec.corrchange(
            "break", features=["x0", "x1"],
            kind="monitor",      # the constancy test; "window": how big the change is
            span_rows=100,       # nothing is reported until a span closes
        )
        out = po.ModelBank([c]).fit_predict(df).unnest("break")
        due = out.filter(pl.col("stat").is_not_null()).select("t", "stat", "crit", "flag")

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, ``ValueError`` for ``span_rows`` below the kind's minimum, for
    ``halflife``/``lam`` without ``scalar``, and for ``reset`` under
    ``"monitor"``.
    """
    model: dict[str, Any] = {
        "type": "corrchange",
        "kind": kind,
        "span_rows": span_rows,
        "alpha": alpha,
        "alpha_adjust": alpha_adjust,
        "bandwidth": bandwidth,
        "scalar": scalar,
        "crit": crit,
        "n_perm": n_perm,
        "permute_every": permute_every,
        "perm_block": perm_block,
        "norm": norm,
        "seed": seed,
        "reset": reset,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


@_checked
def hmm(
    name: str,
    *,
    features: list[str],
    k: int,
    precision_prior: float,
    covariance: str = "full",
    learn: bool = True,
    transition_prior: float | None = None,
    transition: list[float] | None = None,
    means: list[float] | None = None,
    covs: list[float] | None = None,
    warm_rows: int | None = None,
    seed_rule: str | None = None,
    seed: int | None = None,
    exog_tvtp: str | None = None,
    tvtp_coef: list[list[float]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A Gaussian hidden Markov model, filtered online: which regime are we in?

    :func:`ew_class` classifies a row against labelled Gaussians. This does the
    same arithmetic with no labels: the state is hidden, and a transition matrix
    carries information from one row to the next. That is the difference between
    "which regime does this row look like" and "which regime are we in", and the
    second is usually the question. Not a regression: no targets, and nothing
    residual-based applies.

    .. rubric:: The recursion

    Hamilton's filter, one row at a time. Before the row, from the filtered ``p``
    the previous row left (uniform before the first):

    .. code-block:: text

        p1_l   = sum_k p_k * Pi_kl                     the predicted state
        f_l    = N(x | mu_l, Sigma_l + r_l I)          the state's density
        loglik = log sum_l p1_l f_l                    the row's surprise
        p_l   <- p1_l f_l / sum                        the filtered state

    Everything reported is read before the row is learned from, so an ``hmm``
    output is safe as a feature for that same row. The densities go through the
    path :func:`ew_class` uses, with the same decaying ``precision_prior`` ridge.
    Each state's accumulator then takes the row at weight ``w * p_l``; the
    responsibilities sum to ``w``, so ``n_eff`` is the shared recursion untouched.
    The transition matrix is learned from the filtered joint of consecutive
    states:

    .. code-block:: text

        xi_kl = p_k(t-1) Pi_kl f_l / sum over all pairs
        A_kl <- decay * A_kl + w * xi_kl
        Pi_kl = (A_kl + tau_kl) / sum_l (A_kl + tau_kl)

    with ``tau`` a Dirichlet pseudo-count per cell, which is what keeps a
    never-visited row of ``Pi`` a distribution.

    Two limitations worth knowing. A single extreme row can be captured by one
    state, moving its mean far from the data; in mean form a state with zero
    responsibility keeps its moments, so a state that stops winning never forgets,
    and the mixture is left short one state. A larger ``precision_prior``, given
    states (``learn = False``) or cleaning upstream are the mitigations. And a
    regime that lives only in the covariance needs the covariances to start from.
    The default seeding is k-means over the rows, and zero-mean states differ in
    nothing k-means can see, so it splits the rows by direction and the filter
    never recovers. On streams that stay in one of two zero-mean states, it puts
    57% of the rows in the true state, about half of those after seeding, against
    98% for the same filter given ``covs`` (`docs/REGIMES.md
    <https://github.com/hgilde/polars-online/blob/main/docs/REGIMES.md>`_ §1).
    Pass ``means`` and ``covs``, or a feature in which the regime is a shift in
    location. And keep ``precision_prior`` small against the data's scale: with
    states given and not learned, a ridge of 1.0 on data whose variance is about
    1.0 halves every correlation.

    .. rubric:: Parameters

    ``k``
        The number of hidden states; required.
    ``precision_prior``
        A ridge on every state's covariance, required here as it is for
        :func:`ew_class`, because a state's centred co-moments start at zero and a
        zero matrix has no density.
    ``covariance``
        ``"full"`` (the default), ``"shared"`` or ``"diagonal"``, as for
        :func:`ew_class`.
    ``learn``
        Whether the states move with the rows. ``False`` with no states given is
        refused: there would be nothing to filter with.
    ``transition_prior``, ``transition``
        The Dirichlet pseudo-count per cell (default 1), and a matrix to spread
        that mass over instead of flat, so the given matrix is the prior mean.
    ``means``, ``covs``
        The states given outright (``K x d`` and ``K`` matrices of ``d x d``, both
        flattened row-major), and then there is no warm-up. Each ``covs`` block
        must be symmetric and positive definite, and the pair enters at one row's
        weight, so under ``learn = True`` the stream washes the given states out
        at the ordinary rate and under ``learn = False`` they are held exactly.
    ``warm_rows``, ``seed_rule``, ``seed``
        Without given states, the first ``warm_rows`` learned rows (default 50)
        are buffered, :func:`kmeans`' ``seed_rule`` (``"lloyd"`` by default;
        ``"first"``, ``"farthest"``, ``"kmeanspp"``) chooses centres among them
        with ``seed`` (default 0), and the buffer is replayed through those
        centres as hard assignments; every output is null until then. The buffered
        rows age as ``n_eff`` does, so the states start at the weight ``n_eff``
        says, not at the rows' raw weights. The buffer should span more than one
        regime, or the seeds are two halves of one.
    ``exog_tvtp``, ``tvtp_coef``
        A column (declared like ``weight``, not a feature) whose value drives the
        matrix instead: ``Pi_kl(t) = softmax_l(A_kl + B_kl z_t)`` from the fixed
        ``tvtp_coef = [A, B]``. The count-based learning is off under it; ``A``
        and ``B`` are fitted elsewhere. A row whose ``exog_tvtp`` is null or
        non-finite is filtered with ``z = 0``, the base transition, and is
        otherwise an ordinary row. :meth:`polars_online.ModelBank.predict` reads
        the column too.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    ``min_periods`` gates what is reported, never what is learned: a row below it
    moves the filter and shows nulls. A row whose state densities cannot be
    evaluated is counted in :meth:`polars_online.ModelBank.solve_failures` and
    leaves the filter where it stands.

    .. rubric:: Output

    One struct column named after the spec, all read before the row is learned
    (`docs/OUTPUTS.md#hmm
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#hmm>`_):

    ``p_<j>``, ``p1_<j>``
        The filtered and the predicted probability of each state.
    ``state``
        The most likely state.
    ``loglik``
        The row's surprise: its log-likelihood under the predicted mixture.
    ``n_eff``
        As everywhere.
    ``coef``
        The state means, one row per state, each one entry per feature
        (:func:`coef_index`).

    .. rubric:: Example

    .. code-block:: python

        h = po.spec.hmm(
            "regime", features=["x0", "x1"], halflife=500.0,
            k=2,
            precision_prior=1e-2,    # required: a zero matrix has no density
            warm_rows=100,           # seeds the states from this many rows; null until then
        )
        out = po.ModelBank([h]).fit_predict(df).unnest("regime")
        states = out.select("p_0", "p_1", "state", "loglik").tail(3)

    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, and ``ValueError`` for ``learn = False`` without states, for
    ``means`` or ``covs`` of the wrong shape or a ``covs`` block that is not
    positive definite, and for ``transition`` that is not ``k x k``.
    """
    model: dict[str, Any] = {
        "type": "hmm",
        "k": k,
        "covariance": covariance,
        "precision_prior": precision_prior,
        "learn": learn,
        "transition_prior": transition_prior,
        "transition": transition,
        "means": means,
        "covs": covs,
        "warm_rows": warm_rows,
        "seed_rule": seed_rule,
        "seed": seed,
        "exog_tvtp": exog_tvtp,
        "tvtp_coef": tvtp_coef,
    }
    targets = [exog_tvtp] if exog_tvtp is not None else [features[0]]
    return _common(name, model, targets=targets, features=features, **common)


@_checked
def rcov(
    name: str,
    *,
    features: list[str],
    kind: str = "kernel",
    kernel: str = "parzen",
    bandwidth: int | None = None,
    jitter: int = 2,
    theta: float = 1.0,
    psd: bool = True,
    block_rows: int | None = None,
    max_bandwidth: int | None = None,
    preavg_rows: int | None = None,
    noise_stride: int | None = None,
    iv_stride: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A block's realised covariance, robust to microstructure noise.

    A plain realised covariance over ticks is biased by noise (each price is the
    efficient one plus an error, and the error's variance accumulates with every
    tick) and attenuated by asynchrony. Both are estimated away by published
    estimators that are sums over lags, which is exactly what a stream can
    accumulate. Rows are returns: difference upstream (``.diff().over(by)`` after
    :func:`polars_online.prep.refresh_time`). There is no decay and no per-row
    output but ``n_eff``, because the value is the block: ``rcov`` requires
    ``group`` and ``group_close``, and the estimate rides in the row that close
    emits.

    .. rubric:: The estimators

    ``"plain"`` is ``sum(x x')``, which at close equals ``n`` times an
    ``ew_cov(lam=1)``'s uncentred second moment to the bit -- the cross-check, and
    the reference the other two are measured against.

    ``"kernel"`` (the default) is Barndorff-Nielsen, Hansen, Lunde & Shephard's
    multivariate realised kernel:

    .. code-block:: text

        K = sum_{h=-H}^{H} k(h / (H + 1)) * Gamma_h
        Gamma_h = sum_j x_j x'_{j-h},  Gamma_{-h} = Gamma_h'

    with the Parzen kernel, a positive-definite function, so ``K`` is PSD up to
    rounding, and 0.97 efficient against the quadratic spectral's 0.93. The
    Bartlett kernel is not consistent here and is not offered. The end points are
    jittered by averaging the first and last ``jitter`` observations.

    ``"preavg"`` is Christensen, Kinnebrock & Podolskij's modulated realised
    covariance: the returns are pre-averaged over ``k_n = floor(theta *
    sqrt(block_rows))`` with ``g(x) = min(x, 1 - x)``, which averages the noise
    away, and the residual bias is subtracted.

    .. rubric:: Parameters

    ``kind``, ``kernel``
        ``"plain"``, ``"kernel"`` (the default) or ``"preavg"``; ``kernel`` takes
        only ``"parzen"``.
    ``bandwidth``, ``block_rows``, ``max_bandwidth``
        ``bandwidth`` is a fixed ``H``. Left out, it is their ``H = ceil(c*
        xi^(4/5) n^(3/5))`` with ``c* = 3.5134``, which needs ``block_rows``: the
        ring has to be sized before the first row and ``n`` is known only at the
        close. ``block_rows`` is a sizing hint, not a limit: a longer block runs,
        clipped, and reports ``bandwidth_used``. ``max_bandwidth`` fixes the ring
        depth itself (default ``ceil(c* block_rows^(3/5))`` under the automatic
        bandwidth, the depth at which the noise equals the block's integrated
        variance); it must not cap the ring below a fixed ``bandwidth``.
    ``jitter``
        Observations averaged at each end. Default 2; ``1`` is no jitter, and the
        paper's own ``m = 1..4`` move the estimate by under 0.5%.
    ``theta``, ``psd``, ``preavg_rows``
        ``"preavg"``'s window scale (default 1.0), form and override. ``psd =
        False`` is the balanced, bias-corrected form (optimal rate, not guaranteed
        PSD); ``psd = True`` (the default) is the longer window ``k_n =
        ceil(theta * block_rows^0.6)`` without the bias term, and clips any
        negative eigenvalue, reporting ``psd_repaired``. ``preavg_rows`` fixes
        ``k_n`` (at least 2) instead of deriving it from ``block_rows``.
    ``noise_stride``, ``iv_stride``
        The two subsampled grids behind an automatic bandwidth (defaults 1 and
        20): the noise variance ``omega2`` from the dense one, deliberately biased
        upward as BNHLS accept, and the integrated variance ``iv_sparse`` from the
        sparse one, each averaged over the stride's offsets.

    The stream parameters every builder takes are in :mod:`polars_online.spec`:
    ``clock``, ``halflife``, ``max_dclock``, ``min_periods``, ``group`` and the
    rest.

    ``weight`` is taken only as 0 or 1, since a sum over returns has no fractional
    row, and a zero-weight row advances the clock and enters no ring.
    ``halflife``/``lam`` are refused: the block boundary is ``group_close``'s, not
    a decay's. A gap over ``max_dclock``, or a session change, splits the block
    into stretches: the returns on either side of it are not adjacent, and a
    covariance of adjacent returns is the whole statistic. Each stretch is closed
    as the last one is (leading jitter, interior, trailing jitter), and the lagged
    sums add over stretches, so no product pairs two returns across the break. A
    stretch too short to reach its trailing jitter contributes only what it had
    already emitted, and ``rcov_n`` says how many effective returns there were in
    total. Nothing reads a future row: the jittered end point is formed at close
    from observations already in state.

    .. rubric:: Output

    One struct column named after the spec holding ``n_eff`` alone
    (`docs/OUTPUTS.md#rcov
    <https://github.com/hgilde/polars-online/blob/main/docs/OUTPUTS.md#rcov>`_).
    The estimate is in the row :meth:`polars_online.ModelBank.closed_groups` emits
    when the group closes:

    ``rcov``, ``rcorr``
        The covariance and correlation estimates, as ``vech`` of the upper
        triangle.
    ``rcov_n``, ``rcov_kind``, ``bandwidth_used``
        The effective returns, the estimator, and the ``H`` used.
    ``omega2``, ``iv_sparse``
        The noise variance and the sparse integrated variance behind an automatic
        bandwidth.
    ``iq``
        A realised-quarticity proxy, and labelled one.
    ``psd_repaired``
        Whether a negative eigenvalue was clipped.

    All null for a block too short to estimate from.

    .. rubric:: Example

    .. code-block:: python

        r = po.spec.rcov(
            "rk", features=["x0", "x1"],           # rows are returns
            group="block", group_close="monotone",
            kind="kernel",       # the multivariate realised kernel with Parzen weights
            block_rows=100,      # a sizing hint for the ring; a longer block runs, clipped
        )
        bank = po.ModelBank([r])
        bank.fit_predict(by_block.select("x0", "x1", "block"))
        blocks = bank.closed_groups()    # rcov, rcorr, rcov_n, bandwidth_used, omega2, iv_sparse


    .. rubric:: Raises

    As every builder does (:mod:`polars_online.spec`); ``TypeError`` for
    ``targets``, ``ValueError`` for ``halflife``/``lam``, for a missing ``group``
    or ``group_close``, for an automatic bandwidth without ``block_rows``, and for
    a ``max_bandwidth`` below a fixed ``bandwidth``.
    """
    model: dict[str, Any] = {
        "type": "rcov",
        "kind": kind,
        "kernel": kernel,
        "bandwidth": bandwidth,
        "jitter": jitter,
        "theta": theta,
        "psd": psd,
        "block_rows": block_rows,
        "max_bandwidth": max_bandwidth,
        "preavg_rows": preavg_rows,
        "noise_stride": noise_stride,
        "iv_stride": iv_stride,
    }
    return _common(name, model, targets=[features[0]], features=features, **common)


#: The model types with no target column: their outputs are read from the
#: state before each row, their ``targets`` mirror ``features[0]`` for the
#: plumbing, and nothing residual-based applies to them. ``ew_class`` is
#: not one -- its label column travels as the target -- though it predicts
#: no number either, and refuses the residual switches the same way.
UNSUPERVISED = frozenset(
    {"ew_cov", "kmeans", "micro", "deco", "rcov", "hmm", "corrchange", "bocpd"}
)

_NUMERIC_KEYS = _numeric_keys()
