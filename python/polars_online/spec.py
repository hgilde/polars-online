"""Spec builders: ``polars_online.spec.ewridge("m", targets=[...], ...)``.

A spec is a plain dict -- JSON-able, the same thing a ``[[specs]]`` entry in
the CLI's TOML is -- naming a model, the columns it reads and the stream
parameters every model shares. A builder assembles one, checks it, and
returns it; :class:`polars_online.ModelBank`, the ``online`` namespaces and
:func:`polars_online.run` take lists of them. The builders are the
documented way to write one: a dict written by hand is checked the same way
when it is used, but only a builder can tell a misspelt keyword from a
missing one at the call.

Every model takes the stream parameters (the README's "Common parameters",
docs/PLAN.md section 3): ``targets`` and ``features`` (column names; a
target is what the model predicts, a feature what it reads from the same
row), ``add_intercept``, ``clock`` (a monotone numeric column; row count
when ``None``), ``halflife`` or ``lam`` (the decay, in clock units; a list
of halflives means one fit per value), ``max_dclock`` (a ceiling on the
clock delta, required with ``clock``; ``inf`` for none), ``on_clock_reset``
(``"max"``, ``"zero"``, ``"reset_state"`` or ``"error"``), ``session`` and
``session_gap``, ``weight``, ``min_periods`` (in ``n_eff`` units; outputs
are null until it is reached), ``coef_every``, the ``emit_*`` switches, and
``group`` (one state per key). The builder's signature lists them with their
defaults; ``polars_online.spec.output_index`` says what struct fields a spec
produces.

A builder raises ``TypeError`` for a name that is not a str, a keyword it
has not got, or a value of the wrong shape -- naming the parameter and what
it takes (``spec "m": halflife must be a number or a list of numbers, got
str '10'``) -- and ``ValueError``, naming the spec and the parameter, for a
value the model refuses: a count below 0, ``NaN`` anywhere, ``inf`` where
it means nothing (it is allowed where it does: ``halflife``, ``max_dclock``,
``min_periods``, ``session_gap`` and a few model parameters that say so),
neither ``halflife`` nor ``lam``, ``clock`` without ``max_dclock``, a column
listed twice or as both target and feature, a level outside ``(0, 1)``, an
option not in the list the message gives, and each model's own rules. A
spec that came back from a builder is valid; one edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored.
"""

from polars_online._spec import (
    coef_fields,
    coef_index,
    ew_cov,
    ewridge,
    ftrl,
    holt,
    huber,
    kalman,
    kmeans,
    lasso,
    output_fields,
    output_index,
    pa,
    quantile,
    rls,
    sgd,
)

__all__ = [
    "ew_cov",
    "ewridge",
    "ftrl",
    "holt",
    "huber",
    "kalman",
    "kmeans",
    "lasso",
    "output_fields",
    "coef_fields",
    "coef_index",
    "output_index",
    "pa",
    "quantile",
    "rls",
    "sgd",
]
