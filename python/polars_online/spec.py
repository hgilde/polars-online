"""Spec builders: ``polars_online.spec.ewridge("m", targets=[...], ...)``.

A spec is a plain dict: JSON-able, and the same thing as a ``[[specs]]``
entry in the CLI's TOML. It names a model, the columns the model reads, and
the stream parameters every model shares. A builder assembles one, checks
it and returns it; :class:`polars_online.ModelBank`, the ``online``
namespaces and :func:`polars_online.run` take lists of them.

The builders are the documented way to write a spec. A dict written by hand
is checked the same way when it is used, but only a builder can tell a
misspelt keyword from a missing one at the call.

**The stream parameters**, shared by every model (the README's "How a bank
sees a stream"; ``docs/PLAN.md`` section 3):

- ``targets``, ``features``: column names. A target is what the model
  predicts; a feature is what it reads from the same row.
- ``add_intercept``: prepend a constant 1 to the features (default ``True``).
- ``clock``: a monotone numeric column; the row count when ``None``.
- ``halflife`` or ``lam``: the decay, in clock units. A list of halflives
  means one fit per value.
- ``max_dclock``: a ceiling on the clock delta. Required with ``clock``;
  ``inf`` for no ceiling.
- ``on_clock_reset``: what a backwards clock means -- ``"max"``, ``"zero"``,
  ``"reset_state"`` or ``"error"``.
- ``session``, ``session_gap``: the clock delta to apply where the session
  column changes.
- ``weight``: a row-weight column.
- ``min_periods``: in ``n_eff`` units; outputs are null until it is reached.
- ``coef_every``: how often to snapshot the coefficients.
- ``label_delay``: hold each row back from learning until the clock has
  moved this much further on.
- ``group``, ``group_close``: one state per key, and when a key is finished.
- the ``emit_*`` switches, ``conformal``, ``resid_quantiles`` and the other
  diagnostics.

Each builder's signature lists them with their defaults, and
:func:`polars_online.spec.output_index` says which struct fields a spec
produces.

**Errors.** A builder raises ``TypeError`` for a name that is not a str, a
keyword it has not got, or a value of the wrong shape, naming the parameter
and what it takes::

    spec "m": halflife must be a number or a list of numbers, got str '10'

It raises ``ValueError``, naming the spec and the parameter, for a value the
model refuses: a count below 0; ``NaN`` anywhere; ``inf`` where it means
nothing (it is allowed where it does -- ``halflife``, ``max_dclock``,
``min_periods``, ``session_gap`` and the model parameters that say so);
neither ``halflife`` nor ``lam``; ``clock`` without ``max_dclock``; a column
listed twice, or as both target and feature; a level outside ``(0, 1)``; an
option not in the list the message gives; and each model's own rules.

A spec that came back from a builder is valid. One edited afterwards is
checked again wherever it is used, and a key no spec has is refused there
rather than ignored.
"""

from polars_online._spec import (
    bocpd,
    coef_fields,
    coef_index,
    corrchange,
    deco,
    ew_class,
    ew_cov,
    ewridge,
    ftrl,
    hmm,
    holt,
    huber,
    kalman,
    kmeans,
    lasso,
    marginal,
    micro,
    output_fields,
    output_index,
    pa,
    quantile,
    rcov,
    rls,
    seqtest,
    sgd,
)

__all__ = [
    "ew_class",
    "ew_cov",
    "ewridge",
    "ftrl",
    "hmm",
    "holt",
    "huber",
    "kalman",
    "kmeans",
    "lasso",
    "bocpd",
    "corrchange",
    "deco",
    "marginal",
    "rcov",
    "micro",
    "output_fields",
    "coef_fields",
    "coef_index",
    "output_index",
    "pa",
    "quantile",
    "rls",
    "seqtest",
    "sgd",
]
