"""A seeded simulator for correlation regimes.

The detectors in this library (``deco``, ``hmm``, ``corrchange``, ``bocpd``)
are claims about streams whose correlation structure changes. A claim like
that can only be measured against data whose truth is known. :func:`regimes`
produces such data from one seed, and hands back the truth beside it. The
truth has to include the awkward parts of real streams, and each is a
parameter:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - the awkward part
     - the parameter
   * - series that report at their own times
     - ``async_rates``
   * - levels observed with additive noise
     - ``noise``
   * - autocorrelated increments
     - ``phi``
   * - a scale that moves with the regime
     - ``scale_state``
   * - a periodic pattern in the correlation
     - ``cycle_profile``, ``cycle_rows``
   * - an activity clock
     - ``activity``

Everything is drawn from ``numpy.random.default_rng(seed)`` in one order, so
two calls with the same seed give byte-identical frames under one numpy
version: numpy's ``Generator`` makes no promise across versions. Requires
numpy, an optional extra of this package (``pip install
polars-online[numpy]``), and nothing else.

.. warning::
    This module is considered **unstable**: its keywords, its frames and the
    draws behind them may change in any release without that counting as a
    breaking change. Under ``POLARS_ONLINE_WARN_UNSTABLE=1`` each call raises
    :class:`polars_online.UnstableWarning`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import polars as pl

from polars_online._warnings import unstable

__all__ = ["regimes"]


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "po.sim: it draws with numpy, but numpy is not installed. "
            "Install it with `pip install numpy` or `pip install polars-online[numpy]`."
        )
        raise ModuleNotFoundError(msg) from e
    return np


def _states(np: Any, states: Sequence[Any], m: int) -> list[Any]:
    """The `K` correlation matrices, from matrices or equicorrelations."""
    out = []
    for i, s in enumerate(states):
        arr = np.asarray(s, dtype=float)
        if arr.ndim == 0:
            rho = float(arr)
            if not -1.0 / (m - 1) < rho < 1.0:
                # The bound to three figures, rounded toward 0, so a value
                # above the figure shown is above the bound.
                low = -1.0 / (m - 1)
                scale = 10.0 ** (2 - math.floor(math.log10(-low)))
                shown = math.ceil(low * scale) / scale
                msg = (
                    f"po.sim.regimes: states[{i}] is an equicorrelation of {rho}, but for {m} "
                    f"series an equicorrelation must be above {shown:.3g} and below 1 to be a "
                    f"correlation matrix. Give states[{i}] a correlation in that range."
                )
                raise ValueError(msg)
            r = np.full((m, m), rho)
            np.fill_diagonal(r, 1.0)
        else:
            r = arr
            if r.shape != (m, m):
                msg = (
                    f"po.sim.regimes: states[{i}] must be a {m} by {m} matrix, one row and column "
                    f"per series, but its shape is {r.shape}. Give a {m} by {m} matrix."
                )
                raise ValueError(msg)
            if not np.allclose(np.diag(r), 1.0):
                msg = (
                    f"po.sim.regimes: states[{i}] must be a correlation matrix, but its diagonal "
                    f"is not all 1. Set its diagonal to 1."
                )
                raise ValueError(msg)
            if not np.allclose(r, r.T):
                msg = (
                    f"po.sim.regimes: states[{i}] must be a correlation matrix, but it is not "
                    f"symmetric. Give a matrix equal to its transpose."
                )
                raise ValueError(msg)
            if np.linalg.eigvalsh(r).min() < -1e-10:
                msg = (
                    f"po.sim.regimes: states[{i}] must be a correlation matrix, but it has a "
                    f"negative eigenvalue. Give a positive semi-definite matrix."
                )
                raise ValueError(msg)
        out.append(r)
    return out


def _chain(
    np: Any,
    rng: Any,
    transition: Any,
    n_blocks: int,
    k: int,
    durations: Sequence[int] | None,
) -> list[int]:
    """The state per block: a Markov chain, or a deterministic sojourn with
    the diagonal removed (the recurring-state design)."""
    p = np.asarray(transition, dtype=float)
    if p.shape != (k, k):
        msg = (
            f"po.sim.regimes: transition must be a {k} by {k} matrix for the {k} entries of "
            f"states, but its shape is {p.shape}. Give one row and one column per state."
        )
        raise ValueError(msg)
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0):
        bad = next(j for j in range(k) if (p[j] < 0).any() or not np.isclose(p[j].sum(), 1.0))
        msg = (
            f"po.sim.regimes: each row of transition must be probabilities, at least 0 and "
            f"summing to 1, but row {bad} is {p[bad].tolist()}. Give rows of probabilities "
            f"that sum to 1."
        )
        raise ValueError(msg)
    # `np.allclose` accepts a row summing to `1 + 1e-6`; `rng.choice` does
    # not, and raises from inside numpy with nothing to say which row. The
    # rows are a valid distribution by the test above, so normalise them and
    # draw from that (docs/REVIEW-E54-E64.md S1).
    p = p / p.sum(axis=1, keepdims=True)
    if durations is None:
        out = [0]
        for _ in range(1, n_blocks):
            out.append(int(rng.choice(k, p=p[out[-1]])))
        return out
    d = list(durations)
    if len(d) != k or any(v < 1 for v in d):
        msg = (
            f"po.sim.regimes: durations must hold one block count of at least 1 for each of "
            f"the {k} states, but it is {d}. Give {k} counts of 1 or more."
        )
        raise ValueError(msg)
    # The sojourn is `durations[s]` blocks; the next state is drawn from the
    # row with its own entry removed, so a state never "transitions" to
    # itself and the durations mean what they say.
    off = p - np.diag(np.diag(p))
    row_sums = off.sum(axis=1)
    seq: list[int] = []
    s = 0
    while len(seq) < n_blocks:
        seq.extend([s] * d[s])
        s = int(rng.choice(k, p=off[s] / row_sums[s])) if row_sums[s] > 0 else s
    return seq[:n_blocks]


@unstable("polars_online.sim.regimes")
def regimes(
    m: int,
    *,
    states: Sequence[Any],
    transition: Any,
    n_blocks: int,
    rows_per_block: int,
    durations: Sequence[int] | None = None,
    design: str = "step",
    smooth_rows: int = 0,
    phi: float | Sequence[float] = 0.0,
    scale_state: float | Sequence[float] = 0.0,
    async_rates: Sequence[float] | None = None,
    noise: float = 0.0,
    cycle_profile: Sequence[float] | None = None,
    cycle_rows: int | None = None,
    activity: tuple[float, float] | None = None,
    seed: int = 0,
) -> dict[str, pl.DataFrame]:
    """A simulated stream of ``m`` series whose correlation changes by regime, with
    its truth.

    The latent returns are ``eps_t ~ N(0, R_t)``, filtered to ``y_it = phi_i
    y_i,t-1 + eps_it`` and scaled by ``exp(scale_state * s_t)``. The observed
    series are their levels, the cumulative sums, plus noise. The documented truth
    is the innovation correlation: an AR filter moves the return correlation of a
    pair with unequal ``phi``, which is exactly what
    :func:`polars_online.corr.fisher_se`'s inflation is about.

    .. rubric:: Parameters

    ``m``
        The number of series, at least 2.
    ``states``
        ``K`` correlation matrices (``m x m``, unit diagonal, PSD) or ``K``
        floats, each an equicorrelation.
    ``transition``
        The ``K x K`` row-stochastic matrix that drives one state per block.
    ``n_blocks``, ``rows_per_block``
        The blocks and the rows in each.
    ``durations``
        One positive block count per state: the sojourn becomes deterministic, and
        the next state is drawn from ``transition`` with its diagonal removed --
        the recurring-state design, where each state lasts exactly as long as it
        says.
    ``design``, ``smooth_rows``
        ``"step"`` switches at the block boundary; ``"smooth"`` interpolates the
        correlation matrix linearly over ``smooth_rows`` rows centred on it:
        row ``t`` follows ``(1 - mix) R_before + mix R_after``, the states of
        the blocks before and after the boundary, on both sides of it. A convex
        combination of two correlation matrices is one, so every matrix along the
        way is valid.
    ``phi``
        The AR(1) coefficient per series (a scalar for all), ``|phi| < 1``.
    ``scale_state``
        The returns' scale per state, as the exponent of its multiplier
        ``exp(scale_state * s_t)``. A scalar makes the scale rise with the
        state index; a list gives one exponent per state.
    ``async_rates``
        Expected observations per row, per series: a row where series ``i``
        reported nothing carries ``null`` for ``x_i``. Last-observation sampling
        is one ``forward_fill`` away, and ``unpivot`` over the non-null rows is
        :func:`polars_online.stream.refresh_time`'s long input.
    ``noise``
        The standard deviation of the noise added to each observed level. Noise on
        the level is what the literature models, and it makes the observed return
        an MA(1) with a negative first autocorrelation, the microstructure effect
        ``rcov`` exists to undo.
    ``cycle_profile``, ``cycle_rows``
        ``cycle_rows`` multipliers in ``(0, 1]`` applied to the off-diagonal of
        ``R_t`` at row ``t mod cycle_rows``, a mix towards the identity, so the
        matrix stays PSD. ``cycle_rows`` defaults to ``rows_per_block``, and
        ``session = t // cycle_rows``.
    ``activity``
        ``(mean, shape)`` for a Gamma count per row, with the mean scaled by the
        same scale multiplier; ``clock`` is then the cumulative activity, and the
        row index otherwise.
    ``seed``
        Two calls with the same seed give byte-identical frames under one numpy
        version; numpy's ``Generator`` makes no promise across versions.

    Returns ``{"rows", "truth_rows", "truth_blocks"}``:

    ``rows``
        What a consumer sees: ``entity``, ``t``, ``clock``, ``session``, the
        levels ``x_1 .. x_m`` (null where a series did not report), and
        ``activity`` (null without ``activity``).
    ``truth_rows``
        Per row: ``t``, ``block``, ``state``, ``scale_mult`` and ``mix``, the
        weight of the later block's state in the row's blend: 0 away from a
        boundary, rising to 1 across it.
    ``truth_blocks``
        Per block: ``block``, ``state``, ``n_rows``, and ``corr``, the block's
        true correlation matrix as ``vech``, its upper triangle with the
        diagonal, row by row. That matrix is the mean over the block's rows of
        the states' matrices, before ``cycle_profile`` mixes them toward the
        identity.

    .. code-block:: python

        sim = po.sim.regimes(
            3, states=[0.2, 0.7], transition=[[0.9, 0.1], [0.1, 0.9]],
            n_blocks=4, rows_per_block=50, phi=0.3, seed=1,
        )
        rows, truth = sim["rows"], sim["truth_blocks"]

    Raises ``ValueError`` for:

    - fewer than two series, or ``n_blocks`` or ``rows_per_block`` below 1;
    - a state that is not a correlation matrix;
    - a ``transition`` that is not row-stochastic or not ``K x K``, and
      ``durations`` that do not give one positive count per state;
    - a ``design`` that is neither;
    - a ``phi`` outside ``(-1, 1)``, or a ``phi`` or ``scale_state`` list of
      the wrong length;
    - a ``cycle_rows`` below 1, and a ``cycle_profile`` of the wrong length
      or outside ``(0, 1]``;
    - an ``activity`` that is not two positive numbers, and ``async_rates``
      that are not ``m`` positive rates.
    """
    np = _np()
    if m < 2:
        msg = f"po.sim.regimes: m must be at least 2, but it is {m}. Set m to 2 or more series."
        raise ValueError(msg)
    if design not in ("step", "smooth"):
        msg = (
            f'po.sim.regimes: design must be "step" or "smooth", but it is {design!r}. '
            f"Set design to one of those words."
        )
        raise ValueError(msg)
    if n_blocks < 1 or rows_per_block < 1:
        msg = (
            f"po.sim.regimes: n_blocks and rows_per_block must each be at least 1, but they are "
            f"{n_blocks} and {rows_per_block}. Set both to 1 or more."
        )
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    mats = _states(np, states, m)
    k = len(mats)
    chain = _chain(np, rng, transition, n_blocks, k, durations)
    n = n_blocks * rows_per_block
    cycle_len = cycle_rows if cycle_rows is not None else rows_per_block
    if cycle_len < 1:
        msg = (
            f"po.sim.regimes: cycle_rows must be at least 1, but it is {cycle_len}. "
            f"Set cycle_rows to 1 or more."
        )
        raise ValueError(msg)
    if cycle_profile is not None and len(cycle_profile) != cycle_len:
        msg = (
            f"po.sim.regimes: cycle_profile must hold one multiplier per row of the cycle, "
            f"{cycle_len}, but it holds {len(cycle_profile)}. Give {cycle_len} multipliers."
        )
        raise ValueError(msg)
    if cycle_profile is not None and not all(0.0 < d <= 1.0 for d in cycle_profile):
        bad = next(j for j, d in enumerate(cycle_profile) if not 0.0 < d <= 1.0)
        msg = (
            f"po.sim.regimes: each multiplier of cycle_profile must be above 0 and at most 1, "
            f"but multiplier {bad} is {cycle_profile[bad]}. Give multipliers in that range."
        )
        raise ValueError(msg)

    phis = np.asarray(phi, dtype=float)
    if phis.ndim == 0:
        phis = np.full(m, phis)
    if phis.shape != (m,):
        msg = (
            f"po.sim.regimes: phi must be one number or {m} values, one per series, but its "
            f"shape is {phis.shape}. Give one number or {m}."
        )
        raise ValueError(msg)
    if (np.abs(phis) >= 1.0).any():
        bad = int(np.argmax(np.abs(phis) >= 1.0))
        msg = (
            f"po.sim.regimes: each phi must be above -1 and below 1 for a stationary series, "
            f"but series {bad}'s is {phis[bad]}. Give values in that range."
        )
        raise ValueError(msg)
    vol_exp = np.asarray(scale_state, dtype=float)
    if vol_exp.ndim == 0:
        vol_exp = vol_exp * np.arange(k, dtype=float)
    elif vol_exp.shape != (k,):
        msg = (
            f"po.sim.regimes: scale_state must be one number or {k} values, one per state, but "
            f"its shape is {vol_exp.shape}. Give one number or {k}."
        )
        raise ValueError(msg)

    block_of = np.arange(n) // rows_per_block
    state_of = np.array([chain[b] for b in block_of])
    # The interpolation fraction: 0 away from a boundary, ramping to 1
    # across `smooth_rows` centred on it, and the two states it blends: the
    # block before the boundary's (`before`) and the block after it
    # (`after`). Row `t` follows `(1 - mix) R_before + mix R_after`, on
    # either side of the boundary, so `truth_rows`' `mix` is what the data
    # follow. A row away from every boundary follows its own block's state.
    mix = np.zeros(n)
    before = state_of.copy()
    after = state_of.copy()
    if design == "smooth" and smooth_rows > 0:
        half = smooth_rows / 2.0
        for b in range(1, n_blocks):
            edge = b * rows_per_block
            for t in range(max(0, int(edge - half)), min(n, int(edge + half) + 1)):
                mix[t] = np.clip(0.5 + (t - edge) / smooth_rows, 0.0, 1.0)
                before[t], after[t] = chain[b - 1], chain[b]

    def corr_at(t: int) -> Any:
        """`R_t` before `cycle_profile`: its block's state, or the blend of
        the two blocks' across a boundary."""
        if mix[t] > 0.0:
            return (1.0 - mix[t]) * mats[int(before[t])] + mix[t] * mats[int(after[t])]
        return mats[int(before[t])]

    chol: dict[tuple[Any, ...], Any] = {}

    def factor(t: int) -> Any:
        """The Cholesky factor of `R_t`, cached per (state, cycle_profile
        row), and per blend across a boundary -- one factorisation per state
        under `"step"`."""
        d = t % cycle_len if cycle_profile is not None else 0
        key: tuple[Any, ...] = (int(before[t]), int(after[t]), d)
        if mix[t] > 0.0:
            key = (*key, round(float(mix[t]), 12))
        if key in chol:
            return chol[key]
        r = corr_at(t)
        if cycle_profile is not None:
            scale = float(cycle_profile[d])
            r = scale * r + (1.0 - scale) * np.eye(m)
        # A correlation matrix on the boundary of the cone needs a nudge for
        # Cholesky; the jitter is at rounding scale.
        try:
            f = np.linalg.cholesky(r)
        except np.linalg.LinAlgError:
            f = np.linalg.cholesky(r + 1e-12 * np.eye(m))
        chol[key] = f
        return f

    eps = np.empty((n, m))
    z = rng.standard_normal((n, m))
    for t in range(n):
        eps[t] = factor(t) @ z[t]
    vol = np.exp(vol_exp[state_of])
    y = np.empty((n, m))
    prev_row = np.zeros(m)
    for t in range(n):
        prev_row = phis * prev_row + eps[t]
        y[t] = prev_row * vol[t]
    levels = np.cumsum(y, axis=0)
    if noise > 0.0:
        levels = levels + noise * rng.standard_normal((n, m))

    activity_col = None
    clock = np.arange(n, dtype=float)
    if activity is not None:
        mean, shape = activity
        if mean <= 0.0 or shape <= 0.0:
            msg = (
                f"po.sim.regimes: activity must be (mean, shape), both above 0, but it is "
                f"{activity!r}. Give two positive numbers."
            )
            raise ValueError(msg)
        activity_col = rng.gamma(shape, scale=mean / shape, size=n) * vol
        clock = np.cumsum(activity_col)

    observed = levels
    if async_rates is not None:
        rates = np.asarray(async_rates, dtype=float)
        if rates.shape != (m,):
            msg = (
                f"po.sim.regimes: async_rates must hold one expected report rate per series, "
                f"{m}, but its shape is {rates.shape}. Give {m} rates."
            )
            raise ValueError(msg)
        if (rates <= 0).any():
            bad = int(np.argmax(rates <= 0))
            msg = (
                f"po.sim.regimes: each rate of async_rates must be above 0, but series {bad}'s "
                f"is {rates[bad]}. Give positive rates."
            )
            raise ValueError(msg)
        reports = rng.poisson(rates, size=(n, m))
        observed = np.where(reports > 0, levels, np.nan)

    rows = pl.DataFrame(
        {
            "entity": ["sim"] * n,
            "t": np.arange(n, dtype=np.int64),
            "clock": clock,
            "session": (np.arange(n) // cycle_len).astype(np.int64),
            **{f"x_{i + 1}": pl.Series(observed[:, i]).fill_nan(None) for i in range(m)},
            # A float column whether or not `activity` was asked for, so the
            # schema does not depend on the call (review round 4, YB14).
            "activity": pl.Series(
                activity_col if activity_col is not None else [None] * n, dtype=pl.Float64
            ),
        }
    )
    truth_rows = pl.DataFrame(
        {
            "t": np.arange(n, dtype=np.int64),
            "block": block_of.astype(np.int64),
            "state": state_of.astype(np.int64),
            "scale_mult": vol,
            "mix": mix,
        }
    )
    iu = np.triu_indices(m)
    blocks = []
    for b in range(n_blocks):
        idx = np.flatnonzero(block_of == b)
        r = np.mean([corr_at(t) for t in idx], axis=0)
        blocks.append(
            {
                "block": b,
                "state": int(chain[b]),
                "n_rows": len(idx),
                "corr": r[iu].tolist(),
            }
        )
    truth_blocks = pl.DataFrame(
        blocks,
        schema={
            "block": pl.Int64,
            "state": pl.Int64,
            "n_rows": pl.Int64,
            "corr": pl.List(pl.Float64),
        },
    )
    return {"rows": rows, "truth_rows": truth_rows, "truth_blocks": truth_blocks}
