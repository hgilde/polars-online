"""A seeded simulator for correlation regimes (docs/ENHANCEMENTS.md E64).

The detectors in this library — `deco`, `hmm`, `corrchange`, `bocpd` — are
claims about streams whose correlation structure changes. A claim like that
can only be measured against data whose truth is known, and the truth has to
include the awkward parts: series that report at their own times, levels
observed with additive noise, autocorrelated increments, a scale
that moves with the regime, a periodic pattern, and an activity clock.

:func:`regimes` produces all of it from one seed, and hands back the truth
beside the data:

``rows``
    what a consumer sees: **levels** ``x_1 .. x_m`` (so
    :func:`polars_online.prep.refresh_time` and then ``.diff()`` apply),
    a clock, a cycle index and an optional activity count.
``truth_rows``
    per row: the block, the state, the scale multiplier and the
    interpolation fraction.
``truth_blocks``
    per block: the state, the row count, and the block's true correlation
    matrix as ``vech``.

Everything is drawn from ``numpy.random.default_rng(seed)`` in one order, so
two calls with the same seed give byte-identical frames. numpy only; no
scipy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

__all__ = ["regimes"]


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as e:  # pragma: no cover - exercised by a stub
        msg = (
            "polars_online.sim draws with numpy, and numpy is not installed. "
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
                msg = (
                    f"sim.regimes: state {i} is an equicorrelation of {rho}, which is outside "
                    f"(-1/{m - 1}, 1) and so not a correlation matrix"
                )
                raise ValueError(msg)
            r = np.full((m, m), rho)
            np.fill_diagonal(r, 1.0)
        else:
            r = arr
            if r.shape != (m, m):
                msg = f"sim.regimes: state {i} is {r.shape}, not ({m}, {m})"
                raise ValueError(msg)
            if not np.allclose(np.diag(r), 1.0):
                msg = f"sim.regimes: state {i} has a diagonal that is not 1"
                raise ValueError(msg)
            if not np.allclose(r, r.T):
                msg = f"sim.regimes: state {i} is not symmetric"
                raise ValueError(msg)
            if np.linalg.eigvalsh(r).min() < -1e-10:
                msg = f"sim.regimes: state {i} is not positive semi-definite"
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
        msg = f"sim.regimes: transition is {p.shape}, not ({k}, {k}) for {k} states"
        raise ValueError(msg)
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0):
        msg = "sim.regimes: transition must be row-stochastic (non-negative rows summing to 1)"
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
        msg = f"sim.regimes: durations needs one positive block count per state ({k})"
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
    """A simulated stream of ``m`` series whose correlation changes by regime.

    ``states`` is a list of ``K`` correlation matrices (``m x m``, unit
    diagonal, PSD) or ``K`` floats, each an equicorrelation. ``transition``
    is the ``K x K`` row-stochastic matrix that drives one state per block,
    and there are ``n_blocks`` blocks of ``rows_per_block`` rows.
    ``durations`` (one positive block count per state) makes the sojourn
    deterministic instead and draws the *next* state from ``transition``
    with its diagonal removed — the recurring-state design, where each state
    lasts exactly as long as it says.

    ``design="step"`` switches at the block boundary. ``design="smooth"``
    interpolates the correlation matrix linearly over ``smooth_rows`` rows
    around it; a convex combination of two correlation matrices is one, so
    every matrix along the way is valid.

    The latent returns are ``eps_t ~ N(0, R_t)``, filtered to
    ``y_it = phi_i y_i,t-1 + eps_it`` and scaled by ``exp(scale_state * s_t)``.
    **The documented truth is the innovation correlation**: an AR filter
    moves the *return* correlation of a pair with unequal ``phi``, which is
    exactly what :func:`polars_online.corr.fisher_se`'s inflation is about.
    ``scale_state`` as a scalar makes the noise scale rise with the state index; as
    a list it gives one multiplier exponent per state.

    ``cycle_profile`` is ``cycle_rows`` multipliers in ``(0, 1]`` applied to the
    *off-diagonal* of ``R_t`` at row ``t mod cycle_rows`` — a mix towards
    the identity, so the matrix stays PSD. ``cycle_rows`` defaults to
    ``rows_per_block``, and ``session = t // cycle_rows``.

    ``activity`` is ``(mean, shape)`` for a Gamma count per row, with the
    mean scaled by the same scale multiplier; ``clock`` is then the
    cumulative activity, and the row index otherwise.

    ``x_1 .. x_m`` are **levels**: the cumulative sum of the latent returns
    plus ``noise * N(0, 1)`` per observed row. Noise on the *level* is what
    the literature models, and it makes the observed return an MA(1) with a
    negative first autocorrelation — the microstructure effect `rcov`
    exists to undo. With ``async_rates`` (expected observations per row, per
    series) a row where series ``i`` reported nothing carries ``null`` for
    ``x_i``: last-observation sampling is one ``forward_fill`` away, and
    ``unpivot`` over the non-null rows is
    :func:`polars_online.prep.refresh_time`'s long input.

    Returns ``{"rows", "truth_rows", "truth_blocks"}``. Two calls with the
    same ``seed`` give byte-identical frames.
    """
    np = _np()
    if m < 2:
        msg = f"sim.regimes: at least two series are needed, got {m}"
        raise ValueError(msg)
    if design not in ("step", "smooth"):
        msg = f'sim.regimes: design must be "step" or "smooth", got {design!r}'
        raise ValueError(msg)
    if n_blocks < 1 or rows_per_block < 1:
        msg = "sim.regimes: n_blocks and rows_per_block must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    mats = _states(np, states, m)
    k = len(mats)
    chain = _chain(np, rng, transition, n_blocks, k, durations)
    n = n_blocks * rows_per_block
    cycle_len = cycle_rows if cycle_rows is not None else rows_per_block
    if cycle_len < 1:
        msg = "sim.regimes: cycle_rows must be >= 1"
        raise ValueError(msg)
    if cycle_profile is not None and len(cycle_profile) != cycle_len:
        msg = f"sim.regimes: cycle_profile needs one multiplier per cycle row ({cycle_len})"
        raise ValueError(msg)
    if cycle_profile is not None and not all(0.0 < d <= 1.0 for d in cycle_profile):
        msg = "sim.regimes: cycle_profile multipliers must be in (0, 1]"
        raise ValueError(msg)

    phis = np.asarray(phi, dtype=float)
    if phis.ndim == 0:
        phis = np.full(m, phis)
    if phis.shape != (m,):
        msg = f"sim.regimes: phi must be a scalar or {m} values, got {phis.shape}"
        raise ValueError(msg)
    if (np.abs(phis) >= 1.0).any():
        msg = "sim.regimes: |phi| must be < 1 for a stationary series"
        raise ValueError(msg)
    vol_exp = np.asarray(scale_state, dtype=float)
    if vol_exp.ndim == 0:
        vol_exp = vol_exp * np.arange(k, dtype=float)
    elif vol_exp.shape != (k,):
        msg = f"sim.regimes: scale_state must be a scalar or {k} values"
        raise ValueError(msg)

    block_of = np.arange(n) // rows_per_block
    state_of = np.array([chain[b] for b in block_of])
    # The interpolation fraction: 0 away from a boundary, ramping to 1
    # across `smooth_rows` centred on it.
    mix = np.zeros(n)
    if design == "smooth" and smooth_rows > 0:
        half = smooth_rows / 2.0
        for b in range(1, n_blocks):
            edge = b * rows_per_block
            for t in range(max(0, int(edge - half)), min(n, int(edge + half) + 1)):
                mix[t] = np.clip(0.5 + (t - edge) / smooth_rows, 0.0, 1.0)

    chol: dict[tuple[int, int, int], Any] = {}

    def factor(t: int) -> Any:
        """The Cholesky factor of `R_t`, cached per (state, previous state,
        cycle_profile row) — one factorisation per state under `"step"`."""
        s = int(state_of[t])
        prev = int(state_of[max(0, t - 1)]) if mix[t] > 0.0 else s
        d = t % cycle_len if cycle_profile is not None else 0
        key = (s, prev, d if cycle_profile is not None else 0)
        if mix[t] > 0.0:
            key = (s, prev, d, round(mix[t], 12))  # type: ignore[assignment]
        if key in chol:
            return chol[key]
        r = mats[s]
        if mix[t] > 0.0:
            r = (1.0 - mix[t]) * mats[prev] + mix[t] * mats[s]
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
            msg = "sim.regimes: activity is (mean, shape), both > 0"
            raise ValueError(msg)
        activity_col = rng.gamma(shape, scale=mean / shape, size=n) * vol
        clock = np.cumsum(activity_col)

    observed = levels
    if async_rates is not None:
        rates = np.asarray(async_rates, dtype=float)
        if rates.shape != (m,):
            msg = f"sim.regimes: async_rates needs one expected report rate per series ({m})"
            raise ValueError(msg)
        if (rates <= 0).any():
            msg = "sim.regimes: async_rates must be > 0"
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
            "activity": activity_col if activity_col is not None else [None] * n,
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
        r = np.mean(
            [
                (
                    (1.0 - mix[t]) * mats[int(state_of[max(0, t - 1)])]
                    + mix[t] * mats[int(state_of[t])]
                    if mix[t] > 0.0
                    else mats[int(state_of[t])]
                )
                for t in idx
            ],
            axis=0,
        )
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
