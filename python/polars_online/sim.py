"""A seeded simulator for correlation regimes (docs/ENHANCEMENTS.md E64).

The detectors in this library — `deco`, `hmm`, `corrchange`, `bocpd` — are
claims about streams whose correlation structure changes. A claim like that
can only be measured against data whose truth is known, and the truth has to
include the awkward parts: series that tick at their own times, prices
observed with microstructure noise, autocorrelated returns, a volatility
that moves with the regime, an intraday pattern, and a volume clock.

:func:`regimes` produces all of it from one seed, and hands back the truth
beside the data:

``bars``
    what a consumer sees: **levels** ``x_1 .. x_m`` (so
    :func:`polars_online.prep.refresh_time` and then ``.diff()`` apply),
    a clock, a session and an optional volume.
``truth_rows``
    per bar: the block, the state, the volatility multiplier and the
    interpolation fraction.
``truth_blocks``
    per block: the state, the bar count, and the block's true correlation
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
    rows = off.sum(axis=1)
    seq: list[int] = []
    s = 0
    while len(seq) < n_blocks:
        seq.extend([s] * d[s])
        s = int(rng.choice(k, p=off[s] / rows[s])) if rows[s] > 0 else s
    return seq[:n_blocks]


def regimes(
    m: int,
    *,
    states: Sequence[Any],
    transition: Any,
    n_blocks: int,
    bars_per_block: int,
    durations: Sequence[int] | None = None,
    design: str = "step",
    smooth_bars: int = 0,
    phi: float | Sequence[float] = 0.0,
    vol_state: float | Sequence[float] = 0.0,
    async_rates: Sequence[float] | None = None,
    noise: float = 0.0,
    diurnal: Sequence[float] | None = None,
    session_bars: int | None = None,
    volume: tuple[float, float] | None = None,
    seed: int = 0,
) -> dict[str, pl.DataFrame]:
    """A simulated stream of ``m`` series whose correlation changes by regime.

    ``states`` is a list of ``K`` correlation matrices (``m x m``, unit
    diagonal, PSD) or ``K`` floats, each an equicorrelation. ``transition``
    is the ``K x K`` row-stochastic matrix that drives one state per block,
    and there are ``n_blocks`` blocks of ``bars_per_block`` bars.
    ``durations`` (one positive block count per state) makes the sojourn
    deterministic instead and draws the *next* state from ``transition``
    with its diagonal removed — the recurring-state design, where each state
    lasts exactly as long as it says.

    ``design="step"`` switches at the block boundary. ``design="smooth"``
    interpolates the correlation matrix linearly over ``smooth_bars`` bars
    around it; a convex combination of two correlation matrices is one, so
    every matrix along the way is valid.

    The latent returns are ``eps_t ~ N(0, R_t)``, filtered to
    ``y_it = phi_i y_i,t-1 + eps_it`` and scaled by ``exp(vol_state * s_t)``.
    **The documented truth is the innovation correlation**: an AR filter
    moves the *return* correlation of a pair with unequal ``phi``, which is
    exactly what :func:`polars_online.corr.fisher_se`'s inflation is about.
    ``vol_state`` as a scalar makes volatility rise with the state index; as
    a list it gives one multiplier exponent per state.

    ``diurnal`` is ``session_bars`` multipliers in ``(0, 1]`` applied to the
    *off-diagonal* of ``R_t`` at bar ``t mod session_bars`` — a mix towards
    the identity, so the matrix stays PSD. ``session_bars`` defaults to
    ``bars_per_block``, and ``session = t // session_bars``.

    ``volume`` is ``(mean, shape)`` for a Gamma volume per bar, with the
    mean scaled by the same volatility multiplier; ``clock`` is then the
    cumulative volume, and the bar index otherwise.

    ``x_1 .. x_m`` are **levels**: the cumulative sum of the latent returns
    plus ``noise * N(0, 1)`` per observed bar. Noise on the *level* is what
    the literature models, and it makes the observed return an MA(1) with a
    negative first autocorrelation — the microstructure effect `rcov`
    exists to undo. With ``async_rates`` (expected ticks per bar, per
    series) a bar where series ``i`` drew no tick carries ``null`` for
    ``x_i``: previous-tick sampling is one ``forward_fill`` away, and
    ``unpivot`` over the non-null rows is
    :func:`polars_online.prep.refresh_time`'s long input.

    Returns ``{"bars", "truth_rows", "truth_blocks"}``. Two calls with the
    same ``seed`` give byte-identical frames.
    """
    np = _np()
    if m < 2:
        msg = f"sim.regimes: at least two series are needed, got {m}"
        raise ValueError(msg)
    if design not in ("step", "smooth"):
        msg = f'sim.regimes: design must be "step" or "smooth", got {design!r}'
        raise ValueError(msg)
    if n_blocks < 1 or bars_per_block < 1:
        msg = "sim.regimes: n_blocks and bars_per_block must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    mats = _states(np, states, m)
    k = len(mats)
    chain = _chain(np, rng, transition, n_blocks, k, durations)
    n = n_blocks * bars_per_block
    sess_bars = session_bars if session_bars is not None else bars_per_block
    if sess_bars < 1:
        msg = "sim.regimes: session_bars must be >= 1"
        raise ValueError(msg)
    if diurnal is not None and len(diurnal) != sess_bars:
        msg = f"sim.regimes: diurnal needs one multiplier per session bar ({sess_bars})"
        raise ValueError(msg)
    if diurnal is not None and not all(0.0 < d <= 1.0 for d in diurnal):
        msg = "sim.regimes: diurnal multipliers must be in (0, 1]"
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
    vol_exp = np.asarray(vol_state, dtype=float)
    if vol_exp.ndim == 0:
        vol_exp = vol_exp * np.arange(k, dtype=float)
    elif vol_exp.shape != (k,):
        msg = f"sim.regimes: vol_state must be a scalar or {k} values"
        raise ValueError(msg)

    block_of = np.arange(n) // bars_per_block
    state_of = np.array([chain[b] for b in block_of])
    # The interpolation fraction: 0 away from a boundary, ramping to 1
    # across `smooth_bars` centred on it.
    mix = np.zeros(n)
    if design == "smooth" and smooth_bars > 0:
        half = smooth_bars / 2.0
        for b in range(1, n_blocks):
            edge = b * bars_per_block
            for t in range(max(0, int(edge - half)), min(n, int(edge + half) + 1)):
                mix[t] = np.clip(0.5 + (t - edge) / smooth_bars, 0.0, 1.0)

    chol: dict[tuple[int, int, int], Any] = {}

    def factor(t: int) -> Any:
        """The Cholesky factor of `R_t`, cached per (state, previous state,
        diurnal bar) — one factorisation per state under `"step"`."""
        s = int(state_of[t])
        prev = int(state_of[max(0, t - 1)]) if mix[t] > 0.0 else s
        d = t % sess_bars if diurnal is not None else 0
        key = (s, prev, d if diurnal is not None else 0)
        if mix[t] > 0.0:
            key = (s, prev, d, round(mix[t], 12))  # type: ignore[assignment]
        if key in chol:
            return chol[key]
        r = mats[s]
        if mix[t] > 0.0:
            r = (1.0 - mix[t]) * mats[prev] + mix[t] * mats[s]
        if diurnal is not None:
            scale = float(diurnal[d])
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

    vol_col = None
    clock = np.arange(n, dtype=float)
    if volume is not None:
        mean, shape = volume
        if mean <= 0.0 or shape <= 0.0:
            msg = "sim.regimes: volume is (mean, shape), both > 0"
            raise ValueError(msg)
        vol_col = rng.gamma(shape, scale=mean / shape, size=n) * vol
        clock = np.cumsum(vol_col)

    observed = levels
    if async_rates is not None:
        rates = np.asarray(async_rates, dtype=float)
        if rates.shape != (m,):
            msg = f"sim.regimes: async_rates needs one expected tick rate per series ({m})"
            raise ValueError(msg)
        if (rates <= 0).any():
            msg = "sim.regimes: async_rates must be > 0"
            raise ValueError(msg)
        ticks = rng.poisson(rates, size=(n, m))
        observed = np.where(ticks > 0, levels, np.nan)

    bars = pl.DataFrame(
        {
            "instrument": ["sim"] * n,
            "t": np.arange(n, dtype=np.int64),
            "clock": clock,
            "session": (np.arange(n) // sess_bars).astype(np.int64),
            **{f"x_{i + 1}": pl.Series(observed[:, i]).fill_nan(None) for i in range(m)},
            "volume": vol_col if vol_col is not None else [None] * n,
        }
    )
    truth_rows = pl.DataFrame(
        {
            "t": np.arange(n, dtype=np.int64),
            "block": block_of.astype(np.int64),
            "state": state_of.astype(np.int64),
            "vol_mult": vol,
            "mix": mix,
        }
    )
    iu = np.triu_indices(m)
    blocks = []
    for b in range(n_blocks):
        rows = np.flatnonzero(block_of == b)
        r = np.mean(
            [
                (
                    (1.0 - mix[t]) * mats[int(state_of[max(0, t - 1)])]
                    + mix[t] * mats[int(state_of[t])]
                    if mix[t] > 0.0
                    else mats[int(state_of[t])]
                )
                for t in rows
            ],
            axis=0,
        )
        blocks.append(
            {
                "block": b,
                "state": int(chain[b]),
                "n_bars": len(rows),
                "corr": r[iu].tolist(),
            }
        )
    truth_blocks = pl.DataFrame(
        blocks,
        schema={
            "block": pl.Int64,
            "state": pl.Int64,
            "n_bars": pl.Int64,
            "corr": pl.List(pl.Float64),
        },
    )
    return {"bars": bars, "truth_rows": truth_rows, "truth_blocks": truth_blocks}
