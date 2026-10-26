"""Class 9 of task 278: long-run stability, the measurements.

Each case streams ``rows`` rows through a bank in chunks, the test itself
O(chunk) in memory, and at checkpoints spaced geometrically compares what
the bank reads against a recomputation from scratch of the same quantity in
exact (``fractions``) or compensated (``math.fsum``) arithmetic. A case
returns the relative error at each checkpoint, so a test reads both the
worst error and its *rate*: an error that grows with the rows is drift, the
finding this class exists for, where rounding that does not accumulate stays
flat.

The cases, by model (the table :data:`CASES` at the bottom, which the shared
convergence registry folds in: each entry is a model, a case name and a
function of the row count):

- ``ew_cov``: NIST StRD NumAcc4 cycled (certified mean and standard
  deviation at a level of 1e7 with a spread of 0.1); a decayed stream at a
  level of 1e6, against the weighted moments of the last 64 half-lives; the
  same under a ``window_size``, against the window's rows (the running
  products of the per-row decay factors, task 275).
- ``ewridge``: no decay, against the least-squares fit of every row so far
  (cross-products accumulated with ``math.fsum``, solved by NumPy); decayed,
  against statsmodels' WLS on the last 64 half-lives at their weights.
- ``kalman``: ``P`` read from the state at each checkpoint, positive definite
  (numpy's eigenvalues) and within a band of its value once settled.
- the clock: ``weight_sum`` against its closed-form steady state
  ``1 / (1 - 2^(-d/h))`` on a float clock at an epoch-sized origin, and on a
  ``Datetime`` clock in nanoseconds.

The 64 half-lives: a row older than that carries ``2^-64`` of the newest
row's weight, below a double's rounding of the sum, so the truncated sum
is the full one to the precision compared.

Every case takes ``oracle``. With ``oracle=False`` it feeds the same rows to
the same checkpoints and returns what the bank read there, ``(rows, ...)``,
without the recomputation: the outputs the oracle ledger's hash guard holds
(``tests/ledger.py``), in milliseconds at a few thousand rows, where the
recomputation from scratch is the slow half. With ``oracle=True``, the
default and the soak's, it returns the errors as before, from the same
reads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from fractions import Fraction

import numpy as np
import polars as pl
import statsmodels.api as sm

import polars_online as po

CHUNK = 200_000


def checkpoints(rows: int, first: int = 1_000) -> list[int]:
    """Rows at which a case compares: 1, 2 and 5 times each power of ten from
    ``first`` up to ``rows``, and ``rows`` itself."""
    out = []
    p = first
    while p <= rows:
        for m in (1, 2, 5):
            if m * p <= rows:
                out.append(m * p)
        p *= 10
    if out[-1] != rows:
        out.append(rows)
    return out


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want)


# ---------------------------------------------------------------------------
# ew_cov: NumAcc4 cycled


def numacc_cycled(values: np.ndarray, rows: int, oracle: bool = True) -> list:
    """``ew_cov`` at ``half_life = inf`` over ``values`` repeated to ``rows``
    rows. At each checkpoint (a whole number of cycles) the mean and the
    sample standard deviation (``bias=False``, Bessel's ``n - 1``) against
    the exact moments of the doubles fed, computed in fractions: the mean of
    a whole number of cycles is the cycle's, and its sum of squared
    deviations is the cycle's times the cycles. Returns ``(rows, mean's
    relative error, std's relative error)``."""
    n0 = len(values)
    if oracle:
        exact = [Fraction(v) for v in values]
        mean = sum(exact) / n0
        ss = sum((e - mean) ** 2 for e in exact)
    spec = po.spec.ew_cov("c", features=["x"], half_life=float("inf"), stats=["mean", "std"])
    bank = po.ModelBank([spec])
    per_chunk = max(1, CHUNK // n0)
    block = np.tile(values, per_chunk)
    marks = sorted({max(1, c // n0) * n0 for c in checkpoints(rows, first=10 * n0)})
    out = []
    fed = 0
    for mark in marks:
        while fed < mark:
            take = min(len(block), mark - fed)
            bank.fit_predict(pl.DataFrame({"x": block[:take]}))
            fed += take
        # The fields are read before the row: the next row reads all `fed`.
        probe = bank.predict(pl.DataFrame({"x": values[:1]})).unnest("c")
        m_got, s_got = probe["mean_x"][0], probe["std_x"][0]
        if not oracle:
            out.append((fed, m_got, s_got))
            continue
        cycles = fed // n0
        sd = math.sqrt(ss * cycles / (fed - 1))
        out.append((fed, float(abs(Fraction(m_got) - mean) / mean), _rel(s_got, sd)))
    return out


# ---------------------------------------------------------------------------
# A decayed stream with an irregular clock, chunk by chunk


def _stream(
    rows: int, seed: int, k: int, level: float, beta: np.ndarray | None
) -> Iterator[pl.DataFrame]:
    """An irregular clock (exponential steps of mean 1), ``k`` features at
    ``level`` plus unit noise, and, when ``beta`` is given, a target
    ``level + x . beta + noise``."""
    rng = np.random.default_rng(seed)
    t0 = 0.0
    for start in range(0, rows, CHUNK):
        n = min(CHUNK, rows - start)
        t = t0 + np.cumsum(rng.exponential(1.0, n))
        t0 = float(t[-1])
        x = level + rng.standard_normal((n, k))
        cols: dict[str, np.ndarray] = {"t": t}
        for j in range(k):
            cols[f"x{j}"] = x[:, j]
        if beta is not None:
            cols["y"] = level + (x - level) @ beta + 0.5 * rng.standard_normal(n)
        yield pl.DataFrame(cols)


class _Tail:
    """The last ``keep`` rows of a stream, for a recomputation from scratch."""

    def __init__(self, keep: int):
        self.keep = keep
        self.frame: pl.DataFrame | None = None

    def push(self, chunk: pl.DataFrame) -> None:
        both = chunk if self.frame is None else pl.concat([self.frame, chunk])
        self.frame = both.tail(self.keep)


def _decayed_weights(t: np.ndarray, now: float, half_life: float) -> np.ndarray:
    return np.exp2(-(now - t) / half_life)


def _feed_to_checkpoints(
    bank: po.ModelBank, stream: Iterator[pl.DataFrame], rows: int, keep: int, read: Callable
) -> list:
    """Feed ``stream`` and call ``read(fed, tail_frame, next_row)`` once the
    bank has learned each checkpoint's rows, ``tail_frame`` the last ``keep``
    of them and ``next_row`` the row after, whose fields the state of the rows
    before it gives."""
    marks = set(checkpoints(rows - 1))
    tail = _Tail(keep)
    out = []
    fed = 0
    due = False
    for chunk in stream:
        while chunk.height:
            if due:
                out.append(read(fed, tail.frame, chunk.head(1)))
                due = False
            nxt = min((m for m in marks if m > fed), default=fed + chunk.height)
            take = min(chunk.height, nxt - fed)
            head, chunk = chunk.head(take), chunk.slice(take)
            bank.fit_predict(head)
            tail.push(head)
            fed += take
            due = fed in marks
    return out


def _slopes(bank: po.ModelBank) -> np.ndarray:
    """The first target's slopes after the last row learned, intercept left out."""
    return bank.coef()["coef"].to_numpy()[1:]


def ew_cov_decayed(
    rows: int,
    half_life: float = 50.0,
    window: float | None = None,
    seed: int = 0,
    oracle: bool = True,
) -> list:
    """``ew_cov`` over one feature at a level of 1e6 on an irregular clock,
    decayed (and windowed when ``window``), against the weighted mean and
    variance of the rows the state should hold, recomputed with ``fsum``.
    Returns ``(rows, mean's relative error, std's relative error)``."""
    level = 1e6
    kw: dict = {"window_size": window} if window is not None else {}
    spec = po.spec.ew_cov(
        "c",
        features=["x0"],
        clock="t",
        gap_cap=1e9,
        half_life=half_life,
        stats=["mean", "std"],
        **kw,
    )
    bank = po.ModelBank([spec])
    keep = int(64 * half_life * 1.5) + 100 if oracle else 1

    def read(fed, frame, nxt):
        out = bank.predict(nxt).unnest("c")
        if not oracle:
            return fed, out["mean_x0"][0], out["std_x0"][0]
        t = frame["t"].to_numpy()
        x = frame["x0"].to_numpy()
        now = t[-1]
        w = _decayed_weights(t, now, half_life)
        if window is not None:
            w = np.where(t > now - window, w, 0.0)
        sw = math.fsum(w)
        m = math.fsum(w * x) / sw
        # Deviations from the mean in compensated form: the level is 1e6.
        d = (x - level) - (m - level)
        c = math.fsum(w * d * d) / sw
        q = math.fsum(w * w)
        var = c * sw * sw / (sw * sw - q)
        return (
            fed,
            _rel(out["mean_x0"][0] - level, m - level) * abs(m - level) / abs(m),
            _rel(out["std_x0"][0], math.sqrt(var)),
        )

    return _feed_to_checkpoints(bank, _stream(rows, seed, 1, level, None), rows, keep, read)


# ---------------------------------------------------------------------------
# ewridge


BETA = np.array([2.0, -0.5, 0.25])


def ewridge_no_decay(rows: int, seed: int = 0, oracle: bool = True) -> list:
    """``ewridge`` with no decay and no ridge over three features at a level of
    100, against the least-squares fit of every row so far: the centred
    cross-products accumulated per chunk in ``math.fsum``, solved by NumPy.
    Returns ``(rows, the largest relative error of a slope)``."""
    level = 100.0
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        ridge=0.0,
    )
    bank = po.ModelBank([spec])
    k = 3
    # Sums of z, y, zz', zy over every row, each a list of fsum partials.
    sums: dict[str, list[float]] = {}

    def acc(key: str, v: float) -> None:
        sums.setdefault(key, []).append(v)

    def total(key: str) -> float:
        return math.fsum(sums[key])

    marks = checkpoints(rows)
    out = []
    fed = 0
    for chunk in _stream(rows, seed, k, level, BETA):
        mark_iter = [m for m in marks if fed < m <= fed + chunk.height]
        start = 0
        for mark in [*mark_iter, fed + chunk.height]:
            piece = chunk.slice(start, mark - fed - start)
            if piece.height == 0:
                continue
            start = mark - fed
            bank.fit_predict(piece)
            if not oracle:
                if mark in mark_iter:
                    out.append((fed + start, _slopes(bank)))
                continue
            z = piece.select("x0", "x1", "x2").to_numpy() - level
            y = piece["y"].to_numpy() - level
            for i in range(k):
                acc(f"z{i}", math.fsum(z[:, i]))
                acc(f"zy{i}", math.fsum(z[:, i] * y))
                for j in range(i, k):
                    acc(f"zz{i}{j}", math.fsum(z[:, i] * z[:, j]))
            acc("y", math.fsum(y))
            n_now = fed + start
            if mark in mark_iter:
                n = n_now
                mz = np.array([total(f"z{i}") / n for i in range(k)])
                my = total("y") / n
                s = np.empty((k, k))
                for i in range(k):
                    for j in range(i, k):
                        s[i, j] = s[j, i] = total(f"zz{i}{j}") / n - mz[i] * mz[j]
                r = np.array([total(f"zy{i}") / n - mz[i] * my for i in range(k)])
                want = np.linalg.solve(s, r)
                coef = _slopes(bank)
                out.append((n, float(np.max(np.abs(coef - want) / np.abs(want)))))
        fed += chunk.height
    return out


def ewridge_decayed(
    rows: int, half_life: float = 200.0, seed: int = 0, oracle: bool = True
) -> list:
    """``ewridge`` at a finite half-life on an irregular clock, against
    statsmodels' WLS on the last 64 half-lives at their decayed weights.
    Returns ``(rows, the largest relative error of a slope)``."""
    level = 100.0
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=1e9,
        half_life=half_life,
        ridge=0.0,
        max_rows_between_solves=1,
    )
    bank = po.ModelBank([spec])
    keep = int(64 * half_life * 1.5) + 100 if oracle else 1

    def read(fed, frame, nxt):
        coef = _slopes(bank)
        if not oracle:
            return fed, coef
        t = frame["t"].to_numpy()
        w = _decayed_weights(t, t[-1], half_life)
        x = sm.add_constant(frame.select("x0", "x1", "x2").to_numpy() - level)
        want = sm.WLS(frame["y"].to_numpy() - level, x, weights=w).fit().params[1:]
        return fed, float(np.max(np.abs(coef - want) / np.abs(want)))

    return _feed_to_checkpoints(bank, _stream(rows, seed, 3, level, BETA), rows, keep, read)


# ---------------------------------------------------------------------------
# kalman: P stays positive definite and bounded


def _kalman_p(bank: po.ModelBank) -> np.ndarray:
    state = json.loads(bank.write_json(pretty=False))
    found: list = []

    def walk(node):
        if isinstance(node, dict):
            if "p" in node and "beta" in node:
                found.append(node["p"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(state)
    assert len(found) == 1, f"expected one kalman state, found {len(found)}"
    p = np.array(found[0][0], dtype=float)
    k = int(round(math.sqrt(p.size)))
    return p.reshape(k, k)


def kalman_p(rows: int, seed: int = 0, oracle: bool = True) -> list:
    """``kalman`` with drifting coefficients on a stationary stream. At each
    checkpoint ``P`` (the first target's) from the state: returns ``(rows,
    the smallest eigenvalue, the largest)``, numpy's ``eigvalsh`` on the
    symmetrised matrix, with the asymmetry checked by the caller through the
    ratio of the two (``oracle=False``: ``(rows, P)``, no eigenvalues)."""
    spec = po.spec.kalman(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=1e9,
        half_life=500.0,
        coef_half_life=2000.0,
    )
    bank = po.ModelBank([spec])
    out = []

    def read(fed, frame, nxt):
        p = _kalman_p(bank)
        if not oracle:
            return fed, p
        asym = float(np.max(np.abs(p - p.T)) / np.max(np.abs(p)))
        ev = np.linalg.eigvalsh((p + p.T) / 2)
        return fed, float(ev[0]), float(ev[-1]), asym

    out = _feed_to_checkpoints(bank, _stream(rows, seed, 3, 0.0, BETA), rows, 10, read)
    return out


# ---------------------------------------------------------------------------
# The clock


def clock_steady_state(
    rows: int, temporal: bool, half_life_rows: float = 1_000.0, oracle: bool = True
) -> list:
    """``ew_cov``'s ``weight_sum`` over ``rows`` evenly spaced rows, one
    millisecond apart, against the weight recomputed from scratch,
    ``sum_i 2^(-(t_last - t_i)/h)`` over the last 64 half-lives in ``fsum``.
    A float clock in epoch seconds at 2024 (``1.7e9``, where a double
    resolves ``2.4e-7`` s: each step is rounded, so the recomputation reads
    the doubles fed, whose differences there are exact), or a ``Datetime``
    clock in nanoseconds, which the model reads against an origin it keeps,
    so its steps are exact. Returns ``(rows, weight_sum's relative error)``."""
    step_ns = 1_000_000
    h_ns = int(half_life_rows * step_ns)
    origin_ns = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z
    if temporal:
        spec = po.spec.ew_cov(
            "c",
            features=["x"],
            clock="t",
            gap_cap="1h",
            half_life=f"{h_ns}ns",
            stats=["mean"],
        )
    else:
        spec = po.spec.ew_cov(
            "c", features=["x"], clock="t", gap_cap=3600.0, half_life=h_ns / 1e9, stats=["mean"]
        )
    bank = po.ModelBank([spec])
    keep = int(64 * half_life_rows)
    marks = checkpoints(rows - 1)
    out = []
    fed = 0
    tail = np.empty(0)
    while fed < rows:
        n = min(CHUNK, rows - fed)
        i = np.arange(fed, fed + n, dtype=np.int64)
        if temporal:
            t = pl.Series("t", origin_ns + i * step_ns).cast(pl.Datetime("ns"))
            # Ages in half-lives from whole nanoseconds, exact: a division of
            # the nanoseconds since the origin by 1e9 would round each time
            # to 1.8e-12 s by 1e4 s, an error of the recomputation's own.
            secs = i.astype(np.float64)
        else:
            t = pl.Series("t", (origin_ns + i * step_ns) / 1e9)
            secs = t.to_numpy()
        got = bank.fit_predict(pl.DataFrame([t, pl.Series("x", np.zeros(n))])).unnest("c")
        ws = got["weight_sum"].to_numpy()
        both = np.concatenate([tail, secs])
        for m in marks:
            if fed <= m < fed + n and not oracle:
                out.append((m, ws[m - fed]))
            elif fed <= m < fed + n:
                # Row m reads the m rows before it, the last of them row m - 1.
                j = len(tail) + (m - fed)  # row m's index in `both`
                seen = both[max(0, j - keep) : j]
                unit = half_life_rows if temporal else h_ns / 1e9
                want = math.fsum(np.exp2(-(seen[-1] - seen) / unit))
                out.append((m, _rel(ws[m - fed], want)))
        tail = both[-keep:]
        fed += n
    return out


#: The cases by model, for the shared convergence registry: ``(model, case,
#: function of the row count returning its checkpoints)``. The bounds each
#: is held to are `tests/test_long_run.py`'s ``BOUNDS``, by case.
CASES: list[tuple[str, str, Callable[[int], list]]] = [
    ("ew_cov", "decayed", lambda rows: ew_cov_decayed(rows)),
    ("ew_cov", "windowed", lambda rows: ew_cov_decayed(rows, window=150.0)),
    ("ewridge", "no_decay", ewridge_no_decay),
    ("ewridge", "decayed_ridge", ewridge_decayed),
    ("kalman", "p_bounded", kalman_p),
    ("clock", "float", lambda rows: clock_steady_state(rows, temporal=False)),
    ("clock", "temporal", lambda rows: clock_steady_state(rows, temporal=True)),
]
