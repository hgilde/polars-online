"""Test data: seeded synthetic generator and a cached public intraday download.

Hard rule 1 (CLAUDE.md): tests download or generate their own data. No data files
in the repo. Downloads are cached under ``.cache/`` (gitignored); tests needing
them call :func:`public_intraday` and are skipped when offline.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# Binance's public data dump: stable URLs, no auth, permissive terms.
_PUBLIC_URL_FMT = (
    "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{date}.zip"
)
_DEFAULT_DATES = ("2024-01-02",)
#: The ten days ``scripts/validate.py`` runs over (``docs/VALIDATION.md``);
#: here so the test that regenerates the document warms the same cache.
VALIDATION_DATES = tuple(f"2024-01-{d:02d}" for d in range(2, 12))
_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def synthetic(
    seed: int = 0,
    n_groups: int = 3,
    n_rows: int = 400,
    k: int = 3,
    n_targets: int = 1,
    null_frac: float = 0.02,
    beta_sigma: float = 0.02,
    noise_sigma: float = 0.5,
    session_every: int = 120,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Seeded stream generator with known, time-varying beta.

    Per group: an irregular monotone clock ``t``, session breaks every
    ``session_every`` rows, a volume clock ``vol`` that resets per session, a row
    weight ``w``, features ``x0..``, and targets ``y0..`` following
    ``y_j = x . beta_j(t) + eps`` where each ``beta_j`` is a random walk.

    Returns ``(df, betas)``; ``betas[group]`` has shape ``(n_rows, n_targets, k)``.
    """
    rng = np.random.default_rng(seed)
    frames: list[pl.DataFrame] = []
    betas: dict[str, np.ndarray] = {}
    for g in range(n_groups):
        name = f"g{g}"
        dt = rng.exponential(scale=10.0, size=n_rows)
        dt[0] = 0.0
        t = np.cumsum(dt)
        session = (np.arange(n_rows) // session_every).astype(np.int64)
        vol_step = rng.lognormal(mean=0.0, sigma=1.0, size=n_rows)
        vol = np.empty(n_rows)
        for s in np.unique(session):
            m = session == s
            vol[m] = np.cumsum(vol_step[m])
        w = rng.uniform(0.5, 1.5, size=n_rows)
        x = rng.standard_normal((n_rows, k))
        beta = np.empty((n_rows, n_targets, k))
        beta[0] = rng.standard_normal((n_targets, k))
        for i in range(1, n_rows):
            beta[i] = beta[i - 1] + beta_sigma * rng.standard_normal((n_targets, k))
        y = np.einsum("ik,ijk->ij", x, beta) + noise_sigma * rng.standard_normal(
            (n_rows, n_targets)
        )
        betas[name] = beta

        cols: dict[str, object] = {
            "group": [name] * n_rows,
            "t": t,
            "session": session,
            "vol": vol,
            "w": w,
        }
        for j in range(k):
            xj = x[:, j].copy()
            if null_frac > 0:
                xj[rng.random(n_rows) < null_frac] = np.nan
            cols[f"x{j}"] = xj
        for j in range(n_targets):
            yj = y[:, j].copy()
            if null_frac > 0:
                yj[rng.random(n_rows) < null_frac] = np.nan
            cols[f"y{j}"] = yj
        frames.append(pl.DataFrame(cols))
    df = pl.concat(frames)
    # NaN -> proper nulls; the library treats null and NaN in inputs identically,
    # but tests standardize on nulls.
    df = df.with_columns(pl.col(c).fill_nan(None) for c, d in df.schema.items() if d == pl.Float64)
    return df, betas


#: Errors a retry can fix: no route, a timeout, a reset, and a response cut
#: short (``http.client.IncompleteRead``, which is not an ``OSError`` -- one
#: took a CI run down as a crash rather than a skip).
_TRANSIENT = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)


def _download(url: str, attempts: int = 3) -> bytes:
    """One public file, retried with a short pause; ``RuntimeError("offline")``
    once every attempt has failed."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except _TRANSIENT as e:
            if attempt + 1 == attempts:
                raise RuntimeError("offline") from e
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _one_day(date: str) -> pl.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / f"BTCUSDT-1m-{date}.parquet"
    if cached.exists():
        return pl.read_parquet(cached)
    raw = _download(_PUBLIC_URL_FMT.format(date=date))
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_bytes = zf.read(zf.namelist()[0])
    df = pl.read_csv(io.BytesIO(csv_bytes), has_header=False, new_columns=_KLINE_COLS)
    df = (
        df.select(
            # open_time is microseconds in recent dumps, milliseconds in older ones.
            t=pl.when(pl.col("open_time") > 10**15)
            .then(pl.col("open_time") / 1_000_000)
            .otherwise(pl.col("open_time") / 1_000),
            open=pl.col("open").cast(pl.Float64),
            high=pl.col("high").cast(pl.Float64),
            low=pl.col("low").cast(pl.Float64),
            close=pl.col("close").cast(pl.Float64),
            volume=pl.col("volume").cast(pl.Float64),
            n_trades=pl.col("n_trades").cast(pl.Float64),
        )
        .sort("t")
        .with_columns(group=pl.lit("BTCUSDT"))
    )
    df.write_parquet(cached)
    return df


def public_intraday(dates: tuple[str, ...] = _DEFAULT_DATES) -> pl.DataFrame:
    """BTCUSDT 1-minute bars from Binance's public dump, cached per day.

    ``dates`` are ``YYYY-MM-DD`` strings; days are concatenated in order, so the
    clock stays monotone. Raises ``RuntimeError("offline")`` when a download
    fails; test callers turn that into a skip via :func:`public_intraday_or_skip`.
    """
    frames = [_one_day(d) for d in dates]
    return pl.concat(frames).sort("t")


def public_intraday_or_skip(dates: tuple[str, ...] = _DEFAULT_DATES) -> pl.DataFrame:
    import pytest

    try:
        return public_intraday(dates)
    except RuntimeError:
        pytest.skip("offline: could not download public intraday data")
