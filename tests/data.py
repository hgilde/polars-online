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


class Offline(RuntimeError):
    """Every attempt at a download failed for a reason a retry could fix.
    The one failure a test turns into a skip."""


#: Errors a retry can fix: no route, a timeout, a reset, and a response cut
#: short (``http.client.IncompleteRead``, which is not an ``OSError`` -- one
#: took a CI run down as a crash rather than a skip). An HTTP error is an
#: ``OSError`` too, and is sorted by its status first (:func:`is_transient`).
_TRANSIENT = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)


def is_transient(e: BaseException) -> bool:
    """Whether a retry can fix ``e``. A 5xx is the server failing and a 429
    is it asking to be called later; any other HTTP status is its answer,
    such as a file that moved (404) or is refused (403), which no retry and
    no skip should hide (review 2026-10-05, TA3)."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500 or e.code == 429
    return isinstance(e, _TRANSIENT)


def _download(url: str, attempts: int = 3) -> bytes:
    """One public file, retried with a short pause; :class:`Offline` once
    every attempt has failed. An HTTP status a retry cannot fix is raised at
    once, as itself."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except _TRANSIENT as e:
            if not is_transient(e):
                e.add_note(f"{url}: the server's answer, not the network's, so not offline")
                raise
            if attempt + 1 == attempts:
                raise Offline("offline") from e
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
    """BTCUSDT 1-minute rows from Binance's public dump, cached per day.

    ``dates`` are ``YYYY-MM-DD`` strings; days are concatenated in order, so the
    clock stays monotone. Raises :class:`Offline` when the network fails;
    test callers turn that into a skip via :func:`public_intraday_or_skip`.
    """
    frames = [_one_day(d) for d in dates]
    return pl.concat(frames).sort("t")


def public_intraday_or_skip(dates: tuple[str, ...] = _DEFAULT_DATES) -> pl.DataFrame:
    import pytest

    try:
        return public_intraday(dates)
    except Offline:
        pytest.skip("offline: could not download public intraday data")


# ---------------------------------------------------------------------------
# Interleaved quotes and trades (docs/PLAN.md task 143)

_BINANCE_UM = "https://data.binance.vision/data/futures/um/daily"


def public_quotes_and_trades(symbol: str = "ETCUSDT", date: str = "2024-01-02") -> pl.DataFrame:
    """One symbol-day of Binance USD-M futures ``bookTicker`` (the best bid and
    ask at each update) and ``trades``, interleaved by their millisecond
    stamps, cached as parquet under ``.cache/microstructure``.

    Columns: ``ts`` (Datetime, microseconds), ``kind`` (``"quote"`` or
    ``"trade"``), ``bid``, ``ask``, ``bid_qty``, ``ask_qty`` (the book as of
    each row: a trade carries the last quote's), ``price``, ``quantity`` and
    ``side`` (``"buy"`` when the taker bought; null on a quote), ``mid``. At an
    equal stamp a quote precedes a trade, as a trade is reported against the
    book it hit; the order of a trade and a quote in one millisecond is not
    in the data (about half the trades share a millisecond with a quote).
    Downloads once (two zips, about 33 MB); raises :class:`Offline` when the
    network fails, which :func:`public_quotes_and_trades_or_skip` turns into
    a skip.
    """
    import io
    import zipfile

    out_dir = CACHE_DIR / "microstructure"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / f"{symbol}-{date}.parquet"
    if cached.exists():
        return pl.read_parquet(cached)

    def csv(kind: str) -> pl.DataFrame:
        name = f"{symbol}-{kind}-{date}"
        raw = out_dir / f"{name}.csv"
        if not raw.exists():
            data = _download(f"{_BINANCE_UM}/{kind}/{symbol}/{name}.zip")
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                raw.write_bytes(z.read(f"{name}.csv"))
        return pl.read_csv(raw)

    quotes = csv("bookTicker").select(
        ts=pl.col("transaction_time"),
        bid=pl.col("best_bid_price"),
        ask=pl.col("best_ask_price"),
        bid_qty=pl.col("best_bid_qty"),
        ask_qty=pl.col("best_ask_qty"),
    )
    trades = csv("trades").select(
        ts=pl.col("time"),
        price=pl.col("price"),
        quantity=pl.col("qty"),
        side=pl.when(pl.col("is_buyer_maker")).then(pl.lit("sell")).otherwise(pl.lit("buy")),
    )
    both = (
        pl.concat(
            [
                quotes.with_columns(
                    kind=pl.lit("quote"),
                    price=pl.lit(None, pl.Float64),
                    quantity=pl.lit(None, pl.Float64),
                    side=pl.lit(None, pl.String),
                    order=pl.lit(0),
                ),
                trades.with_columns(
                    kind=pl.lit("trade"),
                    bid=pl.lit(None, pl.Float64),
                    ask=pl.lit(None, pl.Float64),
                    bid_qty=pl.lit(None, pl.Float64),
                    ask_qty=pl.lit(None, pl.Float64),
                    order=pl.lit(1),
                ),
            ],
            how="diagonal",
        )
        .sort(["ts", "order"], maintain_order=True)
        .drop("order")
        .with_columns(ts=pl.from_epoch("ts", time_unit="ms").dt.cast_time_unit("us"))
        .with_columns(pl.col("bid", "ask", "bid_qty", "ask_qty").forward_fill())
        .with_columns(mid=(pl.col("bid") + pl.col("ask")) / 2)
        .select(
            "ts", "kind", "bid", "ask", "bid_qty", "ask_qty", "price", "quantity", "side", "mid"
        )
    )
    both.write_parquet(cached)
    return both


def public_quotes_and_trades_or_skip(
    symbol: str = "ETCUSDT", date: str = "2024-01-02"
) -> pl.DataFrame:
    """:func:`public_quotes_and_trades`, skipping the test when offline.
    Its caller caught ``OSError``, which this module never raised offline, so
    offline it errored instead of skipping (review 2026-10-05, TB2)."""
    import pytest

    try:
        return public_quotes_and_trades(symbol, date)
    except Offline as e:
        pytest.skip(f"offline: could not download {symbol} quotes and trades ({e.__cause__})")
