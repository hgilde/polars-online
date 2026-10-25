"""Test data: seeded synthetic generator and a cached public intraday download.

Hard rule 1 (CLAUDE.md): tests download or generate their own data. No data files
in the repo. Downloads are cached under ``.cache/`` (gitignored); tests needing
them call :func:`public_intraday` and are skipped when offline.
"""

from __future__ import annotations

import http.client
import io
import os
import ssl
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


def _download(
    url: str,
    attempts: int = 3,
    headers: dict[str, str] | None = None,
    context: ssl.SSLContext | None = None,
) -> bytes:
    """One public file, retried with a short pause; :class:`Offline` once
    every attempt has failed. An HTTP status a retry cannot fix is raised at
    once, as itself. ``headers`` go with the request (NIST's server answers
    urllib's default agent with a 403); ``context`` is the TLS context, for
    a server whose certificate needs one of its own."""
    opened = {"timeout": 30} if context is None else {"timeout": 30, "context": context}
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, **opened) as resp:
                return resp.read()
        except _TRANSIENT as e:
            if not is_transient(e):
                e.add_note(f"{url}: the server's answer, not the network's, so not offline")
                raise
            if attempt + 1 == attempts:
                raise Offline("offline") from e
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def write_atomically(path: Path, write) -> None:
    """Write a cached file whole or not at all: ``write`` writes a temporary
    file beside ``path``, named for this process, which then replaces
    ``path`` in one step. The suite runs in several processes at once
    (pytest-xdist), and a reader that saw a half-written file through
    ``path.exists()`` would read a truncated parquet; with the replace, a
    reader sees no file or the whole one, and two writers each write the
    same bytes."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    write_atomically(cached, df.write_parquet)
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
                body = z.read(f"{name}.csv")
            write_atomically(raw, lambda tmp: tmp.write_bytes(body))
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
    write_atomically(cached, both.write_parquet)
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


# ---------------------------------------------------------------------------
# NIST's Statistical Reference Datasets, linear regression (task 278 (b))

#: Where NIST serves the StRD linear least-squares files: a public-domain work
#: of the United States government (NIST/ITL, Statistical Engineering
#: Division; https://www.itl.nist.gov/div898/strd/), each a few kilobytes.
_STRD_URL = "https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/{name}.dat"
#: The linear datasets with certified values, by NIST's own order of
#: difficulty (lower, average, higher).
STRD_LINEAR = (
    "Norris",
    "Pontius",
    "NoInt1",
    "NoInt2",
    "Filip",
    "Longley",
    "Wampler1",
    "Wampler2",
    "Wampler3",
    "Wampler4",
    "Wampler5",
)


def nist_strd_linear(name: str) -> dict[str, object]:
    """One StRD linear least-squares dataset, cached under ``.cache/strd``.

    Returns ``x`` (the predictors as the model states them: the six columns
    of ``Longley``, or the powers ``x, x**2, ...`` of a polynomial model,
    computed in double precision), ``y``, ``coef`` (the certified
    estimates, ``B0`` first), ``se`` (their certified standard deviations),
    ``resid_sd`` (the certified residual standard deviation),
    ``intercept`` (whether the model has ``B0``) and ``degree`` (the
    polynomial's degree, 0 for ``Longley``). Raises :class:`Offline` when the
    network fails, which :func:`nist_strd_linear_or_skip` turns into a skip.
    """
    import re

    out_dir = CACHE_DIR / "strd"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / f"{name}.dat"
    if not cached.exists():
        body = _download(_STRD_URL.format(name=name), headers={"User-Agent": "polars-online-tests"})
        write_atomically(cached, lambda tmp: tmp.write_bytes(body))
    text = cached.read_text(encoding="utf-8")
    lines = text.splitlines()
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    coef, se = [], []
    for line in lines:
        m = re.match(rf"^\s*B(\d+)\s+({number})\s+({number})\s*$", line)
        if m:
            coef.append(float(m.group(2)))
            se.append(float(m.group(3)))
    m = re.search(rf"Residual\s+Standard Deviation\s+({number})", text)
    if m is None:
        raise ValueError(f"{name}: no certified residual standard deviation")
    resid_sd = float(m.group(1))
    start = next(i for i, line in enumerate(lines) if re.match(r"^Data:\s+y\s", line))
    rows = [
        [float(v) for v in line.split()]
        for line in lines[start + 1 :]
        if line.strip() and re.match(rf"^\s*{number}(\s+{number})+\s*$", line)
    ]
    data = np.array(rows)
    y = data[:, 0]
    intercept = "B0" in text.split("Model:")[1].split("Certified")[0]
    if data.shape[1] == 2 and (len(coef) > 2 or not intercept):
        # A polynomial in the one predictor: its powers, as the model reads.
        degree = len(coef) - 1 if intercept else len(coef)
        x = np.column_stack([data[:, 1] ** j for j in range(1, degree + 1)])
    else:
        degree = 0
        x = data[:, 1:]
    return {
        "x": x,
        "y": y,
        "coef": np.array(coef),
        "se": np.array(se),
        "resid_sd": resid_sd,
        "intercept": intercept,
        "degree": degree,
    }


def nist_strd_linear_or_skip(name: str) -> dict[str, object]:
    """:func:`nist_strd_linear`, skipping the test when offline (hard rule 1)."""
    import pytest

    try:
        return nist_strd_linear(name)
    except Offline as e:
        pytest.skip(f"offline: could not download NIST StRD {name} ({e.__cause__})")


# ---------------------------------------------------------------------------
# Classic robust-regression datasets (docs/PLAN.md task 278)

#: Rdatasets (Vincent Arel-Bundock's collection of the datasets R packages
#: ship), served as CSV from its GitHub pages. The robust-regression sets are
#: Rousseeuw and Leroy's (*Robust Regression and Outlier Detection*, 1987),
#: as R's ``robustbase`` and ``datasets`` packages distribute them.
_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv/{package}/{name}.csv"


def rdataset(package: str, name: str) -> pl.DataFrame:
    """One Rdatasets table, cached under ``.cache/rdatasets/``; its row-name
    column is dropped. Raises :class:`Offline` when the network fails."""
    folder = CACHE_DIR / "rdatasets"
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / f"{package}-{name}.parquet"
    if cached.exists():
        return pl.read_parquet(cached)
    raw = _download(_RDATASETS.format(package=package, name=name))
    df = pl.read_csv(io.BytesIO(raw)).drop("rownames")
    df = df.with_columns(pl.col(c).cast(pl.Float64) for c in df.columns)
    write_atomically(cached, df.write_parquet)
    return df


def rdataset_or_skip(package: str, name: str) -> pl.DataFrame:
    """:func:`rdataset`, skipping the test when offline (hard rule 1)."""
    import pytest

    try:
        return rdataset(package, name)
    except Offline as e:
        pytest.skip(f"offline: could not download Rdatasets {package}/{name} ({e.__cause__})")
