"""Shared helpers for the task-324 prototypes (docs/records/TREE-FEATURES.md).
Data are read from the repository's .cache, which fetch.py fills through
tests/data.py; nothing else downloads."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

SLOT = Path(__file__).resolve().parents[2]  # the repository
CACHE = SLOT / ".cache"
#: Where each script writes its numbers as JSON.
OUT = Path(os.environ.get("TREE_FEATURES_OUT", str(CACHE / "tree-features" / "results")))
OUT.mkdir(parents=True, exist_ok=True)


def load_flights() -> pl.DataFrame:
    """nycflights13 flights, cancelled flights dropped, in scheduled-departure
    order. t: minutes since 2013-01-01 00:00 local; avail: when the label
    (arr_delay) is known, approximated as actual departure + air time + 20
    minutes of taxi."""
    df = pl.read_parquet(CACHE / "tree-features" / "nycflights13-flights.parquet")
    df = df.filter(pl.col("arr_delay").is_not_null() & pl.col("air_time").is_not_null())
    df = (
        df.with_columns(
            ts=pl.datetime(
                pl.col("year"),
                pl.col("month"),
                pl.col("day"),
                pl.col("sched_dep_time") // 100,
                pl.col("sched_dep_time") % 100,
            ),
        )
        .with_columns(
            t=(pl.col("ts") - pl.datetime(2013, 1, 1)).dt.total_minutes().cast(pl.Float64),
        )
        .with_columns(
            avail=pl.col("t") + pl.col("dep_delay").cast(pl.Float64) + pl.col("air_time") + 20.0,
            y=pl.col("arr_delay").cast(pl.Float64),
            dep_min=((pl.col("sched_dep_time") // 100) * 60 + pl.col("sched_dep_time") % 100).cast(
                pl.Float64
            ),
            arr_min=((pl.col("sched_arr_time") // 100) * 60 + pl.col("sched_arr_time") % 100).cast(
                pl.Float64
            ),
            wday=pl.col("ts").dt.weekday().cast(pl.Float64),
            doy=pl.col("ts").dt.ordinal_day().cast(pl.Float64),
            distance=pl.col("distance").cast(pl.Float64),
            monthf=pl.col("month").cast(pl.Float64),
            flight_id=pl.col("carrier") + pl.col("flight").cast(pl.String),
            route=pl.col("origin") + "-" + pl.col("dest"),
            tailnum=pl.col("tailnum").fill_null("NA"),
        )
        .sort("t", "rownames")
    )
    return df.select(
        "t",
        "avail",
        "y",
        "dep_min",
        "arr_min",
        "wday",
        "doy",
        "distance",
        "monthf",
        "carrier",
        "origin",
        "dest",
        "tailnum",
        "flight_id",
        "route",
    )


def codes(s: pl.Series) -> np.ndarray:
    """Integer codes in order of first appearance (a stream's view)."""
    first: dict = {}
    out = np.empty(len(s), dtype=np.int64)
    for i, v in enumerate(s.to_list()):
        out[i] = first.setdefault(v, len(first))
    return out


def ordered_ts(
    t,
    avail,
    key,
    y,
    *,
    half_life=math.inf,
    prior_weight=10.0,
    prior_half_life=None,
    freeze_at=None,
    cap=None,
    cms=None,
    seed=0,
):
    """Ordered target statistic of `key`, row by row in `t` order.

    A row's encoding reads only labels whose `avail` < the row's `t` (an
    embargo by label time). Per category: decayed W = sum w, S = sum w y,
    both decayed by 0.5 ** (dt / half_life) to the reading time. The prior is
    the global decayed mean (its own half-life, default the same) with weight
    `prior_weight`: enc = (S + a p) / (W + a). Also returns W (the weight
    behind the encoding) and the row's decayed frequency share.

    freeze_at: rows with t >= freeze_at read the state as it stood at
      freeze_at (no labels after it; CatBoost's test-time rule).
    cap: keep at most `cap` categories; inserting into a full table evicts the
      one of least decayed W (ties: oldest); evicted/unknown read the prior.
    cms: (depth, width) -- a count-min style table of (W, S) per cell; a
      category reads the cell of least W among its `depth` hashed cells.
    """
    n = len(t)
    hl = half_life
    phl = prior_half_life if prior_half_life is not None else hl
    ln2 = math.log(2.0)
    enc = np.empty(n)
    wout = np.empty(n)
    order_l = np.argsort(avail, kind="stable")
    j = 0
    # global prior
    gW = 0.0
    gS = 0.0
    gt = 0.0
    if cms is None:
        tab: dict[int, list[float]] = {}  # key -> [W, S, t_last]
    else:
        depth, width = cms
        rng = np.random.default_rng(seed)
        a_h = rng.integers(1, 2**31 - 1, size=depth)
        b_h = rng.integers(0, 2**31 - 1, size=depth)
        P = 2**31 - 1
        cW = np.zeros((depth, width))
        cS = np.zeros((depth, width))
        cT = np.zeros((depth, width))

    def decay(dt, h):
        return 1.0 if h == math.inf else math.exp(-ln2 * dt / h)

    def cells(k):
        return [(d, int((a_h[d] * (k + 1) + b_h[d]) % P % width)) for d in range(depth)]

    def learn(k, yk, tk):
        nonlocal gW, gS, gt
        f = decay(tk - gt, phl)
        gW = gW * f + 1.0
        gS = gS * f + yk
        gt = tk
        if cms is not None:
            for d, c in cells(k):
                f = decay(tk - cT[d, c], hl)
                cW[d, c] = cW[d, c] * f + 1.0
                cS[d, c] = cS[d, c] * f + yk
                cT[d, c] = tk
            return
        e = tab.get(k)
        if e is None:
            if cap is not None and len(tab) >= cap:
                # evict least decayed weight
                victim = min(tab.items(), key=lambda kv: kv[1][0] * decay(tk - kv[1][2], hl))[0]
                del tab[victim]
            tab[k] = [1.0, yk, tk]
        else:
            f = decay(tk - e[2], hl)
            e[0] = e[0] * f + 1.0
            e[1] = e[1] * f + yk
            e[2] = tk

    def read(k, tk):
        p = gS / gW if gW > 0 else 0.0
        if cms is not None:
            best = None
            for d, c in cells(k):
                f = decay(tk - cT[d, c], hl)
                Wc = cW[d, c] * f
                if best is None or Wc < best[0]:
                    best = (Wc, cS[d, c] * f)
            W, S = best
        else:
            e = tab.get(k)
            if e is None:
                W, S = 0.0, 0.0
            else:
                f = decay(tk - e[2], hl)
                W, S = e[0] * f, e[1] * f
        return (S + prior_weight * p) / (W + prior_weight), W

    frozen = freeze_at is not None
    for i in range(n):
        ti = t[i]
        lim = min(ti, freeze_at) if frozen else ti
        while j < n and avail[order_l[j]] < lim:
            r = order_l[j]
            learn(key[r], y[r], avail[r])
            j += 1
        enc[i], wout[i] = read(key[i], lim if frozen and ti >= freeze_at else ti)
    return enc, wout


def decayed_freq(t, key, half_life=math.inf):
    """A row's category's decayed count among rows strictly before it
    (needs no label), and that count over the decayed total."""
    n = len(t)
    out = np.empty(n)
    share = np.empty(n)
    tab: dict[int, list[float]] = {}
    tot = 0.0
    tt = 0.0
    ln2 = math.log(2.0)
    for i in range(n):
        ti = t[i]
        f = 1.0 if half_life == math.inf else math.exp(-ln2 * (ti - tt) / half_life)
        tot *= f
        tt = ti
        e = tab.get(key[i])
        if e is None:
            c = 0.0
        else:
            c = e[0] * (1.0 if half_life == math.inf else math.exp(-ln2 * (ti - e[1]) / half_life))
        out[i] = c
        share[i] = c / tot if tot > 0 else 0.0
        tab[key[i]] = [c + 1.0, ti]
        tot += 1.0
    return out, share


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def load_line() -> str:
    return os.popen("uptime").read().strip()
