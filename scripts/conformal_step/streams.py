"""The streams the step rules are measured on, each run once through the bank
under `label_clock` to fix its served predictions (docs/records/CONFORMAL-STEP.md).

- generated: tests/data.py's seeded stream (`synthetic`: three features and a
  coefficient vector that walks), its noise replaced by one of four kinds,
  its clock either the generator's (a steady row rate) or warped so the row
  rate swings twentyfold over a cycle, and each label late by a delay that is
  none, short (about 5 rows), long (about 300 rows) or drawn per row
  (exponential, mean about 300 rows);
- flights: nycflights13's arrival delays less a frozen LightGBM model's
  prediction (as docs/records/TREE-FEATURES.md §2.3 builds it), each label
  at its landing, or at a constant 0, 1 or 8 hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
SLOT = HERE.parents[1]
sys.path.insert(0, str(SLOT / "tests"))
sys.path.insert(0, str(SLOT / "scripts" / "tree_features"))
CACHE = SLOT / ".cache" / "conformal-step"
CACHE.mkdir(parents=True, exist_ok=True)

import polars_online as po  # noqa: E402

N_ROWS = 100_000
GAP = 10.0  # the generator's mean step of clock
DELAYS = {"none": 0.0, "short": 5 * GAP, "long": 300 * GAP, "variable": -300 * GAP}
NOISES = ("gauss", "t3", "hetero", "shift")
RATES = ("steady", "bursty")


def generated(seed: int, noise: str, rate: str, delay: str):
    import data  # tests/data.py

    df, betas = data.synthetic(
        seed=seed,
        n_groups=1,
        n_rows=N_ROWS,
        k=3,
        n_targets=1,
        null_frac=0.0,
        beta_sigma=0.002,
        noise_sigma=1.0,
        session_every=N_ROWS,
    )
    x = df.select("x0", "x1", "x2").to_numpy()
    signal = np.einsum("ik,ik->i", x, betas["g0"][:, 0, :])
    rng = np.random.default_rng(1000 + seed)
    i = np.arange(N_ROWS)
    eps = rng.standard_normal(N_ROWS)
    if noise == "t3":
        eps = rng.standard_t(3, N_ROWS) / np.sqrt(3.0)
    elif noise == "hetero":
        # a slow swing of the scale by e^{±0.7}, and a stretch at three times
        eps = (
            eps
            * np.exp(0.7 * np.sin(2 * np.pi * i / 20_000))
            * np.where((i > 60_000) & (i < 70_000), 3.0, 1.0)
        )
    elif noise == "shift":
        # the level jumps by three noise sds at 50% and back at 75%
        eps = eps + 3.0 * ((i > N_ROWS // 2) & (i < 3 * N_ROWS // 4))
    y = signal + eps
    t = df["t"].to_numpy()
    if rate == "bursty":
        dt = np.diff(t, prepend=0.0)
        # rows twentyfold denser at the cycle's peak than at its trough, at
        # the same mean step of clock
        m = np.exp(1.5 * np.sin(2 * np.pi * i / 2_000))
        dt = dt * m / m.mean()
        t = np.cumsum(dt)
    d = DELAYS[delay]
    # a negative delay names the mean of an exponential delay drawn per row
    late = d if d >= 0 else np.random.default_rng(2000 + seed).exponential(-d, N_ROWS)
    avail = t + late
    frame = pl.DataFrame(
        {"t": t, "avail": avail, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y}
    )
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        clock="t",
        half_life=500 * GAP,
        gap_cap=50 * GAP,
        label_clock="avail",
        conformal=0.9,
        emit_sigma=True,
    )
    mask = i >= N_ROWS // 10
    return run(frame, spec, 500 * GAP, 50 * GAP, mask)


def flights_residual():
    f = CACHE / "flights.parquet"
    if f.exists():
        return pl.read_parquet(f)
    import common as C
    from leafrefit import train_trees

    df = C.load_flights()
    t = df["t"].to_numpy()
    yf = df["y"].to_numpy()
    tr, va = t < 243 * 1440.0, (t >= 243 * 1440.0) & (t < 273 * 1440.0)
    K = [C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")]
    X = np.column_stack([df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")] + K)
    b = train_trees(X, yf, tr, va, 31, 1000)
    out = pl.DataFrame(
        {
            "t": t,
            "avail": df["avail"].to_numpy(),
            "r": yf - b.predict(X, num_iteration=b.best_iteration),
        }
    )
    out.write_parquet(f)
    return out


FLIGHT_DELAYS = {"none": 0.0, "1 h": 60.0, "8 h": 480.0, "landing": None}


def flights(delay: str):
    df = flights_residual()
    t = df["t"].to_numpy()
    d = FLIGHT_DELAYS[delay]
    avail = df["avail"].to_numpy() if d is None else t + d
    frame = pl.DataFrame(
        {"c": t, "avail": avail, "y": df["r"].to_numpy(), "zero": np.zeros(len(t))}
    )
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["zero"],
        clock="c",
        gap_cap=600.0,
        half_life=7 * 1440.0,
        ridge=1e-6,
        label_clock="avail",
        conformal=0.9,
        emit_sigma=True,
    )
    return run(frame, spec, 7 * 1440.0, 600.0, t >= 273 * 1440.0)


def run(frame, spec, half_life, gap_cap, mask):
    out = po.ModelBank([spec]).fit_predict(frame)["m"].struct
    pred = out.field("pred_y").to_numpy().astype(float)
    y = frame["y"].to_numpy()
    return {
        "t": frame[frame.columns[0]].to_numpy().astype(float),
        "avail": frame["avail"].to_numpy().astype(float),
        "resid": y - pred,
        "pred": pred,
        "w": np.ones(frame.height),
        "lo": out.field("lo_y").to_numpy().astype(float),
        "sigma": out.field("sigma_y").to_numpy().astype(float),
        "half_life": half_life,
        "gap_cap": gap_cap,
        "mask": mask,
    }
