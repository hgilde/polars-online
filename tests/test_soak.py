"""T-E11: long-stream soak.

docs/PLAN.md section 7 claims the EW accumulators are stable under arbitrarily
long runs because they hold weighted *means*, not sums. That is an argument;
this is a measurement.

Opt-in (it takes tens of seconds): `uv run pytest -m soak`.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

pytestmark = pytest.mark.soak

ROWS = 10_000_000
CHUNK = 500_000


def _chunks(n_rows, chunk, seed=0):
    """Generate the stream in chunks so the test itself stays O(chunk)."""
    rng = np.random.default_rng(seed)
    t0 = 0.0
    for start in range(0, n_rows, chunk):
        n = min(chunk, n_rows - start)
        dt = rng.exponential(1.0, n)
        t = t0 + np.cumsum(dt)
        t0 = t[-1]
        x0 = rng.standard_normal(n)
        x1 = rng.standard_normal(n)
        yield pl.DataFrame(
            {
                "t": t,
                "x0": x0,
                "x1": x1,
                "y0": 2.0 * x0 - 0.5 * x1 + 0.1 * rng.standard_normal(n),
            }
        )


def test_ten_million_rows_stay_bounded_and_accurate():
    spec = po.spec.ewridge(
        "m",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=10.0,
        halflife=1000.0,
        min_periods=20.0,
    )
    bank = po.ModelBank([spec])
    n_eff_seen, last_coef, rows = [], None, 0
    for chunk in _chunks(ROWS, CHUNK):
        out = bank.fit_predict(chunk)
        rows += chunk.height
        neff = out["m"].struct.field("n_eff").to_numpy().astype(float)
        n_eff_seen.append((np.nanmin(neff), np.nanmax(neff)))
        coefs = [c for c in out["m"].struct.field("coef").to_list() if c is not None]
        if coefs:
            last_coef = np.array(coefs[-1], dtype=float)

    assert rows == ROWS
    # n_eff must settle near the steady state 1/(1 - 2^(-dt/halflife)) and never
    # grow without bound -- that is the whole point of mean-form accumulators.
    highs = [hi for _, hi in n_eff_seen]
    assert np.isfinite(highs).all()
    assert max(highs) < 5000.0, f"n_eff grew to {max(highs)}"
    assert max(highs[-3:]) == pytest.approx(max(highs[3:6]), rel=0.05), (
        "n_eff drifted between the start and end of the stream"
    )
    # And the fit is still right after 10M rows.
    assert last_coef is not None
    assert last_coef[1] == pytest.approx(2.0, abs=0.05)
    assert last_coef[2] == pytest.approx(-0.5, abs=0.05)


def test_state_stays_small_and_resumable_after_a_long_run():
    spec = po.spec.ewridge(
        "m",
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        max_dclock=10.0,
        halflife=1000.0,
        min_periods=20.0,
    )
    bank = po.ModelBank([spec])
    for chunk in _chunks(2_000_000, CHUNK, seed=1):
        bank.fit_predict(chunk)

    blob = bank.save_bytes()
    # Memory is O(state), not O(data): a 2M-row stream still serializes tiny.
    assert len(blob) < 4096, f"state grew to {len(blob)} bytes"

    resumed = po.ModelBank.load_bytes(blob, specs=[spec])
    tail = next(_chunks(1000, 1000, seed=2))
    a = bank.fit_predict(tail).select("m").unnest("m")
    b = resumed.fit_predict(tail).select("m").unnest("m")
    assert a.equals(b, null_equal=True)
