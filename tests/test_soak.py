"""T-E11: long-stream soak.

docs/PLAN.md section 7 claims the EW accumulators are stable under arbitrarily
long runs because they hold weighted *means*, not sums. That is an argument;
this is a measurement.

Deselected from the default run by pyproject's addopts, and run by the
weekly leak-check workflow: `uv run pytest -m soak`. Until task 160 (TC11) no
workflow ran it, and the resume test below had failed unseen since task 120
made a step back a refusal: its tail restarted the clock at 0.
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
        gap_cap=10.0,
        half_life=1000.0,
        min_weight=20.0,
    )
    bank = po.ModelBank([spec])
    n_eff_seen, last_coef, rows = [], None, 0
    for chunk in _chunks(ROWS, CHUNK):
        out = bank.fit_predict(chunk)
        rows += chunk.height
        neff = out["m"].struct.field("weight_sum").to_numpy().astype(float)
        n_eff_seen.append((np.nanmin(neff), np.nanmax(neff)))
        coefs = [c for c in out["m"].struct.field("coef").to_list() if c is not None]
        if coefs:
            last_coef = np.array(coefs[-1], dtype=float)

    assert rows == ROWS
    # weight_sum must settle near the steady state 1/(1 - 2^(-dt/half-life)) and never
    # grow without bound -- that is the whole point of mean-form accumulators.
    highs = [hi for _, hi in n_eff_seen]
    assert np.isfinite(highs).all()
    assert max(highs) < 5000.0, f"weight_sum grew to {max(highs)}"
    assert max(highs[-3:]) == pytest.approx(max(highs[3:6]), rel=0.05), (
        "weight_sum drifted between the start and end of the stream"
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
        gap_cap=10.0,
        half_life=1000.0,
        min_weight=20.0,
    )
    bank = po.ModelBank([spec])
    last_t = 0.0
    for chunk in _chunks(2_000_000, CHUNK, seed=1):
        bank.fit_predict(chunk)
        last_t = chunk["t"][-1]

    blob = bank.save_bytes()
    # Memory is O(state), not O(data): a 2M-row stream still serializes tiny.
    assert len(blob) < 4096, f"state grew to {len(blob)} bytes"

    resumed = po.ModelBank.load_bytes(blob, specs=[spec])
    # The tail continues the stream's clock, as a resumed feed does.
    tail = next(_chunks(1000, 1000, seed=2)).with_columns(pl.col("t") + last_t)
    a = bank.fit_predict(tail).select("m").unnest("m")
    b = resumed.fit_predict(tail).select("m").unnest("m")
    assert a.equals(b, null_equal=True)
