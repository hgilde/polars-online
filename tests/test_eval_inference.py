"""Where `po.eval.diebold_mariano` and `po.eval.clark_west` read null, in the
frame and over a query alike (task 382b, review of `po.eval` B4, B7 and C2).

- **B4.** The per-row form (``half_life=``) applies ``min_samples``: a key's
  statistic is null until it has ``min_samples`` rows with a value, 30 by
  default. It was taken and ignored, and a key's second row read a
  statistic that passed 1.96 on 32% of streams of noise.
- **B7.** Both forms null a key's statistic where it has at most
  ``4 (lags + 1)`` rows with a value. The deviations from the mean sum to
  zero, so the Bartlett sum of a short key shrinks toward zero as ``lags``
  nears its rows, and its statistic grows without meaning (11.7 on two rows
  of noise at ``lags=10``).
- **C2.** Without ``group``, a query returns a target no row has a value of
  as the frame does, as a row with ``n = 0`` and null statistics. It left
  the target out.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

CALLS = {
    "diebold_mariano": ("dm", {"a": "a", "b": "b"}),
    "clark_west": ("cw", {"big": "a", "small": "b"}),
}


def _two_specs(
    ra: np.ndarray, rb: np.ndarray, g: np.ndarray | None = None, z: bool = False
) -> pl.DataFrame:
    """Two specs' outputs holding ``resid_y``, and with ``z`` a second
    target ``resid_z`` whose ``a`` side is null on every row."""
    fields_a = {"resid_y": ra}
    fields_b = {"resid_y": rb}
    if z:
        fields_a["resid_z"] = np.full(len(ra), None)
        fields_b["resid_z"] = rb
    df = pl.DataFrame(
        {
            "a": pl.DataFrame(fields_a, schema=dict.fromkeys(fields_a, pl.Float64)).to_struct(),
            "b": pl.DataFrame(fields_b, schema=dict.fromkeys(fields_b, pl.Float64)).to_struct(),
        }
    )
    return df if g is None else df.with_columns(pl.Series("g", g))


def _run(df: pl.DataFrame, call: str, monkeypatch, rows: int | None, **kw) -> pl.DataFrame:
    """The call over ``df``, or over a query of it read ``rows`` at a time."""
    _, sides = CALLS[call]
    fn = getattr(po.eval, call)
    if rows is None:
        return fn(df, **sides, **kw)
    monkeypatch.setattr(po.eval, "_chunk_rows", lambda: rows)
    got = fn(df.lazy(), **sides, **kw)
    return got.collect() if isinstance(got, pl.LazyFrame) else got


def _per_row(df: pl.DataFrame, call: str, monkeypatch, rows: int | None, **kw) -> list:
    field, _ = CALLS[call]
    got = _run(df, call, monkeypatch, rows, **kw)
    return got[call].struct.field(f"{field}_y").to_list()


def _interleaved(n: int, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Two keys of ``n`` rows each, interleaved, with a null residual on
    every seventh row of key 0, and each row's count of rows with a value
    in its key so far, itself included (0 where it has none)."""
    rng = np.random.default_rng(seed)
    g = np.tile([0, 1], n)
    ra, rb = rng.normal(size=(2, 2 * n))
    ra[np.arange(0, 2 * n, 14)] = np.nan
    seen = {0: 0, 1: 0}
    counts = np.zeros(2 * n, dtype=int)
    for i, (key, value) in enumerate(zip(g, ra, strict=True)):
        if np.isfinite(value):
            seen[key] += 1
            counts[i] = seen[key]
    return _two_specs(ra, rb, g), counts


class TestThePerRowFormAppliesMinSamples:
    """B4: ``min_samples`` with ``half_life`` nulls a key's rows before its
    ``min_samples``-th row with a value, and moves no other row."""

    @pytest.mark.parametrize("call", list(CALLS))
    @pytest.mark.parametrize("lags", [0, 2])
    @pytest.mark.parametrize("min_samples", [20, 45])
    def test_null_until_the_key_has_min_samples_rows(self, monkeypatch, call, lags, min_samples):
        df, counts = _interleaved(120, seed=min_samples + lags)
        kw = dict(lags=lags, half_life=10.0, group="g")
        free = _per_row(df, call, monkeypatch, None, min_samples=1, **kw)
        for rows in (None, 1, 7, 10**6):
            got = _per_row(df, call, monkeypatch, rows, min_samples=min_samples, **kw)
            for i, (value, count) in enumerate(zip(got, counts, strict=True)):
                if count < min_samples:
                    assert value is None, (rows, i, count, value)
                else:
                    assert value is not None, (rows, i, count)
                    assert value == pytest.approx(free[i], rel=1e-12, abs=0.0), (rows, i)

    @pytest.mark.parametrize("call", list(CALLS))
    def test_by_default_a_keys_first_29_rows_read_null(self, monkeypatch, call):
        df, counts = _interleaved(60, seed=3)
        for rows in (None, 13):
            got = _per_row(df, call, monkeypatch, rows, half_life=20.0, group="g")
            assert all(v is None for v, c in zip(got, counts, strict=True) if c < 30), rows
            assert all(v is not None for v, c in zip(got, counts, strict=True) if c >= 30), rows


class TestAKeyOfFewRowsForItsLags:
    """B7: a key with at most ``4 (lags + 1)`` rows with a value reads a
    null statistic, its ``n`` and ``mean`` kept; one more row and it reads
    one. Whatever ``min_samples`` says."""

    @pytest.mark.parametrize("call", list(CALLS))
    @pytest.mark.parametrize("lags", [0, 2, 10])
    def test_run_once(self, monkeypatch, call, lags):
        field, _ = CALLS[call]
        floor = 4 * (lags + 1)
        rng = np.random.default_rng(lags)
        sizes = [2, floor, floor + 1, floor + 40]
        g = np.repeat(np.arange(len(sizes)), sizes)
        ra, rb = rng.normal(size=(2, len(g)))
        df = _two_specs(ra, rb, g)
        for rows in (None, 5, 10**6):
            got = _run(df, call, monkeypatch, rows, lags=lags, group="g", min_samples=1)
            assert got["n"].to_list() == sizes, rows
            assert got["mean"].null_count() == 0, rows
            assert got[field].is_null().to_list() == [True, True, False, False], rows
            assert got["p_value"].is_null().to_list() == [True, True, False, False], rows

    @pytest.mark.parametrize("call", list(CALLS))
    @pytest.mark.parametrize("lags", [0, 3])
    def test_per_row(self, monkeypatch, call, lags):
        floor = 4 * (lags + 1)
        df, counts = _interleaved(80, seed=11 + lags)
        kw = dict(lags=lags, half_life=30.0, group="g", min_samples=1)
        for rows in (None, 1, 9):
            got = _per_row(df, call, monkeypatch, rows, **kw)
            assert all(v is None for v, c in zip(got, counts, strict=True) if c <= floor), rows
            assert all(v is not None for v, c in zip(got, counts, strict=True) if c > floor), rows


class TestATargetWithNoValueOverAQuery:
    """C2: a target no row has a value of reads as the frame reads it: one
    row with ``n = 0`` and null statistics without ``group``, and no row
    with one."""

    @pytest.mark.parametrize("call", list(CALLS))
    @pytest.mark.parametrize("grouped", [False, True])
    def test_the_query_returns_the_frames_rows(self, monkeypatch, call, grouped):
        rng = np.random.default_rng(5)
        n = 300
        ra, rb = rng.normal(size=(2, n))
        g = np.arange(n) % 3 if grouped else None
        df = _two_specs(ra, rb, g, z=True)
        kw = dict(lags=2, group="g") if grouped else dict(lags=2)
        frame = _run(df, call, monkeypatch, None, **kw)
        targets = frame["target"].to_list()
        assert targets == (["y"] * 3 if grouped else ["y", "z"])
        if not grouped:
            assert frame.row(1) == ("z", 0, None, None, None)
        for rows in (7, 10**6):
            got = _run(df, call, monkeypatch, rows, **kw)
            assert got.schema == frame.schema, rows
            assert got["target"].to_list() == targets, rows
            assert got["n"].to_list() == frame["n"].to_list(), rows
            for column in ("mean", CALLS[call][0], "p_value"):
                np.testing.assert_allclose(
                    got[column].to_numpy().astype(float),
                    frame[column].to_numpy().astype(float),
                    rtol=1e-9,
                    err_msg=f"{rows} {column}",
                )
