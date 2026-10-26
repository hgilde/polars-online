"""The refit of ``quantile``'s first rows, ``quantile_refit_rows`` (task 278,
D-Q2): once a target's count of learned rows is reached, its rows are refit
to the minimum of the smoothed check loss at the band its own residuals
draw, the sums are rebuilt at that fit and the scale from its residuals, and
the stream goes on online.

One pass takes each row as the fit before it sees it: its warm-up's rows as
least squares, then a row inside the band as a least-squares row. Without
decay those rows stay in the sums at full weight, so wild targets in the
first rows keep pulling where the batch fit reads them as outliers.

Held here: the refit lands near statsmodels' ``QuantReg`` on the same rows
(the third-party oracle; the refit's fit is the smoothed loss's, within the
band's smoothing of it), it frees the stream the wild rows held, chunk
invariance (hard rule 3), a save and a load before, at and after the refit,
the weight it reports (hard rule 8), the default count, the refusals and the
budget. That the fit is the smoothed loss's stationary point is held in
Rust, against ``crate::oracle``'s LU (``robust/quantile_refit_tests.rs``).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import polars_online as po

TIER = "essential"
INF = float("inf")


def _stream(n: int, seed: int, wild: int = 0) -> pl.DataFrame:
    """``y = 1 + 2x - z + 0.5 Laplace``, its first ``wild`` targets 1e4 off."""
    rng = np.random.default_rng(seed)
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = 1.0 + 2.0 * x - z + 0.5 * rng.laplace(size=n)
    y[:wild] += 1e4
    return pl.DataFrame({"x": x, "z": z, "y": y})


def _spec(**kw) -> dict:
    c = dict(targets=["y"], features=["x", "z"], half_life=INF, quantile=0.5, solve_every=0.0)
    return po.spec.quantile("m", **{**c, **kw})


def _coef(spec: dict, df: pl.DataFrame) -> np.ndarray:
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return np.array(bank.coef("m")["coef"].to_list())


def _quantreg(df: pl.DataFrame) -> np.ndarray:
    X = sm.add_constant(df.select("x", "z").to_numpy())
    return np.asarray(sm.QuantReg(df["y"].to_numpy(), X).fit(q=0.5, p_tol=1e-10).params)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_refit_lands_near_quantreg_on_the_buffer(seed):
    """The stream ends on the row the buffer fills, so the fit read after it
    is the refit's, of 400 rows whose first five are 1e4 off: within the
    band's smoothing of statsmodels' ``QuantReg`` on the same rows, where the
    one pass, its warm-up's least squares still in its sums, is far off.
    Measured 0.014-0.017 against the one pass's 12-25."""
    n = 400
    df = _stream(n, seed, wild=5)
    oracle = _quantreg(df)
    refit = _coef(_spec(quantile_refit_rows=n), df)
    one_pass = _coef(_spec(quantile_refit_rows=0), df)
    gap, before = np.max(np.abs(refit - oracle)), np.max(np.abs(one_pass - oracle))
    print(f"seed {seed}: refit {gap:.2e}, one pass {before:.2e}")
    assert gap < 0.03
    assert before > 5 * gap


def test_the_refit_frees_the_stream_its_first_rows_held():
    """Without decay, after 4,000 rows: five wild first rows leave the one
    pass well off ``QuantReg`` on every row, and the refit of the first 500
    rows leaves it where a clean start would be."""
    df = _stream(4000, 3, wild=5)
    oracle = _quantreg(df)
    on = np.max(np.abs(_coef(_spec(), df) - oracle))
    off = np.max(np.abs(_coef(_spec(quantile_refit_rows=0), df) - oracle))
    clean = _stream(4000, 3)
    floor = np.max(np.abs(_coef(_spec(), clean) - _quantreg(clean)))
    print(f"refit {on:.2e}, one pass {off:.2e}, clean {floor:.2e}")
    assert off > 0.05
    assert on < 3 * max(floor, 0.003)


def _frame(n: int = 1400) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n)
    y0 = 1000.0 + 2.0 * x + 0.5 * rng.laplace(size=n)
    y0[:4] += 1e3
    y1 = -3.0 * x + rng.standard_t(3, n)
    y1[rng.random(n) < 0.1] = None
    w = np.where(np.arange(n) % 11 == 5, 0.0, rng.uniform(0.5, 1.5, n))
    t = np.cumsum(rng.integers(1, 4, n)).astype(float)
    return pl.DataFrame({"x": x, "y0": y0, "y1": y1, "w": w, "t": t})


def _specs() -> list[dict]:
    c = dict(
        targets=["y0", "y1"],
        features=["x"],
        clock="t",
        gap_cap=50.0,
        coef_every=1,
        weight="w",
        quantile_refit_rows=200,
    )
    return [
        po.spec.quantile("m", quantile=0.5, half_life=INF, **c),
        po.spec.quantile("m", quantile=0.8, half_life=300.0, standardize=True, **c),
    ]


IDS = ["median", "0.8 decayed standardized"]


@pytest.mark.parametrize("spec", _specs(), ids=IDS)
def test_one_chunk_or_many_give_one_run(spec):
    """Hard rule 3: the buffer ends on a count of learned rows, never at a
    chunk's end, so chunks of 1, 7 and 600 rows give the one chunk's output
    to the bit."""
    df = _frame()
    whole = po.ModelBank([spec]).fit_predict(df)
    for rows in (1, 7, 600):
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
        assert pl.concat(parts).equals(whole, null_equal=True), rows


@pytest.mark.parametrize("spec", _specs(), ids=IDS)
def test_a_bank_saved_before_at_or_after_the_refit_resumes_to_the_bit(spec):
    """The buffer is state: saved and loaded mid-buffer, around the refit's
    row and after it, a bank goes on as the one that never stopped."""
    df = _frame()
    whole = po.ModelBank([spec]).fit_predict(df)
    for cut in (1, 37, 150, 220, 250, 1001):
        bank = po.ModelBank([spec])
        head = bank.fit_predict(df.slice(0, cut))
        loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
        tail = loaded.fit_predict(df.slice(cut))
        assert pl.concat([head, tail]).equals(whole, null_equal=True), cut


def _with_rows(spec: dict, rows: int | None) -> dict:
    model = {k: v for k, v in spec["model"].items() if k != "quantile_refit_rows"}
    if rows is not None:
        model["quantile_refit_rows"] = rows
    return {**spec, "model": model}


@pytest.mark.parametrize("spec", _specs(), ids=IDS)
def test_the_refit_moves_the_fit_after_its_row_and_not_the_weight(spec):
    """The activation check: with the refit on, the predictions part from the
    run without it only after a target's buffer fills -- its 200th learned
    row, a row of weight above 0 with its target -- and the weight the bank
    reports and gates on does not move (hard rule 8). A spec that does not
    name the count is the spec at 500, the default at one feature."""
    df = _frame()
    on = po.ModelBank([spec]).fit_predict(df)["m"]
    off = po.ModelBank([_with_rows(spec, 0)]).fit_predict(df)["m"]
    default = po.ModelBank([_with_rows(spec, None)]).fit_predict(df)["m"]
    at_500 = po.ModelBank([_with_rows(spec, 500)]).fit_predict(df)["m"]
    assert default.equals(at_500, null_equal=True)
    assert not default.equals(on, null_equal=True), "500 rows refit after 200"
    assert on.struct.field("weight_sum").equals(off.struct.field("weight_sum"))
    learned = df.with_row_index("i").filter(pl.col("y0").is_not_null() & (pl.col("w") > 0))
    fills = int(learned["i"][199])
    a = on.struct.field("pred_y0").to_numpy()
    b = off.struct.field("pred_y0").to_numpy()
    differ = np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))
    assert differ.size > 0
    assert differ[0] > fills, (differ[0], fills)


def test_a_count_inside_the_warm_up_is_refused():
    """The count is at least the warm-up's three rows a coefficient (six at
    one feature and an intercept), and the budget above 0."""
    df = _frame(50)
    c = dict(targets=["y0"], features=["x"], half_life=INF, quantile=0.5)
    po.ModelBank([po.spec.quantile("m", quantile_refit_rows=6, **c)]).fit_predict(df)
    with pytest.raises(ValueError, match="quantile_refit_rows must be 0 or at least three rows"):
        po.ModelBank([po.spec.quantile("m", quantile_refit_rows=5, **c)])
    with pytest.raises(ValueError, match="refit_budget must be > 0 MiB"):
        po.ModelBank([po.spec.quantile("m", refit_budget=0.0, **c)])


def test_the_refit_budget_gives_up_the_later_groups_and_counts_them():
    """``refit_budget`` bounds the buffers of all the spec's groups, as for
    ``huber``: at 50 rows, one feature and two targets a group reserves
    16,800 bytes, so 0.05 MiB holds three groups' buffers; the fourth to
    appear and every later one keep their one-pass fits, one notice names
    ``quantile_refit_rows``, and ``summary()`` counts their targets. Each
    admitted group is the group a bank with no bound fits, to the bit, and
    each given-up group the group with the refit off."""
    rng = np.random.default_rng(6)
    groups, rows = 8, 120
    n = groups * rows
    x = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "g": np.tile(np.arange(groups), rows).astype(str),
            "x": x,
            "y": 10.0 + 2.0 * x + 0.5 * rng.laplace(size=n) + 8.0 * (rng.random(n) < 0.1),
            "z": -x + rng.standard_normal(n),
        }
    )
    c = dict(targets=["y", "z"], features=["x"], group="g", half_life=INF, coef_every=0)

    def bank(**kw) -> tuple[po.ModelBank, pl.DataFrame, list[str]]:
        b = po.ModelBank([po.spec.quantile("m", quantile=0.5, quantile_refit_rows=50, **c, **kw)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = b.fit_predict(df)
        return b, out, [str(w.message) for w in caught if "refit_budget" in str(w.message)]

    bounded, out, notices = bank(refit_budget=0.05)
    free, out_free, none = bank(refit_budget=INF)
    off_spec = po.spec.quantile("m", quantile=0.5, quantile_refit_rows=0, **c)
    out_off = po.ModelBank([off_spec]).fit_predict(df)
    assert len(notices) == 1 and 'at group "3"' in notices[0], notices
    assert "quantile_refit_rows" in notices[0], notices
    assert none == []
    given = dict(bounded.summary().select("group", "refit_given_up").iter_rows())
    assert given == {str(g): (0 if g < 3 else 2) for g in range(groups)}, given
    assert set(free.summary()["refit_given_up"]) == {0}
    for g in range(groups):
        mine = out.filter(pl.col("g") == str(g))["m"]
        like = (out_free if g < 3 else out_off).filter(pl.col("g") == str(g))["m"]
        assert mine.equals(like, null_equal=True), g
