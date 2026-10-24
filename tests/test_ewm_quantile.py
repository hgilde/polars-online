"""``po.ewm_quantile``: the exponentially weighted quantile of a column over
a window of the stream's clock (docs/PLAN.md task 272 (d)).

The operator reads each value as the bucket it falls in, each bucket
spanning a ratio of ``e**(1/64)``, so it is held to numpy's weighted
quantile (``np.quantile(..., weights=, method="inverted_cdf")``: the
smallest value whose share of the weight at or below it reaches the level)
within the sketch's relative accuracy, ``tanh(1/128)``, over the window the
other operators hold, each row weighing ``lam**|t_j - t|``. The EW-MAD
recipe is held to statsmodels' MAD over an evenly weighted window, and the
rest is the stream's own rules: chunk invariance, a save and a load, nulls,
``min_samples``, and a scale of a target that leaves the row's own label
out.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm
from statsmodels.robust.scale import mad

import polars_online as po

TIER = "essential"

#: The sketch's relative accuracy: ``tanh(1 / 128)``.
ALPHA = math.tanh(1 / 128)

#: ``Φ⁻¹(3/4)``, the MAD of a standard normal.
Z75 = 0.6744897501960817


def stream(n: int, seed: int, *, repeats: bool = True, shift_at: int | None = None) -> pl.DataFrame:
    """Irregular stamps (repeated ones too, with ``repeats``), signed values
    over several decades with nulls and an exact zero or two, and a level
    shift of +50 from row ``shift_at``."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([0.0, 0.5, 1.0, 1.0, 2.0, 3.5], n) if repeats else rng.uniform(0.2, 2.0, n)
    t = np.cumsum(steps)
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 3, n)
    if shift_at is not None:
        x[shift_at:] += 50.0
    x[rng.random(n) < 0.1] = np.nan
    x[rng.random(n) < 0.01] = 0.0
    return pl.DataFrame({"t": t, "x": x}).with_columns(pl.col("x").fill_nan(None))


def members(
    t: np.ndarray, i: int, *, window: float | None, closed: str, forward: bool
) -> np.ndarray:
    """The rows in row ``i``'s window, as Polars' rolling windows hold them:
    a set of timestamps, ``closed`` saying which ends are in."""
    near = closed in (("left", "both") if forward else ("right", "both"))
    far = closed in (("right", "both") if forward else ("left", "both"))
    d = t - t[i] if forward else t[i] - t
    keep = d >= 0 if near else d > 0
    if window is not None:
        keep &= d <= window if far else d < window
    return np.flatnonzero(keep)


def oracle(
    df: pl.DataFrame,
    q: float,
    *,
    half_life: float,
    window: float | None = None,
    closed: str = "right",
    forward: bool = False,
) -> list[float | None]:
    """Row by row, numpy's weighted quantile at ``q`` over the window's
    values, each weighing ``2 ** (-|t_j - t| / half_life)``. A forward window
    no later row has closed is unresolved: null."""
    t = df["t"].to_numpy()
    x = df["x"].to_numpy(allow_copy=True).astype(float)
    out: list[float | None] = []
    for i in range(len(t)):
        if forward:
            past = t > t[i] + window if closed in ("right", "both") else t >= t[i] + window
            if not past.any():
                out.append(None)
                continue
        m = members(t, i, window=window, closed=closed, forward=forward)
        m = m[~np.isnan(x[m])]
        if len(m) == 0:
            out.append(None)
            continue
        w = 2.0 ** (-np.abs(t[m] - t[i]) / half_life)
        out.append(float(np.quantile(x[m], q, weights=w, method="inverted_cdf")))
    return out


def within(got: list[float | None], want: list[float | None]) -> list[tuple[int, Any, Any]]:
    """The rows where the operator is not within the sketch's accuracy of the
    oracle, a null where the other has a value included. A magnitude below
    the smallest normal double reads 0."""
    bad = []
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if (g is None) != (w is None):
            bad.append((i, g, w))
        elif w is not None and g is not None:
            if abs(w) < 2.2250738585072014e-308:
                ok = g == 0.0
            else:
                ok = abs(g - w) <= ALPHA * abs(w) * (1 + 1e-12)
            if not ok:
                bad.append((i, g, w))
    return bad


CLOCK = {"clock": "t", "gap_cap": 1e9}


@pytest.mark.parametrize("q", [0.02, 0.25, 0.5, 0.8, 0.99])
def test_the_running_quantile_is_numpys_weighted_quantile(q: float) -> None:
    """With no window every row since the start counts, decayed: through a
    level shift, nulls, zeros and repeated stamps, every row is within the
    sketch's accuracy of numpy's weighted quantile, and the level shift
    shows (the median crosses to the new level within a few half-lives)."""
    df = stream(600, 1, shift_at=300)
    out = po.stream.with_windows(df, y=po.ewm_quantile("x", quantile=q, half_life=4.0), **CLOCK)[
        "y"
    ].to_list()
    assert within(out, oracle(df, q, half_life=4.0)) == []
    if q == 0.5:
        assert out[290] < 10 and out[-1] > 40, (out[290], out[-1])


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_a_windowed_quantile_is_numpys_over_the_rows_a_window_holds(closed: str, q: float) -> None:
    """A hard window holds the rows the other operators' windows hold, per
    ``closed``, each weighed by its decay: rows leaving it take their weight
    with them, so the quantile after the shift is the new level's alone."""
    df = stream(500, 2, shift_at=250)
    out = po.stream.with_windows(
        df,
        y=po.ewm_quantile("x", quantile=q, half_life=6.0, window_size=15.0, closed=closed),
        **CLOCK,
    )["y"].to_list()
    assert within(out, oracle(df, q, half_life=6.0, window=15.0, closed=closed)) == []
    # Once the window holds only rows after the shift, the median is the new
    # level's.
    if q == 0.5:
        shift = df["t"][250]
        rows = list(zip(out, df["t"], strict=True))
        before = [v for v, ti in rows if ti < shift and v is not None]
        after = [v for v, ti in rows if ti > shift + 15.0 and v is not None]
        assert abs(np.median(before)) < 5 and 45 < np.median(after) < 55


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_a_quantile_looking_ahead_is_numpys_over_the_rows_after(closed: str) -> None:
    """``reverse=True``: the rows after each row, at most ``window_size``
    later, each weighing ``lam**(t_j - t)``."""
    df = stream(400, 3, shift_at=200)
    out = po.stream.with_windows(
        df,
        y=po.ewm_quantile(
            "x", quantile=0.3, half_life=3.0, window_size=8.0, closed=closed, reverse=True
        ),
        **CLOCK,
    )["y"].to_list()
    want = oracle(df, 0.3, half_life=3.0, window=8.0, closed=closed, forward=True)
    assert within(out, want) == []
    assert sum(v is not None for v in out) > 300


def test_an_even_window_is_the_rolling_quantile_and_the_ew_mad_is_statsmodels_mad() -> None:
    """``half_life=inf`` weighs the window evenly. On a unit clock a window
    of 101 holds the last 101 rows, and the recipe -- the median of ``|r|``
    over ``Φ⁻¹(3/4)`` -- is statsmodels' MAD about 0 of those rows, within
    the sketch's accuracy; the constant is scipy's ``norm.ppf(0.75)``."""
    assert abs(norm.ppf(0.75) - Z75) <= 1e-15
    rng = np.random.default_rng(4)
    r = rng.standard_normal(1500) * 2.0
    r[rng.random(1500) < 0.05] = 40.0  # wild rows, a twentieth of them
    df = pl.DataFrame({"t": np.arange(1500.0), "r": r})
    out = po.stream.with_windows(
        df,
        scale=po.ewm_quantile(
            pl.col("r").abs(), quantile=0.5, half_life=math.inf, window_size=101.0
        )
        / Z75,
        **CLOCK,
    )["scale"].to_numpy()
    for i in range(100, 1500, 7):
        want = mad(r[i - 100 : i + 1], c=Z75, center=0.0)
        assert abs(out[i] - want) <= ALPHA * want * (1 + 1e-12), (i, out[i], want)
    # The wild rows barely move it: about 2.1, against a root mean square
    # of about 9.
    assert 1.8 < np.median(out[100:]) < 2.4
    assert np.sqrt(np.mean(r**2)) > 8


def test_a_scale_of_a_target_leaves_the_rows_own_label_out_under_closed_left() -> None:
    """Under ``closed="left"`` the row's own value is outside its window, so
    a weight from a target's scale never reads the label it weighs: a wild
    label does not move its own row's scale, and moves the next row's. (A
    row weighs ``1 - lam``, 0.067 of a window at a half-life of 10, so the
    0.95 quantile reaches one wild row.)"""
    t = np.arange(60.0)
    y = np.sin(t)
    y[40] = 1e6
    df = pl.DataFrame({"t": t, "y": y})
    kw = {"quantile": 0.95, "half_life": 10.0}
    out = po.stream.with_windows(
        df,
        left=po.ewm_quantile(pl.col("y").abs(), **kw, closed="left"),
        right=po.ewm_quantile(pl.col("y").abs(), **kw),
        **CLOCK,
    )
    left, right = out["left"].to_list(), out["right"].to_list()
    assert left[0] is None and left[40] < 2 and left[41] > 1e5
    assert right[40] > 1e5
    # The left window at row i is the right window at row i - 1.
    assert left[1:] == right[:-1]


def test_nulls_count_for_nothing_and_min_samples_counts_values() -> None:
    """A null row gives the window as it stands; ``min_samples`` nulls a
    window with fewer values."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [5.0, None, 1.0, None, 3.0]})
    out = po.stream.with_windows(
        df,
        a=po.ewm_quantile("x", quantile=0.5, half_life=math.inf),
        b=po.ewm_quantile("x", quantile=0.5, half_life=math.inf, min_samples=2),
        **CLOCK,
    )
    a, b = out["a"].to_list(), out["b"].to_list()
    assert within(a, [5.0, 5.0, 1.0, 1.0, 3.0]) == []
    assert b[:2] == [None, None] and within(b[2:], [1.0, 1.0, 3.0]) == []


def mixed() -> dict[str, pl.Expr]:
    h = {"half_life": 2.0}
    return {
        "run": po.ewm_quantile("x", quantile=0.5, **h),
        "win": po.ewm_quantile(pl.col("x").abs(), quantile=0.9, **h, window_size=5.0),
        "left": po.ewm_quantile("x", quantile=0.2, **h, window_size=4.0, closed="left"),
        "ahead": po.ewm_quantile("x", quantile=0.7, **h, window_size=4.0, reverse=True),
    }


def frame(n: int, seed: int) -> pl.DataFrame:
    df = stream(n, seed, shift_at=n // 2)
    g = np.random.default_rng(seed).integers(0, 3, n)
    return df.with_columns(g=pl.Series([f"g{k}" for k in g]))


@pytest.mark.parametrize("rows", [1, 7, 600])
def test_chunking_changes_no_bit(rows: int) -> None:
    df = frame(1500, 5)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **mixed(), **kw, chunk_size=100_000)
    out = po.stream.with_windows(df, **mixed(), **kw, chunk_size=rows)
    assert out.equals(one)
    # Each operator is valued on most rows: a window of 4 within one of three
    # groups is empty on many.
    assert all(one[c].null_count() < 0.6 * len(one) for c in mixed())


def test_a_save_and_load_at_any_row_is_one_run(tmp_path: Any) -> None:
    df = frame(300, 6)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **mixed(), **kw)
    for at in (1, 2, 37, 151, 299):
        state = tmp_path / f"q{at}.state"
        first = po.stream.with_windows(df[:at], **mixed(), **kw, save_state=state)
        second = po.stream.with_windows(df[at:], **mixed(), **kw, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_a_quantiles_parameters_are_checked_by_name() -> None:
    for bad, err, says in [
        ({"quantile": 0.0}, ValueError, "quantile must be above 0 and below 1, got 0.0"),
        ({"quantile": 1}, ValueError, "got 1"),
        ({"quantile": math.nan}, ValueError, "got nan"),
        ({"quantile": True}, TypeError, "quantile must be a number, got bool"),
        ({"quantile": "0.5"}, TypeError, "got str"),
    ]:
        with pytest.raises(err, match=says):
            po.ewm_quantile("x", half_life=1.0, **bad)
    with pytest.raises(TypeError, match=r"reverse=True\): window_size is required"):
        po.ewm_quantile("x", quantile=0.5, half_life=1.0, reverse=True)
    with pytest.raises(ValueError, match="not another window operator"):
        po.ewm_quantile(po.ewm_mean("x", half_life=1.0), quantile=0.5, half_life=1.0)
    with pytest.raises(TypeError):
        po.ewm_quantile("x", half_life=1.0)  # type: ignore[call-arg]


def test_two_levels_are_two_columns_and_one_asked_twice_is_one() -> None:
    a = po.ewm_quantile("x", quantile=0.5, half_life=1.0)
    b = po.ewm_quantile("x", quantile=0.5, half_life=1.0)
    c = po.ewm_quantile("x", quantile=0.6, half_life=1.0)
    assert a.meta.output_name() == b.meta.output_name() != c.meta.output_name()


def test_a_quantile_looking_ahead_is_a_target_the_bank_learns_as_its_column() -> None:
    """``reverse=True`` makes a target: the bank resolves it in its own
    window core, and learns each row once the window and the embargo have
    passed, giving what the same target as a column (``like=``) gives."""
    rng = np.random.default_rng(8)
    n = 400
    x0 = rng.standard_normal(n)
    df = pl.DataFrame(
        {"t": np.cumsum(rng.uniform(0.5, 1.5, n)), "x0": x0, "y": 2 * x0 + rng.standard_normal(n)}
    )
    fwd = po.ewm_quantile("y", quantile=0.5, half_life=3.0, window_size=6.0, reverse=True)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 50.0, "half_life": 40.0, "embargo": 8.0}
    spec = po.spec.ewridge("m", targets=[fwd.alias("fwd_med")], features=["x0"], **kw)
    by_expr = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_fwd_med")
    column = po.stream.with_windows(df, fwd.alias("fwd_med"), like=spec)
    by_col = po.spec.ewridge("c", targets=["fwd_med"], features=["x0"], **kw)
    by_column = po.ModelBank([by_col]).fit_predict(column)["c"].struct.field("pred_fwd_med")
    assert by_expr.drop_nulls().len() > 300
    assert by_expr.equals(by_column)
