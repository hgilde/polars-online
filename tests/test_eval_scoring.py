"""The scores' numbers at their edges, and their refusals (task 382a).

Each test here failed on main at 6befe4e9, from the review of `po.eval` of
2026-11-10 (`.claude/coord/review-eval/A-scoring.md` and `C-query-grid.md`,
finding IDs in each test's docstring). The oracles are scikit-learn's
`r2_score` and scipy's `pearsonr` where they compute the score, and exact
rational arithmetic (`fractions.Fraction`) for the sums themselves:

- a part of the rows that all weigh 0 adds its rows and nothing else to a
  merge, through `merge_sums` and through a query's chunks (A1 = C1);
- a target or a prediction that never varies reads the documented null,
  whatever the constant, the weights or the split (A2);
- `ic` is right at every scale inside the input bound of 1e100 (A3), at a
  far level (A5), and never past 1 (A16);
- a `Float32` target is scored in double precision on every path (A4 = C8),
  and a frame's scores are a one-chunk query's to the bit, `ic` too (A5);
- a key, clock or weight column named like a column the scores make, or an
  `Object` key, is refused by name on both paths (A6, A11, C9);
- a NaN or infinite float clock is in no window (A7); a narrow integer
  clock's window edges are refused where its dtype cannot hold them (A13);
- `merge_sums` and `from_sums` refuse a frame that is not a sums frame
  (A12); `min_samples` and `targets` are checked for kind (A14); a target
  that is not 0/1 is refused under `binary` (A15);
- the refusals of `compare_specs` and `grid_search` name the call made (C10).
"""

from __future__ import annotations

import decimal
import math
from fractions import Fraction

import numpy as np
import polars as pl
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import polars_online as po

TIER = "essential"


def _frame(y, pred, **more) -> pl.DataFrame:
    """A bank's output as the scores read it: the target ``y`` and a record
    column ``m`` holding ``pred_y``."""
    return pl.DataFrame({"y": y, **more}).with_columns(m=pl.struct(pl.Series("pred_y", pred)))


def _exact(y, p, w=None) -> dict[str, Fraction]:
    """The weighted moments in rational arithmetic, from their definitions."""
    w = [1.0] * len(y) if w is None else w
    ws, ys, ps = ([Fraction(v) for v in col] for col in (w, y, p))
    tw = sum(ws)
    my = sum(a * b for a, b in zip(ws, ys, strict=True)) / tw
    mp = sum(a * b for a, b in zip(ws, ps, strict=True)) / tw
    return {
        "w": tw,
        "m2_y": sum(a * (b - my) ** 2 for a, b in zip(ws, ys, strict=True)),
        "m2_pred": sum(a * (b - mp) ** 2 for a, b in zip(ws, ps, strict=True)),
        "cov": sum(a * (b - my) * (c - mp) for a, b, c in zip(ws, ys, ps, strict=True)),
    }


def _exact_ic(e: dict[str, Fraction]) -> float:
    """``cov / sqrt(m2_y * m2_pred)`` to 50 digits."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50

        def d(f: Fraction) -> decimal.Decimal:
            return decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator)

        return float(d(e["cov"]) / (d(e["m2_y"]) * d(e["m2_pred"])).sqrt())


@pytest.fixture
def chunks_of(monkeypatch):
    """Read a query ``rows`` rows at a time."""

    def set_rows(rows: int) -> None:
        monkeypatch.setattr(po.eval, "_chunk_rows", lambda: rows)

    return set_rows


# --- A1 = C1: a part that weighs nothing ------------------------------------


def _weighted(n=200, zero=slice(0, 50), seed=0, keys=False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    w = rng.uniform(0.2, 3.0, n)
    w[zero] = 0.0
    more = {"w": w}
    if keys:
        more["g"] = np.where(np.arange(n) % 2 == 0, "a", "b")
    return _frame(y, y + 0.5 * rng.standard_normal(n), **more)


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_a_part_of_zero_weight_adds_only_its_rows_to_a_merge(where):
    """A1: a part whose rows all weigh 0 turned the merged moments to NaN,
    and `from_sums` read `r2` and `ic` as NaN."""
    df = _weighted(
        zero={"first": slice(0, 50), "middle": slice(75, 125), "last": slice(150, 200)}[where]
    )
    cut = {"first": (50, 50), "middle": (75, 125), "last": (150, 150)}[where]
    parts = (
        [df[: cut[0]], df[cut[0] : cut[1]], df[cut[1] :]]
        if cut[0] != cut[1]
        else [df[: cut[0]], df[cut[0] :]]
    )
    parts = [p for p in parts if p.height]
    merged = po.eval.merge_sums(*(po.eval.sums(p, "m", weight="w") for p in parts))
    whole = po.eval.sums(df, "m", weight="w")
    assert merged["n"].to_list() == whole["n"].to_list()
    for c in ("w", "mean_y", "mean_pred", "m2_y", "m2_pred", "cov", "sse"):
        assert math.isfinite(merged[c][0]), (c, merged[c][0])
        assert merged[c][0] == pytest.approx(whole[c][0], rel=1e-13), c
    got = po.eval.from_sums(merged).row(0, named=True)
    sk = r2_score(df["y"], df["m"].struct.field("pred_y"), sample_weight=df["w"])
    assert got["r2"] == pytest.approx(sk, rel=1e-13)
    keep = df.filter(pl.col("w") > 0)
    want_ic = pearsonr(keep["y"], keep["m"].struct.field("pred_y")).statistic
    # The weighted correlation of the rows that weigh: numpy's, as the oracle.
    cov = np.cov(keep["y"], keep["m"].struct.field("pred_y"), aweights=keep["w"])
    assert got["ic"] == pytest.approx(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]), rel=1e-13)
    assert math.isfinite(want_ic)


def test_one_weighted_part_beside_zero_weight_parts_keeps_its_sums_to_the_bit():
    """A1: the part that carries the weight is the merge's, as a key in one
    part alone is."""
    df = _weighted(zero=slice(0, 100))
    weighted = po.eval.sums(df[100:], "m", weight="w")
    merged = po.eval.merge_sums(po.eval.sums(df[:50], "m", weight="w"), weighted,
                                po.eval.sums(df[50:100], "m", weight="w"))  # fmt: skip
    assert merged.drop("n").equals(weighted.drop("n"))
    assert merged["n"][0] == 200


def test_parts_that_all_weigh_nothing_still_read_null():
    df = _weighted(zero=slice(0, 200))
    merged = po.eval.merge_sums(po.eval.sums(df[:80], "m", weight="w"),
                                po.eval.sums(df[80:], "m", weight="w"))  # fmt: skip
    got = po.eval.from_sums(merged, min_samples=1).row(0, named=True)
    assert got["n"] == 200
    assert [got[c] for c in ("r2", "ic", "mse", "rmse")] == [None] * 4


def test_a_query_whose_chunk_weighs_nothing_is_the_frame(chunks_of):
    """C1: the same, with no `merge_sums` call in sight: a query of a frame
    whose first chunk (or one key's rows in it) all weigh 0."""
    chunks_of(50)
    df = _weighted(zero=slice(0, 50), keys=True)
    got = po.eval.sums(df.lazy(), "m", group="g", weight="w")
    want = po.eval.sums(df, "m", group="g", weight="w")
    assert got["n"].to_list() == want["n"].to_list()
    for c in ("w", "mean_y", "mean_pred", "m2_y", "m2_pred", "cov", "sse"):
        assert got[c].to_numpy() == pytest.approx(want[c].to_numpy(), rel=1e-13), c
    scores = po.eval.from_sums(got)
    assert scores["r2"].is_not_null().all() and scores["ic"].is_not_null().all()


# --- A2: a constant target or prediction ------------------------------------


def _constant_cases(seed: int, trials: int):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        c = float(
            rng.choice(
                [rng.uniform(-2, 2), rng.uniform(0, 1) * 1e-3, 100 + rng.uniform(0, 1),
                 0.01 * rng.integers(1, 100), 0.35, 0.1, 100.07]
            )
        )  # fmt: skip
        n = int(rng.integers(2, 400))
        yield c, n, rng.standard_normal(n), rng.uniform(0.2, 3, n), int(rng.integers(1, n))


def _all_paths(df: pl.DataFrame, k: int) -> dict[str, dict]:
    n = df.height
    weighted = [
        po.eval.sums(df.head(k), "m", weight="w"),
        po.eval.sums(df.tail(n - k), "m", weight="w"),
    ]
    return {
        "metrics": po.eval.metrics(df, "m", min_samples=1).row(0, named=True),
        "sums": po.eval.from_sums(po.eval.sums(df, "m"), min_samples=1).row(0, named=True),
        "weighted": po.eval.from_sums(po.eval.sums(df, "m", weight="w"), min_samples=1).row(
            0, named=True
        ),
        "merged": po.eval.from_sums(po.eval.merge_sums(*weighted), min_samples=1).row(
            0, named=True
        ),
    }


def test_a_constant_target_reads_null_r2_and_ic():
    """A2: the computed mean of a constant is not the constant in general,
    so its spread was rounding, about 1e-30, and `r2` read about -1e32."""
    for c, n, pred, w, k in _constant_cases(1, 80):
        df = _frame([c] * n, pred, w=w)
        for path, got in _all_paths(df, k).items():
            assert got["r2"] is None and got["ic"] is None, (path, c, n, k, got)


def test_a_constant_prediction_reads_null_ic():
    """A2: an intercept-only fit, or one whose features are all dropped,
    read an `ic` such as 0.14."""
    for c, n, y, w, k in _constant_cases(2, 80):
        df = _frame(y, [c] * n, w=w)
        for path, got in _all_paths(df, k).items():
            assert got["ic"] is None, (path, c, n, k, got)
            assert got["r2"] is not None, (path, c, n, k, got)


def test_a_constant_stays_constant_through_a_query_of_many_chunks(chunks_of):
    chunks_of(7)
    rng = np.random.default_rng(3)
    df = _frame([0.35] * 300, rng.standard_normal(300), w=rng.uniform(0.2, 3.0, 300))
    for kw in ({}, {"weight": "w"}):
        s = po.eval.sums(df.lazy(), "m", **kw)
        assert s["m2_y"][0] == 0.0 and s["mean_y"][0] == 0.35 and s["cov"][0] == 0.0, kw
    assert po.eval.metrics(df.lazy(), "m")["r2"][0] is None


def test_equal_means_merge_to_that_mean_exactly():
    """A2(b): parts whose means are all one value pool to that value, so
    the parallel-axis terms are exactly 0; a mean that differs keeps the
    weighted pooling."""
    rng = np.random.default_rng(4)
    a = _frame([0.1] * 40, rng.standard_normal(40), w=rng.uniform(0.2, 3, 40))
    b = _frame([0.1] * 70, rng.standard_normal(70), w=rng.uniform(0.2, 3, 70))
    m = po.eval.merge_sums(po.eval.sums(a, "m", weight="w"), po.eval.sums(b, "m", weight="w"))
    assert (m["mean_y"][0], m["m2_y"][0], m["cov"][0]) == (0.1, 0.0, 0.0)
    c = _frame([0.2] * 30, rng.standard_normal(30), w=rng.uniform(0.2, 3, 30))
    m = po.eval.merge_sums(m, po.eval.sums(c, "m", weight="w"))
    every = pl.concat([a, b, c])
    pred = every["m"].struct.field("pred_y").to_list()
    e = _exact(every["y"].to_list(), pred, every["w"].to_list())
    for f in ("m2_y", "m2_pred", "cov"):
        assert m[f][0] == pytest.approx(float(e[f]), rel=1e-13), f


# --- A3, A5, A16: ic ---------------------------------------------------------


@pytest.mark.parametrize(
    ("sy", "sp"),
    [(1e-90, 1e-90), (1e-80, 1e-80), (1e78, 1e78), (1.25e99, 1.25e99), (1e-60, 1e-110),
     (1e60, 1.25e99)],
)  # fmt: skip
def test_ic_is_scale_free_inside_the_input_bound(sy, sp):
    """A3: the product of the two moments overflowed to inf, and `ic` read
    0.0, or underflowed to 0, and `ic` read null; the roots are taken
    before the product."""
    rng = np.random.default_rng(0)
    z, e = rng.standard_normal((2, 2000))
    ref = pearsonr(z + 0.5 * e, z).statistic
    df = _frame(sy * z, sp * (z + 0.5 * e))
    for got in (po.eval.metrics(df, "m"), po.eval.from_sums(po.eval.sums(df, "m"))):
        assert got["ic"][0] == pytest.approx(ref, rel=1e-13), (sy, sp, got["ic"][0])
        assert got["r2"][0] == pytest.approx(r2_score(sy * z, sp * (z + 0.5 * e)), rel=1e-12)


@pytest.mark.parametrize("level", [1e6, 1e8, 1e10])
def test_a_frames_ic_at_a_level_is_the_exact_one(level):
    """A5: a frame's `ic` was Polars' `corr`, which loses the level over
    the spread times the rounding (4e-9 at 1e9, against 6e-15 from the
    centred sums)."""
    rng = np.random.default_rng(5)
    z, e = rng.standard_normal((2, 6000))
    y, p = level + z, level + z + 0.7 * e
    df = _frame(y, p)
    want = _exact_ic(_exact(list(y), list(p)))
    got = po.eval.metrics(df, "m")["ic"][0]
    # The centred sums' own rounding, second order in the level over the
    # spread (6e-15 at 1e9 measured); Polars' `corr` is first order.
    eps = 2.0**-52
    assert abs(got - want) <= (1e-14 + 50 * (eps * level) ** 2) * abs(want), (got, want)


def test_ic_never_passes_one():
    """A16: two rows that agree read `ic` 1.0000000000000002 from the
    rounding of the roots; it is clipped to [-1, 1]."""
    worst = 0.0
    rng = np.random.default_rng(6)
    for _ in range(300):
        y = rng.standard_normal(2) * 10.0 ** rng.integers(-5, 5)
        a = rng.uniform(0.1, 10)
        for sign in (1.0, -1.0):
            df = _frame(list(y), list(sign * a * y))
            for got in (po.eval.metrics(df, "m", min_samples=1),
                        po.eval.from_sums(po.eval.sums(df, "m"), min_samples=1)):  # fmt: skip
                ic = got["ic"][0]
                assert -1.0 <= ic <= 1.0, (y, a, ic)
                worst = max(worst, abs(ic))
    assert worst == 1.0


# --- A4 = C8, A5: a frame is a one-chunk query ----------------------------------


def test_a_float32_target_is_scored_in_double_precision():
    """A4: `metrics` on a frame squared a `Float32` target's deviations in
    single precision (1.7e-8 relative), and `from_sums` and a query did
    not."""
    rng = np.random.default_rng(7)
    y = (50_000 + rng.standard_normal(3000)).astype(np.float32)
    pred = y.astype(np.float64) + 0.5 * rng.standard_normal(3000)
    df = _frame(pl.Series(y, dtype=pl.Float32), pred)
    assert df.schema["y"] == pl.Float32
    got = po.eval.metrics(df, "m")
    assert got.equals(po.eval.from_sums(po.eval.sums(df, "m")))
    assert got.equals(po.eval.metrics(df.lazy(), "m"))
    assert got["r2"][0] == pytest.approx(r2_score(y.astype(np.float64), pred), rel=1e-14)


def test_a_float32_prediction_is_scored_in_double_precision():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(3000) + 300.0
    pred = (y + 0.5 * rng.standard_normal(3000)).astype(np.float32)
    df = _frame(y, pl.Series(pred, dtype=pl.Float32))
    got = po.eval.metrics(df, "m")
    assert got.equals(po.eval.from_sums(po.eval.sums(df, "m")))
    assert got["r2"][0] == pytest.approx(r2_score(y, pred.astype(np.float64)), rel=1e-14)
    assert got["ic"][0] == pytest.approx(pearsonr(y, pred.astype(np.float64)).statistic, rel=1e-14)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("group", [None, "g"])
def test_a_frames_scores_are_a_one_chunk_querys_to_the_bit(binary, group):
    """A5: every column, `ic` too, now that the frame's is read from the
    same centred sums."""
    rng = np.random.default_rng(9)
    n = 2000
    if binary:
        y = (rng.random(n) < 0.4).astype(float)
        pred = np.clip(0.4 + 0.3 * (y - 0.4) + 0.2 * rng.standard_normal(n), 0.01, 0.99)
    else:
        y = 1e6 + rng.standard_normal(n)
        pred = y + rng.standard_normal(n)
    df = _frame(y, pred, g=rng.choice(["a", "b", "c"], n))
    kw = {"group": group, "binary": binary}
    assert po.eval.metrics(df, "m", **kw).equals(po.eval.metrics(df.lazy(), "m", **kw))
    t = rng.integers(0, 1000, n)
    dfc = df.with_columns(t=pl.Series(np.sort(t)))
    wk = {"clock": "t", "every": 100, **kw}
    assert po.eval.window_metrics(dfc, "m", **wk).equals(
        po.eval.window_metrics(dfc.lazy(), "m", **wk)
    )


# --- A6, A11, C9: names the scores make, and Object keys ----------------------


TAKEN = ["y", "pred", "slot", "target", "n", "w", "mean_y", "cov", "sse", "hits", "r2", "ic",
         "hit_rate", "mse", "accuracy", "log_loss", "window_start"]  # fmt: skip


def _labelled(name: str) -> pl.DataFrame:
    rng = np.random.default_rng(10)
    n = 60
    ret = rng.standard_normal(n)
    return pl.DataFrame(
        {"ret": ret, name: rng.choice(["a", "b"], n), "t": np.arange(n), "wt": np.ones(n)}
    ).with_columns(m=pl.struct(pl.Series("pred_ret", ret + rng.standard_normal(n))))


@pytest.mark.parametrize("name", TAKEN)
@pytest.mark.parametrize("lazy", [False, True])
def test_a_key_named_like_a_column_the_scores_make_is_refused(name, lazy):
    """A6: a key named `y` or `pred` keyed the scores by the target's or
    the prediction's value, one row per input row; A11: a key named `n`,
    `w`, `slot`, `target` or `window_start` raised Polars' DuplicateError."""
    df = _labelled(name)
    src = df.lazy() if lazy else df
    calls = {
        "metrics": lambda: po.eval.metrics(src, "m", group=name),
        "sums": lambda: po.eval.sums(src, "m", group=name),
        "window_metrics": lambda: po.eval.window_metrics(src, "m", clock="t", every=10, group=name),
        "compare_specs": lambda: po.eval.compare_specs(src, ["m"], group=name),
    }
    for call, fn in calls.items():
        with pytest.raises(ValueError, match=f"po.eval.{call}: group column {name!r} is named"):
            fn()


@pytest.mark.parametrize("name", ["y", "pred", "slot", "window_start"])
def test_a_clock_or_weight_named_like_a_column_the_scores_make_is_refused(name):
    """A6: a clock or weight named like a column of the long form was
    replaced by it: the target's value read as the weight."""
    df = _labelled("g").with_columns(pl.col("t").alias(name))
    with pytest.raises(ValueError, match=f"clock column {name!r} is named"):
        po.eval.window_metrics(df, "m", clock=name, every=10)
    if name != "window_start":
        with pytest.raises(ValueError, match=f"weight column {name!r} is named"):
            po.eval.sums(df.lazy(), "m", weight=name)


def test_a_clock_or_weight_named_like_a_sum_is_read():
    """Only a name the long form takes is refused for a clock or a weight:
    neither is a column of the result, so ``w`` or ``n`` reads as it is."""
    df = _labelled("g").with_columns(w=pl.col("wt"), n=pl.col("t"))
    assert po.eval.sums(df, "m", weight="w").equals(po.eval.sums(df, "m", weight="wt"))
    assert po.eval.window_metrics(df, "m", clock="n", every=10, min_samples=1).equals(
        po.eval.window_metrics(df, "m", clock="t", every=10, min_samples=1)
    )


def test_a_key_named_spec_is_refused_by_compare_specs():
    df = _labelled("spec")
    with pytest.raises(ValueError, match="po.eval.compare_specs: group column 'spec' is named"):
        po.eval.compare_specs(df, ["m"], group="spec")


@pytest.mark.parametrize("lazy", [False, True])
def test_an_object_key_is_refused_on_both_paths(lazy):
    """C9: an `Object` key worked on a frame and failed on a query with
    Polars' internals (a hash join, a unique of objects)."""
    df = _labelled("g").with_columns(pl.Series("o", [object()] * 60, dtype=pl.Object))
    df = df.with_columns(n2=pl.struct(pl.Series("pred_ret", df["ret"] * 0.5),
                                      pl.Series("resid_ret", df["ret"] * 0.5)),
                         n1=pl.struct(pl.Series("pred_ret", df["ret"] * 0.2),
                                      pl.Series("resid_ret", df["ret"] * 0.8)))  # fmt: skip
    src = df.lazy() if lazy else df
    for call, fn in {
        "metrics": lambda: po.eval.metrics(src, "m", group="o"),
        "sums": lambda: po.eval.sums(src, "m", group="o"),
        "diebold_mariano": lambda: po.eval.diebold_mariano(src, a="n1", b="n2", group="o"),
        "seqtest": lambda: po.eval.seqtest(src, a="n1", b="n2", group="o"),
    }.items():
        with pytest.raises(TypeError, match=f"po.eval.{call}: group column 'o' is an Object"):
            fn()


# --- A7, A13, A16: clocks ------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_a_nan_or_infinite_float_clock_is_in_no_window(lazy, chunks_of):
    """A7: such rows were scored as windows labelled NaN, inf and -inf."""
    chunks_of(7)
    rng = np.random.default_rng(11)
    n = 40
    t = np.arange(n, dtype=float)
    t[[3, 17, 29]] = [np.nan, np.inf, -np.inf]
    df = _frame(rng.standard_normal(n), rng.standard_normal(n), t=t)
    got = po.eval.window_metrics(
        df.lazy() if lazy else df, "m", clock="t", every=10.0, min_samples=1
    )
    assert got["window_start"].to_list() == [0.0, 10.0, 20.0, 30.0]
    assert got["n"].sum() == n - 3


@pytest.mark.parametrize("lazy", [False, True])
def test_a_narrow_clock_whose_window_edge_it_cannot_hold_is_refused(lazy):
    """A13: an `Int8` clock with `every=200` raised Polars' cast error, and
    with `every=100` a window starting at -200 was labelled 56, wrapped."""
    rng = np.random.default_rng(12)
    t = pl.Series(np.arange(-120, 0, 2), dtype=pl.Int8)
    df = _frame(rng.standard_normal(60), rng.standard_normal(60), t=t)
    src = df.lazy() if lazy else df
    with pytest.raises(
        ValueError,
        match="po.eval.window_metrics: every is 200, which clock column 't', Int8, cannot hold",
    ):
        po.eval.window_metrics(src, "m", clock="t", every=200)
    with pytest.raises(
        ValueError, match="starts at -200, which clock column 't', Int8, cannot hold"
    ):
        po.eval.window_metrics(src, "m", clock="t", every=100)
    ok = po.eval.window_metrics(src, "m", clock="t", every=8, min_samples=1)
    assert ok.schema["window_start"] == pl.Int8 and ok["window_start"].min() == -120


# --- A12, A14, A15: refusals ---------------------------------------------------


def _sums_frame():
    rng = np.random.default_rng(13)
    return _frame(rng.standard_normal(50), rng.standard_normal(50), g=["a", "b"] * 25)


def test_a_metrics_frame_is_not_a_sums_frame():
    """A12: raw Polars errors, `ColumnNotFoundError: w`, or a `ValueError`
    naming the scores as keys."""
    df = _sums_frame()
    scores = po.eval.metrics(df, "m", min_samples=1)
    s = po.eval.sums(df, "m")
    for fn in (lambda: po.eval.merge_sums(scores), lambda: po.eval.merge_sums(s, scores),
               lambda: po.eval.from_sums(scores)):  # fmt: skip
        with pytest.raises(ValueError, match="holds scores, not sums.*po.eval.metrics"):
            fn()
    with pytest.raises(ValueError, match=r"po.eval.merge_sums: a frame lacks \['hits'\]"):
        po.eval.merge_sums(s, s.drop("hits"))
    other = po.eval.sums(df.with_columns(pl.col("g").cast(pl.Categorical)), "m", group="g")
    with pytest.raises(ValueError, match="po.eval.merge_sums: key column 'g' is String in one"):
        po.eval.merge_sums(po.eval.sums(df, "m", group="g"), other)


@pytest.mark.parametrize("value", [None, True, 20.5, "30", -5])
def test_min_samples_is_checked_for_kind(value):
    """A14: `None`, `True` and -5 scored every key, 20.5 thinned a key of
    20 rows, and "30" raised Polars' ComputeError."""
    df = _sums_frame()
    for call, fn in {
        "metrics": lambda: po.eval.metrics(df, "m", min_samples=value),
        "window_metrics": lambda: po.eval.window_metrics(
            df.with_row_index("t"), "m", clock="t", every=10, min_samples=value
        ),
        "compare_specs": lambda: po.eval.compare_specs(df, ["m"], min_samples=value),
        "from_sums": lambda: po.eval.from_sums(po.eval.sums(df, "m"), min_samples=value),
    }.items():
        with pytest.raises((TypeError, ValueError), match=f"po.eval.{call}: min_samples must be"):
            fn()


def test_a_bare_string_of_targets_is_one_target():
    """A14: `targets="ret"` was read letter by letter."""
    df = _labelled("g")
    assert po.eval.metrics(df, "m", targets="ret").equals(po.eval.metrics(df, "m", targets=["ret"]))
    assert po.eval.unpack(df, "m", targets="ret").equals(po.eval.unpack(df, "m", targets=["ret"]))


@pytest.mark.parametrize("lazy", [False, True])
def test_a_target_that_is_not_0_or_1_is_refused_under_binary(lazy, chunks_of):
    """A15: labels 0.5, 2 and -1 under `binary` read a negative log loss."""
    chunks_of(7)
    rng = np.random.default_rng(14)
    y = (rng.random(40) < 0.5).astype(float)
    y[23] = 2.0
    df = _frame(y, np.clip(rng.random(40), 0.01, 0.99), t=np.arange(40))
    src = df.lazy() if lazy else df
    for call, fn in {
        "metrics": lambda: po.eval.metrics(src, "m", binary=True),
        "sums": lambda: po.eval.sums(src, "m", binary=True),
        "window_metrics": lambda: po.eval.window_metrics(
            src, "m", clock="t", every=10, binary=True
        ),
    }.items():
        with pytest.raises(ValueError, match=f"po.eval.{call}: under binary .* is 2.0"):
            fn()
    ok = po.eval.metrics(df.with_columns(pl.col("y").clip(0.0, 1.0)), "m", binary=True)
    assert ok["log_loss"][0] > 0


# --- C10: whose refusal ---------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_compare_specs_refuses_in_its_own_name(lazy):
    df = _sums_frame()
    with pytest.raises(ValueError, match="po.eval.compare_specs: group column 'nope'"):
        po.eval.compare_specs(df.lazy() if lazy else df, ["m"], group="nope")
    with pytest.raises(KeyError, match='po.eval.compare_specs: spec "nope"'):
        po.eval.compare_specs(df.lazy() if lazy else df, ["m", "nope"])


def test_grid_search_refuses_a_chunk_size_in_its_own_name():
    rng = np.random.default_rng(15)
    df = pl.DataFrame({"x": rng.standard_normal(100), "y": rng.standard_normal(100)})
    spec = po.spec.ewridge("r", targets=["y"], features=["x"], half_life=50.0)
    for bad in (0, -3, 2.5, True):
        with pytest.raises(
            (TypeError, ValueError), match="po.eval.grid_search: chunk_size must be"
        ):
            po.eval.grid_search(df, spec, chunk_size=bad)


# --- C3(b), C5: grid_search's choice and what it ranks -------------------------


def test_the_choice_after_the_first_reads_the_test_spans_as_the_rank_does():
    """C3(b): every pick read from row 0, so a model whose first scored
    error was 11.55 stayed out of every choice though it was best on every
    test span (the review's stream, n = 257, seed 3)."""
    rng = np.random.default_rng(3)
    x0, x1 = rng.standard_normal((2, 257))
    df = pl.DataFrame({"x0": x0, "x1": x1, "y": x0 - 0.3 * x1 + rng.standard_normal(257)})
    specs = [
        po.spec.ewridge("one", targets=["y"], features=["x0"], half_life=50.0),
        po.spec.ewridge("two", targets=["y"], features=["x0", "x1"], half_life=50.0),
    ]
    table = po.eval.grid_search(df, specs, n_splits=5)
    chosen = dict(table.filter(kind="candidate").select("spec", "folds_chosen").iter_rows())
    assert chosen == {"one": 1, "two": 4}, chosen
    rank1 = table.filter(rank=1).row(0, named=True)
    assert rank1["spec"] == "two"
    # The rule's regret: with no gap the choice cannot beat the best in hindsight.
    assert table.filter(kind="choice")["mse"].item() >= rank1["mse"]


def test_loglik_over_specs_of_different_columns_is_refused():
    """C5(a): `bocpd` over `y`, over a column 1000 times its scale and over
    two columns were ranked 1, 3, 2 by their data's scale and dimension."""
    rng = np.random.default_rng(16)
    df = pl.DataFrame({"y": rng.standard_normal(300), "big": 1000 * rng.standard_normal(300),
                       "x0": rng.standard_normal(300)})  # fmt: skip
    specs = [po.spec.bocpd("on_y", features=["y"]), po.spec.bocpd("on_big", features=["big"]),
             po.spec.bocpd("on_y_too", features=["y"], hazard=50.0)]  # fmt: skip
    with pytest.raises(
        ValueError, match=r"po.eval.grid_search: scoring=\"loglik\" .*\['on_big'\] model \['big'\]"
    ):
        po.eval.grid_search(df, specs, scoring="loglik", n_splits=2)
    both = [
        po.spec.bocpd("a", features=["y", "x0"]),
        po.spec.bocpd("b", features=["x0", "y"], hazard=50.0),
    ]
    assert po.eval.grid_search(df, both, scoring="loglik", n_splits=2).height == 3


def test_one_target_name_read_from_two_columns_is_refused():
    """C5(b): `po.target("y", name="ret")` and `po.target("big", name="ret")`
    were ranked as one target, mse 1.06 against 1.04e6."""
    rng = np.random.default_rng(17)
    x = rng.standard_normal(300)
    df = pl.DataFrame(
        {"x": x, "y": x + rng.standard_normal(300), "big": 1000 * rng.standard_normal(300)}
    )
    specs = [
        po.spec.ewridge(
            "from_y", targets=[po.target("y", name="ret")], features=["x"], half_life=50.0
        ),
        po.spec.ewridge(
            "from_big", targets=[po.target("big", name="ret")], features=["x"], half_life=50.0
        ),
    ]
    with pytest.raises(
        ValueError, match="po.eval.grid_search: target 'ret' is read from different columns"
    ):
        po.eval.grid_search(df, specs, n_splits=2)
    also = po.target("y", name="ret")
    same = [specs[0], po.spec.ewridge("also_y", targets=[also], features=["x"], half_life=500.0)]
    assert po.eval.grid_search(df, same, n_splits=2).height == 3
