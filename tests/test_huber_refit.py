"""The Huber refit, ``huber_refit_rows`` on ``huber`` and on
``sgd(loss="huber")`` (task 277, docs/HUBER.md §3): each target's first rows
refit by iteratively reweighted least squares to convergence, the residual
scale rebuilt from the converged residuals, then the stream goes on online.

Held here: the refit lands on statsmodels' converged ``RLM`` on the same rows
(the third-party oracle), it removes the capture it is there for, chunk
invariance (hard rule 3), a save and a load before, at and after the refit,
the weight it reports (hard rule 8), ``0`` restoring the run without it, and
the refusals. That a state 1.3.0 saved goes on to the bit, and that ``0`` gives
1.3.0's numbers to the bit under ``huber_scale="rms"`` (under the MAD the anchored
sketch moves them, task 278), is held against the released wheel in
``tests/test_released_state.py``."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

import polars_online as po
from test_huber_scale import _frame, stream

TIER = "essential"
INF = float("inf")


def _model(kind: str, **kw) -> dict:
    """A one-target, one-feature spec of ``kind``: ``huber``, or ``sgd``
    under the Huber loss."""
    c = dict(targets=["y"], features=["x"], half_life=INF, **kw)
    if kind == "huber":
        return po.spec.huber("m", **c)
    return po.spec.sgd("m", loss="huber", **c)


def _coef(spec: dict, df: pl.DataFrame) -> np.ndarray:
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return np.array(bank.coef("m")["coef"].to_list())


@pytest.mark.parametrize("kind", ["huber", "sgd"])
@pytest.mark.parametrize("share", [0.05, 0.125])
def test_the_refit_is_statsmodels_converged_fit_on_the_buffer(kind, share):
    """The stream ends on the row the buffer fills, so the fit read after it
    is the refit's: statsmodels' ``RLM`` with the MAD scale on the same rows
    (an odd count, where the weighted median is ``np.median``), to the
    ridge's size (1e-6 on the mean-form system). The one-pass fit is far
    from it there, so the refit is what closes the distance: measured,
    ``huber``'s 2e-3 to 1e-1 and ``sgd``'s 1e-2 to 1e3 (the intercept,
    which a constant rate has not reached at a level of 1,000)."""
    n = 401
    df = stream(n, 2 * share, 0, level=1000.0)
    X = sm.add_constant(df["x"].to_numpy())
    rlm = sm.RLM(df["y"].to_numpy(), X, M=HuberT(1.345))
    oracle = rlm.fit(scale_est="mad", conv="coefs", tol=1e-12, maxiter=200).params
    refit = _coef(_model(kind, huber_refit_rows=n), df)
    plain = _coef(_model(kind, huber_refit_rows=0), df)
    assert np.abs(refit - oracle).max() < 1e-5, (refit, oracle)
    assert np.abs(plain - oracle).max() > 1e-3, (plain, oracle)


@pytest.mark.parametrize(
    ("kind", "rows", "seed", "share", "before", "after"),
    [("huber", 20_000, 110, 0.25, 0.4257, 0.0025), ("sgd", 2_000, 100, 0.4, 2.9735, 0.0349)],
)
def test_the_refit_frees_a_captured_stream(kind, rows, seed, share, before, after):
    """The capture the refit is there for (the module doc of
    ``huber_refit.rs``): with outliers and no decay, the one-pass fit of a
    captured stream ends far from statsmodels' converged ``RLM``, and with
    the default refit near it (the gaps in slope measured and pinned here:
    at 12.5% outliers 13 of 40 ``huber`` streams of 20,000 rows were
    captured; at 20%, 28 of 40 ``sgd`` streams of 2,000. At 12.5% all 40
    ``sgd`` streams were, seed 101's by 1.1057, before its intercept took
    the level from the first row, task 278 D-S; now the worst is 0.157).
    The one-pass gap of seed 100 read 2.9419 before its scale learned from
    the first rows, which ``min_weight`` withheld (task 278, D10). On the
    MAD sketch's fixed buckets, before it kept ``|e|`` in units of the first
    (task 278), the one-pass gaps read 0.4261 and 2.9756, and ``sgd``'s
    refitted one 0.0351."""
    df = stream(rows, share, seed, level=1000.0)
    X = sm.add_constant(df["x"].to_numpy())
    oracle = sm.RLM(df["y"].to_numpy(), X, M=HuberT(1.345)).fit(scale_est="mad").params
    plain = abs(_coef(_model(kind, huber_refit_rows=0), df)[1] - oracle[1])
    default = abs(_coef(_model(kind), df)[1] - oracle[1])
    assert plain == pytest.approx(before, abs=1e-4)
    assert default == pytest.approx(after, abs=1e-4)


def _specs() -> list[dict]:
    """Two targets, nulls, zero weights and a clock, the refit filling
    inside the stream: `huber` with a scale memory of its own and its scale
    emitted, and `sgd` standardized, under AdaGrad and under `inv_scaling`."""
    c = dict(
        targets=["y0", "y1"],
        features=["x"],
        clock="t",
        gap_cap=50.0,
        half_life=300.0,
        coef_every=1,
        weight="w",
        huber_refit_rows=200,
    )
    return [
        po.spec.huber("m", huber_scale_half_life=60.0, emit_huber_scale=True, **c),
        po.spec.sgd("m", loss="huber", schedule="adagrad", learning_rate=0.5, **c),
        po.spec.sgd("m", loss="huber", schedule="inv_scaling", learning_rate=0.5, **c),
    ]


IDS = ["huber", "sgd adagrad", "sgd inv_scaling"]


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
    """The buffer is state: saved and loaded before the refit (mid-buffer),
    around its row and after it, a bank goes on as the one that never
    stopped."""
    df = _frame()
    whole = po.ModelBank([spec]).fit_predict(df)
    for cut in (1, 37, 150, 220, 250, 1001):
        bank = po.ModelBank([spec])
        head = bank.fit_predict(df.slice(0, cut))
        loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
        tail = loaded.fit_predict(df.slice(cut))
        assert pl.concat([head, tail]).equals(whole, null_equal=True), cut


def _without_refit(spec: dict, rows: int | None) -> dict:
    model = {k: v for k, v in spec["model"].items() if k != "huber_refit_rows"}
    if rows is not None:
        model["huber_refit_rows"] = rows
    return {**spec, "model": model}


@pytest.mark.parametrize("spec", _specs(), ids=IDS)
def test_the_refit_moves_the_fit_after_its_row_and_not_the_weight(spec):
    """The activation check: with the refit on, the predictions part from the
    run without it only after a target's buffer fills -- here its 200th learned
    row -- and the weight the bank reports and gates on does not move (hard
    rule 8). The default at one feature is 500 rows, so a spec that does not
    name it is the spec at 500."""
    df = _frame()
    on = po.ModelBank([spec]).fit_predict(df)["m"]
    off = po.ModelBank([_without_refit(spec, 0)]).fit_predict(df)["m"]
    default = po.ModelBank([_without_refit(spec, None)]).fit_predict(df)["m"]
    at_500 = po.ModelBank([_without_refit(spec, 500)]).fit_predict(df)["m"]
    assert default.equals(at_500, null_equal=True)
    assert not default.equals(on, null_equal=True), "500 rows refit after 200"
    assert on.struct.field("weight_sum").equals(off.struct.field("weight_sum"))
    learned = df.with_row_index("i").filter(
        pl.col("y0").is_not_null() & pl.col("y0").is_finite() & (pl.col("w") > 0)
    )
    fills = int(learned["i"][199])
    a = on.struct.field("pred_y0").to_numpy()
    b = off.struct.field("pred_y0").to_numpy()
    differ = np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))
    assert differ.size > 0
    assert differ[0] > fills, (differ[0], fills)


def test_the_refit_solves_its_own_target_alone():
    """On the row a target's buffer fills, only that target's fit moves: the
    other target, present on every row, is refit on another row, and until
    then its predictions are the run without any refit's to the bit, the
    solve cadence included."""
    rng = np.random.default_rng(4)
    n = 600
    x = rng.standard_normal(n)
    y0 = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
    y0[(rng.random(n) < 0.3) & (x > 0)] += 50.0
    y0[np.arange(n) % 3 == 0] = np.nan  # y0 fills its buffer later than y1
    y1 = -3.0 * x + rng.standard_normal(n)
    df = pl.DataFrame({"x": x, "y0": y0, "y1": y1}).with_columns(pl.col("y0").fill_nan(None))
    c = dict(targets=["y0", "y1"], features=["x"], half_life=INF)
    spec = po.spec.huber("m", huber_refit_rows=100, **c)
    on = po.ModelBank([spec]).fit_predict(df)["m"]
    # Each target's refit alone: y1's at its 100th row, y0's at its 100th.
    off = po.ModelBank([po.spec.huber("m", huber_refit_rows=0, **c)]).fit_predict(df)["m"]
    p_on = on.struct.field("pred_y0").to_numpy()
    p_off = off.struct.field("pred_y0").to_numpy()
    y0_fills = int(np.flatnonzero(~np.isnan(y0))[99])
    assert y0_fills > 120, "y1's refit (row 99) falls well before y0's"
    same = (p_on[: y0_fills + 1] == p_off[: y0_fills + 1]) | np.isnan(p_off[: y0_fills + 1])
    assert same.all(), np.flatnonzero(~same)[:5]


def test_the_refit_is_refused_where_it_does_not_apply():
    with pytest.raises(ValueError, match='huber_refit_rows is for loss "huber"'):
        po.spec.sgd("s", targets=["y"], features=["x"], half_life=INF, huber_refit_rows=10)
    c = dict(targets=["y"], features=["x"], half_life=INF, ridge_scale="sum")
    with pytest.raises(ValueError, match="huber_refit_rows is for huber with a finite huber_delta"):
        po.ModelBank([po.spec.huber("h", huber_refit_rows=10, **c)])
    # Unnamed, the sum-scale prior has none, and takes the spec as it did.
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.5, 2.0]})
    off = po.ModelBank([po.spec.huber("h", huber_refit_rows=0, **c)]).fit_predict(df)
    assert po.ModelBank([po.spec.huber("h", **c)]).fit_predict(df).equals(off)


def test_the_largest_feature_set_is_the_refit_huber_on_its_features():
    """Under ``feature_sets`` a row's weight, the scale and so the refit read
    the largest set's residuals: a largest set of every feature is the
    ``huber`` on those features, refit included, to the bit; a smaller set
    is solved from the same rebuilt sums."""
    rng = np.random.default_rng(8)
    n = 800
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = 1000.0 + 2.0 * x - z + 0.5 * rng.standard_normal(n)
    y[(rng.random(n) < 0.25) & (x > 0)] += 50.0
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    c = dict(targets=["y"], half_life=INF, min_weight=3.0)
    sets = po.spec.huber(
        "m", features=["x", "z"], feature_sets={"x": ["x"], "all": ["x", "z"]}, **c
    )
    plain = po.spec.huber("m", features=["x", "z"], **c)
    a = po.ModelBank([sets]).fit_predict(df)["m"].struct.field("pred_y__all")
    b = po.ModelBank([plain]).fit_predict(df)["m"].struct.field("pred_y")
    assert a.equals(b, null_equal=True)
    off = po.ModelBank([{**plain, "model": {**plain["model"], "huber_refit_rows": 0}}])
    assert not off.fit_predict(df)["m"].struct.field("pred_y").equals(b, null_equal=True)


def test_the_default_count_grows_with_the_features_and_stops_above_100():
    """The default count is ``max(500, 10 * features)`` up to 100 features
    and none above: at 60 features the spec that does not name it is the
    spec at 600, which parts from the one at 500; at 101 it is the spec with
    the refit off, to the bit. A count given is taken at any width."""
    rng = np.random.default_rng(5)

    def frame(k: int, n: int) -> tuple[pl.DataFrame, list[str]]:
        X = rng.standard_normal((n, k))
        y = 1000.0 + X @ np.linspace(1.0, 2.0, k) + 0.5 * rng.standard_normal(n)
        y[(rng.random(n) < 0.25) & (X[:, 0] > 0)] += 50.0
        names = [f"x{i}" for i in range(k)]
        return pl.DataFrame({**{c: X[:, i] for i, c in enumerate(names)}, "y": y}), names

    def preds(df: pl.DataFrame, names: list[str], **kw) -> pl.Series:
        spec = po.spec.huber("m", targets=["y"], features=names, half_life=INF, **kw)
        return po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y")

    df, names = frame(60, 700)
    default = preds(df, names)
    assert default.equals(preds(df, names, huber_refit_rows=600), null_equal=True)
    assert not default.equals(preds(df, names, huber_refit_rows=500), null_equal=True)
    assert not default.equals(preds(df, names, huber_refit_rows=0), null_equal=True)
    df, names = frame(101, 260)
    default = preds(df, names)
    assert default.equals(preds(df, names, huber_refit_rows=0), null_equal=True)
    given = preds(df, names, huber_refit_rows=200)
    assert not given.equals(default, null_equal=True), "a count given is taken above 100"


@pytest.mark.parametrize("kind", ["huber", "sgd"])
def test_the_refit_budget_gives_up_the_later_groups_and_counts_them(kind):
    """``refit_budget`` bounds the buffers of all the spec's groups: each new
    group reserves ``(features + 20) * 8`` bytes a target for each row of the
    count when it first appears. At 50 rows, one feature and two targets that
    is 16,800 bytes a group, so 0.05 MiB holds three groups' buffers; the
    fourth to appear and every later one keep their one-pass fits, one
    notice says so, and ``summary()`` counts their targets. Each admitted
    group is the group a bank with no bound fits, to the bit, and each
    given-up group the group with the refit off."""
    rng = np.random.default_rng(6)
    groups, rows = 8, 120
    n = groups * rows
    x = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "g": np.tile(np.arange(groups), rows).astype(str),
            "x": x,
            "y": 10.0 + 2.0 * x + 0.5 * rng.standard_normal(n) + 8.0 * (rng.random(n) < 0.1),
            "z": -x + rng.standard_normal(n),
        }
    )
    build = po.spec.huber if kind == "huber" else po.spec.sgd
    loss = {} if kind == "huber" else {"loss": "huber"}
    c = dict(targets=["y", "z"], features=["x"], group="g", half_life=INF, coef_every=0, **loss)

    def bank(**kw) -> tuple[po.ModelBank, pl.DataFrame, list[str]]:
        b = po.ModelBank([build("m", huber_refit_rows=50, **c, **kw)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = b.fit_predict(df)
        return b, out, [str(w.message) for w in caught if "refit_budget" in str(w.message)]

    bounded, out, notices = bank(refit_budget=0.05)
    free, out_free, none = bank(refit_budget=INF)
    out_off = po.ModelBank([build("m", huber_refit_rows=0, **c)]).fit_predict(df)
    assert len(notices) == 1 and 'at group "3"' in notices[0], notices
    assert none == []
    given = dict(bounded.summary().select("group", "refit_given_up").iter_rows())
    assert given == {str(g): (0 if g < 3 else 2) for g in range(groups)}, given
    assert set(free.summary()["refit_given_up"]) == {0}
    for g in range(groups):
        mine = out.filter(pl.col("g") == str(g))["m"]
        like = (out_free if g < 3 else out_off).filter(pl.col("g") == str(g))["m"]
        assert mine.equals(like, null_equal=True), g
    with pytest.raises(ValueError, match="refit_budget must be > 0 MiB"):
        po.ModelBank([build("m", refit_budget=0.0, **c)])


def test_the_refit_budget_is_refused_beside_another_loss():
    with pytest.raises(ValueError, match='refit_budget is for loss "huber"'):
        po.spec.sgd("s", targets=["y"], features=["x"], half_life=INF, refit_budget=10.0)
