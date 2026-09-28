"""A clock that steps back within a group (docs/PLAN.md task 120, decided by the
user on 2026-09-28).

It is easy to feed rows out of order by accident, and the policies that
absorbed a step back -- ``"max"``, which took the cap as the step, and
``"zero"`` -- turned it into plausible, wrong output. Both are gone.
``on_clock_reset = "error"``, the default, refuses every step back.
``"reset_state"`` starts the model over at a step back larger than
``min_backwards_jump``, which it requires, and refuses one no larger (a late
row: the comparison is inclusive). A refusal is chunk-level, so the bank is
untouched, and the error names the step, the row and the way out. Scoring
learns nothing, so ``predict`` scores a row before the last learned clock
against the state as it stands, under either policy.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

STEP = 10.0


def frame(n=50, seed=0, start=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "x0": x,
            "y": 2.0 * x + 0.1 * rng.standard_normal(n),
            "t": start + STEP * np.arange(float(n)),
        }
    )


def ticks(n=1000, seed=0, start=0.0):
    """One clock unit per row: a seconds-valued stream."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {"x0": x, "y": 2.0 * x + 0.1 * rng.standard_normal(n), "t": start + np.arange(float(n))}
    )


def spec(**kw):
    """Ten-unit steps under a cap of 100."""
    d = dict(targets=["y"], features=["x0"], halflife=10.0, clock="t", max_dclock=100.0)
    d.update(kw)
    return po.spec.ewridge("m", **d)


def spec1(**kw):
    """One-unit ticks under a cap of 300: five minutes of seconds."""
    d = dict(targets=["y"], features=["x0"], halflife=5000.0, clock="t", max_dclock=300.0)
    d.update(kw)
    return po.spec.ewridge("m", **d)


def restart(minimum, **kw):
    """`spec1` under `"reset_state"`: a step back past `minimum` starts over."""
    return spec1(on_clock_reset="reset_state", min_backwards_jump=minimum, **kw)


def swapped(df, i):
    """`df` with rows `i` and `i + 1` exchanged: one row one tick late."""
    order = list(range(df.height))
    order[i], order[i + 1] = order[i + 1], order[i]
    return df[order]


def late_by(df, i, k):
    """Row `i` delivered `k` rows late: one backwards jump of `k` clock units."""
    order = list(range(df.height))
    order.remove(i)
    order.insert(i + k, i)
    return df[order]


def resets(bank):
    return bank.summary()["resets"].sum()


def test_a_transposed_pair_is_refused_by_default_and_names_the_row():
    with pytest.raises(ValueError, match="goes backwards by 10 at row 21") as e:
        po.ModelBank([spec()]).fit_predict(swapped(frame(), 20))
    msg = str(e.value)
    assert 'on_clock_reset = "error", the default' in msg and "not updated" in msg
    assert "skip_learned" in msg and '"reset_state"' in msg


def test_the_default_refuses_every_step_back_however_large():
    """Thirty seconds late, four minutes, or a whole day: under `"error"` none
    is absorbed, the day boundary included -- that is `"reset_state"`'s to
    say, or a `session` column's."""
    for k in (30, 240):
        with pytest.raises(ValueError, match="goes backwards by"):
            po.ModelBank([spec1()]).fit_predict(late_by(ticks(), 500, k))
    day = pl.concat([ticks(1000, 0, 34_200.0), ticks(1000, 1, 34_200.0)])
    with pytest.raises(ValueError, match="goes backwards by 999 at row 1000"):
        po.ModelBank([spec1()]).fit_predict(day)


def test_reset_state_starts_over_past_the_minimum_and_refuses_a_late_row():
    day = pl.concat([ticks(1000, 0, 34_200.0), ticks(1000, 1, 34_200.0)])
    bank = po.ModelBank([restart(300.0)])
    out = bank.fit_predict(day)
    assert out.height == 2000 and resets(bank) == 1
    # The model starts over at the new day: the row after it has no weight.
    assert out["m"].struct.field("n_eff")[1000] == 0.0
    with pytest.raises(ValueError, match="no more than min_backwards_jump = 300") as e:
        po.ModelBank([restart(300.0)]).fit_predict(late_by(ticks(), 500, 30))
    assert "a late row" in str(e.value) and "row 530" in str(e.value)


def test_the_minimum_is_inclusive():
    """A step back as large as the minimum is a late row (task 120: a `Date`
    clock's one-day step back passed a one-day minimum, `<` being strict)."""
    exact = pl.concat([frame(20, 0, 0.0), frame(20, 1, 90.0)])  # 190 -> 90: back by 100
    with pytest.raises(ValueError, match="no more than min_backwards_jump = 100"):
        po.ModelBank([spec(on_clock_reset="reset_state", min_backwards_jump=100.0)]).fit_predict(
            exact
        )
    below = po.ModelBank([spec(on_clock_reset="reset_state", min_backwards_jump=99.5)])
    assert below.fit_predict(exact).height == 40 and resets(below) == 1


def test_the_first_backwards_jump_is_judged_like_any_other():
    """No warmup and no first-jump exemption."""
    m = dict(on_clock_reset="reset_state", min_backwards_jump=100.0)
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec(**m)]).fit_predict(pl.concat([frame(1, 0, 5.0), frame(10, 1, 0.0)]))
    df = pl.concat([frame(1, 0, 150.0), frame(10, 1, 0.0)])
    bank = po.ModelBank([spec(**m)])
    assert bank.fit_predict(df).height == 11 and resets(bank) == 1


def test_zero_starts_over_at_every_step_back():
    bank = po.ModelBank([spec(on_clock_reset="reset_state", min_backwards_jump=0.0)])
    assert bank.fit_predict(swapped(frame(), 20)).height == 50 and resets(bank) == 1


def test_a_shuffled_frame_is_refused():
    df = frame(200, seed=3)
    df = df[np.random.default_rng(1).permutation(df.height).tolist()]
    with pytest.raises(ValueError, match="goes backwards by"):
        po.ModelBank([spec()]).fit_predict(df)


def test_the_refused_chunk_leaves_the_bank_untouched():
    first = frame(40, seed=0)
    bad = swapped(frame(40, seed=1, start=400.0), 10)
    third = frame(40, seed=2, start=800.0)
    bank = po.ModelBank([spec()])
    bank.fit_predict(first)
    with pytest.raises(ValueError, match="goes backwards by"):
        bank.fit_predict(bad)
    got = bank.fit_predict(third)
    ref = po.ModelBank([spec()])
    ref.fit_predict(first)
    assert got.equals(ref.fit_predict(third))


def test_the_minimum_is_the_callers_to_give():
    """Required with `"reset_state"`, refused with `"error"`, and no default
    from the cap: what a late row is, only the caller knows."""
    with pytest.raises(ValueError, match="min_backwards_jump is required"):
        spec(on_clock_reset="reset_state")
    with pytest.raises(ValueError, match='applies only under on_clock_reset = "reset_state"'):
        spec(min_backwards_jump=5.0)
    with pytest.raises(ValueError, match="needs clock"):
        po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            halflife=10.0,
            on_clock_reset="reset_state",
            min_backwards_jump=5.0,
        )
    for bad in (-1.0, float("inf")):
        with pytest.raises(ValueError, match="min_backwards_jump must be finite"):
            spec(on_clock_reset="reset_state", min_backwards_jump=bad)
    with pytest.raises(ValueError, match="min_backwards_jump must not be NaN"):
        spec(on_clock_reset="reset_state", min_backwards_jump=float("nan"))


def test_the_removed_settings_are_refused():
    for key in ("min_session_clock", "backwards_jitter_ratio"):
        with pytest.raises(TypeError):
            spec(**{key: 0.0})
    for policy in ("max", "zero"):
        with pytest.raises(ValueError, match="unknown variant"):
            spec(on_clock_reset=policy)


def test_max_dclock_is_finite_and_above_zero():
    with pytest.raises(ValueError, match="max_dclock must be finite"):
        spec(max_dclock=float("inf"))
    with pytest.raises(ValueError, match='max_dclock must be > 0.*halflife = "inf"'):
        spec(max_dclock=0.0)


def test_session_gap_is_finite_or_reset():
    with pytest.raises(ValueError, match='session_gap must be finite.*"reset"'):
        spec(session="s", session_gap=float("inf"))


def test_the_audits_stream_never_hands_a_model_an_infinite_step():
    """The input task 120's audit reproduced: t = 0, 1, 2, 3, 1, 2, 3, 4. Under
    an infinite cap `"max"` handed the models a step of `+inf`, and `holt`
    predicted null for good. Now the cap is finite, the default refuses the
    step back, and `"reset_state"` starts over and goes on predicting."""
    t = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0] * 4
    t = [v + 10.0 * (i // 8) for i, v in enumerate(t)]
    df = pl.DataFrame({"t": t, "y": np.sin(np.arange(len(t)))})
    holt = dict(targets=["y"], clock="t", max_dclock=5.0, level_halflife=3.0, min_periods=1.0)
    with pytest.raises(ValueError, match="goes backwards by 2 at row 4"):
        po.ModelBank([po.spec.holt("h", **holt)]).fit_predict(df)
    bank = po.ModelBank(
        [po.spec.holt("h", on_clock_reset="reset_state", min_backwards_jump=0.0, **holt)]
    )
    pred = bank.fit_predict(df)["h"].struct.field("pred_y").to_numpy()
    assert np.isfinite(pred[-3:]).all(), pred
    assert resets(bank) == 4


def test_a_declared_session_change_may_step_the_clock_back():
    """With a `session` column the boundary is declared and the step back is
    the session gap's business; the same rows undeclared are refused."""
    a = frame(30, seed=0, start=0.0)  # 0 .. 290
    b = frame(30, seed=1, start=285.0)  # 5 back from 290
    df = pl.concat([a, b])
    with pytest.raises(ValueError, match="goes backwards by 5"):
        po.ModelBank([spec()]).fit_predict(df)
    declared = df.with_columns(s=pl.Series(["s0"] * 30 + ["s1"] * 30))
    assert po.ModelBank([spec(session="s", session_gap=1.0)]).fit_predict(declared).height == 60


def test_a_step_back_at_a_session_change_with_no_session_gap():
    """REVIEW-2026-09-18 V14: with `group_close = "session"` a session change
    needs no `session_gap`, and a step back there is neither refused nor a
    step: the group closes and the new session starts fresh."""
    a = frame(30, seed=0, start=0.0)
    b = frame(30, seed=1, start=100.0)  # 290 -> 100: back by 190
    df = pl.concat([a, b]).with_columns(s=pl.Series(["s0"] * 30 + ["s1"] * 30), g=pl.lit("a"))
    bank = po.ModelBank([spec(session="s", group="g", group_close="session")])
    out = bank.fit_predict(df)
    assert out.height == 60
    assert out["m"].struct.field("n_eff")[30] == 0.0
    assert bank.summary()["clock_backwards"].sum() == 0


def test_a_skipped_row_with_an_out_of_order_clock_is_still_refused():
    late = swapped(frame(), 20).with_columns(
        pl.when(pl.int_range(pl.len()) == 21).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    with pytest.raises(ValueError, match="goes backwards by"):
        po.ModelBank([spec()]).fit_predict(late)


@pytest.mark.parametrize(
    "policy",
    [{}, dict(on_clock_reset="reset_state", min_backwards_jump=0.0)],
    ids=["error", "reset_state"],
)
def test_predict_scores_rows_before_the_last_learned_clock_as_they_stand(policy):
    """Scoring learns nothing: a row before the last learned clock is scored
    against the state as it stands -- a step of 0 -- under either policy,
    never refused and never a fresh stream. `holt` extrapolates over the step,
    so its score of an early row is its score at the last clock."""
    df = pl.DataFrame({"t": np.arange(60.0), "y": 0.5 * np.arange(60.0)})
    holt = dict(targets=["y"], clock="t", max_dclock=5.0, level_halflife=10.0, **policy)
    bank = po.ModelBank([po.spec.holt("h", **holt)])
    bank.fit_predict(df)
    early = bank.predict(df.slice(10, 20))["h"].struct
    at_last = bank.predict(df.slice(59, 1).with_columns(t=pl.lit(59.0)))["h"].struct
    assert early.field("pred_y").is_finite().all()
    assert (early.field("pred_y") == at_last.field("pred_y")[0]).all()
    assert (early.field("n_eff") == at_last.field("n_eff")[0]).all()


def test_a_group_column_is_the_remedy_for_interleaved_streams():
    a = frame(30, seed=0, start=5.0).with_columns(g=pl.lit("a"))
    b = frame(30, seed=1, start=0.0).with_columns(g=pl.lit("b"))
    pairs = zip(a.iter_rows(named=True), b.iter_rows(named=True), strict=True)
    df = pl.DataFrame([r for pair in pairs for r in pair])
    with pytest.raises(ValueError, match="goes backwards by"):
        po.ModelBank([spec()]).fit_predict(df)
    assert po.ModelBank([spec(group="g")]).fit_predict(df).height == 60
