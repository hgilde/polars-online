"""Obviously out-of-order rows are refused by default (design note of
2026-09-19, reduced to one rule 2026-09-20).

It is easy to feed rows out of order by accident, and every policy but
``on_clock_reset = "error"`` absorbed a backwards clock into plausible, wrong
output. One check refuses a backwards jump that cannot be a session boundary,
whatever the policy says: ``max_dclock`` is the most two adjacent rows can be
apart and a session is longer than that, so a jump back by less than
``min_backwards_jump`` (default ``max_dclock``) is a late row, not a
boundary. A jump of at least that much takes the policy. The refusal is
chunk-level, so the bank is untouched, and the error names the jump, the
minimum and the key; ``0`` switches the check off.
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
    """Ten-unit steps under a cap of 100, so the minimum defaults to 100."""
    d = dict(targets=["y"], features=["x0"], halflife=10.0, clock="t", max_dclock=100.0)
    d.update(kw)
    return po.spec.ewridge("m", **d)


def spec1(**kw):
    """One-unit ticks under a cap of 300: five minutes of seconds."""
    d = dict(targets=["y"], features=["x0"], halflife=5000.0, clock="t", max_dclock=300.0)
    d.update(kw)
    return po.spec.ewridge("m", **d)


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


def test_a_transposed_pair_is_refused_and_names_the_key():
    with pytest.raises(ValueError, match="min_backwards_jump") as e:
        po.ModelBank([spec()]).fit_predict(swapped(frame(), 20))
    msg = str(e.value)
    assert "out-of-order rows" in msg and "not updated" in msg and "row 21" in msg
    assert "defaults to max_dclock" in msg


def test_a_row_late_by_less_than_max_dclock_is_refused():
    """Thirty seconds late, or four minutes, under a five-minute cap: adjacent
    rows are never that far apart, so the jump back cannot be a boundary."""
    for k in (30, 240):
        with pytest.raises(ValueError, match="min_backwards_jump"):
            po.ModelBank([spec1()]).fit_predict(late_by(ticks(), 500, k))


def test_a_jump_of_at_least_max_dclock_takes_the_policy():
    """A real day boundary jumps back by a session. A jump of exactly the
    minimum meets it (strict `<`), so it is a boundary too."""
    day = pl.concat([ticks(1000, 0, 34_200.0), ticks(1000, 1, 34_200.0)])
    assert po.ModelBank([spec1()]).fit_predict(day).height == 2000
    exact = pl.concat([frame(20, 0, 0.0), frame(20, 1, 90.0)])  # 190 -> 90: back by 100
    assert po.ModelBank([spec()]).fit_predict(exact).height == 40


def test_the_first_backwards_jump_is_judged_like_any_other():
    """No warmup and no first-jump exemption: the very first delta, back by
    less than the minimum, is refused; back by the minimum, it is a boundary."""
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec()]).fit_predict(pl.concat([frame(1, 0, 5.0), frame(10, 1, 0.0)]))
    df = pl.concat([frame(1, 0, 100.0), frame(10, 1, 0.0)])
    assert po.ModelBank([spec()]).fit_predict(df).height == 11


def test_a_shuffled_frame_is_refused():
    df = frame(200, seed=3)
    df = df[np.random.default_rng(1).permutation(df.height).tolist()]
    with pytest.raises(ValueError, match="out-of-order rows"):
        po.ModelBank([spec()]).fit_predict(df)


def test_a_single_backwards_jump_that_holds_is_a_session_boundary():
    """Two days concatenated: the clock restarts once, by a session's span,
    then runs forward. A boundary, absorbed by the policy as before."""
    day1 = frame(100, seed=0, start=0.0)
    day2 = frame(60, seed=1, start=0.0)
    assert po.ModelBank([spec()]).fit_predict(pl.concat([day1, day2])).height == 160


def test_the_refused_chunk_leaves_the_bank_untouched():
    first = frame(40, seed=0)
    bad = swapped(frame(40, seed=1, start=400.0), 10)
    third = frame(40, seed=2, start=800.0)
    bank = po.ModelBank([spec()])
    bank.fit_predict(first)
    with pytest.raises(ValueError, match="min_backwards_jump"):
        bank.fit_predict(bad)
    got = bank.fit_predict(third)
    ref = po.ModelBank([spec()])
    ref.fit_predict(first)
    assert got.equals(ref.fit_predict(third))


def test_the_minimum_is_a_setting_and_zero_switches_it_off():
    late = late_by(ticks(), 500, 30)
    assert po.ModelBank([spec1(min_backwards_jump=10.0)]).fit_predict(late).height == 1000
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec1(min_backwards_jump=10.0)]).fit_predict(late_by(ticks(), 500, 5))
    df = swapped(frame(), 20)
    assert po.ModelBank([spec(min_backwards_jump=0.0)]).fit_predict(df).height == df.height


def test_with_max_dclock_inf_the_check_is_off():
    """`inf` gives the check nothing to compare against, so the default is 0
    there and the policy absorbs every jump, a transposed pair included --
    the price of an unbounded cap, pinned so it is a choice. Explicit `inf`
    is no setting: to refuse every backwards jump, use `on_clock_reset =
    "error"`. An explicit finite value switches the check on under `inf`."""
    two_days = pl.concat([frame(100, seed=0, start=0.0), frame(60, seed=1, start=0.0)])
    unbounded = spec(max_dclock=float("inf"))
    assert po.ModelBank([unbounded]).fit_predict(two_days).height == 160
    assert po.ModelBank([unbounded]).fit_predict(swapped(frame(), 20)).height == 50
    with pytest.raises(ValueError, match="min_backwards_jump must be finite"):
        spec(min_backwards_jump=float("inf"))
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec(max_dclock=float("inf"), min_backwards_jump=100.0)]).fit_predict(
            swapped(frame(), 20)
        )


def test_the_check_needs_a_clock_and_a_non_negative_setting():
    with pytest.raises(ValueError, match="min_backwards_jump needs clock"):
        po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=10.0, min_backwards_jump=5.0)
    with pytest.raises(ValueError, match="min_backwards_jump"):
        spec(min_backwards_jump=-1.0)
    with pytest.raises(ValueError):
        spec(min_backwards_jump=float("nan"))


def test_the_removed_keys_are_gone():
    for key in ("min_session_clock", "backwards_jitter_ratio"):
        with pytest.raises(TypeError):
            spec(**{key: 0.0})


def test_the_error_policy_is_unchanged():
    with pytest.raises(ValueError, match='on_clock_reset = "error"'):
        po.ModelBank([spec(on_clock_reset="error")]).fit_predict(
            pl.concat([frame(10, seed=0, start=0.0), frame(10, seed=1, start=0.0)])
        )


@pytest.mark.parametrize("policy", ["max", "zero", "reset_state"])
def test_every_absorbing_policy_yields_to_the_check(policy):
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec(on_clock_reset=policy)]).fit_predict(swapped(frame(), 20))


def test_a_declared_session_change_may_step_the_clock_back():
    """With a `session` column the boundary is declared and the step back is
    the session gap's business; the same rows undeclared are refused."""
    a = frame(30, seed=0, start=0.0)  # 0 .. 290
    b = frame(30, seed=1, start=285.0)  # 5 back from 290: under the minimum
    df = pl.concat([a, b])
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec()]).fit_predict(df)
    declared = df.with_columns(s=pl.Series(["s0"] * 30 + ["s1"] * 30))
    assert po.ModelBank([spec(session="s", session_gap=1.0)]).fit_predict(declared).height == 60


def test_a_skipped_row_with_an_out_of_order_clock_is_still_refused():
    late = swapped(frame(), 20).with_columns(
        pl.when(pl.int_range(pl.len()) == 21).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    with pytest.raises(ValueError, match="min_backwards_jump"):
        po.ModelBank([spec()]).fit_predict(late)


def test_predict_scores_rows_before_the_last_learned_clock():
    """The check guards learning: scoring rows before the bank's clock takes
    the policy as it always did, and only `"error"` refuses there."""
    df = frame(60)
    bank = po.ModelBank([spec()])
    bank.fit_predict(df)
    assert bank.predict(df.slice(10, 20)).height == 20
    strict = po.ModelBank([spec(on_clock_reset="error")])
    strict.fit_predict(df)
    with pytest.raises(ValueError, match='on_clock_reset = "error"'):
        strict.predict(df.slice(10, 20))


def test_a_group_column_is_the_remedy_for_interleaved_streams():
    a = frame(30, seed=0, start=5.0).with_columns(g=pl.lit("a"))
    b = frame(30, seed=1, start=0.0).with_columns(g=pl.lit("b"))
    pairs = zip(a.iter_rows(named=True), b.iter_rows(named=True), strict=True)
    df = pl.DataFrame([r for pair in pairs for r in pair])
    with pytest.raises(ValueError, match="out-of-order rows"):
        po.ModelBank([spec()]).fit_predict(df)
    assert po.ModelBank([spec(group="g")]).fit_predict(df).height == 60
