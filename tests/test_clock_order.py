"""Obviously out-of-order rows are refused by default (design note of
2026-09-19).

It is easy to feed rows out of order by accident, and every policy but
``on_clock_reset = "error"`` absorbed a backwards clock into plausible, wrong
output. Two checks now refuse a backwards jump that is *obviously* not a
session boundary, whatever the policy says: a step back no larger than one
typical forward step (``backwards_jitter_ratio``, a transposed pair or a row
one tick late), and a second backwards jump within ``min_session_clock`` of the
previous one (default ``max_dclock``: a "session" shorter than one adjacency
gap is not a session). A single jump that then holds is a boundary and takes
the policy. The refusal is chunk-level, so the bank is untouched, and the
error names the rule and the key that disables it.
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


def spec(**kw):
    d = dict(
        targets=["y"],
        features=["x0"],
        halflife=10.0,
        clock="t",
        max_dclock=100.0,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


def swapped(df, i):
    """`df` with rows `i` and `i + 1` exchanged: one row one tick late."""
    order = list(range(df.height))
    order[i], order[i + 1] = order[i + 1], order[i]
    return df[order]


def test_a_transposed_pair_is_refused_as_jitter_and_names_the_key():
    with pytest.raises(ValueError, match="backwards_jitter_ratio") as e:
        po.ModelBank([spec()]).fit_predict(swapped(frame(), 20))
    msg = str(e.value)
    assert "out-of-order rows" in msg and "not updated" in msg and "row 21" in msg


def test_a_shuffled_frame_is_refused():
    df = frame(200, seed=3)
    df = df[np.random.default_rng(1).permutation(df.height).tolist()]
    with pytest.raises(ValueError, match="out-of-order rows"):
        po.ModelBank([spec()]).fit_predict(df)


def test_a_single_backwards_jump_that_holds_is_a_session_boundary():
    """Two days concatenated: the clock restarts once, then runs forward. That
    is a boundary, absorbed by the policy as before -- not refused."""
    day1 = frame(100, seed=0, start=0.0)
    day2 = frame(60, seed=1, start=0.0)
    out = po.ModelBank([spec()]).fit_predict(pl.concat([day1, day2]))
    assert out.height == 160


def test_two_boundaries_closer_than_min_session_clock_are_refused():
    """A jump back, three rows, and another jump back: the middle "session"
    spans 30 clock units against a `min_session_clock` of 100 (the
    `max_dclock`), so the second jump is out-of-order data."""
    a = frame(100, seed=0, start=0.0)  # 0 .. 990
    b = frame(4, seed=1, start=500.0)  # 500 .. 530: a boundary, then 3 rows
    c = frame(10, seed=2, start=100.0)  # a second jump back, too soon
    with pytest.raises(ValueError, match="min_session_clock") as e:
        po.ModelBank([spec()]).fit_predict(pl.concat([a, b, c]))
    assert "defaults to the larger of max_dclock and the halflife" in str(e.value)
    # Give the middle session room and the second jump is a boundary too.
    b = frame(20, seed=1, start=500.0)  # 500 .. 690: a span of 190 >= 100
    out = po.ModelBank([spec()]).fit_predict(pl.concat([a, b, c]))
    assert out.height == 130


def test_the_refused_chunk_leaves_the_bank_untouched():
    """A good chunk, a refused one, a good one: the bank continues from the
    first as if the second had never arrived -- bit for bit the same as a
    bank that only ever saw the two good chunks."""
    first, bad, third = (
        frame(40, seed=0),
        swapped(frame(40, seed=1, start=400.0), 10),
        frame(40, seed=2, start=800.0),
    )
    bank = po.ModelBank([spec()])
    bank.fit_predict(first)
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        bank.fit_predict(bad)
    got = bank.fit_predict(third)
    ref = po.ModelBank([spec()])
    ref.fit_predict(first)
    want = ref.fit_predict(third)
    assert got.equals(want)


def test_each_check_is_switched_off_with_zero():
    df = swapped(frame(), 20)
    # Jitter off: the late row takes the `max` policy as before.
    out = po.ModelBank([spec(backwards_jitter_ratio=0.0)]).fit_predict(df)
    assert out.height == df.height
    a = frame(100, seed=0, start=0.0)
    b = frame(4, seed=1, start=500.0)
    c = frame(10, seed=2, start=100.0)
    # Frequency rule off: two close boundaries take the policy as before.
    out = po.ModelBank([spec(min_session_clock=0.0)]).fit_predict(pl.concat([a, b, c]))
    assert out.height == 114


def test_the_checks_need_a_clock_and_a_finite_setting():
    with pytest.raises(ValueError, match="min_session_clock needs clock"):
        po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=10.0, min_session_clock=5.0)
    with pytest.raises(ValueError, match="backwards_jitter_ratio needs clock"):
        po.spec.ewridge(
            "m", targets=["y"], features=["x0"], halflife=10.0, backwards_jitter_ratio=2.0
        )
    for key in ("min_session_clock", "backwards_jitter_ratio"):
        with pytest.raises(ValueError, match=f"{key} must be finite"):
            spec(**{key: float("inf")})
        with pytest.raises(ValueError, match=key):
            spec(**{key: -1.0})


def test_the_error_policy_is_unchanged():
    """`"error"` refused every backwards jump before and still does, with its
    own message."""
    with pytest.raises(ValueError, match='on_clock_reset = "error"'):
        po.ModelBank([spec(on_clock_reset="error")]).fit_predict(
            pl.concat([frame(10, seed=0, start=0.0), frame(10, seed=1, start=0.0)])
        )


# --- edge cases ---------------------------------------------------------------


@pytest.mark.parametrize("policy", ["max", "zero", "reset_state"])
def test_every_absorbing_policy_yields_to_the_checks(policy):
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        po.ModelBank([spec(on_clock_reset=policy)]).fit_predict(swapped(frame(), 20))


def test_a_declared_session_change_may_step_the_clock_back():
    """With a `session` column the boundary is declared, and a small step back
    at the change is the session gap's business, not jitter; the same rows
    without the column are refused."""
    a = frame(30, seed=0, start=0.0)  # 0 .. 290
    b = frame(30, seed=1, start=285.0)  # 5 back from 290: jitter-sized
    df = pl.concat([a, b])
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        po.ModelBank([spec()]).fit_predict(df)
    declared = df.with_columns(s=pl.Series(["s0"] * 30 + ["s1"] * 30))
    out = po.ModelBank([spec(session="s", session_gap=1.0)]).fit_predict(declared)
    assert out.height == 60


def test_a_skipped_row_with_an_out_of_order_clock_is_still_refused():
    """A null feature skips the row's learning, not its clock: the row's place
    in the stream is still evidence of disorder, as it is under `"error"`."""
    late = swapped(frame(), 20).with_columns(
        pl.when(pl.int_range(pl.len()) == 21).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        po.ModelBank([spec()]).fit_predict(late)


def test_the_memory_survives_a_chunk_boundary_and_a_save():
    """Where the inferred session began and the typical step live in the clock
    state, so a second jump in a later chunk, or after a save and load, is
    still caught."""
    a = frame(100, seed=0, start=0.0)
    b = frame(4, seed=1, start=500.0)  # a boundary, 3 rows on
    c = frame(10, seed=2, start=100.0)  # too soon
    bank = po.ModelBank([spec()])
    bank.fit_predict(a)
    bank.fit_predict(b)
    with pytest.raises(ValueError, match="min_session_clock"):
        bank.fit_predict(c)
    bank = po.ModelBank([spec()])
    bank.fit_predict(a)
    bank.fit_predict(b)
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    with pytest.raises(ValueError, match="min_session_clock"):
        loaded.fit_predict(c)
    # The typical step too: a transposed pair after a load is jitter.
    bank = po.ModelBank([spec()])
    bank.fit_predict(frame(40, seed=0))
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        loaded.fit_predict(swapped(frame(40, seed=1, start=400.0), 10))


def test_predict_scores_rows_before_the_last_learned_clock():
    """The checks guard learning: scoring rows before the bank's clock is
    ordinary and takes the policy as it always did. Only `"error"` refuses
    there, as the user chose."""
    df = frame(60)
    bank = po.ModelBank([spec()])
    bank.fit_predict(df)
    assert bank.predict(df.slice(10, 20)).height == 20
    strict = po.ModelBank([spec(on_clock_reset="error")])
    strict.fit_predict(df)
    with pytest.raises(ValueError, match='on_clock_reset = "error"'):
        strict.predict(df.slice(10, 20))


def test_a_group_column_is_the_remedy_for_interleaved_streams():
    """Two sources interleaved row by row read as one clock that steps back on
    every other row; grouped, each has its own clock and nothing is out of
    order."""
    a = frame(30, seed=0, start=5.0).with_columns(g=pl.lit("a"))
    b = frame(30, seed=1, start=0.0).with_columns(g=pl.lit("b"))
    pairs = zip(a.iter_rows(named=True), b.iter_rows(named=True), strict=True)
    rows = [r for pair in pairs for r in pair]
    df = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="out-of-order rows"):
        po.ModelBank([spec()]).fit_predict(df)
    assert po.ModelBank([spec(group="g")]).fit_predict(df).height == 60


def test_a_first_delta_that_steps_back_is_a_boundary():
    """Before any forward step there is no typical step to judge by, so the
    very first delta, even backwards, is a boundary and takes the policy."""
    df = pl.concat([frame(1, seed=0, start=100.0), frame(20, seed=1, start=0.0)])
    assert po.ModelBank([spec()]).fit_predict(df).height == 21


def test_with_max_dclock_inf_the_default_falls_to_the_halflife():
    """`inf` is no scale, so the default is the halflife alone: 10 here, under
    which a span of 30 is two boundaries; set explicitly, it refuses. The
    jitter rule needs no scale and stays on."""
    a = frame(100, seed=0, start=0.0)
    b = frame(4, seed=1, start=500.0)
    c = frame(10, seed=2, start=100.0)
    df = pl.concat([a, b, c])
    assert po.ModelBank([spec(max_dclock=float("inf"))]).fit_predict(df).height == 114
    with pytest.raises(ValueError, match="min_session_clock"):
        po.ModelBank([spec(max_dclock=float("inf"), min_session_clock=100.0)]).fit_predict(df)
    with pytest.raises(ValueError, match="backwards_jitter_ratio"):
        po.ModelBank([spec(max_dclock=float("inf"))]).fit_predict(swapped(frame(), 20))


def test_a_gap_over_max_dclock_does_not_inflate_the_typical_step():
    """A forward gap over `max_dclock` is not a step, by the spec's own cap.
    Fed into the typical step, one weekend would refuse a real boundary in the
    rows after it as jitter against a typical step the data never shows."""
    a = frame(100, seed=0, start=0.0)  # 0 .. 990 in steps of 10
    b = frame(20, seed=1, start=1_000_000.0)  # a gap of 999010: capped
    c = frame(20, seed=2, start=999_690.0)  # 500 back: fifty steps, a boundary
    out = po.ModelBank([spec()]).fit_predict(pl.concat([a, b, c]))
    assert out.height == 140


def test_the_frequency_rule_defaults_to_the_larger_of_max_dclock_and_the_halflife():
    """A session shorter than one adjacency gap is not a session, and neither
    is one shorter than the halflife: with a halflife of 1000 against a
    `max_dclock` of 100, two boundaries 290 apart are out-of-order data. The
    default is the larger of the two scales, so where the halflife is the
    smaller one the floor holds and the same rows are two boundaries."""
    a = frame(100, seed=0, start=0.0)  # 0 .. 990
    b = frame(30, seed=1, start=500.0)  # 500 .. 790: a boundary, then a span of 290
    c = frame(10, seed=2, start=100.0)  # a second jump back
    df = pl.concat([a, b, c])
    with pytest.raises(ValueError, match="min_session_clock") as e:
        po.ModelBank([spec(halflife=1000.0)]).fit_predict(df)
    assert "larger of max_dclock and the halflife" in str(e.value)
    assert po.ModelBank([spec(halflife=10.0)]).fit_predict(df).height == 140


def test_the_default_reads_lam_as_a_halflife_too():
    """`lam` is the same scale under another name: 0.5 ** (1/1000) per clock
    unit is a halflife of 1000, and the default follows it."""
    a = frame(100, seed=0, start=0.0)
    b = frame(30, seed=1, start=500.0)
    c = frame(10, seed=2, start=100.0)
    df = pl.concat([a, b, c])
    with pytest.raises(ValueError, match="min_session_clock"):
        po.ModelBank([spec(halflife=None, lam=0.5 ** (1 / 1000.0))]).fit_predict(df)
