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
    assert "defaults to max_dclock" in str(e.value)
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
