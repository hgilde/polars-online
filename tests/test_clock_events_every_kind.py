"""The clock's events hold on every kind (docs/PLAN.md task 278, batch 14).

The stream parameters that cut a stream -- ``restart_after_step_back``,
``session`` with ``session_gap``, ``group_close`` and ``group`` -- were
held to their definitions on the ten regressions only; the twelve other
kinds ran with none of them. Each definition is oracle-free, so each is a
relation between two runs of this library:

- **A step back is refused by default**, naming its row, and the bank is
  not updated (its saved bytes are unchanged).
- **A step back at or below ``restart_after_step_back`` is a late row**,
  refused the same way (the bound is inclusive), and a step back one ulp
  past it restarts. A tie (a step of 0) is no step back, even at 0.
- **A restart is a fresh run**: from the row that stepped back, every
  field equals a fresh bank fed only the rows from there, to the bit.
- **``session_gap="reset"`` is a fresh run per session**, likewise.
- **A numeric ``session_gap`` is the clock rewritten**: the same stream
  with no session and the clock's step at each boundary set to the gap
  equals it, to the bit, including a gap past ``gap_cap`` (capped in both).
- **``group_close``**: ``"monotone"`` moves no output field and closes each
  finished key once; ``"session"`` equals ``session_gap="reset"`` and
  closes each finished session once, each closed row the one
  ``"monotone"`` writes keyed on the session column.
- **``group`` is separate banks**: the rows of each key equal a bank fed
  that key's rows alone.

A relation whose two sides could agree because the event did nothing would
pass on a dead parameter, so each case prints, and asserts, evidence that
the event took effect: a restarted or reset run differs from the run that
carried on, and the summary counts the restarts; a numeric session gap
differs from the clock's own steps, or the kind is named as one whose
output never reads a step (``bocpd`` at a hazard in rows, which a duration
hazard on a temporal clock then shows reading it, and ``seqtest``, whose
e-process does not forget); the grouped and the ungrouped runs differ.
Where a kind's rows say little (``marginal``'s and ``rcov``'s hold only
``weight_sum``), the evidence is in the table or the closed rows it is
read from.

The clock is integer-valued (steps of 1 and some of 40), so a rewritten
clock is exact and the comparisons are to the bit.

Documented exceptions, each held to its own form rather than skipped:

- ``audit`` keeps its counts through a restart and a session reset ("an
  audit is a record of what the stream held, not a fit"), so its
  ``weight_sum`` carries on from before the event and every other field is
  the fresh run's; its tables show the runs of equal values starting over.
  It records the clock as fed, so a numeric ``session_gap`` moves none of
  its rows or tables (the rewritten clock moves its clock table). It
  refuses ``group_close``.
- ``rcov`` needs ``group`` and ``group_close="monotone"`` (its block is a
  group), so its runs keep them; its estimate is the row each group writes
  at its close, compared too. A session change splits its block whatever
  the gap, as a gap past ``gap_cap`` does, so a numeric ``session_gap``
  equals the clock rewritten to a step past the cap. A closed row's span
  counts (``rows_fed``, ``rows_learned``, ``clock_min``, ``clock_max``) are
  the summary's, counted since the group's state began, so a restart or
  reset inside the span leaves them counting every row.
- ``corrchange`` abandons its span, windows or cycle at a session change,
  as at a gap past the cap, so it too equals the clock rewritten to a step
  past the cap.

Essentials: ``ew_cov``, ``kmeans`` and ``marginal`` (moments with a clock,
a warm-up and seeding, and a state read from a table). Extended: every
other kind. ``huber`` waits for task 277, which changes its warm-up (its
refit must restart with the model); its leg builds and runs today
(``test_the_waiting_leg_is_ready``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_online as po
from test_model_registry import MINIMAL

TIER = "mixed"

N = 400
#: The row whose clock steps back, in the restart cases: inside rcov's second
#: group, since each group keeps its own clock.
BACK = 150
#: How far it steps back.
STEP_BACK = 10.0
#: The rows where the session changes: three sessions.
SESSIONS = (150, 280)
#: Above every step of the clock (the largest is 40), so no step is capped
#: unless a case makes one.
GAP_CAP = 50.0

ESSENTIAL = ("ew_cov", "kmeans", "marginal")
#: Task 277 changes its warm-up; it joins ``KINDS`` when that merges.
AFTER_TASK_277 = ("huber",)
KINDS = sorted(set(MINIMAL) - set(AFTER_TASK_277))
#: The kinds the regressions' own sweeps never ran ``group`` on.
OTHER_KINDS = sorted(
    {
        "ew_cov",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "corrchange",
        "bocpd",
        "hmm",
        "rcov",
        "audit",
    }
)


def _param(kind: str):
    if kind in ESSENTIAL:
        return kind
    return pytest.param(
        kind, marks=pytest.mark.extended(reason="a kind of the clock-event sweep beyond the three")
    )


EVERY = [_param(k) for k in KINDS]
OTHERS = [_param(k) for k in OTHER_KINDS]


def base() -> pl.DataFrame:
    r = np.random.default_rng(5)
    t = np.cumsum(np.where(r.random(N) < 0.05, 40.0, 1.0))
    t[0] = 0.0
    x0, x1 = r.normal(size=N), r.normal(size=N)
    y = 1.0 + 2.0 * x0 - x1 + 0.3 * r.normal(size=N) + np.where(np.arange(N) > 200, 3.0, 0.0)
    i = np.arange(N)
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": x1,
            "y": y,
            "s": np.searchsorted(SESSIONS, i, side="right"),
            # A key that only grows, 100 rows each: rcov's group, and the
            # group_close="monotone" cases'.
            "g": i // 100,
            # Interleaved keys at random, for group == separate banks.
            "gg": r.choice(["a", "b", "c"], size=N),
            "k": ["a"] * N,
        }
    )


def frame(kind: str) -> pl.DataFrame:
    df = base()
    if kind == "ftrl":
        df = df.with_columns(y=(pl.col("y") > 1).cast(pl.Float64))
    if kind == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 1).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


def stepped(kind: str) -> pl.DataFrame:
    """Row ``BACK``'s clock steps back by ``STEP_BACK``; the rows after it
    keep their own steps."""
    df = frame(kind)
    t = df["t"].to_numpy().copy()
    t[BACK:] = t[BACK:] - t[BACK] + (t[BACK - 1] - STEP_BACK)
    return df.with_columns(t=pl.Series(t))


def spec(kind: str, **kw: Any) -> dict:
    a: dict[str, Any] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": 50.0}
    a.update(MINIMAL[kind])
    if kind in ("kmeans", "hmm"):
        # Seeded inside the stream, so a restart has a warm-up to redo.
        a["warm_rows"] = 20
    if kind == "rcov":
        a["block_rows"] = 50
    a.update(kw)
    a = {k: v for k, v in a.items() if v is not None}
    build = getattr(po.spec, kind)
    try:
        # Every row a coefficient row, so `coef` is compared on each.
        return build("m", coef_every=0, **a)
    except ValueError as e:
        if "coef_every" not in str(e):
            raise
        # A kind that reports no coefficients refuses it.
        return build("m", **a)


def clocked(kind: str, **kw: Any) -> dict:
    return spec(kind, clock="t", gap_cap=GAP_CAP, **kw)


def run(bank: po.ModelBank, df: pl.DataFrame) -> pl.DataFrame:
    with warnings.catch_warnings():
        # A fresh start's readiness notices are expected here.
        warnings.simplefilter("ignore")
        return bank.fit_predict(df).select("m").unnest("m")


def fit(s: dict, df: pl.DataFrame) -> tuple[pl.DataFrame, po.ModelBank]:
    bank = po.ModelBank([s], threads=1)
    return run(bank, df), bank


def closed(bank: po.ModelBank) -> pl.DataFrame:
    return bank.closed_groups("m")


def differs(a: pl.DataFrame, b: pl.DataFrame) -> list[str]:
    """The fields on which two runs of equal length differ."""
    return [c for c in a.columns if not a[c].equals(b[c], null_equal=True)]


def held(kind: str, event: pl.DataFrame, fresh: pl.DataFrame, carried_weight: float) -> None:
    """``event`` (the rows from a restart or a reset) equals ``fresh``; for
    ``audit``, every field but ``weight_sum``, which carries the counts."""
    if kind == "audit":
        assert_frame_equal(event.drop("weight_sum"), fresh.drop("weight_sum"), check_exact=True)
        np.testing.assert_array_equal(
            event["weight_sum"].to_numpy(), fresh["weight_sum"].to_numpy() + carried_weight
        )
        return
    assert_frame_equal(event, fresh, check_exact=True)


# ---------------------------------------------------------------------------
# restart_after_step_back


@pytest.mark.parametrize("kind", EVERY)
def test_a_step_back_is_refused_by_default(kind):
    """Unset, a step back is refused, naming its row, and the bank keeps the
    state it had before the call."""
    s = clocked(kind)
    df = stepped(kind)
    bank = po.ModelBank([s], threads=1)
    run(bank, df.head(100))
    before = bank.save_bytes()
    with pytest.raises(ValueError, match=f"goes backwards by 10 at row {BACK - 100} "):
        run(bank, df.slice(100))
    assert bank.save_bytes() == before
    # And the stream carries on from where it was: the rows refused were
    # never learned.
    one, _ = fit(s, frame(kind))
    assert_frame_equal(run(bank, frame(kind).slice(100)), one.slice(100), check_exact=True)


@pytest.mark.parametrize("kind", EVERY)
def test_a_step_back_at_or_below_the_threshold_is_a_late_row(kind):
    """A step back no larger than ``restart_after_step_back`` is refused as
    a late row; at exactly the bound too (it is inclusive). One ulp past it,
    the stream restarts there, as at a bound of 0."""
    df = stepped(kind)
    for bound in (STEP_BACK, 2 * STEP_BACK):
        bank = po.ModelBank([clocked(kind, restart_after_step_back=bound)], threads=1)
        before = bank.save_bytes()
        with pytest.raises(
            ValueError, match=f"goes backwards by 10 at row {BACK}, no more than restart"
        ):
            run(bank, df)
        assert bank.save_bytes() == before
    below, _ = fit(clocked(kind, restart_after_step_back=math.nextafter(STEP_BACK, 0.0)), df)
    zero, _ = fit(clocked(kind, restart_after_step_back=0.0), df)
    assert_frame_equal(below, zero, check_exact=True)


@pytest.mark.parametrize("kind", EVERY)
def test_a_tie_is_no_step_back(kind):
    """A step of 0 is no step back: at ``restart_after_step_back=0`` a
    repeated clock value neither restarts nor is refused."""
    df = frame(kind)
    t = df["t"].to_numpy().copy()
    t[BACK:] -= t[BACK] - t[BACK - 1]
    tied = df.with_columns(t=pl.Series(t))
    zero, bank = fit(clocked(kind, restart_after_step_back=0.0), tied)
    unset, _ = fit(clocked(kind), tied)
    assert_frame_equal(zero, unset, check_exact=True)
    if kind != "rcov":  # its summary keeps only the group still open
        assert bank.summary()["resets"].sum() == 0


@pytest.mark.parametrize("kind", EVERY)
def test_a_restart_is_a_fresh_run(kind):
    """From the row that stepped back, every field equals a fresh bank fed
    only the rows from there; so does the state a kind is read from."""
    s = clocked(kind, restart_after_step_back=0.0)
    df = stepped(kind)
    out, bank = fit(s, df)
    fresh, fresh_bank = fit(clocked(kind), df.slice(BACK))
    # audit counts one a row, so its count carries the rows before.
    carried = float(BACK) if kind == "audit" else 0.0
    held(kind, out.slice(BACK), fresh, carried)
    if kind == "marginal":
        assert_frame_equal(bank.marginal("m"), fresh_bank.marginal("m"), check_exact=True)
    if kind == "rcov":
        # Group 0 closed before the restart; the restart dropped group 1's
        # rows before it, which a fresh bank never saw.
        mine = closed(bank)
        held_closed(mine.filter(pl.col("group") != "0"), closed(fresh_bank))
    carried_on, carried_bank = fit(clocked(kind), frame(kind))
    evidence(kind, "restart", carried_on.slice(BACK), fresh, bank, carried_bank, events=1)


# ---------------------------------------------------------------------------
# session and session_gap


def sessions(df: pl.DataFrame) -> list[tuple[int, int]]:
    edges = [0, *SESSIONS, df.height]
    return list(zip(edges[:-1], edges[1:], strict=True))


@pytest.mark.parametrize("kind", EVERY)
def test_a_session_reset_is_a_fresh_run_per_session(kind):
    """Under ``session_gap="reset"`` each session's rows equal a fresh bank
    fed that session alone."""
    s = clocked(kind, session="s", session_gap="reset")
    df = frame(kind)
    out, bank = fit(s, df)
    closes = []
    for a, b in sessions(df):
        fresh, fresh_bank = fit(s, df.slice(a, b - a))
        carried = float(a) if kind == "audit" else 0.0
        held(kind, out.slice(a, b - a), fresh, carried)
        if kind == "rcov":
            closes.append(closed(fresh_bank))
    if kind == "marginal":
        assert_frame_equal(bank.marginal("m"), fresh_bank.marginal("m"), check_exact=True)
    if kind == "rcov":
        # The groups each session closed, a straddling group's rows before
        # the change dropped.
        held_closed(closed(bank), pl.concat(closes))
    carried_on, carried_bank = fit(clocked(kind), df)
    evidence(
        kind,
        "reset",
        carried_on.slice(SESSIONS[0]),
        out.slice(SESSIONS[0]),
        bank,
        carried_bank,
        events=len(SESSIONS),
    )


#: The columns of a closed row that report the span's own summary: counted
#: since the group's state began, undecayed, as ``ModelBank.summary`` counts
#: them, so a restart or a reset inside the span does not start them over.
SPAN_COUNTS = ("rows_fed", "rows_learned", "clock_min", "clock_max")


def held_closed(mine: pl.DataFrame, fresh: pl.DataFrame) -> None:
    """rcov's closed rows equal a fresh bank's, every estimate to the bit;
    the span's counts still cover every row the group was fed (100)."""
    assert_frame_equal(mine.drop(SPAN_COUNTS), fresh.drop(SPAN_COUNTS), check_exact=True)
    print("closed", mine["group"].to_list(), "rows_fed", mine["rows_fed"].to_list())
    assert (mine["rows_fed"] == 100).all()


def evidence(
    kind: str,
    event: str,
    carried_on: pl.DataFrame,
    after: pl.DataFrame,
    bank: po.ModelBank,
    carried_bank: po.ModelBank,
    events: int,
) -> None:
    """The event took effect: the run that carried on through it differs
    from the rows after it. audit's rows carry its counts, so its trace is
    in its tables: the runs of equal values start over at each event."""
    if kind == "audit":
        adjacent = bank.audit("m")["adjacent"].to_list()
        whole = carried_bank.audit("m")["adjacent"].to_list()
        print(f"audit: adjacent rows {adjacent} after the {event}, {whole} without")
        assert adjacent == [n - events for n in whole]
        return
    moved = differs(carried_on, after)
    print(f"{kind}: without the {event}, {len(moved)} fields differ: {moved}")
    assert moved
    if kind == "marginal":
        # Its rows hold only weight_sum; the fit is its table.
        table = differs(carried_bank.marginal("m"), bank.marginal("m"))
        print(f"marginal: without the {event}, its table's {table} differ")
        assert {"corr", "beta"} <= set(table)
    if kind != "rcov":  # its summary keeps only the group still open
        print("summary resets", bank.summary()["resets"].to_list())
        assert bank.summary()["resets"].sum() == events


#: The kinds whose output reads no clock step, at any size: ``seqtest``'s
#: e-process does not forget, and ``bocpd``'s hazard counts rows unless it is
#: a duration (``test_bocpd_duration_hazard_reads_the_session_gap``). The
#: gap's only trace on them is the session counted.
STEP_BLIND: frozenset[str] = frozenset({"bocpd", "seqtest"})
#: The kinds a session change breaks whatever its gap, as a gap past
#: ``gap_cap`` does: ``rcov``'s block is split there (no product pairs two
#: returns across it), and ``corrchange`` abandons its span, windows or cycle.
BREAK_AT_A_SESSION: frozenset[str] = frozenset({"rcov", "corrchange"})


def rewrite(df: pl.DataFrame, step: float) -> pl.DataFrame:
    """No session; the clock's step at each session change set to ``step``."""
    t = df["t"].to_numpy()
    steps = np.diff(t, prepend=t[0])
    assert (steps[list(SESSIONS)] != step).all(), "the rewrite must move the clock"
    steps[list(SESSIONS)] = step
    return df.with_columns(t=pl.Series(np.cumsum(steps)))


@pytest.mark.parametrize("gap", [0.0, 3.0, 2 * GAP_CAP], ids=["0", "3", "past-the-cap"])
@pytest.mark.parametrize("kind", EVERY)
def test_a_numeric_session_gap_is_the_clock_rewritten(kind, gap):
    """``session_gap=g`` equals the same stream with no session and the
    clock's step at each session change set to ``g``. A ``g`` past
    ``gap_cap`` is capped, as the rewritten step is.

    audit records the clock as it was fed, so its rows and tables equal the
    run with no session at all; and against the rewritten clock its clock
    table differs, so the rewrite was visible to it."""
    df = frame(kind)
    out, bank = fit(clocked(kind, session="s", session_gap=gap), df)
    own, own_bank = fit(clocked(kind), df)
    if kind == "audit":
        assert_frame_equal(out, own, check_exact=True)
        for table in ("columns", "clock"):
            assert_frame_equal(
                bank.audit("m", table=table), own_bank.audit("m", table=table), check_exact=True
            )
        _, plain_bank = fit(clocked(kind), rewrite(df, gap))
        moved = differs(own_bank.audit("m", table="clock"), plain_bank.audit("m", table="clock"))
        print(f"audit, gap {gap}: the rewritten clock moves its clock table's {moved}")
        assert moved
        return
    at = max(gap, 2 * GAP_CAP) if kind in BREAK_AT_A_SESSION else gap
    plain, plain_bank = fit(clocked(kind), rewrite(df, at))
    assert_frame_equal(out, plain, check_exact=True)
    if kind == "marginal":
        assert_frame_equal(bank.marginal("m"), plain_bank.marginal("m"), check_exact=True)
    if kind == "rcov":
        # clock_min and clock_max report the clock column as fed.
        mine = closed(bank).drop("clock_min", "clock_max")
        assert_frame_equal(
            mine, closed(plain_bank).drop("clock_min", "clock_max"), check_exact=True
        )
        # Its rows hold only weight_sum; the split shows in the estimates.
        moved = differs(closed(own_bank).drop("clock_min", "clock_max"), mine)
        print(f"rcov, gap {gap}: against the clock's own steps, the closed rows' {moved} differ")
        assert moved
        return
    if kind != "rcov":  # its summary keeps only the group still open
        assert bank.summary()["session_changes"].sum() == len(SESSIONS)
    moved = differs(own, out)
    print(f"{kind}, gap {gap}: against the clock's own steps, {len(moved)} fields differ: {moved}")
    if kind not in STEP_BLIND:
        assert moved


@pytest.mark.extended(reason="bocpd's temporal leg of the numeric session gap")
def test_bocpd_duration_hazard_reads_the_session_gap():
    """With its hazard a duration, bocpd reads the clock's step, so there the
    numeric gap is seen: it equals the rewritten clock and differs from the
    clock's own steps."""
    df = frame("bocpd")
    origin = pl.datetime(2024, 1, 1)

    def temporal(d: pl.DataFrame) -> pl.DataFrame:
        return d.with_columns(ts=origin + pl.duration(seconds=pl.col("t")))

    def bocpd(**kw: Any) -> dict:
        return spec("bocpd", clock="ts", gap_cap="50s", hazard="100s", **kw)

    out, _ = fit(bocpd(session="s", session_gap="3s"), temporal(df))
    plain, _ = fit(bocpd(), temporal(rewrite(df, 3.0)))
    own, _ = fit(bocpd(), temporal(df))
    assert_frame_equal(out, plain, check_exact=True)
    moved = differs(own, out)
    print(f"bocpd, a duration hazard: {len(moved)} fields differ from the own clock: {moved}")
    assert moved


# ---------------------------------------------------------------------------
# group_close


def _name(p) -> str:
    return p if isinstance(p, str) else p.values[0]


#: ``audit`` refuses ``group_close``; ``rcov`` requires one, so it has no
#: open run to compare ``"monotone"`` with.
MONOTONE = [p for p in EVERY if _name(p) not in ("audit", "rcov")]
SESSION_CLOSE = [p for p in EVERY if _name(p) != "audit"]


@pytest.mark.parametrize("kind", MONOTONE)
def test_monotone_close_moves_no_output(kind):
    """``group_close="monotone"`` moves no output field, and closes each key
    once a larger one arrives: 0, 1 and 2, with 3 still open."""
    df = frame(kind)
    open_, _ = fit(spec(kind, group="g"), df)
    out, bank = fit(spec(kind, group="g", group_close="monotone"), df)
    assert_frame_equal(out, open_, check_exact=True)
    rows = closed(bank)
    print(f"{kind}: {rows.height} closed rows, keys {rows['group'].unique().sort().to_list()}")
    assert rows["group"].unique(maintain_order=True).to_list() == ["0", "1", "2"]
    assert closed(bank).height == 0, "read once, the closed rows are gone"


@pytest.mark.parametrize("kind", SESSION_CLOSE)
def test_session_close_is_a_session_reset(kind):
    """``group_close="session"`` gives the output ``session_gap="reset"``
    does, and closes each finished session: two, with the third open. Each
    closed row is the one ``"monotone"`` writes keyed on the session column,
    which closes the same spans."""
    df = frame(kind)
    # rcov needs a close, and "monotone" on one key never closes it.
    keep = dict(group_close="monotone") if kind == "rcov" else {}
    reset, _ = fit(spec(kind, group="k", session="s", session_gap="reset", **keep), df)
    out, bank = fit(spec(kind, group="k", session="s", group_close="session"), df)
    assert_frame_equal(out, reset, check_exact=True)
    rows = closed(bank)
    print(f"{kind}: {rows.height} closed rows, sessions {rows['session'].to_list()}")
    assert rows["session"].unique(maintain_order=True).cast(pl.Int64).to_list() == [0, 1]
    by_key, key_bank = fit(spec(kind, group="s", group_close="monotone"), df)
    assert_frame_equal(by_key, out, check_exact=True)
    keyed = closed(key_bank)
    assert keyed["group"].to_list() == rows["session"].to_list()
    assert_frame_equal(
        keyed.drop("group", "session"), rows.drop("group", "session"), check_exact=True
    )


def test_audit_refuses_group_close_and_rcov_requires_it():
    with pytest.raises(ValueError, match="group_close"):
        spec("audit", group="g", group_close="monotone")
    with pytest.raises(ValueError, match="group_close"):
        po.spec.rcov("m", features=["x0", "x1"], group="g")


# ---------------------------------------------------------------------------
# group


@pytest.mark.parametrize("kind", OTHERS)
def test_a_group_is_a_separate_bank(kind):
    """On a clock, each key's rows equal a bank fed that key's rows alone,
    and so does each key's table where the kind is read from one."""
    df = frame(kind)
    key = "g" if kind == "rcov" else "gg"  # rcov's groups close: a growing key
    s = clocked(kind, group=key)
    out, bank = fit(s, df)
    for k in df[key].unique(maintain_order=True):
        mine = df[key] == k
        solo, solo_bank = fit(s, df.filter(mine))
        assert_frame_equal(out.filter(mine), solo, check_exact=True)
        if kind in ("marginal", "audit"):
            read = bank.marginal if kind == "marginal" else bank.audit
            solo_read = solo_bank.marginal if kind == "marginal" else solo_bank.audit
            assert_frame_equal(
                read("m").filter(pl.col("group") == str(k)), solo_read("m"), check_exact=True
            )
    # Evidence: the keys' streams are not one stream (rcov's one stream is
    # one group, as it needs a group).
    whole, _ = fit(clocked(kind, group="k") if kind == "rcov" else clocked(kind), df)
    moved = differs(whole, out)
    print(f"{kind}: ungrouped, {len(moved)} fields differ: {moved}")
    assert moved


# ---------------------------------------------------------------------------
# Keeping the sweep honest


def test_every_kind_is_swept_or_waits():
    assert set(KINDS) | set(AFTER_TASK_277) == set(MINIMAL)
    assert set(OTHER_KINDS) <= set(KINDS)


@pytest.mark.parametrize("kind", AFTER_TASK_277)
def test_the_waiting_leg_is_ready(kind):
    """The ``huber`` leg builds every case and runs, so adding it is a move
    into ``KINDS``, not a build."""
    for kw in (
        dict(restart_after_step_back=0.0),
        dict(session="s", session_gap=3.0),
        dict(session="s", session_gap="reset"),
        dict(group="g", group_close="monotone"),
        dict(group="k", session="s", group_close="session"),
    ):
        df = stepped(kind) if "restart_after_step_back" in kw else frame(kind)
        out, _ = fit(clocked(kind, **kw), df)
        assert out.height == N
