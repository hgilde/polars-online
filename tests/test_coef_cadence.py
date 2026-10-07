"""``coef_every`` counts the clock, as ``solve_every`` does (docs/PLAN.md task
178): a ``coef`` row once the clock has moved ``coef_every`` since the
group's last one, or after ``max_rows_between_coefs`` accepted rows,
whichever comes first; ``0`` every row; and with neither, each group's last
row in each chunk, the default.

Each test reads the rows that carry ``coef`` from the output and holds them to
the rule written out over the stream's own clock, in whole numbers so that
the oracle's arithmetic is exact:

- the clock is the one the models are stepped on, measured from the group's
  first row: each step capped at ``gap_cap``, the steps of a run of skipped
  rows folded into the next accepted row's and the total capped again;
  without a clock column it is the row's number, the first row being 1;
- an accepted row -- one whose features and weight are usable, rows of
  weight zero and rows with a null target included -- is a ``coef`` row when
  the clock is ``coef_every`` past the last ``coef`` row's (the group's
  first row's before the first), or when it is the ``max_rows_between_coefs``-th
  accepted row since; a reset of the clock starts the count over;
- a ``coef`` row shows the model's coefficients, so it is null where the
  model has none yet: the expected rows are the rule's rows on which the
  same spec at ``coef_every = 0`` writes one, and the values are that run's.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po

#: Fitting a ridge from a handful of rows says so; beside the point here.
pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")

T0 = datetime(2024, 1, 2, 9, 30)
#: Seconds: the cap on a step, and the clock cadence most tests use.
GAP_CAP, EVERY = 600, 900


def irregular(n: int = 400, seed: int = 1) -> np.ndarray:
    """Steps of 1 to 120 seconds, the first row's 0."""
    steps = np.random.default_rng(seed).integers(1, 121, n)
    steps[0] = 0
    return steps


def frame(steps: np.ndarray, seed: int = 0, skipped: bool = True) -> pl.DataFrame:
    """Rows at those steps on a ``Datetime`` clock and on a number clock in
    seconds, in whole seconds so the clock's sums are exact; two features, a
    numeric, a binary and a label target, and weights. One row in eleven has a
    null feature, which skips it (but not the first, so every model has a
    first row); one in seven a weight of zero and one in thirteen a null
    target, which are accepted and counted."""
    rng = np.random.default_rng(seed)
    n = len(steps)
    seconds = np.cumsum(steps)
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 0.5 + 2.0 * x1 - x2 + 0.1 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.random(n) < 1 / 7] = 0.0
    df = pl.DataFrame(
        {
            "ts": [T0 + timedelta(seconds=int(s)) for s in seconds],
            "t": seconds.astype(float),
            "x1": x1,
            "x2": x2,
            "y": y,
            "b": (y > 0.5).astype(float),
            "c": np.where(y > 0.5, "a", "b"),
            "w": w,
        }
    )
    i = pl.int_range(pl.len())
    if skipped:
        df = df.with_columns(x1=pl.when(i % 11 == 5).then(None).otherwise(pl.col("x1")))
    return df.with_columns(
        y=pl.when(i % 13 == 6).then(None).otherwise(pl.col("y")),
        b=pl.when(i % 13 == 6).then(None).otherwise(pl.col("b")),
        c=pl.when(i % 13 == 6).then(None).otherwise(pl.col("c")),
    )


#: Every kind with a ``coef``, by builder, with what it needs beyond the
#: stream parameters: ``h`` is a half-life in the clock's own form.
KINDS = [
    "ewridge",
    "lasso",
    "rls",
    "kalman",
    "huber",
    "quantile",
    "ftrl",
    "sgd",
    "pa",
    "holt",
    "kmeans",
    "micro",
    "ew_class",
    "hmm",
    "deco",
]


def kind(name: str, h: float | str, **common) -> dict:
    two = ["x1", "x2"]
    own: dict[str, object] = {
        "ewridge": dict(targets=["y"], features=two),
        "lasso": dict(targets=["y"], features=two, lasso_path=[0.1, 0.0]),
        "rls": dict(targets=["y"], features=two),
        "kalman": dict(targets=["y"], features=two, coef_half_life=h),
        "huber": dict(targets=["y"], features=two),
        "quantile": dict(targets=["y"], features=two, quantile=0.5),
        "ftrl": dict(targets=["b"], features=two),
        "sgd": dict(targets=["y"], features=two),
        "pa": dict(targets=["y"], features=two),
        "holt": dict(targets=["y"]),
        "kmeans": dict(features=two, k=2, warm_rows=50),
        "micro": dict(features=two, eps=0.3),
        "ew_class": dict(label="c", classes=["a", "b"], features=two, precision_prior=1.0),
        "hmm": dict(features=two, k=2, precision_prior=0.1),
        "deco": dict(features=two),
    }[name]
    return getattr(po.spec, name)("m", half_life=h, weight="w", **own, **common)


def temporal(name: str = "ewridge", **kw) -> dict:
    return kind(name, "30m", clock="ts", gap_cap="10m", **kw)


def run(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([spec]).fit_predict(df).unnest("m")


def observed(out: pl.DataFrame) -> list[int]:
    """The rows that carry ``coef``."""
    return out.with_row_index("i").filter(pl.col("coef").is_not_null())["i"].to_list()


def accepted_rows(df: pl.DataFrame, spec: dict) -> list[bool]:
    """Whether the stream accepts each row: its features and weight usable."""
    cols = [*spec["features"], *([spec["weight"]] if spec.get("weight") else [])]
    if not cols:
        return [True] * df.height
    return df.select(pl.all_horizontal(pl.col(cols).is_not_null()))[:, 0].to_list()


def decayed_clock(
    seconds: list[int] | None,
    accepted: list[bool],
    gap_cap: int | None,
    sessions: list[str] | None = None,
    session_gap: int = 0,
) -> list[int]:
    """The clock ``coef_every`` measures at each row, from the group's first
    row: each step capped at ``gap_cap``, at a change of ``sessions`` the
    step ``session_gap`` capped the same way, and a run of skipped rows'
    steps folded into the next accepted row's and the total capped again.
    Without a clock column (``seconds`` None) it is the row's number, the
    first row being 1. A skipped row's entry is the clock as it stood;
    nothing reads it."""
    if seconds is None:
        return list(range(1, len(accepted) + 1))
    cap = float("inf") if gap_cap is None else gap_cap
    out, clock, held, prev = [], 0, 0, None
    for i, (s, ok) in enumerate(zip(seconds, accepted, strict=True)):
        if prev is None:
            step = 0
        elif sessions is not None and sessions[i] != sessions[i - 1]:
            step = min(session_gap, cap)
        else:
            step = min(s - prev, cap)
        prev = s
        if ok:
            clock += min(held + step, cap)
            held = 0
        else:
            held += step
        out.append(clock)
    return out


def rule(
    clock: list[int],
    accepted: list[bool],
    *,
    every: float | None,
    cap: int | None,
    resets: frozenset[int] = frozenset(),
) -> list[int]:
    """The rows the rule writes ``coef`` on. A reset at row ``r`` starts the
    clock and the count over there, the clock reading 0 at ``r``."""
    rows, last, count = [], 0, 0
    for i, ok in enumerate(accepted):
        if i in resets:
            last, count = clock[i], 0
        if not ok:
            continue
        count += 1
        if (every is not None and clock[i] - last >= every) or (cap is not None and count >= cap):
            rows.append(i)
            last, count = clock[i], 0
    return rows


def seconds_of(df: pl.DataFrame) -> list[int]:
    return [int(v) for v in df["t"]]


def where_fitted(spec: dict, df: pl.DataFrame, rows: list[int]) -> list[int]:
    """The rows of ``rows`` on which the model has a fit to show: those the
    same spec at ``coef_every = 0`` writes ``coef`` on."""
    every_row = run({**spec, "coef_every": 0, "max_rows_between_coefs": None}, df)
    fitted = set(observed(every_row))
    return [r for r in rows if r in fitted]


def held_to_the_rule(spec: dict, df: pl.DataFrame, want: list[int]) -> list[int]:
    """Assert the rows ``spec`` writes ``coef`` on are ``want`` where the model
    has a fit -- a row the same spec at ``coef_every = 0`` writes one on --
    with that run's values; return them."""
    out = run(spec, df)
    every_row = run({**spec, "coef_every": 0, "max_rows_between_coefs": None}, df)
    fitted = set(observed(every_row))
    got = observed(out)
    assert got == [r for r in want if r in fitted], (got[:12], want[:12])
    for r in got:
        assert out["coef"][r].to_list() == every_row["coef"][r].to_list(), r
    return got


class TestTheRule:
    @pytest.mark.parametrize("name", KINDS)
    def test_a_coef_row_follows_the_clock(self, name):
        df = frame(irregular())
        spec = temporal(name, coef_every="15m")
        acc = accepted_rows(df, spec)
        want = rule(decayed_clock(seconds_of(df), acc, GAP_CAP), acc, every=EVERY, cap=None)
        got = held_to_the_rule(spec, df, want)
        assert len(got) >= 10, got
        # The clock, not the rows: the gaps between coef rows in rows vary.
        assert len(set(np.diff(want))) > 3, want

    def test_a_capped_gap_counts_as_its_cap(self):
        """Rows a minute apart write ``coef`` every fifteen of them; a two-hour
        gap just after one counts as the ten minutes ``gap_cap`` allows, so the
        next comes five rows later, not at the gap."""
        steps = np.full(120, 60)
        steps[0] = 0
        df0 = frame(steps, skipped=False)
        before = observed(run(temporal(coef_every="15m"), df0))
        assert np.all(np.diff(before) == 15), before
        gap = before[1] + 1
        steps[gap] = 7200
        df = frame(steps, skipped=False)
        spec = temporal(coef_every="15m")
        acc = accepted_rows(df, spec)
        want = rule(decayed_clock(seconds_of(df), acc, GAP_CAP), acc, every=EVERY, cap=None)
        got = held_to_the_rule(spec, df, want)
        assert gap not in got and gap + 5 in got, (gap, got)
        uncapped = rule(decayed_clock(seconds_of(df), acc, None), acc, every=EVERY, cap=None)
        assert gap in uncapped, "the case needs the cap to matter"

    @pytest.mark.parametrize("name", KINDS)
    def test_a_row_cap_writes_whatever_the_clock(self, name):
        """``max_rows_between_coefs`` counts accepted rows as ``coef_every``
        counted them before task 178: rows of weight zero and rows with a null
        target included, a skipped row not."""
        df = frame(irregular())
        spec = temporal(name, max_rows_between_coefs=25)
        acc = accepted_rows(df, spec)
        want = rule(decayed_clock(seconds_of(df), acc, GAP_CAP), acc, every=None, cap=25)
        held_to_the_rule(spec, df, want)
        # The 25th, 50th, ... accepted row, as the old row count had it.
        nth = [i for i, k in enumerate(np.cumsum(acc)) if acc[i] and k % 25 == 0]
        assert want == nth

    @pytest.mark.parametrize("name", KINDS)
    def test_whichever_comes_first(self, name):
        df = frame(irregular())
        # Fifteen rows average about fifteen minutes here, so each binds
        # first somewhere.
        spec = temporal(name, coef_every="15m", max_rows_between_coefs=15)
        acc = accepted_rows(df, spec)
        clock = decayed_clock(seconds_of(df), acc, GAP_CAP)
        want = rule(clock, acc, every=EVERY, cap=15)
        held_to_the_rule(spec, df, want)
        by_clock = rule(clock, acc, every=EVERY, cap=None)
        by_rows = rule(clock, acc, every=None, cap=15)
        got = where_fitted(spec, df, want)
        assert got != where_fitted(spec, df, by_clock), "the clock must bind somewhere"
        assert got != where_fitted(spec, df, by_rows), "the rows must bind somewhere"

    @pytest.mark.parametrize("name", KINDS)
    def test_zero_writes_every_accepted_row(self, name):
        """``0`` meant the default before task 178; it is every row now, as in
        every clock cadence, and a skipped row still writes none."""
        df = frame(irregular())
        spec = temporal(name, coef_every=0)
        out = run(spec, df)
        acc = accepted_rows(df, spec)
        got = observed(out)
        assert len(got) > df.height // 2, got
        assert got == [i for i in range(got[0], df.height) if acc[i]]

    def test_unset_writes_each_group_s_last_accepted_row_in_each_chunk(self):
        df = frame(irregular()).with_columns(
            g=pl.when(pl.int_range(pl.len()) % 3 == 0).then(pl.lit("a")).otherwise(pl.lit("b"))
        )
        spec = temporal(group="g")
        parts = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_rows=50))
        got = observed(parts.unnest("m"))
        fitted = set(observed(run({**spec, "coef_every": 0}, df)))
        acc = accepted_rows(df, spec)
        want = []
        for start in range(0, df.height, 50):
            for g in ("a", "b"):
                rows = [i for i in range(start, min(start + 50, df.height)) if df["g"][i] == g]
                # The group's last accepted row in the chunk, which the last
                # row was alone, and none when it was skipped (review round 4,
                # PB2).
                kept = [i for i in rows if acc[i]]
                if kept:
                    want.append(kept[-1])
        assert got == [r for r in sorted(want) if r in fitted]
        assert len(got) >= 12, got

    def test_without_a_clock_column_it_counts_rows(self):
        """The clock of a spec without one is the row's number, the first row
        being 1, so a number ``coef_every`` writes every that many rows, a
        skipped row counted as the clock counts it; with no row skipped that
        is the row count ``coef_every`` kept before."""
        df = frame(irregular())
        spec = kind("ewridge", 30.0, coef_every=7)
        acc = accepted_rows(df, spec)
        want = rule(decayed_clock(None, acc, None), acc, every=7, cap=None)
        held_to_the_rule(spec, df, want)
        clean = frame(irregular(), skipped=False)
        got = held_to_the_rule(spec, clean, list(range(6, clean.height, 7)))
        assert got[:3] == [6, 13, 20]

    def test_a_number_clock_counts_its_own_units(self):
        df = frame(irregular())
        spec = kind("ewridge", 1800.0, clock="t", gap_cap=600.0, coef_every=900.0)
        acc = accepted_rows(df, spec)
        want = rule(decayed_clock(seconds_of(df), acc, GAP_CAP), acc, every=EVERY, cap=None)
        assert len(held_to_the_rule(spec, df, want)) >= 10

    @pytest.mark.parametrize("clock", ["ts", "t", None])
    def test_the_clock_runs_from_the_group_s_first_row_skipped_or_not(self, clock):
        """The group's first three rows skipped: the clock still starts at
        its first row, the skipped rows' steps folded into the first accepted
        row's, and without a clock column a skipped row has its number too."""
        df = frame(irregular()).with_columns(
            x1=pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("x1"))
        )
        if clock == "ts":
            spec, every, secs = temporal(coef_every="15m"), EVERY, seconds_of(df)
        elif clock == "t":
            spec = kind("ewridge", 1800.0, clock="t", gap_cap=600.0, coef_every=900.0)
            every, secs = EVERY, seconds_of(df)
        else:
            spec, every, secs = kind("ewridge", 30.0, coef_every=7), 7, None
        acc = accepted_rows(df, spec)
        assert acc[:4] == [False, False, False, True]
        clk = decayed_clock(secs, acc, GAP_CAP)
        want = rule(clk, acc, every=every, cap=None)
        held_to_the_rule(spec, df, want)
        # Measured from the first accepted row instead, the rows would move.
        late = rule([c - clk[3] for c in clk], acc, every=every, cap=None)
        assert where_fitted(spec, df, want) != where_fitted(spec, df, late), (
            "the case needs the skipped rows to count"
        )

    def test_a_session_change_counts_its_session_gap(self):
        """The clock is the one the models are stepped on: at a session
        change ``session_gap``, capped at ``gap_cap``, stands for the
        column's step."""
        df = frame(irregular()).with_columns(s=(pl.int_range(pl.len()) // 50).cast(pl.String))
        spec = temporal(coef_every="15m", session="s", session_gap="8m")
        acc = accepted_rows(df, spec)
        sessions = df["s"].to_list()
        clk = decayed_clock(seconds_of(df), acc, GAP_CAP, sessions=sessions, session_gap=480)
        want = rule(clk, acc, every=EVERY, cap=None)
        held_to_the_rule(spec, df, want)
        plain = rule(decayed_clock(seconds_of(df), acc, GAP_CAP), acc, every=EVERY, cap=None)
        assert where_fitted(spec, df, want) != where_fitted(spec, df, plain), (
            "the case needs the session gap to matter"
        )

    def test_each_group_counts_its_own_clock(self):
        df = frame(irregular()).with_columns(
            g=pl.when(pl.int_range(pl.len()) % 3 == 0).then(pl.lit("a")).otherwise(pl.lit("b"))
        )
        spec = temporal(group="g", coef_every="15m", max_rows_between_coefs=10)
        want = []
        for g in ("a", "b"):
            idx = [i for i in range(df.height) if df["g"][i] == g]
            sub = df[idx]
            acc = accepted_rows(sub, spec)
            mine = rule(decayed_clock(seconds_of(sub), acc, GAP_CAP), acc, every=EVERY, cap=10)
            want += [idx[r] for r in mine]
        held_to_the_rule(spec, df, sorted(want))

    def test_a_reset_starts_the_count_over(self):
        """A step back larger than ``restart_after_step_back`` starts the
        models over, and the cadence with them: the clock is measured from the
        reset row, and the rows counted from it."""
        steps = irregular(300)
        # The clock steps back at row 150, by 4,000 seconds: past the 3,600
        # that restarts.
        steps[150] = -4000
        df = frame(steps)
        spec = temporal(coef_every="15m", max_rows_between_coefs=40, restart_after_step_back="1h")
        acc = accepted_rows(df, spec)
        secs = seconds_of(df)
        head = decayed_clock(secs[:150], acc[:150], GAP_CAP)
        tail = decayed_clock(secs[150:], acc[150:], GAP_CAP)
        clock = head + tail
        want = rule(clock, acc, every=EVERY, cap=40, resets=frozenset({150}))
        out = held_to_the_rule(spec, df, want)
        assert any(r >= 150 for r in out) and any(r < 150 for r in out)
        unbroken = where_fitted(spec, df, rule(clock, acc, every=EVERY, cap=40))
        assert out != unbroken, "the case needs the reset to matter"

    def test_a_millisecond_clock_is_counted_exactly(self):
        """The clock is held exactly, as a window's edge is (task 175): rows
        a millisecond apart under ``"2s"`` write every 2,000th row. Summed in
        doubles, 2,000 steps of 0.001 s come to less than 2 s, a row late."""
        n = 8001
        df = pl.DataFrame(
            {
                "ts": pl.datetime_range(
                    T0, T0 + timedelta(milliseconds=n - 1), "1ms", eager=True
                ).dt.cast_time_unit("ms"),
                "x1": np.random.default_rng(5).standard_normal(n),
                "x2": np.random.default_rng(6).standard_normal(n),
                "y": np.random.default_rng(7).standard_normal(n),
                "w": np.ones(n),
            }
        )
        spec = kind("ewridge", "1s", clock="ts", gap_cap="1s", coef_every="2s")
        got = observed(run(spec, df))
        assert got == [2000, 4000, 6000, 8000], got
        total = 0.0
        for _ in range(2000):
            total += 0.001
        assert total < 2.0, "a double sum would have been a row late"


class TestChunkingAndResume:
    @pytest.mark.parametrize("name", KINDS)
    def test_an_explicit_cadence_does_not_depend_on_the_chunking(self, name):
        """With a cadence given, ``coef`` is written where the rule says and
        nowhere else, so the field is the same however the stream is chunked;
        unset, it follows the chunks by design (PLAN §3)."""
        df = frame(irregular())
        spec = temporal(name, coef_every="15m", max_rows_between_coefs=12)
        whole = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("coef")
        assert whole.null_count() < df.height
        for rows in (1, 7, 37):
            parts = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_rows=rows))
            assert parts["m"].struct.field("coef").to_list() == whole.to_list(), rows

    @pytest.mark.parametrize("name", KINDS)
    def test_a_save_between_coef_rows_resumes_where_the_unbroken_run_does(self, name):
        df = frame(irregular())
        spec = temporal(name, coef_every="15m", max_rows_between_coefs=40)
        whole = run(spec, df)
        rows = observed(whole)
        assert len(rows) > 8, rows
        for cut in (rows[3] + 2, rows[7] + 5):
            assert cut not in rows
            first = po.ModelBank([spec])
            first.fit_predict(df.head(cut))
            resumed = po.ModelBank.load_bytes(first.save_bytes(), specs=[spec])
            tail = resumed.fit_predict(df.slice(cut)).unnest("m")
            assert tail["coef"].to_list() == whole["coef"][cut:].to_list(), cut

    def test_the_state_exports_to_json(self):
        """The cadence's place is state: a number clock's, and a temporal
        clock's, which is held in integer nanoseconds."""
        df = frame(irregular())
        for spec in (
            temporal(coef_every="15m"),
            kind("ewridge", 1800.0, clock="t", gap_cap=600.0, coef_every=900.0),
            kind("ewridge", 30.0, max_rows_between_coefs=9),
        ):
            bank = po.ModelBank([spec])
            bank.fit_predict(df.head(77))
            doc = json.loads(bank.to_json())
            assert "coef_cadence" in json.dumps(doc), spec
        unset = po.ModelBank([temporal()])
        unset.fit_predict(df.head(77))
        assert "coef_cadence" not in unset.to_json(), "a default spec writes what it did"


class TestRefusals:
    def test_a_temporal_spec_refuses_a_plain_number(self):
        with pytest.raises(ValueError, match="coef_every is a plain number"):
            temporal(coef_every=900.0)

    def test_a_number_spec_refuses_a_duration(self):
        with pytest.raises(ValueError, match="coef_every is a duration"):
            kind("ewridge", 1800.0, clock="t", gap_cap=600.0, coef_every="15m")

    def test_without_a_clock_a_duration_is_refused(self):
        with pytest.raises(ValueError, match="coef_every is a duration"):
            kind("ewridge", 30.0, coef_every="15m")

    @pytest.mark.parametrize("bad", [-5.0, -1])
    def test_a_negative_cadence_is_refused(self, bad):
        with pytest.raises(ValueError, match="coef_every must be"):
            kind("ewridge", 30.0, coef_every=bad)

    def test_a_negative_cadence_in_a_dict_is_refused_by_name(self):
        spec = {**kind("ewridge", 30.0), "coef_every": -5.0}
        with pytest.raises(ValueError, match=r"coef_every must be finite and >= 0 clock units"):
            po.ModelBank([spec])

    def test_an_infinite_cadence_is_refused(self):
        spec = {**kind("ewridge", 30.0), "coef_every": float("inf")}
        with pytest.raises(ValueError, match=r"coef_every must be finite and >= 0 clock units"):
            po.ModelBank([spec])

    def test_a_row_cap_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="max_rows_between_coefs must be >= 1"):
            kind("ewridge", 30.0, max_rows_between_coefs=0)
        spec = {**kind("ewridge", 30.0), "max_rows_between_coefs": 0}
        with pytest.raises(ValueError, match="max_rows_between_coefs must be >= 1"):
            po.ModelBank([spec])

    def test_a_model_without_coefficients_refuses_both(self):
        for kw in ({"coef_every": 0}, {"max_rows_between_coefs": 5}):
            with pytest.raises(ValueError, match=next(iter(kw))):
                po.spec.ew_cov("c", features=["x1", "x2"], half_life=30.0, **kw)
