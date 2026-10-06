"""``ew_cov``'s principal components refresh on the clock (docs/PLAN.md
task 161): every ``pca_every`` clock units or every ``max_rows_between_pca``
rows, whichever comes first, as the regressions' ``solve_every`` and
``max_rows_between_solves`` schedule a solve; every row with neither.

Each test reads the rows a refresh happened at from the output, where a row's
loadings are the ones in force before it, so a refresh at row ``t`` shows as
a change between rows ``t`` and ``t + 1``. It holds them to the rule written
out over the stream's own clock: the first refresh at the first row whose
weight reaches ``min_weight``, then each time the clock or the rows since the
last reach theirs, a gap counting at most ``gap_cap``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po

FEATURES = ["x0", "x1", "x2"]
LOADINGS = [f"pc0_loading_{f}" for f in FEATURES]
MIN_WEIGHT = 5.0


def frame(seconds: np.ndarray, seed: int = 0) -> pl.DataFrame:
    """Rows at the given steps, in whole seconds so the clock's sums are
    exact, with three correlated columns."""
    rng = np.random.default_rng(seed)
    n = len(seconds)
    t0 = datetime(2024, 1, 2, 9, 30)
    ts = [t0 + timedelta(seconds=int(s)) for s in np.cumsum(seconds)]
    f = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "ts": ts,
            "t": np.cumsum(seconds).astype(float),
            "x0": f + 0.3 * rng.standard_normal(n),
            "x1": -f + 0.3 * rng.standard_normal(n),
            "x2": 0.5 * rng.standard_normal(n),
        }
    )


def irregular(n: int = 400, seed: int = 1) -> np.ndarray:
    """Steps of 1 to 120 seconds, the first row's 0."""
    steps = np.random.default_rng(seed).integers(1, 121, n)
    steps[0] = 0
    return steps


def spec(**kw):
    d = dict(
        features=FEATURES,
        clock="ts",
        half_life="30m",
        gap_cap="10m",
        stats=["mean"],
        pca=1,
        min_weight=MIN_WEIGHT,
    )
    d.update(kw)
    return po.spec.ew_cov("c", **d)


def run(s, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([s]).fit_predict(df).unnest("c")


def observed(out: pl.DataFrame) -> list[int]:
    """The rows a refresh happened at: where the loadings in force change
    between a row and the next, a first loading included."""
    loadings = out.select(LOADINGS).to_numpy()
    rows = []
    for t in range(len(loadings) - 1):
        a, b = loadings[t], loadings[t + 1]
        if np.isnan(a).all() and np.isnan(b).all():
            continue
        if np.isnan(a).all() != np.isnan(b).all() or not np.array_equal(a, b):
            rows.append(t)
    return rows


def expected(
    steps: np.ndarray,
    out: pl.DataFrame,
    *,
    every: float | None,
    cap: int | None,
    gap_cap: float,
) -> list[int]:
    """The rule, written out. A row's ``weight_sum`` is the weight before it
    (hard rule 8), so the weight after row ``t`` is row ``t + 1``'s; the last
    row's is not read, and nor is a refresh there."""
    after = out["weight_sum"].to_numpy()[1:]
    rows, clock, count = [], 0.0, 0
    for t in range(len(steps) - 1):
        clock += min(float(steps[t]), gap_cap)
        count += 1
        if not rows:
            due = after[t] >= MIN_WEIGHT
        elif every is None and cap is None:
            due = True
        else:
            due = (every is not None and (every <= 0 or clock >= every)) or (
                cap is not None and count >= cap
            )
        if due:
            rows.append(t)
            clock, count = 0.0, 0
    return rows


class TestTheSchedule:
    def test_a_refresh_follows_the_clock(self):
        steps = irregular()
        df = frame(steps)
        out = run(spec(pca_every="15m"), df)
        want = expected(steps, out, every=900.0, cap=None, gap_cap=600.0)
        assert observed(out) == want
        # The clock, not the rows: the gaps between refreshes in rows vary.
        assert len(set(np.diff(want))) > 3, want

    def test_a_capped_gap_counts_as_its_cap(self):
        """Rows a minute apart refresh every fifteen of them; a two-hour gap
        just after a refresh counts as the ten minutes `gap_cap` allows, so
        the refresh comes five rows later, not at the gap."""
        steps = np.full(80, 60)
        steps[0] = 0
        out0 = run(spec(pca_every="15m"), frame(steps))
        before = observed(out0)
        assert np.all(np.diff(before) == 15), before
        gap = before[1] + 1
        steps[gap] = 7200
        out = run(spec(pca_every="15m"), frame(steps))
        got = observed(out)
        assert got == expected(steps, out, every=900.0, cap=None, gap_cap=600.0)
        assert gap not in got and gap + 5 in got, (gap, got)
        uncapped = expected(steps, out, every=900.0, cap=None, gap_cap=float("inf"))
        assert gap in uncapped, "the case needs the cap to matter"

    def test_a_row_cap_refreshes_whatever_the_clock(self):
        steps = irregular()
        out = run(spec(max_rows_between_pca=25), frame(steps))
        want = expected(steps, out, every=None, cap=25, gap_cap=600.0)
        assert observed(out) == want
        assert set(np.diff(want)) == {25}, want

    def test_whichever_comes_first(self):
        steps = irregular()
        # Fifteen rows average about fifteen minutes here, so each binds
        # first somewhere.
        out = run(spec(pca_every="15m", max_rows_between_pca=15), frame(steps))
        got = observed(out)
        assert got == expected(steps, out, every=900.0, cap=15, gap_cap=600.0)
        by_clock = expected(steps, out, every=900.0, cap=None, gap_cap=600.0)
        by_rows = expected(steps, out, every=None, cap=15, gap_cap=600.0)
        assert got != by_clock and got != by_rows, "both must bind somewhere"

    @pytest.mark.parametrize("kw", [{}, {"pca_every": 0}], ids=["neither", "pca_every=0"])
    def test_every_row_with_neither_or_at_zero(self, kw):
        steps = irregular()
        out = run(spec(**kw), frame(steps))
        got = observed(out)
        assert got == list(range(got[0], len(steps) - 1)), got

    def test_without_a_clock_column_it_counts_rows(self):
        """The clock of a spec without one is the row's number, so a number
        `pca_every` refreshes every that many rows, as it did before."""
        steps = irregular()
        s = po.spec.ew_cov(
            "c",
            features=FEATURES,
            half_life=100.0,
            stats=["mean"],
            pca=1,
            pca_every=7,
            min_weight=MIN_WEIGHT,
        )
        out = run(s, frame(steps))
        assert set(np.diff(observed(out))) == {7}

    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    def test_millisecond_rows_refresh_on_the_exact_clock(self, unit):
        """Task 180: two thousand steps of 1 ms summed in doubles are
        1.9999999999998905 s, so ``pca_every="2s"`` refreshed a row late,
        and every refresh after it a row later again. The cadence is decided
        on the decayed clock held exactly: after the first refresh, one at
        the first row whose instant is two seconds past the last's, from the
        raw nanoseconds -- every 2,000th row, in every unit."""
        n = 6_100
        ns = 1_704_067_200_000_000_000 + np.arange(n, dtype=np.int64) * 1_000_000
        df = frame(np.zeros(n)).with_columns(
            pl.Series("ts", ns).cast(pl.Datetime("ns")).cast(pl.Datetime(unit))
        )
        got = observed(run(spec(pca_every="2s"), df))
        want, last = [got[0]], got[0]
        for t in range(last + 1, n - 1):
            if int(ns[t]) - int(ns[last]) >= 2_000_000_000:
                want.append(t)
                last = t
        assert got == want
        assert np.diff(got).tolist() == [2_000] * 3, got

    def test_a_number_clock_counts_its_own_units(self):
        steps = irregular()
        s = po.spec.ew_cov(
            "c",
            features=FEATURES,
            clock="t",
            gap_cap=600.0,
            half_life=1800.0,
            stats=["mean"],
            pca=1,
            pca_every=900.0,
            min_weight=MIN_WEIGHT,
        )
        out = run(s, frame(steps))
        assert observed(out) == expected(steps, out, every=900.0, cap=None, gap_cap=600.0)


class TestChunkingAndResume:
    def test_the_refreshes_do_not_depend_on_the_chunking(self):
        df = frame(irregular())
        s = spec(pca_every="15m", max_rows_between_pca=12)
        whole = po.ModelBank([s]).fit_predict(df)
        for rows in (1, 7, 37):
            parts = pl.concat(po.ModelBank([s]).fit_predict_batches(df, chunk_rows=rows))
            for f in LOADINGS + ["pc0_score", "pc0_var"]:
                a = whole["c"].struct.field(f).to_numpy()
                b = parts["c"].struct.field(f).to_numpy()
                assert np.array_equal(a, b, equal_nan=True), (rows, f)

    def test_a_save_between_refreshes_resumes_where_the_unbroken_run_does(self):
        df = frame(irregular())
        s = spec(pca_every="15m", max_rows_between_pca=40)
        whole = run(s, df)
        refreshes = observed(whole)
        for cut in (refreshes[3] + 2, refreshes[8] + 5):
            first = po.ModelBank([s])
            first.fit_predict(df.head(cut))
            resumed = po.ModelBank.load_bytes(first.save_bytes(), specs=[s])
            tail = resumed.fit_predict(df.slice(cut)).unnest("c")
            for f in LOADINGS + ["pc0_score"]:
                a = whole[f].to_numpy()[cut:]
                b = tail[f].to_numpy()
                assert np.array_equal(a, b, equal_nan=True), (cut, f)


class TestRefusals:
    def test_a_temporal_spec_refuses_a_plain_number(self):
        with pytest.raises(ValueError, match="pca_every is a plain number"):
            spec(pca_every=900.0)

    def test_a_number_spec_refuses_a_duration(self):
        """The mirror: a spec in plain numbers, read on a number clock, with
        `pca_every` a duration, is one spec mixing the two."""
        with pytest.raises(ValueError, match="pca_every is a duration"):
            po.spec.ew_cov(
                "c",
                features=FEATURES,
                clock="t",
                half_life=1800.0,
                gap_cap=600.0,
                stats=["mean"],
                pca=1,
                pca_every="15m",
            )

    def test_a_negative_cadence_is_refused(self):
        with pytest.raises(ValueError, match="pca_every must be finite and >= 0 clock units"):
            po.spec.ew_cov(
                "c", features=FEATURES, half_life=100.0, stats=["mean"], pca=1, pca_every=-5.0
            )


def test_the_state_exports_to_json_under_a_row_cap_alone():
    """A row cap alone leaves the clock cadence infinite, which serde_json
    writes as null and cannot read back: `to_json` refused every such bank
    (task 161's miss, fixed in task 163)."""
    df = frame(irregular())
    for kw in ({"max_rows_between_pca": 12}, {"pca_every": "15m"}, {}):
        bank = po.ModelBank([spec(**kw)])
        bank.fit_predict(df.head(60))
        assert '"pca_every"' in bank.to_json(), kw
