"""`seqtest` -- a sequential test of a sign by betting (docs/ENHANCEMENTS.md
E42, docs/PLAN.md task 30): two e-processes per target, one for "positive"
and one for "negative", each a Kelly bettor with a Krichevsky-Trofimov stake
on the sign counts so far; and, given ``a`` and ``b``, the same test on
``|resid_b| - |resid_a|`` of two specs of the bank -- "is a closer than b".

Not a regression: nothing is predicted, and the outputs (``log_e_pos``,
``log_e_neg``, the counts) are the *evidence* as it stood before the row.
Four kinds of check:

- **The oracle.** ``reference.seqtest_ref`` replays the recursion in scalar
  numpy (nulls, zeros, resets, ``min_periods``) and is held to the bank to
  ``1e-12``; where the clip never binds the wealth has the closed form
  ``2^n B(n_pos + 1/2, n_neg + 1/2) / pi``, held through ``math.lgamma``;
  and ``po.eval.seqtest`` -- the same computation in polars expressions --
  is held to the bank bit for bit over a million rows, in both modes.
- **The guarantee.** Type I error over twenty thousand fair-coin streams,
  and twenty thousand *dependent* streams that are still null (the sign is
  never more likely positive than negative, given the past): the crossing
  rate of ``1/alpha`` is at most ``alpha``, at every row, as Ville's
  inequality says. Power at 60% and 55%; the size of the values is
  invisible; ties are not trials.
- **The comparison.** The two-phase bank: a comparison reads the same
  out-of-sample residuals the two structs report, a grid instance is picked
  by ``a_suffix``, a row either side sits out is no trial, columns come
  back in spec order whichever phase produced them, scoring reads the state
  before the chunk, a refused chunk updates neither phase.
- **Edge cases and plumbing.** Warmup, per-target ``min_periods``, null and
  zero and out-of-bound values, a session reset, groups and a null key,
  chunk invariance, save/load and pickle, the lazy path, the runner, the
  CLI, the expression path (column mode; a comparison is refused with the
  way to write it), ``output_index`` dtypes, no coefficients, and every
  refusal by name.
"""

from __future__ import annotations

import math
import pickle
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference

LN20 = math.log(20.0)  # level 0.05
LN100 = math.log(100.0)  # level 0.01


def signs(n, p=0.5, seed=0, scale=1.0):
    """`n` values, positive with probability `p`, of a size that does not
    matter to the test (``scale`` sets it)."""
    rng = np.random.default_rng(seed)
    s = np.where(rng.random(n) < p, 1.0, -1.0)
    return s * scale * (0.5 + rng.random(n))


def frame(n=600, m=1, seed=0, p=0.5, null_every=0, zero_every=0, groups=None):
    """Columns ``d0 .. d{m-1}`` to test the sign of, ``t`` a clock, and ``g``
    a group key cycling through ``groups`` when given."""
    cols = {}
    for j in range(m):
        d = signs(n, p=p, seed=seed * 100 + j)
        d = [
            None
            if null_every and i % null_every == 3
            else (0.0 if zero_every and i % zero_every == 5 else v)
            for i, v in enumerate(d)
        ]
        cols[f"d{j}"] = pl.Series(d, dtype=pl.Float64)
    cols["t"] = pl.Series(np.arange(n, dtype=float) * 2.0)
    if groups is not None:
        cols["g"] = pl.Series([groups[i % len(groups)] for i in range(n)])
    return pl.DataFrame(cols)


def spec(name="m", targets=("d0",), **kw):
    return po.spec.seqtest(name, targets=list(targets), **kw)


def unnested(out, name="m"):
    return out.select(name).unnest(name)


def as_ref(out, name="m", targets=("d0",)):
    """The bank's struct as `reference.seqtest_ref` lays its output out."""
    u = unnested(out, name)
    stack = lambda k: np.column_stack(  # noqa: E731
        [u[f"{k}_{t}"].cast(pl.Float64).fill_null(float("nan")).to_numpy() for t in targets]
    )
    return {
        "log_e_pos": stack("log_e_pos"),
        "log_e_neg": stack("log_e_neg"),
        "n_pos": stack("n_pos"),
        "n_neg": stack("n_neg"),
        "n_eff": u["n_eff"].fill_null(float("nan")).to_numpy(),
    }


def close(got, want, tol=1e-12):
    """Equal to `tol` (relative), NaN where NaN."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), "the NaN pattern differs"
    ok = ~np.isnan(got)
    err = np.abs(got[ok] - want[ok]) / np.maximum(1.0, np.abs(want[ok]))
    assert err.max(initial=0.0) <= tol, err.max()


def log_e_closed_form(n_pos, n_neg):
    """`ln(2^n B(n_pos + 1/2, n_neg + 1/2) / B(1/2, 1/2))`: the wealth of the
    unclipped KT bettor after `n_pos` positives and `n_neg` negatives."""
    n = n_pos + n_neg
    return (
        n * math.log(2.0)
        + math.lgamma(n_pos + 0.5)
        + math.lgamma(n_neg + 0.5)
        - math.lgamma(n + 1.0)
        - math.log(math.pi)
    )


# --------------------------------------------------------------- the oracle


class TestOracle:
    def test_the_replay_holds_with_nulls_zeros_and_three_targets(self):
        df = frame(n=5000, m=3, seed=1, p=0.6, null_every=7, zero_every=11)
        out = po.ModelBank([spec(targets=["d0", "d1", "d2"], min_periods=4.0)]).fit_predict(df)
        Y = df.select("d0", "d1", "d2").fill_null(float("nan")).to_numpy()
        want = reference.seqtest_ref(Y, min_periods=4.0)
        got = as_ref(out, targets=["d0", "d1", "d2"])
        for k in want:
            close(got[k], want[k])
        # The counts are exact integers and the outputs are the state before
        # the row: the last row's counts are one short of the column's.
        u = unnested(out)
        for t in ("d0", "d1", "d2"):
            col = df[t]
            npos, nneg = (col > 0).sum(), (col < 0).sum()
            last = df[t][-1]
            assert u[f"n_pos_{t}"][-1] == npos - (1 if last is not None and last > 0 else 0)
            assert u[f"n_neg_{t}"][-1] == nneg - (1 if last is not None and last < 0 else 0)

    def test_groups_are_separate_replays(self):
        df = frame(n=3000, seed=2, p=0.55, null_every=5, groups=["p", "q", "r"])
        out = po.ModelBank([spec(group="g")]).fit_predict(df)
        for g in ("p", "q", "r"):
            mask = (df["g"] == g).to_numpy()
            Y = df.filter(pl.col("g") == g).select("d0").fill_null(float("nan")).to_numpy()
            want = reference.seqtest_ref(Y)
            got = as_ref(out.filter(pl.Series(mask)))
            for k in want:
                close(got[k], want[k])

    def test_a_session_reset_restarts_the_replay(self):
        n = 2000
        df = frame(n=n, seed=3, p=0.7).with_columns(
            pl.Series("s", ["a"] * 700 + ["b"] * 500 + ["c"] * 800)
        )
        s = spec(clock="t", max_dclock=10.0, session="s", session_gap="reset")
        out = po.ModelBank([s]).fit_predict(df)
        reset = np.zeros(n, dtype=bool)
        reset[[700, 1200]] = True
        want = reference.seqtest_ref(df.select("d0").to_numpy(), reset=reset)
        got = as_ref(out)
        for k in want:
            close(got[k], want[k])
        u = unnested(out)
        assert u["n_eff"][700] == 0.0 and u["n_pos_d0"][700] == 0 and u["log_e_pos_d0"][700] == 0.0
        assert u["n_eff"][699] == 699.0

    @pytest.mark.parametrize("side", ["pos", "neg"])
    def test_the_closed_form_where_the_clip_never_binds(self, side):
        # Two of one sign, then one of the other, repeated: the leading side
        # is never behind, so its stake is the unclipped KT stake and its
        # wealth is the Beta(1/2, 1/2) mixture; the other side never leads,
        # never bets, and its log e-value is exactly zero.
        lead = 1.0 if side == "pos" else -1.0
        pattern = [lead, lead, -lead] * 400
        df = pl.DataFrame({"d0": pattern})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        n_lead, n_other = 0, 0
        for i, v in enumerate(pattern):
            want = log_e_closed_form(n_lead, n_other)
            got = u[f"log_e_{side}_d0"][i]
            assert abs(got - want) <= 1e-10 * max(1.0, abs(want)), (i, got, want)
            other = "neg" if side == "pos" else "pos"
            assert u[f"log_e_{other}_d0"][i] == 0.0
            if v == lead:
                n_lead += 1
            else:
                n_other += 1
        # And it is evidence: 800 of 1200 rows on one side is far past 1/0.01.
        assert u[f"log_e_{side}_d0"][-1] > LN100 + 50

    def test_the_two_sided_e_value_is_the_average(self):
        # (E_pos + E_neg) / 2 is itself an e-value; where one side never
        # bets its E is 1, so the average is (E + 1) / 2.
        df = pl.DataFrame({"d0": [1.0, 1.0, -1.0] * 100})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        e2 = (np.exp(u["log_e_pos_d0"].to_numpy()) + np.exp(u["log_e_neg_d0"].to_numpy())) / 2
        assert np.allclose(e2, (np.exp(u["log_e_pos_d0"].to_numpy()) + 1.0) / 2.0)

    def test_the_eval_twin_equals_the_bank_over_a_million_rows(self):
        n, k = 1_000_000, 200
        rng = np.random.default_rng(4)
        d0 = signs(n, p=0.52, seed=40)
        d1 = signs(n, p=0.48, seed=41, scale=1e6)
        d0[rng.random(n) < 0.05] = np.nan
        d1[rng.random(n) < 0.05] = 0.0
        df = pl.DataFrame(
            {
                "d0": pl.Series(d0).fill_nan(None),
                "d1": d1,
                "g": (np.arange(n) % k).astype(np.int64),
            }
        )
        s = spec(targets=["d0", "d1"], group="g", min_periods=3.0)
        bank = unnested(po.ModelBank([s]).fit_predict(df))
        twin = po.eval.seqtest(df, targets=["d0", "d1"], by=["g"], min_periods=3.0)
        assert bank.equals(twin["seqtest"].struct.unnest(), null_equal=True)
        assert bank.dtypes == [pl.Float64, pl.Float64, pl.Int64, pl.Int64] * 2 + [pl.Float64]
        # 52% positive (d0) and 48% (d1) over 5000 rows per group: read at
        # the last row, the right side has crossed 1/0.05 in a good share of
        # the groups (0.24 and 0.26 measured) and the wrong side in almost
        # none -- a fixed-time read is dominated by the sup Ville bounds.
        last = bank.filter(pl.col("n_eff") == n / k - 1)
        assert (last["log_e_pos_d0"] > LN20).mean() > 0.15
        assert (last["log_e_neg_d1"] > LN20).mean() > 0.15
        assert (last["log_e_neg_d0"] > LN20).mean() <= 0.05
        assert (last["log_e_pos_d1"] > LN20).mean() <= 0.05

    def test_the_eval_twin_equals_the_banks_comparison(self):
        df = regression(300_000, seed=5, groups=["p", "q", "r", "s"])
        common = dict(targets=["y"], features=["x0", "x1"], group="g", min_periods=5.0)
        specs = [
            po.spec.ewridge("ridge", halflife=[50.0, 500.0], **common),
            po.spec.kalman("kalman", halflife=100.0, coef_halflife=50.0, **common),
            po.spec.seqtest("c", targets=["y"], a="ridge", a_suffix="@h50", b="kalman", group="g"),
        ]
        out = po.ModelBank(specs).fit_predict(df)
        twin = po.eval.seqtest(out, a="ridge", b="kalman", a_suffix="@h50", by=["g"])
        assert unnested(out, "c").equals(twin["seqtest"].struct.unnest(), null_equal=True)
        # `targets=None` finds the residual the two sides share.
        again = po.eval.seqtest(
            out, targets=["y"], a="ridge", b="kalman", a_suffix="@h50", by=["g"]
        )
        assert twin.equals(again, null_equal=True)

    def test_the_eval_twin_reads_what_the_bank_does_not_learn_from_as_no_sign(self):
        # An infinity and a magnitude beyond the input bound are missing to
        # the bank (docs/PLAN.md section 3); polars would order them as
        # signs, so the twin takes the bank's view.
        vals = [1.0, float("inf"), -1.0, float("-inf"), 1e101, -1e101, float("nan"), None, 0.0, 2.0]
        df = pl.DataFrame({"d0": pl.Series(vals, dtype=pl.Float64)})
        bank = unnested(po.ModelBank([spec()]).fit_predict(df))
        twin = po.eval.seqtest(df, targets=["d0"])["seqtest"].struct.unnest()
        assert bank.equals(twin, null_equal=True)
        assert bank["n_pos_d0"][-1] == 1 and bank["n_neg_d0"][-1] == 1
        assert bank["n_eff"][-1] == 9.0  # every row but the last, whatever it held


# ------------------------------------------------------------- the guarantee


def crossing_rates(df, alpha_logs, by="g", field="log_e_pos_d0", **kw):
    """Per group, whether `field` ever reaches each threshold: the rate at
    which a test read at every row and stopped at the first crossing
    rejects."""
    u = po.ModelBank([spec(group=by, **kw)]).fit_predict(df)
    peak = (
        u.select(pl.col(by), pl.col("m").struct.field(field).alias("e"))
        .group_by(by)
        .agg(pl.col("e").max())["e"]
        .to_numpy()
    )
    return {a: float((peak >= a).mean()) for a in alpha_logs}


class TestTheGuarantee:
    STREAMS = 20_000
    ROWS = 300

    def fair(self, seed):
        n = self.STREAMS * self.ROWS
        return pl.DataFrame(
            {"d0": signs(n, p=0.5, seed=seed), "g": (np.arange(n) % self.STREAMS).astype(np.int64)}
        )

    def test_type_i_error_under_a_fair_coin(self):
        rates = crossing_rates(self.fair(10), [LN20, LN100])
        # Ville: P(sup E >= 1/alpha) <= alpha. Twenty thousand streams put
        # three binomial standard deviations at 0.0046 for alpha = 0.05;
        # the KT bettor sits well inside, and a bettor that never bet would
        # sit at zero, which the lower bounds catch.
        assert rates[LN20] <= 0.05, rates
        assert rates[LN100] <= 0.01, rates
        assert rates[LN20] >= 0.005 and rates[LN100] >= 0.0005, rates
        # And the same for the negative side and the two-sided average.
        neg = crossing_rates(self.fair(11), [LN20], field="log_e_neg_d0")
        assert 0.005 <= neg[LN20] <= 0.05, neg

    def test_type_i_error_under_a_dependent_null(self):
        # Not a coin: the next sign depends on the last one (P(+ | last +) =
        # 0.3, P(+ | last -) = 0.5), so the rows are neither independent nor
        # identically distributed -- but given the past a positive is never
        # the more likely sign, which is all the null asks, and the crossing
        # rate of "positive" is still at most alpha.
        rng = np.random.default_rng(12)
        n = self.STREAMS * self.ROWS
        u = rng.random(n).reshape(self.STREAMS, self.ROWS)
        s = np.empty_like(u)
        s[:, 0] = np.where(u[:, 0] < 0.5, 1.0, -1.0)
        for i in range(1, self.ROWS):
            p = np.where(s[:, i - 1] > 0, 0.3, 0.5)
            s[:, i] = np.where(u[:, i] < p, 1.0, -1.0)
        df = pl.DataFrame(
            {
                "d0": s.T.ravel(),  # row-major over streams: `g` cycles
                "g": (np.arange(n) % self.STREAMS).astype(np.int64),
            }
        )
        rates = crossing_rates(df, [LN20, LN100])
        assert rates[LN20] <= 0.05 and rates[LN100] <= 0.01, rates

    def test_the_two_sided_e_value_is_valid_too(self):
        out = po.ModelBank([spec(group="g")]).fit_predict(self.fair(13))
        e2 = (
            out.select(
                pl.col("g"),
                (
                    (
                        pl.col("m").struct.field("log_e_pos_d0").exp()
                        + pl.col("m").struct.field("log_e_neg_d0").exp()
                    )
                    / 2.0
                ).alias("e"),
            )
            .group_by("g")
            .agg(pl.col("e").max())["e"]
        )
        assert (e2 >= 20.0).mean() <= 0.05

    @pytest.mark.parametrize("p, rows, low, high", [(0.6, 1000, 0.99, 1.0), (0.55, 1000, 0.3, 0.9)])
    def test_power(self, p, rows, low, high):
        k = 2000
        n = k * rows
        df = pl.DataFrame({"d0": signs(n, p=p, seed=14), "g": (np.arange(n) % k).astype(np.int64)})
        rate = crossing_rates(df, [LN20])[LN20]
        assert low <= rate <= high, rate
        # Reading it at the end instead of at the peak is a weaker test.
        u = po.ModelBank([spec(group="g")]).fit_predict(df)
        last = u.filter(pl.col("m").struct.field("n_eff") == rows - 1)
        assert (last["m"].struct.field("log_e_pos_d0") >= LN20).mean() <= rate

    def test_more_rows_more_power(self):
        k = 1000
        rates = []
        for rows in (200, 1000, 4000):
            n = k * rows
            df = pl.DataFrame(
                {"d0": signs(n, p=0.55, seed=15), "g": (np.arange(n) % k).astype(np.int64)}
            )
            rates.append(crossing_rates(df, [LN20])[LN20])
        assert rates[0] < rates[1] < rates[2], rates

    def test_the_size_of_the_values_is_invisible(self):
        # Up by a hair 60% of the time, down by a mile the rest: the mean is
        # hugely negative and the sign test still finds "positive" -- it is
        # a test of the sign, and says so.
        rng = np.random.default_rng(16)
        n = 5000
        d = np.where(rng.random(n) < 0.6, 1e-9, -1e6)
        u = unnested(po.ModelBank([spec()]).fit_predict(pl.DataFrame({"d0": d})))
        assert d.mean() < -1e5
        assert u["log_e_pos_d0"][-1] > LN100 and u["log_e_neg_d0"][-1] <= 0.0

    def test_ties_are_not_trials(self):
        rng = np.random.default_rng(17)
        n = 4000
        d = np.where(rng.random(n) < 0.5, 0.0, signs(n, p=0.5, seed=18))
        df = pl.DataFrame({"d0": d})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        dense = unnested(po.ModelBank([spec()]).fit_predict(df.filter(pl.col("d0") != 0.0)))
        # The nonzero rows alone give the same evidence and counts...
        keep = (d != 0.0).nonzero()[0]
        for f in ("log_e_pos_d0", "log_e_neg_d0", "n_pos_d0", "n_neg_d0"):
            assert u[f].gather(keep).to_list() == dense[f].to_list(), f
        # ...while every row counts toward n_eff (a tie is a row seen).
        assert u["n_eff"][-1] == n - 1


# ----------------------------------------------------------- the comparison


def regression(n, seed, groups=None, null_every=0):
    """A stream with a slope that drifts, where a shorter memory wins."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    slope = 1.0 + np.sin(np.arange(n) / 90.0)
    y = 0.5 + slope * x0 - 0.7 * x1 + 0.3 * rng.standard_normal(n)
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": y})
    if null_every:
        df = df.with_columns(
            pl.when(pl.int_range(pl.len()) % null_every == 3).then(None).otherwise("y").alias("y")
        )
    if groups is not None:
        df = df.with_columns(pl.Series("g", [groups[i % len(groups)] for i in range(n)]))
    return df


def two_sides(halflife_a=20.0, halflife_b=400.0, **kw):
    # `coef_every=1`: by default `coef` is reported on each chunk's last row
    # (README, "Two guarantees"), so whole frames compare across chunkings
    # only when it is reported on every row.
    common = dict(targets=["y"], features=["x0", "x1"], min_periods=5.0, coef_every=1, **kw)
    return [
        po.spec.ewridge("fast", halflife=halflife_a, **common),
        po.spec.ewridge("slow", halflife=halflife_b, **common),
    ]


class TestTheComparison:
    def test_the_comparison_is_column_mode_on_the_residual_difference(self):
        df = regression(4000, seed=20, groups=["p", "q"], null_every=13)
        sides = two_sides(group="g")
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow", group="g")
        out = po.ModelBank([*sides, c]).fit_predict(df)
        # By hand: |resid_slow| - |resid_fast| as a column, column mode on it.
        diff = out.with_columns(
            (
                pl.col("slow").struct.field("resid_y").abs()
                - pl.col("fast").struct.field("resid_y").abs()
            ).alias("d")
        )
        by_hand = unnested(po.ModelBank([spec(targets=["d"], group="g")]).fit_predict(diff))
        got = unnested(out, "c")
        for a, b in (
            ("log_e_a_y", "log_e_pos_d"),
            ("log_e_b_y", "log_e_neg_d"),
            ("wins_a_y", "n_pos_d"),
            ("wins_b_y", "n_neg_d"),
            ("n_eff", "n_eff"),
        ):
            assert got[a].equals(by_hand[b], null_equal=True), a
        # The short memory tracks the drifting slope: it wins.
        last = got.filter(pl.col("n_eff") == 1999)
        assert (last["wins_a_y"] > last["wins_b_y"]).all()
        assert (last["log_e_a_y"] > LN100).all() and (last["log_e_b_y"] <= 0.0).all()

    def test_it_reads_the_out_of_sample_residuals_the_structs_report(self):
        # Regenerate the residuals from the predictions the sides emitted and
        # the twin agrees: the comparison saw exactly those.
        df = regression(3000, seed=21)
        sides = two_sides()
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow")
        out = po.ModelBank([*sides, c]).fit_predict(df)
        for side in ("fast", "slow"):
            pred = out[side].struct.field("pred_y")
            resid = out[side].struct.field("resid_y")
            assert ((out["y"] - pred) - resid).abs().fill_null(0.0).max() < 1e-12
        twin = po.eval.seqtest(out, a="fast", b="slow")
        assert unnested(out, "c").equals(twin["seqtest"].struct.unnest(), null_equal=True)

    def test_a_suffix_picks_the_grid_instance(self):
        df = regression(3000, seed=22)
        common = dict(targets=["y"], features=["x0", "x1"], min_periods=5.0)
        grid = po.spec.ewridge("grid", halflife=[20.0, 400.0], ridge=[1e-6, 0.5], **common)
        kalman = po.spec.kalman("kalman", halflife=100.0, coef_halflife=50.0, **common)
        picked = po.spec.seqtest("c", targets=["y"], a="grid", a_suffix="__r0.5@h20", b="kalman")
        out = po.ModelBank([grid, kalman, picked]).fit_predict(df)
        plain = po.spec.ewridge("plain", halflife=20.0, ridge=0.5, **common)
        c = po.spec.seqtest("c", targets=["y"], a="plain", b="kalman")
        want = po.ModelBank([plain, kalman, c]).fit_predict(df)
        assert unnested(out, "c").equals(unnested(want, "c"), null_equal=True)
        # And on both sides, against the grid's own other instance.
        both = po.spec.seqtest(
            "c", targets=["y"], a="grid", a_suffix="__r0.5@h20", b="grid", b_suffix="__r0.5@h400"
        )
        assert unnested(po.ModelBank([grid, both]).fit_predict(df), "c")["wins_a_y"][-1] > 1000

    def test_a_row_either_side_sits_out_is_no_trial(self):
        df = regression(2000, seed=23, null_every=9)
        common = dict(targets=["y"], features=["x0", "x1"])
        # Different warmups: rows 5..39 have a fast residual and no slow one.
        sides = [
            po.spec.ewridge("fast", halflife=20.0, min_periods=5.0, **common),
            po.spec.ewridge("slow", halflife=400.0, min_periods=40.0, **common),
        ]
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow")
        out = po.ModelBank([*sides, c]).fit_predict(df)
        u = unnested(out, "c")
        both = (
            out["fast"].struct.field("resid_y").is_not_null()
            & out["slow"].struct.field("resid_y").is_not_null()
        )
        trials = u["wins_a_y"] + u["wins_b_y"]
        assert trials[-1] == both.sum() - (1 if both[-1] else 0)
        # Nothing moved before the first row both sides scored...
        first = both.arg_true()[0]
        assert first >= 40
        assert (u["wins_a_y"][: first + 1] == 0).all() and (
            u["log_e_a_y"][: first + 1] == 0.0
        ).all()
        # ...while every row is a row seen.
        assert u["n_eff"].to_list() == list(map(float, range(2000)))

    def test_multiple_targets_and_two_comparisons_in_one_bank(self):
        df = regression(3000, seed=24).with_columns((pl.col("y") * -2.0 + 1.0).alias("z"))
        common = dict(targets=["y", "z"], features=["x0", "x1"], min_periods=5.0)
        sides = [
            po.spec.ewridge("fast", halflife=20.0, **common),
            po.spec.ewridge("slow", halflife=400.0, **common),
        ]
        ab = po.spec.seqtest("ab", targets=["y", "z"], a="fast", b="slow")
        ba = po.spec.seqtest("ba", targets=["z"], a="slow", b="fast")
        out = po.ModelBank([ab, *sides, ba]).fit_predict(df)
        assert out.columns == ["t", "x0", "x1", "y", "z", "ab", "fast", "slow", "ba"]
        u, v = unnested(out, "ab"), unnested(out, "ba")
        assert u.columns == [
            "log_e_a_y",
            "log_e_b_y",
            "wins_a_y",
            "wins_b_y",
            "log_e_a_z",
            "log_e_b_z",
            "wins_a_z",
            "wins_b_z",
            "n_eff",
        ]
        # `z` is an affine image of `y`, so its comparison is the same.
        for f in ("log_e_a", "log_e_b", "wins_a", "wins_b"):
            assert u[f"{f}_y"].equals(u[f"{f}_z"], null_equal=True), f
        # The reverse comparison is the same test with the sides swapped.
        assert u["log_e_a_z"].equals(v["log_e_b_z"]) and u["wins_a_z"].equals(v["wins_b_z"])
        assert u["log_e_b_z"].equals(v["log_e_a_z"]) and u["wins_b_z"].equals(v["wins_a_z"])

    def test_scoring_reads_the_state_before_the_chunk(self):
        df = regression(2000, seed=25)
        sides = two_sides()
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow")
        bank = po.ModelBank([*sides, c])
        bank.fit_predict(df.slice(0, 1000))
        rest = df.slice(1000, 1000)
        scored = bank.predict(rest)
        # Scoring learns nothing: score it twice, then fit, and compare.
        assert scored.equals(bank.predict(rest), null_equal=True)
        fitted = bank.fit_predict(rest)
        # The first row agrees (the same state before it); the comparison
        # under scoring stays at the state of row 1000 while the fit moves.
        u, v = unnested(scored, "c"), unnested(fitted, "c")
        assert u.row(0) == v.row(0)
        assert u["wins_a_y"].n_unique() == 1 and v["wins_a_y"].n_unique() > 1
        # And the residuals it compared were the scored sides' residuals.
        twin = po.eval.seqtest(scored, a="fast", b="slow")["seqtest"].struct.unnest()
        assert twin["wins_a_y"][-1] + twin["wins_b_y"][-1] > 900

    def test_a_refused_chunk_updates_neither_phase(self):
        df = regression(1500, seed=26)
        sides = two_sides(clock="t", max_dclock=5.0, on_clock_reset="error")
        c = po.spec.seqtest(
            "c",
            targets=["y"],
            a="fast",
            b="slow",
            clock="t",
            max_dclock=5.0,
            on_clock_reset="error",
        )
        bank = po.ModelBank([*sides, c])
        bank.fit_predict(df.slice(0, 500))
        bad = df.slice(500, 500).with_columns(
            pl.when(pl.int_range(pl.len()) == 250).then(-1.0).otherwise("t").alias("t")
        )
        with pytest.raises(ValueError, match="clock"):
            bank.fit_predict(bad)
        fresh = po.ModelBank([*sides, c])
        fresh.fit_predict(df.slice(0, 500))
        rest = df.slice(1000, 500)
        assert bank.fit_predict(rest).equals(fresh.fit_predict(rest), null_equal=True)

    def test_a_comparison_is_chunk_invariant_over_interleaved_groups(self):
        df = regression(3000, seed=27, groups=["p", "q", "r"], null_every=11)
        sides = two_sides(group="g")
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow", group="g")
        one = po.ModelBank([*sides, c]).fit_predict(df)
        for size in (1, 61, 500):
            bank = po.ModelBank([*sides, c])
            many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 3000, size)])
            assert one.equals(many, null_equal=True), size

    def test_the_sides_can_be_grouped_while_the_comparison_pools(self):
        df = regression(3000, seed=28, groups=["p", "q"])
        sides = two_sides(group="g")
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow")
        out = po.ModelBank([*sides, c]).fit_predict(df)
        u = unnested(out, "c")
        assert u["n_eff"][-1] == 2999.0
        assert u["wins_a_y"][-1] + u["wins_b_y"][-1] > 2900
        twin = po.eval.seqtest(out, a="fast", b="slow")["seqtest"].struct.unnest()
        assert u.equals(twin, null_equal=True)


# ------------------------------------------------------------- edge cases


class TestEdgeCases:
    def test_outputs_are_null_until_min_periods_and_n_eff_always(self):
        df = frame(n=50, seed=30)
        u = unnested(po.ModelBank([spec(min_periods=10.0)]).fit_predict(df))
        for f in ("log_e_pos_d0", "log_e_neg_d0", "n_pos_d0", "n_neg_d0"):
            assert u[f][:10].null_count() == 10 and u[f][10:].null_count() == 0, f
        assert u["n_eff"].to_list() == list(map(float, range(50)))
        # min_periods defaults to 0: the first row already reports.
        u0 = unnested(po.ModelBank([spec()]).fit_predict(df))
        assert u0.null_count().sum_horizontal()[0] == 0
        assert u0.row(0) == (0.0, 0.0, 0, 0, 0.0)

    def test_min_periods_per_target(self):
        df = frame(n=40, m=2, seed=31)
        u = unnested(
            po.ModelBank([spec(targets=["d0", "d1"], min_periods=[0.0, 20.0])]).fit_predict(df)
        )
        assert u["log_e_pos_d0"].null_count() == 0
        assert u["n_pos_d1"][:20].null_count() == 20 and u["n_pos_d1"][20:].null_count() == 0

    def test_null_zero_and_out_of_bound_rows_are_seen_but_not_trials(self):
        vals = [1.0, None, -2.0, 0.0, float("nan"), 3.0, 1e101, -0.0, 5e-324, -5e-324, 1e100]
        df = pl.DataFrame({"d0": pl.Series(vals, dtype=pl.Float64)})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        assert u["n_eff"].to_list() == list(map(float, range(len(vals))))
        # Trials: 1, -2, 3, 5e-324 (a denormal is a sign), -5e-324, 1e100 (at the bound).
        assert u["n_pos_d0"].to_list() == [0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3]
        assert u["n_neg_d0"].to_list() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]
        # A row that is not a trial leaves the evidence where it was.
        assert u["log_e_pos_d0"][1] == u["log_e_pos_d0"][2]
        assert u["log_e_pos_d0"][3] == u["log_e_pos_d0"][4] == u["log_e_pos_d0"][5]

    def test_an_integer_column_is_a_target(self):
        df = pl.DataFrame({"d0": pl.Series([1, -1, 1, 1, 0, -1, 1], dtype=pl.Int32)})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        assert u["n_pos_d0"][-1] == 3 and u["n_neg_d0"][-1] == 2

    def test_a_losing_bet_never_empties_the_wealth(self):
        # 1 - lambda >= 1 / (n + 1): the worst row costs ln(n + 1), so a run
        # of one sign followed by the other side's run stays finite and the
        # process recovers when the other side leads.
        df = pl.DataFrame({"d0": [1.0] * 300 + [-1.0] * 900})
        u = unnested(po.ModelBank([spec()]).fit_predict(df))
        assert u["log_e_pos_d0"].is_finite().all() and u["log_e_neg_d0"].is_finite().all()
        assert u["log_e_pos_d0"][300] > LN100  # after 300 positives
        assert u["log_e_pos_d0"][-1] < 0.0  # and then it lost
        assert u["log_e_neg_d0"][-1] > LN100  # while the negative side won

    def test_chunk_invariance(self):
        df = frame(n=1500, m=2, seed=32, p=0.6, null_every=7, zero_every=10, groups=["p", "q", "r"])
        s = spec(targets=["d0", "d1"], group="g", min_periods=3.0)
        one = po.ModelBank([s]).fit_predict(df)
        for size in (1, 7, 97, 1000):
            bank = po.ModelBank([s])
            many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 1500, size)])
            assert one.equals(many, null_equal=True), size

    def test_save_load_and_pickle(self, tmp_path):
        df = frame(n=1000, seed=33, p=0.6, null_every=9, groups=["p", "q"])
        s = spec(group="g")
        for cut in (1, 250, 999):
            a = po.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            path = tmp_path / f"c{cut}.state"
            a.save(path)
            b = po.ModelBank.load(path, specs=[s])
            c = pickle.loads(pickle.dumps(a))
            rest = df.slice(cut, 1000 - cut)
            want = a.fit_predict(rest)
            assert want.equals(b.fit_predict(rest), null_equal=True), cut
            assert want.equals(c.fit_predict(rest), null_equal=True), cut

    def test_groups_are_independent_and_a_null_key_is_a_group(self):
        df = frame(n=900, seed=34, p=0.7, groups=["p", "q", None])
        both = po.ModelBank([spec(group="g")]).fit_predict(df)
        for g in ("p", None):
            mask = pl.col("g").is_null() if g is None else pl.col("g") == g
            solo = po.ModelBank([spec()]).fit_predict(df.filter(mask).drop("g"))
            assert unnested(both.filter(mask)).equals(unnested(solo), null_equal=True), g
        assert sorted(
            po.ModelBank([spec(group="g")]).fit_predict(df)["g"].unique().to_list(), key=str
        )

    def test_the_clock_neither_decays_nor_stops_it(self):
        # No decay: the same stream on a regular clock, an irregular one and
        # none at all gives the same evidence. What the clock adds is the
        # restart, under `reset_state`.
        df = frame(n=800, seed=35, p=0.6)
        plain = unnested(po.ModelBank([spec()]).fit_predict(df))
        clocked = unnested(po.ModelBank([spec(clock="t", max_dclock=1.0)]).fit_predict(df))
        assert plain.equals(clocked, null_equal=True)
        jumpy = df.with_columns((pl.col("t") ** 2).alias("t"))
        assert plain.equals(
            unnested(po.ModelBank([spec(clock="t", max_dclock=1e6)]).fit_predict(jumpy)),
            null_equal=True,
        )
        back = df.with_columns(
            pl.when(pl.int_range(pl.len()) >= 400)
            .then(pl.col("t") - 5000.0)
            .otherwise("t")
            .alias("t")
        )
        restarted = unnested(
            po.ModelBank(
                [spec(clock="t", max_dclock=10.0, on_clock_reset="reset_state")]
            ).fit_predict(back)
        )
        assert restarted["n_eff"][400] == 0.0 and restarted["n_eff"][399] == 399.0
        assert restarted[:400].equals(plain[:400], null_equal=True)
        fresh = unnested(po.ModelBank([spec()]).fit_predict(df[400:]))
        assert restarted[400:].equals(fresh, null_equal=True)
        kept = unnested(po.ModelBank([spec(clock="t", max_dclock=10.0)]).fit_predict(back))
        assert kept.equals(plain, null_equal=True)

    def test_lazy_path_equals_bank(self):
        df = frame(n=2000, m=2, seed=36, null_every=5, groups=["p", "q"])
        s = spec(targets=["d0", "d1"], group="g", min_periods=2.0)
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s], chunk_rows=128).collect()
        assert bank.equals(lazy, null_equal=True)
        assert df.online.fit_predict([s]).equals(bank, null_equal=True)

    def test_the_runner_agrees_with_the_bank(self, tmp_path):
        df = regression(2000, seed=37, groups=["p", "q", "r"], null_every=7)
        sides = two_sides(group="g")
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow", group="g")
        want = po.ModelBank([*sides, c]).fit_predict(df)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        po.run(input=str(src), output=str(dst), specs=[*sides, c], chunk_rows=300)
        assert want.equals(pl.read_parquet(dst), null_equal=True)

    def test_determinism(self):
        df = frame(n=3000, seed=38, groups=["p", "q"])
        runs = [po.ModelBank([spec(group="g")]).fit_predict(df) for _ in range(3)]
        assert runs[0].equals(runs[1]) and runs[0].equals(runs[2])

    def test_expression_equals_bank(self):
        df = frame(n=800, m=2, seed=39, null_every=6, groups=["p", "q"])
        bank = unnested(
            po.ModelBank([spec(targets=["d0", "d1"], group="g", min_periods=2.0)]).fit_predict(df)
        )
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("d0").online.seqtest(extra_targets=["d1"], min_periods=2.0).over("g")
            ).unnest("d0")
        assert bank.equals(expr, null_equal=True)
        with pytest.warns(po.InMemoryExpressionWarning):
            typed = df.select(po.online(pl.col("d0")).seqtest(min_periods=2.0).over("g")).unnest(
                "d0"
            )
        assert typed.equals(
            bank.select([c for c in bank.columns if not c.endswith("_d1")]), null_equal=True
        )

    def test_the_expression_refuses_a_comparison_and_says_how_to_write_it(self):
        with pytest.raises(
            TypeError, match=r"seqtest a, b compare two specs of a bank.*\|resid_b\|"
        ):
            pl.col("d").online.seqtest(a="fast", b="slow")
        with pytest.raises(TypeError, match="seqtest a_suffix compare"):
            pl.col("d").online.seqtest(a_suffix="@h20")
        # The way it says: column mode on the difference.
        df = regression(1500, seed=40)
        sides = two_sides()
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow")
        out = po.ModelBank([*sides, c]).fit_predict(df)
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = out.select(
                (
                    pl.col("slow").struct.field("resid_y").abs()
                    - pl.col("fast").struct.field("resid_y").abs()
                )
                .alias("d")
                .online.seqtest()
            ).unnest("d")
        assert expr["log_e_pos_d"].equals(out["c"].struct.field("log_e_a_y"), null_equal=True)
        assert expr["n_neg_d"].equals(out["c"].struct.field("wins_b_y"), null_equal=True)

    def test_output_index_declares_the_fields_and_dtypes(self):
        s = spec(targets=["d0", "d1"])
        idx = po.spec.output_index(s)
        assert idx["field"].to_list() == po.spec.output_fields(s)
        assert idx["dtype"].to_list() == ["f64", "f64", "i64", "i64"] * 2 + ["f64"]
        assert idx["kind"].to_list() == ["log_e_pos", "log_e_neg", "n_pos", "n_neg"] * 2 + ["n_eff"]
        assert idx["target"].to_list() == ["d0"] * 4 + ["d1"] * 4 + [None]
        assert idx["halflife"].null_count() == idx.height  # nothing decays
        c = po.spec.seqtest("c", targets=["y"], a="p", b="q")
        assert po.spec.output_fields(c) == [
            "log_e_a_y",
            "log_e_b_y",
            "wins_a_y",
            "wins_b_y",
            "n_eff",
        ]
        assert po.spec.output_index(c)["dtype"].to_list() == ["f64", "f64", "i64", "i64", "f64"]
        out = po.ModelBank([s]).fit_predict(frame(n=5, m=2))
        assert out.schema["m"] == pl.Struct(
            {
                "log_e_pos_d0": pl.Float64,
                "log_e_neg_d0": pl.Float64,
                "n_pos_d0": pl.Int64,
                "n_neg_d0": pl.Int64,
                "log_e_pos_d1": pl.Float64,
                "log_e_neg_d1": pl.Float64,
                "n_pos_d1": pl.Int64,
                "n_neg_d1": pl.Int64,
                "n_eff": pl.Float64,
            }
        )
        unnest = po.ModelBank([s]).fit_predict(frame(n=5, m=2)).online.unnest([s])
        assert "n_pos_d1" in unnest.columns and "m" not in unnest.columns

    def test_it_has_no_coefficients(self):
        s = spec()
        assert "coef" not in po.spec.output_fields(s)
        assert po.spec.coef_fields(s).height == 0
        with pytest.raises(ValueError, match="seqtest emits evidence .* not coefficients"):
            po.spec.coef_index(s)
        bank = po.ModelBank([s])
        bank.fit_predict(frame(n=20))
        with pytest.raises(ValueError, match="seqtest emits evidence"):
            bank.coef("m")
        assert bank.solve_failures() == {"m": {"": 0}}
        assert bank.gram("m") == []

    def test_unpack_says_what_a_seqtest_struct_holds(self):
        out = po.ModelBank([spec()]).fit_predict(frame(n=20))
        with pytest.raises(TypeError, match="a seqtest struct evidence, not predictions"):
            po.eval.unpack(out, "m")

    def test_the_cli_runs_both_modes(self, tmp_path, online_cli):
        df = regression(2000, seed=41, groups=["p", "q"], null_every=7)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            "\n".join(
                [
                    f'input = "{src.as_posix()}"',
                    f'output = "{dst.as_posix()}"',
                    "chunk_rows = 300",
                    "[[specs]]",
                    'name = "fast"',
                    'targets = ["y"]',
                    'features = ["x0", "x1"]',
                    "halflife = 20.0",
                    "min_periods = 5.0",
                    "coef_every = 1",
                    'group = "g"',
                    "[specs.model]",
                    'type = "ew_ridge"',
                    "[[specs]]",
                    'name = "slow"',
                    'targets = ["y"]',
                    'features = ["x0", "x1"]',
                    "halflife = 400.0",
                    "min_periods = 5.0",
                    "coef_every = 1",
                    'group = "g"',
                    "[specs.model]",
                    'type = "ew_ridge"',
                    "[[specs]]",
                    'name = "c"',
                    'targets = ["y"]',
                    "features = []",
                    'group = "g"',
                    "[specs.model]",
                    'type = "seqtest"',
                    'a = "fast"',
                    'b = "slow"',
                    "[[specs]]",
                    'name = "sign"',
                    'targets = ["x0"]',
                    "features = []",
                    "[specs.model]",
                    'type = "seqtest"',
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = pl.read_parquet(dst)
        sides = two_sides(group="g")
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="slow", group="g")
        sign = po.spec.seqtest("sign", targets=["x0"])
        want = po.ModelBank([*sides, c, sign]).fit_predict(df)
        assert got.equals(want, null_equal=True)
        assert got.schema["c"].fields[2].dtype == pl.Int64

    def test_the_cli_dry_run_lists_the_residual_fields_a_mismatch_needs(self, tmp_path, online_cli):
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            "\n".join(
                [
                    'input = "in.parquet"',
                    'output = "out.parquet"',
                    "[[specs]]",
                    'name = "ridge"',
                    'targets = ["y"]',
                    'features = ["x0"]',
                    "halflife = 20.0",
                    "[specs.model]",
                    'type = "ew_ridge"',
                    "ridge = [1e-6, 0.1]",
                    "[[specs]]",
                    'name = "c"',
                    'targets = ["y"]',
                    "features = []",
                    "[specs.model]",
                    'type = "seqtest"',
                    'a = "ridge"',
                    'a_suffix = "__r0.1"',
                    'b = "ridge"',
                    'b_suffix = "__r0.5"',
                ]
            )
        )
        res = subprocess.run(
            [str(online_cli), "--config", str(cfg), "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode != 0, res.stdout
        assert 'target "y" names no residual of b = "ridge"' in res.stderr, res.stderr
        assert 'it has no field "resid_y__r0.5"' in res.stderr
        assert '["resid_y__r0.000001", "resid_y__r0.1"]' in res.stderr
        # And the two instances it has compare, both ways, in a dry run and a bank.
        ok = cfg.read_text().replace('b_suffix = "__r0.5"', 'b_suffix = "__r0.000001"')
        cfg.write_text(ok)
        res = subprocess.run(
            [str(online_cli), "--config", str(cfg), "--dry-run"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "c (seqtest):" in res.stdout and "c.wins_a_y" in res.stdout


# --------------------------------------------------------------- refusals


class TestRefusals:
    @pytest.mark.parametrize(
        "kw, msg",
        [
            (dict(features=["x0"]), "seqtest takes no features"),
            (dict(a="p"), "seqtest a and b go together"),
            (dict(b="q"), "seqtest a and b go together"),
            (dict(a="p", b="p"), 'seqtest a and b are both "p" with the same suffix'),
            (dict(a="p", b="p", a_suffix="@h20", b_suffix="@h20"), "with the same suffix"),
            (dict(a="m", b="q"), "seqtest a/b name the spec itself"),
            (dict(a_suffix="@h20"), "seqtest a_suffix/b_suffix pick a side's grid instance"),
            (dict(weight="w"), "weight does not apply to seqtest"),
            (dict(halflife=20.0), "halflife/lam do not apply to seqtest"),
            (dict(lam=0.99), "halflife/lam do not apply to seqtest"),
            (dict(emit_sigma=True), "emit_sigma does not apply to seqtest"),
            (dict(emit_resid_z=True), "emit_resid_z does not apply to seqtest"),
            (dict(conformal=0.9), "conformal does not apply to seqtest"),
            (dict(emit_drift=True), "emit_drift does not apply to seqtest"),
            (dict(emit_metrics=True), "emit_metrics does not apply to seqtest"),
            (dict(min_periods=-1.0), "min_periods must be >= 0"),
        ],
    )
    def test_the_spec_refuses_by_name(self, kw, msg):
        with pytest.raises((ValueError, TypeError), match=msg):
            spec(**kw)

    def test_no_targets_is_refused(self):
        with pytest.raises(ValueError, match="targets"):
            po.spec.seqtest("m", targets=[])

    def test_a_side_the_bank_has_not_got(self):
        c = po.spec.seqtest("c", targets=["y"], a="fast", b="nope")
        with pytest.raises(
            ValueError,
            match=r'seqtest b = "nope" is not a spec of this bank \(the bank has \["fast", "c"\]\)',
        ):
            po.ModelBank([two_sides()[0], c])

    def test_a_side_that_is_itself_a_test(self):
        c1 = po.spec.seqtest("c1", targets=["d0"])
        c2 = po.spec.seqtest("c2", targets=["d0"], a="c1", b="c1_")
        with pytest.raises(
            ValueError, match='seqtest a = "c1" is itself a seqtest and has no residuals'
        ):
            po.ModelBank([c1, po.spec.seqtest("c1_", targets=["d0"]), c2])

    def test_a_target_neither_side_has(self):
        sides = two_sides()
        c = po.spec.seqtest("c", targets=["z"], a="fast", b="slow")
        with pytest.raises(
            ValueError,
            match=(
                r'target "z" names no residual of a = "fast": it has no field "resid_z" '
                r'\(its residual fields are \["resid_y"\]\)'
            ),
        ):
            po.ModelBank([*sides, c])
        grid = po.spec.ewridge("grid", targets=["y"], features=["x0"], halflife=[20.0, 400.0])
        c = po.spec.seqtest("c", targets=["y"], a="grid", b="slow")
        with pytest.raises(ValueError, match=r'\["resid_y@h20", "resid_y@h400"\]'):
            po.ModelBank([grid, sides[1], c])

    def test_the_eval_twin_refuses_the_same_things(self):
        df = regression(200, seed=42)
        sides = two_sides()
        out = po.ModelBank(sides).fit_predict(df)
        with pytest.raises(ValueError, match="a and b go together"):
            po.eval.seqtest(out, a="fast")
        with pytest.raises(ValueError, match="targets name the columns whose sign is tested"):
            po.eval.seqtest(out)
        with pytest.raises(ValueError, match=r"target 'z' names no residual of 'fast'"):
            po.eval.seqtest(out, a="fast", b="slow", targets=["z"])
        with pytest.raises(ValueError, match=r"target 'y' names no residual of 'slow'"):
            po.eval.seqtest(out, a="fast", b="slow", targets=["y"], b_suffix="@h400")
        with pytest.raises(ValueError, match="share no residual field"):
            po.eval.seqtest(out, a="fast", b="slow", a_suffix="@h20")
        with pytest.raises(KeyError):
            po.eval.seqtest(out, a="fast", b="nope")
        with pytest.raises(TypeError, match="not a model-output struct"):
            po.eval.seqtest(out, a="fast", b="y")

    def test_a_wrong_type_names_the_parameter(self):
        with pytest.raises(TypeError, match='spec "m": a must be a str, got int 1'):
            po.spec.seqtest("m", targets=["d"], a=1, b="q")
        with pytest.raises(TypeError, match="seqtest\\(\\) got an unexpected keyword argument 'c'"):
            po.spec.seqtest("m", targets=["d"], c="q")
