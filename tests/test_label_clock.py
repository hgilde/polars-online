"""`label_clock`: each row's label time as a column (task 341).

A row is scored where it sits, held until the first row whose clock reaches
its label time, and learned at that label time, with the prediction and
interval it was served scored against its label. The Rust tests
(`crates/online-polars/tests/label_clock.rs`) hold it to the doubled stream
and to `embargo`; here the oracle is a fold, written out in numpy, of each
served residual `y - pred` in release order: the residual spread `sigma`,
the conformal interval and its coverage, and `hit_rate`.
"""

import math
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from test_conformal import norm_ppf

TIER = "essential"

GAP_CAP = 25.0
HALF_LIFE = 40.0


def frame(n=600, seed=0):
    """An integer clock of whole steps; a label time 0 to 30 units after its
    row, so a short delay often overtakes a long one, null on every
    seventeenth row; a weight of 0 on every nineteenth; `x` null on every
    twenty-third, which skips the row."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(0, 4, n))
    delay = rng.integers(0, 31, n)
    avail = [None if i % 17 == 5 else int(t[i] + delay[i]) for i in range(n)]
    x = [None if i % 23 == 7 else float(v) for i, v in enumerate(rng.standard_normal(n))]
    y = np.array([2.0 * (v or 0.0) for v in x]) + rng.standard_normal(n) * (
        1 + (np.arange(n) > n // 2)
    )
    w = np.where(np.arange(n) % 19 == 3, 0.0, rng.uniform(0.5, 1.5, n))
    return pl.DataFrame(
        {"t": t, "avail": avail, "x": x, "y": y, "w": w},
        schema_overrides={"avail": pl.Int64},
    )


def spec(**kw):
    kw.setdefault("label_clock", "avail")
    return po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=GAP_CAP,
        half_life=HALF_LIFE,
        weight="w",
        min_weight=0.0,
        max_error_inflation=float("inf"),
        conformal=0.9,
        emit_sigma=True,
        emit_metrics=True,
        **kw,
    )


def exp_neg(x):
    """`e^{-x}` as `online_core::conformal::exp_neg` computes it, in basic
    arithmetic: `x = k ln 2 + r`, the Taylor series of `e^{-r}`, times
    `2^{-k}`. The gain is held to scipy's normal density separately
    (`test_the_gain_is_the_normal_density`)."""
    if x <= 0.0:
        return 1.0
    ln2_hi, ln2_lo = 0.6931471803691238, 1.9082149292705877e-10
    k = math.floor(x / 0.6931471805599453)
    r = (x - k * ln2_hi) - k * ln2_lo
    term = total = 1.0
    for i in range(1, 21):
        term *= -r / float(i)
        total += term
    return total * 2.0 ** (-k)


def gain(alpha):
    """`g(α) = 2φ(Φ⁻¹(1 − α/2))`, the step's loop gain per unit of rate."""
    z = norm_ppf(1.0 - 0.5 * alpha)
    return 2.0 * 0.3989422804014327 * exp_neg(0.5 * z * z)


def fold(df, pred, coverage=0.9, rate=0.05, auto=False):
    """The served residuals folded in release order: at each accepted row,
    the held rows whose label time the row's clock has reached are released
    in order of label time and then arrival, each decaying the folds by its
    step from the label time released before it (capped) and folding
    `y - pred` at its weight, its interval scored against the radius it was
    shown; then the row reads `sigma`, its interval and the coverage, and is
    held, unless its weight is 0 or its label time null.

    `auto`, `conformal_rate="auto"`: each release with a residual also
    folds the count of rows still waiting, itself aside, into `n̄`, an EW
    mean on the count `w̄` is taken over, and the step is `rate · sigma`
    divided by `1 + rate · g(α) · n̄ / (1/2)`."""
    t = df["t"].to_list()
    avail = df["avail"].to_list()
    x = df["x"].to_list()
    y = df["y"].to_list()
    w = df["w"].to_list()
    n = df.height
    alpha = 1.0 - coverage
    var = rw = 0.0
    q, cov, cov_w, w_mean, w_rows, nbar = math.nan, 0.0, 0.0, 0.0, 0.0, 0.0
    g = gain(alpha)
    hits = hit_w = 0.0
    last = None
    held = []  # (label, arrival, shown radius)
    sigma, lo, hi, cover, hit_rate = (np.full(n, np.nan) for _ in range(5))
    for i in range(n):
        if x[i] is None:
            continue
        due = sorted(h for h in held if h[0] <= t[i])
        held = [h for h in held if h[0] > t[i]]
        for k, (label, j, shown) in enumerate(due):
            d = 0.0 if last is None else min(float(label - last), GAP_CAP)
            last = label
            lam = math.exp2(-(d / HALF_LIFE))
            r = y[j] - pred[j]
            s = math.sqrt(max(var, 0.0)) if rw > 0.0 else math.nan
            if not math.isfinite(r):
                # Served no prediction: no residual, and every fold ages.
                cov_w *= lam
                w_rows *= lam
                rw *= lam
                hits *= lam
                hit_w *= lam
                continue
            # The conformal step, against the radius the row was shown.
            w_rows = lam * w_rows + 1.0
            w_mean += (w[j] - w_mean) / w_rows
            if auto:
                waiting = len(held) + len(due) - k - 1
                nbar += (waiting - nbar) / w_rows
            usable = math.isfinite(s) and s > 0.0
            if not math.isfinite(q):
                if usable:
                    q = s * norm_ppf(1.0 - 0.5 * alpha)
                cov_w *= lam
            elif not math.isfinite(shown):
                cov_w *= lam
            else:
                miss = abs(r) > shown
                cw = lam * cov_w + w[j]
                cov = (lam * cov_w * cov + w[j] * (0.0 if miss else 1.0)) / cw
                cov_w = cw
                eta = rate * s if usable else 0.0
                if nbar > 0.0:
                    eta /= 1.0 + rate * g * nbar / 0.5
                q = max(q + eta * (w[j] / w_mean) * ((1.0 if miss else 0.0) - alpha), 0.0)
            # The spread.
            w_new = lam * rw + w[j]
            var = (lam * rw * var + w[j] * r * r) / w_new
            rw = w_new
            # The hit rate, by sign.
            hit_w = lam * hit_w + w[j]
            hits = lam * hits + w[j] * float(np.sign(pred[j]) == np.sign(y[j]))
        if rw > 0.0:
            sigma[i] = math.sqrt(max(var, 0.0))
        lo[i], hi[i] = pred[i] - q, pred[i] + q
        if cov_w > 0.0:
            cover[i] = cov
        if hit_w > 0.0:
            hit_rate[i] = hits / hit_w
        if avail[i] is not None and w[i] > 0.0:
            held.append((avail[i], i, q))
    return {"sigma_y": sigma, "lo_y": lo, "hi_y": hi, "coverage_y": cover, "hit_rate_y": hit_rate}


def field(out, name):
    return out["m"].struct.field(name).to_numpy().astype(float)


class TestTheServedResidualsInReleaseOrder:
    def test_sigma_the_interval_and_the_coverage_are_the_fold(self):
        df = frame()
        out = po.ModelBank([spec()]).fit_predict(df)
        pred = field(out, "pred_y")
        want = fold(df, pred)
        for key, exp in want.items():
            got = field(out, key)
            assert (np.isnan(got) == np.isnan(exp)).all(), key
            fin = np.isfinite(exp)
            assert fin.sum() > 300, key
            if key == "hit_rate_y":
                # The bank keeps the rate as a running mean; the fold as a
                # ratio of two sums, which rounds apart in the last bits.
                np.testing.assert_allclose(got[fin], exp[fin], rtol=1e-12, atol=0, err_msg=key)
            else:
                # The same operations in the same order: to the bit.
                np.testing.assert_array_equal(got[fin], exp[fin], err_msg=key)

    def test_the_residual_on_a_row_is_its_served_one(self):
        df = frame()
        out = po.ModelBank([spec()]).fit_predict(df)
        resid, pred = field(out, "resid_y"), field(out, "pred_y")
        fin = np.isfinite(pred)
        np.testing.assert_array_equal(resid[fin], df["y"].to_numpy()[fin] - pred[fin])

    def test_without_the_release_order_the_coverage_would_differ(self):
        """The fold is not one any order passes: in arrival order (each label
        learned at its row's own clock plus the longest delay) the coverage
        reads other numbers."""
        df = frame()
        out = po.ModelBank([spec()]).fit_predict(df)
        pred = field(out, "pred_y")
        shifted = df.with_columns(avail=pl.col("t") + 30)
        other = fold(shifted, pred)["coverage_y"]
        got = field(out, "coverage_y")
        fin = np.isfinite(got) & np.isfinite(other)
        assert not np.array_equal(got[fin], other[fin])


class TestTheAutoStep:
    """`conformal_rate="auto"`: the step divided by `1 + 0.05 · g(α) · n̄ / c`,
    `n̄` the EW mean of the rows waiting at each release (task 342b)."""

    def test_the_interval_and_the_coverage_are_the_fold_to_the_bit(self):
        df = frame(n=2000, seed=1)
        out = po.ModelBank([spec(conformal_rate="auto")]).fit_predict(df)
        pred = field(out, "pred_y")
        want = fold(df, pred, auto=True)
        fixed = fold(df, pred)
        for key in ("sigma_y", "lo_y", "hi_y", "coverage_y"):
            got, exp = field(out, key), want[key]
            assert (np.isnan(got) == np.isnan(exp)).all(), key
            fin = np.isfinite(exp)
            assert fin.sum() > 1500, key
            np.testing.assert_array_equal(got[fin], exp[fin], err_msg=key)
        # The rows held move the step: the fixed fold is another interval.
        fin = np.isfinite(want["lo_y"])
        assert not np.array_equal(want["lo_y"][fin], fixed["lo_y"][fin])

    def test_the_gain_is_the_normal_density(self):
        """`g(α)` against scipy's normal: the density of `|Z|` at its `1 − α`
        quantile, within the inverse normal's own error."""
        from scipy.stats import norm

        for coverage in (0.5, 0.8, 0.9, 0.95, 0.99):
            alpha = 1.0 - coverage
            want = 2.0 * norm.pdf(norm.ppf(1.0 - alpha / 2.0))
            assert gain(alpha) == pytest.approx(want, rel=1e-8), coverage

    def test_the_spec_writes_it_and_refuses_another_word(self):
        assert spec(conformal_rate="auto")["conformal_rate"] == "auto"
        with pytest.raises(
            ValueError,
            match=r'^spec "m": conformal_rate must be a number or "auto", but it is "fast"\. '
            r"Use one of those\.$",
        ):
            spec(conformal_rate="fast")
        hand = {**spec(), "conformal_rate": "fast"}
        with pytest.raises(
            ValueError, match='conformal_rate must be a number or "auto", but it is "fast"'
        ):
            po.ModelBank([hand])
        with pytest.raises(ValueError, match="conformal_rate sets how fast the interval adapts"):
            po.spec.ewridge("m", targets=["y"], features=["x"], conformal_rate="auto")

    def test_the_cli_takes_it(self, tmp_path, online_cli):
        df = frame(n=300, seed=12)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        s = spec(conformal_rate="auto")
        want = po.ModelBank([s]).fit_predict(df).select("m").unnest("m")
        out = tmp_path / "cli.parquet"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
output = "{out.as_posix()}"
chunk_size = 100

[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
clock = "t"
half_life = {HALF_LIFE}
gap_cap = {GAP_CAP}
weight = "w"
min_weight = 0.0
max_error_inflation = inf
conformal = 0.9
conformal_rate = "auto"
emit_sigma = true
emit_metrics = true
label_clock = "avail"
[specs.model]
type = "ewridge"
""",
            encoding="utf-8",
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = pl.read_parquet(out).select("m").unnest("m")
        for key in ("lo_y", "hi_y", "coverage_y"):
            assert got[key].equals(want[key], null_equal=True), key


class TestTheSurfaces:
    def test_the_spec_writes_the_key_only_when_given(self):
        assert "label_clock" not in spec(label_clock=None)
        assert spec()["label_clock"] == "avail"
        assert spec(label_clock=("avail",))["label_clock"] == ["avail"]
        s = spec(max_held_rows=100, held_overflow="drop_oldest")
        assert (s["max_held_rows"], s["held_overflow"]) == (100, "drop_oldest")

    def test_the_lazy_plan_reads_the_label_column_and_agrees_with_the_bank(self):
        df = frame(n=300, seed=3)
        s = spec()
        bank = po.ModelBank([s]).fit_predict(df)
        # The plan selects only the output, so the label column is read
        # because the spec names it.
        lazy = df.lazy().online.fit_predict([s]).select("m").collect()
        assert lazy.unnest("m").equals(bank.select("m").unnest("m"), null_equal=True)

    def test_the_summary_counts_labels_missing_and_dropped(self):
        df = frame(n=400, seed=4)
        bank = po.ModelBank([spec(max_held_rows=3, held_overflow="drop_oldest")])
        bank.fit_predict(df)
        row = bank.summary().row(0, named=True)
        accepted = df.filter(pl.col("x").is_not_null())
        assert row["labels_missing"] == accepted["avail"].null_count()
        assert row["labels_dropped"] > 0

    def test_the_cli_takes_the_key(self, tmp_path, online_cli):
        df = frame(n=300, seed=12)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        s = spec()
        out = tmp_path / "out.parquet"
        run_online(online_cli, tmp_path, [s], input=src, output=out, chunk_size=64)

        def fields(frame_):
            return frame_.select("m").unnest("m").drop("coef", "support_coef")

        want = fields(po.ModelBank([s]).fit_predict(df))
        assert fields(pl.read_parquet(out)).equals(want, null_equal=True)
        cli_out = tmp_path / "cli.parquet"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
output = "{cli_out.as_posix()}"
chunk_size = 100

[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
clock = "t"
half_life = {HALF_LIFE}
gap_cap = {GAP_CAP}
weight = "w"
min_weight = 0.0
max_error_inflation = inf
conformal = 0.9
emit_sigma = true
emit_metrics = true
label_clock = "avail"
[specs.model]
type = "ewridge"
""",
            encoding="utf-8",
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        assert fields(pl.read_parquet(cli_out)).equals(want, null_equal=True)


class TestRefusals:
    def test_beside_embargo(self):
        with pytest.raises(ValueError, match="label_clock does not work beside embargo"):
            spec(embargo=5.0)

    def test_a_list_whose_entries_differ(self):
        with pytest.raises(
            ValueError,
            match='label_clock lists "a" and "b", but a spec reads one time a row at which its '
            "targets become known",
        ):
            po.spec.ewridge(
                "m",
                targets=["y", "x"],
                features=["x"],
                clock="t",
                gap_cap=5.0,
                half_life=5.0,
                label_clock=["a", "b"],
            )

    def test_holt(self):
        with pytest.raises(ValueError, match="label_clock does not work with holt"):
            po.spec.holt(
                "m", targets=["y"], clock="t", gap_cap=5.0, half_life=5.0, label_clock="avail"
            )

    def test_a_label_before_its_row(self):
        df = frame(n=50, seed=5).with_columns(avail=pl.col("t") - 1)
        bank = po.ModelBank([spec()])
        with pytest.raises(ValueError, match='label_clock column "avail" is 1 before clock column'):
            bank.fit_predict(df)

    def test_a_full_buffer(self):
        df = frame(n=200, seed=6)
        with pytest.raises(ValueError, match="above max_held_rows = 2"):
            po.ModelBank([spec(max_held_rows=2)]).fit_predict(df)

    def test_a_label_column_of_another_kind_than_the_clock(self):
        df = frame(n=50, seed=7).with_columns(avail=pl.col("avail").cast(pl.String))
        with pytest.raises(ValueError, match='label_clock column "avail" is String'):
            po.ModelBank([spec()]).fit_predict(df)
