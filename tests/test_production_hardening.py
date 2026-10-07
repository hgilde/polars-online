"""Production-hardening round: the checks other libraries run that we did not.

Calibrated against river 0.26.1's own `river.checks` battery (37 checks — the
relevant ones here are `check_shuffle_features_no_impact`, `check_pickling`,
`check_predict_one_before_any_learn`, `check_no_state_aliasing_with_input`),
sklearn's estimator checks (sample-weight and column-order invariances), and
statsmodels' convention of oracle comparisons — which this suite already has
in `tests/reference.py` and the KKT tests.

This round found and fixed two real defects before writing a single test:

* **a target listed as its own feature was accepted**, producing
  corr(pred, y) = 1.0 — perfect leakage through the door hard rule 2 does not
  guard, and exactly the accident a long feature list invites;
* **duplicate feature names were accepted**, silently splitting the
  coefficient across identical slots on an exactly singular system.

Both are now spec-validation errors, pinned below.
"""

import ast
import builtins
import contextlib
import copy
import itertools
import os
import pickle
import re
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_version import needs_polars
from test_ffi_memory import run_isolated

REPO = Path(__file__).resolve().parent.parent


def _df(n=2000, k=2, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.standard_normal(n) for i in range(k)}
    beta = np.arange(1, k + 1, dtype=float)
    cols["y0"] = sum(beta[i] * cols[f"x{i}"] for i in range(k)) + 0.05 * rng.standard_normal(n)
    return pl.DataFrame(cols)


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=200.0,
        min_weight=5.0,
        max_rows_between_solves=1,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


class TestLeakageAndDuplicateRejection:
    """The two defects this review found, pinned."""

    def test_target_as_feature_is_rejected(self):
        with pytest.raises(ValueError, match="both a target and a feature"):
            _spec(features=["x0", "y0"])

    def test_the_rejection_message_offers_the_right_fix(self):
        with pytest.raises(ValueError, match="lagged copy"):
            _spec(features=["y0"])

    def test_duplicate_features_are_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            _spec(features=["x0", "x1", "x0"])

    def test_duplicate_targets_are_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            _spec(targets=["y0", "y0"])

    def test_every_model_rejects_the_leak(self):
        for name, extra in [
            ("rls", {}),
            ("kalman", {"coef_half_life": 100.0}),
            ("lasso", {"lasso_path": [0.0]}),
            ("sgd", {"learning_rate": 0.01}),
            ("ftrl", {}),
        ]:
            with pytest.raises(ValueError, match="both a target and a feature"):
                getattr(po.spec, name)(
                    "m", targets=["y0"], features=["y0"], half_life=100.0, **extra
                )


class TestFeatureOrderInvariance:
    """river's `check_shuffle_features_no_impact`, which named features make a
    real promise here: neither the order of the `features` list nor the order
    of the DataFrame's columns may change a number."""

    def test_spec_feature_order(self):
        df = _df(k=4)
        a = po.ModelBank([_spec(features=["x0", "x1", "x2", "x3"])]).fit_predict(df)
        b = po.ModelBank([_spec(features=["x3", "x1", "x0", "x2"])]).fit_predict(df)
        pa_ = a["m"].struct.field("pred_y0").to_list()
        pb = b["m"].struct.field("pred_y0").to_list()
        for i, (u, v) in enumerate(zip(pa_, pb, strict=True)):
            if u is None or v is None:
                assert u == v, f"row {i}"
            else:
                assert u == pytest.approx(v, rel=1e-12), f"row {i}"

    def test_dataframe_column_order_and_extra_columns(self):
        """Extraction is by name: shuffling the frame's columns and adding
        unrelated ones must be invisible (river's emerging-features analogue —
        ours is 'extra columns are ignored', by design)."""
        df = _df(k=2)
        spec = _spec()
        a = po.ModelBank([spec]).fit_predict(df).select("m")
        shuffled = df.select(["y0", "x1", "x0"]).with_columns(
            junk=pl.lit("z"), extra=pl.int_range(0, df.height)
        )
        b = po.ModelBank([spec]).fit_predict(shuffled).select("m")
        assert a.equals(b, null_equal=True)


class TestDegenerateColumns:
    """Whole-column pathologies, not just scattered nulls."""

    def test_all_null_feature_skips_every_row(self):
        df = _df().with_columns(x0=pl.lit(None, dtype=pl.Float64))
        out = po.ModelBank([_spec()]).fit_predict(df)
        for f in out.schema["m"].fields:
            vals = out["m"].struct.field(f.name).to_list()
            assert all(v is None for v in vals), f"{f.name} produced a value from no data"

    def test_all_null_target_is_predict_only_forever(self):
        """`weight_sum` counts accepted rows, and a null target does not reject a
        row -- so its whole trajectory must be *identical* to the same frame
        with targets present."""
        df = _df()
        with_targets = po.ModelBank([_spec()]).fit_predict(df)
        out = po.ModelBank([_spec()]).fit_predict(
            df.with_columns(y0=pl.lit(None, dtype=pl.Float64))
        )
        assert (
            out["m"].struct.field("weight_sum").to_list()
            == with_targets["m"].struct.field("weight_sum").to_list()
        )
        assert all(v is None for v in out["m"].struct.field("resid_y0").to_list())

    def test_all_zero_weight_never_learns_and_never_breaks(self):
        df = _df().with_columns(w=pl.lit(0.0))
        out = po.ModelBank([_spec(weight="w")]).fit_predict(df)
        preds = out["m"].struct.field("pred_y0").to_list()
        assert all(v is None for v in preds), "nothing carried weight, nothing may predict"

    def test_constant_target(self):
        df = _df().with_columns(y0=pl.lit(7.0))
        out = po.ModelBank([_spec()]).fit_predict(df)
        preds = [v for v in out["m"].struct.field("pred_y0").to_list() if v is not None]
        assert preds, "a constant target is perfectly learnable"
        assert preds[-1] == pytest.approx(7.0, abs=1e-6)


class TestFirstRowPathologies:
    """The first row seeds means, scales and Holt's level; an outlier there is
    the worst-placed outlier a stream can have."""

    @pytest.mark.parametrize("standardize", [False, True])
    def test_e12_outlier_first_row_washes_out(self, standardize):
        """A 1e12 first row injects ~1e24 into the second moments, so washout
        needs ~80 half-lives -- inherent to EW accumulators, not a defect. 4000
        rows at half-life 40 is 100 half-lives: recovery must be complete."""
        df = _df(n=4000)
        df = pl.concat([pl.DataFrame({"x0": [1e12], "x1": [-1e12], "y0": [1e12]}), df])
        out = po.ModelBank([_spec(standardize=standardize, half_life=40.0)]).fit_predict(df)
        coef = out["m"].struct.field("coef").to_list()[-1]
        assert coef[1] == pytest.approx(1.0, abs=0.05), f"slope x0: {coef[1]}"
        assert coef[2] == pytest.approx(2.0, abs=0.05), f"slope x1: {coef[2]}"

    def test_holt_seeded_by_an_outlier_recovers(self):
        n = 4000
        y = np.concatenate([[1e9], 3.0 + 0.5 * np.arange(n)])
        df = pl.DataFrame({"y0": y})
        # 80 half-lives of washout for the poisoned trend (the first update
        # sees a slope of -1e9).
        spec = po.spec.holt(
            "m", targets=["y0"], half_life=50.0, trend_half_life=50.0, min_weight=3.0
        )
        out = po.ModelBank([spec]).fit_predict(df)
        level, trend = out["m"].struct.field("coef").to_list()[-1]
        assert trend == pytest.approx(0.5, abs=0.05)
        assert level == pytest.approx(3.0 + 0.5 * (n - 1), rel=0.01)

    def test_leading_nulls_then_data(self):
        df = _df(n=1000)
        nulls = pl.DataFrame(
            {
                "x0": [None] * 50,
                "x1": [None] * 50,
                "y0": [None] * 50,
            },
            schema={"x0": pl.Float64, "x1": pl.Float64, "y0": pl.Float64},
        )
        out = po.ModelBank([_spec()]).fit_predict(pl.concat([nulls, df]))
        coef = out["m"].struct.field("coef").to_list()[-1]
        assert coef[1] == pytest.approx(1.0, abs=0.05)


class TestParameterExtremes:
    """Every model at parameter values from the edges of its documented range:
    the assertion is only 'finite-or-null and does not panic', which is what a
    production stream needs at 3 a.m."""

    CASES = [
        ("ewridge", {"ridge": [1e-15]}),
        ("ewridge", {"ridge": [1e15]}),
        ("kalman", {"coef_half_life": 100.0, "p0": 1e-12}),
        ("kalman", {"coef_half_life": 100.0, "p0": 1e12}),
        ("quantile", {"quantile": 0.01}),
        ("quantile", {"quantile": 0.99}),
        ("huber", {"huber_delta": 1e-6}),
        ("huber", {"huber_delta": 1e6}),
        ("pa", {"c": 1e-9}),
        ("pa", {"c": 1e9}),
        ("sgd", {"learning_rate": 1e-9}),
        ("sgd", {"learning_rate": 0.5, "clip_gradient": float("inf")}),
        ("ftrl", {"alpha": 1e-6, "l1": 100.0}),
        ("lasso", {"lasso_path": [1e6, 0.0]}),
    ]

    @pytest.mark.parametrize(
        ("model", "extra"), CASES, ids=[f"{m}-{i}" for i, (m, _) in enumerate(CASES)]
    )
    def test_extreme_parameters_never_produce_nonfinite(self, model, extra):
        df = _df(n=1500)
        kw = dict(targets=["y0"], features=["x0", "x1"], half_life=100.0, min_weight=5.0)
        if model not in ("rls", "kalman", "ftrl", "sgd", "pa"):
            kw["max_rows_between_solves"] = 8
        kw.update(extra)
        out = po.ModelBank([getattr(po.spec, model)("m", **kw)]).fit_predict(df)
        for f in out.schema["m"].fields:
            # `coef`/`support_coef` are lists; `withheld_reason` is an enum.
            if f.name.startswith("coef") or not f.dtype.is_float():
                continue
            for i, v in enumerate(out["m"].struct.field(f.name).to_list()):
                assert v is None or np.isfinite(v), f"{f.name}[{i}] = {v}"

    def test_lam_is_halflife_by_another_name(self):
        df = _df()
        h = 137.0
        a = po.ModelBank([_spec(half_life=h)]).fit_predict(df)
        b = po.ModelBank([_spec(half_life=None, lam=0.5 ** (1.0 / h))]).fit_predict(df)
        pa_ = a["m"].struct.field("pred_y0").to_numpy().astype(float)
        pb = b["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(pa_)
        np.testing.assert_allclose(pa_[m], pb[m], rtol=1e-12)

    def test_pure_ridge_lasso_limit_matches_ewridge(self):
        """`l1_ratio = 0` turns the coordinate descent into pure ridge; at a
        negligible penalty both it and ewridge must land on OLS."""
        df = _df(n=3000)
        lasso = po.spec.lasso(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            lasso_path=[1e-10],
            l1_ratio=0.0,
            half_life=1e9,
            min_weight=5.0,
            max_rows_between_solves=1,
        )
        a = po.ModelBank([lasso]).fit_predict(df)
        b = po.ModelBank([_spec(half_life=1e9, ridge=[1e-10])]).fit_predict(df)
        ca = a["m"].struct.field("coef").to_list()[-1][:3]
        cb = b["m"].struct.field("coef").to_list()[-1]
        np.testing.assert_allclose(ca, cb, rtol=1e-6)


#: Run by `test_an_interrupted_save_keeps_the_last_good_state` in a fresh
#: interpreter. `STATE_PATH` is substituted; the soft file-size limit makes
#: the second save fail partway through, the way a full disk would.
SAVE_INTERRUPTED = """
import os, resource, signal
import numpy as np, polars as pl, polars_online as po

path = STATE_PATH
rng = np.random.default_rng(0)
df = pl.DataFrame({
    "x0": rng.standard_normal(2000),
    "y0": rng.standard_normal(2000),
    "g": [f"g{i % 200}" for i in range(2000)],
})
bank = po.ModelBank([po.spec.ewridge("m", targets=["y0"], features=["x0"],
                                     half_life=50.0, group="g")])
bank.fit_predict(df)
bank.save(path)
good = open(path, "rb").read()

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # EFBIG instead of a kill
resource.setrlimit(resource.RLIMIT_FSIZE, (len(good) // 3, resource.RLIM_INFINITY))
bank.fit_predict(df)
try:
    bank.save(path)
except Exception as e:
    assert path in str(e), f"the error does not name the file: {e}"
else:
    raise AssertionError("the save reported success under a file-size limit")
resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

assert open(path, "rb").read() == good, "the failed save damaged the state"
po.ModelBank.load(path)          # and it still resumes
left = [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]
assert not left, f"temporary left behind: {left}"
print("ok")
"""


class TestScoringWithoutLearning:
    """The deployment path: load a fit and score new rows. The README says to
    do it with weight 0, and says what a null target does instead; both are
    pinned here (IMPROVEMENTS U8, docs/ENHANCEMENTS.md E31, docs/PLAN.md task
    81)."""

    def _fitted(self, df, **kw):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            gap_cap=5.0,
            half_life=20.0,
            min_weight=3.0,
            weight="w",
            max_rows_between_solves=1,
            coef_every=0,
            **kw,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        return bank

    def _frame(self, n, seed=0, weight=1.0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        return pl.DataFrame(
            {
                "t": np.arange(float(n)),
                "x0": x,
                "y": 1.0 + 2.0 * x + 0.01 * rng.standard_normal(n),
                "w": np.full(n, weight),
            }
        )

    def test_zero_weight_rows_score_and_freeze_the_fit_exactly(self):
        """Mean-form accumulators decayed with nothing added are themselves, so
        the coefficients do not move by one bit while scoring."""
        fit = self._frame(100)
        bank = self._fitted(fit)
        before = bank.fit_predict(fit.tail(1))["m"].struct.field("coef").to_list()[-1]

        scored = self._frame(60, seed=1, weight=0.0).with_columns(pl.col("t") + 100.0)
        out = po.ModelBank.load_bytes(bank.save_bytes()).fit_predict(scored)["m"].struct
        after = out.field("coef").to_list()[-1]
        assert after == before, "scoring moved the coefficients"
        assert out.field("pred_y")[0] is not None, "scoring emitted nothing"

    def _score_null_targets(self, **kw):
        """The coefficients before and after 60 rows whose target is null,
        and the `weight_sum` those rows report."""
        fit = self._frame(100)
        bank = self._fitted(fit, **kw)
        before = bank.fit_predict(fit.tail(1))["m"].struct.field("coef").to_list()[-1]
        scored = self._frame(60, seed=1).with_columns(
            pl.col("t") + 100.0, pl.lit(None, dtype=pl.Float64).alias("y")
        )
        out = po.ModelBank.load_bytes(bank.save_bytes()).fit_predict(scored)["m"].struct
        return before, out.field("coef").to_list()[-1], out.field("weight_sum").to_list()

    def test_a_null_target_leaves_its_fit_where_it_was(self):
        """Under `target_gaps="own_rows"`, the default, a target's Gram is over
        the rows it is present on, so a row without it ages the Gram and the
        cross-moments alike, and mean-form accumulators aged with nothing
        added are themselves: the coefficients do not move by one bit, as
        with weight 0. What differs from weight 0 is `weight_sum`, which counts
        every row and so keeps counting (docs/PLAN.md task 81)."""
        before, after, weight_sum = self._score_null_targets()
        assert after == before, "a row without the target moved its fit"
        # Weight 0 decays `weight_sum` toward nothing; here it holds at the steady
        # state of a half-life of 20 rows, `1 / (1 - lam)`.
        steady = 1.0 / (1.0 - 0.5 ** (1.0 / 20.0))
        assert weight_sum[-1] == pytest.approx(steady, rel=1e-2), (
            "every row counts toward weight_sum"
        )

    def test_under_pairwise_a_null_target_still_moves_the_fit(self):
        """Under `"pairwise"` the one Gram is over every row, so a row without
        the target still moves the feature moments while the target's
        cross-moments stand still: the two halves of the fit end up over
        different rows, and it wanders with the features. That is why the
        README says to score with weight 0 there."""
        before, after, _ = self._score_null_targets(target_gaps="pairwise")
        assert after != before, "if this ever stops drifting, the README can say so"

    def test_a_long_scoring_tail_decays_n_eff_under_min_periods(self):
        """The documented cost of scoring with weight 0: the clock still
        advances, so `weight_sum` decays and eventually `min_weight` blanks the
        output even though the fit behind it is unchanged."""
        bank = self._fitted(self._frame(100))
        scored = self._frame(200, seed=2, weight=0.0).with_columns(pl.col("t") + 100.0)
        out = po.ModelBank.load_bytes(bank.save_bytes()).fit_predict(scored)["m"].struct
        weight_sum = out.field("weight_sum").to_list()
        assert weight_sum[0] > 3.0 > weight_sum[-1], (weight_sum[0], weight_sum[-1])
        preds = out.field("pred_y").to_list()
        assert preds[0] is not None and preds[-1] is None
        # ... and it is only the gate: the fit is still there underneath.
        coefs = [c for c in out.field("coef").to_list() if c is not None]
        assert coefs[0] == coefs[-1]


class TestSerializationRobustness:
    """State bytes are the resume path; a corrupt file must be an error, never
    a panic or worse. (`SECURITY.md` already says to treat state like pickle;
    this is the test that a *damaged own* file still fails cleanly.)"""

    def _bytes(self):
        b = po.ModelBank([_spec()])
        b.fit_predict(_df(n=500))
        return b.save_bytes()

    def test_corruption_never_crashes_and_the_header_always_detects(self):
        """Two different guarantees. A flip in the *structure* (magic,
        versions, layout -- the first bytes) must raise. A flip in the
        *payload* is often just a different float, which msgpack cannot know
        is wrong -- there the requirement is a clean outcome, never a panic."""
        blob = bytearray(self._bytes())
        rng = np.random.default_rng(1)
        # The one region that MUST always detect: the magic string. (The
        # first-64-bytes region also holds `package_version`, which is
        # deliberately informational, so it is not a valid target.)
        magic = blob.find(b"polars-online-bank")
        assert magic >= 0
        for i in range(magic, magic + len(b"polars-online-bank")):
            old = blob[i]
            new = int(rng.integers(0, 256))
            blob[i] = (new + 1) % 256 if new == old else new
            with pytest.raises(Exception, match="."):
                po.ModelBank.load_bytes(bytes(blob))
            blob[i] = old
        detected = 0
        probe = _df(n=3)
        for _ in range(300):  # anywhere at all: clean outcome only
            i = int(rng.integers(0, len(blob)))
            old = blob[i]
            blob[i] = int(rng.integers(0, 256))
            try:
                loaded = po.ModelBank.load_bytes(bytes(blob))
            except Exception:
                detected += 1
            else:
                # A state that loaded must also learn without a panic: a
                # flipped length in a model's vectors loaded and then
                # indexed out of bounds on the first row (review
                # 2026-09-18, B3). A ValueError is a clean outcome; a
                # `PanicException` is not, and is not an `Exception`.
                with contextlib.suppress(ValueError):
                    loaded.fit_predict(probe)
            finally:
                blob[i] = old
        assert detected > 0, "not a single payload corruption was detected"

    def test_random_truncation_errors_cleanly(self):
        blob = self._bytes()
        rng = np.random.default_rng(2)
        for _ in range(50):
            cut = int(rng.integers(0, len(blob)))
            with pytest.raises(Exception, match="."):
                po.ModelBank.load_bytes(blob[:cut])

    @pytest.mark.skipif(os.name != "posix", reason="RLIMIT_FSIZE is POSIX")
    def test_an_interrupted_save_keeps_the_last_good_state(self, tmp_path):
        """`save` used to be `fs::write`, which truncates the destination and
        then writes into it: a kill, a full disk or a quota left a truncated
        file *and* took the last good state with it, so a resume loop started
        the stream over (IMPROVEMENTS C6). RLIMIT_FSIZE reproduces exactly
        that, deterministically. In a subprocess, because the limit is
        process-wide and would otherwise apply to pytest's own writing."""
        path = tmp_path / "state.msgpack"
        r = run_isolated(SAVE_INTERRUPTED.replace("STATE_PATH", repr(str(path))))
        assert "ok" in r.stdout

    def test_pickle_and_deepcopy_resume_exactly(self):
        df = _df(n=1000)
        b = po.ModelBank([_spec()])
        b.fit_predict(df.slice(0, 500))
        clones = [pickle.loads(pickle.dumps(b)), copy.deepcopy(b)]
        rest = df.slice(500, 500)
        want = b.fit_predict(rest)
        for c in clones:
            assert want.equals(c.fit_predict(rest), null_equal=True)


class TestConcurrency:
    """Two threads on one bank is a user error (chunks must arrive in stream
    order), but it must be a *safe* error: either serialized or refused, never
    interpreter corruption, and the bank must still work afterwards."""

    def test_concurrent_fit_predict_is_safe(self):
        df = _df(n=5000)
        bank = po.ModelBank([_spec()])
        errors: list[BaseException] = []

        def work():
            try:
                for i in range(0, 5000, 500):
                    bank.fit_predict(df.slice(i, 500))
            except BaseException as e:  # noqa: BLE001 — recording, not hiding
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Whatever happened above, the object must not be wedged.
        fresh = po.ModelBank([_spec()])
        again = fresh.fit_predict(df)
        assert again.height == df.height
        for e in errors:
            assert isinstance(e, Exception), f"non-Exception escaped: {e!r}"


class TestOddNames:
    """Column names are interpolated into struct field names; they must pass
    through untouched, not be parsed."""

    def test_unicode_and_spacey_names_roundtrip(self, tmp_path):
        rng = np.random.default_rng(3)
        df = pl.DataFrame(
            {
                "价格 Δ": rng.standard_normal(500),
                "my target": rng.standard_normal(500),
                "g": ["α", "β"] * 250,
            }
        ).with_columns((pl.col("价格 Δ") * 2).alias("my target"))
        spec = po.spec.ewridge(
            "m",
            targets=["my target"],
            features=["价格 Δ"],
            group="g",
            half_life=100.0,
            min_weight=5.0,
            max_rows_between_solves=1,
        )
        bank = po.ModelBank([spec])
        out = bank.fit_predict(df)
        names = [f.name for f in out.schema["m"].fields]
        assert "pred_my target" in names, names
        p = tmp_path / "s.state"
        bank.save(p)
        b2 = po.ModelBank.load(p, specs=[spec])
        assert bank.fit_predict(df).equals(b2.fit_predict(df), null_equal=True)

    def test_missing_column_names_the_column(self):
        with pytest.raises(Exception, match="x9"):
            po.ModelBank([_spec(features=["x0", "x9"])]).fit_predict(_df())


def _doc_blocks(rel: str, fence: str = "python") -> list[tuple[str, int, str]]:
    """Every ```<fence> block in one document, with the line it starts on."""
    out: list[tuple[str, int, str]] = []
    in_block, buf, start = False, [], 0
    text = (REPO / rel).read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(f"```{fence}"):
            in_block, buf, start = True, [], i
        elif line.strip() == "```" and in_block:
            in_block = False
            out.append((rel, start, "\n".join(buf)))
        elif in_block:
            buf.append(line)
    return out


# The README, and the runner guide its file-to-file examples live in (docs/PLAN.md
# task 68): the fixture below already writes the files those examples read.
README_BLOCKS = _doc_blocks("README.md") + _doc_blocks("docs/RUNNER.md")


def _block_param(path: str, line: int, code: str):
    """One block as a test case. A block that streams a DuckDB, ADBC or
    pyarrow source reads it with `pl.scan_arrow_c_stream`, and so needs the
    py-polars the README names for it."""
    marks = []
    if "scan_arrow_c_stream" in code:
        marks.append(
            needs_polars("1.43.0", "pl.scan_arrow_c_stream, which py-polars added in 1.43.0")
        )
    return pytest.param(path, line, code, id=f"{path}:L{line}", marks=marks)


def _names(code: str) -> tuple[set[str], list[str]]:
    """The names a block builds -- assigns, imports, defines or takes as an
    argument -- and the names it reads, in order."""
    stored: set[str] = set()
    loaded: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name):
            (loaded.append if isinstance(node.ctx, ast.Load) else stored.add)(node.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            stored.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            stored.add(node.name)
        elif isinstance(node, ast.arg):
            stored.add(node.arg)
    return stored, loaded


#: The README's own python blocks, in order: the README's definitions.
README_ONLY = [b for b in README_BLOCKS if b[0] == "README.md"]


def _built_by_the_readme(code: str, line: int, ns: dict, memo: dict) -> dict:
    """``ns``, with each name ``code`` reads and does not build itself
    replaced by the README's own: the value the README's latest block before
    ``line`` that assigns it builds, that block run the same way, in a copy
    of ``ns``. So a block runs on what the README built, not on the
    fixture's copy, which can differ: the fixture's ``grid`` had a
    ``min_weight`` the README's has not, and the README defines ``spec``
    twice (review 2026-10-05, TC5). ``memo`` keeps each (name, block) built
    once per test."""
    stored, loaded = _names(code)
    out = dict(ns)
    for name in dict.fromkeys(loaded):
        if name in stored:
            continue
        defining = [(ln, c) for _, ln, c in README_ONLY if ln < line and name in _names(c)[0]]
        if not defining:
            continue
        ln, c = defining[-1]
        if (name, ln) not in memo:
            local = _built_by_the_readme(c, ln, ns, memo)
            exec(compile(c, f"README.md:{ln}", "exec"), local)
            # A comprehension's or a lambda's variable is not the block's.
            memo[name, ln] = local.get(name, ns.get(name))
        if memo[name, ln] is not None:
            out[name] = memo[name, ln]
    return out


def _example_data_block() -> str:
    """The README's *Example data* block: the frames every example may read."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index("\n### Example data\n")
    section = text[start : text.index("\n## ", start)]
    blocks = [code for _, _, code in README_BLOCKS if code in section]
    assert len(blocks) == 1, f"Example data should hold one python block, not {len(blocks)}"
    return blocks[0]


#: The runner guide's shell blocks, run against the built `online` binary.
#: Only this document's: the README's ```sh blocks are `pip install`, `uv sync`
#: and the development commands, which must never run from a test.
SHELL_BLOCKS = _doc_blocks("docs/RUNNER.md", "sh")


def _rst_python_blocks(where: str, doc: str) -> list[tuple[str, int, str]]:
    """Every ``.. code-block:: python`` in one docstring (dedented, as
    `inspect.getdoc` gives it), with the line it starts on."""
    out: list[tuple[str, int, str]] = []
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == ".. code-block:: python":
            indent = len(line) - len(line.lstrip())
            j, buf = i + 1, []
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip() == "":
                    buf.append("")
                elif len(nxt) - len(nxt.lstrip()) > indent:
                    buf.append(nxt)
                else:
                    break
                j += 1
            body = "\n".join(buf).strip("\n")
            pad = min(len(b) - len(b.lstrip()) for b in body.splitlines() if b.strip())
            out.append((where, i + 1, "\n".join(b[pad:] for b in body.splitlines())))
            i = j
        else:
            i += 1
    return out


def _docstring_blocks() -> list[tuple[str, int, str]]:
    """Every ``.. code-block:: python`` in the docstrings the API reference is
    built from: each module in the reference, what its ``__all__`` exports,
    and the public methods of an exported class. A docstring example is a
    claim the reader trusts as much as a README block, so it runs the same
    way (docs/WRITING.md, "Every code block runs")."""
    import inspect

    modules = [po, po.spec, po._bank, po._frame]
    modules += [po.gram, po.eval, po.corr, po.stream, po.sim]
    seen: set[int] = set()
    out: list[tuple[str, int, str]] = []
    for mod in modules:
        objs = [(mod.__name__, mod)]
        for name in getattr(mod, "__all__", ()):
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            objs.append((f"{mod.__name__}.{name}", obj))
            if inspect.isclass(obj):
                for mname, member in vars(obj).items():
                    if not mname.startswith("_") and callable(member):
                        objs.append((f"{mod.__name__}.{name}.{mname}", member))
        for where, obj in objs:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            doc = inspect.getdoc(obj)
            if doc:
                out.extend(_rst_python_blocks(where, doc))
    return out


DOCSTRING_BLOCKS = _docstring_blocks()


def _closed_rows(df: pl.DataFrame) -> pl.DataFrame:
    """What `bank.closed_groups()` gives for the README's block example."""
    spec = po.spec.ew_cov(
        "cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone"
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df.with_columns(block=pl.int_range(pl.len()) // 100))
    return bank.closed_groups()


def _trades(n: int = 3000) -> pl.DataFrame:
    """Quotes, each with a `mid`, and trades between them -- three rows in ten
    -- with a `side`, a `quantity` and a `price`, null on the quotes; two
    symbols, about two rows a second on the clock `ts`."""
    rng = np.random.default_rng(7)
    trade = rng.random(n) < 0.3
    start = datetime(2024, 1, 2, 9, 30)
    ts = [start + timedelta(microseconds=int(u)) for u in np.cumsum(rng.exponential(5e5, n))]
    mid = 100 + np.cumsum(rng.normal(0.0, 0.01, n))
    return pl.DataFrame(
        {
            "ts": ts,
            "symbol": rng.choice(["AAA", "BBB"], n),
            "mid": mid,
            "side": [
                s if k else None for s, k in zip(rng.choice(["buy", "sell"], n), trade, strict=True)
            ],
            "quantity": [
                float(q) if k else None for q, k in zip(rng.integers(1, 10, n), trade, strict=True)
            ],
            "price": [
                float(m + e) if k else None
                for m, e, k in zip(mid, rng.normal(0.0, 0.02, n), trade, strict=True)
            ],
        }
    )


def _readme_namespace(tmp_path: Path) -> dict[str, object]:
    """What the README's examples have built by the time a block runs: the
    frames *Example data* builds, a spec with the grid its field-name
    examples assume, a fed bank, an output frame, and the files the doc
    examples read. A block that needs something not here fails with a
    `NameError`. That finds a name missing from this namespace, not one the
    README never built: every block here gets `df` ready-made, so a README
    that only described `df` passed for four rewrites (README-ITERATIONS,
    C11). `test_every_name_a_readme_example_reads_was_built_by_an_earlier_one`
    checks the README itself."""
    n = 400
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "t": np.arange(float(n)),
            # A timestamp, a minute apart: the README's clock="ts" specs give durations.
            "ts": pl.datetime_range(
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 30) + timedelta(minutes=n - 1),
                "1m",
                eager=True,
            ),
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "signal_a": rng.standard_normal(n),
            "signal_b": rng.standard_normal(n),
            "y": rng.standard_normal(n),
            "ret": rng.standard_normal(n),
            "stock_id": [f"b{i % 4}" for i in range(n)],
            "group": [f"g{i % 3}" for i in range(n)],
            "session": ["m"] * (n // 2) + ["a"] * (n - n // 2),
            "venue": ["X", "Y"] * (n // 2),
        }
    )
    common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0)
    # `spec` as section 2 shows it, and `grid` as the "output field names"
    # section introduces it (ridge 0.5, half-life 500 are what its blocks filter on).
    spec = po.spec.ewridge(
        "ridge",
        targets=["y"],
        features=["x0", "x1", "x2"],
        clock="t",
        half_life=600.0,
        gap_cap=300.0,
        group="stock_id",
        ridge=[1e-6, 0.1],
        standardize=True,
    )
    grid = po.spec.ewridge("m", half_life=[100.0, 500.0], ridge=[1e-6, 0.5], **common)
    scored = po.ModelBank(
        [
            po.spec.ewridge("ridge", half_life=500.0, group="stock_id", **common),
            po.spec.kalman(
                "kalman", half_life=500.0, coef_half_life=100.0, group="stock_id", **common
            ),
        ]
    ).fit_predict(df)
    graded = po.ModelBank([grid]).fit_predict(df)
    out = df.hstack(scored.select("ridge", "kalman")).hstack(graded.select("m"))
    # The rows after `df`'s, as "today" is after the stream a bank has
    # learned: the same rows again would step every group's clock back,
    # which the default refuses (task 120).
    today = df.with_columns(pl.col("t") - df["t"].min() + df["t"].max() + 1.0)
    df.write_parquet(tmp_path / "ticks.parquet")
    today.write_parquet(tmp_path / "today.parquet")
    _trades().write_parquet(tmp_path / "trades.parquet")
    (tmp_path / "ticks").mkdir()  # the `ticks/*.parquet` glob: a stream in two files
    df[:200].write_parquet(tmp_path / "ticks" / "part-0.parquet")
    df[200:].write_parquet(tmp_path / "ticks" / "part-1.parquet")
    today.write_csv(tmp_path / "today.csv")
    # `--input-format ipc` names a file whose extension does not say so.
    df.write_ipc(tmp_path / "feed.dat")
    fed = po.ModelBank([spec])
    fed.fit_predict(df)
    fed.save(tmp_path / "bank.state")  # what section 2's `bank.save` left behind
    cli_spec = po.spec.ewridge(
        "ridge", targets=["y"], features=["x0"], half_life=500.0, min_weight=5.0
    )
    (tmp_path / "bank.toml").write_text(
        'input = "ticks.parquet"\noutput = "fitted.parquet"\n\n'
        '[[specs]]\nname = "ridge"\ntargets = ["y"]\nfeatures = ["x0"]\n'
        'half_life = 500.0\nmin_weight = 5.0\n[specs.model]\ntype = "ewridge"\n',
        encoding="utf-8",
    )
    # The guide's `--load-state` examples need a state *its own config's* spec
    # saved: a state refuses to load under specs it was not saved from, which
    # is the point of that check. `bank.state` belongs to the README, which
    # writes and reads it throughout, so the CLI's gets a name of its own.
    cli_bank = po.ModelBank([cli_spec])
    cli_bank.fit(df.lazy())
    cli_bank.save(tmp_path / "run.state")
    # The sidecar example needs a spec that closes groups; the config above
    # has none, and the CLI refuses `--closed-groups` without one.
    (tmp_path / "blocks.toml").write_text(
        'input = "ticks.parquet"\noutput = "fitted.parquet"\n\n'
        '[[specs]]\nname = "cov"\nfeatures = ["x0", "x1"]\n'
        'half_life = 500.0\ngroup = "stock_id"\ngroup_close = "session"\n'
        'session = "session"\n[specs.model]\ntype = "ew_cov"\n',
        encoding="utf-8",
    )
    return {
        "pl": pl,
        "np": np,
        "po": po,
        "df": df,
        # The "one row per finished group" section introduces both, and the
        # run block after it reuses them.
        "blocks": po.spec.ew_cov(
            "cov",
            features=["x0", "x1"],
            lam=1.0,
            group="block",
            group_close="monotone",
        ),
        "by_block": df.with_columns(block=pl.int_range(pl.len()) // 100),
        # The "one row per finished group" block's output, which the
        # "reading a correlation matrix" section reads.
        "closed": _closed_rows(df),
        # The "series that tick at their own times" section's eight ticks, which
        # its first block builds and its two-run block reads.
        "ticks": pl.DataFrame(
            {
                "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],
                "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],
                "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],
            }
        ),
        # The "windowed means" section's stream: quotes with a mid, and
        # trades between them, for two symbols on the clock `ts`.
        "trades": _trades(),
        # The "relative and look-ahead targets" section builds it in its first
        # block, and its `with_columns` return block reuses it.
        "flows": pl.scan_parquet(tmp_path / "trades.parquet").with_columns(
            flow=pl.when(pl.col("side") == "buy")
            .then(pl.col("quantity"))
            .otherwise(-pl.col("quantity"))
        ),
        "today": today,
        # The query forms, as *Example data* makes them: each file read back.
        "later": pl.scan_parquet(tmp_path / "today.parquet"),
        "lf": pl.scan_parquet(tmp_path / "ticks.parquet"),
        "spec": spec,
        "grid": grid,
        "bank": po.ModelBank([spec]),
        "out": out,
        "now": df["t"].max(),
    }


class TestReadmeExamples:
    """The README's python blocks are the first code anyone runs, so they are
    run here -- not just compiled, which is all this checked before and which
    let `po.spec.holt(..., level_half_life=200.0)` sit in the README raising
    "one of half-life/lam is required" (IMPROVEMENTS U6).

    Each block runs in its own copy of a namespace holding what the prose has
    already introduced, so blocks do not depend on each other's order or
    leftovers."""

    def test_there_are_blocks_to_check(self):
        assert len(README_BLOCKS) >= 8, README_BLOCKS

    @pytest.mark.parametrize(
        ("path", "line", "code"), [_block_param(*block) for block in README_BLOCKS]
    )
    def test_a_readme_block_runs(self, path, line, code, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # the doc examples write next to the inputs
        # The parallelism blocks set thread-count variables; both pools are
        # long built in this process, so they change nothing here, but the
        # subprocess tests inherit the environment, so put it back.
        env = dict(os.environ)
        ns = _readme_namespace(tmp_path)
        try:
            if path == "README.md":
                ns = _built_by_the_readme(code, line, ns, {})
            exec(compile(code, f"{path}:{line}", "exec"), ns)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_the_readme_builds_the_frames_its_examples_read(self, tmp_path, monkeypatch):
        """*Example data* shows the code that builds `df`, `trades` and
        `today`, and writes the files some examples scan, so a reader can run
        any example. It must build exactly the frames the namespace above
        gives every block, or the README would show one stream and test its
        examples on another."""
        ns = _readme_namespace(tmp_path)
        # The block runs where the fixture wrote nothing, so the files read
        # back below are the ones it writes.
        readme = tmp_path / "readme"
        readme.mkdir()
        monkeypatch.chdir(readme)
        shown: dict[str, object] = {}
        exec(compile(_example_data_block(), "README.md: Example data", "exec"), shown)
        for name in ("df", "trades", "today"):
            assert shown[name].equals(ns[name]), name
        assert shown["lf"].collect().equals(ns["df"])
        assert shown["later"].collect().equals(ns["today"])
        for file, name in [("ticks", "df"), ("today", "today"), ("trades", "trades")]:
            assert pl.read_parquet(readme / f"{file}.parquet").equals(ns[name]), file

    #: The names the namespace hands out that the README also builds, and
    #: which of the README blocks that assign one it copies.
    COPIES = {
        "spec": "first",
        "grid": "last",
        "now": "first",
        "ticks": "first",
        "flows": "first",
        "blocks": "first",
        "by_block": "first",
        "closed": "first",
    }

    def test_the_namespace_copies_what_the_readme_builds(self, tmp_path, monkeypatch):
        """A README block runs on the names the README itself built
        (`_built_by_the_readme`), but the API reference's examples still
        read the namespace's copies (TC4), so each copy is what the README
        block it copies builds. `out` copies two at once: the evaluation
        section's ``ridge`` and ``kalman`` columns and the field-names
        section's ``m``; `bank` is `po.ModelBank([spec])` before any row,
        as the README's blocks start it. The fixture's `grid` had a
        `min_weight` and its `now` was 400.0, where the README's have none
        and 399.0; and `spec` was checked against nothing (review
        2026-10-05, TC5)."""
        monkeypatch.chdir(tmp_path)
        ns = _readme_namespace(tmp_path)

        def built(name: str, which: str, text: str = "") -> dict:
            blocks = [(ln, c) for _, ln, c in README_ONLY if name in _names(c)[0] and text in c]
            ln, c = blocks[0] if which == "first" else blocks[-1]
            local = _built_by_the_readme(c, ln, ns, {})
            exec(compile(c, f"README.md:{ln}", "exec"), local)
            return local

        def same(a: object, b: object) -> bool:
            if isinstance(a, pl.LazyFrame):
                a, b = a.collect(), b.collect()  # type: ignore[union-attr]
            if isinstance(a, pl.DataFrame):
                return a.equals(b)  # type: ignore[arg-type]
            return a == b

        for name, which in self.COPIES.items():
            assert same(built(name, which)[name], ns[name]), name
        evaluated = built("out", "first", "fit_predict([ridge, kalman])")["out"]
        assert ns["out"].select("ridge", "kalman").equals(evaluated.select("ridge", "kalman"))
        fields = built("grid", "last")["out"]
        assert ns["out"].select("m").equals(fields.select("m"))
        assert ns["bank"].save_bytes() == po.ModelBank([ns["spec"]]).save_bytes()

    def test_every_name_a_readme_example_reads_was_built_by_an_earlier_one(self):
        """A reader starts with nothing and runs the examples in order. The
        namespace above hands each block `df` and the rest ready-made, so a
        block that reads a frame the README only describes still runs there.
        I7's *The examples from here on read two frames* described `df` that
        way for four rewrites before a reader found it out of context
        (README-ITERATIONS, C11). So each block is parsed, not run, and every
        name it reads must be one that a block before it, or the block itself,
        assigns, imports or defines."""
        known = set(dir(builtins))
        unbuilt: dict[str, int] = {}
        for _, line, code in (b for b in README_BLOCKS if b[0] == "README.md"):
            stored, loaded = _names(code)
            for name in loaded:
                if name not in known and name not in stored:
                    unbuilt.setdefault(name, line)
            known |= stored
        assert not unbuilt, (
            "README examples read names that no example before them builds, "
            f"each at the line of its first use: {unbuilt}"
        )

    def test_an_example_on_the_example_data_says_so_just_above_it(self):
        """The user, 2026-10-05: when a code block reads the example data, the
        line of prose just above it says so, with a link to *Example data*
        (README-ITERATIONS, C15). A block reads it when it reads a name
        *Example data* builds without assigning that name itself, or a file
        *Example data* writes."""
        text = (REPO / "README.md").read_text(encoding="utf-8").split("\n")
        built = {"df", "lf", "trades", "today", "later"}
        written = re.compile(r"[\"'](ticks/|(ticks|today|trades)\.parquet[\"'])")
        missing = []
        for path, line, code in README_BLOCKS:
            if path != "README.md":
                continue
            names = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)]
            stored = {n.id for n in names if not isinstance(n.ctx, ast.Load)}
            loaded = {n.id for n in names if isinstance(n.ctx, ast.Load)}
            if stored >= built:  # Example data's own block, which builds them
                continue
            if not ((loaded - stored) & built or written.search(code)):
                continue
            above = line - 2  # the fence is at line - 1, counted from 0
            while above >= 0 and not text[above].strip():
                above -= 1
            if "](#example-data)" not in text[above]:
                missing.append(line)
        assert not missing, (
            "README blocks that read the example data with no line just above linking "
            f"*Example data*, at the lines of their fences: {missing}"
        )

    def test_no_example_calls_lazy(self):
        """The user, 2026-10-06: "We don't want the example code in the readme
        or anywhere else to be sprinkled with calls to .lazy(), show the
        example by saving tables to disk and then reading them from a lazy
        frame." An example's query reads a file with `pl.scan_parquet`, as a
        stream too large to hold is read, and *Example data* saves each frame
        it builds (docs/WRITING.md §3)."""
        docs = sorted(
            str(p.relative_to(REPO))
            for p in (REPO / "docs").rglob("*.md")
            if "_build" not in p.parts
        )
        blocks = [*_doc_blocks("README.md"), *(b for d in docs for b in _doc_blocks(d))]
        blocks += _docstring_blocks()
        assert len(blocks) > 100, "the examples were not found"
        found = [f"{where}:{line}" for where, line, code in blocks if ".lazy()" in code]
        assert not found, f"examples that call .lazy(): {found}"

    def test_there_are_shell_blocks_to_check(self):
        assert len(SHELL_BLOCKS) >= 4, SHELL_BLOCKS

    @pytest.mark.parametrize(
        ("path", "line", "code"),
        SHELL_BLOCKS,
        ids=[f"{p}:L{ln}" for p, ln, _ in SHELL_BLOCKS],
    )
    def test_a_shell_block_runs(self, path, line, code, tmp_path, monkeypatch, online_cli):
        """The runner guide is all command line now, so its examples are shell.
        They run the same way the python blocks do: against the files the
        fixture writes, in a directory of their own, with the built `online` on
        PATH so the block runs exactly as written.

        Only this document's: the README's ```sh blocks install the package and
        drive the toolchain, which a test must not do."""
        monkeypatch.chdir(tmp_path)
        _readme_namespace(tmp_path)  # writes bank.toml, bank.state and the inputs
        monkeypatch.setenv("PATH", f"{online_cli.parent}{os.pathsep}{os.environ['PATH']}")
        for command in (c for c in code.splitlines() if c.strip()):
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            assert res.returncode == 0, f"{command}\n{res.stderr}"

    def test_the_config_the_guide_shows_is_the_one_its_blocks_run(self, tmp_path):
        """The runner guide shows the `bank.toml` its shell blocks run against,
        and the fixture writes that file itself. Parsed, the two must be one
        configuration, or the guide would show a config its examples never ran."""
        import tomllib

        shown = [
            code
            for _, _, code in _doc_blocks("docs/RUNNER.md", "toml")
            if 'input = "ticks.parquet"' in code
        ]
        assert len(shown) == 1, shown
        _readme_namespace(tmp_path)
        written = (tmp_path / "bank.toml").read_text(encoding="utf-8")
        assert tomllib.loads(shown[0]) == tomllib.loads(written)

    def test_the_readme_config_gives_the_numbers_its_python_spec_gives(
        self, tmp_path, monkeypatch, online_cli
    ):
        """The README's TOML block mirrors its Python `spec` and says the
        `online` command line gives "the numbers the bank gives in Python".
        So the block runs here as written, on the example data's
        `ticks.parquet`, and every prediction, residual and weight it writes
        must be that spec's in Python, bit for bit. The coefficients are left
        out: they come on each group's last accepted row of each chunk, and the two
        runs chunk differently (PLAN §3)."""
        monkeypatch.chdir(tmp_path)
        shown = [code for _, _, code in _doc_blocks("README.md", "toml")]
        assert len(shown) == 1, shown
        ns = _readme_namespace(tmp_path)  # writes ticks.parquet
        (tmp_path / "readme.toml").write_text(shown[0], encoding="utf-8")
        subprocess.run(
            [str(online_cli), "--config", "readme.toml", "--quiet"],
            check=True,
            capture_output=True,
        )
        got = pl.read_parquet(tmp_path / "fitted.parquet")["ridge"].struct.unnest()
        want = po.ModelBank([ns["spec"]]).fit_predict(ns["df"])["ridge"].struct.unnest()
        assert got.columns == want.columns
        numbers = [c for c in want.columns if c.startswith(("pred_", "resid_", "weight_sum"))]
        assert len(numbers) == 5, want.columns  # pred and resid at each ridge, and the weight
        assert got.select(numbers).equals(want.select(numbers))

    def test_there_are_docstring_blocks_to_check(self):
        assert len(DOCSTRING_BLOCKS) >= 10, DOCSTRING_BLOCKS

    @pytest.mark.parametrize(
        ("path", "line", "code"),
        DOCSTRING_BLOCKS,
        ids=[f"{p}:L{ln}" for p, ln, _ in DOCSTRING_BLOCKS],
    )
    def test_a_docstring_block_runs(self, path, line, code, tmp_path, monkeypatch):
        """The API reference's examples, run in the README's namespace. That
        namespace hands every block the names the README builds, so a block
        that reads a name its own page never shows still runs here: the next
        test is the one that finds it (review 2026-10-05, TC4)."""
        monkeypatch.chdir(tmp_path)
        ns = _readme_namespace(tmp_path)
        exec(compile(code, f"{path}:{line}", "exec"), ns)

    def test_every_name_a_docstring_example_reads_is_built_on_its_page(self):
        """The README's rule (the test above it) for the API reference and
        the runner guide. A reader of one docstring has that docstring and
        the README's *Example data*, not the README's every name, which the
        namespace hands each block: `ModelBank.skip_learned`'s example read
        `rerun`, defined nowhere, and ran green (review 2026-10-05, TC4). So
        each block is parsed, not run, and every name it reads must be one
        that *Example data* builds, an earlier block of the same docstring
        or page builds, or the block itself builds."""
        known = set(dir(builtins)) | _names(_example_data_block())[0]
        runner = [b for b in README_BLOCKS if b[0] != "README.md"]
        unbuilt: dict[str, list[str]] = {}
        for where, group in itertools.groupby(DOCSTRING_BLOCKS + runner, key=lambda b: b[0]):
            built: set[str] = set()
            for _, line, code in group:
                stored, loaded = _names(code)
                missing = sorted({n for n in loaded if n not in known | built | stored})
                if missing:
                    unbuilt[f"{where}:L{line}"] = missing
                built |= stored
        assert not unbuilt, (
            f"docstring examples that read names their page never builds, by block: {unbuilt}"
        )
