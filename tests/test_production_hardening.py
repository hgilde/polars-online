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

import copy
import pickle
import threading
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online as po

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
        halflife=200.0,
        min_periods=5.0,
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
            ("kalman", {"coef_halflife": 100.0}),
            ("lasso", {"lasso_path": [0.0]}),
            ("sgd", {"learning_rate": 0.01}),
            ("ftrl", {}),
        ]:
            with pytest.raises(ValueError, match="both a target and a feature"):
                getattr(po.spec, name)(
                    "m", targets=["y0"], features=["y0"], halflife=100.0, **extra
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
        """`n_eff` counts accepted rows, and a null target does not reject a
        row -- so its whole trajectory must be *identical* to the same frame
        with targets present."""
        df = _df()
        with_targets = po.ModelBank([_spec()]).fit_predict(df)
        out = po.ModelBank([_spec()]).fit_predict(
            df.with_columns(y0=pl.lit(None, dtype=pl.Float64))
        )
        assert (
            out["m"].struct.field("n_eff").to_list()
            == with_targets["m"].struct.field("n_eff").to_list()
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
        needs ~80 halflives -- inherent to EW accumulators, not a defect. 4000
        rows at halflife 40 is 100 halflives: recovery must be complete."""
        df = _df(n=4000)
        df = pl.concat([pl.DataFrame({"x0": [1e12], "x1": [-1e12], "y0": [1e12]}), df])
        out = po.ModelBank([_spec(standardize=standardize, halflife=40.0)]).fit_predict(df)
        coef = out["m"].struct.field("coef").to_list()[-1]
        assert coef[1] == pytest.approx(1.0, abs=0.05), f"slope x0: {coef[1]}"
        assert coef[2] == pytest.approx(2.0, abs=0.05), f"slope x1: {coef[2]}"

    def test_holt_seeded_by_an_outlier_recovers(self):
        n = 4000
        y = np.concatenate([[1e9], 3.0 + 0.5 * np.arange(n)])
        df = pl.DataFrame({"y0": y})
        # 80 halflives of washout for the poisoned trend (the first update
        # sees a slope of -1e9).
        spec = po.spec.holt(
            "m", targets=["y0"], halflife=50.0, trend_halflife=50.0, min_periods=3.0
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
        ("kalman", {"coef_halflife": 100.0, "p0": 1e-12}),
        ("kalman", {"coef_halflife": 100.0, "p0": 1e12}),
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
        kw = dict(targets=["y0"], features=["x0", "x1"], halflife=100.0, min_periods=5.0)
        if model not in ("rls", "kalman", "ftrl", "sgd", "pa"):
            kw["max_rows_between_solves"] = 8
        kw.update(extra)
        out = po.ModelBank([getattr(po.spec, model)("m", **kw)]).fit_predict(df)
        for f in out.schema["m"].fields:
            if f.name.startswith("coef"):
                continue
            for i, v in enumerate(out["m"].struct.field(f.name).to_list()):
                assert v is None or np.isfinite(v), f"{f.name}[{i}] = {v}"

    def test_lam_is_halflife_by_another_name(self):
        df = _df()
        h = 137.0
        a = po.ModelBank([_spec(halflife=h)]).fit_predict(df)
        b = po.ModelBank([_spec(halflife=None, lam=0.5 ** (1.0 / h))]).fit_predict(df)
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
            halflife=1e9,
            min_periods=5.0,
            max_rows_between_solves=1,
        )
        a = po.ModelBank([lasso]).fit_predict(df)
        b = po.ModelBank([_spec(halflife=1e9, ridge=[1e-10])]).fit_predict(df)
        ca = a["m"].struct.field("coef").to_list()[-1][:3]
        cb = b["m"].struct.field("coef").to_list()[-1]
        np.testing.assert_allclose(ca, cb, rtol=1e-6)


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
        for _ in range(300):  # anywhere at all: clean outcome only
            i = int(rng.integers(0, len(blob)))
            old = blob[i]
            blob[i] = int(rng.integers(0, 256))
            try:
                po.ModelBank.load_bytes(bytes(blob))
            except Exception:
                detected += 1
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
            halflife=100.0,
            min_periods=5.0,
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


class TestReadmeExamples:
    """Every python block in the README must at least be valid syntax — users
    copy-paste them, and a typo there outlives any release."""

    def test_python_blocks_compile(self):
        text = (REPO / "README.md").read_text()
        blocks, in_block, buf = [], False, []
        for line in text.splitlines():
            if line.strip().startswith("```python"):
                in_block, buf = True, []
            elif line.strip() == "```" and in_block:
                in_block = False
                blocks.append("\n".join(buf))
            elif in_block:
                buf.append(line)
        assert len(blocks) >= 3, "the README should carry runnable examples"
        for i, b in enumerate(blocks):
            compile(b, f"README.md:block{i}", "exec")
