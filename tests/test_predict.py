"""E31: ``ModelBank.predict(df)`` scores a frame against the bank as it stands
and learns nothing.

The contract is stated as one equality: row ``i`` of ``predict(df)`` is row 0
of ``fit_predict(df.slice(i, 1))`` on a fresh copy of the bank -- the same
struct, field for field, that the row would get as the next row of its
stream. Everything else here is a consequence: the state is untouched (the
bytes say so), targets and weights are optional, unknown groups are null, a
trend model extrapolates over the clock, and any number of threads may score
at once.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from test_semantics_all_models import IDS, MODELS

# --- the oracle ----------------------------------------------------------------


def _fields(out: pl.DataFrame, col: str) -> list[str]:
    return [f.name for f in out.schema[col].fields]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def assert_row_oracle(bank, specs, df: pl.DataFrame, *, skip=("drift", "coef")) -> None:
    """``predict(df)[i] == fresh_clone.fit_predict(df[i])[0]`` for every row
    and every field except the ones documented to differ (``drift`` never
    fires when scoring; ``coef`` is placed on the last accepted row rather
    than by ``coef_every``)."""
    snap = bank.save_bytes()
    got = bank.predict(df)
    assert bank.save_bytes() == snap, "predict changed the bank"
    names = [s["name"] for s in specs]
    for i in range(df.height):
        fresh = po.ModelBank.load_bytes(snap, specs)
        want = fresh.fit_predict(df.slice(i, 1))
        for name in names:
            a, b = got[name][i], want[name][0]
            for k in a:
                if k.startswith(skip):
                    continue
                assert _same(a[k], b[k]), (
                    f"{name}.{k} at row {i}: predict {a[k]!r} != step {b[k]!r}"
                )


def _frame(n=60, seed=0, binary=False, groups=("a", "b")):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    lin = 0.5 + x0 * 2.0 - x1
    y = (lin > 0.5).astype(float) if binary else lin + 0.1 * rng.standard_normal(n)
    t = np.cumsum(rng.exponential(2.0, n))
    return pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "y0": y,
            "t": t,
            "w": rng.uniform(0.5, 1.5, n),
            "g": [groups[i % len(groups)] for i in range(n)],
        }
    )


def _spec(model, extra, **kw):
    opts = dict(
        targets=["y0"],
        features=["x0", "x1"],
        halflife=30.0,
        min_periods=2.0,
        clock="t",
        max_dclock=8.0,
        group="g",
        weight="w",
        emit_sigma=True,
        emit_resid_z=True,
        emit_metrics=True,
        resid_quantiles=[0.5],
        emit_autocorr=True,
        conformal=0.9,
        coef_every=7,
    )
    opts.update(extra)
    opts.update(kw)
    return getattr(po.spec, model)("m", **opts)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
def test_predict_is_fit_predict_of_the_next_row(model, extra):
    """The contract, per model, with every diagnostic on and a clock, groups
    and weights in play."""
    df = _frame(binary=model == "ftrl")
    specs = [_spec(model, extra)]
    bank = po.ModelBank(specs)
    bank.fit_predict(df.head(40))
    later = df.tail(20).with_columns(pl.col("t") + 3.0)
    # Some rows lack a target, some a feature: resid null vs. row skipped.
    later = later.with_columns(
        y0=pl.when(pl.arange(0, 20) % 6 == 1).then(None).otherwise(pl.col("y0")),
        x0=pl.when(pl.arange(0, 20) % 9 == 4).then(None).otherwise(pl.col("x0")),
    )
    assert_row_oracle(bank, specs, later)


def test_all_the_shared_options_at_once():
    """Two halflives, selection and averaging, sessions, a drift detector, a
    lasso path and a trend model in one bank, over the synthetic stream."""
    df, _ = synthetic(seed=3, n_groups=2, n_rows=120, k=2)
    specs = [
        po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            halflife=[50.0, 200.0],
            clock="t",
            max_dclock=100.0,
            group="group",
            session="session",
            session_gap=50.0,
            weight="w",
            min_periods=3.0,
            emit_sigma=True,
            emit_resid_z=True,
            emit_drift=True,
            emit_metrics=True,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
            conformal=0.95,
            emit_selected=True,
            emit_averaged=True,
            coef_every=10,
        ),
        po.spec.holt(
            "h",
            targets=["y0"],
            features=[],
            halflife=20.0,
            clock="t",
            group="group",
            max_dclock=30.0,
        ),
        po.spec.lasso(
            "l",
            targets=["y0"],
            features=["x0", "x1"],
            halflife=50.0,
            clock="t",
            max_dclock=100.0,
            group="group",
            lasso_path=[1.0, 0.1, 0.0],
        ),
    ]
    bank = po.ModelBank(specs)
    bank.fit_predict(df.head(150))
    assert_row_oracle(bank, specs, df.tail(90))


POLICIES = [
    pytest.param(dict(session="s", session_gap="reset"), id="session-reset"),
    pytest.param(dict(session="s", session_gap=5.0), id="session-gap"),
    pytest.param(
        dict(session="s", session_gap=5.0, session_shrink=0.5, long_halflife=300.0),
        id="session-shrink",
    ),
    pytest.param(dict(on_clock_reset="reset_state"), id="clock-reset-state"),
    pytest.param(dict(on_clock_reset="zero"), id="clock-zero"),
    pytest.param(dict(on_clock_reset="max"), id="clock-max"),
]


@pytest.mark.parametrize("policy", POLICIES)
def test_session_and_clock_policies_hold(policy):
    """A row that would reset the stream is scored by a fresh one, a row that
    would blend toward the long run by a blended copy, a session gap or a
    backwards clock by the delta the policy defines -- each exactly as
    `fit_predict` would have it for that row."""
    df = _frame(n=60, groups=("a",)).with_columns(s=pl.lit("one"))
    specs = [_spec("ewridge", {"max_rows_between_solves": 1}, **policy)]
    bank = po.ModelBank(specs)
    bank.fit_predict(df.head(40))
    later = df.tail(20)
    # Rows 5..9 start a new session; rows 12..14 are dated before the last
    # learned row.
    later = later.with_columns(
        s=pl.when(pl.arange(0, 20).is_between(5, 9)).then(pl.lit("two")).otherwise(pl.col("s")),
        t=pl.when(pl.arange(0, 20).is_between(12, 14))
        .then(pl.col("t") - 200.0)
        .otherwise(pl.col("t")),
    )
    assert_row_oracle(bank, specs, later)
    out = bank.predict(later)["m"]
    n_eff = out.struct.field("n_eff").to_list()
    pred = out.struct.field("pred_y0").to_list()
    if policy.get("session_gap") == "reset" or policy.get("on_clock_reset") == "reset_state":
        # A fresh stream has nothing to say -- and the rows around it are
        # scored by the bank as it stands, unaffected.
        fresh = range(5, 10) if "session" in policy else range(12, 15)
        assert all(pred[i] is None and n_eff[i] == 0.0 for i in fresh)
        assert all(pred[i] is not None for i in range(20) if i not in fresh)
    elif "session_shrink" in policy:
        # The blend changes the accumulated weight, and only for those rows.
        assert len({n_eff[i] for i in range(5, 10)}) == 1
        assert n_eff[5] != n_eff[0]
        assert all(n_eff[i] == n_eff[0] for i in range(20) if i not in range(5, 10))
    else:
        assert len(set(n_eff)) == 1


def test_a_backwards_clock_under_error_is_the_same_error():
    df = _frame(n=60, groups=("a",))
    specs = [_spec("ewridge", {}, on_clock_reset="error")]
    bank = po.ModelBank(specs)
    bank.fit_predict(df.head(40))
    bad = df.tail(20).with_columns(
        t=pl.when(pl.arange(0, 20) == 7).then(pl.col("t") - 200.0).otherwise(pl.col("t"))
    )
    with pytest.raises(ValueError, match=r"goes backwards by .* at row 7") as predict_err:
        bank.predict(bad)
    with pytest.raises(ValueError, match=r"goes backwards by .* at row 7") as fit_err:
        po.ModelBank.load_bytes(bank.save_bytes(), specs).fit_predict(bad)
    # The one difference: when scoring, every row is measured from the last
    # learned row, not from the row before it in the frame.
    back = bank.groups()["last_clock"][0] - bad["t"][7]
    assert f"goes backwards by {back} at row 7" in str(predict_err.value)
    strip = re.compile(r"backwards by [0-9.e+-]+ at")
    assert strip.sub("", str(predict_err.value)) == strip.sub("", str(fit_err.value))
    # Nothing was scored, and nothing changed.
    assert not bank.predict(df.tail(20))["m"].struct.field("pred_y0").is_null().any()


# --- what predict does not need, and does not do -------------------------------


class TestInputs:
    def setup_method(self):
        self.df = _frame()
        self.specs = [_spec("ewridge", {"max_rows_between_solves": 1})]
        self.bank = po.ModelBank(self.specs)
        self.trained = self.bank.fit_predict(self.df.head(40))
        self.later = self.df.tail(20)

    def test_target_and_weight_are_optional(self):
        full = self.bank.predict(self.later)
        bare = self.bank.predict(self.later.drop("y0", "w"))
        for f in _fields(full, "m"):
            a, b = full["m"].struct.field(f), bare["m"].struct.field(f)
            if f in ("resid_y0", "resid_z_y0"):
                # The one thing a target is for.
                assert b.is_null().all(), f
                assert not a.is_null().all(), f
            else:
                assert a.equals(b, null_equal=True), f"{f} differs without a target/weight column"

    def test_session_is_optional_and_feeds_the_gap(self):
        df = self.df.with_columns(s=pl.lit("one"))
        specs = [_spec("ewridge", {}, session="s", session_gap="reset")]
        bank = po.ModelBank(specs)
        bank.fit_predict(df.head(40))
        same = bank.predict(df.tail(20))
        # A new session resets the stream before the row: nothing learned yet.
        other = bank.predict(df.tail(20).with_columns(s=pl.lit("two")))
        assert other["m"].struct.field("pred_y0").is_null().all()
        assert not same["m"].struct.field("pred_y0").is_null().any()
        # Without the column, no session change can be seen.
        assert bank.predict(df.tail(20).drop("s"))["m"].equals(same["m"], null_equal=True)

    def test_features_and_clock_are_still_required(self):
        with pytest.raises(ValueError, match="feature"):
            self.bank.predict(self.later.drop("x0"))
        with pytest.raises(ValueError, match="clock"):
            self.bank.predict(self.later.drop("t"))
        with pytest.raises(ValueError, match="clock"):
            self.bank.predict(self.later.with_columns(t=pl.lit(None, dtype=pl.Float64)))

    def test_unknown_group_is_null_throughout(self):
        out = self.bank.predict(self.later.with_columns(g=pl.lit("never-seen")))
        for f in _fields(out, "m"):
            assert out["m"].struct.field(f).is_null().all(), f

    def test_a_fresh_bank_predicts_nothing(self):
        out = po.ModelBank(self.specs).predict(self.later)
        assert out["m"].struct.field("pred_y0").is_null().all()

    def test_an_empty_frame(self):
        out = self.bank.predict(self.later.clear())
        assert out.height == 0
        assert out.schema["m"] == self.bank.fit_predict(self.later.clear()).schema["m"]

    def test_row_order_does_not_matter(self):
        fwd = self.bank.predict(self.later)
        rev = self.bank.predict(self.later.reverse()).reverse()
        # ...except for where `coef` lands: on each group's last accepted row.
        # Rows alternate between the two groups.
        for f in _fields(fwd, "m"):
            a, b = fwd["m"].struct.field(f), rev["m"].struct.field(f)
            if f == "coef":
                assert a.is_null().to_list() == [i < 18 for i in range(20)]
                assert b.is_null().to_list() == [i >= 2 for i in range(20)]
                assert a.to_list()[18] == b.to_list()[0]  # group a
                assert a.to_list()[19] == b.to_list()[1]  # group b
            else:
                assert a.equals(b, null_equal=True), f

    def test_the_bank_does_not_move(self):
        before = self.bank.rows_seen()
        snap = self.bank.save_bytes()
        for _ in range(3):
            self.bank.predict(self.later)
        assert self.bank.rows_seen() == before
        assert self.bank.save_bytes() == snap
        # ...and the stream picks up exactly where it was.
        want = po.ModelBank.load_bytes(snap, self.specs).fit_predict(self.later)
        assert self.bank.fit_predict(self.later).equals(want, null_equal=True)

    def test_coef_is_on_each_groups_last_accepted_row(self):
        # Row 19 (group b) loses a feature, so group b's last accepted row is
        # 17; group a's is 18.
        later = self.later.with_columns(
            x0=pl.when(pl.arange(0, 20) == 19).then(None).otherwise(pl.col("x0"))
        )
        out = self.bank.predict(later)
        coef = out["m"].struct.field("coef").to_list()
        assert [c is None for c in coef] == [i not in (17, 18) for i in range(20)]
        # They are the coefficients the bank holds: the ones `fit_predict`
        # reported on each group's last training row, which is the state
        # *after* that row (coef is a snapshot of the state, not a prediction).
        trained = self.trained["m"].struct.field("coef").to_list()
        assert coef[18] == trained[38]  # group a
        assert coef[17] == trained[39]  # group b

    def test_type_and_name_errors_match_fit_predict(self):
        with pytest.raises(TypeError, match=r"predict takes a DataFrame, not a LazyFrame"):
            self.bank.predict(self.later.lazy())
        with pytest.raises(TypeError, match="takes a polars DataFrame, got dict"):
            self.bank.predict({"x0": [1.0]})
        with pytest.raises(ValueError, match="same name as an input column"):
            self.bank.predict(self.later.with_columns(m=pl.lit(1)))


# --- the reason it exists ---------------------------------------------------------


def test_the_e31_scenario_no_longer_goes_null():
    """`weight = 0` scored without learning but let the clock run, so `n_eff`
    decayed under `min_periods` and the outputs went null mid-batch
    (measured in docs/ENHANCEMENTS.md E31). `predict` freezes it."""
    n = 100
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    df = pl.DataFrame({"x0": x, "y": 2 * x, "t": np.arange(200.0), "w": np.ones(200)})
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], halflife=20.0, min_periods=10.0, weight="w"
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(n))
    batch = df.tail(n)
    zero_w = po.ModelBank.load_bytes(bank.save_bytes(), [spec]).fit_predict(
        batch.with_columns(w=pl.lit(0.0))
    )
    assert zero_w["m"].struct.field("pred_y").is_null().sum() > 30, "the scenario changed"
    scored = bank.predict(batch)
    assert not scored["m"].struct.field("pred_y").is_null().any()
    n_eff = scored["m"].struct.field("n_eff")
    assert n_eff.n_unique() == 1, "n_eff decayed while scoring"


def test_holt_extrapolates_over_the_clock_distance():
    """A trend model's prediction is `level + trend * h`, where the horizon
    `h` is the row's clock distance from the last learned row, capped by
    `max_dclock` -- measured from where the bank stands, not from the
    previous scored row. `coef` is `[level, trend]`, so the oracle is exact."""
    t = np.arange(50.0)
    df = pl.DataFrame({"y": 3.0 + 0.5 * t, "t": t})
    spec = po.spec.holt("h", targets=["y"], features=[], halflife=10.0, clock="t", max_dclock=20.0)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    last = bank.groups()["last_clock"][0]
    assert last == 49.0
    ahead = pl.DataFrame({"t": [50.0, 55.0, 60.0, 69.0, 80.0, 200.0, 49.0, 30.0]})
    out = bank.predict(ahead)["h"]
    pred = out.struct.field("pred_y").to_list()
    level, trend = out.struct.field("coef").to_list()[-1]
    assert trend > 0.2, "a rising series has a rising trend"
    for ti, p in zip(ahead["t"].to_list(), pred, strict=True):
        # `on_clock_reset = "max"`: a backwards clock is the maximum step.
        h = 20.0 if ti < last else min(ti - last, 20.0)
        assert p == pytest.approx(level + trend * h, abs=1e-12), (ti, h)


# --- threads ----------------------------------------------------------------------


def test_predict_from_many_threads_at_once():
    """Scoring is `&self`: threads never refuse each other and all get the
    same answer. Only a `fit_predict` in flight is refused."""
    n = 400_000
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y": rng.standard_normal(n)})
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=[10.0, 100.0, 1000.0])
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(1000))
    start = threading.Barrier(4)
    results: list[pl.DataFrame] = []
    errors: list[BaseException] = []

    def go():
        start.wait()
        try:
            results.append(bank.predict(df))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(results) == 4
    for r in results[1:]:
        assert r.equals(results[0], null_equal=True)


def test_fit_predict_is_refused_while_scoring():
    """Scoring holds a shared borrow; learning needs the exclusive one, and
    says so instead of waiting on or corrupting the scorers."""
    n = 2_000_000
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y": rng.standard_normal(n)})
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=[10.0, 100.0, 1000.0])
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(1000))
    started = threading.Event()
    errors: list[BaseException] = []

    def score():
        started.set()
        try:
            bank.predict(df)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=score) for _ in range(2)]
    for t in threads:
        t.start()
    started.wait()
    time.sleep(0.02)
    try:
        bank.fit_predict(df.head(10))
    except RuntimeError as e:
        refused = e
    else:
        pytest.skip("the scorers finished before fit_predict reached the bank")
    for t in threads:
        t.join()
    assert not errors, errors
    assert "in use on another thread" in str(refused)
    assert "concurrent `predict` calls are fine" in str(refused)
    assert bank.fit_predict(df.head(10)).height == 10


# --- the runner -------------------------------------------------------------------


class TestRunner:
    def _spec(self):
        return po.spec.ewridge(
            "ridge",
            targets=["y"],
            features=["x0"],
            clock="t",
            max_dclock=10.0,
            halflife=500.0,
            group="g",
            min_periods=20.0,
        )

    def _write(self, path, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal(n)
        df = pl.DataFrame(
            {
                "t": np.arange(float(n)),
                "x0": x0,
                "y": 2 * x0 + 0.1 * rng.standard_normal(n),
                "g": np.where(np.arange(n) % 2 == 0, "a", "b"),
            }
        )
        df.write_parquet(path)
        return df

    def test_predict_scores_the_loaded_bank(self, tmp_path):
        train = self._write(tmp_path / "train.parquet")
        state = tmp_path / "bank.state"
        po.run(
            input=tmp_path / "train.parquet",
            output=tmp_path / "o.parquet",
            specs=[self._spec()],
            save_state=state,
        )
        score = self._write(tmp_path / "score.parquet", n=1000, seed=1).with_columns(
            pl.col("t") + train.height
        )
        score.write_parquet(tmp_path / "score.parquet")
        stats = po.run(
            input=tmp_path / "score.parquet",
            output=tmp_path / "scored.parquet",
            specs=[self._spec()],
            load_state=state,
            predict=True,
            chunk_rows=300,
        )
        assert stats == {"rows": 1000, "chunks": 4}
        got = pl.read_parquet(tmp_path / "scored.parquet")
        want = po.ModelBank.load(state, [self._spec()]).predict(score)
        # `coef` is a reporting cadence (once per chunk per group), so it
        # lands on more rows in the chunked run; every other field is equal.
        for f in _fields(got, "ridge"):
            a, b = got["ridge"].struct.field(f), want["ridge"].struct.field(f)
            if f == "coef":
                assert a.is_null().sum() == 1000 - 8 and b.is_null().sum() == 1000 - 2
                assert set(map(tuple, a.drop_nulls().to_list())) == set(
                    map(tuple, b.drop_nulls().to_list())
                )
            else:
                assert a.equals(b, null_equal=True), f
        # No learning: the state file is untouched, and the same state gives
        # the same answer again.
        again = tmp_path / "scored2.parquet"
        po.run(
            input=tmp_path / "score.parquet",
            output=again,
            specs=[self._spec()],
            load_state=state,
            predict=True,
            chunk_rows=300,
        )
        assert pl.read_parquet(again).equals(got, null_equal=True)

    def test_predict_needs_a_state_and_refuses_to_save_one(self, tmp_path):
        self._write(tmp_path / "in.parquet")
        with pytest.raises(ValueError, match="predict = true needs load_state"):
            po.run(
                input=tmp_path / "in.parquet",
                output=tmp_path / "o.parquet",
                specs=[self._spec()],
                predict=True,
            )
        state = tmp_path / "bank.state"
        po.run(
            input=tmp_path / "in.parquet",
            output=tmp_path / "o.parquet",
            specs=[self._spec()],
            save_state=state,
        )
        with pytest.raises(ValueError, match="save_state has nothing to save"):
            po.run(
                input=tmp_path / "in.parquet",
                output=tmp_path / "o.parquet",
                specs=[self._spec()],
                load_state=state,
                save_state=tmp_path / "again.state",
                predict=True,
            )

    def test_one_toml_serves_both_the_learning_and_the_scoring_run(self, tmp_path, online_cli):
        """A checked-in config carries `save_state` for the learning run;
        `predict=True` from Python and `--predict` from the CLI drop it rather
        than refusing, so the same file drives both sides. An explicit
        `save_state` alongside `predict` is still the contradiction it was."""
        self._write(tmp_path / "in.parquet")
        state = tmp_path / "bank.state"
        toml = tmp_path / "bank.toml"
        toml.write_text(
            "\n".join(
                [
                    f'input = "{(tmp_path / "in.parquet").as_posix()}"',
                    f'output = "{(tmp_path / "o.parquet").as_posix()}"',
                    f'save_state = "{state.as_posix()}"',
                    "",
                    "[[specs]]",
                    'name = "ridge"',
                    'targets = ["y"]',
                    'features = ["x0"]',
                    'clock = "t"',
                    "max_dclock = 10.0",
                    "halflife = 500.0",
                    'group = "g"',
                    "min_periods = 20.0",
                    "",
                    "[specs.model]",
                    'type = "ew_ridge"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        assert po.run(toml) == {"rows": 4000, "chunks": 1}
        mtime = state.stat().st_mtime_ns
        stats = po.run(
            toml,
            output=tmp_path / "scored.parquet",
            load_state=state,
            predict=True,
        )
        assert stats == {"rows": 4000, "chunks": 1}
        assert state.stat().st_mtime_ns == mtime, "predict must not rewrite the state"
        with pytest.raises(ValueError, match="save_state has nothing to save"):
            po.run(
                toml,
                output=tmp_path / "scored.parquet",
                load_state=state,
                save_state=tmp_path / "again.state",
                predict=True,
            )
        self._cli_agrees(
            online_cli, toml, state, mtime, pl.read_parquet(tmp_path / "scored.parquet")
        )

    def _cli_agrees(self, exe, toml, state, mtime, want):
        out = toml.parent / "cli.parquet"
        res = subprocess.run(
            [
                str(exe),
                "--config", str(toml),
                "--output", str(out),
                "--resume", str(state),
                "--predict",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )  # fmt: skip
        assert res.returncode == 0, res.stderr
        assert pl.read_parquet(out).equals(want, null_equal=True)
        assert state.stat().st_mtime_ns == mtime
        res = subprocess.run(
            [
                str(exe),
                "--config", str(toml),
                "--output", str(out),
                "--resume", str(state),
                "--save-state", str(toml.parent / "again.state"),
                "--predict",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )  # fmt: skip
        assert res.returncode != 0
        assert "save_state has nothing to save" in res.stderr

    def test_empty_input_still_writes_the_schema(self, tmp_path):
        df = self._write(tmp_path / "in.parquet")
        state = tmp_path / "bank.state"
        po.run(
            input=tmp_path / "in.parquet",
            output=tmp_path / "o.parquet",
            specs=[self._spec()],
            save_state=state,
        )
        df.clear().write_parquet(tmp_path / "empty.parquet")
        stats = po.run(
            input=tmp_path / "empty.parquet",
            output=tmp_path / "e.parquet",
            specs=[self._spec()],
            load_state=state,
            predict=True,
        )
        assert stats == {"rows": 0, "chunks": 0}
        out = pl.read_parquet(tmp_path / "e.parquet")
        assert out.height == 0
        assert "ridge" in out.columns
