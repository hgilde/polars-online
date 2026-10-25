"""P4 of the code review of 2026-09-12: a window's ring of snapshots, each a
copy of the accumulator, had no bound -- at ``k = 1000`` over a 3,600-row
window about 29 GB per instance, and nothing checked. ``window_budget``
names what to do past a budget in MiB, on the user's decision of
2026-09-15 (docs/PLAN.md task 80): thin the ring, or refuse the run.
"""

from __future__ import annotations

import inspect
import math
import pickle

import numpy as np
import polars as pl
import pytest

import polars_online as po
from reference_paths import WindowRing

TIER = "essential"

#: MiB: about a kilobyte, a handful of one-feature snapshots.
TINY = 0.001


def _frame(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    return pl.DataFrame({"x0": x, "y": 2 * x + 0.1 * rng.standard_normal(n)})


#: This file's spec was frozen on its first solve once (2026-09-21): `min_weight
#: = 0` solves at one row, where the slope is entirely the ridge, and
#: `half_life = 1e9` left no clock cadence to refit, so `support_coef` read
#: 0.00 and a `ReadinessWarning` named it. The weight-share cadence (task 115
#: (b)) refits it as its weight grows: at 0.13.0 the last row reads
#: `support_coef = 1.00` and `coef = [0.006, 1.998]`, and nothing warns
#: (measured for task 116, 2026-10-07). What is under test here is the window
#: ring's bookkeeping (`weight_sum`), which does not depend on the fit either
#: way.


def _spec(**kw):
    kw.setdefault("half_life", 1e9)
    return po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        window_size=50.0,
        snapshot_every=1,
        min_weight=0.0,
        **kw,
    )


def _n_eff(spec, df):
    return po.ModelBank([spec]).fit_predict(df).unnest("m")["weight_sum"].to_numpy()


def test_past_the_budget_the_ring_thins_and_never_keeps_an_older_row():
    """Thinning keeps every other snapshot and doubles the spacing, so the
    window's boundary grows coarser: it drops more rows than asked, and
    never keeps one the window excludes."""
    df = _frame()
    exact = _n_eff(_spec(), df)
    thin = _n_eff(_spec(window_budget={"thin": TINY}), df)
    assert (thin <= exact + 1e-9).all(), "a thinned window holds no row the exact one does not"
    assert (thin < exact - 0.5).any(), "and past the budget it did thin"


def test_past_the_budget_the_chunk_is_refused_whole_and_the_bank_goes_on():
    """The bank replays each chunk's clock schedule on the rings' shadows
    before it learns a row (docs/PLAN.md task 115 (d)), so a chunk that would
    take a ring past a refusing budget is refused whole, naming the way out,
    and the bank is exactly as it was: it saves the same bytes, and takes
    the chunk that fits. The ring found the overrun with the chunk half
    learned before, and every later ``fit_predict``, ``predict`` and ``save``
    was refused."""
    df = _frame()
    bank = po.ModelBank([_spec(window_budget={"refuse": TINY})])
    bank.fit_predict(df.head(1))
    before = bank.save_bytes()
    with pytest.raises(ValueError, match="window_budget") as exc:
        bank.fit_predict(df.slice(1))
    assert "snapshot_every" in str(exc.value)
    assert bank.save_bytes() == before, "nothing of the refused chunk was learned"
    bank.predict(df.head(1))
    bank.fit_predict(df.slice(1, 2))
    # The same rows fed one chunk at a time reach the budget on the same row,
    # refused there: the prediction is the ring's, whatever the chunking.
    fed = po.ModelBank([_spec(window_budget={"refuse": TINY})])
    fed.fit_predict(df.head(1))
    reached = None
    for i in range(1, df.height):
        try:
            fed.fit_predict(df.slice(i, 1))
        except ValueError as e:
            reached = (i, str(e))
            break
    assert reached is not None and "window_budget" in reached[1]
    whole = po.ModelBank([_spec(window_budget={"refuse": TINY})])
    whole.fit_predict(df.head(1))
    whole.fit_predict(df.slice(1, reached[0] - 1))
    with pytest.raises(ValueError, match="window_budget"):
        whole.fit_predict(df.slice(reached[0], 1))


def test_a_broken_bank_names_itself_before_a_bad_column(tmp_path):
    """docs/REVIEW-2026-09-17.md S3 moved the broken-bank refusal ahead of
    the frame's columns in both polars entry points, and left the order
    untested. A frame that is itself bad -- here missing the feature --
    still gets the bank's refusal, which is the one that says what to do;
    the column error would send the reader to a frame that is not the
    problem.

    A bank breaks where the pre-pass cannot see the ring (docs/PLAN.md task
    115 (d)): under ``drift_action="reset"``, whose resets depend on the
    residuals, the ring's own refusal stops the run mid-chunk, as it did
    everywhere before.

    The bad column is a text feature because the frame adapter refuses it
    while casting, which is what the old order ran first. A missing column
    would not tell the orders apart: the adapter leaves that to the bank."""
    df = _frame()
    bank = po.ModelBank(
        [_spec(window_budget={"refuse": TINY}, emit_drift=True, drift_action="reset")]
    )
    bank.fit_predict(df.head(1))
    with pytest.raises(ValueError, match="window_budget"):
        bank.fit_predict(df.slice(1))
    with pytest.raises(ValueError, match="cannot go on.*window_budget"):
        bank.save_bytes()
    # Every way out of the bank refuses it the same way. `save` raised a bare
    # OSError, and `to_json` exported the state `save` refuses (review
    # 2026-10-05, YA2); the file a save would replace is left as it was.
    state = tmp_path / "b.state"
    state.write_bytes(b"the last good state")
    with pytest.raises(ValueError, match="cannot go on.*window_budget") as exc:
        bank.save(state)
    assert not isinstance(exc.value, OSError)
    assert state.read_bytes() == b"the last good state"
    for export in (bank.write_json, lambda: bank.write_json(tmp_path / "b.json")):
        with pytest.raises(ValueError, match="cannot go on.*window_budget"):
            export()
    assert not (tmp_path / "b.json").exists()
    with pytest.raises(ValueError, match="cannot go on.*window_budget"):
        pickle.dumps(bank)
    # A filesystem's refusal is still the OSError of its kind.
    with pytest.raises(FileNotFoundError, match="nowhere"):
        po.ModelBank([_spec()]).save(tmp_path / "nowhere" / "b.state")
    bad = df.head(1).with_columns(pl.col("x0").cast(pl.String))
    # The frame is bad on its own: a healthy bank refuses it for the column.
    with pytest.raises(ValueError, match='"x0" has dtype str; it must be numeric'):
        po.ModelBank([_spec()]).fit_predict(bad)
    for call in (bank.fit_predict, bank.predict):
        with pytest.raises(ValueError, match="cannot go on.*window_budget"):
            call(bad)


def test_a_budget_the_ring_stays_under_changes_nothing():
    df = _frame()
    plain = po.ModelBank([_spec()]).fit_predict(df)
    for action in ("thin", "refuse"):
        roomy = po.ModelBank([_spec(window_budget={action: 64.0})]).fit_predict(df)
        assert roomy.equals(plain), action


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"thin": 0.0}, "window_budget must be > 0"),
        ({"refuse": -1.0}, "window_budget must be > 0"),
        ({"thin": float("nan")}, "must not be NaN"),
        ({"shrink": 1.0}, "thin"),
    ],
)
def test_a_bad_budget_is_refused_by_name(budget, message):
    with pytest.raises(ValueError, match=message):
        _spec(window_budget=budget)


def test_a_budget_without_a_window_is_refused():
    with pytest.raises(ValueError, match="window_budget needs `window_size`"):
        po.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=10.0, window_budget={"thin": 1.0}
        )


def test_an_infinite_budget_is_no_bound_and_survives_a_save():
    bank = po.ModelBank([_spec(window_budget={"thin": math.inf})])
    bank.fit_predict(_frame())
    loaded = po.ModelBank.load_bytes(bank.save_bytes())
    assert loaded.specs == bank.specs
    assert loaded.specs[0]["model"]["window_budget"] == {"thin": math.inf}


def test_the_budget_holds_after_a_load():
    """The budget is the spec's and not the state's: a bank loaded from a
    file thins where the bank that wrote it would have."""
    df = _frame(400)
    spec = _spec(window_budget={"thin": TINY})
    whole = po.ModelBank([spec]).fit_predict(df).unnest("m")
    first = po.ModelBank([spec])
    head = first.fit_predict(df.head(150)).unnest("m")
    tail = po.ModelBank.load_bytes(first.save_bytes()).fit_predict(df.tail(250)).unnest("m")
    both = pl.concat([head, tail])
    for col in ("weight_sum", "pred_y"):
        assert both[col].equals(whole[col]), col


# --- every windowed model ----------------------------------------------------

#: One spec of each windowed kind, over ``_frame``'s columns and a label.
WINDOWED = {
    "ewridge": lambda **kw: po.spec.ewridge("m", targets=["y"], features=["x0"], **kw),
    "lasso": lambda **kw: po.spec.lasso("m", targets=["y"], features=["x0"], penalties=[0.1], **kw),
    "ew_cov": lambda **kw: po.spec.ew_cov("m", features=["x0", "y"], **kw),
    "ew_class": lambda **kw: po.spec.ew_class(
        "m", features=["x0"], label="c", classes=["a", "b"], precision_prior=1.0, **kw
    ),
    "marginal": lambda **kw: po.spec.marginal("m", targets=["y"], features=["x0"], **kw),
}


@pytest.mark.parametrize("kind", sorted(WINDOWED))
def test_every_windowed_model_holds_its_budget(kind):
    """Each windowed model hands the budget to its own ring. The trait's
    defaults do nothing, so a model that forgot to would run on past a tiny
    refusing budget without a word (docs/EXTENDING.md)."""
    label = pl.when(pl.col("x0") > 0).then(pl.lit("a")).otherwise(pl.lit("b"))
    df = _frame().with_columns(c=label)
    spec = WINDOWED[kind](
        half_life=1e9, window_size=50.0, snapshot_every=1, window_budget={"refuse": TINY}
    )
    with pytest.raises(ValueError, match="window_budget"):
        po.ModelBank([spec]).fit_predict(df)


def test_the_budget_table_names_every_windowed_builder():
    """A new builder that takes ``window_budget`` belongs in ``WINDOWED``, or
    the test above cannot check it."""
    takes = set()
    for name in dir(po.spec):
        builder = getattr(po.spec, name)
        if name.startswith("_") or not callable(builder):
            continue
        try:
            params = inspect.signature(builder).parameters
        except (TypeError, ValueError):
            continue
        if "window_budget" in params:
            takes.add(name)
    assert takes == set(WINDOWED)


# --- thinning on every windowed model (task 278, batch 10) ---------------------

#: A half-life the ring's products are read with, on the row-number clock, a
#: window of 50 rows and a snapshot on every row, so a budget of a few
#: snapshots thins it.
THIN = {"half_life": 40.0, "window_size": 50.0, "snapshot_every": 0}


def _labelled(n: int = 300) -> pl.DataFrame:
    label = pl.when(pl.col("x0") > 0).then(pl.lit("a")).otherwise(pl.lit("b"))
    return _frame(n).with_columns(c=label)


def _run(spec, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([spec]).fit_predict(df).unnest("m")


def _capacity(kind: str, df: pl.DataFrame) -> int:
    """How many of ``kind``'s snapshots ``TINY`` MiB holds: the row at which a
    refusing budget of as much refuses, fed a row at a time on a window no
    row leaves, each row adding one snapshot. A thinning budget counts the
    same bytes."""
    bank = po.ModelBank(
        [WINDOWED[kind](**(THIN | {"window_size": 1e9}), window_budget={"refuse": TINY})]
    )
    for i in range(df.height):
        try:
            bank.fit_predict(df.slice(i, 1))
        except ValueError as e:
            assert "window_budget" in str(e), e
            return i
    raise AssertionError(f"{TINY} MiB held every row's snapshot")


def _ring_weights(n: int, capacity: int | None) -> tuple[np.ndarray, WindowRing]:
    """Each row's ``weight_sum`` under ``THIN`` (unit weights on the
    row-number clock) from the front of a ring of at most ``capacity``
    snapshots, and the ring."""
    ring = WindowRing(THIN["window_size"], capacity=capacity)
    starts = [ring.learn(i, float(i)) for i in range(n)]
    want = np.zeros(n)
    for t in range(1, n):
        ages = (t - 1) - np.arange(starts[t - 1], t)
        want[t] = (0.5 ** (ages / THIN["half_life"])).sum()
    return want, ring


@pytest.mark.parametrize("kind", sorted(WINDOWED))
def test_past_a_thinning_budget_every_windowed_model_reads_the_thinned_ring(kind):
    """Each windowed model thins its own ring past the budget: its
    ``weight_sum`` is the weight from the front of a ring that keeps the
    newest snapshot and every second one before it whenever it would pass
    the budget's capacity (``reference_paths.WindowRing``), which drops more
    rows than the exact window and never keeps an older one. Before task 278
    (batch 10) thinning was shown on ``ewridge`` alone."""
    df = _labelled()
    cap = _capacity(kind, df)
    got = _run(WINDOWED[kind](**THIN, window_budget={"thin": TINY}), df)["weight_sum"].to_numpy()
    want, ring = _ring_weights(df.height, cap)
    exact, _ = _ring_weights(df.height, None)
    print(f"{kind}: capacity {cap}, thinned {ring.thinned} times")
    # Measured 2026-10-25: capacities of 4 (ewridge, lasso) to 14 (ew_cov,
    # marginal) snapshots, each ring thinned 2 to 4 times.
    assert ring.thinned >= 1
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    assert (got <= exact + 1e-9).all(), "a thinned window holds no row the exact one does not"
    assert (got < exact - 0.5).any(), "and past the budget it did thin"


@pytest.mark.parametrize("kind", sorted(WINDOWED))
def test_a_budget_every_windowed_ring_stays_under_changes_nothing(kind):
    df = _labelled()
    plain = _run(WINDOWED[kind](**THIN), df)
    for action in ("thin", "refuse"):
        roomy = _run(WINDOWED[kind](**THIN, window_budget={action: 64.0}), df)
        assert roomy.equals(plain), (kind, action)


@pytest.mark.parametrize("kind", sorted(WINDOWED))
def test_a_thinned_ring_does_not_depend_on_the_chunking_or_a_save(kind):
    """Hard rule 3 under a thinning budget, and a save mid-ring: the doubled
    cadence and the thinned ring are state, and the budget is the spec's, so
    a loaded bank goes on thinning where the unbroken one does."""
    df = _labelled()
    spec = WINDOWED[kind](**THIN, window_budget={"thin": TINY})
    whole = _run(spec, df)
    floats = [c for c, t in whole.schema.items() if t.is_float()]
    for rows in (1, 7, 37):
        parts = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_size=rows))
        parts = parts.unnest("m")
        for f in floats:
            assert whole[f].equals(parts[f]), (kind, rows, f)
    for cut in (23, 151):
        first = po.ModelBank([spec])
        first.fit_predict(df.head(cut))
        tail = po.ModelBank.load_bytes(first.save_bytes()).fit_predict(df.slice(cut)).unnest("m")
        for f in floats:
            assert whole[f].slice(cut).equals(tail[f]), (kind, cut, f)
