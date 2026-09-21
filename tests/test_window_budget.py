"""P4 of the code review of 2026-09-12: a window's ring of snapshots, each a
copy of the accumulator, had no bound -- at ``k = 1000`` over a 3,600-row
window about 29 GB per instance, and nothing checked. ``window_budget``
names what to do past a budget in MiB, on the user's decision of
2026-09-15 (docs/PLAN.md task 80): thin the ring, or refuse the run.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

#: MiB: about a kilobyte, a handful of one-feature snapshots.
TINY = 0.001


def _frame(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    return pl.DataFrame({"x0": x, "y": 2 * x + 0.1 * rng.standard_normal(n)})


#: This file's spec is frozen on its first solve by construction: `min_periods
#: = 0` solves at one row, where the slope has no variance to read and is
#: entirely the ridge, and `halflife = 1e9` leaves no solve cadence to refit --
#: `pred_y` is one constant for all 300 rows. `support_coef` reads 0.00 and the
#: warning that names it is *correct*; it is simply beside the point here,
#: where what is under test is the window ring's bookkeeping (`n_eff`), which
#: does not depend on the fit (2026-09-21).
pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def _spec(**kw):
    kw.setdefault("halflife", 1e9)
    return po.spec.ewridge(
        "m", targets=["y"], features=["x0"], window=50.0, window_every=1, min_periods=0.0, **kw
    )


def _n_eff(spec, df):
    return po.ModelBank([spec]).fit_predict(df).unnest("m")["n_eff"].to_numpy()


def test_past_the_budget_the_ring_thins_and_never_keeps_an_older_row():
    """Thinning keeps every other snapshot and doubles the spacing, so the
    window's boundary grows coarser: it drops more rows than asked, and
    never keeps one the window excludes."""
    df = _frame()
    exact = _n_eff(_spec(), df)
    thin = _n_eff(_spec(window_budget={"thin": TINY}), df)
    assert (thin <= exact + 1e-9).all(), "a thinned window holds no row the exact one does not"
    assert (thin < exact - 0.5).any(), "and past the budget it did thin"


def test_past_the_budget_a_refusal_stops_the_run_naming_the_way_out():
    """The budget is checked as the rows are learned, not before, so a chunk
    refused for it has been partly learned -- unlike a chunk refused before
    any stream is touched. The bank says so rather than go on: every later
    ``fit_predict``, ``predict`` and ``save`` is refused, naming the budget."""
    df = _frame()
    bank = po.ModelBank([_spec(window_budget={"refuse": TINY})])
    bank.fit_predict(df.head(1))
    with pytest.raises(ValueError, match="window_budget") as exc:
        bank.fit_predict(df.slice(1))
    assert "window_every" in str(exc.value)
    for call in (
        lambda: bank.fit_predict(df.head(1)),
        lambda: bank.predict(df.head(1)),
        bank.save_bytes,
    ):
        with pytest.raises(ValueError, match="cannot go on.*window_budget"):
            call()


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
    with pytest.raises(ValueError, match="window_budget needs `window`"):
        po.spec.ewridge(
            "m", targets=["y"], features=["x0"], halflife=10.0, window_budget={"thin": 1.0}
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
    for col in ("n_eff", "pred_y"):
        assert both[col].equals(whole[col]), col


# --- every windowed model ----------------------------------------------------

#: One spec of each windowed kind, over ``_frame``'s columns and a label.
WINDOWED = {
    "ewridge": lambda **kw: po.spec.ewridge("m", targets=["y"], features=["x0"], **kw),
    "lasso": lambda **kw: po.spec.lasso(
        "m", targets=["y"], features=["x0"], lasso_path=[0.1], **kw
    ),
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
    spec = WINDOWED[kind](halflife=1e9, window=50.0, window_every=1, window_budget={"refuse": TINY})
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
