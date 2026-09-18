"""A plan whose row order is unspecified is warned about before a bank reads it.

An online model learns in row order, so the order a plan delivers *is* the
model. ``fit(lf)`` and ``lf.online.fit_predict`` run the plan through polars'
streaming engine, and a join, ``group_by`` or ``unique`` without an order
guarantee delivers a different stream there than ``lf.collect()`` gives --
measured on 200k rows: ``collect()`` kept the input order, ``collect_batches()``
did not, and ``maintain_order="left"`` made the two agree. That is silent in
every other respect, so the plan is inspected when it is handed over and an
``OrderNotGuaranteedWarning`` names the node and the fix.

The inspection reads ``LazyFrame.serialize(format="json")``, which polars has
deprecated; it is best-effort by design and falls silent -- never raises -- on
a plan it cannot read. The docs line on each entry point is the guarantee; the
warning is the safety net under it.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import polars_online as po

SPEC = po.spec.ewridge(
    "m", targets=["y"], features=["x0"], clock="t", halflife=10.0, max_dclock=5.0, min_periods=1.0
)


def left() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "k": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "t": [float(i) for i in range(9)],
            "x0": [0.1 * i for i in range(9)],
            "y": [0.2 * i for i in range(9)],
        }
    )


def right() -> pl.LazyFrame:
    return pl.LazyFrame({"k": [1, 2, 3], "z": [10, 20, 30]})


def quiet(fn):
    """Run ``fn`` and fail on any ``OrderNotGuaranteedWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", po.OrderNotGuaranteedWarning)
        return fn()


# ------------------------------------------------------------- what is flagged


def test_a_join_without_maintain_order_is_flagged():
    lf = left().join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning, match="row order is not guaranteed") as rec:
        po.ModelBank([SPEC]).fit(lf, chunk_rows=4)
    text = str(rec[0].message)
    assert "join" in text
    assert 'maintain_order="left"' in text


def test_a_join_that_keeps_the_left_order_is_not():
    lf = left().join(right(), on="k", maintain_order="left")
    quiet(lambda: po.ModelBank([SPEC]).fit(lf, chunk_rows=4))


def test_a_group_by_without_maintain_order_is_flagged():
    lf = left().group_by("k").agg(pl.col("y").sum(), pl.col("x0").mean(), pl.col("t").min())
    with pytest.warns(po.OrderNotGuaranteedWarning, match="group_by") as rec:
        po.ModelBank([SPEC]).fit(lf)
    assert "maintain_order=True" in str(rec[0].message)


def test_a_group_by_that_keeps_order_is_not():
    lf = (
        left()
        .group_by("k", maintain_order=True)
        .agg(pl.col("y").sum(), pl.col("x0").mean(), pl.col("t").min())
    )
    quiet(lambda: po.ModelBank([SPEC]).fit(lf))


@pytest.mark.parametrize("maintain_order", [False, True])
def test_unique_is_flagged_whatever_its_flag(maintain_order):
    """The streaming engine does not honour ``maintain_order`` on ``unique``
    (polars' own note on ``DistinctOptionsDSL``), so the flag is no fix and the
    advice is to sort after it."""
    lf = left().unique("k", maintain_order=maintain_order)
    with pytest.warns(po.OrderNotGuaranteedWarning, match="unique") as rec:
        po.ModelBank([SPEC]).fit(lf)
    assert "sort" in str(rec[0].message)


def test_a_sort_above_the_node_settles_the_order():
    """A sort defines the order of everything beneath it, so a join under a
    sort is not a hazard."""
    lf = left().join(right(), on="k").sort("t")
    quiet(lambda: po.ModelBank([SPEC]).fit(lf, chunk_rows=4))


def test_a_sort_below_the_node_does_not():
    """... but a join above a sort reorders what the sort arranged."""
    lf = left().sort("t").join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning):
        po.ModelBank([SPEC]).fit(lf, chunk_rows=4)


def test_a_plan_with_no_reordering_node_is_quiet():
    lf = left().filter(pl.col("t") > 0).with_columns(w=pl.lit(1.0)).select("t", "x0", "y")
    quiet(lambda: po.ModelBank([SPEC]).fit(lf))


# ------------------------------------------------------------- where it fires


def test_every_plan_entry_point_warns():
    lf = left().join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning, match="fit_predict_batches"):
        list(po.ModelBank([SPEC]).fit_predict_batches(lf, chunk_rows=4))
    with pytest.warns(po.OrderNotGuaranteedWarning, match="fit"):
        po.ModelBank([SPEC]).fit(lf, chunk_rows=4)
    # The plan form warns when the plan is built, before anything runs.
    with pytest.warns(po.OrderNotGuaranteedWarning, match="online.fit_predict"):
        lf.online.fit_predict([SPEC])
    bank = po.ModelBank([SPEC])
    bank.fit(left())
    with pytest.warns(po.OrderNotGuaranteedWarning, match="online.predict"):
        lf.online.predict(bank)


def test_a_frame_and_an_iterator_are_not_inspected():
    """Only a plan has an order that is still to be decided; a frame or a
    sequence of frames arrives in the order it arrives."""
    df = left().join(right(), on="k").collect()
    quiet(lambda: po.ModelBank([SPEC]).fit(df))
    quiet(lambda: po.ModelBank([SPEC]).fit([df.slice(0, 4), df.slice(4)]))


# ------------------------------------------------------------- what it must not do


def test_polars_deprecation_of_json_serialize_is_not_leaked():
    """The inspection reads a deprecated polars format and must not pass
    polars' warning about that on to a caller who never asked for it."""
    lf = left().join(right(), on="k")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        po.ModelBank([SPEC]).fit(lf, chunk_rows=4)
    leaked = [w for w in rec if "serialization format" in str(w.message)]
    assert not leaked, [str(w.message) for w in leaked]
    assert any(isinstance(w.message, po.OrderNotGuaranteedWarning) for w in rec)


def test_it_falls_silent_rather_than_raise_when_the_plan_cannot_be_read(monkeypatch):
    """Best-effort by design: a polars without the format, or a plan it cannot
    serialize, gives a run with no warning, never a run that fails."""

    def unreadable(self, *args, **kwargs):
        raise RuntimeError("no serialization here")

    monkeypatch.setattr(pl.LazyFrame, "serialize", unreadable)
    lf = left().join(right(), on="k")
    quiet(lambda: po.ModelBank([SPEC]).fit(lf, chunk_rows=4))


def test_a_plan_already_holding_a_bank_is_not_readable_and_still_runs():
    """A plan with one of this library's own sources in it cannot be
    serialized (its source is a Python callable); the outer bank runs, and says
    nothing about an order it could not inspect."""
    inner = left().online.fit_predict([SPEC])
    outer = po.spec.ewridge(
        "n",
        targets=["y"],
        features=["x0"],
        clock="t",
        halflife=10.0,
        max_dclock=5.0,
        min_periods=1.0,
    )
    bank = po.ModelBank([outer])
    quiet(lambda: bank.fit(inner))
    assert bank.rows_seen() == 9


def test_the_prefilter_and_the_walk_cannot_drift_apart():
    """``_order_hazards`` skips the deprecated JSON read when ``explain``'s
    text holds none of ``_HAZARD_TAGS``' markers. That is only safe while the
    table lists every tag ``_walk`` reports: one handled there and missing
    here would stop being warned about **silently**, which is the worst way
    this code can fail. ``Sort`` is excluded because it ends the walk rather
    than reporting anything."""
    import inspect
    import re

    from polars_online import _frame

    handled = set(re.findall(r'tag == "(\w+)"', inspect.getsource(_frame._walk))) - {"Sort"}
    assert handled, "the walk's tags could not be read; this guard is not checking anything"
    assert handled == set(_frame._HAZARD_TAGS), (
        f"_walk reports {sorted(handled)} but _HAZARD_TAGS lists "
        f"{sorted(_frame._HAZARD_TAGS)}; add the missing tag's explain marker, "
        "or the pre-filter will skip plans that should warn"
    )


@pytest.mark.parametrize(
    ("how", "marker"),
    [("inner", "JOIN"), ("left", "JOIN"), ("semi", "JOIN"), ("anti", "JOIN"), ("cross", "JOIN")],
)
def test_every_join_variant_carries_its_marker(how, marker):
    """The pre-filter rests on ``explain`` naming each hazard node. Checked
    across the join variants because one unnamed variant would be skipped
    without a warning and without a failure."""
    other = right()
    lf = left().join(other, on=None if how == "cross" else "k", how=how)
    from polars_online._frame import _order_hazards

    assert marker in lf.explain(optimized=False)
    assert _order_hazards(lf), "a join without maintain_order should still be a hazard"


def test_a_plan_with_no_hazard_node_skips_the_json_read(monkeypatch):
    """The saving itself: a plan whose ``explain`` holds no marker must not
    touch ``serialize``, which is the deprecated format this avoids."""
    from polars_online import _frame

    # A `BaseException`, deliberately: `_order_hazards` wraps the JSON path in
    # `except Exception`, so an `AssertionError` raised here would be swallowed
    # and the test would pass whether or not the filter worked.
    class _SerializeCalled(BaseException):
        pass

    def forbidden(self, *args, **kwargs):
        raise _SerializeCalled("serialize was called for a plan with no hazard marker")

    monkeypatch.setattr(pl.LazyFrame, "serialize", forbidden)
    lf = left().filter(pl.col("t") > 0).select("t", "x0", "y")
    assert _frame._order_hazards(lf) == []
    quiet(lambda: po.ModelBank([SPEC]).fit(lf))


def test_a_plan_that_cannot_be_explained_still_reads_the_json(monkeypatch):
    """Fail open: if ``explain`` raises, the filter must fall through to the
    JSON path rather than skip the check, so it can only make the inspection
    cheaper, never blinder."""
    from polars_online import _frame

    monkeypatch.setattr(
        pl.LazyFrame, "explain", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("no plan"))
    )
    lf = left().join(right(), on="k")
    assert _frame._order_hazards(lf), "a hazard must still be found when explain fails"


def test_the_warning_is_a_user_warning_shown_by_default():
    """A ``DeprecationWarning`` is hidden outside ``__main__``, i.e. in the
    pipeline module where this matters; a ``UserWarning`` is shown."""
    assert issubclass(po.OrderNotGuaranteedWarning, UserWarning)
    assert not issubclass(po.OrderNotGuaranteedWarning, DeprecationWarning)


def test_the_warning_points_at_the_caller_not_the_library():
    lf = left().join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning) as rec:
        po.ModelBank([SPEC]).fit(lf, chunk_rows=4)
    assert rec[0].filename == __file__, rec[0].filename


# ------------------------------------ the exception: a fit whose order cannot
# ------------------------------------ change what it leaves behind

#: No decay, no window, nothing that reads a sequence: an accumulation, whose
#: sums commute. `SPEC` above has a `halflife`, so every test before this one
#: is disqualified and warns exactly as it did.
FREE = po.spec.ewridge("m", targets=["y"], features=["x0"], lam=1.0, min_periods=1.0)


def _rows(n: int = 200) -> pl.DataFrame:
    """Enough rows that a shuffle says something; `left()` has nine."""
    import math

    x0 = [math.sin(i * 0.3) for i in range(n)]
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": x0,
            "y": [0.7 * x0[i] + 0.05 * math.sin(i) for i in range(n)],
        }
    )


def _coefs(spec, frame) -> list[float]:
    bank = po.ModelBank([spec])
    bank.fit(frame.lazy())
    return bank.coef("m")["coef"].to_list()


def _spread(spec, frame) -> float:
    """How far the fitted coefficients move when the same rows arrive shuffled."""
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    a, b = _coefs(spec, frame), _coefs(spec, shuffled)
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def test_an_accumulator_with_no_decay_is_recognised_as_order_free():
    from polars_online._frame import _order_free

    assert _order_free([FREE])


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(SPEC, id="halflife"),
        pytest.param(
            po.spec.ewridge(
                "m", targets=["y"], features=["x0"], lam=1.0, window=50, min_periods=1.0
            ),
            id="window",
        ),
        pytest.param(
            po.spec.sgd("m", targets=["y"], features=["x0"], lam=1.0, min_periods=1.0),
            id="gradient-model",
        ),
        pytest.param(
            po.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                lam=1.0,
                min_periods=1.0,
                drift_action="reset",
                drift_threshold=0.5,
                drift_delta=0.01,
                emit_drift=True,
            ),
            id="drift-reset",
        ),
    ],
)
def test_what_is_not_order_free(spec):
    """``window`` lives *inside* the nested ``model`` dict, so a check that
    read only top-level keys would let a windowed spec through -- the worst
    case after a drift reset, which moves the coefficients by 8.9e-01."""
    from polars_online._frame import _order_free

    assert not _order_free([spec])


def test_an_unrecognised_spec_key_is_not_order_free():
    """The property that makes this safe to ship: a key in neither table fails
    the check, so an option added later cannot quietly become exempt."""
    from polars_online._frame import _order_free

    assert not _order_free([{**FREE, "an_option_added_later": 7}])


def test_one_disqualifying_spec_disqualifies_the_bank():
    from polars_online._frame import _order_free

    grad = po.spec.sgd("m2", targets=["y"], features=["x0"], lam=1.0, min_periods=1.0)
    assert not _order_free([FREE, grad])
    assert not _order_free([])


def test_fit_over_an_order_free_spec_says_nothing():
    """The false positive this removes: the plan's order is unspecified, and
    for this fit it cannot matter."""
    quiet(lambda: po.ModelBank([FREE]).fit(left().join(right(), on="k")))


def test_the_same_spec_still_warns_wherever_predictions_are_handed_back():
    """`fit` keeps only the state; everything else returns predictions, and a
    prediction is out-of-sample -- so reordering moves all of them."""
    lf = left().join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning):
        list(po.ModelBank([FREE]).fit_predict_batches(lf))
    with pytest.warns(po.OrderNotGuaranteedWarning):
        lf.online.fit_predict([FREE]).collect()


def test_a_disqualified_fit_still_warns():
    lf = left().join(right(), on="k")
    with pytest.warns(po.OrderNotGuaranteedWarning):
        po.ModelBank([SPEC]).fit(lf)


# --------------------------------------------- the premise, re-derived here so
# --------------------------------------------- the suppression cannot outlive it


def test_an_accumulators_coefficients_commute():
    """To rounding, never to the bit: the Gram sums commute mathematically but
    not in floating point. If this ever fails, the exception above is wrong."""
    assert _spread(FREE, _rows()) < 1e-12


def test_a_gradient_models_coefficients_do_not_commute_even_with_no_decay():
    """Why "no halflife" is not on its own a reason to expect order not to
    matter: this update is not commutative, and no decay does not change that."""
    grad = po.spec.sgd("m", targets=["y"], features=["x0"], lam=1.0, min_periods=1.0)
    assert _spread(grad, _rows()) > 1e-6


def test_predictions_move_even_where_the_coefficients_do_not():
    """The reason the exception stops at `fit`. Same spec, same rows: the
    coefficients agree to rounding and the predictions do not."""
    frame = _rows()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)

    def preds(f: pl.DataFrame) -> list[float]:
        out = po.ModelBank([FREE]).fit_predict(f)
        field = next(x for x in out["m"].struct.fields if x.startswith("pred"))
        return f.with_columns(p=out["m"].struct.field(field)).sort("t")["p"].to_list()

    a, b = preds(frame), preds(shuffled)
    both = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    assert both, "no row was predicted in both runs"
    assert max(abs(x - y) for x, y in both) > 1e-6
