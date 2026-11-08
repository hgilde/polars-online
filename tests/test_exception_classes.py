"""The refusals whose class changed to the one the problem calls for keep
the class they raised before as a second parent (task 354; the user chose
it, 2026-11-07): each site raises its new class, and an ``except`` for the
old class and one for the new both catch it. Task 355 carried the rule to
the Rust side's refusals of a column of the wrong type."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def _df(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": x,
            "y": 0.5 * x + rng.normal(scale=0.3, size=n),
        }
    )


def _spec() -> dict[str, Any]:
    return po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=1e9)


def _gram() -> Any:
    bank = po.ModelBank([_spec()])
    bank.fit_predict(_df())
    return bank.gram("m")[0]


def _embargo_text_clock() -> None:
    po.stream.embargo(_df().with_columns(t=pl.col("t").cast(pl.String)), clock="t", delay=1.0)


def _embargo_wide_clock() -> None:
    po.stream.embargo(_df().with_columns(t=pl.col("t").cast(pl.Int128)), clock="t", delay=1)


def _refresh(clock: pl.DataType, value: pl.DataType) -> Callable[[], None]:
    def call() -> None:
        df = pl.DataFrame(
            {"series": ["a", "b"], "t": [1, 2], "v": [1.0, 2.0]},
        ).with_columns(pl.col("t").cast(clock), pl.col("v").cast(value))
        po.stream.refresh_time(
            df.lazy(), series="series", names=["a", "b"], clock="t", value="v"
        ).collect()

    return call


def _uint128() -> None:
    po.ModelBank([_spec()]).fit_predict(_df().with_columns(w=pl.lit(1, dtype=pl.UInt128)))


def _xgb(**kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        po.gbdt.xgboost_matrix(_df().lazy(), label="y", **kwargs)

    return call


def _xgb_frame(df: pl.DataFrame, **kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        po.gbdt.xgboost_matrix(df.lazy(), label="y", **kwargs)

    return call


def _grid(**kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        po.eval.grid_search(_df(), **kwargs)

    return call


def _builder(name: str, **kw: Any) -> dict[str, Any]:
    return po.spec.ewridge(name, targets=["y"], features=["x0"], **kw)


def _rewm_without_window() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        po.rewm_mean("x0", half_life=10.0, window_size=None)  # type: ignore[arg-type]


def _look_ahead_without_window() -> None:
    """The operator's own check, behind the public ones: no public call
    reaches it without a window today, so it is called directly."""
    from polars_online._formula import operator

    operator("rewm_mean", "x0", half_life=10.0)


def _like_clash() -> None:
    like = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], half_life=10.0, clock="t", gap_cap=5.0
    )
    po.stream.with_windows(_df(), f=po.ewm_mean("x0", half_life=5.0), like=like, clock="t")


def _bank(df: pl.DataFrame, **kw: Any) -> Callable[[], None]:
    """A bank of one ewridge spec fed ``df``: the Rust side's refusals."""

    def call() -> None:
        args: dict[str, Any] = {"targets": ["y"], "features": ["x0"], "half_life": 10.0, **kw}
        po.ModelBank([po.spec.ewridge("m", **args)]).fit_predict(df)

    return call


def _seconds(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(t=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s"))


def _time_of_day(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(t=(pl.col("t").cast(pl.Int64) * 1_000_000_000).cast(pl.Time))


_CLOCKED: dict[str, Any] = {"clock": "t", "gap_cap": 5.0}


def _windows(df: pl.DataFrame, expr: pl.Expr, **kw: Any) -> Callable[[], None]:
    def call() -> None:
        po.stream.with_windows(df, f=expr, **kw)

    return call


def _ipc_second_file_text(tmp: Any) -> None:
    """Two Arrow IPC files whose second holds a feature as text: the native
    reader refuses it, the schema read from the first having passed."""
    a, b = tmp / "a.arrow", tmp / "b.arrow"
    _df().write_ipc(a)
    _df().with_columns(x0=pl.col("x0").cast(pl.String)).write_ipc(b)
    po.gbdt.xgboost_matrix([str(a), str(b)], label="y", features=["x0"])


def _window_metrics(df: pl.DataFrame, every: Any) -> Callable[[], None]:
    def call() -> None:
        out = po.ModelBank([_spec()]).fit_predict(_df())
        po.eval.window_metrics(out.with_columns(t=df["t"]), "m", clock="t", every=every)

    return call


# The Rust side's refusals of a column of the wrong type (task 355): each
# raised ValueError through the extension, and raises WrongTypeError now.
RUST_SITES: list[tuple[str, Callable[[], Any]]] = [
    ("bank, a text feature", _bank(_df().with_columns(x0=pl.col("x0").cast(pl.String)))),
    ("bank, a text target", _bank(_df().with_columns(y=pl.col("y").cast(pl.String)))),
    ("bank, a text weight", _bank(_df().with_columns(w=pl.lit("a")), weight="w")),
    ("bank, a text clock", _bank(_df().with_columns(t=pl.col("t").cast(pl.String)), **_CLOCKED)),
    ("bank, an Int128 clock", _bank(_df().with_columns(t=pl.col("t").cast(pl.Int128)), **_CLOCKED)),
    ("bank, a time-of-day clock", _bank(_time_of_day(_df()), **_CLOCKED)),
    (
        "bank, a temporal clock read as a feature",
        _bank(
            _seconds(_df()),
            features=["x0", "t"],
            clock="t",
            half_life="10s",
            gap_cap="5s",
        ),
    ),
    (
        "bank, a label_clock of another type than the clock",
        _bank(_df().with_columns(lc=pl.lit("a")), label_clock="lc", **_CLOCKED),
    ),
    (
        "bank, a nested group column",
        _bank(_df().with_columns(g=pl.concat_list("x0")), group="g"),
    ),
    (
        "bank, a float group under group_close='monotone'",
        _bank(_df().with_columns(g=pl.col("x0")), group="g", group_close="monotone"),
    ),
    (
        "with_windows, a text input",
        _windows(
            _df().with_columns(x0=pl.col("x0").cast(pl.String)), po.ewm_mean("x0", half_life=5.0)
        ),
    ),
    (
        # A text clock of digits ran, read through a cast to numbers (task 356).
        "with_windows, a text clock",
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.String)),
            po.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ),
    (
        "with_windows, an Int128 clock",
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.Int128)),
            po.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ),
    (
        "with_windows, a time-of-day clock",
        _windows(_time_of_day(_df()), po.ewm_mean("x0", half_life="5s"), clock="t", gap_cap="5s"),
    ),
    ("refresh_time, an Int128 clock", _refresh(pl.Int128, pl.Float64)),
    ("refresh_time, a time-of-day clock", _refresh(pl.Time, pl.Float64)),
]


@pytest.mark.parametrize("call", [s[1] for s in RUST_SITES], ids=[s[0] for s in RUST_SITES])
def test_each_rust_side_wrong_type_raises_wrong_type_error(call: Callable[[], Any]) -> None:
    with pytest.raises(po.WrongTypeError) as caught:
        call()
    assert type(caught.value) is po.WrongTypeError
    with pytest.raises(ValueError):
        call()
    with pytest.raises(TypeError):
        call()


def test_the_native_gbdt_reader_raises_wrong_type_error(tmp_path: Any) -> None:
    with pytest.raises(po.WrongTypeError, match="is Utf8View in the Arrow IPC file") as caught:
        _ipc_second_file_text(tmp_path)
    assert type(caught.value) is po.WrongTypeError


def test_a_rust_side_refusal_for_another_reason_stays_a_value_error() -> None:
    """A duration on a number clock is a combination refused, and a cap
    finer than the clock's step a value out of range: neither is a type."""
    for call in (
        _bank(_df(), clock="t", half_life="10s", gap_cap="5s"),
        _bank(_seconds(_df()), clock="t", half_life="10s", gap_cap="1ns"),
    ):
        with pytest.raises(ValueError) as caught:
            call()
        assert not isinstance(caught.value, TypeError), caught.value


# Every path that refuses the same clock raises the same class (task 355):
# the bank's, the window run's and refresh_time's in Rust, and embargo's
# and window_metrics' in Python.
_SAME_INPUT: dict[str, list[Callable[[], Any]]] = {
    "a text clock": [
        _bank(_df().with_columns(t=pl.col("t").cast(pl.String)), **_CLOCKED),
        _embargo_text_clock,
        _refresh(pl.String, pl.Float64),
        _window_metrics(_df().with_columns(t=pl.col("t").cast(pl.String)), 10.0),
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.String)),
            po.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ],
    "an Int128 clock": [
        _bank(_df().with_columns(t=pl.col("t").cast(pl.Int128)), **_CLOCKED),
        _embargo_wide_clock,
        _refresh(pl.Int128, pl.Float64),
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.Int128)),
            po.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ],
    "a time-of-day clock": [
        _bank(_time_of_day(_df()), **_CLOCKED),
        lambda: po.stream.embargo(_time_of_day(_df()), clock="t", delay="1s"),
        _refresh(pl.Time, pl.Float64),
        _windows(_time_of_day(_df()), po.ewm_mean("x0", half_life="5s"), clock="t", gap_cap="5s"),
        _window_metrics(_time_of_day(_df()), "5s"),
    ],
}


@pytest.mark.parametrize("input_", list(_SAME_INPUT))
def test_python_and_rust_paths_refusing_the_same_clock_raise_one_class(input_: str) -> None:
    for call in _SAME_INPUT[input_]:
        with pytest.raises(po.WrongTypeError) as caught:
            call()
        assert type(caught.value) is po.WrongTypeError, caught.value


def test_a_text_clock_of_digits_is_refused_by_with_windows() -> None:
    """A String clock of digits ("0.0", "1.0", ...) ran under with_windows,
    read through a cast that turned the text into numbers, where every other
    path refuses a text clock by its type (task 356)."""
    df = _df().with_columns(t=pl.col("t").cast(pl.String))
    with pytest.raises(po.WrongTypeError) as caught:
        po.stream.with_windows(df, f=po.ewm_mean("x0", half_life=5.0), clock="t", gap_cap=5.0)
    assert str(caught.value) == (
        'with_windows: clock column "t" must be a number or temporal, but it is String. '
        'Cast it, such as with pl.col("t").cast(pl.Float64), or to a Datetime.'
    )


def test_the_like_clash_names_the_function_in_full() -> None:
    pin = r"^po\.stream\.with_windows: like= takes the clock settings from spec \"m\""
    with pytest.raises(po.RefusedValueError, match=pin):
        _like_clash()


# (site, call, the new class, the class it raised before, the rule's class)
SITES: list[tuple[str, Callable[[], Any], type[Exception], type[Exception], type[Exception]]] = [
    # A value or column of the wrong type: TypeError, once ValueError.
    (
        "ModelBank, one spec",
        lambda: po.ModelBank(_spec()),  # type: ignore[arg-type]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "ModelBank, a str",
        lambda: po.ModelBank("m"),  # type: ignore[arg-type]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    ("ModelBank, a UInt128 column", _uint128, po.WrongTypeError, ValueError, TypeError),
    (
        "refresh_time, a text clock",
        _refresh(pl.String, pl.Float64),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "refresh_time, a text value",
        _refresh(pl.Int64, pl.String),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    # The embargo's two clock refusals raised TypeError; one class with the
    # bank's UInt128 refusal, so the ValueError parent is new there.
    ("embargo, a text clock", _embargo_text_clock, po.WrongTypeError, TypeError, ValueError),
    ("embargo, an Int128 clock", _embargo_wide_clock, po.WrongTypeError, TypeError, ValueError),
    (
        "window_metrics, a text clock",
        _window_metrics(_df().with_columns(t=pl.col("t").cast(pl.String)), 10.0),
        po.WrongTypeError,
        TypeError,
        ValueError,
    ),
    (
        "embargo, a time-of-day clock",
        lambda: po.stream.embargo(_time_of_day(_df()), clock="t", delay="1s"),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "window_metrics, a time-of-day clock",
        _window_metrics(_time_of_day(_df()), "5s"),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "window operator, min_samples=1.5",
        lambda: po.ewm_mean("x0", half_life=5.0, window_size=10.0, min_samples=1.5),  # type: ignore[arg-type]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "grid_search, n_splits=1.5",
        _grid(specs=_spec(), n_splits=1.5),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "grid_search, gap=0.5",
        _grid(specs=_spec(), gap=0.5),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    ("xgboost_matrix, max_bin=2.5", _xgb(max_bin=2.5), po.WrongTypeError, ValueError, TypeError),
    ("xgboost_matrix, threads=1.5", _xgb(threads=1.5), po.WrongTypeError, ValueError, TypeError),
    (
        "xgboost_matrix, sample_rows=1.5",
        _xgb(sample_rows=1.5),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text feature",
        _xgb_frame(_df().with_columns(s=pl.lit("a"))),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a Categorical feature",
        _xgb_frame(_df().with_columns(s=pl.lit("a", dtype=pl.Categorical))),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text label",
        _xgb_frame(_df().with_columns(y=pl.col("y").cast(pl.String)), features=["x0"]),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text margin",
        _xgb_frame(_df().with_columns(m=pl.lit("a")), features=["x0"], base_margin="m"),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "lightgbm_dataset, a float column as categorical",
        lambda: po.gbdt.lightgbm_dataset(_df().lazy(), label="y", categorical=["x0"]),
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "lars_path, max_iter=1.5",
        lambda: po.gram.lars_path(_gram(), max_iter=1.5),  # type: ignore[arg-type]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "solve_subsets, jobs not a list",
        lambda: po.gram.solve_subsets([_gram()], [[0]], jobs=5),  # type: ignore[arg-type]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "solve_subsets, a job not a mapping",
        lambda: po.gram.solve_subsets([_gram()], [[0]], jobs=[5]),  # type: ignore[list-item]
        po.WrongTypeError,
        ValueError,
        TypeError,
    ),
    # A value out of range or a combination refused: ValueError, once TypeError.
    (
        "ewm_mean, reverse=True without window_size",
        lambda: po.ewm_mean("x0", half_life=5.0, reverse=True),
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "rewm_mean without window_size",
        _rewm_without_window,
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "a look-ahead operator without window_size",
        _look_ahead_without_window,
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    ("with_windows, like= and clock=", _like_clash, po.RefusedValueError, TypeError, ValueError),
    (
        "grid_search, param_grid with built specs",
        _grid(specs=_spec(), param_grid={"half_life": [5.0]}),
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "grid_search, an empty param_grid",
        _grid(specs=_builder, param_grid={}),
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "grid_search, an empty list of values",
        _grid(specs=_builder, param_grid={"half_life": []}),
        po.RefusedValueError,
        TypeError,
        ValueError,
    ),
    # A spec's output column the frame has not got: ColumnNotFoundError, once KeyError.
    (
        "unpack, no output column",
        lambda: po.eval.unpack(_df(), "m"),
        po.OutputColumnNotFoundError,
        KeyError,
        pl.exceptions.ColumnNotFoundError,
    ),
    (
        "seqtest, no output column",
        lambda: po.eval.seqtest(_df(), a="m", b="n"),
        po.OutputColumnNotFoundError,
        KeyError,
        pl.exceptions.ColumnNotFoundError,
    ),
]


@pytest.mark.parametrize(
    ("call", "new", "old", "rule"), [s[1:] for s in SITES], ids=[s[0] for s in SITES]
)
def test_each_site_raises_its_class_and_both_parents_catch_it(
    call: Callable[[], Any], new: type[Exception], old: type[Exception], rule: type[Exception]
) -> None:
    with pytest.raises(new) as caught:
        call()
    assert type(caught.value) is new
    with pytest.raises(old):
        call()
    with pytest.raises(rule):
        call()


def test_wrong_kind_inputs_that_are_also_out_of_range_still_split_by_kind() -> None:
    """A whole number out of range stays a plain ``ValueError`` where the check
    for the type and the range was one condition."""
    with pytest.raises(ValueError) as caught:
        po.ewm_mean("x0", half_life=5.0, window_size=10.0, min_samples=0)
    assert not isinstance(caught.value, TypeError)
    with pytest.raises(ValueError) as caught:
        po.eval.grid_search(_df(), _spec(), n_splits=0)
    assert not isinstance(caught.value, TypeError)
    with pytest.raises(TypeError) as caught:
        po.eval.grid_search(_df(), _builder, ["half_life"])  # type: ignore[arg-type]
    assert not isinstance(caught.value, ValueError)


def test_each_class_has_both_parents_and_is_exported() -> None:
    assert issubclass(po.WrongTypeError, TypeError)
    assert issubclass(po.WrongTypeError, ValueError)
    assert issubclass(po.RefusedValueError, ValueError)
    assert issubclass(po.RefusedValueError, TypeError)
    assert issubclass(po.OutputColumnNotFoundError, po.ColumnNotFoundError)
    assert issubclass(po.OutputColumnNotFoundError, pl.exceptions.ColumnNotFoundError)
    assert issubclass(po.OutputColumnNotFoundError, ValueError)
    assert issubclass(po.OutputColumnNotFoundError, KeyError)
    # The class the problem calls for comes first in the MRO, so it is the
    # one a reader of `type(e).__mro__` meets first.
    assert po.WrongTypeError.__mro__[1] is TypeError
    assert po.RefusedValueError.__mro__[1] is ValueError
    for name in ("WrongTypeError", "RefusedValueError", "OutputColumnNotFoundError"):
        assert name in po.__all__


def test_a_missing_output_column_keeps_its_message() -> None:
    """``KeyError``'s own ``str`` quotes the text, as it did before."""
    with pytest.raises(po.OutputColumnNotFoundError) as caught:
        po.eval.unpack(_df(), "m")
    assert str(caught.value).startswith('\'po.eval.unpack: spec "m" has no output column')
