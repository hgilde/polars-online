"""The refusals whose class changed to the one the problem calls for raise
that class alone (task 354 gave each a second parent, the class it raised
before, through 1.x; 2.0 drops it): each site raises its class, and an
``except`` for the one parent the problem names catches it. Task 355 carried
the rule to the Rust side's refusals of a column of the wrong type."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp
from rheon.polars import _gbdt

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
    return rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=1e9)


def _gram() -> Any:
    bank = rp.ModelBank([_spec()])
    bank.fit_predict(_df())
    return bank.gram("m")[0]


def _embargo_text_clock() -> None:
    rp.stream.embargo(_df().with_columns(t=pl.col("t").cast(pl.String)), clock="t", delay=1.0)


def _embargo_wide_clock() -> None:
    rp.stream.embargo(_df().with_columns(t=pl.col("t").cast(pl.Int128)), clock="t", delay=1)


def _refresh(clock: pl.DataType, value: pl.DataType) -> Callable[[], None]:
    def call() -> None:
        df = pl.DataFrame(
            {"series": ["a", "b"], "t": [1, 2], "v": [1.0, 2.0]},
        ).with_columns(pl.col("t").cast(clock), pl.col("v").cast(value))
        rp.stream.refresh_time(
            df.lazy(), series="series", names=["a", "b"], clock="t", value="v"
        ).collect()

    return call


def _uint128() -> None:
    rp.ModelBank([_spec()]).fit_predict(_df().with_columns(w=pl.lit(1, dtype=pl.UInt128)))


def _xgb(**kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        _gbdt.xgboost_matrix(_df().lazy(), label="y", **kwargs)

    return call


def _xgb_frame(df: pl.DataFrame, **kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        _gbdt.xgboost_matrix(df.lazy(), label="y", **kwargs)

    return call


def _grid(**kwargs: Any) -> Callable[[], None]:
    def call() -> None:
        rp.eval.grid_search(_df(), **kwargs)

    return call


def _builder(name: str, **kw: Any) -> dict[str, Any]:
    return rp.spec.ewridge(name, targets=["y"], features=["x0"], **kw)


def _rewm_without_window() -> None:
    rp.ewm_mean("x0", half_life=10.0, reverse=True)  # type: ignore[call-arg]


def _look_ahead_without_window() -> None:
    """The operator's own check, behind the public ones: no public call
    reaches it without a window today, so it is called directly."""
    from rheon.polars._formula import operator

    operator("rewm_mean", "x0", half_life=10.0)


def _like_clash() -> None:
    like = rp.spec.ewridge(
        "m", targets=["y"], features=["x0"], half_life=10.0, clock="t", gap_cap=5.0
    )
    rp.stream.with_windows(_df(), f=rp.ewm_mean("x0", half_life=5.0), like=like, clock="t")


def _bank(df: pl.DataFrame, **kw: Any) -> Callable[[], None]:
    """A bank of one ewridge spec fed ``df``: the Rust side's refusals."""

    def call() -> None:
        args: dict[str, Any] = {"targets": ["y"], "features": ["x0"], "half_life": 10.0, **kw}
        rp.ModelBank([rp.spec.ewridge("m", **args)]).fit_predict(df)

    return call


def _seconds(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(t=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s"))


def _time_of_day(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(t=(pl.col("t").cast(pl.Int64) * 1_000_000_000).cast(pl.Time))


_CLOCKED: dict[str, Any] = {"clock": "t", "gap_cap": 5.0}


def _windows(df: pl.DataFrame, expr: pl.Expr, **kw: Any) -> Callable[[], None]:
    def call() -> None:
        rp.stream.with_windows(df, f=expr, **kw)

    return call


def _ipc_second_file_text(tmp: Any) -> None:
    """Two Arrow IPC files whose second holds a feature as text: the native
    reader refuses it, the schema read from the first having passed."""
    a, b = tmp / "a.arrow", tmp / "b.arrow"
    _df().write_ipc(a)
    _df().with_columns(x0=pl.col("x0").cast(pl.String)).write_ipc(b)
    _gbdt.xgboost_matrix([str(a), str(b)], label="y", features=["x0"])


def _window_metrics(df: pl.DataFrame, every: Any) -> Callable[[], None]:
    def call() -> None:
        out = rp.ModelBank([_spec()]).fit_predict(_df())
        rp.eval.window_metrics(out.with_columns(t=df["t"]), "m", clock="t", every=every)

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
            _df().with_columns(x0=pl.col("x0").cast(pl.String)), rp.ewm_mean("x0", half_life=5.0)
        ),
    ),
    (
        # A text clock of digits ran, read through a cast to numbers (task 356).
        "with_windows, a text clock",
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.String)),
            rp.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ),
    (
        "with_windows, an Int128 clock",
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.Int128)),
            rp.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ),
    (
        "with_windows, a time-of-day clock",
        _windows(_time_of_day(_df()), rp.ewm_mean("x0", half_life="5s"), clock="t", gap_cap="5s"),
    ),
    ("refresh_time, an Int128 clock", _refresh(pl.Int128, pl.Float64)),
    ("refresh_time, a time-of-day clock", _refresh(pl.Time, pl.Float64)),
]


@pytest.mark.parametrize("call", [s[1] for s in RUST_SITES], ids=[s[0] for s in RUST_SITES])
def test_each_rust_side_wrong_type_raises_wrong_type_error(call: Callable[[], Any]) -> None:
    with pytest.raises(rp.WrongTypeError) as caught:
        call()
    assert type(caught.value) is rp.WrongTypeError
    with pytest.raises(TypeError):
        call()


def test_the_native_gbdt_reader_raises_wrong_type_error(tmp_path: Any) -> None:
    with pytest.raises(rp.WrongTypeError, match="is Utf8View in the Arrow IPC file") as caught:
        _ipc_second_file_text(tmp_path)
    assert type(caught.value) is rp.WrongTypeError


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
            rp.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ],
    "an Int128 clock": [
        _bank(_df().with_columns(t=pl.col("t").cast(pl.Int128)), **_CLOCKED),
        _embargo_wide_clock,
        _refresh(pl.Int128, pl.Float64),
        _windows(
            _df().with_columns(t=pl.col("t").cast(pl.Int128)),
            rp.ewm_mean("x0", half_life=5.0),
            **_CLOCKED,
        ),
    ],
    "a time-of-day clock": [
        _bank(_time_of_day(_df()), **_CLOCKED),
        lambda: rp.stream.embargo(_time_of_day(_df()), clock="t", delay="1s"),
        _refresh(pl.Time, pl.Float64),
        _windows(_time_of_day(_df()), rp.ewm_mean("x0", half_life="5s"), clock="t", gap_cap="5s"),
        _window_metrics(_time_of_day(_df()), "5s"),
    ],
}


@pytest.mark.parametrize("input_", list(_SAME_INPUT))
def test_python_and_rust_paths_refusing_the_same_clock_raise_one_class(input_: str) -> None:
    for call in _SAME_INPUT[input_]:
        with pytest.raises(rp.WrongTypeError) as caught:
            call()
        assert type(caught.value) is rp.WrongTypeError, caught.value


def test_a_text_clock_of_digits_is_refused_by_with_windows() -> None:
    """A String clock of digits ("0.0", "1.0", ...) ran under with_windows,
    read through a cast that turned the text into numbers, where every other
    path refuses a text clock by its type (task 356)."""
    df = _df().with_columns(t=pl.col("t").cast(pl.String))
    with pytest.raises(rp.WrongTypeError) as caught:
        rp.stream.with_windows(df, f=rp.ewm_mean("x0", half_life=5.0), clock="t", gap_cap=5.0)
    assert str(caught.value) == (
        'with_windows: clock column "t" must be a number or temporal, but it is String. '
        'Cast it, such as with pl.col("t").cast(pl.Float64), or to a Datetime.'
    )


def test_the_like_clash_names_the_function_in_full() -> None:
    pin = r"^rp\.stream\.with_windows: like= takes the clock settings from spec \"m\""
    with pytest.raises(rp.RefusedValueError, match=pin):
        _like_clash()


# (site, call, the class raised, then two classes the 1.x table gave it, kept as the record
# of the rule: the class raised before, and the class the rule gives)
SITES: list[tuple[str, Callable[[], Any], type[Exception], type[Exception], type[Exception]]] = [
    # A value or column of the wrong type: TypeError, once ValueError.
    (
        "ModelBank, one spec",
        lambda: rp.ModelBank(_spec()),  # type: ignore[arg-type]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "ModelBank, a str",
        lambda: rp.ModelBank("m"),  # type: ignore[arg-type]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    ("ModelBank, a UInt128 column", _uint128, rp.WrongTypeError, ValueError, TypeError),
    (
        "refresh_time, a text clock",
        _refresh(pl.String, pl.Float64),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "refresh_time, a text value",
        _refresh(pl.Int64, pl.String),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    # The embargo's two clock refusals raised TypeError; one class with the
    # bank's UInt128 refusal, so the ValueError parent is new there.
    ("embargo, a text clock", _embargo_text_clock, rp.WrongTypeError, ValueError, TypeError),
    ("embargo, an Int128 clock", _embargo_wide_clock, rp.WrongTypeError, ValueError, TypeError),
    (
        "window_metrics, a text clock",
        _window_metrics(_df().with_columns(t=pl.col("t").cast(pl.String)), 10.0),
        rp.WrongTypeError,
        TypeError,
        ValueError,
    ),
    (
        "embargo, a time-of-day clock",
        lambda: rp.stream.embargo(_time_of_day(_df()), clock="t", delay="1s"),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "window_metrics, a time-of-day clock",
        _window_metrics(_time_of_day(_df()), "5s"),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "window operator, min_samples=1.5",
        lambda: rp.ewm_mean("x0", half_life=5.0, window_size=10.0, min_samples=1.5),  # type: ignore[arg-type]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "grid_search, n_splits=1.5",
        _grid(specs=_spec(), n_splits=1.5),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "grid_search, gap=0.5",
        _grid(specs=_spec(), gap=0.5),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    ("xgboost_matrix, max_bin=2.5", _xgb(max_bin=2.5), rp.WrongTypeError, ValueError, TypeError),
    ("xgboost_matrix, threads=1.5", _xgb(threads=1.5), rp.WrongTypeError, ValueError, TypeError),
    (
        "xgboost_matrix, sample_rows=1.5",
        _xgb(sample_rows=1.5),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text feature",
        _xgb_frame(_df().with_columns(s=pl.lit("a"))),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a Categorical feature",
        _xgb_frame(_df().with_columns(s=pl.lit("a", dtype=pl.Categorical))),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text label",
        _xgb_frame(_df().with_columns(y=pl.col("y").cast(pl.String)), features=["x0"]),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "xgboost_matrix, a text margin",
        _xgb_frame(_df().with_columns(m=pl.lit("a")), features=["x0"], base_margin="m"),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "lightgbm_dataset, a float column as categorical",
        lambda: _gbdt.lightgbm_dataset(_df().lazy(), label="y", categorical=["x0"]),
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "lars_path, max_iter=1.5",
        lambda: rp.gram.lars_path(_gram(), max_iter=1.5),  # type: ignore[arg-type]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "solve_subsets, jobs not a list",
        lambda: rp.gram.solve_subsets([_gram()], [[0]], jobs=5),  # type: ignore[arg-type]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    (
        "solve_subsets, a job not a mapping",
        lambda: rp.gram.solve_subsets([_gram()], [[0]], jobs=[5]),  # type: ignore[list-item]
        rp.WrongTypeError,
        ValueError,
        TypeError,
    ),
    # A value out of range or a combination refused: ValueError, once TypeError.
    (
        "ewm_mean, reverse=True without window_size",
        lambda: rp.ewm_mean("x0", half_life=5.0, reverse=True),
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "ewm_mean(reverse=True) without window_size",
        _rewm_without_window,
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "a look-ahead operator without window_size",
        _look_ahead_without_window,
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    ("with_windows, like= and clock=", _like_clash, rp.RefusedValueError, TypeError, ValueError),
    (
        "grid_search, param_grid with built specs",
        _grid(specs=_spec(), param_grid={"half_life": [5.0]}),
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "grid_search, an empty param_grid",
        _grid(specs=_builder, param_grid={}),
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    (
        "grid_search, an empty list of values",
        _grid(specs=_builder, param_grid={"half_life": []}),
        rp.RefusedValueError,
        TypeError,
        ValueError,
    ),
    # A spec's output column the frame has not got: ColumnNotFoundError, once KeyError.
    (
        "unpack, no output column",
        lambda: rp.eval.unpack(_df(), "m"),
        rp.OutputColumnNotFoundError,
        KeyError,
        pl.exceptions.ColumnNotFoundError,
    ),
    (
        "seqtest, no output column",
        lambda: rp.eval.seqtest(_df(), a="m", b="n"),
        rp.OutputColumnNotFoundError,
        KeyError,
        pl.exceptions.ColumnNotFoundError,
    ),
]


@pytest.mark.parametrize(
    ("call", "new", "old", "rule"), [s[1:] for s in SITES], ids=[s[0] for s in SITES]
)
def test_each_site_raises_its_class_and_the_rules_class_catches_it(
    call: Callable[[], Any], new: type[Exception], old: type[Exception], rule: type[Exception]
) -> None:
    with pytest.raises(new) as caught:
        call()
    assert type(caught.value) is new
    with pytest.raises(new.__bases__[0]):
        call()


def test_wrong_kind_inputs_that_are_also_out_of_range_still_split_by_kind() -> None:
    """A whole number out of range stays a plain ``ValueError`` where the check
    for the type and the range was one condition."""
    with pytest.raises(ValueError) as caught:
        rp.ewm_mean("x0", half_life=5.0, window_size=10.0, min_samples=0)
    assert not isinstance(caught.value, TypeError)
    with pytest.raises(ValueError) as caught:
        rp.eval.grid_search(_df(), _spec(), n_splits=0)
    assert not isinstance(caught.value, TypeError)
    with pytest.raises(TypeError) as caught:
        rp.eval.grid_search(_df(), _builder, ["half_life"])  # type: ignore[arg-type]
    assert not isinstance(caught.value, ValueError)


def test_each_class_has_the_one_parent_its_problem_names_and_is_exported() -> None:
    assert rp.WrongTypeError.__bases__ == (TypeError,)
    assert not issubclass(rp.WrongTypeError, ValueError)
    assert rp.RefusedValueError.__bases__ == (ValueError,)
    assert not issubclass(rp.RefusedValueError, TypeError)
    assert issubclass(rp.OutputColumnNotFoundError, rp.ColumnNotFoundError)
    assert issubclass(rp.OutputColumnNotFoundError, pl.exceptions.ColumnNotFoundError)
    assert issubclass(rp.OutputColumnNotFoundError, ValueError)
    assert not issubclass(rp.OutputColumnNotFoundError, KeyError)
    for name in ("WrongTypeError", "RefusedValueError", "OutputColumnNotFoundError"):
        assert name in rp.__all__


def test_a_missing_output_column_prints_its_message_unquoted() -> None:
    """It is no ``KeyError``, whose ``str`` quotes the text."""
    with pytest.raises(rp.OutputColumnNotFoundError) as caught:
        rp.eval.unpack(_df(), "m")
    assert str(caught.value).startswith('rp.eval.unpack: spec "m" has no output column')
