"""Every exception type the library raises is raised by a test through a
public call, with its exact class, and a test holds the list complete.

Three parts, each of which fails with the place to add what is missing:

1. **The cases** (``CASES``): one public call for each class the library
   raises, each asserting ``type(e) is X`` -- not ``isinstance``, which a
   parent satisfies -- the parents a class keeps beside the one the rule gives
   (``WrongTypeError`` is a ``TypeError`` and a ``ValueError``), and the
   ``errno`` where the call carries one. An error that reaches Python as a
   plain ``OSError`` where its kind has a subclass, or as ``ValueError`` where
   a typed class is meant, fails here. A rule for each I/O kind, by platform,
   is in the I/O section.
2. **The inventory** (``PY_CLASSES``, ``IO_KINDS``, ``CLI_EXITS``): each
   class, kind and exit code, with the cases that raise it, the tests
   elsewhere that raise it too, or the reason no public call can.
3. **The completeness checks**: every ``raise`` in ``python/polars_online``
   (read with ``ast``), every class the package builds that is raised
   elsewhere, every exception class and every ``io::ErrorKind`` in the Rust
   source (read by pattern, see ``_rust_exceptions``) and every exit code of
   the command line names a class the inventory lists. A new raise site,
   class or mapping cannot land without an entry here, and an entry needs a
   case.

An encode-time failure (a state that cannot be written, as against a file
that cannot) is a ``ValueError`` here; the Rust side's counterpart, which
holds that the filesystem's kind survives an error raised while a save
encodes, is the task 388 branch's test in ``online-polars``.
"""

from __future__ import annotations

import ast
import errno as errno_codes
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _polars_online as native
from polars_online._formula import FormulaError

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"
ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# --- The fixtures every case draws on -----------------------------------------


def _df(n: int = 50) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {"t": np.arange(float(n)), "x0": rng.normal(size=n), "y": rng.normal(size=n)}
    )


def _spec(name: str = "m", **kw: Any) -> dict[str, Any]:
    kw.setdefault("half_life", 50.0)
    return po.spec.ewridge(name, targets=["y"], features=["x0"], **kw)


def _closing() -> tuple[dict[str, Any], list[pl.DataFrame]]:
    """A spec that closes groups, and six batches in which each closes."""
    spec = po.spec.ew_cov(
        "c", features=["x0", "y"], half_life=40.0, group="g", group_close="monotone"
    )
    keys = [f"k{i:02d}" for i in range(12)]
    df = pl.DataFrame(
        {
            "g": [k for k in keys for _ in range(5)],
            "x0": [float((i * 7) % 11) for i in range(60)],
            "y": [float((i * 5) % 13) for i in range(60)],
        }
    )
    return spec, [df.slice(i, 10) for i in range(0, df.height, 10)]


def _broken_bank() -> po.ModelBank:
    """A bank a refused chunk broke: the budget refuses after the bank has
    learned part of the chunk, so its state holds rows never returned."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300)
    df = pl.DataFrame({"x0": x, "y": 2 * x + 0.1 * rng.standard_normal(300)})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        half_life=1e9,
        window_size=50.0,
        snapshot_every=1,
        min_weight=0.0,
        window_budget={"refuse": 0.001},
        emit_drift=True,
        drift_action="reset",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(1))
    with pytest.raises(ValueError, match="window_budget"):
        bank.fit_predict(df.slice(1))
    return bank


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """What a case's call reads: a frame, a fitted bank, a scratch folder."""
    df = _df()
    bank = po.ModelBank([_spec()])
    bank.fit_predict(df)
    return SimpleNamespace(df=df, bank=bank, tmp=tmp_path, monkeypatch=monkeypatch)


# --- The cases ----------------------------------------------------------------

_ANY: Any = object()


@dataclass(frozen=True)
class Case:
    """One public call and what it raises.

    ``raises`` is the exact class (``type(e) is raises``), ``bases`` the
    parents the class keeps, ``text`` a fragment of the message (so the case
    cannot pass on another refusal of the same class), and ``errno`` the
    ``OSError``'s errno where the call carries one: ``None`` says it is not
    carried (an error built by the Rust side, where pyo3 makes the
    ``OSError`` from the message alone) and ``_ANY`` says it is not asserted.
    """

    id: str
    call: Callable[[SimpleNamespace], object]
    raises: type[BaseException]
    text: str
    bases: tuple[type, ...] = ()
    errno: Any = _ANY
    guard: tuple[bool, str] = (False, "")  # (skip?, the explained platform guard)


def _no_numpy(w: SimpleNamespace) -> None:
    """numpy not installed: a ``None`` in ``sys.modules`` is how Python says an
    import is blocked, and raises ``ModuleNotFoundError`` for it."""
    w.monkeypatch.setitem(sys.modules, "numpy", None)


def _gram_call(w: SimpleNamespace) -> object:
    _no_numpy(w)
    return w.bank.gram("m")


def _po_gram_call(w: SimpleNamespace) -> object:
    _no_numpy(w)
    return po.gram.condition({"columns": ["a"]})


def _po_corr_call(w: SimpleNamespace) -> object:
    _no_numpy(w)
    return po.corr.to_z(0.5)


def _po_sim_call(w: SimpleNamespace) -> object:
    _no_numpy(w)
    return po.sim.regimes(2, states=[0.5], transition=[[1.0]], n_blocks=2, rows_per_block=5, seed=0)


def _stub_native(w: SimpleNamespace) -> object:
    """A bank whose native layer answers with the wrong number of
    coefficients, the one defect ``coef`` guards against."""

    class Native:
        def coef(self, *a: Any, **k: Any) -> list[Any]:
            return [("", "i", 1.0, [1.0, 2.0, 3.0])]

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    real = w.bank._native
    w.monkeypatch.setattr(w.bank, "_native", Native())
    return w.bank.coef("m")


def _calendar_every_that_is_no_duration(w: SimpleNamespace) -> object:
    out = w.bank.fit_predict(w.df).with_columns(
        t=pl.datetime_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 1) + timedelta(days=w.df.height - 1),
            "1d",
            eager=True,
        )
    )
    return po.eval.window_metrics(out, "m", clock="t", every="1mo foo")


def _unnest_twice(w: SimpleNamespace) -> object:
    other = _spec("n", half_life=40.0)
    nested = po.ModelBank([_spec(), other]).fit_predict(w.df)
    return nested.online.unnest([_spec(), other])


def _save_state_missing_folder(w: SimpleNamespace) -> object:
    plan = w.df.lazy().online.fit_predict([_spec()], save_state=w.tmp / "nope" / "s.state")
    return plan.collect()


def _save_state_folder(w: SimpleNamespace) -> object:
    return w.df.lazy().online.fit_predict([_spec()], save_state=w.tmp).collect()


def _closed_missing_folder(w: SimpleNamespace) -> object:
    spec, batches = _closing()
    path = w.tmp / "nope" / "closed.parquet"
    return list(po.ModelBank([spec]).fit_predict_batches(batches, closed_groups=path))


def _closed_csv_is_a_folder(w: SimpleNamespace) -> object:
    """The runner's own CSV writer is the one that reports a path it cannot
    replace (the Parquet writer is Polars', and raises its own)."""
    spec, batches = _closing()
    target = w.tmp / "closed.csv"
    target.mkdir()
    return list(po.ModelBank([spec]).fit_predict_batches(batches, closed_groups=target))


def _gbdt_glob(w: SimpleNamespace) -> object:
    return po.gbdt.xgboost_matrix(str(w.tmp / "none*.parquet"), label="y")


def _gbdt_label(w: SimpleNamespace) -> object:
    return po.gbdt.xgboost_matrix(w.df.lazy(), label="nope")


def _file(w: SimpleNamespace) -> Path:
    path = w.tmp / "plain-file"
    path.write_text("not a folder", encoding="utf-8")
    return path


def _read_only_folder(w: SimpleNamespace) -> Path:
    folder = w.tmp / "read-only"
    folder.mkdir()
    folder.chmod(0o500)
    return folder


def _the_one_that(win: type[BaseException], elsewhere: type[BaseException]) -> type[BaseException]:
    """The class the platform gives for opening or replacing a folder as a
    file: ``PermissionError`` on Windows (it refuses the open), the kind's
    own ``IsADirectoryError`` everywhere else."""
    return win if WINDOWS else elsewhere


CASES: list[Case] = [
    # -- the classes the library names: a spec, a value, a call ----------------
    Case(
        "value-out-of-range",
        lambda w: _spec(half_life=-1.0),
        ValueError,
        "half_life must be above 0",
    ),
    Case(
        "value-several-problems",
        lambda w: _spec(half_life=-1.0, ridge=-1.0),
        ValueError,
        "2 problems",
    ),
    Case(
        "value-state-is-not-a-state",
        lambda w: po.ModelBank.load_bytes(b"not a bank"),
        ValueError,
        "not a polars-online bank state file",
    ),
    Case(
        "value-a-file-that-is-not-a-state",
        lambda w: po.ModelBank.load(_garbage(w)),
        ValueError,
        "not a polars-online bank state file",
    ),
    Case(
        "value-a-broken-bank-does-not-save",
        lambda w: _broken_bank().save(w.tmp / "b.state"),
        ValueError,
        "cannot go on",
    ),
    Case(
        "value-arrow-struct-exports-once",
        lambda w: _export_twice(w),
        ValueError,
        "already been exported",
    ),
    Case(
        "value-native-config-is-not-json",
        lambda w: native.Windows("{", pl.DataFrame()),
        ValueError,
        "invalid windows",
    ),
    Case(
        "value-threads-too-few",
        lambda w: po.ModelBank([_spec()], threads=0),
        ValueError,
        "is not a number of threads",
    ),
    Case(
        "value-threads-too-many",
        lambda w: po.ModelBank([_spec()], threads=10**9),
        ValueError,
        "more threads than one pool may have",
    ),
    Case(
        "value-gbdt-label-is-also-a-margin",
        lambda w: po.gbdt.xgboost_matrix(w.df.lazy(), label="y", base_margin="y"),
        ValueError,
        "is named as the base_margin and as the label",
    ),
    Case(
        "value-calendar-every-is-not-polars-text",
        _calendar_every_that_is_no_duration,
        ValueError,
        "every must be a duration above 0",
    ),
    Case(
        "type-a-frame-is-needed",
        lambda w: w.bank.fit_predict(5),
        TypeError,
        "takes a Polars DataFrame",
    ),
    Case(
        "type-a-parameter-of-the-wrong-kind",
        lambda w: _spec(ridge="a"),
        TypeError,
        "ridge must be a number",
    ),
    Case(
        "type-threads-is-a-bool",
        lambda w: po.ModelBank([_spec()], threads=True),
        TypeError,
        "is not a number of threads",
    ),
    Case(
        "type-gbdt-features-is-one-name",
        lambda w: po.gbdt.xgboost_matrix(w.df.lazy(), label="y", features="x0"),
        TypeError,
        "features must be a list of column names",
    ),
    Case(
        "wrong-type-python-one-spec",
        lambda w: po.ModelBank(_spec()),
        po.WrongTypeError,
        "specs must be a list of specs",
        bases=(TypeError, ValueError),
    ),
    Case(
        "wrong-type-rust-text-feature",
        lambda w: w.bank.fit_predict(w.df.with_columns(x0=pl.col("x0").cast(pl.String))),
        po.WrongTypeError,
        "must be a number, but it is String",
        bases=(TypeError, ValueError),
    ),
    Case(
        "wrong-type-rust-refresh-clock",
        lambda w: po.stream.refresh_time(
            pl.DataFrame({"series": ["a"], "t": ["x"], "v": [1.0]}).lazy(),
            series="series",
            names=["a"],
            clock="t",
            value="v",
        ).collect(),
        po.WrongTypeError,
        "must be numeric or temporal",
        bases=(TypeError, ValueError),
    ),
    Case(
        "wrong-type-gbdt-feature",
        lambda w: po.gbdt.xgboost_matrix(
            w.df.with_columns(x0=pl.col("x0").cast(pl.String)).lazy(), label="y"
        ),
        po.WrongTypeError,
        "is String",
        bases=(TypeError, ValueError),
    ),
    Case(
        "refused-value-reverse-without-a-window",
        lambda w: po.ewm_mean("x", half_life=10.0, reverse=True),
        po.RefusedValueError,
        "reverse",
        bases=(ValueError, TypeError),
    ),
    Case(
        "column-not-found-rust-feature",
        lambda w: w.bank.fit_predict(w.df.drop("x0")),
        po.ColumnNotFoundError,
        'feature column "x0" is not in the input',
        bases=(pl.exceptions.ColumnNotFoundError, ValueError),
    ),
    Case(
        "column-not-found-python-clock",
        lambda w: po.stream.embargo(w.df, clock="zz", delay=1.0),
        po.ColumnNotFoundError,
        "has no clock column 'zz'",
        bases=(pl.exceptions.ColumnNotFoundError, ValueError),
    ),
    Case(
        "column-not-found-scoring-group",
        lambda w: po.eval.metrics(w.df, "m", group="zz"),
        po.ColumnNotFoundError,
        "group column 'zz' is not in the frame",
        bases=(pl.exceptions.ColumnNotFoundError, ValueError),
    ),
    Case(
        "column-not-found-gbdt-label",
        _gbdt_label,
        po.ColumnNotFoundError,
        "the label column 'nope' is not in the source",
        bases=(pl.exceptions.ColumnNotFoundError, ValueError),
    ),
    Case(
        "output-column-not-found",
        lambda w: po.eval.metrics(w.df, "m"),
        po.OutputColumnNotFoundError,
        'spec "m" has no output column',
        bases=(po.ColumnNotFoundError, pl.exceptions.ColumnNotFoundError, ValueError, KeyError),
    ),
    Case(
        "formula-a-shift-reads-other-rows",
        lambda w: po.ewm_mean(pl.col("x").shift(1), half_life=5.0),
        FormulaError,
        "holds the method shift",
        bases=(ValueError,),
    ),
    Case(
        "key-error-no-such-spec",
        lambda w: w.bank.coef("nope"),
        KeyError,
        "has no spec named 'nope'",
    ),
    Case(
        "key-error-gram-without-columns",
        lambda w: po.gram.condition({}),
        KeyError,
        "a Gram must name its columns",
    ),
    Case(
        "index-error-spec-position",
        lambda w: w.bank.coef(7),
        IndexError,
        "spec index 7 is out of range",
    ),
    Case(
        "attribute-error-a-renamed-method",
        lambda w: w.bank.rows_seen(),
        AttributeError,
        "it was renamed rows_fed",
    ),
    Case(
        "assertion-error-the-model-gave-the-wrong-count",
        _stub_native,
        AssertionError,
        "a defect in polars-online",
    ),
    Case(
        "module-not-found-bank-gram",
        _gram_call,
        ModuleNotFoundError,
        "numpy is not installed",
        bases=(ImportError,),
    ),
    Case(
        "module-not-found-po-gram",
        _po_gram_call,
        ModuleNotFoundError,
        "numpy is not installed",
        bases=(ImportError,),
    ),
    Case(
        "module-not-found-po-corr",
        _po_corr_call,
        ModuleNotFoundError,
        "numpy is not installed",
        bases=(ImportError,),
    ),
    Case(
        "module-not-found-po-sim",
        _po_sim_call,
        ModuleNotFoundError,
        "numpy is not installed",
        bases=(ImportError,),
    ),
    Case(
        "duplicate-error-unnest-shares-a-name",
        _unnest_twice,
        pl.exceptions.DuplicateError,
        "so the columns unnested would share a name",
    ),
    # -- a path the Python layer checks while the plan is built ---------------
    Case(
        "file-not-found-save-state-folder",
        _save_state_missing_folder,
        FileNotFoundError,
        "save_state names a file in the folder",
        errno=None,
    ),
    Case(
        "file-not-found-closed-groups-folder",
        _closed_missing_folder,
        FileNotFoundError,
        "closed_groups names a file in the folder",
        errno=None,
    ),
    Case(
        "file-not-found-gbdt-glob",
        _gbdt_glob,
        FileNotFoundError,
        "no file matches the pattern",
        errno=None,
    ),
    Case(
        "is-a-directory-save-state-is-a-folder",
        _save_state_folder,
        IsADirectoryError,
        "is a folder, not a file",
        errno=None,
    ),
]

# --- The I/O kinds: each filesystem outcome and the class it reaches Python as.
#
# Python opens a state file to load it, so the OSError is Python's own, with
# its errno; the savers are Rust's, which builds the OSError from the kind and
# the message, so pyo3 gives the right subclass and no errno. A save is asked
# for each kind the filesystem gives; the platform's own class is pinned where
# the platforms differ (``_the_one_that``), never one for both.

CASES += [
    Case(
        "io-load-not-found",
        lambda w: po.ModelBank.load(w.tmp / "nope.state"),
        FileNotFoundError,
        "nope.state",
        errno=errno_codes.ENOENT,
    ),
    Case(
        "io-load-a-folder",
        lambda w: po.ModelBank.load(w.tmp),
        _the_one_that(PermissionError, IsADirectoryError),
        "",
        errno=errno_codes.EACCES if WINDOWS else errno_codes.EISDIR,
    ),
    Case(
        "io-load-under-a-file",
        lambda w: po.ModelBank.load(_file(w) / "x.state"),
        FileNotFoundError if WINDOWS else NotADirectoryError,
        "x.state",
        errno=errno_codes.ENOENT if WINDOWS else errno_codes.ENOTDIR,
    ),
    Case(
        "io-save-not-found",
        lambda w: w.bank.save(w.tmp / "nope" / "b.state"),
        FileNotFoundError,
        "nope",
        errno=None,
    ),
    Case(
        "io-save-onto-a-folder",
        lambda w: w.bank.save(w.tmp),
        _the_one_that(PermissionError, IsADirectoryError),
        "(os error",
        errno=None,
    ),
    Case(
        "io-save-under-a-file",
        lambda w: w.bank.save(_file(w) / "b.state"),
        FileNotFoundError if WINDOWS else NotADirectoryError,
        "b.state",
        errno=None,
    ),
    Case(
        "io-save-permission-denied",
        lambda w: w.bank.save(_read_only_folder(w) / "b.state"),
        PermissionError,
        "Permission denied",
        errno=None,
        guard=(
            WINDOWS or ROOT,
            "a folder's write bit stops no one on Windows, and nothing for root: there "
            "the call is the filesystem's, and Windows' PermissionError is the folder case",
        ),
    ),
    Case(
        "io-save-no-file-name",
        lambda w: w.bank.save(Path(w.tmp.anchor)),
        OSError,
        "not a file path",
        errno=None,
    ),
    Case(
        "io-closed-groups-csv-is-a-folder",
        _closed_csv_is_a_folder,
        _the_one_that(PermissionError, IsADirectoryError),
        "closed.csv",
        errno=None,
    ),
]


def _garbage(w: SimpleNamespace) -> Path:
    path = w.tmp / "garbage.state"
    path.write_bytes(b"not a bank")
    return path


def _export_twice(w: SimpleNamespace) -> object:
    struct = w.bank.fit_predict_arrow(w.df)[0]
    struct.__arrow_c_array__()
    return struct.__arrow_c_array__()


def _id(case: Case) -> str:
    return case.id


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_a_public_call_raises_exactly_its_class(case: Case, world: SimpleNamespace) -> None:
    skip, why = case.guard
    if skip:
        pytest.skip(why)
    with pytest.raises(BaseException) as caught:
        case.call(world)
    e = caught.value
    assert type(e) is case.raises, (
        f"{case.id}: raised {type(e).__module__}.{type(e).__qualname__}, "
        f"not exactly {case.raises.__module__}.{case.raises.__qualname__}: {e}"
    )
    for base in case.bases:
        assert isinstance(e, base), f"{case.id}: not a {base.__name__}"
    assert case.text in str(e), f"{case.id}: {case.text!r} is not in {str(e)!r}"
    if isinstance(e, OSError) and case.errno is not _ANY:
        assert e.errno == case.errno, f"{case.id}: errno {e.errno}, expected {case.errno}"


def test_every_case_has_a_distinct_id() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


# --- Two calls a case cannot make as one expression -----------------------------


def test_a_call_while_a_fit_runs_is_refused_with_exactly_runtime_error() -> None:
    """The GIL is released for ``fit_predict``, so a second thread can reach
    the bank: a read of it from this thread, while the fit runs, is refused.
    Polled until it is, and run again if the fit ends first (five rounds, or
    the test tested nothing)."""
    n = 400_000
    big = pl.DataFrame(
        {
            "x0": np.random.default_rng(0).standard_normal(n),
            "y": np.random.default_rng(1).standard_normal(n),
        }
    )
    for _ in range(5):
        bank = po.ModelBank([_spec(half_life=[10.0, 100.0, 1000.0], coef_every=1)])
        small = big.head(5)
        done = threading.Event()

        def fit(bank: po.ModelBank = bank, done: threading.Event = done) -> None:
            bank.fit_predict(big)
            done.set()

        worker = threading.Thread(target=fit)
        worker.start()
        refused: BaseException | None = None
        while not done.is_set() and refused is None:
            try:
                bank.predict(small)
            except RuntimeError as e:
                refused = e
            except ValueError:  # the bank has learned nothing yet
                pass
            time.sleep(0.001)
        worker.join()
        if refused is not None:
            break
    else:
        pytest.fail("no call reached the bank while a fit of 400,000 rows ran")
    assert type(refused) is RuntimeError
    assert "in use on another thread" in str(refused)


def test_the_savers_refuse_a_state_that_will_not_encode_as_a_value_error_not_an_os_error(
    tmp_path: Path,
) -> None:
    """The savers hand the Rust side's one refusal over as ``Other`` or
    ``InvalidData`` and every filesystem error as its own kind, and the
    Python class follows the kind: a refusal is never an ``OSError``, and the
    file a save would replace is left as it was."""
    bank = _broken_bank()
    state = tmp_path / "b.state"
    state.write_bytes(b"the last good state")
    with pytest.raises(ValueError, match="cannot go on") as caught:
        bank.save(state)
    assert type(caught.value) is ValueError
    assert state.read_bytes() == b"the last good state"
    # A filesystem's refusal beside it, from the same method, is the OSError.
    with pytest.raises(OSError) as caught_os:
        po.ModelBank([_spec()]).save(tmp_path / "nope" / "b.state")
    assert type(caught_os.value) is FileNotFoundError


# --- The classes this library defines -----------------------------------------


def test_the_library_s_own_classes_keep_their_parents_in_the_rule_s_order() -> None:
    """The rule's class first in each MRO, the class the refusal raised
    before second (the older parent goes at the next major version), and each
    class exported under its own name."""

    assert po.WrongTypeError.__bases__ == (TypeError, ValueError)
    assert po.RefusedValueError.__bases__ == (ValueError, TypeError)
    assert po.ColumnNotFoundError.__bases__ == (pl.exceptions.ColumnNotFoundError, ValueError)
    assert po.OutputColumnNotFoundError.__bases__ == (po.ColumnNotFoundError, KeyError)
    assert FormulaError.__bases__ == (ValueError,)
    for cls in (
        po.WrongTypeError,
        po.RefusedValueError,
        po.ColumnNotFoundError,
        po.OutputColumnNotFoundError,
    ):
        assert cls.__module__ == "polars_online._errors"
        assert cls.__mro__[1] is cls.__bases__[0]
        assert getattr(po, cls.__name__) is cls
        assert cls.__name__ in po._errors.__all__


# --- The CLI: an exit code and a message for each kind -------------------------

CLI_EXITS = {
    "SUCCESS": "0: a run that finished",
    "FAILURE": "1: every refusal and every error of a run, as `polars-online: <message>` on stderr",
    "usage": "2: a command line clap refuses, as `error: <message>` (clap's own voice)",
}


def _run_cli(online_cli: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(online_cli), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )


def _bad_config(tmp: Path) -> Path:
    path = tmp / "bad.toml"
    path.write_text("input = \n", encoding="utf-8")
    return path


def _job(tmp: Path, **override: str) -> Path:
    """A config that runs, with ``override`` replacing its top-level keys."""
    _df(40).write_parquet(tmp / "in.parquet")
    keys = {"input": str(tmp / "in.parquet"), "output": str(tmp / "out.parquet")}
    keys.update(override)
    body = "".join(f"{k} = '{v}'\n" for k, v in keys.items())
    path = tmp / "job.toml"
    path.write_text(
        body + '\n[[specs]]\nname = "ridge"\ntargets = ["y"]\nfeatures = ["x0"]\nclock = "t"\n'
        'half_life = 20.0\ngap_cap = 5.0\n[specs.model]\ntype = "ewridge"\n',
        encoding="utf-8",
    )
    return path


def test_the_command_line_exits_0_for_a_run(online_cli: Path, tmp_path: Path) -> None:
    done = _run_cli(online_cli, "--config", _job(tmp_path), "-q")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out.parquet").exists()


@pytest.mark.parametrize(
    ("label", "arrange", "fragment"),
    [
        ("a config that is not there", lambda t: ["--config", t / "nowhere.toml"], "(os error 2)"),
        (
            "a config that is a folder",
            lambda t: ["--config", t],
            "(os error 5)" if WINDOWS else "(os error 21)",
        ),
        (
            "an input that is not there",
            lambda t: ["--config", _job(t, input=str(t / "gone.parquet")), "-q"],
            "gone.parquet",
        ),
        (
            "an output in a folder that is not there",
            lambda t: ["--config", _job(t, output=str(t / "nope" / "o.parquet")), "-q"],
            "nope",
        ),
        (
            "a config that does not parse",
            lambda t: ["--config", _bad_config(t)],
            "parsing",
        ),
    ],
    ids=[
        "config-missing",
        "config-is-a-folder",
        "input-missing",
        "output-folder-missing",
        "config-does-not-parse",
    ],
)
def test_the_command_line_exits_1_with_its_prefix_for_an_io_or_config_error(
    online_cli: Path,
    tmp_path: Path,
    label: str,
    arrange: Callable[[Path], list[Any]],
    fragment: str,
) -> None:
    done = _run_cli(online_cli, *arrange(tmp_path))
    assert done.returncode == 1, f"{label}: {done.stderr}"
    assert done.stderr.startswith("polars-online: "), f"{label}: {done.stderr}"
    assert fragment in done.stderr, f"{label}: {done.stderr}"


@pytest.mark.skipif(
    WINDOWS or ROOT,
    reason="a file's read bit stops no one on Windows, and nothing for root: there is no "
    "unreadable file to make without another account; Windows' folder case is above",
)
def test_the_command_line_names_permission_denied_for_a_config_it_cannot_read(
    online_cli: Path, tmp_path: Path
) -> None:
    config = _job(tmp_path)
    config.chmod(0o000)
    try:
        done = _run_cli(online_cli, "--config", config)
    finally:
        config.chmod(0o600)
    assert done.returncode == 1, done.stderr
    assert done.stderr.startswith("polars-online: reading "), done.stderr
    assert "(os error 13)" in done.stderr, done.stderr


@pytest.mark.parametrize(
    "args",
    [[], ["--no-such-flag"], ["--input-format", "xlsx"], ["--chunk-rows", "5"]],
    ids=["no-config", "unknown-flag", "format-clap-cannot-parse", "renamed-flag"],
)
def test_the_command_line_exits_2_in_clap_s_voice_for_a_command_line_it_refuses(
    online_cli: Path, tmp_path: Path, args: list[str]
) -> None:
    head = [] if not args else ["--config", str(_job(tmp_path))]
    done = _run_cli(online_cli, *head, *args)
    assert done.returncode == 2, done.stderr
    assert done.stderr.startswith("error: "), done.stderr


# --- The inventory --------------------------------------------------------------


@dataclass(frozen=True)
class Raised:
    """A class the library raises: the cases that raise it through a public
    call (ids in ``CASES``), any test elsewhere that does (``file::name``, held
    to exist), or the reason no public call can."""

    cases: tuple[str, ...] = ()
    elsewhere: tuple[str, ...] = ()
    gap: str = ""
    surfaces: str = "python"  # python, rust, or both


#: Every class a ``raise`` in ``python/polars_online`` or a ``PyErr`` in
#: ``crates/online-py`` names, by the name the source writes.
PY_CLASSES: dict[str, Raised] = {
    "ValueError": Raised(
        cases=(
            "value-out-of-range",
            "value-several-problems",
            "value-state-is-not-a-state",
            "value-a-file-that-is-not-a-state",
            "value-a-broken-bank-does-not-save",
            "value-arrow-struct-exports-once",
            "value-native-config-is-not-json",
            "value-threads-too-few",
            "value-threads-too-many",
            "value-gbdt-label-is-also-a-margin",
            "value-calendar-every-is-not-polars-text",
        ),
        surfaces="both",
    ),
    "TypeError": Raised(
        cases=(
            "type-a-frame-is-needed",
            "type-a-parameter-of-the-wrong-kind",
            "type-threads-is-a-bool",
            "type-gbdt-features-is-one-name",
        ),
        surfaces="both",
    ),
    "WrongTypeError": Raised(
        cases=(
            "wrong-type-python-one-spec",
            "wrong-type-rust-text-feature",
            "wrong-type-rust-refresh-clock",
            "wrong-type-gbdt-feature",
        ),
        elsewhere=(
            "test_exception_classes.py::test_each_rust_side_wrong_type_raises_wrong_type_error",
        ),
        surfaces="both",
    ),
    "RefusedValueError": Raised(cases=("refused-value-reverse-without-a-window",)),
    "ColumnNotFoundError": Raised(
        cases=(
            "column-not-found-rust-feature",
            "column-not-found-python-clock",
            "column-not-found-scoring-group",
            "column-not-found-gbdt-label",
        ),
        surfaces="both",
    ),
    "OutputColumnNotFoundError": Raised(cases=("output-column-not-found",)),
    "FormulaError": Raised(cases=("formula-a-shift-reads-other-rows",)),
    "KeyError": Raised(cases=("key-error-no-such-spec", "key-error-gram-without-columns")),
    "IndexError": Raised(cases=("index-error-spec-position",)),
    "AttributeError": Raised(cases=("attribute-error-a-renamed-method",)),
    "AssertionError": Raised(
        cases=("assertion-error-the-model-gave-the-wrong-count",),
        # Two sites; this case reaches `_bank.py`'s with the native layer stubbed.
        gap="`gbdt.py` `_refuse_margin`'s last line is a defect guard (`pragma: no cover`): "
        "the checks above it name every way a margin is refused, so no call reaches it",
    ),
    "ModuleNotFoundError": Raised(
        cases=(
            "module-not-found-bank-gram",
            "module-not-found-po-gram",
            "module-not-found-po-corr",
            "module-not-found-po-sim",
        )
    ),
    "pl.exceptions.DuplicateError": Raised(cases=("duplicate-error-unnest-shares-a-name",)),
    "FileNotFoundError": Raised(
        cases=(
            "file-not-found-save-state-folder",
            "file-not-found-closed-groups-folder",
            "file-not-found-gbdt-glob",
        )
    ),
    "IsADirectoryError": Raised(cases=("is-a-directory-save-state-is-a-folder",)),
    "RuntimeError": Raised(
        gap="`gbdt.py`'s out-of-order guard in `lightgbm_dataset`'s row sequence fires only "
        "for a LightGBM that reads its rows out of order, which none that is supported does; "
        'the Rust side\'s "in use on another thread" sites are reached below',
        elsewhere=(
            "test_exception_types.py::test_a_call_while_a_fit_runs_is_refused_with_exactly_runtime_error",
            "test_error_messages.py::test_concurrent_fit_predict_says_so",
            "test_error_messages.py::test_the_stream_operators_release_the_gil",
        ),
        surfaces="both",
    ),
    "OSError": Raised(
        cases=("io-save-no-file-name",),
        surfaces="rust",
    ),
}

#: The names a ``raise`` may call that are not classes, with the classes each
#: returns. A new helper that returns an exception is added here.
HELPERS: dict[str, tuple[str, ...]] = {
    # `_spec_words.refusal`: one problem as it is, several as one TypeError if
    # any is one and one ValueError otherwise.
    "refusal": ("TypeError", "ValueError"),
    "_refuse": ("FormulaError",),
    "_reserved": ("FormulaError",),
    "both": ("TypeError",),
}

#: Where a ``raise`` re-raises an exception the code did not build: the name
#: of what it raises, and the function, so a new one is looked at.
RERAISES = {("gbdt.py", "_native_or_none")}


@dataclass(frozen=True)
class Kind:
    """An I/O outcome the Rust side hands the Python side: the class pyo3
    makes of it, and the cases that raise it."""

    python: str
    cases: tuple[str, ...] = ()
    gap: str = ""


#: Every ``std::io::ErrorKind`` the Rust source names outside its tests, and the
#: kinds a filesystem gives that the source does not name, by the class they
#: reach Python as. pyo3 maps a kind to the subclass of ``OSError`` for it and
#: leaves the rest a plain ``OSError``; an ``io::Error`` built by ``Error::other``
#: or with ``InvalidData`` is the savers' refusal, a ``ValueError``.
IO_KINDS: dict[str, Kind] = {
    "NotFound": Kind(
        "FileNotFoundError", ("io-save-not-found", "io-load-not-found", "io-load-under-a-file")
    ),
    "IsADirectory": Kind(
        "IsADirectoryError (PermissionError on Windows)",
        ("io-save-onto-a-folder", "io-load-a-folder", "io-closed-groups-csv-is-a-folder"),
    ),
    "NotADirectory": Kind(
        "NotADirectoryError (FileNotFoundError on Windows)",
        ("io-save-under-a-file", "io-load-under-a-file"),
    ),
    "PermissionDenied": Kind("PermissionError", ("io-save-permission-denied",)),
    "InvalidInput": Kind("OSError", ("io-save-no-file-name",)),
    "InvalidData": Kind(
        "ValueError",
        ("value-a-broken-bank-does-not-save",),
    ),
    "Other": Kind("ValueError", ("value-a-broken-bank-does-not-save",)),
    # Kinds a save's filesystem calls can give, which no public call can be
    # made to: pyo3 maps each to its subclass, and any other to ``OSError``,
    # the class the ``InvalidInput`` case holds.
    "<disk full>": Kind(
        "OSError",
        gap="a full disk (ENOSPC, `StorageFull`) cannot be made without root or a "
        "small filesystem to mount; the kind has no subclass, so it reaches Python by "
        "the `_ => OSError` arm that the `InvalidInput` case holds",
    ),
    "<an interrupted write>": Kind(
        "InterruptedError",
        gap="std retries `Interrupted` writes, so a save's call never returns it; "
        "an error raised while a save encodes keeps its kind in the task 388 Rust test",
    ),
    "<others pyo3 maps>": Kind(
        "AlreadyExists, BrokenPipe, Connection*, WouldBlock, TimedOut, OutOfMemory",
        gap="no call of a save or a write reaches them (a rename over a non-empty "
        "folder is `IsADirectory` or `PermissionDenied` here); pyo3's own mapping, "
        "which its tests hold",
    ),
}

# --- The completeness checks ------------------------------------------------------


def _python_sources() -> list[Path]:
    return sorted((REPO / "python" / "polars_online").glob("*.py"))


def _enclosing(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _class_names(fn: ast.AST, exc: ast.expr) -> list[str]:
    """The classes a ``raise``'s expression can be: a class called, a helper
    called (``HELPERS``), a name bound in the function to either, or either
    side of a conditional. ``?...`` is a form this does not read."""
    if isinstance(exc, ast.Call):
        name = ast.unparse(exc.func)
        if name in HELPERS:
            return list(HELPERS[name])
        if name.rsplit(".", 1)[-1][:1].isupper():
            return [name]
        return [f"?call of {name}"]
    if isinstance(exc, ast.IfExp):
        return _class_names(fn, exc.body) + _class_names(fn, exc.orelse)
    if isinstance(exc, ast.Name):
        found: list[str] = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == exc.id for t in node.targets
            ):
                found += _class_names(fn, node.value)
        return found or [f"?name {exc.id}"]
    return [f"?{ast.unparse(exc)[:40]}"]


def _python_raised() -> dict[str, set[str]]:
    """Each class a raise site or a built exception names, with the files."""
    out: dict[str, set[str]] = {}
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _enclosing(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                fn: ast.AST = node
                while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn = parents[fn]
                scope = fn if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else tree
                for name in _class_names(scope, node.exc):
                    out.setdefault(name, set()).add(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Raise):
                fn = node
                while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn = parents[fn]
                where = (path.name, getattr(fn, "name", "<module>"))
                if where not in RERAISES:
                    out.setdefault("?bare raise", set()).add(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Call):
                # An exception built and handed on (collected into a list, returned):
                # every class constructor named for an error is a class the package
                # makes, raised or not.
                name = ast.unparse(node.func)
                last = name.rsplit(".", 1)[-1]
                if last[:1].isupper() and re.search(r"(Error|Exception)$", last):
                    out.setdefault(name, set()).add(f"{path.name}:{node.lineno}")
    return out


def _strip_rust(text: str) -> str:
    """A Rust file's code before its test module, with line comments gone."""
    code = text.split("#[cfg(test)]")[0]
    return "\n".join(re.sub(r"//.*", "", line) for line in code.splitlines())


_RUST_PY = re.compile(r"\bPy([A-Z][A-Za-z]*(?:Error|Exception|Interrupt|Iteration|Warning|Exit))\b")


def _rust_exceptions() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """The Python classes ``crates/online-py`` raises, and the
    ``io::ErrorKind``s the Rust source maps, each with where.

    By pattern, since a Rust mapping is no tree to walk: pyo3 names its
    exception types ``Py<Class>`` (``PyValueError::new_err``, an import of
    ``pyo3::exceptions::{...}``), which the first pattern reads; a class
    defined in Python and fetched by name by ``errors_class("<Class>")`` is the
    second; ``create_exception!`` the third. The kinds are every
    ``ErrorKind::<Kind>`` in the code of ``online-py`` and ``online-polars``
    before its test modules, and ``Error::other``, which is the kind ``Other``.
    A mapping through ``PyErr::from(io::Error)`` or ``os_err`` is the kinds', and
    ``PyPolarsErr`` would be a new surface, so its mention fails here.
    """
    classes: dict[str, set[str]] = {}
    kinds: dict[str, set[str]] = {}
    for crate, is_py in (("online-py", True), ("online-polars", False)):
        for path in sorted((REPO / "crates" / crate / "src").rglob("*.rs")):
            if path.name == "tests.rs":
                continue
            code = _strip_rust(path.read_text(encoding="utf-8"))
            where = f"{crate}/{path.relative_to(REPO / 'crates' / crate / 'src')}"
            if is_py:
                for m in _RUST_PY.finditer(code):
                    classes.setdefault(m.group(1), set()).add(where)
                for m in re.finditer(r'errors_class\("(\w+)"', code):
                    classes.setdefault(m.group(1), set()).add(where)
                for m in re.finditer(r"create_exception!\(\s*\w+\s*,\s*(\w+)", code):
                    classes.setdefault(m.group(1), set()).add(where)
                if "PyPolarsErr" in code:
                    classes.setdefault("?PyPolarsErr", set()).add(where)
            for m in re.finditer(r"\bErrorKind::(\w+)", code):
                kinds.setdefault(m.group(1), set()).add(where)
            if re.search(r"\bError::other\(|Error::other\b", code):
                kinds.setdefault("Other", set()).add(where)
    return classes, kinds


def test_every_class_a_raise_names_is_in_the_inventory() -> None:
    raised = _python_raised()
    unlisted = {c: sorted(w)[:3] for c, w in raised.items() if c not in PY_CLASSES}
    assert not unlisted, (
        "a raise site or a built exception names a class that PY_CLASSES in "
        f"tests/test_exception_types.py does not list: {unlisted}. Add the class there, "
        "with a Case in CASES that raises it through a public call and asserts "
        "`type(e) is <the class>` (a name beginning `?` is a form of raise the "
        "collector does not read: teach `_class_names`, or add the helper to HELPERS)."
    )
    stale = sorted(
        c
        for c in PY_CLASSES
        if c not in raised and c not in _rust_exceptions()[0] and c != "OSError"
    )
    assert not stale, f"PY_CLASSES lists {stale}, which no source names any longer: remove it"


def test_every_rust_exception_class_and_io_kind_is_in_the_inventory() -> None:
    classes, kinds = _rust_exceptions()
    # The savers' and the runner's OSError is made from a kind, not named.
    known = set(PY_CLASSES)
    unlisted = {c: sorted(w) for c, w in classes.items() if c not in known}
    assert not unlisted, (
        f"crates/online-py raises {unlisted}, which PY_CLASSES in "
        "tests/test_exception_types.py does not list. Add the class there, with a Case "
        "that raises it through a public call and asserts its exact class."
    )
    unlisted_kinds = {k: sorted(w) for k, w in kinds.items() if k not in IO_KINDS}
    assert not unlisted_kinds, (
        f"the Rust source maps or makes the io::ErrorKind {unlisted_kinds}, which IO_KINDS in "
        "tests/test_exception_types.py does not list. Add the kind with the class it reaches "
        "Python as and a Case that raises it through a public call (a save or a write)."
    )
    assert "ValueError" in classes and "RuntimeError" in classes, (
        "the Rust scan reads no `PyValueError` or `PyRuntimeError`: its pattern no longer "
        "matches what online-py writes"
    )
    assert {"NotFound", "IsADirectory", "InvalidData"} <= set(kinds), (
        "the Rust scan reads no ErrorKind it knows is named: its pattern no longer matches"
    )


def test_the_command_line_s_exit_codes_are_in_the_inventory() -> None:
    code = _strip_rust(
        (REPO / "crates" / "online-cli" / "src" / "main.rs").read_text(encoding="utf-8")
    )
    seen = set(re.findall(r"ExitCode::(\w+)", code))
    if re.search(r"\.exit\(\)", code):
        seen.add("usage")
    explicit = set(re.findall(r"process::exit\(([^)]*)\)", code))
    assert not explicit, (
        f"the command line calls process::exit({explicit}): add its code to CLI_EXITS in "
        "tests/test_exception_types.py, with a test that reaches it"
    )
    assert seen == set(CLI_EXITS), (
        f"the command line's exits are {sorted(seen)} and CLI_EXITS lists {sorted(CLI_EXITS)}: "
        "make CLI_EXITS say each, with a test that reaches it"
    )


def test_each_inventory_entry_has_a_case_or_a_reason() -> None:
    ids = {c.id for c in CASES}
    seen_in_entries: set[str] = set()
    for name, entry in PY_CLASSES.items():
        assert entry.cases or entry.elsewhere or entry.gap, (
            f"{name}: no case raises it and no reason is given. Add a Case to CASES."
        )
        for case in entry.cases:
            assert case in ids, f"{name}: {case} is not in CASES"
            seen_in_entries.add(case)
    for kind, io in IO_KINDS.items():
        assert io.cases or io.gap, f"{kind}: no case raises it and no reason is given"
        for case in io.cases:
            assert case in ids, f"{kind}: {case} is not in CASES"
            seen_in_entries.add(case)
    orphans = sorted(ids - seen_in_entries)
    assert not orphans, (
        f"cases that no inventory entry names: {orphans}. List each under its class."
    )


def test_every_case_raises_the_class_its_entry_lists() -> None:
    """An entry's case is of its class: ``raises`` is the entry, or a class that
    entry's name means (the library's own classes by name)."""
    by_id = {c.id: c for c in CASES}
    for name, entry in PY_CLASSES.items():
        for case_id in entry.cases:
            want = by_id[case_id].raises
            short = want.__qualname__ if want.__module__ == "builtins" else want.__name__
            shown = f"pl.exceptions.{short}" if want.__module__ == "polars.exceptions" else short
            assert shown == name, f"{case_id} raises {shown}, but is listed under {name}"


def test_every_test_listed_as_elsewhere_exists() -> None:
    missing = []
    for entry in PY_CLASSES.values():
        for ref in entry.elsewhere:
            file, _, func = ref.partition("::")
            path = REPO / "tests" / file
            if not path.exists() or not re.search(
                rf"def {re.escape(func)}\b", path.read_text(encoding="utf-8")
            ):
                missing.append(ref)
    assert not missing, f"tests the inventory points to are gone or renamed: {missing}"
