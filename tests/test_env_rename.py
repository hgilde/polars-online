"""The environment variables took the `RHEON_` prefix with the library's name.

Each is read by its new name. The name it had before, `POLARS_ONLINE_...`, is
read while the new one is not set at all, with one `RheonDeprecationWarning`
per process at `import rheon` (the command line prints one notice on standard
error instead). When both are set the new one wins and nothing is said. The
old names are read until the next major release.

Each case runs in a child interpreter of its own: what a process has warned
about, and the pool it has built, cannot be undone. The Rust side reads the
same pairs (`crates/rheon-polars/src/env.rs`, tested there and in
`crates/rheon-cli/tests/surface.rs`); the cases here run its readers through
the Python and command-line surfaces.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import child
from rheon import _env
from rheon._warnings import RheonDeprecationWarning

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: (new name, a value of it that changes nothing a test below depends on), in
#: the order of `_env.RENAMED`'s own pairs.
VARIABLES = [
    ("RHEON_MAX_THREADS", "2"),
    ("RHEON_SKIP_CPU_CHECK", "1"),
    ("RHEON_TIMING", "0"),
    ("RHEON_WARN_UNSTABLE", "0"),
]


def old_of(new: str) -> str:
    return _env.RENAMED[new]


def clean_env(**extra: str) -> dict[str, str]:
    """The environment of a child: none of the variables inherited."""
    env = child.env(**extra)
    for k in list(env):
        if k.startswith(("RHEON_", "POLARS_ONLINE_")) and k not in extra:
            del env[k]
    return env


def python(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    res = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=clean_env(**env),
        cwd=str(REPO),
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res


#: Imports the package twice and reads every variable again, then prints the
#: deprecation warnings issued: (category, message) pairs.
WARNINGS = """
    import json, warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import rheon
        import rheon.polars
        import rheon
        for name in rheon._env.RENAMED:
            rheon._env.get(name)
            rheon._env.get(name)
    print(json.dumps([
        [w.category.__name__, str(w.message)]
        for w in caught
        if issubclass(w.category, rheon.RheonDeprecationWarning)
    ]))
"""


def deprecations(**env: str) -> list[list[str]]:
    out: list[list[str]] = json.loads(python(WARNINGS, **env).stdout)
    return out


# --- the warning -------------------------------------------------------------------


def test_the_pairs_are_the_four_the_library_reads_and_each_new_name_has_the_prefix():
    assert dict(VARIABLES).keys() == _env.RENAMED.keys()
    for new, old in _env.RENAMED.items():
        assert new.startswith("RHEON_") and old == f"POLARS_ONLINE_{new.removeprefix('RHEON_')}"


@pytest.mark.parametrize(("new", "value"), VARIABLES)
def test_an_old_name_alone_is_read_and_warns_once_naming_the_new_one(new, value):
    old = old_of(new)
    got = deprecations(**{old: value})
    assert [c for c, _ in got] == ["RheonDeprecationWarning"], got
    message = got[0][1]
    assert old in message and f"renamed {new}" in message and f"Set {new} instead" in message


@pytest.mark.parametrize(("new", "value"), VARIABLES)
def test_both_set_the_new_one_wins_and_nothing_is_said(new, value):
    assert deprecations(**{new: value, old_of(new): value}) == []


@pytest.mark.parametrize(("new", "value"), VARIABLES)
def test_the_new_name_alone_and_neither_say_nothing(new, value):
    assert deprecations(**{new: value}) == []
    assert deprecations() == []


def test_each_old_name_set_warns_once_and_each_names_its_own_new_one():
    got = deprecations(**{old_of(new): value for new, value in VARIABLES})
    assert len(got) == len(VARIABLES), got
    for new, _ in VARIABLES:
        assert sum(f"renamed {new}." in m for _, m in got) == 1, got


def test_the_warning_points_at_the_line_that_imported_the_package():
    res = python(
        """
        import warnings
        warnings.simplefilter("always")
        import rheon
        """,
        POLARS_ONLINE_MAX_THREADS="2",
    )
    assert "<string>:4: RheonDeprecationWarning" in res.stderr, res.stderr


def test_get_reads_the_new_name_first_and_warns_once_in_process(monkeypatch):
    for new, old in _env.RENAMED.items():
        monkeypatch.delenv(new, raising=False)
        monkeypatch.delenv(old, raising=False)
    monkeypatch.setattr(_env, "_warned", set())
    assert _env.get("RHEON_MAX_THREADS") is None

    monkeypatch.setenv("POLARS_ONLINE_MAX_THREADS", "4")
    with pytest.warns(RheonDeprecationWarning, match="renamed RHEON_MAX_THREADS"):
        assert _env.get("RHEON_MAX_THREADS") == "4"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _env.get("RHEON_MAX_THREADS") == "4"  # the second read is silent

    # An empty new value is set: it wins over the old name, and says nothing.
    monkeypatch.setattr(_env, "_warned", set())
    monkeypatch.setenv("RHEON_MAX_THREADS", "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _env.get("RHEON_MAX_THREADS") == ""


# --- each variable does its work under the old name -------------------------------


def test_max_threads_sizes_the_default_pool_under_either_name_and_the_new_one_wins():
    show = "import rheon; print(rheon.thread_pool_size())"
    assert python(show, RHEON_MAX_THREADS="3").stdout.strip() == "3"
    assert python(show, POLARS_ONLINE_MAX_THREADS="2").stdout.strip() == "2"
    both = python(show, RHEON_MAX_THREADS="3", POLARS_ONLINE_MAX_THREADS="2")
    assert both.stdout.strip() == "3"


def test_a_bad_old_max_threads_is_refused_naming_the_old_variable_the_user_set():
    code = """
        import polars as pl, rheon
        from rheon import polars as rp
        spec = rheon.spec.ewridge('m', targets=['y'], features=['x'], half_life=10.0)
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})
        try:
            rp.ModelBank([spec]).fit_predict(df)
        except ValueError as e:
            print(str(e).split(' is ')[0])
    """
    old = python(code, POLARS_ONLINE_MAX_THREADS="eight").stdout.strip()
    assert old == 'environment variable POLARS_ONLINE_MAX_THREADS: "eight"'
    new = python(code, RHEON_MAX_THREADS="eight", POLARS_ONLINE_MAX_THREADS="2").stdout.strip()
    assert new == 'environment variable RHEON_MAX_THREADS: "eight"'


UNSTABLE = """
    import json, warnings
    import numpy as np, rheon
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rheon.corr.nearest(np.eye(2))
    print(json.dumps([w.category.__name__ for w in caught]))
"""


def test_warn_unstable_turns_the_label_on_under_either_name_and_the_new_one_wins():
    def issued(**env: str) -> list[str]:
        out: list[str] = json.loads(python(UNSTABLE, **env).stdout)
        return [c for c in out if c == "UnstableWarning"]

    assert issued(RHEON_WARN_UNSTABLE="1") == ["UnstableWarning"]
    assert issued(POLARS_ONLINE_WARN_UNSTABLE="1") == ["UnstableWarning"]
    assert issued(RHEON_WARN_UNSTABLE="0", POLARS_ONLINE_WARN_UNSTABLE="1") == []
    assert issued() == []


CPU = """
    import json, warnings
    import rheon
    from rheon import _cpu_check
    _cpu_check.missing_features = lambda: ["avx"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _cpu_check.run()
    print(json.dumps([w.category.__name__ for w in caught]))
"""


def test_skip_cpu_check_skips_under_either_name_and_the_new_one_wins():
    def warned(**env: str) -> list[str]:
        out: list[str] = json.loads(python(CPU, **env).stdout)
        return [c for c in out if c == "RuntimeWarning"]

    assert warned() == ["RuntimeWarning"]
    assert warned(RHEON_SKIP_CPU_CHECK="1") == []
    assert warned(POLARS_ONLINE_SKIP_CPU_CHECK="1") == []
    # The new name, set and empty, is "do not skip", whatever the old one says.
    assert warned(RHEON_SKIP_CPU_CHECK="", POLARS_ONLINE_SKIP_CPU_CHECK="1") == ["RuntimeWarning"]


TIMING = """
    import polars as pl, rheon
    from rheon import polars as rp
    spec = rheon.spec.ewridge('m', targets=['y'], features=['x'], half_life=10.0)
    df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [2.0, 4.0, 6.0]})
    rp.ModelBank([spec]).fit_predict(df)
"""


def test_timing_prints_under_either_name_and_the_new_one_wins():
    def lines(**env: str) -> list[str]:
        err = python(TIMING, **env).stderr
        return [ln for ln in err.splitlines() if ln.startswith("RHEON_TIMING rows=")]

    assert len(lines(RHEON_TIMING="1")) == 1
    assert len(lines(POLARS_ONLINE_TIMING="1")) == 1
    assert lines(RHEON_TIMING="0", POLARS_ONLINE_TIMING="1") == []
    assert lines() == []


# --- the command line --------------------------------------------------------------


def run_cli(cli: Path, **env: str) -> str:
    res = subprocess.run(
        [str(cli), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=clean_env(**env),
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return res.stderr


@pytest.mark.parametrize(
    "new", ["RHEON_MAX_THREADS", "RHEON_TIMING", "RHEON_WARN_UNSTABLE"], ids=lambda n: n
)
def test_the_command_line_says_once_on_stderr_which_name_to_set(rheon_cli, new):
    old = old_of(new)
    err = run_cli(rheon_cli, **{old: "1"})
    notices = [ln for ln in err.splitlines() if "was renamed" in ln]
    assert notices == [
        f"rheon: warning: {old} was renamed {new}, and the old name is read for now. "
        f"Set {new} instead."
    ], err
    # Both set: the new name wins, and nothing is said. Neither: nothing.
    assert run_cli(rheon_cli, **{old: "1", new: "1"}) == ""
    assert run_cli(rheon_cli) == ""


def test_the_command_line_does_not_read_the_python_only_variable(rheon_cli):
    """`RHEON_SKIP_CPU_CHECK` is the import check's alone: the binary has none."""
    assert run_cli(rheon_cli, POLARS_ONLINE_SKIP_CPU_CHECK="1") == ""
