"""The import-time CPU check of the x86-64 wheels.

The x86-64 wheels of ``polars-online`` are compiled for CPUs with AVX2, FMA
and the rest of Polars' feature set, and a CPU without one dies on an
illegal instruction. ``rheon/_cpu_check.py`` reads the CPU's
features in pure Python, first thing in ``__init__``, and warns. These
tests hold the pieces: the build script that records what a build needs
(``crates/rheon-py/build.rs``, compiled and run for real), the comparison
and the warning (against a faked CPU, since the failing path cannot be
reached on a CPU that has every feature), the real reader (on an x86-64
host, against ``/proc/cpuinfo`` where there is one), and the order of the
imports, which is the point: the check runs before the extension module is
asked for.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import yaml

from rheon import _cpu_check

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
BUILD_RS = REPO / "crates" / "rheon-py" / "build.rs"
RELEASE = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text("utf-8"))

#: The flags the release gives the x86-64 wheels of `polars-online`.
X86_FEATURES = RELEASE["env"]["X86_FEATURES"]

ON_X86 = platform.machine().lower() in {"x86_64", "amd64", "x64"} and sys.maxsize > 2**32


# --- the build script ----------------------------------------------------------


@pytest.fixture(scope="module")
def build_script(tmp_path_factory) -> Path:
    """build.rs compiled on its own: it uses the standard library alone."""
    rustc = shutil.which("rustc") or shutil.which("rustc", path=str(Path.home() / ".cargo" / "bin"))
    if rustc is None:
        pytest.fail("rustc is not on PATH; `source scripts/env.sh` first")
    out = tmp_path_factory.mktemp("build-rs") / ("build.exe" if os.name == "nt" else "build")
    res = subprocess.run(
        [rustc, "--edition", "2024", "-o", str(out), str(BUILD_RS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return out


def _run_build_script(
    exe: Path, root: Path, flags: list[str] | None, arch: str = "x86_64", python_dir: bool = True
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run it in a copy of the layout (crates/rheon-py/, python/rheon/)
    and return the process and the file it writes."""
    manifest = root / "crates" / "rheon-py"
    manifest.mkdir(parents=True, exist_ok=True)
    package = root / "python" / "rheon"
    if python_dir:
        package.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO", "RUSTFLAGS"))}
    env["CARGO_MANIFEST_DIR"] = str(manifest)
    env["CARGO_CFG_TARGET_ARCH"] = arch
    if flags is not None:
        env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    res = subprocess.run(
        [str(exe)], env=env, capture_output=True, text=True, encoding="utf-8", check=False
    )
    return res, package / "_build_flags.py"


def _recorded(path: Path) -> str:
    (value,) = re.findall(r'^BUILD_FEATURE_FLAGS = "([^"]*)"$', path.read_text("utf-8"), re.M)
    return value


@pytest.mark.parametrize(
    "flags, want",
    [
        (["-C", "target-feature=+avx2,+fma"], "+avx2,+fma"),
        (["-Ctarget-feature=+avx2,+fma"], "+avx2,+fma"),
        (["--codegen", "target-feature=+avx2"], "+avx2"),
        (["--codegen=target-feature=+avx2"], "+avx2"),
        # Every flag adds; the order is the order given.
        (["-C", "target-feature=+bmi1", "-C", "target-feature=+avx2,+fma"], "+bmi1,+avx2,+fma"),
        # A feature turned off is not needed, and crt-static is no CPU feature.
        (["-C", "target-feature=+avx2,-sse4a,+crt-static,-crt-static"], "+avx2"),
        (["-C", "target-feature=-crt-static"], ""),
        # Other flags do not name features.
        (["-C", "target-cpu=x86-64-v3", "-C", "link-arg=-fuse-ld=lld", "-Zthreads=4"], ""),
        (["-C", "target-feature=+avx2", "-C", "target-cpu=native", "--cfg", "x"], "+avx2"),
        ([], ""),
        (None, ""),
    ],
)
def test_the_build_script_records_the_features_rustc_was_told(build_script, tmp_path, flags, want):
    res, path = _run_build_script(build_script, tmp_path, flags)
    assert res.returncode == 0, res.stderr
    assert _recorded(path) == want
    # The three conditions that make cargo write it again.
    assert "cargo:rerun-if-changed=build.rs" in res.stdout
    assert "cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS" in res.stdout
    assert any(
        line.startswith("cargo:rerun-if-changed=") and line.endswith("_build_flags.py")
        for line in res.stdout.splitlines()
    ), res.stdout


@pytest.mark.parametrize("arch", ["aarch64", "arm", "riscv64"])
def test_a_build_that_is_not_for_x86_records_nothing(build_script, tmp_path, arch):
    res, path = _run_build_script(build_script, tmp_path, ["-C", "target-feature=+neon"], arch)
    assert res.returncode == 0, res.stderr
    assert _recorded(path) == ""


def test_the_recorded_file_names_the_features_and_the_python_reads_them_back(
    build_script, tmp_path
):
    """What the script writes is what ``_cpu_check`` reads: the release's own
    flag list, through both."""
    res, path = _run_build_script(build_script, tmp_path, ["-C", f"target-feature={X86_FEATURES}"])
    assert res.returncode == 0, res.stderr
    namespace: dict = {}
    exec(path.read_text("utf-8"), namespace)
    assert namespace["BUILD_FEATURE_FLAGS"] == X86_FEATURES
    assert _cpu_check.required_features(namespace["BUILD_FEATURE_FLAGS"]) == [
        f.removeprefix("+") for f in X86_FEATURES.split(",")
    ]


def test_a_file_that_is_current_is_left_alone_and_a_stale_one_is_rewritten(build_script, tmp_path):
    """Cargo must not find the file newer than its own record each time, so a
    build that records the same flags does not write it; one that records
    other flags does, and so does one whose file is gone."""
    flags = ["-C", "target-feature=+avx2"]
    _, path = _run_build_script(build_script, tmp_path, flags)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    _run_build_script(build_script, tmp_path, flags)
    assert path.stat().st_mtime == 1_000_000_000
    _run_build_script(build_script, tmp_path, ["-C", "target-feature=+avx2,+fma"])
    assert _recorded(path) == "+avx2,+fma"
    path.unlink()
    _run_build_script(build_script, tmp_path, flags)
    assert _recorded(path) == "+avx2"


def test_a_layout_without_the_python_source_builds_and_records_nothing(build_script, tmp_path):
    res, path = _run_build_script(
        build_script, tmp_path, ["-C", "target-feature=+avx2"], python_dir=False
    )
    assert res.returncode == 0, res.stderr
    assert not path.exists()


# --- the comparison and the warning -----------------------------------------------


@pytest.fixture
def cpu(monkeypatch):
    """A CPU whose features a test sets, and a build whose flags it sets."""

    class Fake:
        def build(self, flags: str) -> None:
            monkeypatch.setattr(_cpu_check, "baked_flags", lambda: flags)

        def has(self, features: dict[str, bool] | None) -> None:
            monkeypatch.setattr(_cpu_check, "read_cpu_features", lambda: features)

    monkeypatch.delenv(_cpu_check.SKIP_ENV, raising=False)
    return Fake()


def _all(**off: bool) -> dict[str, bool]:
    return {name: off.get(name.replace(".", "_"), True) for name in _cpu_check.FEATURE_BITS}


def test_required_features_drop_the_plus_and_the_features_turned_off():
    assert _cpu_check.required_features("+sse3,+avx2,-sse4a,,+fma") == ["sse3", "avx2", "fma"]
    assert _cpu_check.required_features("") == []


def test_every_missing_feature_is_named_in_the_builds_order(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all(bmi1=False, fma=False, avx2=False))
    # The build names avx2, then fma, then bmi1.
    assert _cpu_check.missing_features() == ["avx2", "fma", "bmi1"]


def test_a_cpu_with_every_feature_gets_no_warning(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_missing_feature_warns_with_the_feature_the_remedy_and_the_way_out(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all(avx2=False, fma=False))
    with pytest.warns(RuntimeWarning, match="Missing required CPU features") as caught:
        _cpu_check.run()
    (warning,) = caught
    text = str(warning.message)
    assert "avx2, fma" in text
    # The package to install, which one to remove, Rosetta, and the variable.
    assert "polars-online-compat" in text
    assert "pip uninstall polars-online" in text
    assert "Rosetta" in text
    assert "POLARS_ONLINE_SKIP_CPU_CHECK" in text
    assert "crash" in text
    assert warning.category is RuntimeWarning


def test_the_variable_is_the_documented_one():
    assert _cpu_check.SKIP_ENV == "POLARS_ONLINE_SKIP_CPU_CHECK"


def test_the_variable_skips_the_check(cpu, monkeypatch):
    cpu.build(X86_FEATURES)
    cpu.has(_all(avx2=False))
    monkeypatch.setenv(_cpu_check.SKIP_ENV, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()
    # An empty value is no request to skip, as in Polars.
    monkeypatch.setenv(_cpu_check.SKIP_ENV, "")
    with pytest.warns(RuntimeWarning):
        _cpu_check.run()


@pytest.mark.parametrize("flags", ["", "-avx2", ","])
def test_a_build_that_needs_nothing_reads_nothing(cpu, monkeypatch, flags):
    """No CPU read at all on ARM, on the compat build and on a source build:
    the reader would raise if asked."""
    cpu.build(flags)

    def refuse() -> None:
        raise AssertionError("the CPU was read for a build that needs no feature")

    monkeypatch.setattr(_cpu_check, "read_cpu_features", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_cpu_that_cannot_be_read_is_no_reason_to_warn(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_feature_the_check_cannot_read_is_not_reported_missing(cpu):
    cpu.build("+avx2,+avx512f")
    cpu.has(_all())
    assert _cpu_check.missing_features() == []


def test_a_build_with_no_target_feature_flag_has_nothing_to_check(monkeypatch):
    """Every development build, the compat wheel and a source build record no
    feature, whatever CPU they run on."""
    monkeypatch.setattr(_cpu_check, "baked_flags", lambda: "")
    assert _cpu_check.missing_features() == []


def test_no_record_at_all_is_no_check(monkeypatch):
    monkeypatch.setitem(sys.modules, "rheon._build_flags", None)  # import fails
    assert _cpu_check.baked_flags() == ""


def test_the_releases_features_are_all_features_the_check_can_read():
    """A feature the release builds for but the check cannot read would be a
    feature nothing checks."""
    names = [f.removeprefix("+") for f in X86_FEATURES.split(",")]
    assert set(names) == set(_cpu_check.FEATURE_BITS), set(names) ^ set(_cpu_check.FEATURE_BITS)
    assert len(set(names)) == len(names)


def test_the_cpu_is_never_read_off_x86(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert _cpu_check.read_cpu_features() is None
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert _cpu_check.read_cpu_features() is None


# --- the real reader, where there is an x86-64 CPU ------------------------------------

#: /proc/cpuinfo's name for each feature the check reads.
CPUINFO = {
    "sse3": "pni",
    "pclmulqdq": "pclmulqdq",
    "ssse3": "ssse3",
    "fma": "fma",
    "cmpxchg16b": "cx16",
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
    "movbe": "movbe",
    "popcnt": "popcnt",
    "avx": "avx",
    "bmi1": "bmi1",
    "avx2": "avx2",
    "bmi2": "bmi2",
    "lzcnt": "abm",
}

x86_only = pytest.mark.skipif(not ON_X86, reason="cpuid exists on x86-64 CPUs only")


@x86_only
def test_the_real_reader_answers_for_every_feature_and_the_baseline_is_there():
    features = _cpu_check.read_cpu_features()
    assert features is not None, "cpuid could not be run on an x86-64 host"
    assert set(features) == set(_cpu_check.FEATURE_BITS)
    # Every x86-64 CPU the wheels run on, v1 or v3, has these.
    for name in ("sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cmpxchg16b"):
        assert features[name], name


@x86_only
@pytest.mark.skipif(
    not Path("/proc/cpuinfo").exists(), reason="/proc/cpuinfo is the oracle, and Linux's"
)
def test_the_real_reader_agrees_with_the_kernels_flags():
    """The kernel's list is read by the kernel's own cpuid code, which shares
    nothing with these bytes."""
    text = Path("/proc/cpuinfo").read_text("utf-8")
    (line,) = [ln for ln in text.splitlines() if ln.startswith("flags")][:1]
    flags = set(line.split(":", 1)[1].split())
    features = _cpu_check.read_cpu_features()
    assert features is not None
    for name, flag in CPUINFO.items():
        assert features[name] == (flag in flags), (name, flag)


@x86_only
def test_reading_twice_gives_the_same_answer_and_frees_its_memory():
    first = _cpu_check.read_cpu_features()
    assert all(_cpu_check.read_cpu_features() == first for _ in range(50))


# --- the order of the imports -----------------------------------------------------------


def test_the_check_runs_before_the_extension_module_is_asked_for():
    """A fresh interpreter imports the package with a finder that logs which
    of its modules the import machinery is asked for, in order: the build's
    record (read by the check) comes before the compiled module."""
    code = textwrap.dedent(
        """
        import sys

        asked = []

        class Log:
            def find_spec(self, name, path=None, target=None):
                if name.startswith("rheon"):
                    asked.append(name)
                return None

        sys.meta_path.insert(0, Log())
        import rheon

        print("\\n".join(asked))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != _cpu_check.SKIP_ENV}
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    asked = res.stdout.split()
    # The package, then the check, then what the check reads, and only then
    # anything else of the package: the compiled module among it.
    assert asked[:3] == ["rheon", "rheon._cpu_check", "rheon._build_flags"]
    assert "rheon._rheon" in asked, asked


def test_the_skip_variable_skips_the_whole_check_in_a_fresh_interpreter():
    """With the variable set, the build's record is never asked for."""
    code = textwrap.dedent(
        """
        import sys

        asked = []

        class Log:
            def find_spec(self, name, path=None, target=None):
                asked.append(name)
                return None

        sys.meta_path.insert(0, Log())
        import rheon

        print("rheon._build_flags" in asked)
        """
    )
    env = {**os.environ, _cpu_check.SKIP_ENV: "1"}
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False"
