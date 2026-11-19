"""Two PyPI projects from one source: ``polars-online`` and ``polars-online-compat``.

The x86-64 wheels of ``polars-online`` are built for CPUs with AVX2, FMA and
the rest of the feature set Polars builds its own default runtime for, and
check the CPU when the package is imported. ``polars-online-compat`` is the
same source, version and import name, built for every x86-64 CPU, and ships
for the ARM platforms too, as Polars' ``polars-runtime-compat`` does. These
tests hold what makes that so: the script that renames a checkout, and the
release workflow's build, smoke test, upload and verification of both.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
RELEASE = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text("utf-8"))
JOBS = RELEASE["jobs"]
BUILD = JOBS["build"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat_package = _script("compat_package")
wheel_smoke = _script("wheel_smoke")

#: Polars' own default x86-64 runtime's `BUILD_FEATURE_FLAGS`, read from its
#: 2.0.0 and 2.1.0 wheels (`polars-runtime-32`, manylinux x86-64).
POLARS_FEATURES = (
    "+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,"
    "+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe"
)

# --- the rename ---------------------------------------------------------------


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text("utf-8"))


def test_the_rename_changes_the_distribution_and_nothing_that_names_the_code(tmp_path):
    for name in ("pyproject.toml", "README.md"):
        (tmp_path / name).write_text((REPO / name).read_text("utf-8"), encoding="utf-8")
    assert compat_package.main(["--root", str(tmp_path)]) == 0
    before = _project(REPO / "pyproject.toml")
    after = _project(tmp_path / "pyproject.toml")
    assert before["project"]["name"] == "rheon"
    assert after["project"]["name"] == "polars-online-compat"
    assert after["project"]["description"] == compat_package.COMPAT_DESCRIPTION
    # Everything else is the same: version, dependencies, the module the
    # wheel installs. The import name stays `rheon`.
    for tree in (before, after):
        del tree["project"]["name"], tree["project"]["description"]
    assert before == after
    assert after["tool"]["maturin"]["module-name"] == "rheon._rheon"
    assert after["tool"]["maturin"]["python-source"] == "python"


def test_the_readme_of_the_compat_package_opens_with_a_notice_after_its_title(tmp_path):
    readme = (REPO / "README.md").read_text("utf-8")
    new = compat_package.prepend_notice(readme)
    lines = new.splitlines()
    title = next(i for i, line in enumerate(lines) if line.startswith("# "))
    assert lines[title] == "# polars-online"
    assert "polars-online-compat" in "\n".join(lines[title : title + 12])
    # Nothing is lost: the original text follows the notice.
    assert new.endswith(readme.split("\n", title + 1)[title + 1].lstrip("\n"))
    with pytest.raises(ValueError, match="no `# ` title"):
        compat_package.prepend_notice("no title here\n")


def test_a_checkout_already_renamed_is_refused(tmp_path):
    for name in ("pyproject.toml", "README.md"):
        (tmp_path / name).write_text((REPO / name).read_text("utf-8"), encoding="utf-8")
    assert compat_package.main(["--root", str(tmp_path)]) == 0
    assert compat_package.main(["--root", str(tmp_path)]) == 1


def test_a_pyproject_whose_name_line_moved_is_refused():
    text = (REPO / "pyproject.toml").read_text("utf-8")
    with pytest.raises(ValueError, match="no single"):
        compat_package.rename_pyproject(text.replace('name = "rheon"', 'name = "x"'))


def test_the_compat_names_are_the_ones_the_workflow_uploads_and_verifies():
    assert compat_package.COMPAT_NAME == "polars-online-compat"
    assert wheel_smoke.DISTRIBUTIONS == ("polars-online", "polars-online-compat")
    verify = _script("verify_release")
    assert verify.DISTRIBUTIONS == wheel_smoke.DISTRIBUTIONS
    from rheon import _cpu_check

    # scripts/compat_package.py still names the old project; the packaging task renames it.
    assert _cpu_check.COMPAT_PACKAGE == "rheon-compat"


# --- the build matrix ----------------------------------------------------------------


def _legs() -> list[dict]:
    """The matrix's combinations as GitHub expands them: the product of the
    axes, then each ``include`` entry added to every combination whose axis
    values it does not contradict."""
    matrix = BUILD["strategy"]["matrix"]
    axes = {k: v for k, v in matrix.items() if k != "include"}
    legs = [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
    for extra in matrix["include"]:
        for leg in legs:
            if all(leg[k] == v for k, v in extra.items() if k in axes):
                leg.update(extra)
    return legs


def _step(job: dict, name: str) -> dict:
    return next(s for s in job["steps"] if s.get("name") == name)


def test_every_target_is_built_under_both_names():
    legs = _legs()
    targets = {leg["target"] for leg in legs}
    assert targets == {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
    }
    assert BUILD["strategy"]["matrix"]["variant"] == ["default", "compat"]
    assert len(legs) == 12
    assert all("os" in leg and "bin" in leg for leg in legs), legs
    # Every x86-64 target is marked, and no other.
    assert {leg["target"] for leg in legs if leg.get("x86")} == {
        t for t in targets if t.startswith("x86_64-")
    }


def test_the_default_x86_wheels_are_built_with_the_feature_list_and_no_other_is():
    """Read as GitHub reads the expression: a flag for the default variant of
    an x86-64 target, empty for the compat variant and for ARM."""
    env = _step(BUILD, "build the wheel")["env"]
    assert list(env) == ["RUSTFLAGS"]
    expression = env["RUSTFLAGS"]
    assert expression == (
        "${{ matrix.x86 && matrix.variant == 'default' && "
        "format('-C target-feature={0}', env.X86_FEATURES) || '' }}"
    )
    assert RELEASE["env"]["X86_FEATURES"] == POLARS_FEATURES

    def rustflags(leg: dict) -> str:
        if leg.get("x86") and leg["variant"] == "default":
            return f"-C target-feature={RELEASE['env']['X86_FEATURES']}"
        return ""

    built_for_v3 = {leg["target"] for leg in _legs() if rustflags(leg)}
    assert built_for_v3 == {
        "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
    }
    assert all(not rustflags(leg) for leg in _legs() if leg["variant"] == "compat")
    # Nothing else in the workflow sets rustflags or a target CPU: the wheel's
    # flags are this one expression, which build.rs then records.
    text = (REPO / ".github" / "workflows" / "release.yml").read_text("utf-8")
    assert len(re.findall(r"^\s*RUSTFLAGS:", text, re.M)) == 1
    assert "target-cpu" not in text.replace("`target-cpu`", "")


def test_the_action_that_hands_the_flags_to_its_containers_is_the_one_that_was_read():
    """maturin-action hands a variable on to its manylinux and musllinux
    containers if its name starts with one of a list of prefixes, and
    RUSTFLAGS starts with `RUST`. That was read in the action's source at
    this commit (src/index.ts, `ALLOWED_ENV_PREFIXES`); a new pin is a reason
    to read it again, and to change the commit named here with it."""
    pin = _step(BUILD, "build the wheel")["uses"]
    assert pin == "PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b"


def test_the_compat_project_is_renamed_after_the_readme_and_before_the_build():
    steps = BUILD["steps"]
    names = [s.get("name", "") for s in steps]
    readme = names.index("the README's links point at GitHub, at this ref")
    rename = names.index("the project is polars-online-compat")
    build = names.index("build the wheel")
    assert readme < rename < build
    step = steps[rename]
    assert step["if"] == "matrix.variant == 'compat'"
    assert "scripts/compat_package.py" in step["run"]


def test_only_the_default_variant_builds_the_command_line_binary():
    for name in ("build the CLI (Linux, in the manylinux2014 image)", "build the CLI"):
        assert "matrix.variant == 'default'" in _step(BUILD, name)["if"]
    uploads = [s for s in BUILD["steps"] if "upload-artifact" in str(s.get("uses", ""))]
    wheel, cli = uploads
    assert wheel["with"]["name"] == "wheel-${{ matrix.target }}-${{ matrix.variant }}"
    assert cli["if"] == "matrix.cli && matrix.variant == 'default'"


def test_each_wheel_is_smoke_tested_as_its_project_with_its_features():
    """The wheel's distribution and recorded features are asked for, from the
    variant and target, in the runner's smoke test and in Alpine's."""
    for name in (
        "install the wheel and run it",
        "install the wheel and run it (musl, in Alpine)",
    ):
        step = _step(BUILD, name)
        assert step["env"]["DISTRIBUTION"] == (
            "${{ matrix.variant == 'compat' && 'polars-online-compat' || 'polars-online' }}"
        )
        assert step["env"]["FEATURES"] == (
            "${{ matrix.x86 && matrix.variant == 'default' && env.X86_FEATURES || '' }}"
        )
        assert '--distribution "$DISTRIBUTION" --features "$FEATURES"' in step["run"]
    assert (
        "-e DISTRIBUTION -e FEATURES"
        in _step(BUILD, "install the wheel and run it (musl, in Alpine)")["run"]
    )


def test_the_sdist_job_also_builds_the_compat_source_distribution():
    steps = JOBS["sdist"]["steps"]
    names = [s.get("name", "") for s in steps]
    rename = names.index("the compat package's source distribution")
    assert "scripts/compat_package.py" in steps[rename]["run"]
    # After the first sdist is built and kept, into a folder of its own, and
    # before the install that reads `dist/` alone.
    build = steps[rename + 1]
    assert build["with"]["command"] == "sdist" and "--out dist-compat" in build["with"]["args"]
    upload = steps[rename + 2]
    assert upload["with"] == {"name": "sdist-compat", "path": "dist-compat/*.tar.gz"}
    smoke = next(i for i, s in enumerate(steps) if "wheel_smoke.py" in str(s.get("run", "")))
    assert rename < smoke


def test_both_projects_are_uploaded_from_their_own_folders_compat_first():
    job = JOBS["publish"]
    uploads = [s for s in job["steps"] if "gh-action-pypi-publish" in s.get("uses", "")]
    assert [(s["name"], s["with"]["packages-dir"]) for s in uploads] == [
        ("upload polars-online-compat", "dist-compat"),
        ("upload polars-online", "dist"),
    ]
    # The job can be run again after a failure between the two.
    assert all(s["with"]["skip-existing"] is True for s in uploads)
    assert job["environment"] == "pypi"
    assert job["permissions"] == {"id-token": "write"}


def test_the_release_checks_the_compat_wheels_sizes_too():
    job = JOBS["wheel-size"]
    (download,) = [s for s in job["steps"] if "download-artifact" in s.get("uses", "")]
    # `wheel-*` is every wheel artifact, whichever variant.
    assert download["with"]["pattern"] == "wheel-*"
    assert "wheel-${{ matrix.target }}-${{ matrix.variant }}".startswith("wheel-")


# --- the smoke test's checks ---------------------------------------------------------------


@pytest.fixture
def installed(monkeypatch):
    """Which distributions the environment has installed."""
    have: set[str] = set()

    def version(name: str) -> str:
        if name not in have:
            raise wheel_smoke.importlib.metadata.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(wheel_smoke.importlib.metadata, "version", version)
    return have


def _cpu(monkeypatch, flags: str, missing: list[str]) -> None:
    from rheon import _cpu_check

    monkeypatch.setattr(_cpu_check, "baked_flags", lambda: flags)
    monkeypatch.setattr(_cpu_check, "missing_features", lambda *_: missing)


def test_the_smoke_test_passes_the_right_distribution_with_the_right_features(
    installed, monkeypatch
):
    installed.add("polars-online")
    _cpu(monkeypatch, POLARS_FEATURES, [])
    wheel_smoke.check_build("polars-online", POLARS_FEATURES)
    installed.clear()
    installed.add("polars-online-compat")
    _cpu(monkeypatch, "", [])
    wheel_smoke.check_build("polars-online-compat", "")


def test_the_smoke_test_refuses_the_other_project_or_both_or_neither(installed, monkeypatch):
    _cpu(monkeypatch, "", [])
    with pytest.raises(SystemExit, match="polars-online-compat is not installed"):
        wheel_smoke.check_build("polars-online-compat", "")
    installed.add("polars-online")
    with pytest.raises(SystemExit, match="polars-online is installed"):
        wheel_smoke.check_build("polars-online-compat", "")
    installed.add("polars-online-compat")
    with pytest.raises(SystemExit, match="polars-online is installed"):
        wheel_smoke.check_build("polars-online-compat", "")
    installed.discard("polars-online")
    wheel_smoke.check_build("polars-online-compat", "")
    with pytest.raises(SystemExit, match="unknown distribution"):
        wheel_smoke.check_build("polars", "")


def test_the_smoke_test_refuses_a_build_that_recorded_other_features(installed, monkeypatch):
    installed.add("polars-online")
    _cpu(monkeypatch, "", [])  # the record is missing: a default x86-64 wheel with no check
    with pytest.raises(SystemExit, match="recorded the CPU features ''"):
        wheel_smoke.check_build("polars-online", POLARS_FEATURES)
    _cpu(monkeypatch, POLARS_FEATURES, [])  # the compat wheel built with the flags
    with pytest.raises(SystemExit, match="not the expected ''"):
        wheel_smoke.check_build("polars-online", "")


def test_the_smoke_test_refuses_a_runner_without_a_feature_the_wheel_needs(installed, monkeypatch):
    installed.add("polars-online")
    _cpu(monkeypatch, POLARS_FEATURES, ["avx2"])
    with pytest.raises(SystemExit, match="lacks avx2"):
        wheel_smoke.check_build("polars-online", POLARS_FEATURES)


def test_a_checkout_is_smoke_tested_without_a_distribution():
    """The check this suite runs on the build in this environment, which has
    no distribution to hold to a name."""
    from rheon import polars as rp

    assert wheel_smoke.check(rp.__version__, installed=False).endswith(": ok")


def test_the_wheel_size_check_reads_the_compat_wheels_like_any_other(tmp_path):
    """Both projects' wheels go through the one check and one notes table."""
    import zipfile

    size = _script("wheel_size")
    paths = []
    for name in ("rheon-1-x.whl", "polars_online_compat-1-x.whl"):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("a.bin", b"0" * 100)
        paths.append(path)
    table = size.notes([size.measure(p) for p in paths])
    assert "`rheon-1-x.whl`" in table and "`polars_online_compat-1-x.whl`" in table


# --- the import path -------------------------------------------------------------------------


def test_the_package_runs_the_check_before_any_other_of_its_imports():
    """The extension module is imported through the package's other modules,
    so the check comes before every one of them."""
    tree = ast.parse((REPO / "python" / "rheon" / "__init__.py").read_text("utf-8"))
    body = [
        n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]

    def imports_the_package(node: ast.AST) -> bool:
        return isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rheon")

    first = next(i for i, n in enumerate(body) if imports_the_package(n))
    assert [a.name for a in body[first].names] == ["_cpu_check"]  # type: ignore[attr-defined]
    call = body[first + 1]
    assert isinstance(call, ast.Expr) and ast.unparse(call.value) == "_cpu_check.run()"
    assert not any(imports_the_package(n) for n in body[:first])
    assert not any(isinstance(n, ast.Import) for n in body[:first])


def test_the_check_module_imports_nothing_of_the_package_but_the_build_record():
    """It runs before the package's other modules, so it can use none of
    them, and it reads the build's record by name, never by import."""
    tree = ast.parse((REPO / "python" / "rheon" / "_cpu_check.py").read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            )
            assert not any(n and n.startswith("polars") for n in names), ast.unparse(node)


def test_the_build_record_is_not_committed():
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "python/rheon/_build_flags.py"],
        cwd=REPO,
        check=False,
    )
    assert ignored.returncode == 0
