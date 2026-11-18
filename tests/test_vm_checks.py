"""The VM checks workflow (`.github/workflows/vm-checks.yml`) and `scripts/vm_checks.py`.

The x86 VMs time wall-clock; the workflow builds the wheels they time, so a
VM that restarts mid-build loses nothing. It is a dispatch by hand, never a
push, one job after another steps so a request holds one runner, and it
reads the repository's artifacts and nothing else. These hold its shape to
that, and the script's pure parts -- the layouts, the names, the base's
lookup, the test counts, the digest table -- to what the workflow and the
record (`docs/records/VM-CHECKS.md`) say.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
FILE = WORKFLOWS / "vm-checks.yml"
TEXT = FILE.read_text(encoding="utf-8")
WF = yaml.safe_load(TEXT)
# PyYAML reads the key `on` as the boolean True.
TRIGGERS = WF.get("on", WF.get(True))
JOB = WF["jobs"]["checks"]


def _load_script():
    spec = importlib.util.spec_from_file_location("vm_checks", REPO / "scripts" / "vm_checks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vm_checks"] = module
    spec.loader.exec_module(module)
    return module


vm = _load_script()


# --- the workflow's shape ------------------------------------------------------------


def test_it_is_dispatched_by_hand_and_never_by_a_push_or_a_pull_request():
    """A build nobody asked for saves nobody's time, and a pull request from a
    fork would run it."""
    assert set(TRIGGERS) == {"workflow_dispatch"}


def test_the_dispatch_inputs():
    inputs = TRIGGERS["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "ref",
        "variants",
        "base",
        "digests",
        "tests",
        "test_package",
        "qemu_cpu",
    }
    assert inputs["ref"]["required"] is True
    assert inputs["variants"]["default"] == "default,aligned"
    assert inputs["base"]["default"] == ""
    assert inputs["digests"]["type"] == "boolean"
    assert inputs["digests"]["default"] is False
    assert inputs["tests"]["default"] == ""
    assert inputs["test_package"]["default"] == "online-core"
    assert inputs["qemu_cpu"]["default"] == ""


def test_one_job_and_no_matrix():
    """A request holds one runner: a cap on concurrent jobs is shared with CI."""
    assert list(WF["jobs"]) == ["checks"]
    assert "strategy" not in JOB
    assert "matrix" not in TEXT.replace("no matrix", "")


def test_the_job_is_bounded_and_on_the_image_the_others_use():
    assert JOB["timeout-minutes"] <= 120
    assert JOB["runs-on"] == "ubuntu-24.04"
    budgets = yaml.safe_load((WORKFLOWS / "budgets.yml").read_text(encoding="utf-8"))
    assert budgets["jobs"]["instructions"]["runs-on"] == JOB["runs-on"]


def test_a_second_request_for_a_branch_replaces_the_first():
    """Per branch asked for: dispatched on main, `github.ref` would be one
    group for every request."""
    assert WF["concurrency"]["cancel-in-progress"] is True
    group = WF["concurrency"]["group"]
    assert group.startswith("vm-checks-")
    assert "inputs.ref" in group


def test_it_reads_the_code_and_the_artifacts_and_nothing_more():
    assert WF["permissions"] == {"contents": "read", "actions": "read"}
    assert JOB["permissions"] == {"contents": "read", "actions": "read"}
    # No secret beyond the job's own token.
    assert "secrets." not in TEXT
    assert set(re.findall(r"\$\{\{\s*github\.token\s*\}\}", TEXT)) <= {"${{ github.token }}"}


def _uses() -> list[str]:
    out = [s["uses"] for s in JOB["steps"] if "uses" in s]
    assert out
    return out


def test_every_action_is_pinned_to_a_full_sha_the_other_workflows_use():
    """Not a tag a maintainer can move; and the same commit the other
    workflows run, so there is one version of each to review."""
    others = " ".join(
        p.read_text(encoding="utf-8") for p in WORKFLOWS.glob("*.yml") if p.name != FILE.name
    )
    for use in _uses():
        if use.startswith("./"):
            assert (REPO / use / "action.yml").exists(), use
            continue
        action, _, ref = use.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", ref), use
        assert f"{action}@{ref}" in others, f"{use}: not a pin the other workflows use"


def test_the_untrusted_inputs_reach_a_script_through_the_environment():
    """An input spliced into a `run:` is a command injection: a branch name
    is chosen by whoever pushes it."""
    for step in JOB["steps"]:
        assert "inputs." not in step.get("run", ""), step.get("name")
        assert "steps.plan.outputs" not in step.get("run", ""), step.get("name")


def test_the_digest_and_test_steps_are_conditional_and_follow_the_uploads():
    steps = JOB["steps"]
    names = [s.get("name", "") for s in steps]
    digests = next(s for s in steps if s.get("name", "").startswith("compare the outputs"))
    tests = next(s for s in steps if s.get("name", "").startswith("the tests"))
    assert digests["if"] == "inputs.digests"
    assert tests["if"] == "inputs.tests != ''"
    last_wheel_upload = max(
        i for i, n in enumerate(names) if n.startswith("keep the") and "wheel" in n
    )
    assert names.index(digests["name"]) > last_wheel_upload
    assert names.index(tests["name"]) > last_wheel_upload
    # Correctness tests: the run fails with them, not despite them.
    assert "continue-on-error" not in digests and "continue-on-error" not in tests
    assert "scripts/vm_checks.py tests" in tests["run"]


def test_the_tools_come_from_the_workflows_commit_and_the_branch_is_a_subdirectory():
    """A vm/ branch predates the tools: the workspace root is the commit the
    workflow runs from (a checkout with no `ref`), the branch is checked out in
    `branch/` with its history, and every command that touches the branch
    is pointed at it."""
    steps = JOB["steps"]
    checkouts = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 2
    tools, branch = checkouts
    assert "ref" not in tools.get("with", {}) and "path" not in tools.get("with", {})
    assert branch["with"]["ref"] == "${{ inputs.ref }}"
    assert branch["with"]["path"] == "branch"
    assert branch["with"]["fetch-depth"] == 0
    # The tools' checkout comes first, and before the local action it carries.
    names = [s.get("uses", s.get("name")) for s in steps]
    assert (
        steps.index(tools) < steps.index(branch) < names.index("./.github/actions/linux-build-prep")
    )
    cache = next(s for s in steps if s.get("uses", "").startswith("Swatinem/rust-cache"))
    assert cache["with"]["workspaces"] == "branch"
    for s in steps:
        run = s.get("run", "")
        for sub in ("plan", "wheels", "tests"):
            if f"scripts/vm_checks.py {sub}" in run:
                assert "--repo branch" in run, sub
        # The script is the root's, never `branch/scripts/...`.
        assert "branch/scripts" not in run


def test_the_script_runs_the_comparison_tool_from_its_own_tree_not_the_branchs():
    src = (REPO / "scripts" / "vm_checks.py").read_text(encoding="utf-8")
    assert 'REPO / "scripts" / "compare_outputs.py"' in src
    assert 'repo / "scripts"' not in src


def test_a_build_output_is_absolute_because_the_build_runs_in_the_checkout(
    repo, monkeypatch, tmp_path
):
    seen = {}

    def run(cmd, cwd=None, env=None, check=None):
        seen["out"] = Path(cmd[cmd.index("--out") + 1])
        seen["cwd"] = cwd
        (seen["out"] / "x-py3-none-linux_x86_64.whl").write_bytes(b"")

    monkeypatch.setattr(vm, "_run", lambda cmd, **kw: "/usr/bin/python3\n")
    monkeypatch.setattr(vm, "maturin_requirement", lambda repo: "maturin")
    monkeypatch.setattr(vm.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    vm.build_wheel(repo, "default", tmp_path / "t", Path("dist") / "w")
    assert seen["out"].is_absolute() and seen["out"] == tmp_path / "dist" / "w"
    assert seen["cwd"] == repo


def test_qemu_is_installed_only_when_a_cpu_is_asked_for():
    prep = next(s for s in JOB["steps"] if s.get("uses", "").endswith("linux-build-prep"))
    packages = prep["with"]["packages"]
    assert "inputs.qemu_cpu != ''" in packages and "qemu-user" in packages


def test_a_wheel_is_uploaded_under_the_name_a_vm_looks_for():
    """`wheel-<sha8>-<layout>`, for each layout, the branch's and the base's."""
    uploads = [s for s in JOB["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    names = {s["with"]["name"] for s in uploads}
    for role in ("head8", "base8"):
        for layout in vm.LAYOUTS:
            assert f"wheel-${{{{ steps.plan.outputs.{role} }}}}-{layout}" in names
    for s in uploads:
        if s["with"]["name"].startswith("wheel-"):
            # Nothing to upload where a layout was not asked for or a base reused.
            assert s["with"]["if-no-files-found"] == "ignore"
            assert s["with"]["path"].startswith("dist/wheel-")


def test_the_workflow_calls_only_commands_the_script_has():
    runs = " ".join(s.get("run", "") for s in JOB["steps"])
    subcommands = set(re.findall(r"scripts/vm_checks\.py (\w+)", runs))
    assert subcommands == {"plan", "wheels", "digests", "tests"}
    helps = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "vm_checks.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    for name in subcommands:
        assert name in helps


def test_every_layout_the_workflow_names_is_one_the_script_builds():
    assert set(vm.LAYOUTS) == {"default", "aligned", "v3"}
    assert "align-all-functions=6" in vm.LAYOUTS["aligned"]
    assert "target-cpu=x86-64-v3" in vm.LAYOUTS["v3"]
    assert vm.LAYOUTS["default"] == ""


# --- the script ---------------------------------------------------------------------------


def test_layouts_are_parsed_in_order_without_repeats():
    assert vm.parse_layouts("default,aligned") == ["default", "aligned"]
    assert vm.parse_layouts("v3 aligned,v3") == ["v3", "aligned"]
    with pytest.raises(ValueError, match="unknown variant 'avx512'"):
        vm.parse_layouts("default,avx512")
    with pytest.raises(ValueError, match="no variant"):
        vm.parse_layouts("")


def test_digests_need_the_default_layout_and_get_it():
    assert vm.parse_layouts("aligned", digests=True) == ["default", "aligned"]
    assert vm.parse_layouts("default,v3", digests=True) == ["default", "v3"]


def test_every_build_names_the_linker_and_only_the_layout_differs():
    assert vm.rustflags("default") == vm.LINKER
    for layout in ("aligned", "v3"):
        flags = vm.rustflags(layout)
        assert flags.endswith(vm.LINKER) and vm.LAYOUTS[layout] in flags


def test_the_artifact_name_is_the_short_sha_and_the_layout():
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert vm.wheel_name(sha, "aligned") == "wheel-01234567-aligned"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "f").write_text("0", encoding="utf-8")
    _git(tmp_path, "add", "f")
    _git(tmp_path, "commit", "-q", "-m", "m0")
    # A remote-tracking `origin/main` without a remote.
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(tmp_path, "checkout", "-q", "-b", "vm/x")
    (tmp_path / "f").write_text("1", encoding="utf-8")
    _git(tmp_path, "commit", "-q", "-am", "m1")
    return tmp_path


def test_the_base_is_the_merge_base_with_origin_main_unless_named(repo):
    main = _git(repo, "rev-parse", "origin/main")
    head = _git(repo, "rev-parse", "HEAD")
    assert vm.plan(repo) == (head, main)
    assert vm.plan(repo, main[:10]) == (head, main)
    # A branch only the remote holds is found by its short name.
    _git(repo, "update-ref", "refs/remotes/origin/other", main)
    assert vm.plan(repo, "other") == (head, main)
    with pytest.raises(ValueError, match="names no commit"):
        vm.plan(repo, "no-such-branch")


def _listing(*artifacts: dict) -> str:
    return json.dumps({"total_count": len(artifacts), "artifacts": list(artifacts)})


def _artifact(name: str, run: int, created: str, expired: bool = False) -> dict:
    return {
        "name": name,
        "expired": expired,
        "created_at": created,
        "workflow_run": {"id": run},
    }


def test_an_earlier_run_is_the_newest_unexpired_artifact_of_that_name(monkeypatch):
    seen = []

    def gh(*args):
        seen.append(args)
        return _listing(
            _artifact("wheel-aaaaaaaa-default", 1, "2026-11-01T00:00:00Z"),
            _artifact("wheel-aaaaaaaa-default", 3, "2026-11-03T00:00:00Z", expired=True),
            _artifact("wheel-aaaaaaaa-default", 2, "2026-11-02T00:00:00Z"),
            _artifact("wheel-bbbbbbbb-default", 9, "2026-11-09T00:00:00Z"),
        )

    monkeypatch.setattr(vm, "_gh", gh)
    assert vm.earlier_run("wheel-aaaaaaaa-default", "o/r") == "2"
    assert "name=wheel-aaaaaaaa-default" in seen[0][1] and "repos/o/r/" in seen[0][1]
    assert vm.earlier_run("wheel-cccccccc-default", "o/r") is None


def _fake_builds(monkeypatch, tmp_path, kept: dict):
    """Builds that write a wheel and note which commit and layout; lookups
    answered from ``kept`` (artifact name -> run)."""
    built = []

    def build_wheel(repo, layout, target_dir, out):
        out.mkdir(parents=True, exist_ok=True)
        wheel = out / f"polars_online-{layout}.whl"
        wheel.write_text("", encoding="utf-8")
        built.append((Path(repo).name, layout, Path(target_dir)))
        return wheel

    def fetch(name, run, repository, into):
        into.mkdir(parents=True, exist_ok=True)
        wheel = into / "kept.whl"
        wheel.write_text("", encoding="utf-8")
        return wheel

    monkeypatch.setattr(vm, "build_wheel", build_wheel)
    monkeypatch.setattr(vm, "fetch", fetch)
    monkeypatch.setattr(vm, "earlier_run", lambda name, repository: kept.get(name))
    return built


def test_the_base_is_built_unless_an_earlier_run_kept_its_wheel(repo, monkeypatch, tmp_path):
    head, base = vm.plan(repo)
    kept = {vm.wheel_name(base, "default"): "42"}
    built = _fake_builds(monkeypatch, tmp_path, kept)
    result = vm.make_wheels(
        repo, ["default", "aligned"], head, base, tmp_path / "dist", tmp_path / "got", "o/r"
    )
    assert result["reused"] == {vm.wheel_name(base, "default"): "42"}
    # The branch in both layouts, the base in the one nobody kept.
    assert [(who, layout) for who, layout, _ in built] == [
        (repo.name, "default"),
        (repo.name, "aligned"),
        (".vm-checks-base", "aligned"),
    ]
    assert result["wheels"]["base"]["default"].startswith(str(tmp_path / "got"))
    assert result["wheels"]["base"]["aligned"].startswith(str(tmp_path / "dist"))
    assert (tmp_path / "dist" / vm.wheel_name(head, "aligned")).is_dir()
    # The scratch target directories of the other layouts are gone; the
    # default layout's is the checkout's own, which the job's cache keeps.
    scratch = [t for _, layout, t in built if layout == "aligned"]
    assert scratch and not any(t.exists() for t in scratch)
    assert not (repo / ".vm-checks-base").exists()
    assert _git(repo, "worktree", "list").count("\n") == 0  # the base's worktree is removed


def test_a_relative_checkout_puts_the_base_worktree_inside_it_once(repo, monkeypatch, tmp_path):
    # The workflow passes `--repo branch`, relative to the tools' checkout;
    # `git -C branch worktree add branch/...` would nest the worktree inside
    # itself, and the base's build then found no pyproject.toml (the first
    # dispatch, run 44320126091).
    head, base = vm.plan(repo)
    built = _fake_builds(monkeypatch, tmp_path, {})
    monkeypatch.chdir(repo.parent)
    vm.make_wheels(
        Path(repo.name), ["default"], head, base, tmp_path / "dist", tmp_path / "got", "o/r"
    )
    bases = [where for who, _, where in built if who == ".vm-checks-base"]
    assert [who for who, _, _ in built] == [repo.name, ".vm-checks-base"]
    assert not (repo / repo.name).exists()
    assert _git(repo, "worktree", "list").count("\n") == 0
    assert bases


def test_a_branch_at_its_base_builds_once(repo, monkeypatch, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    built = _fake_builds(monkeypatch, tmp_path, {})
    result = vm.make_wheels(repo, ["default"], head, head, tmp_path / "d", tmp_path / "g", "o/r")
    assert len(built) == 1
    assert result["wheels"]["base"] == result["wheels"]["branch"]


# --- the test step ---------------------------------------------------------------------------


OUTPUT = """\
running 3 tests
test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 10 filtered out; finished in 0.1s

running 2 tests
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s
"""


def test_counts_are_summed_over_every_test_binary():
    assert vm.parse_counts(OUTPUT) == {"passed": 4, "failed": 1, "ignored": 1}
    assert vm.parse_counts("nothing") == {"passed": 0, "failed": 0, "ignored": 0}


def test_a_run_fails_on_an_exit_a_failure_or_a_filter_that_matched_nothing():
    ok = {"exit": 0, "passed": 5, "failed": 0, "ignored": 0}
    assert vm.verdict(ok) is None
    assert "exited 101" in vm.verdict({**ok, "exit": 101})
    assert "1 failed" in vm.verdict({**ok, "failed": 1})
    # `cargo test` exits 0 on a filter that matches no test.
    assert "matched no test" in vm.verdict({**ok, "passed": 0})


def test_the_test_binaries_are_the_test_profile_artifacts_with_their_package_dirs():
    lines = [
        {"reason": "compiler-artifact", "executable": None, "profile": {"test": False}},
        {
            "reason": "compiler-artifact",
            "executable": "/t/release/deps/online_core-abc",
            "profile": {"test": True},
            "manifest_path": "/w/crates/online-core/Cargo.toml",
        },
        {
            "reason": "compiler-artifact",
            "executable": "/t/release/deps/some_bin",
            "profile": {"test": False},
            "manifest_path": "/w/crates/online-core/Cargo.toml",
        },
        {"reason": "build-finished", "success": True},
    ]
    text = "\n".join(json.dumps(m) for m in lines) + "\nnot json\n"
    assert vm.built_test_binaries(text) == [
        (Path("/t/release/deps/online_core-abc"), Path("/w/crates/online-core"))
    ]


def test_the_tests_table_says_each_run():
    rows = [
        {"run": "native", "exit": 0, "passed": 7, "failed": 0, "ignored": 1},
        {"run": "qemu -cpu Nehalem", "exit": 0, "passed": 0, "failed": 0, "ignored": 0},
    ]
    table = vm.tests_table("online-core", ["solve::"], rows)
    assert "| native | 7 | 0 | 1 | ok |" in table
    assert "| qemu -cpu Nehalem | 0 | 0 | 0 | the filter matched no test |" in table
    assert "cargo test -p online-core --release solve::" in table


# --- the digest table -----------------------------------------------------------------------


def test_the_digest_table_names_the_columns_that_moved():
    report = {
        "floor": 1e-12,
        "cases": [
            {
                "name": "smoke/a",
                "status": "ok",
                "state": "same",
                "columns": {
                    "m.pred": {"rows_differing": 3, "max_rel": 4.4e-16},
                    "m.coef": {"rows_differing": 0, "max_rel": 0.0},
                },
                "summary": {
                    "max_rel": 4.4e-16,
                    "rows_any_differ": 3,
                    "flipped_columns": {},
                    "decision_flips": {"flag": 2},
                },
            },
            {"name": "smoke/b", "status": "not comparable", "why": "base: crashed"},
        ],
    }
    table = vm.digest_table(report, "a" * 40, "b" * 40)
    row = next(line for line in table.splitlines() if line.startswith("| smoke/a"))
    # columns moved, max relative, in ulps (4.4e-16 / 2^-52), rows differing, flips, state
    assert row == "| smoke/a | ok | `m.pred` | 4.4e-16 | 1.98 | 3 | 2 | same |"
    assert "not comparable: base: crashed" in table
    assert "`aaaaaaaa`" in table and "`bbbbbbbb`" in table


def test_a_case_that_moved_nothing_says_none():
    report = {
        "cases": [
            {
                "name": "c",
                "status": "ok",
                "columns": {"x": {"rows_differing": 0}},
                "summary": {
                    "max_rel": 0.0,
                    "rows_any_differ": 0,
                    "flipped_columns": {},
                    "decision_flips": {},
                },
            }
        ]
    }
    assert "| none | 0 |" in vm.digest_table(report, "a" * 40, "b" * 40)


def _fake_tools(tmp_path: Path, monkeypatch, passed: int) -> list[str]:
    """A `cargo` and a `qemu-x86_64` on the PATH that answer as the real ones
    do; the qemu one records its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "qemu.log"
    exe = tmp_path / "online_core-abc"
    exe.write_text("", encoding="utf-8")
    artifact = json.dumps(
        {
            "reason": "compiler-artifact",
            "executable": str(exe),
            "profile": {"test": True},
            "manifest_path": str(tmp_path / "Cargo.toml"),
        }
    )
    result = f"test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    cargo = bin_dir / "cargo"
    cargo.write_text(
        f"#!/bin/sh\ncase \"$*\" in *--no-run*) echo '{artifact}';;\n*) echo '{result}';;\nesac\n",
        encoding="utf-8",
    )
    qemu = bin_dir / "qemu-x86_64"
    qemu.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\necho '{result}'\n", encoding="utf-8")
    for tool in (cargo, qemu):
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return [str(log)]


def test_the_tests_run_natively_and_under_qemu_with_the_same_filter(tmp_path, monkeypatch):
    (log,) = _fake_tools(tmp_path, monkeypatch, passed=4)
    rows = vm.run_tests(tmp_path, "online-core", ["solve::"], "Nehalem")
    assert [r["run"] for r in rows] == ["native", "qemu -cpu Nehalem"]
    assert [r["passed"] for r in rows] == [4, 4]
    assert all(vm.verdict(r) is None for r in rows)
    ran = Path(log).read_text(encoding="utf-8")
    assert ran.startswith("-cpu Nehalem ") and "solve:: --include-ignored" in ran


def test_a_filter_that_matches_nothing_fails_the_step(tmp_path, monkeypatch, capsys):
    _fake_tools(tmp_path, monkeypatch, passed=0)
    args = vm.argparse.Namespace(
        repo=tmp_path, package="online-core", filter="nothing::", qemu_cpu=""
    )
    assert vm._cmd_tests(args) == 1
    assert "the filter matched no test" in capsys.readouterr().out


def test_an_empty_filter_is_refused_not_run_as_every_test(tmp_path):
    args = vm.argparse.Namespace(repo=tmp_path, package="p", filter="  ", qemu_cpu="")
    with pytest.raises(SystemExit, match="empty filter"):
        vm._cmd_tests(args)
