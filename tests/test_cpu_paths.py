"""`.github/workflows/cpu-paths.yml`, `scripts/cpu_paths.py`,
`scripts/cpu_probe.rs` and the `oracle` mark (docs/TESTING.md, "The machines
and the CPU paths"; task 273 (b)).

The workflow forces the CPU paths the numeric tests take, weekly, instead of
sampling GitHub's runner pool: online-core's numeric tests under Intel SDE
posing as chips with and without AVX2 and AVX-512 and with a resized L3, and
the Python oracle tests with OpenBLAS's kernels and NumPy's loops pinned.
None of it can run here, so these hold its parts: the script's arithmetic on
canned inputs, the probe's build, the mark's selection, and the workflow's
shape, each leg proving its path took effect before its tests count."""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

import oracle_marks

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cpu_paths", REPO / "scripts/cpu_paths.py")
assert _spec and _spec.loader
cp = importlib.util.module_from_spec(_spec)
sys.modules["cpu_paths"] = cp
_spec.loader.exec_module(cp)


def _workflow() -> dict:
    d = yaml.safe_load((REPO / ".github/workflows/cpu-paths.yml").read_text(encoding="utf-8"))
    d["on"] = d.pop(True, d.get("on"))
    return d


WF = _workflow()


def _artifact(kind: str, exe: str | None) -> str:
    return json.dumps(
        {
            "reason": "compiler-artifact",
            "target": {"kind": [kind], "name": "x"},
            "executable": exe,
        }
    )


class TestTheScript:
    def test_the_library_test_executable_is_the_one_lib_artifact(self):
        lines = [
            "not json",
            _artifact("lib", None),  # the library itself, no executable
            _artifact("test", "/t/golden-1"),
            _artifact("lib", "/t/online_core-2"),
            json.dumps({"reason": "build-finished", "success": True}),
        ]
        assert cp.lib_test_executable(lines) == "/t/online_core-2"

    @pytest.mark.parametrize("count", [0, 2])
    def test_none_or_two_is_refused(self, count):
        lines = [_artifact("lib", f"/t/online_core-{i}") for i in range(count)]
        with pytest.raises(SystemExit, match="expected one library test executable"):
            cp.lib_test_executable(lines)

    #: A CPUID definition file's shape: leaf 4's four subleaves of a chip
    #: with a 32 KiB L1d, a 32 KiB L1i, a 256 KiB L2 and a 6 MiB L3 of 12
    #: ways, 64-byte lines and 8,192 sets; and two other leaves.
    DEF = (
        "# a comment\n"
        "00000000 ******** => 0000000d 756e6547 6c65746e 49656e69\n"
        "00000004 00000000 => 1c004121 01c0003f 0000003f 00000000\n"
        "00000004 00000001 => 1c004122 01c0003f 0000003f 00000000\n"
        "00000004 00000002 => 1c004143 01c0003f 000001ff 00000000\n"
        "00000004 00000003 => 1c03c163 02c0003f 00001fff 00000006\n"
        "80000006 ******** => 00000000 00000000 01006040 00000000\n"
    )

    @staticmethod
    def _l3_kib(text: str) -> int:
        for line in text.splitlines():
            m = cp.DEF_LINE.match(line)
            if m and m["leaf"] == "00000004" and int(m["eax"], 16) >> 5 & 7 == 3:
                ebx, ecx = int(m["ebx"], 16), int(m["ecx"], 16)
                ways, parts, size = (ebx >> 22) + 1, ((ebx >> 12) & 0x3FF) + 1, (ebx & 0xFFF) + 1
                return ways * parts * size * (ecx + 1) // 1024
        raise AssertionError("no L3")

    @pytest.mark.parametrize("kib", [2048, 6144, 1536, 8192])
    def test_the_l3_is_resized_and_nothing_else_moves(self, kib):
        assert self._l3_kib(self.DEF) == 6144
        out = cp.resize_l3(self.DEF, kib)
        assert self._l3_kib(out) == kib
        changed = [
            (a, b) for a, b in zip(self.DEF.splitlines(), out.splitlines(), strict=True) if a != b
        ]
        assert len(changed) == (kib != 6144), changed
        for a, b in changed:
            assert a.startswith("00000004 00000003 => 1c03c163 ") and b.endswith(" 00000006")

    def test_a_file_without_an_l3_is_refused(self):
        no_l3 = "".join(ln + "\n" for ln in self.DEF.splitlines() if "00000003 =>" not in ln)
        with pytest.raises(SystemExit, match="one L3 subleaf"):
            cp.resize_l3(no_l3, 2048)

    def test_the_probe_is_held_to_each_expectation(self):
        seen = cp.parse_probe("vendor=GenuineIntel\nbrand=x=y\navx2=true\nl3_kib=2048\n")
        assert seen["brand"] == "x=y"
        assert cp.unmet(seen, ["avx2=true", "l3_kib=2048"]) == []
        assert cp.unmet(seen, ["avx2=false", "avx512f=true"]) == [
            "avx2: expected false, saw true",
            "avx512f: expected true, saw None",
        ]

    LIST = "a::one: test\na::two: test\nb::bench: benchmark\n\n2 tests, 1 benchmark\n"

    def test_a_run_passes_only_when_every_listed_test_passed(self):
        assert cp.listed(self.LIST) == 2
        ok = "running 2 tests\n..\ntest result: ok. 2 passed; 0 failed; 0 ignored;"
        assert cp.passed_all(ok, 2) is None
        assert "of 3 listed" in cp.passed_all(ok, 3)
        bad = "test result: FAILED. 1 passed; 1 failed; 0 ignored;"
        assert cp.passed_all(bad, 2) == "FAILED: 1 passed, 1 failed, of 2 listed"
        assert cp.passed_all("", 2) == "no test result line"

    DISPATCH = ["X86_V3", "X86_V4", "AVX512_ICL", "AVX512_SPR"]

    def test_numpy_keeps_what_it_is_told_and_disables_the_rest(self):
        assert cp.numpy_disable(self.DISPATCH, ["X86_V3"]) == ["X86_V4", "AVX512_ICL", "AVX512_SPR"]
        assert cp.numpy_disable(self.DISPATCH, []) == self.DISPATCH
        with pytest.raises(SystemExit, match="not among NumPy's dispatched features"):
            cp.numpy_disable(self.DISPATCH, ["AVX2"])

    def test_the_workflow_passes_sdes_knobs_as_values(self):
        """argparse reads `--sde-args -hsw` as two options and stops; the
        workflow writes `--sde-args=...`."""
        runs = " ".join(s.get("run", "") for s in WF["jobs"]["sde"]["steps"])
        assert runs.count("--sde-args=") == 2 and "--sde-args " not in runs, runs
        with pytest.raises(SystemExit):
            cp.main(["probe", "p", "--sde", "s", "--sde-args", "-hsw", "--expect", "a=b"])

    def test_sde_keeps_the_definition_files_leaves(self):
        assert cp._sde("sde.exe", "-cpuid_in D:/a/cpuid.def") == [
            "sde.exe",
            "-cpuid_in",
            "D:/a/cpuid.def",
            "-update_cpuid_from_host",
            "0",
            "--",
        ]


class TestTheProbe:
    def test_it_builds_and_reports_on_this_machine(self, tmp_path):
        exe = tmp_path / ("cpu_probe.exe" if sys.platform == "win32" else "cpu_probe")
        cmd = ["rustc", "-O", "--edition", "2024", str(REPO / "scripts/cpu_probe.rs"), "-o"]
        try:
            subprocess.run(
                [*cmd, str(exe)], check=True, capture_output=True, text=True, encoding="utf-8"
            )
        except FileNotFoundError:
            pytest.fail("rustc is not on PATH; `source scripts/env.sh` first")
        run = subprocess.run(
            [str(exe)], check=True, capture_output=True, text=True, encoding="utf-8"
        )
        out = cp.parse_probe(run.stdout)
        if platform.machine().lower() in ("x86_64", "amd64"):
            for key in ("vendor", "sse3", "avx", "avx2", "fma", "avx512f"):
                assert key in out, (key, out)
            # Every x86-64 machine has SSE3.
            assert out["sse3"] == "true", out
        else:
            assert "arch" in out, out


class TestTheOracleMark:
    def test_it_is_registered(self):
        meta = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        markers = meta["tool"]["pytest"]["ini_options"]["markers"]
        assert any(m.startswith("oracle:") for m in markers)

    @pytest.mark.parametrize(
        ("source", "oracle"),
        [
            ("import scipy.stats\n", True),
            ("from statsmodels.api import OLS\n", True),
            ("def f():\n    from sklearn.linear_model import Ridge\n", True),
            ("import reference\n", True),
            ("from reference_paths import sgd_ref\n", True),
            ("import numpy as np\nx = np.linalg.solve\n", True),
            ("from numpy import linalg\n", True),
            ("import numpy as np\nx = np.exp(1.0)\n", False),
            ("from . import reference\n", False),
            ("import polars as pl\n", False),
        ],
    )
    def test_a_module_is_one_by_what_it_imports_or_calls(self, tmp_path, source, oracle):
        path = tmp_path / "m.py"
        path.write_text(source, encoding="utf-8")
        assert oracle_marks.is_oracle_module(path) is oracle

    def test_this_repos_modules_are_sorted_and_the_cli_is_left_out(self):
        second = REPO / "tests/test_second_opinion.py"
        assert oracle_marks.is_oracle_test(second, ["tmp_path"])
        assert not oracle_marks.is_oracle_test(second, ["tmp_path", "online_cli"])
        assert not oracle_marks.is_oracle_module(REPO / "tests/test_ci_cost_policy.py")

    def test_a_test_here_is_not_one(self, request):
        assert "oracle" not in {m.name for m in request.node.iter_markers()}


class TestTheWorkflow:
    def test_it_runs_weekly_and_by_hand_on_windows(self):
        assert set(WF["on"]) == {"schedule", "workflow_dispatch"}
        (cron,) = (e["cron"] for e in WF["on"]["schedule"])
        minute, hour, day, month, weekday = cron.split()
        assert (day, month) == ("*", "*") and weekday.isdigit(), cron
        for job in WF["jobs"].values():
            assert job["runs-on"] == "windows-latest"

    @staticmethod
    def _steps(job: str) -> list[dict]:
        return WF["jobs"][job]["steps"]

    def _at(self, job: str, needle: str) -> int:
        for i, step in enumerate(self._steps(job)):
            if needle in f"{step.get('uses', '')} {step.get('run', '')}":
                return i
        raise AssertionError(f"{job}: no step with {needle!r}")

    def test_sde_is_intels_kit_checked_by_its_hash(self):
        env = WF["jobs"]["sde"]["env"]
        assert re.fullmatch(r"[0-9A-F]{64}", env["SDE_SHA256"]), env["SDE_SHA256"]
        setup = self._at("sde", "petarpetrovt/setup-sde@")
        step = self._steps("sde")[setup]
        assert step["with"]["sdeVersion"] == "${{ env.SDE_VERSION }}"
        assert setup < self._at("sde", "$env:SDE_SHA256") < self._at("sde", "cpu_paths.py probe")

    def test_each_sde_leg_proves_its_chip_before_its_tests(self):
        assert self._at("sde", "cpu_paths.py probe") < self._at("sde", "cpu_paths.py sde-tests")
        legs = WF["jobs"]["sde"]["strategy"]["matrix"]["include"]
        expects = [dict(e.split("=") for e in leg["expect"].split()) for leg in legs]
        for leg, expect in zip(legs, expects, strict=True):
            assert {"avx2", "avx512f", "l3_kib"} <= set(expect), leg
            assert ("sde" in leg) != ("cpuid" in leg), leg
            if "cpuid" in leg:
                assert expect["l3_kib"] == str(leg["l3_kib"]), leg
        # A chip without AVX2, one with AVX2 alone, one with AVX-512, each a
        # chip of SDE's own, and a resized L3 under each of the latter two.
        kinds = {
            (e["avx2"], e["avx512f"]) for e, leg in zip(expects, legs, strict=True) if "sde" in leg
        }
        assert kinds == {("false", "false"), ("true", "false"), ("true", "true")}, kinds
        resized = {
            (e["avx2"], e["avx512f"])
            for e, leg in zip(expects, legs, strict=True)
            if "cpuid" in leg
        }
        assert resized == {("true", "false"), ("true", "true")}, resized

    def test_the_sde_tests_cover_every_module_running_products_or_the_oracle(self):
        src = REPO / "crates/online-core/src"
        needed = set()
        for path in src.rglob("*.rs"):
            text = path.read_text(encoding="utf-8")
            top = path.relative_to(src).parts[0].removesuffix(".rs")
            if top != "lib" and re.search(r"\bfaer\b|crate::oracle|\boracle::", text):
                needed.add(f"{top}::")
        assert {"gramfit::", "solve::", "pieces::", "ewcov::"} <= needed, needed
        listed = set(shlex.split(WF["jobs"]["sde"]["env"]["SDE_TESTS"]))
        assert needed <= listed, sorted(needed - listed)
        assert all(f.endswith("::") for f in listed), listed

    def test_the_sde_build_keeps_the_debug_assertions(self):
        env = WF["jobs"]["sde"]["env"]
        assert env["CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS"] == "true"
        assert env["CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS"] == "true"
        build = self._steps("sde")[self._at("sde", "cargo build --release")]["run"]
        assert "-p online-core --tests" in build and "cpu_paths.py lib-tests" in build

    def test_each_oracle_leg_proves_its_paths_before_its_tests(self):
        legs = WF["jobs"]["oracles"]["strategy"]["matrix"]["include"]
        cores = {leg["openblas"] for leg in legs}
        assert {"Prescott", "Haswell"} <= cores, cores
        # Native, so no kernel a machine of the pool may lack.
        assert not cores & {"SkylakeX", "Cooperlake", "SapphireRapids"}, cores
        assert {leg["keep"] for leg in legs} == {"", "X86_V3"}
        env = self._at("oracles", "cpu_paths.py numpy-env")
        assert env < self._at("oracles", "cpu_paths.py check") < self._at("oracles", "pytest")
        assert self._at("oracles", "maturin develop") < env
        (pytest_run,) = [s["run"] for s in self._steps("oracles") if "pytest" in s.get("run", "")]
        assert pytest_run == 'uv run pytest -m "oracle and not soak"'
