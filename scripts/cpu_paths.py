"""The steps of `.github/workflows/cpu-paths.yml` that need more than a line.

The workflow forces the CPU paths the tests take instead of sampling the
runner pool, and shows each one took effect before it believes a green run:

    python scripts/cpu_paths.py lib-tests < cargo.json
    python scripts/cpu_paths.py sde-cpuid KIT CHIP --l3-kib 2048 --out l3.def
    python scripts/cpu_paths.py probe PROBE --sde SDE --sde-args=-hsw --expect avx2=true
    python scripts/cpu_paths.py sde-tests EXE --sde SDE --sde-args=-hsw --filters "solve::"
    python scripts/cpu_paths.py numpy-env --openblas Haswell --keep X86_V3 >> "$GITHUB_ENV"
    python scripts/cpu_paths.py check

`lib-tests` reads cargo's `--message-format=json` and prints the path of
the library's unit-test executable. `sde-cpuid` writes a copy of one of
Intel SDE's CPUID definition files with its L3 cache resized, for SDE's
`-cpuid_in`, and prints its absolute path. `probe` runs the program
`scripts/cpu_probe.rs` builds on the host and under SDE, and fails unless
SDE's run reports what `--expect` says.
`sde-tests` lists the tests the filters select on the host, runs them under
SDE, and fails unless every listed test passed: a filter that matches
nothing would otherwise pass with nothing run. `numpy-env` prints the two
environment variables that pin OpenBLAS's kernels and NumPy's loops, and
`check`, run in that environment, fails unless both took effect.

`--sde-args` takes its value after an `=`, since SDE's knobs start with a
dash, which argparse would read as an option of its own.

Standard library only, but for `check`, which imports NumPy, SciPy and
threadpoolctl from the project's environment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

#: SDE's knob that keeps the definition file's CPUID leaves as written. By
#: default SDE updates some leaves from the host, so a cache size could be
#: the runner's rather than the chip's the workflow asked for.
SDE_FIXED = ["-update_cpuid_from_host", "0"]


def lib_test_executable(messages: Iterable[str]) -> str:
    """The library's unit-test executable among cargo's JSON messages: the
    one artifact of a `lib` target built as a test. Exactly one, or an
    error, so a build that made none (or a workspace build that made
    several) cannot hand the next step the wrong binary."""
    found = []
    for line in messages:
        line = line.strip()
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("reason") != "compiler-artifact" or not msg.get("executable"):
            continue
        kinds = msg["target"]["kind"]
        if "lib" in kinds or "rlib" in kinds:
            found.append(msg["executable"])
    if len(found) != 1:
        raise SystemExit(f"expected one library test executable, found {found}")
    return found[0]


#: A CPUID definition file's line: the leaf and subleaf (`********` for
#: any), then EAX, EBX, ECX and EDX, all hexadecimal.
DEF_LINE = re.compile(
    r"^(?P<leaf>[0-9a-fA-F]{8}) (?P<sub>[0-9a-fA-F*]{8}) => "
    r"(?P<eax>[0-9a-fA-F]{8}) (?P<ebx>[0-9a-fA-F]{8}) "
    r"(?P<ecx>[0-9a-fA-F]{8}) (?P<edx>[0-9a-fA-F]{8})(?P<rest>.*)$"
)


def resize_l3(text: str, l3_kib: int) -> str:
    """A CPUID definition file with its L3 cache `l3_kib` KiB.

    Leaf 4 describes one cache a subleaf: EBX holds the ways, partitions and
    line size, each less one (bits 31-22, 21-12 and 11-0), and ECX the sets
    less one; the size is their product. The line size and partitions stay;
    the ways stay if they divide the new size into whole sets, and become
    the first of 16, 8, 4, 2 and 1 that does otherwise.
    """
    out, done = [], 0
    for line in text.splitlines():
        m = DEF_LINE.match(line)
        if m and m["leaf"] == "00000004" and m["sub"] != "********":
            eax, ebx = int(m["eax"], 16), int(m["ebx"], 16)
            if eax & 0x1F in (1, 3) and (eax >> 5) & 0x7 == 3:
                ways = (ebx >> 22) + 1
                partitions = ((ebx >> 12) & 0x3FF) + 1
                line_bytes = (ebx & 0xFFF) + 1
                size = l3_kib * 1024
                for w in (ways, 16, 8, 4, 2, 1):
                    if size % (w * partitions * line_bytes) == 0:
                        ways = w
                        break
                else:
                    raise SystemExit(f"no way count divides {l3_kib} KiB into whole sets")
                sets = size // (ways * partitions * line_bytes)
                ebx = ((ways - 1) << 22) | ((partitions - 1) << 12) | (line_bytes - 1)
                line = (
                    f"{m['leaf']} {m['sub']} => {m['eax']} {ebx:08x} "
                    f"{sets - 1:08x} {m['edx']}{m['rest']}"
                )
                done += 1
        out.append(line)
    if done != 1:
        raise SystemExit(f"expected one L3 subleaf in leaf 4, found {done}")
    return "\n".join(out) + "\n"


def parse_probe(output: str) -> dict[str, str]:
    """`cpu_probe`'s `key=value` lines as a mapping."""
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def unmet(seen: dict[str, str], expect: Sequence[str]) -> list[str]:
    """Each `key=value` of `expect` the probe did not report as given."""
    wrong = []
    for item in expect:
        key, value = item.split("=", 1)
        if seen.get(key) != value:
            wrong.append(f"{key}: expected {value}, saw {seen.get(key)}")
    return wrong


def listed(output: str) -> int:
    """How many tests a libtest `--list` printed."""
    return sum(1 for line in output.splitlines() if line.endswith(": test"))


#: libtest's summary line.
RESULT = re.compile(r"^test result: (\w+)\. (\d+) passed; (\d+) failed;", re.M)


def passed_all(output: str, expected: int) -> str | None:
    """None if the run's last summary says all `expected` tests passed,
    else what it said instead."""
    results = RESULT.findall(output)
    if not results:
        return "no test result line"
    verdict, passed, failed = results[-1]
    if verdict != "ok" or int(failed) or int(passed) != expected:
        return f"{verdict}: {passed} passed, {failed} failed, of {expected} listed"
    return None


def numpy_disable(dispatch: Sequence[str], keep: Sequence[str]) -> list[str]:
    """The NumPy features to disable so that its dispatched loops go no
    higher than `keep`: every dispatched feature but those. NumPy refuses
    to disable a baseline feature, so the baseline is never named."""
    unknown = [k for k in keep if k not in dispatch]
    if unknown:
        raise SystemExit(f"not among NumPy's dispatched features {list(dispatch)}: {unknown}")
    return [f for f in dispatch if f not in keep]


#: The name `openblas_get_corename` reports for a forced core, where it is
#: not the core's own. NumPy's and SciPy's OpenBLAS on Windows are the
#: scipy-openblas builds (MacPython/openblas-libs,
#: `tools/build_steps_windows.sh`), made with `DYNAMIC_LIST="PRESCOTT
#: NEHALEM SANDYBRIDGE HASWELL SKYLAKEX"`; in such a build OpenBLAS's
#: `driver/others/dynamic.c` defines each core left out as an alias of one
#: kept, `gotoblas_KATMAI` among them as `gotoblas_PRESCOTT`, and
#: `gotoblas_corename` tests Katmai before Prescott, so Prescott's kernels
#: report "Katmai" (the first run of cpu-paths.yml, both libraries). A
#: runner's own detection picks Haswell or SkylakeX (Zen is an alias of
#: Haswell in these builds), never these kernels, so the name still tells a
#: forced Prescott from the core a runner picks for itself.
REPORTED_AS = {"prescott": "katmai"}


def reported_core(core: str) -> str:
    """The name, lower-cased, OpenBLAS reports when `core` is forced."""
    return REPORTED_AS.get(core.lower(), core.lower())


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", **kw)


def _sde(sde: str, sde_args: str) -> list[str]:
    return [sde, *shlex.split(sde_args), *SDE_FIXED, "--"]


def main(argv: Sequence[str] | None = None) -> int:
    # Plain newlines on Windows too: a path a step reads with `$(...)` or a
    # line it appends to `$GITHUB_ENV` would otherwise end in a carriage
    # return, which the shell keeps.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(newline="\n")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lib-tests")
    p = sub.add_parser("sde-cpuid")
    p.add_argument("kit", type=Path)
    p.add_argument("chip")
    p.add_argument("--l3-kib", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("probe")
    p.add_argument("probe")
    p.add_argument("--sde", required=True)
    p.add_argument("--sde-args", required=True)
    p.add_argument("--expect", nargs="+", required=True)
    p = sub.add_parser("sde-tests")
    p.add_argument("exe")
    p.add_argument("--sde", required=True)
    p.add_argument("--sde-args", required=True)
    p.add_argument("--filters", required=True)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("numpy-env")
    p.add_argument("--openblas", required=True)
    p.add_argument("--keep", nargs="*", default=[])
    sub.add_parser("check")
    args = parser.parse_args(argv)

    if args.cmd == "lib-tests":
        print(lib_test_executable(sys.stdin))
    elif args.cmd == "sde-cpuid":
        source = args.kit / "misc" / "cpuid" / args.chip / "cpuid.def"
        args.out.write_text(resize_l3(source.read_text(encoding="utf-8"), args.l3_kib), "utf-8")
        print(f"{args.out}: {args.chip}'s CPUID with a {args.l3_kib} KiB L3", file=sys.stderr)
        # Absolute, with forward slashes, which Windows takes and `shlex`
        # leaves alone: the tests run from another directory.
        print(args.out.resolve().as_posix())
    elif args.cmd == "probe":
        probe = str(Path(args.probe).resolve())
        host = _run([probe], check=True).stdout
        print("on the host:\n" + host)
        res = _run([*_sde(args.sde, args.sde_args), probe])
        print(f"under SDE ({args.sde_args}):\n{res.stdout}{res.stderr}")
        wrong = unmet(parse_probe(res.stdout), args.expect)
        if res.returncode or wrong:
            print(f"SDE did not pose as asked (exit {res.returncode}): {wrong}", file=sys.stderr)
            return 1
    elif args.cmd == "sde-tests":
        filters = shlex.split(args.filters)
        lst = _run([args.exe, "--list", "--include-ignored", *filters], check=True).stdout
        expected = listed(lst)
        print(f"{expected} tests listed on the host")
        cmd = [
            *_sde(args.sde, args.sde_args),
            args.exe,
            "--include-ignored",
            f"--test-threads={args.threads}",
            *filters,
        ]
        print("+", shlex.join(cmd), flush=True)
        # Streamed, so a slow emulated run shows its progress in the log.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
        assert proc.stdout is not None
        lines = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        code = proc.wait()
        problem = passed_all("".join(lines), expected)
        if code or problem or expected == 0:
            print(f"exit {code}: {problem or 'no test selected'}", file=sys.stderr)
            return 1
    elif args.cmd == "numpy-env":
        from numpy._core import _multiarray_umath as mu

        disable = numpy_disable(mu.__cpu_dispatch__, args.keep)
        print(f"OPENBLAS_CORETYPE={args.openblas}")
        print(f"NPY_DISABLE_CPU_FEATURES={' '.join(disable)}")
    elif args.cmd == "check":
        return check()
    return 0


def check() -> int:
    """Fail unless OpenBLAS runs the kernels `OPENBLAS_CORETYPE` names, in
    NumPy's copy and SciPy's alike, under the name it reports for them
    (`reported_core`), and NumPy's loops leave out every
    feature `NPY_DISABLE_CPU_FEATURES` names. NumPy refuses a baseline
    feature with an error and only warns of a name it does not dispatch,
    so that warning is an error here."""
    warnings.simplefilter("error", ImportWarning)
    import numpy as np
    import scipy.linalg  # noqa: F401  (loads SciPy's own OpenBLAS)
    import threadpoolctl
    from numpy._core import _multiarray_umath as mu

    np.show_runtime()
    print("baseline:", mu.__cpu_baseline__)
    print("dispatch:", mu.__cpu_dispatch__)
    problems = []
    core = os.environ.get("OPENBLAS_CORETYPE", "")
    want = reported_core(core)
    blas = [i for i in threadpoolctl.threadpool_info() if i.get("internal_api") == "openblas"]
    for info in blas:
        print(f"{info['filepath']}: {info.get('architecture')}")
        if str(info.get("architecture", "")).lower() != want:
            got = info.get("architecture")
            problems.append(f"{info['filepath']} runs {got}, not {core} (reported as {want})")
    if core and len(blas) < 2:
        problems.append(f"expected NumPy's and SciPy's OpenBLAS, found {len(blas)}")
    for name in os.environ.get("NPY_DISABLE_CPU_FEATURES", "").split():
        if mu.__cpu_features__.get(name) is not False:
            problems.append(f"NumPy's {name} is {mu.__cpu_features__.get(name)}, not disabled")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
