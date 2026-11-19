"""Peak memory of each way of running a bank over a parquet file, at 3M and
12M rows, and the command line at 24M too: the figures ``docs/PERFORMANCE.md``'s
*Peak memory of each way of running a bank* and *Peak memory of the command
line* give, and the advice to filter after the bank.

    uv run python scripts/memory_bench.py generate .cache/memory
    uv run python scripts/memory_bench.py all .cache/memory
    uv run python scripts/memory_bench.py all .cache/memory --online target/release/rheon

``generate`` writes a seeded file of 3M rows, 20 features, a clock, a weight
and 32 groups (0.5 GB of parquet), and the same rows four times over, the
clock shifted, for 12M (2.0 GB), and eight times over for 24M (4.1 GB): the
state is the same at every size and only the data grows. Each case then runs
in a fresh process under macOS's ``/usr/bin/time -l``, and its figure is the
peak memory footprint, the most physical memory the process held at once. The
spec is ``ewridge`` with a clock, the weight, a minimum weight and the 32
groups, parquet in and parquet out. The cases:

``query``
    ``lf.online.fit_predict([spec]).sink_parquet(..)``
``after`` and ``before``
    the same with a filter that keeps every row, after the bank and before
    it: a filter before it is held by Polars' reader, which is the worst
    case when it keeps every row
``loop``
    ``for chunk in lf.collect_batches(): bank.fit_predict(chunk)``
``released``
    ``query`` with the allocator told to return freed pages at once
    (``_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0``)
``online`` and ``online-prefetch1``
    the command line, given with ``--online``, and the same with
    ``POLARS_ROW_GROUP_PREFETCH_SIZE=1``, at 24M rows as well: a peak that
    grows from 12M to 24M is the allocator holding pages, not the bank

The figure is the peak memory footprint on macOS (``/usr/bin/time -l``). The
command line's cases also run on Linux and Windows (``cli``, below), where the
platform has no footprint and the figure is the closest high-water mark it
gives: Linux's peak resident set (``wait4``'s ``ru_maxrss``) beside the peak
of ``RssAnon`` (resident pages that are not a mapped file) sampled from
``/proc``, and Windows' peak working set (``GetProcessMemoryInfo``) beside its
peak commit (``PeakPagefileUsage``, private memory). A resident set counts a
mapped parquet file's clean pages, so compare a platform's figures with one
another and not with macOS's. The python cases (``all``) stay macOS-only.

    uv run python scripts/memory_bench.py cli .cache/memory --online target/release/rheon \\
        --sizes 3m,12m,24m --json cli-memory.json [--variant K=V,K=V ...]

``cli`` runs only the command line, on any platform, one size at a time
(each size's file is generated, run and deleted, so a runner's disk holds one),
and writes every run, with its wall and CPU time, to the JSON file: the job
``.github/workflows/cli-memory.yml`` runs on Windows and Linux. Each
``--variant`` adds a case, ``online[K=V]``, that runs with those environment
variables set (a stock binary reads ``MIMALLOC_PURGE_DELAY`` and
``_RJEM_MALLOC_CONF`` itself), beside the plain ``online``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

N, K, G = 3_000_000, 20, 32
FEATURES = [f"x{j}" for j in range(K)]
PY_CASES = ("query", "after", "before", "loop", "released")


def _parts(size: str) -> int:
    """How many copies of the 3M-row file make ``size`` (``"12m"`` -> 4)."""
    rows = int(size.removesuffix("m"))
    if not size.endswith("m") or rows % 3 or rows <= 0:
        raise ValueError(f"{size!r} is not a size: a multiple of 3 and an m, as 3m or 12m")
    return rows // 3


def generate(d: Path, sizes: tuple[str, ...] = ("12m", "24m")) -> None:
    """The 3M-row file and the files of ``sizes`` (the base file is always
    written: the others are it, repeated, the clock shifted)."""
    import numpy as np
    import polars as pl

    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((N, K))
    beta = rng.standard_normal(K)
    df = pl.DataFrame(
        {
            "t": np.arange(N, dtype=float),
            "group": rng.integers(0, G, N),
            "w": rng.uniform(0.5, 1.5, N),
            "y": x @ beta + rng.standard_normal(N),
        }
        | {f: x[:, j] for j, f in enumerate(FEATURES)}
    )
    df.write_parquet(d / "r3m.parquet")
    for size in sizes:
        if size != "3m":
            repeat(d, size)


def repeat(d: Path, size: str) -> None:
    """``r<size>.parquet``: the 3M-row file ``size / 3M`` times, clock shifted."""
    import polars as pl

    parts = [
        pl.scan_parquet(d / "r3m.parquet").with_columns(pl.col("t") + i * N)
        for i in range(_parts(size))
    ]
    pl.concat(parts).sink_parquet(d / f"r{size}.parquet")


def _spec():
    import rheon as po

    return po.spec.ewridge(
        "m",
        targets=["y"],
        features=FEATURES,
        clock="t",
        gap_cap=10.0,
        half_life=1000.0,
        weight="w",
        group="group",
        min_weight=20.0,
    )


def run(d: Path, case: str, size: str) -> None:
    """One case, in this process."""
    import polars as pl

    import rheon as po

    spec = _spec()
    keep = pl.col("w") > 0  # every row
    lf = pl.scan_parquet(d / f"r{size}.parquet")
    out = d / f"out_{case}_{size}.parquet"
    if case == "loop":
        bank = po.ModelBank([spec])
        for chunk in lf.collect_batches():
            bank.fit_predict(chunk)
        return
    q = {
        "query": lambda: lf.online.fit_predict([spec]),
        "released": lambda: lf.online.fit_predict([spec]),
        "after": lambda: lf.online.fit_predict([spec]).filter(keep),
        "before": lambda: lf.filter(keep).online.fit_predict([spec]),
    }[case]()
    q.sink_parquet(out)


def _config(d: Path, size: str) -> Path:
    cfg = d / f"bank_{size}.toml"
    inp = (d / f"r{size}.parquet").as_posix()  # forward slashes: TOML escapes a backslash
    out = (d / f"out_online_{size}.parquet").as_posix()
    cfg.write_text(
        f'input = "{inp}"\noutput = "{out}"\n\n'
        '[[specs]]\nname = "m"\ntargets = ["y"]\n'
        f"features = {FEATURES!r}\n".replace("'", '"')
        + 'clock = "t"\ngap_cap = 10.0\nhalf_life = 1000.0\nweight = "w"\ngroup = "group"\n'
        'min_weight = 20.0\n[specs.model]\ntype = "ewridge"\n',
        encoding="utf-8",
    )
    return cfg


def _footprint(cmd: list[str], env: dict[str, str]) -> float:
    return float(measure(cmd, env)["peak_gb"])


def _proc_status_kb(pid: int, field: str) -> int:
    """A ``kB`` field of ``/proc/<pid>/status``, 0 where it cannot be read."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _windows_counters(handle: int) -> dict[str, float]:
    """Peak working set, peak commit and CPU seconds of a finished (or live)
    process, from its handle: ``GetProcessMemoryInfo`` and ``GetProcessTimes``."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = wintypes.HANDLE(handle)
    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    if not kernel32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    ft = [wintypes.FILETIME() for _ in range(4)]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
    if not kernel32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
        raise ctypes.WinError(ctypes.get_last_error())

    def secs(t: Any) -> float:
        return ((t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7

    return {
        "peak_working_set_gb": c.PeakWorkingSetSize / 1e9,
        "peak_commit_gb": c.PeakPagefileUsage / 1e9,
        "cpu_s": secs(ft[2]) + secs(ft[3]),
    }


def measure(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    """Run ``cmd`` once and return its peak memory, wall and CPU time.

    ``peak_gb`` is the figure the platform's metric names: macOS's peak memory
    footprint, Linux's peak resident set, Windows' peak working set. Linux adds
    ``peak_anon_gb`` (the largest ``RssAnon`` seen at 20 ms) and Windows
    ``peak_commit_gb``, the private memory the process asked for.
    """
    if sys.platform == "darwin":
        t0 = time.perf_counter()
        res = subprocess.run(
            ["/usr/bin/time", "-l", *cmd],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        wall = time.perf_counter() - t0
        gb = {}
        for ln in res.stderr.splitlines():
            if "peak memory footprint" in ln:
                gb["peak_gb"] = int(ln.split()[0]) / 1e9
        user = sys_ = 0.0
        for ln in res.stderr.splitlines():
            words = ln.split()
            if "real" in words and "user" in words and "sys" in words:
                user = float(words[words.index("user") - 1])
                sys_ = float(words[words.index("sys") - 1])
        return {"metric": "footprint", **gb, "wall_s": wall, "cpu_s": user + sys_}
    if sys.platform == "win32":
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        wall = time.perf_counter() - t0
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=err)
        c = _windows_counters(int(proc._handle))  # type: ignore[attr-defined]
        return {
            "metric": "peak working set",
            "peak_gb": c["peak_working_set_gb"],
            "peak_commit_gb": c["peak_commit_gb"],
            "wall_s": wall,
            "cpu_s": c["cpu_s"],
        }
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    anon_kb = 0
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            break
        anon_kb = max(anon_kb, _proc_status_kb(proc.pid, "RssAnon"))
        time.sleep(0.02)
    wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)  # reaped here, not by Popen
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return {
        "metric": "peak resident set",
        "peak_gb": usage.ru_maxrss * 1024 / 1e9,  # Linux reports kB
        "peak_anon_gb": anon_kb * 1024 / 1e9,
        "wall_s": wall,
        "cpu_s": usage.ru_utime + usage.ru_stime,
    }


def cli_cases(
    d: Path, online: str, sizes: list[str], repeats: int, variants: list[str], out: Path | None
) -> dict[str, Any]:
    """The command line's cases at each of ``sizes``: every run's figures, and
    the machine, as a dict (written to ``out`` as JSON when given)."""
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "r3m.parquet").exists():
        generate(d, ())
    runs: list[dict[str, Any]] = []
    cases: list[tuple[str, dict[str, str]]] = [
        ("online", {}),
        ("online-prefetch1", {"POLARS_ROW_GROUP_PREFETCH_SIZE": "1"}),
    ]
    for v in variants:
        cases.append((f"online[{v}]", dict(kv.split("=", 1) for kv in v.split(","))))
    print(f"{platform.platform()}, {os.cpu_count()} cpus", flush=True)
    print("| case | rows | peak GB (each run) | wall s | cpu s |\n|---|---|---|---|---|")
    for size in sizes:
        if not (d / f"r{size}.parquet").exists():
            repeat(d, size)
        cmd = [online, "--config", str(_config(d, size)), "--quiet"]
        for case, extra in cases:
            got = []
            for i in range(repeats):
                m = measure(cmd, {**os.environ, **extra})
                (d / f"out_online_{size}.parquet").unlink(missing_ok=True)
                runs.append({"case": case, "rows": size, "run": i, **m})
                got.append(m)
            print(
                f"| {case} | {size} | {' / '.join(f'{g["peak_gb"]:.2f}' for g in got)} "
                f"| {' / '.join(f'{g["wall_s"]:.1f}' for g in got)} "
                f"| {' / '.join(f'{g["cpu_s"]:.1f}' for g in got)} |",
                flush=True,
            )
        if size != "3m":
            (d / f"r{size}.parquet").unlink()  # one size on the disk at a time
    result = {
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "online": online,
        "runs": runs,
    }
    if out is not None:
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def all_cases(d: Path, online: str | None, repeats: int = 2) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)
    cases = [(c, s) for s in ("3m", "12m") for c in PY_CASES]
    if online:
        cases += [(c, s) for s in ("3m", "12m", "24m") for c in ("online", "online-prefetch1")]
    print("| case | rows | peak footprint, GB (each run) |\n|---|---|---|")
    for case, size in cases:
        env = dict(os.environ)
        if case == "released":
            env["_RJEM_MALLOC_CONF"] = "dirty_decay_ms:0,muzzy_decay_ms:0"
        if case == "online-prefetch1":
            env["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"
        if case.startswith("online"):
            cmd = [online or "", "--config", str(_config(d, size)), "--quiet"]
        else:
            cmd = [sys.executable, __file__, "run", str(d), case, size]
        got = [_footprint(cmd, env) for _ in range(repeats)]
        print(f"| {case} | {size} | {' / '.join(f'{g:.2f}' for g in got)} |", flush=True)


def main(argv: list[str]) -> None:
    cmd, d = argv[1], Path(argv[2])
    if cmd == "generate":
        generate(d)
    elif cmd == "run":
        run(d, argv[3], argv[4])
    elif cmd == "cli":
        sizes = (
            argv[argv.index("--sizes") + 1].split(",")
            if "--sizes" in argv
            else ["3m", "12m", "24m"]
        )
        reps = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 2
        out = Path(argv[argv.index("--json") + 1]) if "--json" in argv else None
        variants = [argv[i + 1] for i, a in enumerate(argv) if a == "--variant"]
        cli_cases(d, argv[argv.index("--online") + 1], sizes, reps, variants, out)
    elif cmd == "all":
        online = argv[argv.index("--online") + 1] if "--online" in argv else None
        all_cases(d, online)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
