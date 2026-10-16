"""Peak memory of each way of running a bank over a parquet file, at 3M and
12M rows: the figures the README's *Memory* table, its advice to filter after
the bank, and ``docs/RUNNER.md``'s *Memory* cite.

    uv run python scripts/memory_bench.py generate .cache/memory
    uv run python scripts/memory_bench.py all .cache/memory
    uv run python scripts/memory_bench.py all .cache/memory --online target/release/polars-online

``generate`` writes a seeded file of 3M rows, 20 features, a clock, a weight
and 32 groups (0.5 GB of parquet), and the same rows four times over, the
clock shifted, for 12M (2.0 GB): the state is the same at both sizes and only
the data grows. Each case then runs in a fresh process under macOS's
``/usr/bin/time -l``, and its figure is the peak memory footprint, the most
physical memory the process held at once. The spec is ``ewridge`` with a
clock, the weight, a minimum weight and the 32 groups, parquet in and
parquet out. The cases:

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
    ``POLARS_ROW_GROUP_PREFETCH_SIZE=1``

macOS only: the footprint comes from ``/usr/bin/time -l``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

N, K, G = 3_000_000, 20, 32
FEATURES = [f"x{j}" for j in range(K)]
PY_CASES = ("query", "after", "before", "loop", "released")


def generate(d: Path) -> None:
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
    parts = [pl.scan_parquet(d / "r3m.parquet").with_columns(pl.col("t") + i * N) for i in range(4)]
    pl.concat(parts).sink_parquet(d / "r12m.parquet")


def _spec():
    import polars_online as po

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

    import polars_online as po

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
    cfg.write_text(
        f'input = "{d / f"r{size}.parquet"}"\noutput = "{d / f"out_online_{size}.parquet"}"\n\n'
        '[[specs]]\nname = "m"\ntargets = ["y"]\n'
        f"features = {FEATURES!r}\n".replace("'", '"')
        + 'clock = "t"\ngap_cap = 10.0\nhalf_life = 1000.0\nweight = "w"\ngroup = "group"\n'
        'min_weight = 20.0\n[specs.model]\ntype = "ewridge"\n',
        encoding="utf-8",
    )
    return cfg


def _footprint(cmd: list[str], env: dict[str, str]) -> float:
    res = subprocess.run(
        ["/usr/bin/time", "-l", *cmd],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    line = next(ln for ln in res.stderr.splitlines() if "peak memory footprint" in ln)
    return int(line.split()[0]) / 1e9


def all_cases(d: Path, online: str | None, repeats: int = 2) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)
    cases = [(c, s) for s in ("3m", "12m") for c in PY_CASES]
    if online:
        cases += [(c, s) for s in ("3m", "12m") for c in ("online", "online-prefetch1")]
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
    elif cmd == "all":
        online = argv[argv.index("--online") + 1] if "--online" in argv else None
        all_cases(d, online)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
