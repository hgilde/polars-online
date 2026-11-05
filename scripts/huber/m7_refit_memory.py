# The `refit_budget` memory table of MODELS.md's `huber` section and HUBER.md
# §9 (tasks 277 and 331): groups of 150 rows, two targets and 20 features,
# none reaching the default `huber_refit_rows` of 500, so every buffer a group
# reserves holds its rows to the end. The frame is written to parquet once and
# fed back in its batches through a `ModelBank`; each configuration runs in a
# process of its own, which prints its peak resident memory after the fit
# (`ru_maxrss`, the whole process), the saved state's size, and the targets
# whose refits the budget gave up. The table gives the median of three runs.
#
#   uv run python scripts/huber/m7_refit_memory.py --groups 5000 --budget default
#   uv run python scripts/huber/m7_refit_memory.py --groups 20000 --budget off
#
# `--budget` is `default` (no `refit_budget` given), `off`
# (`huber_refit_rows=0`), or a number of MiB, `inf` for no bound. The parquet
# file is cached under `.cache/huber/`, written by a process of its own.
import argparse
import resource
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po

K = 20
ROWS = 150
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "huber"


def path_of(groups: int) -> Path:
    return CACHE / f"refit_memory_{groups}.parquet"


def write(groups: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(groups)
    n = groups * ROWS
    x = rng.standard_normal((n, K))
    beta = rng.standard_normal(K)
    y = 1000.0 + x @ beta + 0.5 * rng.standard_normal(n)
    y[(rng.random(n) < 0.25) & (x[:, 0] > 0)] += 50.0
    z = -(x @ beta) + 0.5 * rng.standard_normal(n)
    cols = {f"x{i}": x[:, i] for i in range(K)}
    df = pl.DataFrame({"g": np.repeat(np.arange(groups), ROWS), **cols, "y": y, "z": z})
    df.write_parquet(path_of(groups), row_group_size=100_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5000)
    ap.add_argument("--budget", default="default")
    ap.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.write:
        write(a.groups)
        return
    path = path_of(a.groups)
    if not path.exists():
        # Written by a process of its own, so the frame it builds is not in
        # this one's peak.
        subprocess.run([sys.executable, __file__, "--groups", str(a.groups), "--write"], check=True)
    kw: dict[str, float | int] = {}
    if a.budget == "off":
        kw["huber_refit_rows"] = 0
    elif a.budget != "default":
        kw["refit_budget"] = float(a.budget)
    spec = po.spec.huber(
        "m",
        targets=["y", "z"],
        features=[f"x{i}" for i in range(K)],
        group="g",
        half_life=float("inf"),
        **kw,  # type: ignore[arg-type]
    )
    bank = po.ModelBank([spec])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for batch in pl.scan_parquet(path).collect_batches():
            bank.fit_predict(batch)
    # `ru_maxrss` is in bytes on macOS and in KiB on Linux.
    unit = 1 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**20
    state = len(bank.save_bytes()) / 2**20
    given_up = int(bank.summary()["refit_given_up"].sum())
    print(
        f"{a.groups} groups, budget {a.budget}: peak {peak:.0f} MiB, state {state:.0f} MiB, "
        f"{given_up} of {2 * a.groups} targets given up"
    )


if __name__ == "__main__":
    main()
