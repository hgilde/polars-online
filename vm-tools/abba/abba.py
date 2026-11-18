#!/usr/bin/env python3
"""One shared ABBA timing harness for builds of polars-online (see README-abba.md).

    python abba.py list
    python abba.py run --build A=PATH --build B=PATH --control A2=PATH \
        --cases bocpd_diag_k4_g100,corrchange_window_w100 --rounds 9 --out layout1.json
    python abba.py combine layout1.json layout2.json
    python abba.py once CASE            # one timed run on the current interpreter (sizing)

Python 3.12, standard library + polars + numpy only (the build under test brings its own).
A PATH is a venv directory, a python executable, or a wheel (a venv is made for it).
The first --build is the baseline A; the other builds are compared to it; --control is a
second copy of A's build in its own environment, so its A2/A spread is the noise floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Case catalogue. Each factory takes the scale and returns (timed_callable, rows). The data
# is generated here (seeded) before the clock starts; the callable builds a fresh bank (or
# runs the frame operation) and returns nothing. Row counts aim at 0.3-2 s on a 2 GHz Xeon.
# ---------------------------------------------------------------------------------------


def _frame(rows: int, k: int, groups: int, seed: int = 0, clock: str = "float", extra: int = 0):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    d: dict = {}
    if clock == "datetime":
        steps = rng.exponential(0.5e9, rows).astype("int64") + 1
        d["t"] = pl.Series("t", steps.cumsum()).cast(pl.Datetime("ns"))
    else:
        d["t"] = np.cumsum(rng.exponential(1.0, rows))
    for j in range(k + extra):
        d[f"x{j}"] = rng.standard_normal(rows)
    x = np.column_stack([d[f"x{j}"] for j in range(max(k, 1))])
    d["y"] = x @ rng.standard_normal(x.shape[1]) + 0.5 * rng.standard_normal(rows)
    d["z"] = rng.standard_normal(rows)
    if groups > 1:
        d["g"] = rng.integers(0, groups, rows)
    return pl.DataFrame(d)


def _feats(k: int) -> list[str]:
    return [f"x{j}" for j in range(k)]


def _bank_case(make_spec: Callable, rows: int, k: int, groups: int, *, chunk: int | None = None,
               clock: str = "float", extra: int = 0):
    def factory(scale: float):
        import polars_online as po

        n = max(1000, int(rows * scale))
        df = _frame(n, k, groups, clock=clock, extra=extra)
        spec = make_spec(po)
        if chunk is None:
            return (lambda: po.ModelBank([spec]).fit_predict(df)), n

        def run():
            for _ in po.ModelBank([spec]).fit_predict_batches(df, chunk_size=chunk):
                pass

        return run, n

    return factory


def _grp(g: int) -> dict:
    return {"group": "g"} if g > 1 else {}


def _window_case(n_ops: int):
    def factory(scale: float):
        import polars_online as po

        n = max(1000, int(2_000_000 * scale))
        df = _frame(n, 1, 100, clock="datetime")

        def run():
            ops = [po.ewm_mean("x0", half_life="5s", window_size="60s")]
            if n_ops == 4:
                ops += [po.ewm_sum("x0", half_life="5s", window_size="60s"),
                        po.ewm_rate("x0", half_life="5s", window_size="60s"),
                        po.ewm_std("x0", half_life="5s", window_size="60s")]
            named = {f"w{i}": e for i, e in enumerate(ops)}
            df.online.with_windows(clock="t", group="g", gap_cap="600s", **named)

        return run, n

    return factory


def _bocpd(k: int, g: int, rows: int, cov: str = "diag", chunk: int | None = None):
    return _bank_case(
        lambda po: po.spec.bocpd("m", features=_feats(k), covariance_type=cov, **_grp(g)),
        rows, k, g, chunk=chunk)


def _corr(kind: str, w: int, rows: int, k: int = 4):
    return _bank_case(
        lambda po: po.spec.corrchange("m", features=_feats(k), kind=kind, window_rows=w),
        rows, k, 1)


def _ew_cov(g: int, rows: int, lags: bool = False, k: int = 5):
    def spec(po):
        kw = dict(features=_feats(k), half_life=200.0, **_grp(g))
        if lags:
            return po.spec.ew_cov("m", stats=["corr", "lag_corr"], lags=[1, 2, 5], **kw)
        return po.spec.ew_cov("m", stats=["corr", "mahal"], precision_prior=0.1, **kw)

    return _bank_case(spec, rows, k, g)


def _kmeans(nc: int, rows: int, k: int = 4):
    return _bank_case(
        lambda po: po.spec.kmeans("m", features=_feats(k), half_life=2000.0, n_clusters=nc,
                                  warm_rows=500),
        rows, k, 1)


def _save_load(which: str):
    def factory(scale: float):
        import polars_online as po

        n, g, k = max(2000, int(400_000 * scale)), max(10, int(20_000 * scale)), 20
        df = _frame(n, k, g)
        spec = po.spec.ewridge("m", targets=["y"], features=_feats(k), clock="t", gap_cap=10.0,
                               half_life=1000.0, min_weight=25.0, group="g")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        path = os.path.join(tempfile.mkdtemp(prefix="abba_"), "bank.state")
        if which == "load":
            bank.save(path)
            return (lambda: po.ModelBank.load(path)), n
        return (lambda: bank.save(path)), n

    return factory


def _eight(scale: float):
    import polars_online as po

    n, k = max(2000, int(600_000 * scale)), 20
    df = _frame(n, k, 1)
    specs = [po.spec.ewridge(f"m{j}", targets=["y"], features=_feats(k), clock="t", gap_cap=10.0,
                             half_life=1000.0 * j) for j in range(1, 9)]
    po.ModelBank(specs).fit_predict(df)  # pool built, pages warm (untimed)
    return (lambda: po.ModelBank(specs).fit_predict(df)), n


def _seqtest(scale: float):
    import polars_online as po

    n = max(1000, int(4_000_000 * scale))
    df = _frame(n, 2, 1)
    spec = po.spec.seqtest("m", targets=["y", "z"])
    return (lambda: po.ModelBank([spec]).fit_predict(df)), n


def _hmm(scale: float):
    import polars_online as po

    n = max(1000, int(600_000 * scale))
    df = _frame(n, 4, 1)
    spec = po.spec.hmm("m", features=_feats(4), n_components=2, precision_prior=0.1,
                       half_life=500.0)
    return (lambda: po.ModelBank([spec]).fit_predict(df)), n


def _ewridge(scale: float):
    import polars_online as po

    n = max(2000, int(4_000_000 * scale))
    df = _frame(n, 5, 1000)
    spec = po.spec.ewridge("m", targets=["y"], features=_feats(5), clock="t", gap_cap=10.0,
                           half_life=1000.0, group="g")
    return (lambda: po.ModelBank([spec]).fit_predict(df)), n


CASES: dict[str, Callable] = {
    "window_ewm1_g100": _window_case(1),
    "window_ewm4_g100": _window_case(4),
}
# bocpd diag: rows shrink as k grows (cost ~ runs * k)
for _k, _rows in [(1, 50_000), (2, 150_000), (4, 300_000), (8, 400_000), (20, 300_000), (200, 40_000)]:
    for _g in (1, 100):
        CASES[f"bocpd_diag_k{_k}_g{_g}"] = _bocpd(_k, _g, _rows)
CASES["bocpd_full_k4_chunked"] = _bocpd(4, 1, 200_000, "full", chunk=10_000)
CASES["bocpd_full_k20_chunked"] = _bocpd(20, 1, 30_000, "full", chunk=2_000)
for _w, _rows in [(20, 150_000), (100, 200_000), (1000, 150_000), (2500, 100_000)]:
    CASES[f"corrchange_window_w{_w}"] = _corr("window", _w, _rows)
CASES["corrchange_monitor_w2500"] = _corr("monitor", 2500, 600_000)
CASES["hmm_d4_k2"] = _hmm
CASES["ew_cov_g1"] = _ew_cov(1, 600_000)
CASES["ew_cov_g100"] = _ew_cov(100, 3_000_000)
CASES["ew_cov_g1000"] = _ew_cov(1000, 3_000_000)
CASES["ew_cov_lags_g1"] = _ew_cov(1, 800_000, lags=True)
CASES["ew_cov_lags_g100"] = _ew_cov(100, 800_000, lags=True)
CASES["seqtest"] = _seqtest
CASES["ewridge_k5_g1000"] = _ewridge
CASES["kmeans_k4"] = _kmeans(4, 1_000_000)
CASES["kmeans_k20"] = _kmeans(20, 300_000, k=8)
CASES["save_400k_g20k"] = _save_load("save")
CASES["load_400k_g20k"] = _save_load("load")
CASES["parallel_eight_one_bank"] = _eight


# ---------------------------------------------------------------------------------------
# Worker: one case, one process, `reps` timed runs.
# ---------------------------------------------------------------------------------------


def worker(case: str, scale: float, reps: int, stat: str, warmup: int) -> None:
    fn, rows = CASES[case](scale)
    for _ in range(warmup):
        fn()
    secs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t)
    import polars_online as po

    out = {"secs": min(secs) if stat == "min" else statistics.median(secs), "all": secs,
           "rows": rows, "file": po.__file__, "version": getattr(po, "__version__", "?")}
    print("RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------


def load1() -> float:
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return float("nan")


def venv_python(venv: Path) -> Path:
    win = venv / "Scripts" / "python.exe"
    return win if win.exists() else venv / "bin" / "python"


def resolve_build(name: str, path: str, workdir: Path) -> str:
    p = Path(path).expanduser()
    if p.suffix == ".whl":
        venv = workdir / f"venv-{name}"
        if not venv_python(venv).exists():
            workdir.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
            py = str(venv_python(venv))
            subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
            subprocess.run([py, "-m", "pip", "install", "-q", str(p.resolve()), "numpy"], check=True)
        return str(venv_python(venv))
    if p.is_dir():
        return str(venv_python(p))
    return str(p)


def parse_named(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {it!r}")
        out[name] = path
    return out


def run_one(py: str, case: str, a: argparse.Namespace, env: dict) -> dict:
    cmd = [py, str(Path(__file__).resolve()), "_worker", case, "--scale", str(a.scale),
           "--reps", str(a.reps), "--stat", a.stat, "--warmup", str(a.warmup)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"worker failed ({py}, {case}):\n{p.stderr[-2000:]}")
    line = next(x for x in p.stdout.splitlines() if x.startswith("RESULT "))
    return json.loads(line[7:])


def wait_for_load(max_load: float | None, max_wait: float) -> float:
    ld = load1()
    if max_load is None or math.isnan(ld):
        return ld
    t0 = time.time()
    while ld > max_load and time.time() - t0 < max_wait:
        time.sleep(5)
        ld = load1()
    return ld


def verdict(ratios: list[float], ctl: list[float], floor: float = 0.005) -> str:
    med = statistics.median(ratios)
    lo, hi = min(min(ctl), 1.0) - floor, max(max(ctl), 1.0) + floor
    if med > hi and min(ratios) > 1.0:
        return "SLOWER"
    if med < lo and max(ratios) < 1.0:
        return "FASTER"
    return "within control"


def rng_s(v: list[float]) -> str:
    return f"{min(v):.3f}..{max(v):.3f}"


def table(res: dict) -> str:
    base, ctl = res["baseline"], res["control"]
    others = [b for b in res["builds"] if b not in (base, ctl)]
    lines = [f"baseline {base}; B/A > 1 means slower. load before rounds: "
             f"{res['load_min']:.2f}..{res['load_max']:.2f}",
             f"{'case':<28} {'A (s)':>8} {'build':<8} {'B/A median':>10} {'min..max':>13} "
             f"{'ctl A2/A':>8} {'ctl range':>13}  verdict"]
    for case, c in res["cases"].items():
        for b in others:
            r = c["ratios"][b]
            cr = c["ratios"].get(ctl, [1.0])
            lines.append(f"{case:<28} {c['median_secs'][base]:>8.3f} {b:<8} "
                         f"{statistics.median(r):>10.3f} {rng_s(r):>13} "
                         f"{statistics.median(cr):>8.3f} {rng_s(cr):>13}  {c['verdict'][b]}")
    return "\n".join(lines)


def cmd_run(a: argparse.Namespace) -> None:
    builds = parse_named(a.build)
    if len(builds) < 2:
        raise SystemExit("need at least two --build")
    control = parse_named(a.control)
    workdir = Path(a.workdir)
    pys = {n: resolve_build(n, p, workdir) for n, p in {**builds, **control}.items()}
    order = list(builds) + list(control)
    base = order[0]
    ctl = next(iter(control), None)
    seq = order + order[::-1]  # A B C C B A: the palindrome of every build, control last
    names = list(CASES) if a.cases == "all" else [c.strip() for c in a.cases.split(",") if c.strip()]
    for n in names:
        if n not in CASES:
            raise SystemExit(f"unknown case {n!r}; `abba.py list` shows them")
    env = {**os.environ, **dict(e.split("=", 1) for e in a.env)}
    res: dict = {"baseline": base, "control": ctl, "builds": order, "cases": {}, "loads": [],
                 "args": vars(a), "host": platform.platform(), "cpus": os.cpu_count(),
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    for case in names:
        per: dict[str, list[float]] = {b: [] for b in order}
        rows = None
        for r in range(a.rounds):
            ld = wait_for_load(a.max_load, a.max_wait)
            res["loads"].append(ld)
            got: dict[str, list[float]] = {b: [] for b in order}
            for b in seq:
                o = run_one(pys[b], case, a, env)
                got[b].append(o["secs"])
                rows = o["rows"]
                res["files"][b] = [o["file"], o["version"]]
            for b in order:  # mean of a build's two ABBA positions cancels linear drift
                per[b].append(statistics.mean(got[b]) if a.positions == "mean" else min(got[b]))
            print(f"{case} round {r + 1}/{a.rounds} load {ld:.2f} "
                  + " ".join(f"{b}={per[b][-1]:.3f}" for b in order), file=sys.stderr, flush=True)
        ratios = {b: [per[b][i] / per[base][i] for i in range(a.rounds)]
                  for b in order if b != base}
        cr = ratios.get(ctl, [1.0]) if ctl else [1.0]
        res["cases"][case] = {
            "rows": rows, "secs": per, "ratios": ratios,
            "median_secs": {b: statistics.median(per[b]) for b in order},
            "verdict": {b: verdict(ratios[b], cr, a.floor) for b in order if b not in (base, ctl)},
        }
    ls = [x for x in res["loads"] if not math.isnan(x)]
    res["load_min"], res["load_max"] = (min(ls), max(ls)) if ls else (float("nan"),) * 2
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(table(res))


def cmd_combine(a: argparse.Namespace) -> None:
    rs = [json.loads(Path(p).read_text()) for p in a.results]
    cases = [c for c in rs[0]["cases"] if all(c in r["cases"] for r in rs)]
    ctl = rs[0]["control"]
    others = [b for b in rs[0]["builds"] if b not in (rs[0]["baseline"], ctl)]
    out = {"layouts": a.results, "cases": {}}
    lines = [f"{'case':<28} {'build':<8} " + " ".join(f"{'L' + str(i + 1) + ' B/A':>16}"
                                                          for i in range(len(rs))) + "  verdict"]
    for c in cases:
        for b in others:
            vs = [r["cases"][c]["verdict"][b] for r in rs]
            if all(v == "SLOWER" for v in vs):
                v = "SLOWER (both layouts)"
            elif all(v == "FASTER" for v in vs):
                v = "FASTER (both layouts)"
            elif "SLOWER" in vs and "FASTER" in vs:
                v = "layout-dependent"
            elif "SLOWER" in vs:
                v = "within (slower in one layout only: layout)"
            elif "FASTER" in vs:
                v = "within (faster in one layout only)"
            else:
                v = "within control"
            out["cases"].setdefault(c, {})[b] = {"verdicts": vs, "verdict": v}
            cells = " ".join(f"{statistics.median(r['cases'][c]['ratios'][b]):>8.3f} {vv[:7]:>7}"
                             for r, vv in zip(rs, vs))
            lines.append(f"{c:<28} {b:<8} {cells}  {v}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    o = sub.add_parser("once")
    o.add_argument("case")
    o.add_argument("--scale", type=float, default=1.0)
    o.add_argument("--reps", type=int, default=3)
    w = sub.add_parser("_worker")
    w.add_argument("case")
    w.add_argument("--scale", type=float, default=1.0)
    w.add_argument("--reps", type=int, default=3)
    w.add_argument("--stat", default="min")
    w.add_argument("--warmup", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--build", action="append", required=True, metavar="NAME=PATH")
    r.add_argument("--control", action="append", default=[], metavar="NAME=PATH")
    r.add_argument("--cases", default="all", help="comma list of names, or 'all'")
    r.add_argument("--rounds", type=int, default=9)
    r.add_argument("--reps", type=int, default=3, help="timed runs inside each process")
    r.add_argument("--stat", choices=["min", "median"], default="min", help="of the reps")
    r.add_argument("--positions", choices=["mean", "min"], default="mean",
                   help="a build runs twice a round (ABBA); combine its two figures")
    r.add_argument("--warmup", type=int, default=0)
    r.add_argument("--floor", type=float, default=0.005,
                   help="widen the control band by this much each side (0.005 = 0.5%%)")
    r.add_argument("--scale", type=float, default=1.0, help="multiplies every case's rows")
    r.add_argument("--max-load", type=float, default=None, help="wait for 1-min load under this")
    r.add_argument("--max-wait", type=float, default=600.0, help="seconds to wait per round")
    r.add_argument("--env", action="append", default=[], metavar="K=V", help="for the workers")
    r.add_argument("--workdir", default=".abba-venvs", help="where wheel venvs are made")
    r.add_argument("--out", default=None, help="results JSON")
    c = sub.add_parser("combine")
    c.add_argument("results", nargs="+")
    c.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(CASES))
    elif a.cmd == "once":
        worker(a.case, a.scale, a.reps, "min", 0)
    elif a.cmd == "_worker":
        worker(a.case, a.scale, a.reps, a.stat, a.warmup)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        cmd_combine(a)


if __name__ == "__main__":
    main()
