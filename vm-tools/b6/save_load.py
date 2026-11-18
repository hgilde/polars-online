"""(d) bank save then load: bytes written, seconds each, sha256 of the bytes."""
import hashlib
import os
import tempfile
import time

import numpy as np
import polars as pl

import polars_online as po
from _abba import main

ROWS, K, G = 400_000, 20, 20_000


def make_cases():
    rng = np.random.default_rng(0)
    d = {f"x{j}": rng.standard_normal(ROWS) for j in range(K)}
    d["y"] = d["x0"] + 0.5 * rng.standard_normal(ROWS)
    d["t"] = np.arange(float(ROWS))
    d["g"] = np.arange(ROWS) % G
    df = pl.DataFrame(d)
    spec = po.spec.ewridge("m", targets=["y"], features=[f"x{j}" for j in range(K)], clock="t",
                           gap_cap=10.0, half_life=1000.0, min_weight=25.0, group="g")
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    path = os.path.join(tempfile.mkdtemp(), "bank.state")
    info = {}

    def save():
        t = time.perf_counter()
        bank.save(path)
        dt = time.perf_counter() - t
        raw = open(path, "rb").read()
        info["bytes"], info["sha256"] = len(raw), hashlib.sha256(raw).hexdigest()
        return dt

    def load():
        t = time.perf_counter()
        po.ModelBank.load(path)
        return time.perf_counter() - t

    def run_save():
        r = save()
        print(f"INFO save bytes={info['bytes']} sha256={info['sha256']}", flush=True)
        return r

    return {"save": (run_save, None), "load": (load, None)}


if __name__ == "__main__":
    main(__file__, make_cases)
