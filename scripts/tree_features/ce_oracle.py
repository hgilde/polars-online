"""category_encoders.CatBoostEncoder (ordered TS in row order) as the oracle
for an undecayed ordered TS with a fixed prior and no embargo."""

import sys
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

rng = np.random.default_rng(0)
n = 50_000
k = np.minimum(rng.zipf(1.3, n), 5000)
y = rng.standard_normal(n) + (k % 5)
for a in (1.0, 10.0):
    enc = (
        ce.CatBoostEncoder(cols=["k"], a=a)
        .fit_transform(pd.DataFrame({"k": k.astype(str)}), pd.Series(y))["k"]
        .to_numpy()
    )
    prior = y.mean()
    # ours with the oracle's fixed prior: prior_half_life tiny would not do; inline the constant prior  # noqa: E501
    t = np.arange(n, dtype=float)
    av = t + 0.5
    W = {}
    S = {}
    mine = np.empty(n)
    e, w = C.ordered_ts(t, av, k, y, prior_weight=a)
    for i in range(n):
        ww, ss = W.get(k[i], 0.0), S.get(k[i], 0.0)
        mine[i] = (ss + a * prior) / (ww + a)
        W[k[i]] = ww + 1
        S[k[i]] = ss + y[i]
    print(
        f"a={a}: max |CatBoostEncoder - ordered TS with its prior| = {np.max(np.abs(enc - mine)):.3e}; "  # noqa: E501
        f"with the ordered prior instead: {np.max(np.abs(enc - e)):.3e} (first 100 rows differ most: {np.max(np.abs(enc - e)[:100]):.3e}, last 10k: {np.max(np.abs(enc - e)[-10000:]):.3e})"  # noqa: E501
    )
