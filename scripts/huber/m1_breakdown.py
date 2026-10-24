# Converged batch Huber IRLS (statsmodels RLM, HuberT 1.345) under three scale estimators,
# with outliers correlated with x (x > 0 lifted by +50), which biases the slope.
import warnings

import numpy as np
import statsmodels.api as sm

warnings.simplefilter("ignore")


def stream(n, frac, level, seed, lift=50.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
    bad = (rng.random(n) < frac) & (x > 0)
    y[bad] += lift
    return sm.add_constant(x), y, bad.mean()


class RmsScale:  # the library's present scale: the plain RMS of every residual
    def __call__(self, resid):
        return np.sqrt(np.mean(resid**2))


scales = {
    "MAD": sm.robust.scale.mad,
    "Proposal 2": sm.robust.scale.HuberScale(),
    "plain RMS": RmsScale(),
}


def summary(v):
    return f"{np.median(v):.3f} [{min(v):.3f},{max(v):.3f}]"


print(
    "converged IRLS slope, truth 2.0, 20,000 rows, level 1,000; "
    "median [min,max] over 10 seeds; outliers on x>0 rows only"
)
print(f"{'contam.':>8} {'OLS':>22} " + " ".join(f"{k:>22}" for k in scales))
for frac in (0.02, 0.10, 0.25, 0.40, 0.60):
    cols = {k: [] for k in ["OLS", *scales]}
    shares = []
    for seed in range(10):
        X, y, share = stream(20_000, frac, 1000.0, seed)
        shares.append(share)
        cols["OLS"].append(sm.OLS(y, X).fit().params[1])
        for k, sc in scales.items():
            cols[k].append(
                sm.RLM(y, X, M=sm.robust.norms.HuberT(1.345))
                .fit(scale_est=sc, tol=1e-10, maxiter=300)
                .params[1]
            )
    print(f"{np.mean(shares):>7.1%} " + " ".join(f"{summary(cols[k]):>22}" for k in cols))
