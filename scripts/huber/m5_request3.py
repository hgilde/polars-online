# Request 3 measured: Huber weights w_t applied to the cross-moments x*y_t only (x*x' shared,
# unweighted) vs full weighted least squares (x*x' weighted too: one Gram per target) vs the
# pseudo-response y* = yhat_ref + s*psi(r/s) on the shared unweighted Gram (Huber 1981's
# algorithm H), iterated as passes. Level 1000, slope 2, 5% one-sided outliers on x>0, MAD scale.
import warnings

import numpy as np
import statsmodels.api as sm

warnings.simplefilter("ignore")
C = 1.345
rng = np.random.default_rng(3)
n = 20_000
x = rng.standard_normal(n)
y = 1000 + 2 * x + 0.5 * rng.standard_normal(n)
bad = (rng.random(n) < 0.10) & (x > 0)
y[bad] += 50
X = np.column_stack([np.ones(n), x])
G = X.T @ X
ref = np.linalg.solve(G, X.T @ y)  # pass 1: least squares as the reference


def scale(r):
    return np.median(np.abs(r)) / 0.6745


def weights(r, s):
    return np.minimum(1.0, C * s / np.maximum(np.abs(r), 1e-300))


truth = (
    sm.RLM(y, X, M=sm.robust.norms.HuberT(C))
    .fit(scale_est=sm.robust.scale.mad, tol=1e-12, maxiter=500)
    .params
)
print(
    f"converged Huber (statsmodels RLM, MAD scale): intercept {truth[0]:.3f} slope {truth[1]:.4f}"
)
print(f"least squares (the reference): intercept {ref[0]:.3f} slope {ref[1]:.4f}")
for name in (
    "cross-moments weighted only (request 3 as asked)",
    "full WLS (a Gram per target)",
    "pseudo-response on the shared Gram (algorithm H)",
):
    b = ref.copy()
    hist = []
    for it in range(1, 31):
        r = y - X @ b
        s = scale(r)
        w = weights(r, s)
        if name.startswith("cross"):
            b = np.linalg.solve(G, X.T @ (w * y))
        elif name.startswith("full"):
            b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        else:
            ystar = X @ b + s * np.clip(r / s, -C, C)
            b = np.linalg.solve(G, X.T @ ystar)
        if it in (1, 2, 5, 10, 30):
            hist.append(f"pass {it}: int {b[0]:9.3f} slope {b[1]:7.4f}")
    print(f"{name}:\n   " + "\n   ".join(hist))
