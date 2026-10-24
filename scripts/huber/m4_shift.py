# Online one-pass Huber with exponential decay (half-life 500 rows) through a +20 level shift at
# row 10,000 and no outliers: rows until the trailing 200-row mean |error| stays under 1.0, for
# the RMS scale (today's), the EW-MAD scale, the EW-MAD with its own shorter memory, and least
# squares (delta = inf). Then the same with 5% one-sided outliers before and after the shift.
import warnings

import numpy as np

warnings.simplefilter("ignore")
DELTA = 1.345
MAD_CONST = 0.6745
LAM = 0.5 ** (1 / 500)


class EwRms:
    def __init__(self, lam):
        self.lam = lam
        self.s2 = 0.0
        self.w = 0.0

    def scale(self):
        return np.sqrt(self.s2) if self.w > 0 else 0.0

    def push(self, e):
        self.w = self.lam * self.w + 1
        self.s2 += (e * e - self.s2) / self.w


class EwMad:  # exact EW median of |e| via weights on a kept sorted list (fine for a simulation)
    def __init__(self, lam):
        self.lam = lam
        self.v = []
        self.w = []

    def scale(self):
        if not self.v:
            return 0.0
        o = np.argsort(self.v)
        cw = np.cumsum(np.array(self.w)[o])
        half = cw[-1] / 2
        return np.array(self.v)[o][np.searchsorted(cw, half)] / MAD_CONST

    def push(self, e):
        self.w = [x * self.lam for x in self.w]
        self.v.append(abs(e))
        self.w.append(1.0)
        if len(self.v) > 4000:
            self.v = self.v[-4000:]
            self.w = self.w[-4000:]


def run(x, y, Scale, delta=DELTA):
    n = len(x)
    G = np.zeros((2, 2))
    r = np.zeros(2)
    beta = np.zeros(2)
    sc = Scale()
    preds = np.zeros(n)
    for i in range(n):
        z = np.array([1.0, x[i]])
        pred = z @ beta
        preds[i] = pred
        e = y[i] - pred
        s = sc.scale()
        w = 1.0 if (s <= 0 or abs(e) <= delta * s) else delta * s / abs(e)
        G = LAM * G + w * np.outer(z, z)
        r = LAM * r + w * z * y[i]
        sc.push(e)
        beta = np.linalg.solve(G + 1e-9 * np.eye(2), r)
    return preds


def reconv(y, preds, at=10_000):
    err = np.abs(y - preds)[at:]
    m = np.convolve(err, np.ones(200) / 200, mode="valid")
    return (
        next((i for i in range(len(m)) if np.all(m[i : i + 500] < 1.0)), None),
        err[:200].mean(),
        err[200:400].mean(),
    )


rng = np.random.default_rng(1)
n = 14_000
x = rng.standard_normal(n)
clean = 2.0 * x + 0.5 * rng.standard_normal(n)
clean[10_000:] += 20.0
dirty = clean.copy()
bad = (rng.random(n) < 0.10) & (x > 0)
dirty[bad] += 50.0
variants = {
    "least squares": (lambda: EwRms(LAM), np.inf),
    "huber, RMS scale (today)": (lambda: EwRms(LAM), DELTA),
    "huber, EW-MAD scale (hl 500)": (lambda: EwMad(LAM), DELTA),
    "huber, EW-MAD scale (hl 100)": (lambda: EwMad(0.5 ** (1 / 100)), DELTA),
}
for label, yy in (("no outliers", clean), ("5% one-sided outliers", dirty)):
    print(f"--- {label}: +20 level shift at row 10,000, fit half-life 500")
    for k, (Sc, d) in variants.items():
        p = run(x, yy, Sc, d)
        back, e1, e2 = reconv(yy, p)
        # slope before the shift as the robustness read
        print(
            f"{k:>30}: rows to re-converge {back}; "
            f"mean |err| rows 0-200 after {e1:5.2f}, 200-400 {e2:5.2f}"
        )
