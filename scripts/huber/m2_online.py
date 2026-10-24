# One-pass online Huber as the library does it (weight by the prior residual against the fit
# before the row, scale from earlier residuals, Gram accumulated, solve every row), no decay,
# level 1,000, intercept + one feature; three scales; plus a trailing re-weight pass over the
# last R rows every R rows (the Gram corrected by the weight change, rounding aside).
import bisect
import warnings

import numpy as np

warnings.simplefilter("ignore")
DELTA = 1.345
P2_CONST = 0.7102
MAD_CONST = 0.6745


def stream(n, frac, level, seed, lift=50.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
    bad = (rng.random(n) < frac) & (x > 0)
    y[bad] += lift
    return x, y


class Rms:
    def __init__(self):
        self.s2 = 0.0
        self.n = 0

    def scale(self):
        return np.sqrt(self.s2) if self.n else 0.0

    def push(self, e):
        self.n += 1
        self.s2 += (e * e - self.s2) / self.n


class P2:
    def __init__(self):
        self.s2 = 0.0
        self.n = 0
        self.rms = Rms()

    def scale(self):
        return np.sqrt(self.s2) if self.n > 22 else self.rms.scale()

    def push(self, e):
        self.rms.push(e)
        s = self.scale()
        self.n += 1
        c = min(e * e, (DELTA * s) ** 2) if s > 0 else e * e
        self.s2 += (c / P2_CONST - self.s2) / self.n


class Mad:
    def __init__(self):
        self.a = []

    def scale(self):
        n = len(self.a)
        if n == 0:
            return 0.0
        m = self.a[n // 2] if n % 2 else 0.5 * (self.a[n // 2 - 1] + self.a[n // 2])
        return m / MAD_CONST

    def push(self, e):
        bisect.insort(self.a, abs(e))


def run(x, y, Scale, R=0, passes=0):
    n = len(x)
    G = np.zeros((2, 2))
    r = np.zeros(2)
    beta = np.zeros(2)
    sc = Scale()
    rows = []  # (z, y, w) of the trailing buffer
    for i in range(n):
        z = np.array([1.0, x[i]])
        pred = z @ beta
        e = y[i] - pred
        s = sc.scale()
        w = 1.0 if (s <= 0 or abs(e) <= DELTA * s) else DELTA * s / abs(e)
        G += w * np.outer(z, z)
        r += w * z * y[i]
        sc.push(e)
        rows.append([z, y[i], w])
        beta = np.linalg.solve(G + 1e-9 * np.eye(2), r)
        if R and (i + 1) % R == 0:
            buf = rows[-R:]
            for _ in range(passes):
                s = sc.scale()
                for row in buf:
                    zz, yy, w_old = row
                    e = yy - zz @ beta
                    w_new = 1.0 if (s <= 0 or abs(e) <= DELTA * s) else DELTA * s / abs(e)
                    G += (w_new - w_old) * np.outer(zz, zz)
                    r += (w_new - w_old) * zz * yy
                    row[2] = w_new
                beta = np.linalg.solve(G + 1e-9 * np.eye(2), r)
            rows = rows[-R:]
    return beta[1]


variants = {
    "online RMS": (Rms, 0, 0),
    "online P2": (P2, 0, 0),
    "online MAD": (Mad, 0, 0),
    "MAD +1 pass/256": (Mad, 256, 1),
    "MAD +3 passes/256": (Mad, 256, 3),
    "MAD +1 pass/2048": (Mad, 2048, 1),
}


def summary(v):
    return f"{np.median(v):.3f} [{min(v):.3f},{max(v):.3f}]"


print(
    "one-pass online Huber slope, truth 2.0, 20,000 rows, level 1,000, no decay; "
    "median [min,max] over 5 seeds"
)
print(f"{'contam.':>8} " + " ".join(f"{k:>20}" for k in variants))
for frac in (0.0, 0.02, 0.10, 0.25):
    cols = {k: [] for k in variants}
    for seed in range(5):
        x, y = stream(20_000, frac, 1000.0, seed)
        for k, (Sc, R, p) in variants.items():
            cols[k].append(run(x, y, Sc, R, p))
    print(f"{frac / 2:>7.1%} " + " ".join(f"{summary(cols[k]):>20}" for k in variants))
