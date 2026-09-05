"""Reference (oracle) for the clustering models: pure Python, **bit-exact**.

`tests/reference.py` holds the regressions to ~1e-9 with numpy; the
clustering models are held tighter, because their arithmetic is a hard
assignment followed by a mean-form update, and a hard assignment that flips
on a 1-ulp tie moves a whole row from one centre to another. So this module
mirrors `crates/online-core/src/cluster/` operation for operation -- the
same sums in the same order, the same guards, the same `splitmix64` behind
the random seeding rules -- in plain Python floats (IEEE doubles, like
Rust's `f64`; numpy's pairwise `sum` would *not* reproduce the Rust loops)
and the tests ask for equality, not tolerance.

Conventions, shared with the core and the bank (docs/PLAN.md section 3):

- Every output is read from the state *before* the row is learned.
- A row with a null / non-finite / out-of-bound feature, or such a weight,
  is skipped: every output null, nothing learned, the clock still advances.
- ``n_eff`` is the EW weight before the row and before its own decay.
- A zero weight advances the clock and learns nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MASK64 = (1 << 64) - 1
INPUT_BOUND = 1e100
BUF_CAP = 1000
LLOYD_ITERS = 10
LLOYD_RESTARTS = 10
FAR_SIGMAS = 4.0
FAR_SHARE = 0.05
FAR_ROWS = 3
RADIUS_ROWS = 10
NAN = float("nan")


class SplitMix64:
    """splitmix64, as `summary.rs` writes it."""

    def __init__(self, seed: int) -> None:
        self.s = seed & MASK64

    def next_u64(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return float(self.next_u64() >> 11) * (1.0 / float(1 << 53))

    def choice(self, weights: list[float]) -> int:
        n = len(weights)
        u = self.uniform()
        total = 0.0
        for w in weights:
            total += w
        if math.isnan(total) or total <= 0.0 or math.isinf(total):
            return min(int(u * float(n)), n - 1)
        target = u * total
        acc = 0.0
        last = 0
        for i, w in enumerate(weights):
            if w > 0.0:
                acc += w
                last = i
                if acc > target:
                    return i
        return last


def dist2(c: list[float], z: list[float], mw: list[float]) -> float:
    acc = 0.0
    for i in range(len(c)):
        t = z[i] - c[i]
        acc += mw[i] * t * t
    return acc


@dataclass
class Summary:
    n: float
    c: list[float]
    r2: float

    @staticmethod
    def empty(p: int) -> Summary:
        return Summary(0.0, [0.0] * p, 0.0)

    def decay(self, lam: float) -> None:
        self.n *= lam

    def absorb_plain(self, z: list[float], w: float, d2: float) -> None:
        n_new = self.n + w
        if n_new <= 0.0:
            return
        b = w / n_new
        for i in range(len(self.c)):
            self.c[i] += b * (z[i] - self.c[i])
        if math.isfinite(d2):
            self.r2 += b * (d2 - self.r2)
        self.n = n_new

    def merge_plain(self, o: Summary) -> None:
        n_new = self.n + o.n
        if n_new <= 0.0:
            return
        b = o.n / n_new
        for i in range(len(self.c)):
            self.c[i] += b * (o.c[i] - self.c[i])
        self.r2 += b * (o.r2 - self.r2)
        self.n = n_new

    def absorb(self, z: list[float], w: float, mw: list[float]) -> None:
        n_new = self.n + w
        if n_new <= 0.0:
            return
        a, b = self.n / n_new, w / n_new
        q = dist2(self.c, z, mw)
        for i in range(len(self.c)):
            self.c[i] += b * (z[i] - self.c[i])
        if math.isfinite(q):
            self.r2 = a * self.r2 + a * b * q
        self.n = n_new

    def merge_welford(self, o: Summary, mw: list[float]) -> None:
        n_new = self.n + o.n
        if n_new <= 0.0:
            return
        a, b = self.n / n_new, o.n / n_new
        q = dist2(self.c, o.c, mw)
        for i in range(len(self.c)):
            self.c[i] += b * (o.c[i] - self.c[i])
        if math.isfinite(q):
            self.r2 = a * self.r2 + b * o.r2 + a * b * q
        self.n = n_new

    def clear(self) -> None:
        self.n = 0.0
        self.r2 = 0.0
        for i in range(len(self.c)):
            self.c[i] = 0.0


@dataclass
class Moments:
    w: float
    mean: list[float]
    var: list[float]

    @staticmethod
    def new(p: int) -> Moments:
        return Moments(0.0, [0.0] * p, [0.0] * p)

    def decay(self, lam: float) -> None:
        self.w *= lam

    def absorb(self, x: list[float], w: float) -> None:
        w_new = self.w + w
        if w_new <= 0.0:
            return
        a, b = self.w / w_new, w / w_new
        for i in range(len(self.mean)):
            d = x[i] - self.mean[i]
            self.mean[i] += b * d
            self.var[i] = a * self.var[i] + a * b * d * d
        self.w = w_new

    def metric(self, standardize: bool) -> list[float]:
        out = []
        for v in self.var:
            inv = 1.0 / v if v != 0.0 else math.inf
            out.append(inv if standardize and v > 0.0 and math.isfinite(inv) else 1.0)
        return out


def _shrink(dd: list[float], buf: list[list[float]], c: list[float], mw: list[float]) -> None:
    for i, z in enumerate(buf):
        q = dist2(c, z, mw)
        if q < dd[i]:
            dd[i] = q


def _farthest(buf: list[list[float]], k: int, mw: list[float]) -> list[list[float]]:
    out = [list(buf[0])]
    dd = [math.inf] * len(buf)
    _shrink(dd, buf, buf[0], mw)
    while len(out) < k:
        idx = 0
        for i in range(1, len(dd)):
            if dd[i] > dd[idx]:
                idx = i
        out.append(list(buf[idx]))
        _shrink(dd, buf, buf[idx], mw)
    return out


def _kmeanspp(
    buf: list[list[float]], w: list[float], k: int, mw: list[float], rng: SplitMix64
) -> list[list[float]]:
    idx0 = rng.choice(w)
    out = [list(buf[idx0])]
    dd = [math.inf] * len(buf)
    _shrink(dd, buf, buf[idx0], mw)
    while len(out) < k:
        pr = [w[i] * dd[i] for i in range(len(buf))]
        idx = rng.choice(pr)
        out.append(list(buf[idx]))
        _shrink(dd, buf, buf[idx], mw)
    return out


def _lloyd(
    buf: list[list[float]],
    w: list[float],
    centres: list[list[float]],
    mw: list[float],
    iters: int,
) -> list[list[float]]:
    k, p = len(centres), len(mw)
    for _ in range(iters):
        sw = [0.0] * k
        sx = [[0.0] * p for _ in range(k)]
        for z, wi in zip(buf, w, strict=True):
            best, best_d = 0, math.inf
            for j, c in enumerate(centres):
                d = dist2(c, z, mw)
                if d < best_d:
                    best_d, best = d, j
            sw[best] += wi
            for i in range(p):
                sx[best][i] += wi * z[i]
        for j in range(k):
            if sw[j] > 0.0:
                for i in range(p):
                    centres[j][i] = sx[j][i] / sw[j]
    return centres


def _inertia(
    buf: list[list[float]], w: list[float], centres: list[list[float]], mw: list[float]
) -> float:
    total = 0.0
    for z, wi in zip(buf, w, strict=True):
        best = math.inf
        for c in centres:
            d = dist2(c, z, mw)
            if d < best:
                best = d
        total += wi * best
    return total


def seed_centres(
    buf: list[list[float]],
    w: list[float],
    k: int,
    rule: str,
    seed: int,
    allow_dup: bool,
    mw: list[float],
) -> list[list[float]] | None:
    if len(buf) < k:
        return None
    if rule == "first":
        out: list[list[float]] = []
        for z in buf:
            if not any(c == z for c in out):
                out.append(list(z))
                if len(out) == k:
                    return out
        return [list(z) for z in buf[:k]] if allow_dup else None
    if rule == "farthest":
        return _farthest(buf, k, mw)
    rng = SplitMix64(seed)
    if rule == "kmeanspp":
        return _kmeanspp(buf, w, k, mw, rng)
    if rule == "lloyd":
        best: tuple[float, list[list[float]]] | None = None
        for _ in range(LLOYD_RESTARTS):
            centres = _lloyd(buf, w, _kmeanspp(buf, w, k, mw, rng), mw, LLOYD_ITERS)
            cost = _inertia(buf, w, centres, mw)
            if best is None or cost < best[0]:
                best = (cost, centres)
        return None if best is None else best[1]
    msg = f"unknown seed_rule {rule!r}"
    raise ValueError(msg)


@dataclass
class KMeansRef:
    """`online_core::KMeans`, operation for operation."""

    p: int
    k: int
    halflife: float = math.inf
    lam: float | None = None
    min_periods: float = 0.0
    warm_rows: int = 500
    seed_rule: str = "lloyd"
    seed: int = 0
    update_every: int = 1
    split_merge: float = 0.5
    sm_every: int = 100
    dead_frac: float = 0.05
    standardize: bool = True
    moments: Moments = field(init=False)
    mw: list[float] = field(init=False)
    clusters: list[Summary] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    batch: list[Summary] = field(default_factory=list)
    buf: list[list[float]] = field(default_factory=list)
    buf_w: list[float] = field(default_factory=list)
    far: list[Summary] = field(default_factory=list)
    far_rows: list[int] = field(default_factory=list)
    far_factor: float = field(init=False)
    r2_typical: float = 0.0
    far_cut: float = math.inf
    window_w: float = 0.0
    since: int = 0
    since_sm: int = 0
    n_merges: int = 0
    n_dead: int = 0

    def __post_init__(self) -> None:
        self.moments = Moments.new(self.p)
        self.mw = [1.0] * self.p
        self.far_factor = 1.0 + FAR_SIGMAS * math.sqrt(2.0 / self.p)

    # -- the decay, as `Decay::factor` --------------------------------------
    def factor(self, d: float) -> float:
        if self.lam is not None:
            return self.lam**d
        if math.isinf(self.halflife):
            return 1.0
        # `exp2(-x)`, as `online_core::Decay::factor` spells it (LLVM
        # compiles `0.5.powf(x)` to exactly this; libm's `pow` can differ
        # from it in the last bit).
        return math.exp2(-(d / self.halflife))

    @property
    def seeded(self) -> bool:
        return bool(self.clusters)

    @property
    def n_eff(self) -> float:
        return self.moments.w

    def coefficients(self) -> list[list[float]] | None:
        return [list(c.c) for c in self.clusters] if self.seeded else None

    def nearest2(self, z: list[float]) -> tuple[int, float, float]:
        best, best_d, second = 0, math.inf, math.inf
        for j, c in enumerate(self.clusters):
            d = dist2(c.c, z, self.mw)
            if d < best_d:
                second = best_d
                best_d = d
                best = j
            elif d < second:
                second = d
        if len(self.clusters) < 2:
            second = NAN
        return best, best_d, second

    def score(self, x: list[float], valid: bool, n_eff: float) -> list[float]:
        if valid and self.seeded and n_eff >= self.min_periods:
            j, d2, d2s = self.nearest2(x)
            return [float(j), math.sqrt(d2), math.sqrt(d2s) if not math.isnan(d2s) else NAN]
        return [NAN, NAN, NAN]

    def learn_row(self, x: list[float], w: float) -> None:
        j, d2, _ = self.nearest2(x)
        far = self.split_merge > 0.0 and d2 > self.far_cut
        self.absorb(j, x, w, d2, far)
        self.since += 1
        self.since_sm += 1
        if self.since >= self.update_every:
            self.checkpoint()

    def absorb(self, j: int, x: list[float], w: float, d2: float, far: bool) -> None:
        if self.split_merge > 0.0:
            self.window_w += w
        if far:
            self.far[j].absorb(x, w, self.mw)
            self.far_rows[j] += 1
        else:
            self.batch[j].absorb_plain(x, w, d2)
            self.rows[j] += 1

    def checkpoint(self) -> None:
        for c, b in zip(self.clusters, self.batch, strict=True):
            c.merge_plain(b)
            b.clear()
        self.since = 0
        if self.split_merge > 0.0:
            self.refresh_far_cut()
            if self.since_sm >= self.sm_every:
                self.since_sm = 0
                self.winsorize_radii()
                self.refresh_far_cut()
                self.split_merge_check()
                self.window_w = 0.0
                self.far_rows = [0] * self.k
                for f in self.far:
                    f.clear()

    def winsorize_radii(self) -> None:
        for c, f in zip(self.clusters, self.far, strict=True):
            if f.n > 0.0:
                c.r2 = (c.n * c.r2 + f.n * self.far_cut) / (c.n + f.n)

    def refresh_far_cut(self) -> None:
        r2s = [
            c.r2
            for c, rows in zip(self.clusters, self.rows, strict=True)
            if rows >= RADIUS_ROWS and c.r2 > 0.0
        ]
        if not r2s:
            self.r2_typical, self.far_cut = 0.0, math.inf
            return
        total, largest = 0.0, 0.0
        for r2 in r2s:
            total += r2
            largest = max(largest, r2)
        typical = total if len(r2s) == 1 else (total - largest) / (len(r2s) - 1)
        self.r2_typical, self.far_cut = typical, typical * self.far_factor

    def far_source(self) -> int | None:
        best: int | None = None
        for idx, f in enumerate(self.far):
            if f.n > 0.0 and (best is None or f.n > self.far[best].n):
                best = idx
        return best

    def merge_source(self, i: int, j: int) -> int | None:
        best = (i, self.far[i].n + self.far[j].n, self.far_rows[i] + self.far_rows[j])
        for idx, f in enumerate(self.far):
            if idx != i and idx != j and f.n > best[1]:
                best = (idx, f.n, self.far_rows[idx])
        source, n, rows = best
        ok = n > 0.0 and rows >= FAR_ROWS and n >= FAR_SHARE * self.window_w
        return source if ok else None

    def split(self, target: int, source: int) -> None:
        self.clusters[source].n *= 0.5
        n = self.clusters[source].n
        c = list(self.far[source].c)
        self.clusters[target] = Summary(n, c, self.r2_typical)
        self.rows[target] = RADIUS_ROWS

    def split_merge_check(self) -> None:
        k = len(self.clusters)
        if k >= 3:
            best, pair = math.inf, (0, 0)
            for i in range(k):
                for j in range(i + 1, k):
                    ri = math.sqrt(max(self.clusters[i].r2, 0.0))
                    rj = math.sqrt(max(self.clusters[j].r2, 0.0))
                    den = ri + rj
                    d = math.sqrt(dist2(self.clusters[i].c, self.clusters[j].c, self.mw))
                    if d == 0.0:
                        ratio = 0.0
                    elif den > 0.0:
                        ratio = d / den
                    else:
                        ratio = math.inf
                    if ratio < best:
                        best, pair = ratio, (i, j)
            if best < self.split_merge:
                i, j = pair
                source = self.merge_source(i, j)
                if source is None:
                    return
                other = Summary(self.clusters[j].n, list(self.clusters[j].c), self.clusters[j].r2)
                self.clusters[i].merge_welford(other, self.mw)
                self.rows[i] += self.rows[j]
                far_j, self.far[j] = self.far[j], Summary.empty(self.p)
                self.far[i].merge_welford(far_j, self.mw)
                self.far_rows[i] += self.far_rows[j]
                self.far_rows[j] = 0
                self.split(j, source)
                self.n_merges += 1
                return
        if k >= 2 and self.dead_frac > 0.0:
            jd = 0
            for j in range(1, k):
                if self.clusters[j].n < self.clusters[jd].n:
                    jd = j
            floor = self.dead_frac * self.moments.w / float(k)
            if self.clusters[jd].n < floor:
                source = self.far_source()
                if source is not None:
                    self.split(jd, source)
                    self.n_dead += 1

    def buffer_cut(self) -> float:
        total, wsum = 0.0, 0.0
        for z, w in zip(self.buf, self.buf_w, strict=True):
            total += w * dist2(self.moments.mean, z, self.mw)
            wsum += w
        mean = total / wsum if wsum > 0.0 else 0.0
        return mean * self.far_factor

    def try_seed(self) -> None:
        target = max(self.warm_rows, self.k)
        if len(self.buf) < target:
            return
        allow_dup = len(self.buf) >= max(target, BUF_CAP)
        cut = self.buffer_cut()
        kept = [i for i, z in enumerate(self.buf) if dist2(self.moments.mean, z, self.mw) <= cut]
        seeds = None
        if self.k <= len(kept) < len(self.buf):
            seeds = seed_centres(
                [self.buf[i] for i in kept],
                [self.buf_w[i] for i in kept],
                self.k,
                self.seed_rule,
                self.seed,
                allow_dup,
                self.mw,
            )
            if seeds is not None and any(a == b for n, a in enumerate(seeds) for b in seeds[:n]):
                seeds = None
        if seeds is None:
            seeds = seed_centres(
                self.buf, self.buf_w, self.k, self.seed_rule, self.seed, allow_dup, self.mw
            )
        if seeds is None:
            return
        self.clusters = [Summary(0.0, c, 0.0) for c in seeds]
        self.rows = [0] * self.k
        self.batch = [Summary.empty(self.p) for _ in range(self.k)]
        self.far = [Summary.empty(self.p) for _ in range(self.k)]
        self.far_rows = [0] * self.k
        buf, buf_w = self.buf, self.buf_w
        self.buf, self.buf_w = [], []
        for z, w in zip(buf, buf_w, strict=True):
            j, d2, _ = self.nearest2(z)
            far = self.split_merge > 0.0 and dist2(self.moments.mean, z, self.mw) > cut
            self.absorb(j, z, w, d2, far)
        for c, b in zip(self.clusters, self.batch, strict=True):
            c.merge_plain(b)
            b.clear()
        if self.split_merge > 0.0:
            self.refresh_far_cut()

    def step(self, x: list[float], d: float, w: float) -> tuple[list[float], float]:
        """One accepted row: `(pred, n_eff)`, as `OnlineModel::step`."""
        lam = self.factor(d)
        n_before = self.moments.w
        valid = all(math.isfinite(v) for v in x)
        learn = w > 0.0 and math.isfinite(w) and valid
        self.moments.decay(lam)
        for c in self.clusters:
            c.decay(lam)
        for b in self.batch:
            b.decay(lam)
        for f in self.far:
            f.decay(lam)
        self.window_w *= lam
        for i in range(len(self.buf_w)):
            self.buf_w[i] *= lam
        pred = self.score(x, valid, n_before)
        if learn:
            self.moments.absorb(x, w)
            if self.seeded:
                self.learn_row(x, w)
            else:
                self.buf.append(list(x))
                self.buf_w.append(w)
                self.try_seed()
        self.mw = self.moments.metric(self.standardize)
        return pred, n_before

    def predict(self, x: list[float]) -> tuple[list[float], float]:
        valid = all(math.isfinite(v) for v in x)
        return self.score(x, valid, self.moments.w), self.moments.w


def _usable(v: float | None) -> bool:
    return v is not None and math.isfinite(v) and abs(v) <= INPUT_BOUND


def kmeans_ref(
    rows: list[list[float | None]],
    *,
    k: int,
    clock: list[float] | None = None,
    weight: list[float | None] | None = None,
    max_dclock: float = math.inf,
    **params: object,
) -> dict[str, list]:
    """The bank's `kmeans` output over `rows` (one list per row), as lists
    with ``None`` for null: ``cluster`` (ints), ``dist``, ``dist2``, ``n_eff``
    and ``coef`` (the flat centres after the last accepted row, as
    ``coef_every=0`` reports on the final row -- ``None`` before seeding).

    The stream plumbing here is the bank's row plan for one group: a skipped
    row (a null or unusable feature or weight) has every output null and its
    clock delta is folded into the next accepted row's; ``max_dclock`` caps
    each raw delta, as the bank does with a clock column.
    """
    p = len(rows[0])
    model = KMeansRef(p=p, k=k, **params)  # type: ignore[arg-type]
    out: dict[str, list] = {"cluster": [], "dist": [], "dist2": [], "n_eff": []}
    pending = 0.0
    for i, row in enumerate(rows):
        w = 1.0 if weight is None else weight[i]
        if clock is None:
            d = 0.0 if i == 0 else 1.0
        else:
            d = 0.0 if i == 0 else min(clock[i] - clock[i - 1], max_dclock)
        pending += d
        accept = all(_usable(v) for v in row) and _usable(w)
        if not accept:
            for key in out:
                out[key].append(None)
            continue
        pred, n_eff = model.step([float(v) for v in row], pending, float(w))  # type: ignore[arg-type]
        pending = 0.0
        out["cluster"].append(None if math.isnan(pred[0]) else int(pred[0]))
        out["dist"].append(None if math.isnan(pred[1]) else pred[1])
        out["dist2"].append(None if math.isnan(pred[2]) else pred[2])
        out["n_eff"].append(n_eff)
    coef = model.coefficients()
    out["coef"] = [None if coef is None else [v for c in coef for v in c]]
    out["model"] = [model]
    return out
