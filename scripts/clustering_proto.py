"""Online clustering, prototyped: every family that can be made to fit the contract.

The numpy models behind ``docs/CLUSTERING.md``. They exist to measure --
accuracy against batch references, what each knob costs, whether the
guarantees hold -- not to be fast; ``scripts/clustering_experiments.py`` runs
the experiments the document quotes.

Semantics match polars-online's models (docs/PLAN.md §2-§3): per row a decay
``lam = 0.5**(d_clock/halflife)``, a weight ``w`` that scales the row (``w = 0``
advances the clock and learns nothing, legal as the first row), the output
computed *before* the update, ``n_eff`` = the EW weight sum before the row and
before its own decay, ``min_periods`` on that ``n_eff``, and a null feature
skipping the row. State is O(parameters), never O(rows): a warm-up buffer of a
fixed size for seeding, then the cluster summaries only.

Every model keeps its whole state on the object and consumes rows one at a time
through ``step``; ``fit_chunk`` is a loop over ``step``. Any chunking of the
stream therefore gives identical output by construction, and the experiments
assert it bitwise. Where a model has a data-parallel form (``update_every > 1``:
summaries frozen between checkpoints, rows contribute plain weighted sums) the
checkpoint schedule counts learned rows, never chunks, so the same holds.

Models, one class each (the document's §6 explains the choices):

  EWKMeans      sequential k-means: per cluster an EW (weight, centre, radius);
                MacQueen / Bottou-Bengio ``1/n_k`` step under decay; seeding
                rules; Huber-weighted centres; spherical (cosine) distance;
                fuzzy memberships (``fuzzifier > 1`` = single-pass fuzzy
                c-means); dead-cluster re-seeding; ``update_every`` mini-batches
  OnlineGMM     online EM for a Gaussian mixture (Cappe-Moulines eq. 15 with the
                EW step): per-component Welford moments weighted by w * r_k;
                spherical / diag / full covariance with a variance floor
  DPMeans       Kulis-Jordan DP-means made online: a new cluster when the
                nearest centre is farther than ``radius``; capped at
                ``max_clusters`` (evict the lightest); leader variant (frozen)
  MicroClusters DenStream-style micro-clusters with fading = decay: potential vs
                outlier micro-clusters by weight, radius cap ``eps``, capped
                count, DenStream's pruning, optional checkpointed macro step
                (single linkage over the potential micro-clusters)
  SOM           self-organising map on a fixed grid, mean-form neighbourhood
                update (each neuron an EW mean with a neighbourhood-scaled
                weight), fixed neighbourhood width
  GNG           growing neural gas (Fritzke 1995) bounded by ``max_nodes``;
                edges age out; a node is inserted every ``insert_every`` rows;
                connected components labelled at those checkpoints
  ODAC          clusters the *variables* by correlation (Rodrigues et al. 2008,
                as implemented in river): one EW correlation matrix,
                Hoeffding-bound split / aggregate tests every ``n_min`` rows

Shared: ``Stream`` (clock, n_eff, decay, standardisation from pre-row moments,
warm-up buffer), ``seed_centres`` (first-k / farthest-first / k-means++ /
Lloyd on the buffer) and the metrics at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NONE = -1  # "no cluster" label (before seeding / min_periods, or a null row)


def decay_factor(d_clock: float, halflife: float) -> float:
    return 1.0 if math.isinf(halflife) else 0.5 ** (d_clock / halflife)


# --------------------------------------------------------------------------- seeding
def seed_centres(
    B: np.ndarray,
    wB: np.ndarray,
    k: int,
    rule: str,
    seed: int = 0,
    lloyd_iters: int = 10,
    allow_dup: bool = False,
    mw: np.ndarray | None = None,
) -> np.ndarray | None:
    """Pick ``k`` centres from the warm-up buffer ``B`` (n x p, weights ``wB``).

    first     the first k distinct rows (MacQueen's initialisation; Bottou-Bengio §3.3)
    farthest  Gonzalez farthest-first from the first row (deterministic; picks outliers)
    kmeanspp  D^2 sampling with a seeded generator (deterministic given ``seed``)
    lloyd     kmeanspp, then ``lloyd_iters`` weighted Lloyd iterations on the buffer

    Returns ``None`` when the buffer has fewer than ``k`` distinct rows and
    ``allow_dup`` is false (the caller keeps buffering).
    """
    n = len(B)
    mw = np.ones(B.shape[1]) if mw is None else mw

    def d2(A, z):
        d = A - z
        return (d * d * mw).sum(-1)

    if rule == "first":
        idx: list[int] = []
        for i in range(n):
            if all(not np.array_equal(B[i], B[j]) for j in idx):
                idx.append(i)
            if len(idx) == k:
                break
        if len(idx) < k:
            if not allow_dup or n < k:
                return None
            idx = list(range(k))
        return B[idx].copy()
    if n < k:
        return None
    if rule == "farthest":
        idx = [0]
        dd = d2(B, B[0])
        while len(idx) < k:
            i = int(np.argmax(dd))  # first max wins
            idx.append(i)
            dd = np.minimum(dd, d2(B, B[i]))
        return B[idx].copy()
    if rule in ("kmeanspp", "lloyd"):
        rng = np.random.default_rng(seed)
        idx = [int(rng.choice(n, p=wB / wB.sum()))]
        dd = d2(B, B[idx[0]])
        while len(idx) < k:
            pr = wB * dd
            i = int(rng.choice(n)) if pr.sum() <= 0.0 else int(rng.choice(n, p=pr / pr.sum()))
            idx.append(i)
            dd = np.minimum(dd, d2(B, B[i]))
        C = B[idx].copy()
        if rule == "lloyd":
            for _ in range(lloyd_iters):
                lab = np.argmin(np.stack([d2(B, c) for c in C], 1), axis=1)
                for j in range(k):
                    m = lab == j
                    if wB[m].sum() > 0:
                        C[j] = (wB[m, None] * B[m]).sum(0) / wB[m].sum()
        return C
    raise ValueError(f"unknown seed rule {rule!r}")


# --------------------------------------------------------------------------- shared
class Stream:
    """Clock, decay, n_eff, EW feature moments (for standardisation) and the warm-up buffer."""

    #: attributes derived from others -- not part of the state a save/load must carry
    DERIVED = ("mw",)

    def canonical_state(self) -> list[np.ndarray]:
        """Every float the state genuinely carries, for a bitwise comparison."""
        out = []
        for name, v in vars(self).items():
            if name in self.DERIVED:
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                out.append(v.copy())
            elif isinstance(v, float):
                out.append(np.array([v]))
        return out

    def __init__(self, p: int, halflife: float, min_periods: float, standardize: bool):
        self.p = p
        self.halflife = halflife
        self.min_periods = min_periods
        self.standardize = standardize
        self.N = 0.0  # n_eff: EW sum of learned weights
        self.L = 0.0  # cumulative log-decay
        self.n_rows = 0
        self.n_learned = 0
        self.m = np.zeros(p)  # EW mean of the features (Welford, centred)
        self.v = np.zeros(p)  # EW centred variance of the features
        self.mw = np.ones(p)  # DERIVED from v (not state): metric weights, 1 / v when standardizing
        self.buf: list[tuple[np.ndarray, float, float]] = []  # (z, w, L) warm-up rows

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """The state always lives in raw units. Standardisation is in the *metric*
        (``self.mw``), never in the coordinates: rescaling the coordinates as the
        moments move slides every stored centre out from under itself, which is
        worse than not standardising at all (docs/CLUSTERING.md §10)."""
        return x

    def _d2(self, C: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Squared distances from every row of ``C`` to ``z`` under the current metric."""
        d = C - z
        return (d * d * self.mw).sum(-1)

    def _set_metric(self) -> None:
        if self.standardize:
            self.mw = np.where(self.v > 0.0, 1.0 / np.where(self.v > 0.0, self.v, 1.0), 1.0)

    def _begin(self, x, d_clock: float, w: float) -> tuple[float, np.ndarray, bool, bool, float]:
        """Per-row bookkeeping shared by every model: decay, moments, n_eff.

        Returns ``(lam, z, valid, learn, n_eff_before)``; ``z`` is the row scaled
        with the moments *before* the row (E24's rule), ``valid`` is false for a
        null feature (no output, nothing learned), ``learn`` is false for a null
        feature or a zero weight (output, nothing learned). ``n_eff`` is read
        before the update and before the row's own decay (CLAUDE.md rule 8)."""
        x = np.asarray(x, dtype=float)
        lam = decay_factor(d_clock, self.halflife)
        self.L += math.log(lam)
        self.n_rows += 1
        n_before = self.N
        valid = not np.isnan(x).any()
        learn = w > 0.0 and valid
        self._set_metric()  # from the moments *before* this row
        z = self._scale(x)
        if learn:
            N_new = lam * self.N + w
            a, b = lam * self.N / N_new, w / N_new
            delta = x - self.m
            self.m = self.m + b * delta
            self.v = a * self.v + a * b * delta * delta
            self.N = N_new
            self.n_learned += 1
        else:
            self.N = lam * self.N
        return lam, z, valid, learn, n_before

    def replay_weights(self) -> np.ndarray:
        """Weights of the buffered rows as of now: each has decayed since its row."""
        return np.array([w * math.exp(self.L - L) for _, w, L in self.buf])

    def fit_chunk(self, X, d_clock, w=None) -> dict[str, np.ndarray]:
        """Predict-then-learn each row; returns the per-row outputs as arrays."""
        n = len(X)
        w = np.ones(n) if w is None else np.asarray(w, dtype=float)
        rows = [self.step(X[i], float(d_clock[i]), float(w[i])) for i in range(n)]
        out: dict[str, np.ndarray] = {}
        for key in rows[0]:
            vals = [r[key] for r in rows]
            out[key] = np.array(vals) if not isinstance(vals[0], np.ndarray) else np.stack(vals)
        return out

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:  # pragma: no cover
        raise NotImplementedError


def _mean_update(
    n_old: float, lam: float, we: float, c: np.ndarray, z: np.ndarray, R: float, mw=None
):
    """Mean-form update of an EW (weight, centre, radius^2) summary by a row of
    effective weight ``we`` (CLAUDE.md rule 9: guarded when both are zero).

    n' = lam n + we,   a = lam n / n',   b = we / n'
    c' = c + b (z - c)
    R' = a R + a b |z - c|^2        (Welford's centred trace, as ewcov.rs)
    """
    n_new = lam * n_old + we
    if n_new <= 0.0:
        return n_new, c, R
    a, b = lam * n_old / n_new, we / n_new
    delta = z - c
    q = float(delta @ delta) if mw is None else float((delta * delta * mw).sum())
    return n_new, c + b * delta, a * R + a * b * q


# --------------------------------------------------------------------------- k-means
@dataclass
class KMeansCfg:
    k: int = 3
    halflife: float = math.inf
    min_periods: float = 0.0
    warm_rows: int = 0  # 0: the first k distinct rows seed; else buffer this many rows
    seed_rule: str = "first"  # first | farthest | kmeanspp | lloyd
    seed: int = 0
    update_every: int = 1  # learned rows between centre updates (1 = fully online)
    huber_c: float = math.inf  # finite: a row farther than c radii gets weight c / r
    spherical: bool = False  # cosine distance on unit vectors (spherical k-means)
    fuzzifier: float = 0.0  # > 1: fuzzy c-means memberships u_j ~ d_j^(-2/(m-1))
    reseed: bool = False  # at a checkpoint, move a dead cluster to the batch's farthest row
    dead_frac: float = 0.05  # dead: n_j < dead_frac * (n_eff / k)
    reseed_factor: float = 3.0  # ...when that row is farther than this many radii
    split_merge: float = 0.0  # > 0: merge two centres closer than this * (r_i + r_j) and
    # re-place the freed one to split the cluster the batch's farthest row belongs to
    sm_every: int = 100  # learned rows between split-merge attempts (its own, slower clock)
    standardize: bool = False


class EWKMeans(Stream):
    """Sequential k-means with EW-decayed per-cluster (weight n_j, centre c_j, radius^2 R_j).

    Between checkpoints (every ``update_every`` learned rows) the centres are
    frozen; a row adds ``we * z`` / ``we`` / ``we * d^2`` to its cluster's pending
    sums (S, W, Q), all decayed with the clock. At the checkpoint
    ``c_j = (n_j c_j + S_j) / (n_j + W_j)`` -- with ``update_every = 1`` that is
    exactly the per-row mean form, i.e. MacQueen's ``1/n_k`` step with the
    count decayed. ``we`` is the row weight times the Huber weight
    ``min(1, c / (d / radius_j))`` and, when fuzzy, times ``u_j^m``.
    """

    def __init__(self, cfg: KMeansCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        k = cfg.k
        self.C: np.ndarray | None = None
        self.n = np.zeros(k)
        self.R = np.zeros(k)
        self.S = np.zeros((k, p))
        self.W = np.zeros(k)
        self.Q = np.zeros(k)
        self.far: tuple[float, np.ndarray | None, float, int] = (-math.inf, None, 0.0, -1)
        self.since = 0
        self.since_sm = 0
        self.n_reseeds = 0
        self.n_merges = 0

    # -- distances --------------------------------------------------------------
    def _dists(self, z: np.ndarray) -> np.ndarray:
        assert self.C is not None
        if self.cfg.spherical:
            nz = float(np.sqrt(z @ z))
            zn = z / nz if nz > 0 else z
            return np.maximum(1.0 - self.C @ zn, 0.0)  # C rows are unit vectors
        return np.sqrt(self._d2(self.C, z))

    def _memberships(self, d: np.ndarray) -> np.ndarray:
        m = self.cfg.fuzzifier
        if m <= 1.0:
            u = np.zeros(len(d))
            u[int(np.argmin(d))] = 1.0  # first minimum wins
            return u
        if (d == 0.0).any():
            u = np.zeros(len(d))
            u[int(np.argmin(d))] = 1.0
            return u
        inv = d ** (-2.0 / (m - 1.0))
        return inv / inv.sum()

    def assign(self, z: np.ndarray) -> tuple[int, float, float, np.ndarray]:
        """(cluster, distance, distance to the second-nearest centre, memberships)."""
        d = self._dists(z)
        u = self._memberships(d)
        j = int(np.argmax(u)) if self.cfg.fuzzifier > 1.0 else int(np.argmin(d))
        second = float(np.min(np.delete(d, j))) if len(d) > 1 else math.nan
        return j, float(d[j]), second, u

    def predict(self, x) -> tuple[int, float]:
        if self.C is None:
            return NONE, math.nan
        j, d, _, _ = self.assign(self._scale(np.asarray(x, dtype=float)))
        return j, d

    # -- learning ---------------------------------------------------------------
    def _effective(self, j: int, d: float, w: float, u: np.ndarray) -> np.ndarray:
        """Per-cluster effective weights of a row (Huber, fuzzy)."""
        we = w * u
        if math.isfinite(self.cfg.huber_c):
            scale = math.sqrt(self.R[j]) if self.R[j] > 0 else 0.0
            if scale > 0 and d > 0:
                r = d / scale
                we = we * min(1.0, self.cfg.huber_c / r)
        return we

    def _accumulate(self, z: np.ndarray, w: float) -> tuple[int, float]:
        """Add a row to the pending sums with the centres frozen; returns (cluster, distance)."""
        j, d, _, u = self.assign(z)
        we = self._effective(j, d, w, u)
        zz = z
        if self.cfg.spherical:
            nz = float(np.sqrt(z @ z))
            zz = z / nz if nz > 0 else z
        self.S += we[:, None] * zz
        self.W += we
        dd = self._dists(zz) if self.cfg.fuzzifier > 1.0 else None
        self.Q += we * (dd**2 if dd is not None else d * d)
        return j, d

    def _learn(self, lam: float, z: np.ndarray, w: float) -> None:
        j, d = self._accumulate(z, w)
        # farthest row of the batch, in units of its cluster's radius
        ratio = d / math.sqrt(self.R[j]) if self.R[j] > 0 else 0.0
        if ratio > self.far[0]:
            self.far = (ratio, z.copy(), w, j)
        self.since += 1
        self.since_sm += 1
        if self.since >= self.cfg.update_every:
            self._checkpoint()

    def _checkpoint(self) -> None:
        assert self.C is not None
        tot = self.n + self.W
        for j in range(self.cfg.k):
            if tot[j] > 0.0:
                self.C[j] = (self.n[j] * self.C[j] + self.S[j]) / tot[j]
                self.R[j] = (self.n[j] * self.R[j] + self.Q[j]) / tot[j]
                if self.cfg.spherical:
                    nc = float(np.sqrt(self.C[j] @ self.C[j]))
                    if nc > 0:
                        self.C[j] /= nc
        self.n = tot
        self.S[:] = 0.0
        self.W[:] = 0.0
        self.Q[:] = 0.0
        if self.cfg.reseed and self.far[1] is not None:
            dead = int(np.argmin(self.n))
            if (
                self.n[dead] < self.cfg.dead_frac * self.N / self.cfg.k
                and self.far[0] > self.cfg.reseed_factor
            ):
                self._replace(dead, self.far[1], self.far[2], 0.0)
                self.n_reseeds += 1
        if (
            self.cfg.split_merge > 0.0
            and self.far[1] is not None
            and self.cfg.k > 2
            and self.since_sm >= self.cfg.sm_every
        ):
            self.since_sm = 0
            self._split_merge()
        self.far = (-math.inf, None, 0.0, -1)
        self.since = 0

    def _replace(self, j: int, z: np.ndarray, n: float, R: float) -> None:
        zz = z / float(np.sqrt(z @ z)) if self.cfg.spherical and float(z @ z) > 0 else z
        self.C[j] = zz
        self.n[j] = n
        self.R[j] = R

    def _split_merge(self) -> None:
        """Merge the most redundant pair of centres and use the freed one to split
        the cluster the batch's farthest row belongs to.

        Redundancy of a pair is ``d_ij / (r_i + r_j)``: two centres sitting inside
        one another's spread describe one component, and no k-means step can
        separate them again -- the failure that costs the plain model ~0.2 ARI
        under drift. The freed centre goes to the farthest row seen since the last
        checkpoint and takes half of that row's cluster's weight: the split half of
        ISODATA, on EW summaries, at O(k^2) per checkpoint."""
        assert self.C is not None
        k = self.cfg.k
        r = np.sqrt(np.maximum(self.R, 0.0))
        D = np.sqrt(np.stack([self._d2(self.C, c) for c in self.C]))
        best, pair = math.inf, None
        for i in range(k):
            for j in range(i + 1, k):
                den = r[i] + r[j]
                ratio = D[i, j] / den if den > 0 else math.inf
                if ratio < best:
                    best, pair = ratio, (i, j)
        if pair is None or best >= self.cfg.split_merge:
            return
        i, j = pair  # keep i, free j
        tot = self.n[i] + self.n[j]
        if tot > 0:
            a, b = self.n[i] / tot, self.n[j] / tot
            delta = self.C[j] - self.C[i]
            self.C[i] = self.C[i] + b * delta
            self.R[i] = (
                a * self.R[i] + b * self.R[j] + a * b * float((delta * delta * self.mw).sum())
            )
            self.n[i] = tot
        donor = self.far[3]
        if donor == j or donor < 0:
            donor = i
        self.n[donor] *= 0.5
        self._replace(j, self.far[1], self.n[donor], self.R[donor])
        self.n_merges += 1

    def _try_seed(self) -> None:
        target = self.cfg.warm_rows if self.cfg.warm_rows > 0 else self.cfg.k
        if len(self.buf) < target:
            return
        B = np.stack([z for z, _, _ in self.buf])
        wB = self.replay_weights()
        C = seed_centres(
            B,
            wB,
            self.cfg.k,
            self.cfg.seed_rule,
            self.cfg.seed,
            allow_dup=len(self.buf) >= max(target, 1000),
            mw=self.mw,
        )
        if C is None:
            return
        if self.cfg.spherical:
            nrm = np.sqrt((C**2).sum(1, keepdims=True))
            C = C / np.where(nrm > 0, nrm, 1.0)
        self.C = C
        # one frozen pass over the buffer (assign to the seeds, then recompute):
        # the seeds get the weight of the rows they attract, each at its decayed
        # weight. A seed that attracts nothing keeps zero weight and is replaced
        # outright by the first row it wins later (MacQueen's 1/n_k with n_k = 0).
        for (z, _, _), we in zip(self.buf, wB, strict=True):
            self._accumulate(z, float(we))
        self._checkpoint()
        self.buf = []

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        self.n *= lam
        self.S *= lam
        self.W *= lam
        self.Q *= lam
        out = {"cluster": NONE, "dist": math.nan, "second": math.nan, "n_eff": n_before}
        if self.cfg.fuzzifier > 1.0:
            out["membership"] = np.full(self.cfg.k, math.nan)
        if not valid:
            return out
        if self.C is not None and n_before >= self.min_periods:
            j, d, second, u = self.assign(z)
            out.update(cluster=j, dist=d, second=second)
            if self.cfg.fuzzifier > 1.0:
                out["membership"] = u
        if not learn:
            return out
        if self.C is None:
            self.buf.append((z, w, self.L))
            self._try_seed()
            return out
        self._learn(lam, z, w)
        return out

    def centres(self) -> np.ndarray | None:
        return None if self.C is None else self.C.copy()

    def state_doubles(self) -> int:
        k, p = self.cfg.k, self.p
        return k * p + 2 * k + (k * p + 2 * k if self.cfg.update_every > 1 else 0) + 2 * p + 4


# --------------------------------------------------------------------------- GMM
@dataclass
class GMMCfg:
    k: int = 3
    halflife: float = math.inf
    min_periods: float = 0.0
    warm_rows: int = 0
    seed_rule: str = "first"
    seed: int = 0
    cov: str = "diag"  # spherical | diag | full
    var_floor: float = 1e-3  # times the global EW variance, added to every component
    warm_iters: int = 3  # EM iterations on the warm-up buffer before going online
    standardize: bool = False  # no-op: the per-component covariance already is the metric


class OnlineGMM(Stream):
    """Online EM for a Gaussian mixture, Cappe & Moulines (2009) eq. 15 with the EW step.

    The expected sufficient statistics of component j are the EW moments of the
    rows weighted by ``w * r_j`` (responsibility from the parameters *before*
    the row): ``n_j``, the centred mean ``mu_j`` and the centred second moment
    ``Sigma_j`` in Welford form -- ewcov.rs's accumulator with a soft weight.
    ``pi_j = n_j / sum n``. Output per row: the argmax component, the
    Mahalanobis distance to it, the responsibilities and the log-likelihood,
    all before the update.
    """

    def __init__(self, cfg: GMMCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        k = cfg.k
        self.mu: np.ndarray | None = None
        self.n = np.zeros(k)
        if cfg.cov == "full":
            self.Sig = np.zeros((k, p, p))
        elif cfg.cov == "diag":
            self.Sig = np.zeros((k, p))
        else:
            self.Sig = np.zeros(k)

    def _floor(self) -> np.ndarray:
        gv = np.where(self.v > 0.0, self.v, 1.0)
        return self.cfg.var_floor * gv

    def _logdens(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-component log-density and squared Mahalanobis distance."""
        assert self.mu is not None
        k, p = self.cfg.k, self.p
        fl = self._floor()
        logd = np.empty(k)
        maha = np.empty(k)
        for j in range(k):
            delta = z - self.mu[j]
            if self.cfg.cov == "full":
                S = self.Sig[j] + np.diag(fl)
                sign, logdet = np.linalg.slogdet(S)
                q = float(delta @ np.linalg.solve(S, delta))
            elif self.cfg.cov == "diag":
                S = self.Sig[j] + fl
                logdet = float(np.log(S).sum())
                q = float((delta * delta / S).sum())
            else:
                s = float(self.Sig[j] + fl.mean())
                logdet = p * math.log(s)
                q = float(delta @ delta) / s
            maha[j] = q
            logd[j] = -0.5 * (p * math.log(2 * math.pi) + logdet + q)
        return logd, maha

    def responsibilities(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        logd, maha = self._logdens(z)
        tot = self.n.sum()
        logpi = (
            np.log(self.n / tot)
            if tot > 0 and (self.n > 0).all()
            else np.full(self.cfg.k, -math.log(self.cfg.k))
        )
        lp = logd + logpi
        mx = lp.max()
        r = np.exp(lp - mx)
        ll = mx + math.log(r.sum())
        return r / r.sum(), maha, ll

    def predict(self, x) -> tuple[int, float]:
        if self.mu is None:
            return NONE, math.nan
        r, maha, _ = self.responsibilities(self._scale(np.asarray(x, dtype=float)))
        j = int(np.argmax(r))
        return j, math.sqrt(maha[j])

    def _learn(self, lam: float, z: np.ndarray, w: float, r: np.ndarray) -> None:
        assert self.mu is not None
        for j in range(self.cfg.k):
            we = w * r[j]
            n_new = lam * self.n[j] + we
            if n_new <= 0.0:
                self.n[j] = n_new
                continue
            a, b = lam * self.n[j] / n_new, we / n_new
            delta = z - self.mu[j]
            self.mu[j] = self.mu[j] + b * delta
            if self.cfg.cov == "full":
                self.Sig[j] = a * self.Sig[j] + a * b * np.outer(delta, delta)
            elif self.cfg.cov == "diag":
                self.Sig[j] = a * self.Sig[j] + a * b * delta * delta
            else:
                self.Sig[j] = a * self.Sig[j] + a * b * float(delta @ delta) / self.p
            self.n[j] = n_new

    def _try_seed(self) -> None:
        target = self.cfg.warm_rows if self.cfg.warm_rows > 0 else self.cfg.k
        if len(self.buf) < target:
            return
        B = np.stack([z for z, _, _ in self.buf])
        wB = self.replay_weights()
        C = seed_centres(
            B,
            wB,
            self.cfg.k,
            self.cfg.seed_rule,
            self.cfg.seed,
            allow_dup=len(self.buf) >= max(target, 1000),
            mw=self.mw,
        )
        if C is None:
            return
        self.mu = C
        # every component starts with the buffer's global variance, then
        # ``warm_iters`` EM iterations on the buffer: responsibilities from the
        # frozen parameters, moments re-accumulated from zero (an M-step). Cappe &
        # Moulines' M-step is only well defined once enough rows have been seen;
        # updating from a zero count would move every component onto the first row.
        mB = (wB[:, None] * B).sum(0) / wB.sum()
        vB = (wB[:, None] * (B - mB) ** 2).sum(0) / wB.sum()
        vB = np.where(vB > 0, vB, 1.0)
        if self.cfg.cov == "full":
            self.Sig[:] = np.diag(vB)
        elif self.cfg.cov == "diag":
            self.Sig[:] = vB
        else:
            self.Sig[:] = vB.mean()
        for _ in range(max(1, self.cfg.warm_iters)):
            resp = [self.responsibilities(z)[0] for z, _, _ in self.buf]
            self.n[:] = 0.0
            for (z, _, _), we, r in zip(self.buf, wB, resp, strict=True):
                self._learn(1.0, z, float(we), r)
        self.buf = []

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        k = self.cfg.k
        out = {
            "cluster": NONE,
            "dist": math.nan,
            "loglik": math.nan,
            "n_eff": n_before,
            "membership": np.full(k, math.nan),
        }
        r = None
        if not valid:
            self.n *= lam
            return out
        if self.mu is not None:
            r, maha, ll = self.responsibilities(z)
            if n_before >= self.min_periods:
                j = int(np.argmax(r))
                out.update(cluster=j, dist=math.sqrt(maha[j]), loglik=ll, membership=r)
        if not learn:
            self.n *= lam
            return out
        if self.mu is None:
            self.buf.append((z, w, self.L))
            self._try_seed()
            return out
        assert r is not None
        self._learn(lam, z, w, r)
        return out

    def centres(self) -> np.ndarray | None:
        return None if self.mu is None else self.mu.copy()

    def state_doubles(self) -> int:
        k, p = self.cfg.k, self.p
        cov = {"full": k * p * p, "diag": k * p, "spherical": k}[self.cfg.cov]
        return k * p + k + cov + 2 * p + 4


# --------------------------------------------------------------------------- DP-means
@dataclass
class DPCfg:
    radius: float = 1.0  # new cluster when the nearest centre is farther than this (sqrt(lambda))
    max_clusters: int = 50
    halflife: float = math.inf
    min_periods: float = 0.0
    move: bool = True  # False: the leader algorithm (centres never move)
    prune_weight: float = 0.0  # > 0: at checkpoints drop clusters lighter than this
    prune_every: int = 100  # learned rows between prune checkpoints
    standardize: bool = False


class DPMeans(Stream):
    """Kulis & Jordan's DP-means, one pass, with EW summaries.

    A row farther than ``radius`` from every centre starts a new cluster at
    itself (their Algorithm 1, ``min_c d_ic > lambda`` with ``radius^2 =
    lambda``); otherwise it joins the nearest and moves its centre by the mean
    form. Cluster ids are allocated monotonically and never reused. The count
    is capped at ``max_clusters``: when full, the lightest cluster (first by
    weight, then oldest) is evicted to make room. The paper's algorithm is
    order-dependent by construction; this one is that dependence made explicit.
    """

    def __init__(self, cfg: DPCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        self.ids: list[int] = []
        self.C = np.zeros((0, p))
        self.n = np.zeros(0)
        self.R = np.zeros(0)
        self.next_id = 0
        self.since = 0
        self.n_evicted = 0

    def _nearest(self, z: np.ndarray) -> tuple[int, float]:
        d = np.sqrt(self._d2(self.C, z))
        j = int(np.argmin(d))
        return j, float(d[j])

    def predict(self, x) -> tuple[int, float]:
        if len(self.ids) == 0:
            return NONE, math.nan
        j, d = self._nearest(self._scale(np.asarray(x, dtype=float)))
        return self.ids[j], d

    def _create(self, z: np.ndarray, w: float) -> None:
        if len(self.ids) >= self.cfg.max_clusters:
            j = int(np.argmin(self.n))  # lightest; ties -> oldest (first)
            self._drop(j)
            self.n_evicted += 1
        self.ids.append(self.next_id)
        self.next_id += 1
        self.C = np.vstack([self.C, z[None, :]])
        self.n = np.append(self.n, w)
        self.R = np.append(self.R, 0.0)

    def _drop(self, j: int) -> None:
        del self.ids[j]
        self.C = np.delete(self.C, j, axis=0)
        self.n = np.delete(self.n, j)
        self.R = np.delete(self.R, j)

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        self.n *= lam
        out = {
            "cluster": NONE,
            "dist": math.nan,
            "new": False,
            "n_clusters": len(self.ids),
            "n_eff": n_before,
        }
        if not valid:
            return out
        j = -1
        d = math.inf
        if len(self.ids) > 0:
            j, d = self._nearest(z)
            if n_before >= self.min_periods:
                out.update(cluster=self.ids[j], dist=d, new=d > self.cfg.radius)
        if not learn:
            return out
        if d > self.cfg.radius:
            self._create(z, w)
        elif self.cfg.move:
            self.n[j], self.C[j], self.R[j] = _mean_update(
                self.n[j], 1.0, w, self.C[j], z, self.R[j], self.mw
            )
        else:
            self.n[j] += w
        self.since += 1
        if self.cfg.prune_weight > 0.0 and self.since >= self.cfg.prune_every:
            self.since = 0
            for jj in [i for i in range(len(self.ids)) if self.n[i] < self.cfg.prune_weight][::-1]:
                self._drop(jj)
        return out

    def centres(self) -> np.ndarray:
        return self.C.copy()

    def state_doubles(self) -> int:
        return self.cfg.max_clusters * (self.p + 3) + 2 * self.p + 4


# --------------------------------------------------------------------------- micro-clusters
@dataclass
class MicroCfg:
    eps: float = 0.5  # maximum micro-cluster radius (RMS distance to its centre)
    beta_mu: float = 3.0  # weight at which an outlier micro-cluster becomes potential
    max_micro: int = 200
    halflife: float = math.inf
    min_periods: float = 0.0
    prune_every: int = 100  # learned rows between prune / macro checkpoints
    macro_link: float = (
        2.0  # p-MCs with centres within macro_link * eps share a macro label (0: none)
    )
    standardize: bool = False


class MicroClusters(Stream):
    """DenStream's online part (Cao et al. 2006) on EW summaries, plus a checkpointed macro step.

    Micro-cluster = (weight n, centre c, radius^2 R) in mean form -- their
    (w, CF1, CF2) with the fading function ``2^(-lambda t)`` being our decay.
    Merging (their Algorithm 1): try the nearest potential micro-cluster; if
    the merged radius stays <= eps, merge; else try the nearest outlier
    micro-cluster the same way, promoting it when its weight reaches
    ``beta_mu``; else open a new outlier micro-cluster at the row. Pruning at
    checkpoints (their Algorithm 2): a potential micro-cluster lighter than
    ``beta_mu`` is deleted; an outlier micro-cluster lighter than
    ``xi(age) = (2^(-lambda (age + Tp)) - 1) / (2^(-lambda Tp) - 1)`` (their eq.
    4.2, ``Tp`` from eq. 4.1) is deleted. Both are evaluated on a learned-row
    schedule rather than a clock one, so that a chunking of the stream cannot
    move them. The count is capped: when full, the lightest outlier
    micro-cluster (else the lightest potential one) is evicted. The macro step
    is single linkage over the potential micro-clusters (centres within
    ``macro_link * eps``), O(M^2) at a checkpoint, labels = the smallest id in
    the component. Rows report the nearest potential micro-cluster's macro
    label, the distance to its centre and whether they would be absorbed by a
    potential micro-cluster at all (``outlier``).
    """

    def __init__(self, cfg: MicroCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        self.ids: list[int] = []
        self.C = np.zeros((0, p))
        self.n = np.zeros(0)
        self.R = np.zeros(0)
        self.Lc: list[float] = []  # log-decay at creation
        self.pot: list[bool] = []
        self.macro: dict[int, int] = {}
        self.next_id = 0
        self.since = 0
        self.n_evicted = 0
        self.n_pruned = 0

    def _merged_radius(self, j: int, z: np.ndarray, w: float) -> float:
        n_new, _, R_new = _mean_update(self.n[j], 1.0, w, self.C[j], z, self.R[j], self.mw)
        return math.sqrt(max(R_new, 0.0))

    def _nearest(self, z: np.ndarray, potential: bool) -> tuple[int, float]:
        idx = [i for i in range(len(self.ids)) if self.pot[i] == potential]
        if not idx:
            return -1, math.inf
        d = np.sqrt(self._d2(self.C[idx], z))
        a = int(np.argmin(d))
        return idx[a], float(d[a])

    def _decide(self, z: np.ndarray, w: float) -> tuple[int, float, bool]:
        """Where the row would go: (index, or -1 for a new micro-cluster; the distance
        to the nearest potential micro-cluster; whether this is an outlier)."""
        jp, dp = self._nearest(z, True)
        if jp >= 0 and self._merged_radius(jp, z, w) <= self.cfg.eps:
            return jp, dp, False
        jo, _ = self._nearest(z, False)
        if jo >= 0 and self._merged_radius(jo, z, w) <= self.cfg.eps:
            return jo, dp, True
        return -1, dp, True

    def predict(self, x) -> tuple[int, float]:
        z = self._scale(np.asarray(x, dtype=float))
        jp, dp = self._nearest(z, True)
        if jp < 0:
            return NONE, math.nan
        return self.macro.get(self.ids[jp], self.ids[jp]), dp

    def _drop(self, j: int) -> None:
        self.macro.pop(self.ids[j], None)
        del self.ids[j]
        del self.Lc[j]
        del self.pot[j]
        self.C = np.delete(self.C, j, axis=0)
        self.n = np.delete(self.n, j)
        self.R = np.delete(self.R, j)

    def _create(self, z: np.ndarray, w: float) -> None:
        if len(self.ids) >= self.cfg.max_micro:
            cand = [i for i in range(len(self.ids)) if not self.pot[i]] or list(
                range(len(self.ids))
            )
            j = min(cand, key=lambda i: (self.n[i], i))
            self._drop(j)
            self.n_evicted += 1
        self.ids.append(self.next_id)
        self.next_id += 1
        self.Lc.append(self.L)
        self.pot.append(w >= self.cfg.beta_mu)
        self.C = np.vstack([self.C, z[None, :]])
        self.n = np.append(self.n, w)
        self.R = np.append(self.R, 0.0)

    def _checkpoint(self) -> None:
        cfg = self.cfg
        # DenStream eq. 4.1 / 4.2 in clock units: lambda = 1 / halflife (fading 2^(-t/halflife))
        if math.isfinite(cfg.halflife) and cfg.beta_mu > 1.0:
            Tp = math.ceil(cfg.halflife * math.log2(cfg.beta_mu / (cfg.beta_mu - 1.0)))
            fTp = 2.0 ** (-Tp / cfg.halflife)
        else:
            Tp, fTp = None, None
        for j in reversed(range(len(self.ids))):
            if self.pot[j]:
                if self.n[j] < cfg.beta_mu:
                    self._drop(j)
                    self.n_pruned += 1
            elif fTp is not None:
                age_decay = math.exp(self.L - self.Lc[j])  # 2^(-lambda (tc - to))
                xi = (age_decay * fTp - 1.0) / (fTp - 1.0)
                if self.n[j] < xi:
                    self._drop(j)
                    self.n_pruned += 1
        # macro step: single linkage over the potential micro-clusters
        idx = [i for i in range(len(self.ids)) if self.pot[i]]
        parent = {self.ids[i]: self.ids[i] for i in idx}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        if cfg.macro_link > 0.0 and len(idx) > 1:
            Cp = self.C[idx]
            D = np.sqrt(np.stack([self._d2(Cp, c) for c in Cp]))
            thr = cfg.macro_link * cfg.eps
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    if D[a, b] <= thr:
                        ra, rb = find(self.ids[idx[a]]), find(self.ids[idx[b]])
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
        self.macro = {self.ids[i]: find(self.ids[i]) for i in idx}

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        self.n *= lam
        out = {
            "cluster": NONE,
            "micro": NONE,
            "dist": math.nan,
            "outlier": False,
            "n_micro": len(self.ids),
            "n_potential": int(sum(self.pot)),
            "n_eff": n_before,
        }
        if not valid:
            return out
        target, dp, outlier = self._decide(z, w if learn else 1.0)
        if n_before >= self.min_periods and len(self.ids) > 0:
            jp, _ = self._nearest(z, True)
            if jp >= 0:
                out.update(cluster=self.macro.get(self.ids[jp], self.ids[jp]), dist=dp)
            out.update(micro=self.ids[target] if target >= 0 else NONE, outlier=outlier)
        if not learn:
            return out
        if target < 0:
            self._create(z, w)
        else:
            self.n[target], self.C[target], self.R[target] = _mean_update(
                self.n[target], 1.0, w, self.C[target], z, self.R[target], self.mw
            )
            if not self.pot[target] and self.n[target] >= self.cfg.beta_mu:
                self.pot[target] = True
        self.since += 1
        if self.since >= self.cfg.prune_every:
            self.since = 0
            self._checkpoint()
        return out

    def centres(self) -> np.ndarray:
        return self.C[[i for i in range(len(self.ids)) if self.pot[i]]].copy()

    def state_doubles(self) -> int:
        return self.cfg.max_micro * (self.p + 5) + 2 * self.p + 4


# --------------------------------------------------------------------------- SOM
@dataclass
class SOMCfg:
    rows: int = 4
    cols: int = 4
    halflife: float = math.inf
    min_periods: float = 0.0
    sigma: float = 1.0  # neighbourhood width in grid units (fixed)
    warm_rows: int = 0
    seed_rule: str = "first"
    seed: int = 0
    standardize: bool = False


class SOM(Stream):
    """Self-organising map on a fixed grid, in mean form.

    Every neuron is an EW mean; a row of weight ``w`` with best-matching unit
    ``b`` reaches neuron ``j`` with weight ``w * h(j, b)``, ``h = exp(-g^2 / 2
    sigma^2)`` for grid distance ``g``. With ``sigma -> 0`` this is EWKMeans on
    ``rows * cols`` centres; Kohonen's shrinking schedule is replaced by a fixed
    ``sigma`` so that the map is a stationary estimator under decay. Output:
    the BMU index, the quantisation error and the second BMU (a topographic
    error is the fraction of rows whose two BMUs are not grid neighbours).
    """

    def __init__(self, cfg: SOMCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        K = cfg.rows * cfg.cols
        self.K = K
        g = np.array([(i // cfg.cols, i % cfg.cols) for i in range(K)], dtype=float)
        G2 = ((g[:, None, :] - g[None, :, :]) ** 2).sum(2)
        self.H = np.exp(-G2 / (2 * cfg.sigma**2))
        self.adj = G2 <= 2.0  # 8-neighbourhood
        self.C: np.ndarray | None = None
        self.n = np.zeros(K)

    def _bmu(self, z: np.ndarray) -> tuple[int, float, int]:
        assert self.C is not None
        d = np.sqrt(self._d2(self.C, z))
        b = int(np.argmin(d))
        b2 = int(np.argmin(np.where(np.arange(self.K) == b, math.inf, d)))
        return b, float(d[b]), b2

    def predict(self, x) -> tuple[int, float]:
        if self.C is None:
            return NONE, math.nan
        b, d, _ = self._bmu(self._scale(np.asarray(x, dtype=float)))
        return b, d

    def _learn(self, lam: float, z: np.ndarray, w: float) -> None:
        assert self.C is not None
        b, _, _ = self._bmu(z)
        we = w * self.H[b]
        n_new = lam * self.n + we
        ok = n_new > 0
        bcoef = np.where(ok, we / np.where(ok, n_new, 1.0), 0.0)
        self.C = self.C + bcoef[:, None] * (z - self.C)
        self.n = n_new

    def _try_seed(self) -> None:
        target = self.cfg.warm_rows if self.cfg.warm_rows > 0 else self.K
        if len(self.buf) < target:
            return
        B = np.stack([z for z, _, _ in self.buf])
        wB = self.replay_weights()
        C = seed_centres(
            B,
            wB,
            self.K,
            self.cfg.seed_rule,
            self.cfg.seed,
            allow_dup=len(self.buf) >= max(target, 1000),
            mw=self.mw,
        )
        if C is None:
            return
        self.C = C
        # one frozen pass (batch-SOM step): neurons take the neighbourhood-weighted
        # mean of the rows, and that weight as their count
        S = np.zeros_like(C)
        W = np.zeros(self.K)
        for (z, _, _), we in zip(self.buf, wB, strict=True):
            b, _, _ = self._bmu(z)
            h = float(we) * self.H[b]
            S += h[:, None] * z
            W += h
        ok = W > 0
        self.C[ok] = S[ok] / W[ok, None]
        self.n = W
        self.buf = []

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        out = {"cluster": NONE, "dist": math.nan, "second": NONE, "n_eff": n_before}
        if not valid:
            self.n *= lam
            return out
        if self.C is not None and n_before >= self.min_periods:
            b, d, b2 = self._bmu(z)
            out.update(cluster=b, dist=d, second=b2)
        if not learn:
            self.n *= lam
            return out
        if self.C is None:
            self.n *= lam
            self.buf.append((z, w, self.L))
            self._try_seed()
            return out
        self._learn(lam, z, w)
        return out

    def centres(self) -> np.ndarray | None:
        return None if self.C is None else self.C.copy()

    def state_doubles(self) -> int:
        return self.K * (self.p + 1) + 2 * self.p + 4


# --------------------------------------------------------------------------- GNG
@dataclass
class GNGCfg:
    max_nodes: int = 30
    insert_every: int = 100  # learned rows between insertions (Fritzke's lambda)
    eps_b: float = 0.05  # step of the winner
    eps_n: float = 0.005  # step of its neighbours
    a_max: int = 50  # edge age limit
    alpha: float = 0.5  # error discount of the two nodes an insertion splits
    halflife: float = math.inf  # error decay (Fritzke's d) rides on the clock decay
    min_periods: float = 0.0
    standardize: bool = False


class GNG(Stream):
    """Growing neural gas (Fritzke 1995), bounded.

    Per row: the two nearest nodes s1, s2; edges from s1 age by one; s1's error
    grows by ``w d^2``; s1 moves by ``w eps_b (z - s1)``, its neighbours by ``w
    eps_n``; the edge s1-s2 is refreshed; edges older than ``a_max`` drop, then
    isolated nodes. Every ``insert_every`` learned rows a node is inserted
    halfway between the max-error node q and its max-error neighbour f (up to
    ``max_nodes``; ids monotone), and the connected components of the graph
    are relabelled (label = the smallest node id in the component). The
    accumulated errors decay with the clock instead of Fritzke's constant
    ``d``. Constant steps make this a constant-gain model like ``sgd``, not an
    averaging one.
    """

    def __init__(self, cfg: GNGCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, cfg.standardize)
        self.cfg = cfg
        self.ids: list[int] = []
        self.C = np.zeros((0, p))
        self.err = np.zeros(0)
        self.edges: dict[tuple[int, int], int] = {}  # (id_a, id_b) a<b -> age
        self.comp: dict[int, int] = {}
        self.next_id = 0
        self.since = 0

    def _pos(self, nid: int) -> int:
        return self.ids.index(nid)

    def _nearest2(self, z: np.ndarray) -> tuple[int, int, float]:
        d = self._d2(self.C, z)
        s1 = int(np.argmin(d))
        s2 = int(np.argmin(np.where(np.arange(len(d)) == s1, math.inf, d))) if len(d) > 1 else -1
        return s1, s2, float(d[s1])

    def predict(self, x) -> tuple[int, float]:
        if len(self.ids) == 0:
            return NONE, math.nan
        s1, _, d2 = self._nearest2(self._scale(np.asarray(x, dtype=float)))
        return self.comp.get(self.ids[s1], self.ids[s1]), math.sqrt(d2)

    def _add_node(self, z: np.ndarray, err: float) -> int:
        self.ids.append(self.next_id)
        self.next_id += 1
        self.C = np.vstack([self.C, z[None, :]])
        self.err = np.append(self.err, err)
        return self.ids[-1]

    def _neighbours(self, nid: int) -> list[int]:
        out = []
        for a, b in self.edges:
            if a == nid:
                out.append(b)
            elif b == nid:
                out.append(a)
        return out

    def _drop_node(self, nid: int) -> None:
        j = self._pos(nid)
        del self.ids[j]
        self.C = np.delete(self.C, j, axis=0)
        self.err = np.delete(self.err, j)
        self.comp.pop(nid, None)

    def _components(self) -> None:
        parent = {i: i for i in self.ids}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in self.edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        self.comp = {i: find(i) for i in self.ids}

    def _insert(self) -> None:
        if len(self.ids) >= self.cfg.max_nodes or len(self.ids) < 2:
            return
        q = int(np.argmax(self.err))
        qid = self.ids[q]
        nb = self._neighbours(qid)
        if not nb:
            return
        fid = max(nb, key=lambda i: (self.err[self._pos(i)], -i))
        f = self._pos(fid)
        r = (self.C[q] + self.C[f]) / 2.0
        self.err[q] *= self.cfg.alpha
        self.err[f] *= self.cfg.alpha
        rid = self._add_node(r, self.err[q])
        self.edges.pop((min(qid, fid), max(qid, fid)), None)
        self.edges[(min(qid, rid), max(qid, rid))] = 0
        self.edges[(min(fid, rid), max(fid, rid))] = 0

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        self.err *= lam
        out = {
            "cluster": NONE,
            "node": NONE,
            "dist": math.nan,
            "n_nodes": len(self.ids),
            "n_eff": n_before,
        }
        if not valid:
            return out
        if len(self.ids) > 0 and n_before >= self.min_periods:
            s1, _, d2 = self._nearest2(z)
            out.update(
                cluster=self.comp.get(self.ids[s1], self.ids[s1]),
                node=self.ids[s1],
                dist=math.sqrt(d2),
            )
        if not learn:
            return out
        if len(self.ids) < 2:
            self._add_node(z, 0.0)
            if len(self.ids) == 2:
                self.edges[(self.ids[0], self.ids[1])] = 0
                self._components()
            return out
        s1, s2, d2 = self._nearest2(z)
        id1, id2 = self.ids[s1], self.ids[s2]
        for key in list(self.edges):
            if id1 in key:
                self.edges[key] += 1
        self.err[s1] += w * d2
        self.C[s1] += min(1.0, w * self.cfg.eps_b) * (z - self.C[s1])
        for nid in self._neighbours(id1):
            j = self._pos(nid)
            self.C[j] += min(1.0, w * self.cfg.eps_n) * (z - self.C[j])
        self.edges[(min(id1, id2), max(id1, id2))] = 0
        for key in [k for k, age in self.edges.items() if age > self.cfg.a_max]:
            del self.edges[key]
        linked = {a for e in self.edges for a in e}
        for nid in [i for i in self.ids if i not in linked]:
            if len(self.ids) > 2:
                self._drop_node(nid)
        self.since += 1
        if self.since >= self.cfg.insert_every:
            self.since = 0
            self._insert()
            self._components()
        return out

    def centres(self) -> np.ndarray:
        return self.C.copy()

    def state_doubles(self) -> int:
        m = self.cfg.max_nodes
        return m * (self.p + 2) + 3 * m + 2 * self.p + 4  # nodes + up to ~3m edges


# --------------------------------------------------------------------------- ODAC
@dataclass
class ODACCfg:
    halflife: float = math.inf
    min_periods: float = 0.0
    n_min: int = 100  # learned rows between structure tests
    confidence: float = 0.9  # Hoeffding bound: e = sqrt(ln(1 / confidence) / (2 n))
    tau: float = 0.1  # split when the Hoeffding bound has shrunk below tau (ties)


class _Leaf:
    __slots__ = ("id", "vars", "d1_split", "e_split", "n", "parent", "active", "children")

    def __init__(self, id_: int, vars_: list[int], parent: _Leaf | None):
        self.id = id_
        self.vars = vars_
        self.d1_split = math.nan
        self.e_split = math.nan
        self.n = 0.0
        self.parent = parent
        self.active = True
        self.children: tuple[_Leaf, _Leaf] | None = None


class ODAC(Stream):
    """Online divisive-agglomerative clustering of the *variables* (Rodrigues, Gama & Pedroso 2008).

    Dissimilarity ``rnomc(a, b) = sqrt((1 - corr(a, b)) / 2)`` from one EW
    correlation matrix (ewcov.rs's Welford co-moments; river keeps a separate
    Pearson accumulator per pair per leaf and resets it on aggregation, where
    the damped window here forgets instead). Every ``n_min`` learned rows each
    active leaf computes ``d0 = min``, ``d1 = max``, ``d2 = second max`` and the
    average ``avg`` of its pairwise dissimilarities and the Hoeffding bound
    ``e``; it splits at ``d1``'s pair when ``(d1 - d2 > e or tau > e) and (d1 -
    d0) |d1 + d0 - 2 avg| > e``, and a leaf whose ``d1`` exceeds its parent's
    ``d1`` at the time of the split by more than ``max(e_parent, e)`` folds
    back into the parent (river/cluster/odac.py ``test_split`` /
    ``test_aggregate``; ``avg`` here is over pairs, river divides by the
    observation count). Output: the leaf id of every variable -- a per-variable
    label vector, so a ``coef``-shaped output, not a per-row one.
    """

    def __init__(self, cfg: ODACCfg, p: int):
        super().__init__(p, cfg.halflife, cfg.min_periods, False)
        self.cfg = cfg
        self.Cc = np.zeros((p, p))  # EW centred co-moments of the variables
        self.root = _Leaf(0, list(range(p)), None)
        self.next_id = 1
        self.since = 0
        self.n_splits = 0
        self.n_merges = 0

    def _leaves(self) -> list[_Leaf]:
        out, stack = [], [self.root]
        while stack:
            nd = stack.pop()
            if nd.active:
                out.append(nd)
            elif nd.children is not None:
                stack.extend(nd.children)
        return out

    def labels(self) -> np.ndarray:
        lab = np.full(self.p, NONE)
        for leaf in self._leaves():
            lab[leaf.vars] = leaf.id
        return lab

    def _rnomc(self, vars_: list[int]) -> np.ndarray:
        sub = self.Cc[np.ix_(vars_, vars_)]
        sd = np.sqrt(np.diag(sub))
        sd = np.where(sd > 0, sd, 1.0)
        corr = np.clip(sub / np.outer(sd, sd), -1.0, 1.0)
        return np.sqrt(np.abs((1.0 - corr) / 2.0))

    def _test(self, leaf: _Leaf) -> None:
        cfg = self.cfg
        if len(leaf.vars) < 2 or leaf.n <= 0:
            return
        D = self._rnomc(leaf.vars)
        iu = np.triu_indices(len(leaf.vars), 1)
        vals = D[iu]
        e = math.sqrt(math.log(1.0 / cfg.confidence) / (2.0 * leaf.n))
        order = np.argsort(-vals, kind="stable")
        d1 = float(vals[order[0]])
        d2 = float(vals[order[1]]) if len(vals) > 1 else math.nan
        d0 = float(vals.min())
        avg = float(vals.mean())
        # aggregate first (river's order): fold into the parent when this leaf grew apart
        par = leaf.parent
        if (
            par is not None
            and math.isfinite(par.d1_split)
            and d1 - par.d1_split > max(par.e_split, e)
        ):
            par.active = True
            par.children = None
            par.n = 0.0
            self.n_merges += 1
            return
        if (
            not math.isnan(d2)
            and ((d1 - d2) > e or cfg.tau > e)
            and (d1 - d0) * abs(d1 + d0 - 2 * avg) > e
        ):
            a, b = int(iu[0][order[0]]), int(iu[1][order[0]])
            pa, pb = leaf.vars[a], leaf.vars[b]
            va, vb = [pa], [pb]
            for i, v in enumerate(leaf.vars):
                if v in (pa, pb):
                    continue
                (va if D[i, a] < D[i, b] else vb).append(v)
            leaf.d1_split, leaf.e_split = d1, e
            leaf.children = (_Leaf(self.next_id, va, leaf), _Leaf(self.next_id + 1, vb, leaf))
            self.next_id += 2
            leaf.active = False
            self.n_splits += 1

    def step(self, x, d_clock: float, w: float = 1.0) -> dict:
        x = np.asarray(x, dtype=float)
        N_old, m_old = self.N, self.m.copy()
        lam, z, valid, learn, n_before = self._begin(x, d_clock, w)
        out = {"labels": self.labels(), "n_leaves": len(self._leaves()), "n_eff": n_before}
        if learn:
            N_new = lam * N_old + w
            a, b = lam * N_old / N_new, w / N_new
            delta = x - m_old
            self.Cc = a * self.Cc + a * b * np.outer(delta, delta)
        else:
            return out
        for leaf in self._leaves():
            leaf.n = lam * leaf.n + w
        self.since += 1
        if self.since >= self.cfg.n_min:
            self.since = 0
            for leaf in self._leaves():
                self._test(leaf)
        return out

    def state_doubles(self) -> int:
        return self.p * self.p + 2 * self.p + 4 + 4 * (2 * self.p)


# --------------------------------------------------------------------------- metrics
def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(M, (ia, ib), 1)
    return M


def ari(truth: np.ndarray, pred: np.ndarray) -> float:
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    M = contingency(truth, pred)
    n = M.sum()
    if n < 2:
        return math.nan

    def c2(v):
        return (v * (v - 1) / 2.0).sum()

    s_ij, s_a, s_b = c2(M), c2(M.sum(1)), c2(M.sum(0))
    expected = s_a * s_b / c2(np.array([n]))
    mx = 0.5 * (s_a + s_b)
    return float(1.0 if mx == expected else (s_ij - expected) / (mx - expected))


def purity(truth: np.ndarray, pred: np.ndarray) -> float:
    M = contingency(truth, pred)
    return float(M.max(0).sum() / M.sum())


def nmi(truth: np.ndarray, pred: np.ndarray) -> float:
    M = contingency(truth, pred).astype(float)
    n = M.sum()
    pa, pb = M.sum(1) / n, M.sum(0) / n
    P = M / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(P * np.log(P / np.outer(pa, pb)))
        ha = -np.nansum(pa * np.log(pa))
        hb = -np.nansum(pb * np.log(pb))
    return float(mi / math.sqrt(ha * hb)) if ha > 0 and hb > 0 else 0.0


def simplified_silhouette(dist: np.ndarray, second: np.ndarray) -> float:
    """Mean of (b - a) / max(a, b) with a = distance to own centre, b = to the nearest other."""
    m = np.isfinite(dist) & np.isfinite(second)
    if not m.any():
        return math.nan
    a, b = dist[m], second[m]
    den = np.maximum(a, b)
    return float(np.mean(np.where(den > 0, (b - a) / np.where(den > 0, den, 1.0), 0.0)))
