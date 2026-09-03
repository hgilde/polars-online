"""Online gradient-boosted trees, prototyped: EW-decayed gradient histograms.

The numpy model behind ``docs/BOOSTED-TREES.md``. It exists to measure
behaviour -- accuracy against XGBoost, what each knob costs, whether the
guarantees hold -- not to be fast; ``scripts/ogbt_experiments.py`` runs the
experiments the document quotes.

Semantics match polars-online's models: per row a decay
``lam = 0.5**(d_clock/halflife)``, a weight ``w`` that scales the row
(``w = 0`` advances the clock and learns nothing), ``pred`` computed before
the update, and ``n_eff`` = the EW weight sum before the row.

The ensemble (tree structure, leaf values, base score) changes only at
*checkpoints*, every ``grow_every`` learned rows (``grow_every = 1`` is fully
online). Between checkpoints the ensemble is frozen, so the rows of a segment
are independent of one another: predictions and gradients come from the
frozen ensemble and the histogram contributions are plain weighted sums. That
is what makes the fit data-parallel and what makes any chunking of the stream
give identical output.

Decay is lazy and exact: the stream keeps a cumulative log-decay ``L`` and
every node stores the ``L`` at which it was last brought up to date; bringing
a node up to date multiplies its sums by ``exp(L_now - L_node)``.

Two designs share the file. ``OnlineGBT`` grows, prunes and refreshes its own
trees (optionally warm-started batch-style on the warm-up buffer);
``LeafRefresh`` keeps a fixed structure from any batch fit and refreshes the
leaf values only -- the baseline that XGBoost's ``refresh`` updater would give
with decay added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Cfg:
    n_trees: int = 20
    max_depth: int = 4
    n_bins: int = 32
    bin_rows: int = 500  # warm-up rows buffered to place the bin edges
    halflife: float = math.inf  # clock units; inf = no decay
    eta: float = 0.3
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    grace: int = 50  # rows through a node between split / collapse evaluations
    min_periods: float = 0.0
    grow_every: int = 1
    prune: bool = True  # collapse a split whose gain has decayed below zero
    colsample: float = 1.0  # fraction of features a tree may split on (seeded, per tree)
    seed: int = 0
    hoeffding_delta: float = 0.0  # >0: require best-vs-second gain margin (see split())
    gamma_rel: float = 0.0  # gain must exceed gamma_rel * (EW variance of g in the node)
    collapse_rel: float = 0.5  # collapse when gain < collapse_rel * gamma_rel * var
    stage_rows: int = 0  # >0: stagewise -- one tree grows at a time, for this many rows
    stagger_rows: int = 0  # >0: tree m is born at learned row m * stagger_rows; all grow
    grow_trees: int = 0  # >0: only the youngest grow_trees trees keep histograms (can split)
    warm_start: bool = False  # grow the ensemble batch-style on the warm-up buffer first
    freeze_after_warm: bool = False  # warm start, then leaf refresh only (no growth, no prune)
    hist_pool: int = 0  # >0: at most this many leaves (ensemble-wide) hold histograms
    recycle: bool = False  # stagewise/stagger with a full ensemble: retire the oldest tree


class Node:
    __slots__ = (
        "feat",
        "cut",
        "left",
        "right",
        "depth",
        "G",
        "H",
        "Q",
        "stamp",
        "hist_G",
        "hist_H",
        "n_since_eval",
        "value",
    )

    def __init__(self, depth: int, stamp: float):
        self.feat = -1  # -1 = leaf
        self.cut = -1
        self.left: Node | None = None
        self.right: Node | None = None
        self.depth = depth
        self.G = 0.0
        self.H = 0.0
        self.Q = 0.0  # EW sum of c * g^2, for the gradient variance
        self.stamp = stamp
        self.hist_G: np.ndarray | None = None  # (n_sub_features, n_bins)
        self.hist_H: np.ndarray | None = None
        self.n_since_eval = 0
        self.value = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feat < 0

    def bring_up_to_date(self, L_now: float) -> None:
        f = math.exp(L_now - self.stamp)
        if f != 1.0:
            self.G *= f
            self.H *= f
            self.Q *= f
            if self.hist_G is not None:
                self.hist_G *= f
                self.hist_H *= f
        self.stamp = L_now


class Tree:
    def __init__(self, cfg: Cfg, n_features: int, sub: np.ndarray, stamp: float):
        self.cfg = cfg
        self.sub = sub  # feature indices this tree may split on
        self.root = Node(0, stamp)
        self.root.hist_G = np.zeros((len(sub), cfg.n_bins))
        self.root.hist_H = np.zeros((len(sub), cfg.n_bins))
        self.n_leaves = 1
        self.n_nodes = 1
        self.frozen = False

    # -- prediction with the frozen values ------------------------------------------
    def leaf_values(self, Xb: np.ndarray) -> np.ndarray:
        out = np.empty(len(Xb))
        self._assign(self.root, np.arange(len(Xb)), Xb, out, None, None, 0.0)
        return out

    # -- accumulate a segment's contributions ---------------------------------------
    def accumulate(self, Xb: np.ndarray, cg: np.ndarray, ch: np.ndarray, L_end: float) -> None:
        out = np.empty(len(Xb))
        self._assign(self.root, np.arange(len(Xb)), Xb, out, cg, ch, L_end)

    def _assign(self, node, idx, Xb, out, cg, ch, L_end):
        if cg is not None:
            node.bring_up_to_date(L_end)
            node.G += float(cg[idx].sum())
            node.H += float(ch[idx].sum())
            node.Q += float((cg[idx] * cg[idx] / ch[idx]).sum())
            node.n_since_eval += len(idx)
        if node.is_leaf:
            out[idx] = node.value
            if cg is not None and node.hist_G is not None:
                B = self.cfg.n_bins
                xb = Xb[idx]
                for k, j in enumerate(self.sub):
                    node.hist_G[k] += np.bincount(xb[:, j], weights=cg[idx], minlength=B)
                    node.hist_H[k] += np.bincount(xb[:, j], weights=ch[idx], minlength=B)
            return
        go_left = Xb[idx, node.feat] <= node.cut
        li, ri = idx[go_left], idx[~go_left]
        if len(li):
            self._assign(node.left, li, Xb, out, cg, ch, L_end)
        if len(ri):
            self._assign(node.right, ri, Xb, out, cg, ch, L_end)

    # -- checkpoint: values, splits, collapses --------------------------------------
    def checkpoint(self, L_now: float) -> None:
        self._visit(self.root, L_now)

    def _visit(self, node: Node, L_now: float) -> None:
        cfg = self.cfg
        if node.is_leaf:
            node.value = self._leaf_value(node)
            if (
                node.hist_G is not None
                and node.depth < cfg.max_depth
                and node.n_since_eval >= cfg.grace
            ):
                node.n_since_eval = 0
                self._try_split(node, L_now)
            return
        # internal node: maybe collapse, else recurse
        if cfg.prune and node.n_since_eval >= cfg.grace:
            node.n_since_eval = 0
            lft, rgt = node.left, node.right
            lft.bring_up_to_date(L_now)
            rgt.bring_up_to_date(L_now)
            node.bring_up_to_date(L_now)
            gain = self._gain(lft.G, lft.H, rgt.G, rgt.H)
            if gain < self.cfg.collapse_rel * self.cfg.gamma_rel * self._var_g(node):
                self._collapse(node, L_now)
                node.value = self._leaf_value(node)
                return
        self._visit(node.left, L_now)
        self._visit(node.right, L_now)

    def _leaf_value(self, node: Node) -> float:
        return -node.G / (node.H + self.cfg.reg_lambda)

    def _var_g(self, node: Node) -> float:
        """EW variance of the (unit-weight) gradient in the node."""
        if node.H <= 0.0:
            return 0.0
        m = node.G / node.H
        return max(node.Q / node.H - m * m, 0.0)

    def _gain(self, GL, HL, GR, HR) -> float:
        lam = self.cfg.reg_lambda
        return (
            0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam))
            - self.cfg.gamma
        )

    def _try_split(self, node: Node, L_now: float) -> None:
        cfg = self.cfg
        lam, mcw = cfg.reg_lambda, cfg.min_child_weight
        hG, hH = node.hist_G, node.hist_H
        GL = np.cumsum(hG, axis=1)[:, :-1]
        HL = np.cumsum(hH, axis=1)[:, :-1]
        Gt = hG.sum(axis=1, keepdims=True)
        Ht = hH.sum(axis=1, keepdims=True)
        GR, HR = Gt - GL, Ht - HL
        gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - Gt**2 / (Ht + lam)) - cfg.gamma
        gain = np.where((mcw <= HL) & (mcw <= HR), gain, -np.inf)
        k, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
        best = gain[k, b]
        if not np.isfinite(best) or best <= self.cfg.gamma_rel * self._var_g(node):
            return
        if cfg.hoeffding_delta > 0.0:
            # Domingos & Hulten style margin: the best candidate must beat the runner-up
            # by eps = R * sqrt(ln(1/delta) / (2 n)), with the gain range R taken as the
            # best gain itself (a scale-free heuristic, not the paper's fixed R).
            flat = np.sort(gain[np.isfinite(gain)].ravel())
            second = flat[-2] if len(flat) > 1 else 0.0
            n = float(Ht[k, 0])
            eps = best * math.sqrt(math.log(1.0 / cfg.hoeffding_delta) / (2.0 * max(n, 1.0)))
            if best - second < eps:
                return
        # split: the split feature's histogram partitions exactly; other features start cold
        node.feat = int(self.sub[k])
        node.cut = int(b)
        nb = cfg.n_bins
        for side, sl in (("left", slice(0, b + 1)), ("right", slice(b + 1, nb))):
            child = Node(node.depth + 1, L_now)
            child.G = float(hG[k, sl].sum())
            child.H = float(hH[k, sl].sum())
            # g^2 is not in the histogram: seed the child's variance with the parent's
            child.Q = self._var_g(node) * child.H + child.G**2 / max(child.H, 1e-300)
            if child.depth < cfg.max_depth:
                child.hist_G = np.zeros((len(self.sub), nb))
                child.hist_H = np.zeros((len(self.sub), nb))
                child.hist_G[k, sl] = hG[k, sl]
                child.hist_H[k, sl] = hH[k, sl]
            child.value = self._leaf_value(child)
            setattr(node, side, child)
        node.hist_G = node.hist_H = None
        self.n_leaves += 1
        self.n_nodes += 2

    def _collapse(self, node: Node, L_now: float) -> None:
        removed_leaves = self._count_leaves(node) - 1
        removed_nodes = self._count_nodes(node) - 1
        node.feat = -1
        node.cut = -1
        node.left = node.right = None
        if self.frozen:
            node.hist_G = node.hist_H = None
        else:
            node.hist_G = np.zeros((len(self.sub), self.cfg.n_bins))
            node.hist_H = np.zeros((len(self.sub), self.cfg.n_bins))
        self.n_leaves -= removed_leaves
        self.n_nodes -= removed_nodes

    def _count_leaves(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _count_nodes(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def nodes(self):
        stack = [self.root]
        while stack:
            nd = stack.pop()
            yield nd
            if not nd.is_leaf:
                stack.append(nd.left)
                stack.append(nd.right)

    def freeze(self) -> None:
        """Drop the histograms: the structure can no longer grow (it can still collapse)."""
        self.frozen = True
        stack = [self.root]
        while stack:
            nd = stack.pop()
            nd.hist_G = nd.hist_H = None
            if not nd.is_leaf:
                stack.append(nd.left)
                stack.append(nd.right)

    def state_doubles(self) -> int:
        """Doubles held by this tree: 4 per node (G, H, Q, stamp) + histograms on leaves."""
        n = 0
        stack = [self.root]
        while stack:
            nd = stack.pop()
            n += 4
            if nd.hist_G is not None:
                n += 2 * nd.hist_G.size
            if not nd.is_leaf:
                stack.append(nd.left)
                stack.append(nd.right)
        return n


class OnlineGBT:
    """Boosted online trees. Feed rows in order with ``step`` or chunks with ``fit_chunk``."""

    def __init__(self, cfg: Cfg, n_features: int):
        self.cfg = cfg
        self.p = n_features
        self.L = 0.0  # cumulative log-decay
        self.n_eff = 0.0
        self.n_learned = 0  # learned rows so far (checkpoint schedule)
        self.edges: list[np.ndarray] | None = None
        self.buf: list[tuple[np.ndarray, float, float, float]] = []  # warm-up rows (x, y, w, L)
        self.base_G = 0.0  # EW sums for the base score (EW mean of y)
        self.base_H = 0.0
        self.base_stamp = 0.0
        self.base = 0.0
        self.rng = np.random.default_rng(cfg.seed)
        self.trees: list[Tree] = []
        self.n_born = 0
        if cfg.stage_rows > 0 or cfg.stagger_rows > 0:
            self.trees.append(self._new_tree())
        else:
            for _ in range(cfg.n_trees):
                self.trees.append(self._new_tree())
        self._seg: list[tuple[np.ndarray, float, float, float]] = []  # pending rows of a segment

    def _new_tree(self) -> Tree:
        k = max(1, int(round(self.cfg.colsample * self.p)))
        sub = np.sort(self.rng.choice(self.p, size=k, replace=False))
        self.n_born += 1
        return Tree(self.cfg, self.p, sub, self.L)

    def _apply_hist_pool(self, L_now: float) -> None:
        """Histogram pool: only the `hist_pool` heaviest splittable leaves (by EW weight H,
        ties by tree and node order) keep histograms. Decided at checkpoints only, so the
        row stream between checkpoints does not see it (chunk invariance holds)."""
        cand = []
        for ti, t in enumerate(self.trees):
            if t.frozen:
                continue
            for ni, nd in enumerate(t.nodes()):
                if nd.is_leaf and nd.depth < self.cfg.max_depth:
                    nd.bring_up_to_date(L_now)
                    cand.append((-nd.H, ti, ni, nd, t))
        cand.sort(key=lambda c: c[:3])
        for rank, (_, _, _, nd, t) in enumerate(cand):
            if rank < self.cfg.hist_pool:
                if nd.hist_G is None:
                    nd.hist_G = np.zeros((len(t.sub), self.cfg.n_bins))
                    nd.hist_H = np.zeros((len(t.sub), self.cfg.n_bins))
                    nd.n_since_eval = 0
            else:
                nd.hist_G = nd.hist_H = None

    def _stage_boundary(self) -> None:
        """Stagewise: freeze the growing tree and start the next one."""
        self.trees[-1].freeze()
        if len(self.trees) < self.cfg.n_trees:
            self.trees.append(self._new_tree())
        elif self.cfg.recycle:
            self.trees.pop(0)
            self.trees.append(self._new_tree())
        else:
            self.trees[-1].frozen = False  # keep the last tree growing forever
            self._regrow_hist(self.trees[-1])

    def _regrow_hist(self, t: Tree) -> None:
        stack = [t.root]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                if nd.depth < self.cfg.max_depth:
                    nd.hist_G = np.zeros((len(t.sub), self.cfg.n_bins))
                    nd.hist_H = np.zeros((len(t.sub), self.cfg.n_bins))
            else:
                stack.append(nd.left)
                stack.append(nd.right)

    # ---------------------------------------------------------------------------
    def _bin(self, X: np.ndarray) -> np.ndarray:
        Xb = np.empty(X.shape, dtype=np.int64)
        for j in range(self.p):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return Xb

    def _predict_frozen(self, Xb: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return (pred, partial sums F_{m-1} for m = 1..M) with the frozen ensemble."""
        F = np.full(len(Xb), self.base)
        partials = []
        for t in self.trees:
            partials.append(F.copy())
            F = F + self.cfg.eta * t.leaf_values(Xb)
        return F, partials

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.edges is None:
            return np.full(len(X), np.nan)
        return self._predict_frozen(self._bin(X))[0]

    # ---------------------------------------------------------------------------
    def fit_chunk(self, X, y, d_clock, w=None):
        """Predict-then-learn each row of the chunk; returns (pred, n_eff) per row."""
        n = len(X)
        w = np.ones(n) if w is None else np.asarray(w, dtype=float)
        pred = np.full(n, np.nan)
        n_eff = np.empty(n)
        for i in range(n):
            pred[i], n_eff[i] = self.step(X[i], y[i], d_clock[i], w[i])
        return pred, n_eff

    def step(self, x, y, d_clock, w=1.0):
        cfg = self.cfg
        lam = 1.0 if math.isinf(cfg.halflife) else 0.5 ** (d_clock / cfg.halflife)
        self.L += math.log(lam)
        n_eff_before = self.n_eff
        self.n_eff = lam * self.n_eff + w
        learn = (y is not None) and not (isinstance(y, float) and math.isnan(y)) and w > 0.0
        # prediction, before the update, with the frozen ensemble
        if self.edges is None or n_eff_before < cfg.min_periods:
            pred = math.nan
        else:
            pred = float(self._predict_frozen(self._bin(np.asarray(x)[None, :]))[0][0])
        if not learn:
            return pred, n_eff_before
        row = (np.asarray(x, dtype=float), float(y), float(w), self.L)
        if self.edges is None:
            self.buf.append(row)
            if len(self.buf) >= cfg.bin_rows:
                self._place_edges()
                if cfg.warm_start:
                    self.batch_warm_start(
                        np.stack([r[0] for r in self.buf]),
                        np.array([r[1] for r in self.buf]),
                        np.array([r[2] for r in self.buf]),
                        np.array([r[3] for r in self.buf]),
                    )
                else:
                    for r in self.buf:
                        self._push(r)
                self.buf = []
            return pred, n_eff_before
        self._push(row)
        return pred, n_eff_before

    def batch_warm_start(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        Lrow: np.ndarray,
        depth: int | None = None,
    ) -> None:
        """Grow the ensemble batch-style on the buffer (level-wise histogram GBDT with the
        same gain and leaf formulas), then replay the buffer through the online path so the
        sums and histograms are what a stream would have left. Called once, at warm-up."""
        cfg = self.cfg
        depth = cfg.max_depth if depth is None else depth
        Xb = self._bin(X)
        c = w * np.exp(self.L - Lrow)
        F = np.full(len(y), float((c * y).sum() / c.sum()))
        self.trees = []
        self.n_born = 0
        for _ in range(cfg.n_trees):
            t = self._new_tree()
            g = c * (F - y)
            h = c.copy()
            self._grow_batch(t, Xb, g, h, depth)
            self.trees.append(t)
            F = F + cfg.eta * t.leaf_values(Xb)
        self.base_G, self.base_H = float((c * y).sum()), float(c.sum())
        self.base_stamp = self.L
        self.base = self.base_G / self.base_H
        self.n_learned += len(y)
        if cfg.freeze_after_warm:
            for t in self.trees:
                t.freeze()

    def _grow_batch(self, t: Tree, Xb, g, h, depth) -> None:
        cfg = self.cfg
        lam, mcw = cfg.reg_lambda, cfg.min_child_weight
        frontier = [(t.root, np.arange(len(g)))]
        t.root.G, t.root.H = float(g.sum()), float(h.sum())
        t.root.Q = float((g * g / h).sum())
        t.root.value = -t.root.G / (t.root.H + lam)
        for _ in range(depth):
            nxt = []
            for node, idx in frontier:
                if len(idx) < 2:
                    continue
                xb = Xb[idx]
                hG = np.stack(
                    [np.bincount(xb[:, j], weights=g[idx], minlength=cfg.n_bins) for j in t.sub]
                )
                hH = np.stack(
                    [np.bincount(xb[:, j], weights=h[idx], minlength=cfg.n_bins) for j in t.sub]
                )
                node.hist_G, node.hist_H = hG, hH  # kept if the node stays a leaf
                GL = np.cumsum(hG, axis=1)[:, :-1]
                HL = np.cumsum(hH, axis=1)[:, :-1]
                Gt, Ht = hG.sum(1, keepdims=True), hH.sum(1, keepdims=True)
                GR, HR = Gt - GL, Ht - HL
                gain = (
                    0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - Gt**2 / (Ht + lam)) - cfg.gamma
                )
                gain = np.where((mcw <= HL) & (mcw <= HR), gain, -np.inf)
                k, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
                if not np.isfinite(gain[k, b]) or gain[k, b] <= 0.0:
                    continue
                node.feat, node.cut = int(t.sub[k]), int(b)
                node.hist_G = node.hist_H = None
                go_left = xb[:, node.feat] <= node.cut
                for side, sub_idx in (("left", idx[go_left]), ("right", idx[~go_left])):
                    child = Node(node.depth + 1, self.L)
                    child.G, child.H = float(g[sub_idx].sum()), float(h[sub_idx].sum())
                    child.Q = float((g[sub_idx] ** 2 / h[sub_idx]).sum())
                    child.value = -child.G / (child.H + lam)
                    if child.depth < cfg.max_depth:
                        xs = Xb[sub_idx]
                        child.hist_G = np.stack(
                            [
                                np.bincount(xs[:, j], weights=g[sub_idx], minlength=cfg.n_bins)
                                for j in t.sub
                            ]
                        )
                        child.hist_H = np.stack(
                            [
                                np.bincount(xs[:, j], weights=h[sub_idx], minlength=cfg.n_bins)
                                for j in t.sub
                            ]
                        )
                    setattr(node, side, child)
                    nxt.append((child, sub_idx))
                t.n_leaves += 1
                t.n_nodes += 2
            frontier = nxt

    def _place_edges(self) -> None:
        X = np.stack([r[0] for r in self.buf])
        qs = np.arange(1, self.cfg.n_bins) / self.cfg.n_bins
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(self.p)]

    def _push(self, row) -> None:
        self._seg.append(row)
        self.n_learned += 1
        if self.n_learned % self.cfg.grow_every == 0:
            self._flush_segment()

    def _flush_segment(self) -> None:
        if not self._seg:
            return
        X = np.stack([r[0] for r in self._seg])
        y = np.array([r[1] for r in self._seg])
        w = np.array([r[2] for r in self._seg])
        Lrow = np.array([r[3] for r in self._seg])
        self._seg = []
        L_end = self.L
        c = w * np.exp(L_end - Lrow)  # each row's weight as seen from the segment end
        Xb = self._bin(X)
        # gradients from the frozen ensemble (squared loss: g = F - y, h = 1)
        _, partials = self._predict_frozen(Xb)
        for t, Fm in zip(self.trees, partials, strict=True):
            g = Fm - y
            t.accumulate(Xb, c * g, c, L_end)
        # base score: EW mean of y
        f = math.exp(L_end - self.base_stamp)
        self.base_G = f * self.base_G + float((c * y).sum())
        self.base_H = f * self.base_H + float(c.sum())
        self.base_stamp = L_end
        # checkpoint: values, splits, collapses
        self.base = self.base_G / self.base_H if self.base_H > 0 else 0.0
        for t in self.trees:
            t.checkpoint(L_end)
        if self.cfg.hist_pool > 0:
            self._apply_hist_pool(L_end)
        if self.cfg.stage_rows > 0 and self.n_learned % self.cfg.stage_rows == 0:
            self._stage_boundary()
        if self.cfg.stagger_rows > 0 and self.n_learned % self.cfg.stagger_rows == 0:
            if len(self.trees) < self.cfg.n_trees:
                self.trees.append(self._new_tree())
            elif self.cfg.recycle:
                self.trees.pop(0)
                self.trees.append(self._new_tree())
            if self.cfg.grow_trees > 0:
                for t in self.trees[: -self.cfg.grow_trees]:
                    if not t.frozen:
                        t.freeze()

    # ---------------------------------------------------------------------------
    def n_leaves(self) -> int:
        return sum(t.n_leaves for t in self.trees)

    def state_doubles(self) -> int:
        return sum(t.state_doubles() for t in self.trees) + sum(len(e) for e in self.edges or [])


# ---------------------------------------------------------------------------------
# Level 1: fixed structure from a batch fit, leaves refreshed online.
# ---------------------------------------------------------------------------------
class LeafRefresh:
    """Trees from an xgboost booster; per-leaf EW (G, H) refreshed online.

    ``leaf_ids(X)`` must return an int array (n, M) of leaf indices per tree. The
    online part is identical to OnlineGBT with growth disabled.
    """

    def __init__(
        self,
        leaf_ids,
        n_trees: int,
        eta: float,
        reg_lambda: float,
        halflife: float,
        grow_every: int = 1,
        base: float = 0.0,
    ):
        self.leaf_ids = leaf_ids
        self.M = n_trees
        self.eta, self.lam, self.halflife = eta, reg_lambda, halflife
        self.grow_every = grow_every
        self.L = 0.0
        self.n_eff = 0.0
        self.n_learned = 0
        self.base = base
        # per tree: dict leaf -> [G, H, stamp, value]
        self.leaves: list[dict[int, list[float]]] = [dict() for _ in range(n_trees)]
        self._seg: list = []

    def _values(self, ids: np.ndarray) -> np.ndarray:
        """(n, M) frozen leaf values."""
        out = np.zeros(ids.shape)
        for m in range(self.M):
            d = self.leaves[m]
            out[:, m] = [d[i][3] if i in d else 0.0 for i in ids[:, m]]
        return out

    def step(self, x, y, d_clock, w=1.0):
        lam = 1.0 if math.isinf(self.halflife) else 0.5 ** (d_clock / self.halflife)
        self.L += math.log(lam)
        n_eff_before = self.n_eff
        self.n_eff = lam * self.n_eff + w
        ids = self.leaf_ids(np.asarray(x, dtype=float)[None, :])
        vals = self._values(ids)[0]
        pred = self.base + self.eta * float(vals.sum())
        learn = (y is not None) and w > 0.0
        if not learn:
            return pred, n_eff_before
        self._seg.append((ids[0], float(y), float(w), self.L, vals))
        self.n_learned += 1
        if self.n_learned % self.grow_every == 0:
            self._flush()
        return pred, n_eff_before

    def _flush(self):
        L_end = self.L
        for ids, y, w, Lrow, vals in self._seg:
            c = w * math.exp(L_end - Lrow)
            F = self.base
            for m in range(self.M):
                g = F - y
                d = self.leaves[m]
                ent = d.get(int(ids[m]))
                if ent is None:
                    ent = [0.0, 0.0, L_end, 0.0]
                    d[int(ids[m])] = ent
                f = math.exp(L_end - ent[2])
                ent[0] = f * ent[0] + c * g
                ent[1] = f * ent[1] + c
                ent[2] = L_end
                F += self.eta * vals[m]
        self._seg = []
        for d in self.leaves:
            for ent in d.values():
                ent[3] = -ent[0] / (ent[1] + self.lam)


# ---------------------------------------------------------------------------------
# Data: Friedman-1 with optional drift, irregular clock, row weights.
# ---------------------------------------------------------------------------------
def friedman(n: int, p: int = 10, seed: int = 0, drift: str = "none", noise: float = 1.0):
    """y = a(t)*10 sin(pi x1 x2) + b(t)*20 (x3-0.5)^2 + c(t)*10 x4 + d(t)*5 x5 + eps.

    drift: 'none' (constant), 'walk' (coefficients random-walk, sigma 0.02/step scaled),
    'abrupt' (at n/2 the roles of x4 and x5 swap and the quadratic flips sign).
    Returns X, y, f (noise-free), d_clock, w.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    t = np.arange(n)
    a = b = c = d = np.ones(n)
    if drift == "walk":
        walk = np.cumsum(rng.normal(0.0, 1.0 / math.sqrt(n), size=(n, 4)), axis=0)
        a, b, c, d = (1.0 + 0.8 * walk[:, i] for i in range(4))
    f = a * 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + b * 20 * (X[:, 2] - 0.5) ** 2
    if drift == "abrupt":
        h = n // 2
        f = f + np.where(t < h, 10 * X[:, 3] + 5 * X[:, 4], 5 * X[:, 3] + 10 * X[:, 4])
        f = f - np.where(t < h, 0.0, 40 * (X[:, 2] - 0.5) ** 2)
    else:
        f = f + c * 10 * X[:, 3] + d * 5 * X[:, 4]
    y = f + rng.normal(0.0, noise, size=n)
    d_clock = rng.exponential(1.0, size=n)
    w = np.ones(n)
    return X, y, f, d_clock, w


def mse(pred, y, start):
    m = np.isfinite(pred[start:])
    return float(np.mean((pred[start:][m] - y[start:][m]) ** 2))
