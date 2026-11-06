//! The exact weighted median of the residuals a refit holds, read in
//! `O(log n)` a row (`huber`, `quantile`, `sgd(loss = "huber")` and the
//! marginal bins' scales while their refit waits; `huber_refit`'s
//! `RefitBuf::warm_scale`, `margbins`' `HeldDevs`).
//!
//! **The weights.** Row `i`, of raw weight `w_i`, counts at `v_i = w_i ·
//! D_now / D_i` in the median, `D_i` the product of every decay the scale
//! took up to and including row `i`'s own. Every decay after row `i`
//! multiplies all the rows held alike, so a row's weight relative to the
//! others is fixed once it is held: `r_i = w_i / D_i`, the common `D_now`
//! moving no median. So each row is inserted once, at `r_i`, into a tree
//! ordered by `|e|`, and the median is a walk down it.
//!
//! **The decay, in parts.** `D` falls without bound -- a half-life of one
//! row halves it every row -- so it is kept as `m · 2^k`, `m` in `[1/2,
//! 1)`: a decay `λ = m_λ · 2^{k_λ}` multiplies the mantissas, one rounding,
//! and adds the exponents, and the product is brought back into `[1/2, 1)`
//! by an exact power of two (`frexp`). A run of `c` equal decays is `λ^c`
//! by squaring in the same parts. A row is held at `r_i = (w_i / m_i) ·
//! 2^{K − k_i}`, `K` a reference exponent; once `K − k` reaches 512 every
//! weight held is multiplied by `2^{−512}` and `K` lowered by 512, exactly
//! (a weight that falls below the smallest normal float is rounded there,
//! and one past `2^{−1074}` is 0: a row that far below the rest weighs
//! nothing, and a weight of 0 counts nowhere).
//!
//! **The read**, by definition: over the held `(|e_i|, r_i)` with `r_i >
//! 0`, in ascending `|e|` (rows of equal `|e|` in the order they came), the
//! first `|e|` at which the running sum of `r` reaches half the sum of all
//! of them, both sums taken in that order -- `huber_refit::weighted_median_sorted`
//! of the held values at those weights. The tree finds the row where its
//! own sums cross half, and that is the definition's row whenever the
//! crossing is clear of both sums' rounding: the sum below it under half by
//! more than `4 (n + 256) u` of the total and the sum through it over half
//! by as much (the bound `weighted_median`'s selection takes), each sum of
//! `n` positive terms being within `(n − 1) u` of its own total whatever
//! the order of its additions. Where it is not, the read walks the tree in
//! order and takes the definition itself, `O(n)`: a crossing within
//! `10^{−12}` of half on a thousand rows of generic weights. While every
//! weight is a multiple of `2^{−20}` and their sum under `2^{33}` -- no decay
//! and dyadic raw weights, the plain stream's -- every sum is exact in any
//! order, and the tree's row is the definition's with no margin.
//!
//! The tree is not state: a restored buffer rebuilds it from the rows it
//! holds, replaying their decays in order, and reaches the same weights to
//! the bit, so the read is the same as had the stream not stopped.

#[cfg(test)]
thread_local! {
    /// The reads that took the definition's walk ([`Tree::by_definition`]).
    pub(crate) static FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// No child: an index past every node.
const NIL: u32 = u32::MAX;

/// The power of two every held weight is rescaled by when the reference
/// exponent falls this far behind the decay's.
const RESCALE_BITS: i64 = 512;

/// `2^{-512}`, exactly.
const RESCALE: f64 = 7.458_340_731_200_207e-155;

/// A positive normal or subnormal `x` as `(m, e)`, `x = m · 2^e`, `m` in
/// `[1/2, 1)`: exact (`frexp`). `x` is finite and above 0.
pub(crate) fn frexp(x: f64) -> (f64, i64) {
    debug_assert!(x > 0.0 && x.is_finite());
    let (x, bias) = if x < f64::MIN_POSITIVE {
        (x * 18_446_744_073_709_551_616.0, -64) // 2^64
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64;
    // The mantissa with the exponent of 2^-1, so in [1/2, 1).
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022u64 << 52));
    (m, exp - 1022 + bias)
}

/// A product kept as `m · 2^k`, `m` in `[1/2, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Split {
    m: f64,
    k: i64,
}

impl Split {
    const ONE: Split = Split { m: 0.5, k: 1 };

    /// `self · other`: the mantissas' product, one rounding, brought back
    /// into `[1/2, 1)` exactly.
    fn times(self, other: Split) -> Split {
        let (m, e) = frexp(self.m * other.m);
        Split {
            m,
            k: self.k + other.k + e,
        }
    }

    fn of(x: f64) -> Split {
        let (m, k) = frexp(x);
        Split { m, k }
    }

    /// `x^n` by squaring, each product in parts: no libm, no overflow and
    /// no underflow, so a run of a million decays of `1/2` is `2^{−10^6}`
    /// exactly.
    fn pow(x: f64, mut n: u64) -> Split {
        let (mut out, mut base) = (Split::ONE, Split::of(x));
        while n > 0 {
            if n & 1 == 1 {
                out = out.times(base);
            }
            base = base.times(base);
            n >>= 1;
        }
        out
    }
}

/// One held row in a tree: `|e|`, the row's place in the order rows came
/// (its heap priority is a hash of it, [`priority`]), its weight, the
/// weight of its subtree, and its children: 40 bytes a row.
#[derive(Debug, Clone)]
struct Node {
    e: f64,
    seq: u32,
    w: f64,
    sum: f64,
    l: u32,
    r: u32,
}

/// A treap of one target's held rows, ordered by `(|e|, seq)`, each node
/// keeping its subtree's weight.
#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
    root: u32,
    /// Every weight a multiple of `2^{−20}` and their sum under `2^{33}`:
    /// every sum exact.
    exact: bool,
}

/// SplitMix64: a node's priority from its place, the same on every build.
fn priority(seq: u32) -> u64 {
    let mut z = u64::from(seq).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A weight every sum of which is exact: a multiple of `2^{−20}` under
/// `2^{33}`.
fn dyadic(w: f64) -> bool {
    let s = w * 1_048_576.0; // 2^20
    s.fract() == 0.0 && w < 8_589_934_592.0 // 2^33
}

impl Tree {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: NIL,
            exact: true,
        }
    }

    fn sum(&self, at: u32) -> f64 {
        if at == NIL {
            0.0
        } else {
            self.nodes[at as usize].sum
        }
    }

    fn pull(&mut self, at: u32) {
        let n = &self.nodes[at as usize];
        let (l, r, w) = (n.l, n.r, n.w);
        self.nodes[at as usize].sum = w + self.sum(l) + self.sum(r);
    }

    fn before(&self, a: u32, e: f64, seq: u32) -> bool {
        let n = &self.nodes[a as usize];
        e.total_cmp(&n.e).then(seq.cmp(&n.seq)).is_lt()
    }

    /// Node `new` into the subtree at `at`; the subtree's new root.
    fn put(&mut self, at: u32, new: u32) -> u32 {
        if at == NIL {
            return new;
        }
        let (e, seq) = (self.nodes[new as usize].e, self.nodes[new as usize].seq);
        if self.before(at, e, seq) {
            let l = self.put(self.nodes[at as usize].l, new);
            self.nodes[at as usize].l = l;
            if priority(self.nodes[l as usize].seq) > priority(self.nodes[at as usize].seq) {
                // Rotate right.
                self.nodes[at as usize].l = self.nodes[l as usize].r;
                self.nodes[l as usize].r = at;
                self.pull(at);
                self.pull(l);
                return l;
            }
        } else {
            let r = self.put(self.nodes[at as usize].r, new);
            self.nodes[at as usize].r = r;
            if priority(self.nodes[r as usize].seq) > priority(self.nodes[at as usize].seq) {
                // Rotate left.
                self.nodes[at as usize].r = self.nodes[r as usize].l;
                self.nodes[r as usize].l = at;
                self.pull(at);
                self.pull(r);
                return r;
            }
        }
        self.pull(at);
        at
    }

    fn insert(&mut self, e: f64, w: f64, seq: u32) {
        let at = self.nodes.len() as u32;
        self.nodes.push(Node {
            e,
            seq,
            w,
            sum: w,
            l: NIL,
            r: NIL,
        });
        self.root = self.put(self.root, at);
        self.exact &= dyadic(w) && dyadic(self.sum(self.root));
    }

    /// Every weight times `2^{−512}`.
    fn rescale(&mut self) {
        for n in &mut self.nodes {
            n.w *= RESCALE;
            n.sum *= RESCALE;
        }
        self.exact = self.nodes.iter().all(|n| n.w == 0.0);
    }

    /// The median by definition ([`HeldMedian`]'s module doc): the in-order
    /// walk's running sum, `O(n)`.
    fn by_definition(&self) -> Option<f64> {
        #[cfg(test)]
        FALLBACKS.with(|c| c.set(c.get() + 1));
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack = Vec::new();
        let mut at = self.root;
        while at != NIL || !stack.is_empty() {
            while at != NIL {
                stack.push(at);
                at = self.nodes[at as usize].l;
            }
            let x = stack.pop().expect("a node");
            order.push(x);
            at = self.nodes[x as usize].r;
        }
        let kept = || {
            order
                .iter()
                .map(|&i| &self.nodes[i as usize])
                .filter(|n| n.w > 0.0)
        };
        let total: f64 = kept().map(|n| n.w).sum();
        let mut cum = 0.0;
        for n in kept() {
            cum += n.w;
            if cum >= 0.5 * total {
                return Some(n.e);
            }
        }
        None
    }

    /// The median: the tree's row where its sums cross half, where that is
    /// clear of their rounding, and the definition's otherwise.
    fn median(&self) -> Option<f64> {
        let total = self.sum(self.root);
        if total.is_nan() || total <= 0.0 {
            return None;
        }
        if !total.is_finite() {
            return self.by_definition();
        }
        let half = 0.5 * total;
        let (mut at, mut below) = (self.root, 0.0);
        while at != NIL {
            let n = &self.nodes[at as usize];
            let left = below + self.sum(n.l);
            if left >= half && n.l != NIL {
                at = n.l;
            } else if left + n.w >= half {
                let margin = if self.exact {
                    0.0
                } else {
                    4.0 * (self.nodes.len() + 256) as f64 * f64::EPSILON * total
                };
                let clear = if self.exact {
                    left < half
                } else {
                    left < half - margin && left + n.w >= half + margin
                };
                return if clear {
                    Some(n.e)
                } else {
                    self.by_definition()
                };
            } else {
                below = left + n.w;
                at = n.r;
            }
        }
        self.by_definition()
    }
}

/// The held rows of one or more targets that share a clock: the decay in
/// parts, the reference exponent, and a tree per target (the module doc).
#[derive(Debug, Clone)]
pub(crate) struct HeldMedian {
    d: Split,
    /// The decay where the current run of equal decays began.
    run_base: Split,
    /// The reference exponent `K`, set by the first row held.
    reference: Option<i64>,
    trees: Vec<Tree>,
    seq: u32,
}

/// No tree yet: each target's is made by its first row
/// ([`HeldMedian::insert`]), and an absent tree reads as an empty one, so
/// the default holds no rows rather than refusing them.
impl Default for HeldMedian {
    fn default() -> Self {
        Self::new(0)
    }
}

impl PartialEq for HeldMedian {
    /// Not state: built from the rows held, which the owner compares.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl HeldMedian {
    pub(crate) fn new(targets: usize) -> Self {
        Self {
            d: Split::ONE,
            run_base: Split::ONE,
            reference: None,
            trees: vec![Tree::new(); targets],
            seq: 0,
        }
    }

    /// Every row let go, as [`Self::new`] of `targets` would hold them, each
    /// tree keeping the room it had: what a hold whose weight went to 0
    /// starts again from, allocating nothing.
    pub(crate) fn clear(&mut self, targets: usize) {
        self.d = Split::ONE;
        self.run_base = Split::ONE;
        self.reference = None;
        self.seq = 0;
        self.trees.resize_with(targets, Tree::new);
        for t in &mut self.trees {
            t.nodes.clear();
            t.root = NIL;
            t.exact = true;
        }
    }

    /// A row's decay `λ` in `[0, 1]`. A decay of 0 leaves every row held
    /// weighing nothing, and the next row starts the scale afresh.
    pub(crate) fn decay(&mut self, lam: f64) {
        if lam > 0.0 {
            self.d = self.d.times(Split::of(lam));
        } else {
            for t in &mut self.trees {
                for n in &mut t.nodes {
                    n.w = 0.0;
                    n.sum = 0.0;
                }
                t.exact = true;
            }
            self.d = Split::ONE;
            self.reference = None;
        }
        self.run_base = self.d;
    }

    /// A run of decays, `λ` `count` times since [`Self::begin_run`]: the
    /// decay where the run began times `λ^count`, so a run lengthened row
    /// by row and one replayed whole reach the same decay.
    pub(crate) fn run(&mut self, lam: f64, count: u64) {
        self.d = self.run_base.times(Split::pow(lam, count));
    }

    /// A run of equal decays begins here ([`Self::run`]).
    pub(crate) fn begin_run(&mut self) {
        self.run_base = self.d;
    }

    /// Target `t`'s row `|e|` at raw weight `w`, after its decay: held at
    /// `w / D` in the reference's units. Anything but a finite `e` at
    /// least 0 at a weight above 0 is not held.
    pub(crate) fn insert(&mut self, t: usize, e: f64, w: f64) {
        if !(e >= 0.0 && e.is_finite() && w > 0.0 && w.is_finite()) {
            return;
        }
        let k = self.d.k;
        let reference = *self.reference.get_or_insert(k);
        // Lowered by 512 at a time, every weight with it; past five such
        // steps every finite weight is 0, so the rest move the reference
        // alone.
        let steps = (reference - k).div_euclid(RESCALE_BITS);
        for _ in 0..steps.min(5) {
            self.trees.iter_mut().for_each(Tree::rescale);
        }
        self.reference = Some(reference - steps * RESCALE_BITS);
        let reference = reference - steps * RESCALE_BITS;
        let r = w / self.d.m;
        // `2^{K − k}` in [1, 2^511]: exact.
        let up = f64::from_bits(((reference - k + 1023) as u64) << 52);
        let seq = self.seq;
        self.seq += 1;
        // A target's tree is made by its first row: an empty tree takes no
        // part in a decay of 0 or a rescale, so one made here is the one
        // that would have stood empty until now.
        if t >= self.trees.len() {
            self.trees.resize_with(t + 1, Tree::new);
        }
        self.trees[t].insert(e, r * up, seq);
    }

    /// Room for `rows` rows in all of each of `targets` targets' trees,
    /// allocated once where a tree has less: a refit buffer knows its count
    /// from its first row, and a restored one from its configuration.
    pub(crate) fn reserve_to(&mut self, targets: usize, rows: usize) {
        if self.trees.len() < targets {
            self.trees.resize_with(targets, Tree::new);
        }
        for t in &mut self.trees {
            t.nodes.reserve_exact(rows.saturating_sub(t.nodes.len()));
        }
    }

    /// Target `t`'s median `|e|`, or `None` while no row of positive
    /// weight is held.
    pub(crate) fn median(&self, t: usize) -> Option<f64> {
        self.trees.get(t).and_then(Tree::median)
    }

    /// The bytes the trees' nodes take, as each vector reports its
    /// allocation: what a reservation of the held rows must count.
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.trees
            .iter()
            .map(|t| t.nodes.capacity() * std::mem::size_of::<Node>())
            .sum()
    }

    /// The weights held for target `t`, in the order the rows came: what
    /// a test holds the reads to.
    #[cfg(test)]
    pub(crate) fn weights(&self, t: usize) -> Vec<(f64, f64)> {
        let mut v: Vec<&Node> = self.trees[t].nodes.iter().collect();
        v.sort_by_key(|n| n.seq);
        v.into_iter().map(|n| (n.e, n.w)).collect()
    }
}

#[cfg(test)]
#[path = "held_median_tests.rs"]
mod tests;
