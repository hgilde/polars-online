//! The centred moments of two adjacent sliding windows of rows, merged from
//! summaries of fixed blocks: `corrchange(kind = "window")`'s correlations
//! in `O(b·k²)` a row whatever the windows' length, where reading both
//! windows afresh is `O(window_rows · k²)`.
//!
//! # A summary
//!
//! A summary of `n` rows of `k` columns keeps `n`, an anchor `α` (one of
//! its rows), the offset of the mean from it, `o = m − α`, and the centred
//! co-moments `M = Σ (x − m)(x − m)ᵀ`, lower triangle row by row. Two
//! summaries of disjoint rows, `A` before `B`, merge by Chan, Golub &
//! LeVeque's rule (1979, "Updating formulae and a pairwise algorithm for
//! computing sample variances"):
//!
//! ```text
//! δ = m_B − m_A = (α_B − α_A) + (o_B − o_A)
//! M = M_A + M_B + (n_A n_B / n) δδᵀ
//! o = o_A + (n_B / n) δ          against α_A, or
//! o = o_B − (n_A / n) δ          against α_B, the larger part's anchor
//! ```
//!
//! The rule is associative in exact arithmetic and needs no inverse: a row
//! leaving a window is never subtracted, so nothing drifts and nothing
//! needs repair. The mean is kept as an anchor and an offset so that its
//! rounding is at the scale of the rows' spread, not of their level: at a
//! level of 1e8 a double's spacing is 1.5e-8, and a mean held whole would
//! carry that much error into every `δ`. Against an anchor, `δ` is a
//! difference of two rows plus a difference of two offsets, each rounded
//! relative to the spread.
//!
//! # The windows
//!
//! Rows are numbered from 0 since the ring last started. Block `j` is rows
//! `[j·b, (j+1)·b)`; a window's full blocks are summarised once and merged,
//! and its partial blocks at either end are summarised from the rows
//! themselves, at most `2b` rows a window. The full blocks are aggregated
//! by van Herk and Gil & Werman's scheme (1992, 1993): block indices are cut
//! into chunks of `C` blocks, and each chunk keeps the suffix sums of its
//! blocks (each block merged with the suffix after it) and, as the window's
//! right end passes through it, the running prefix sum from its first
//! block. A run of between `C − 1` and `C + 1` blocks meets at most two
//! chunks, and inside one it starts at the chunk's start or ends at its
//! end, so it is one suffix, one prefix, or a suffix and a prefix. With
//! `G = ⌊w/b⌋`, the windows' runs are `G − 2` to `G` blocks long, so `C =
//! G − 1`.
//!
//! A window's summary is `((head ⊕ suffix) ⊕ prefix) ⊕ tail`, each part
//! present where the window has it, and every part's rows and every merge
//! a function of the rows' numbers alone: the same rows give the same bits
//! however the stream was fed, and a state that keeps the ring and the
//! count rebuilds the same sums on loading.

use std::fmt;

/// The fewest full blocks, `G = ⌊w/b⌋`, a window must hold for its sums to
/// be merged; a shorter window is read whole, which costs no more there.
/// Measured 2026-11-05 on arm64 (release, the least of five passes of
/// 20,000 rows, a report a row): at `G = 3` reading whole took 248 ns a row
/// against the merged form's 275 at two columns and 366 against 382 at
/// four, and the merged form was ahead at eight (654 against 914); at `G =
/// 4` the two were 253 and 259 at two columns and the merged form ahead
/// from four (393 against 447), and from there it holds still while
/// reading whole grows with the window (`G = 8`: 253 against 386 at two
/// columns, 2,524 against 11,701 at twenty).
pub(crate) const MIN_BLOCKS: usize = 4;

/// Rows a block summary covers at `k` columns: `⌈k/2⌉ + 3`, the fewest at
/// which a summary (`k²/2 + 5k/2 + 1` numbers) takes no more memory than
/// its block's rows in the ring (`b·k`).
pub(crate) fn block_rows(k: usize) -> usize {
    k.div_ceil(2) + 3
}

/// One summary's layout over `k` columns: `[n, α (k), o (k), M (k(k+1)/2)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    k: usize,
}

impl Layout {
    pub(crate) fn new(k: usize) -> Self {
        Self { k }
    }

    /// Numbers a summary takes.
    pub(crate) fn stride(self) -> usize {
        1 + 2 * self.k + self.k * (self.k + 1) / 2
    }

    /// Where the co-moments start.
    fn at_m(self) -> usize {
        1 + 2 * self.k
    }

    /// The summary of rows `lo..hi`, read by `row`, into `out`: two passes
    /// about the first row, the offsets' sums and then the deviations'
    /// products, `dev` holding one row's deviations. No rows, `n = 0`.
    pub(crate) fn leaf<'a>(
        self,
        out: &mut [f64],
        dev: &mut [f64],
        lo: u64,
        hi: u64,
        row: &impl Fn(u64) -> &'a [f64],
    ) {
        let k = self.k;
        out.fill(0.0);
        if hi <= lo {
            return;
        }
        let n = (hi - lo) as f64;
        let (count, rest) = out.split_at_mut(1);
        let (anchor, rest) = rest.split_at_mut(k);
        let (off, m) = rest.split_at_mut(k);
        count[0] = n;
        anchor.copy_from_slice(&row(lo)[..k]);
        for q in lo..hi {
            for ((o, x), a) in off.iter_mut().zip(row(q)).zip(&*anchor) {
                *o += x - a;
            }
        }
        off.iter_mut().for_each(|o| *o /= n);
        for q in lo..hi {
            for (((d, x), a), o) in dev.iter_mut().zip(row(q)).zip(&*anchor).zip(&*off) {
                *d = (x - a) - o;
            }
            let mut p = 0;
            for a in 0..k {
                let da = dev[a];
                for (s, db) in m[p..=p + a].iter_mut().zip(&dev[..=a]) {
                    *s += da * db;
                }
                p += a + 1;
            }
        }
    }

    /// `acc ← acc ⊕ other`, `acc`'s rows before `other`'s, by the rule in
    /// the module doc, `delta` holding `δ` and then `c·δ`. The merged mean
    /// is kept against the larger part's anchor (`acc`'s on a tie), so its
    /// offset moves by the smaller part's share, at most half of `δ`; and
    /// the larger part's co-moments are added last, so one rounding a
    /// merge is at the scale of the merged rows.
    pub(crate) fn merge(self, acc: &mut [f64], other: &[f64], delta: &mut [f64]) {
        let k = self.k;
        let (na, nb) = (acc[0], other[0]);
        if nb == 0.0 {
            return;
        }
        if na == 0.0 {
            acc.copy_from_slice(other);
            return;
        }
        let n = na + nb;
        for (i, d) in delta.iter_mut().enumerate().take(k) {
            *d = (other[1 + i] - acc[1 + i]) + (other[1 + k + i] - acc[1 + k + i]);
        }
        let c = na * nb / n;
        let at = self.at_m();
        if nb > na {
            let f = na / n;
            for (i, d) in delta.iter().enumerate().take(k) {
                acc[1 + i] = other[1 + i];
                acc[1 + k + i] = other[1 + k + i] - f * d;
            }
        } else {
            let f = nb / n;
            for (i, d) in delta.iter().enumerate().take(k) {
                acc[1 + k + i] += f * d;
            }
        }
        // `c·δ_a` once per column, then `(c·δ_a)·δ_b` per entry.
        let (dv, cd) = delta.split_at_mut(k);
        for (x, d) in cd.iter_mut().zip(&*dv) {
            *x = c * d;
        }
        let (m, theirs) = (&mut acc[at..], &other[at..]);
        let mut p = 0;
        for a in 0..k {
            let ca = cd[a];
            let row = m[p..=p + a]
                .iter_mut()
                .zip(&theirs[p..=p + a])
                .zip(&dv[..=a]);
            if nb > na {
                for ((s, t), db) in row {
                    *s = t + (*s + ca * db);
                }
            } else {
                for ((s, t), db) in row {
                    *s += t + ca * db;
                }
            }
            p += a + 1;
        }
        acc[0] = n;
    }

    /// The correlation of columns `a > b` from a summary: `M_ab / √(M_aa
    /// M_bb)`, NaN where either column is constant over its rows.
    pub(crate) fn corr(self, s: &[f64], a: usize, b: usize) -> f64 {
        let m = &s[self.at_m()..];
        let (aa, bb) = (m[a * (a + 1) / 2 + a], m[b * (b + 1) / 2 + b]);
        let den = (aa * bb).sqrt();
        if den > 0.0 {
            m[a * (a + 1) / 2 + b] / den
        } else {
            f64::NAN
        }
    }
}

/// A chunk's suffix sums: `sums` holds `S_{end−1}, S_{end−2}, …, S_low`,
/// `S_i` the merge of blocks `i..end`, `end` the chunk's last block plus
/// one. Extended downward as a window's start needs, trimmed from below as
/// the oldest window's start passes.
#[derive(Debug, Clone)]
struct Suffixes {
    chunk: u64,
    low: u64,
    sums: Vec<f64>,
}

/// A running prefix sum: the merge of blocks `chunk·C .. upto`.
#[derive(Debug, Clone, Default)]
struct Prefix {
    valid: bool,
    chunk: u64,
    upto: u64,
    sum: Vec<f64>,
}

/// Which merged sums a window of rows reads (the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    /// Its full blocks, `j0..j1`.
    j0: u64,
    j1: u64,
    /// The chunk whose suffix from `j0` it reads, if any.
    suffix: Option<u64>,
    /// The chunk whose prefix to `j1` it reads, if any.
    prefix: Option<u64>,
}

/// The merged sums the next report's two windows read; see the module doc.
/// A cache, rebuilt from the ring and its row count, so it is neither saved
/// nor compared.
#[derive(Clone)]
pub(crate) struct WindowSums {
    layout: Layout,
    /// `b`, `C` and `w`.
    block: u64,
    chunk: u64,
    window: u64,
    suffixes: Vec<Suffixes>,
    /// The prefix each window reads: `[pre, post]`.
    prefixes: [Prefix; 2],
    /// A block's summary, `δ` and `c·δ`, and a row's deviations.
    leaf: Vec<f64>,
    delta: Vec<f64>,
    dev: Vec<f64>,
}

impl fmt::Debug for WindowSums {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowSums")
            .field("block", &self.block)
            .field("chunk", &self.chunk)
            .field("chunks", &self.suffixes.len())
            .finish()
    }
}

/// The sums are a function of the ring and its row count, which the state
/// compares already.
impl PartialEq for WindowSums {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl WindowSums {
    /// The sums of two windows of `w` rows over `k` columns, or `None`
    /// where a window holds fewer than [`MIN_BLOCKS`] full blocks and is
    /// read whole.
    pub(crate) fn new(k: usize, w: usize) -> Option<Self> {
        let b = block_rows(k);
        let g = w / b;
        if g < MIN_BLOCKS {
            return None;
        }
        let layout = Layout::new(k);
        let s = layout.stride();
        Some(Self {
            layout,
            block: b as u64,
            chunk: (g - 1) as u64,
            window: w as u64,
            suffixes: Vec::new(),
            prefixes: [Prefix::default(), Prefix::default()],
            leaf: vec![0.0; s],
            delta: vec![0.0; 2 * k],
            dev: vec![0.0; k],
        })
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    /// Rows a block covers.
    #[cfg(test)]
    pub(crate) fn block(&self) -> u64 {
        self.block
    }

    /// Forget every sum: the rows are renumbered from 0.
    pub(crate) fn clear(&mut self) {
        self.suffixes.clear();
        for p in &mut self.prefixes {
            p.valid = false;
        }
    }

    /// The numbers the sums hold, for the memory they take.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.suffixes.iter().map(|s| s.sums.len()).sum::<usize>()
            + self.prefixes.iter().map(|p| p.sum.len()).sum::<usize>()
    }

    /// The two windows of the report on row `pos` (numbered since the ring
    /// started): `pre` is rows `[pos + 1 − 2w, pos + 1 − w)` and `post`
    /// rows `[pos + 1 − w, pos + 1)`, `pos` itself the row being reported.
    fn windows(&self, pos: u64) -> [(u64, u64); 2] {
        let w = self.window;
        let end = pos + 1;
        [(end - 2 * w, end - w), (end - w, end)]
    }

    /// What the window of rows `[s, e)` reads, on the report of row `pos`:
    /// its full blocks are those whose rows are all before `pos`, the row
    /// being reported not yet in the ring.
    fn plan(&self, s: u64, e: u64, pos: u64) -> Plan {
        let (b, c) = (self.block, self.chunk);
        let j0 = s.div_ceil(b);
        let j1 = e.min(pos) / b;
        // Between `G − 2` and `G` blocks (the module doc), so at least 2.
        assert!(j1 > j0, "a window of {} rows holds no full block", e - s);
        let (c0, c1) = (j0 / c, (j1 - 1) / c);
        if c0 == c1 {
            if j0 % c == 0 {
                Plan {
                    j0,
                    j1,
                    suffix: None,
                    prefix: Some(c1),
                }
            } else {
                assert_eq!(j1, (c0 + 1) * c, "a run inside one chunk ends at its end");
                Plan {
                    j0,
                    j1,
                    suffix: Some(c0),
                    prefix: None,
                }
            }
        } else {
            assert_eq!(c1, c0 + 1, "a run meets two chunks at most");
            Plan {
                j0,
                j1,
                suffix: Some(c0),
                prefix: Some(c1),
            }
        }
    }

    /// Make ready the sums the report of row `pos` reads, `row(q)` reading
    /// each row `q < pos` from the ring; and drop the sums no later report
    /// reads.
    pub(crate) fn ready<'a>(&mut self, pos: u64, row: &impl Fn(u64) -> &'a [f64]) {
        let windows = self.windows(pos);
        let mut oldest = None;
        for (side, (s, e)) in windows.into_iter().enumerate() {
            let plan = self.plan(s, e, pos);
            if side == 0 {
                oldest = Some(plan.j0);
            }
            if let Some(c) = plan.prefix {
                self.advance_prefix(side, c, plan.j1, row);
            }
            if let Some(c) = plan.suffix {
                self.extend_suffix(c, plan.j0, row);
            }
        }
        // No later report starts before the older window's first block.
        let low = oldest.expect("two windows");
        let (c, stride) = (self.chunk, self.layout.stride());
        self.suffixes.retain(|s| s.chunk >= low / c);
        for s in &mut self.suffixes {
            if s.low < low {
                let end = (s.chunk + 1) * c;
                s.sums.truncate((end - low) as usize * stride);
                s.low = low;
            }
        }
    }

    /// The prefix `side` reads, carried to block `j1` of chunk `chunk`:
    /// from where it is, or from the chunk's first block.
    fn advance_prefix<'a>(
        &mut self,
        side: usize,
        chunk: u64,
        j1: u64,
        row: &impl Fn(u64) -> &'a [f64],
    ) {
        let (b, c, layout) = (self.block, self.chunk, self.layout);
        let p = &mut self.prefixes[side];
        if !(p.valid && p.chunk == chunk && p.upto <= j1) {
            p.sum.clear();
            p.sum.resize(layout.stride(), 0.0);
            p.valid = true;
            p.chunk = chunk;
            p.upto = chunk * c;
        }
        while p.upto < j1 {
            let j = p.upto;
            layout.leaf(&mut self.leaf, &mut self.dev, j * b, (j + 1) * b, row);
            layout.merge(&mut p.sum, &self.leaf, &mut self.delta);
            p.upto += 1;
        }
    }

    /// The suffix sums of chunk `chunk`, extended down to block `j0`.
    fn extend_suffix<'a>(&mut self, chunk: u64, j0: u64, row: &impl Fn(u64) -> &'a [f64]) {
        let (b, c, layout) = (self.block, self.chunk, self.layout);
        let stride = layout.stride();
        let at = match self.suffixes.iter().position(|s| s.chunk == chunk) {
            Some(i) => i,
            None => {
                self.suffixes.push(Suffixes {
                    chunk,
                    low: (chunk + 1) * c,
                    sums: Vec::new(),
                });
                self.suffixes.len() - 1
            }
        };
        let s = &mut self.suffixes[at];
        while s.low > j0 {
            let i = s.low - 1;
            let before = s.sums.len();
            s.sums.resize(before + stride, 0.0);
            let (done, new) = s.sums.split_at_mut(before);
            layout.leaf(new, &mut self.dev, i * b, (i + 1) * b, row);
            if before > 0 {
                layout.merge(new, &done[before - stride..], &mut self.delta);
            }
            s.low = i;
        }
    }

    /// The suffix sum `S_j0` of chunk `chunk`.
    fn suffix(&self, chunk: u64, j0: u64) -> &[f64] {
        let s = self
            .suffixes
            .iter()
            .find(|s| s.chunk == chunk)
            .expect("the suffixes a report reads are ready");
        let end = (chunk + 1) * self.chunk;
        assert!(s.low <= j0 && j0 < end, "suffix {j0} of chunk {chunk}");
        let i = (end - 1 - j0) as usize * self.layout.stride();
        &s.sums[i..i + self.layout.stride()]
    }

    /// The summaries of the report of row `pos`'s two windows, `[pre,
    /// post]`, into `out` (two summaries' room), `tmp` (one summary, `δ`
    /// and `c·δ`, and a row's deviations) its working space; `row(q)` reads
    /// row `q`, `pos` the row being reported.
    pub(crate) fn summaries<'a>(
        &self,
        pos: u64,
        row: &impl Fn(u64) -> &'a [f64],
        out: &mut [f64],
        tmp: &mut [f64],
    ) {
        let (layout, b, k) = (self.layout, self.block, self.layout.k);
        let stride = layout.stride();
        let (leaf, rest) = tmp.split_at_mut(stride);
        let (delta, dev) = rest.split_at_mut(2 * k);
        for (side, ((s, e), acc)) in self
            .windows(pos)
            .into_iter()
            .zip(out.chunks_exact_mut(stride))
            .enumerate()
        {
            let plan = self.plan(s, e, pos);
            layout.leaf(acc, dev, s, plan.j0 * b, row);
            if let Some(c) = plan.suffix {
                layout.merge(acc, self.suffix(c, plan.j0), delta);
            }
            if let Some(c) = plan.prefix {
                let p = &self.prefixes[side];
                assert!(
                    p.valid && p.chunk == c && p.upto == plan.j1,
                    "the prefix a report reads is ready"
                );
                layout.merge(acc, &p.sum, delta);
            }
            layout.leaf(leaf, dev, plan.j1 * b, e, row);
            layout.merge(acc, leaf, delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every report's two windows read a run of full blocks that one
    /// suffix, one prefix, or a suffix and a prefix cover, at every window
    /// length a width merges and every row number through three periods of
    /// the chunks: `plan`'s asserts hold, and head, blocks and tail tile
    /// each window.
    #[test]
    fn every_windows_blocks_are_a_suffix_and_a_prefix() {
        let mut plans = 0usize;
        for k in 2..=24 {
            let b = block_rows(k) as u64;
            for w in 3..=260usize {
                let Some(sums) = WindowSums::new(k, w) else {
                    assert!((w as u64) < MIN_BLOCKS as u64 * b, "k {k}, w {w}");
                    continue;
                };
                assert_eq!(sums.chunk, w as u64 / b - 1, "k {k}, w {w}");
                let period = sums.block * sums.chunk;
                for pos in (2 * w as u64 - 1)..(2 * w as u64 - 1 + 3 * period) {
                    for (s, e) in sums.windows(pos) {
                        let p = sums.plan(s, e, pos);
                        assert!(p.j0 * b >= s && p.j0 * b - s < b, "head of k {k}, w {w}");
                        assert!(p.j1 * b <= e && e - p.j1 * b <= b, "tail of k {k}, w {w}");
                        assert!(p.j1 * b <= pos, "an incomplete block, k {k}, w {w}");
                        assert!(p.suffix.is_some() || p.prefix.is_some());
                        plans += 1;
                    }
                }
            }
        }
        assert!(plans > 1_000_000, "{plans}");
    }

    /// The merge is Chan, Golub & LeVeque's rule: two halves of a set of
    /// rows merged give the set's own summary, to rounding, whichever half
    /// is the larger (the larger's anchor is kept), and merging nothing or
    /// into nothing copies.
    #[test]
    fn a_merge_of_two_parts_is_the_whole() {
        let k = 3;
        let layout = Layout::new(k);
        let s = layout.stride();
        let mut x = 7u64;
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|_| {
                (0..k)
                    .map(|j| {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        1e3 * j as f64 + ((x >> 11) as f64 / (1u64 << 53) as f64)
                    })
                    .collect()
            })
            .collect();
        let row = |q: u64| rows[q as usize].as_slice();
        let (mut dev, mut delta) = (vec![0.0; k], vec![0.0; 2 * k]);
        let mut whole = vec![0.0; s];
        layout.leaf(&mut whole, &mut dev, 0, 40, &row);
        for cut in [1u64, 13, 20, 33, 39] {
            let (mut a, mut b) = (vec![0.0; s], vec![0.0; s]);
            layout.leaf(&mut a, &mut dev, 0, cut, &row);
            layout.leaf(&mut b, &mut dev, cut, 40, &row);
            layout.merge(&mut a, &b, &mut delta);
            assert_eq!(a[0], 40.0);
            let anchor = if cut >= 20 { 0 } else { cut as usize };
            assert_eq!(
                &a[1..=k],
                &rows[anchor][..],
                "cut {cut}: the larger part's anchor"
            );
            for i in 0..k {
                let (m, want) = (a[1 + i] + a[1 + k + i], whole[1 + i] + whole[1 + k + i]);
                assert!(
                    (m - want).abs() < 1e-12 * (1.0 + want.abs()),
                    "cut {cut}: mean {i}"
                );
            }
            for (j, (g, w)) in a[1 + 2 * k..].iter().zip(&whole[1 + 2 * k..]).enumerate() {
                assert!(
                    (g - w).abs() < 1e-12 * (1.0 + w.abs()),
                    "cut {cut}: entry {j}: {g} vs {w}"
                );
            }
        }
        let empty = vec![0.0; s];
        let mut copy = empty.clone();
        layout.merge(&mut copy, &whole, &mut delta);
        assert_eq!(copy, whole);
        layout.merge(&mut copy, &empty, &mut delta);
        assert_eq!(copy, whole);
    }

    /// A block's summary takes no more numbers than its rows hold in the
    /// ring, at every width.
    #[test]
    fn a_blocks_summary_is_no_larger_than_its_rows() {
        for k in 2..=400 {
            assert!(Layout::new(k).stride() <= block_rows(k) * k, "k {k}");
        }
    }
}
